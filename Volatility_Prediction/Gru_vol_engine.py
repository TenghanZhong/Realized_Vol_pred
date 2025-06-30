import math, random, warnings, os
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

PRICE_WINDOW = 60
def default_cfg():
    return dict(
        seq_len=60, batch=64, hidden=256, dropout=0.2,
        horizon=5, train_window_days=960,
        lr=3e-4, epochs=300, early_stop_patience=12, seed=SEED
    )

class HVGRU(nn.Module):
    def __init__(self, n_feat, cfg):
        super().__init__()
        self.gru = nn.GRU(n_feat, cfg['hidden'], num_layers=2,
                          batch_first=True, dropout=cfg['dropout'])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg['hidden']),
            nn.Linear(cfg['hidden'], cfg['hidden']//2), nn.ReLU(),
            nn.Linear(cfg['hidden']//2, 1)
        )
    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)

def roll_z_per_row(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w, min_periods=w).mean()
    sd = s.rolling(w, min_periods=w).std().add(1e-12)
    return (s - mu) / sd

def roll_z_fixed(s: pd.Series, w: int) -> pd.Series:
    mu = s.rolling(w, w//2).mean()
    sd = s.rolling(w, w//2).std().add(1e-12)
    return (s - mu) / sd

def base_features(df: pd.DataFrame, H: int) -> pd.DataFrame:
    d = df.copy(); d.columns = d.columns.str.lower()
    out = d.copy()
    out['oc_ret']   = (out['close'] - out['open']) / out['open']
    ln_ret = np.log(out['close']).diff()
    out['ret1_z'] = roll_z_fixed(out['close'].pct_change(), 20)
    out['ret3_z'] = roll_z_fixed(out['close'].pct_change(3), 20)
    out['iv'] = out['iv'] / 100
    out[f'hv{H}_lag'] = ln_ret.rolling(H, H).std(ddof=1).shift(1) * math.sqrt(365)
    out['hl_pct_log'] = np.log1p(((out['high'] - out['low'])/out['close']).clip(lower=0))
    vol10 = out['volume usdt'].rolling(10,5).mean().add(1e-12)
    out['vol_spike_10'] = np.log1p(out['volume usdt']/vol10)
    out[f'skew{H}'] = ln_ret.rolling(H,H).skew().shift(1).fillna(0)
    out[f'kurt{H}'] = ln_ret.rolling(H,H).kurt().shift(1).fillna(0)
    return out[['iv', f'hv{H}_lag', 'hl_pct_log', 'oc_ret',
                'vol_spike_10', 'fear_index', f'skew{H}', f'kurt{H}',
                'ret1_z', 'ret3_z']]

def future_hv(df: pd.DataFrame, k: int, ann: int = 365) -> pd.DataFrame:
    d = df.copy(); d.columns = d.columns.str.lower()
    log_ret = np.log(d['close']).diff()
    d[f'hvCC_f{k}'] = log_ret.rolling(k).std(ddof=1).shift(-k+1) * math.sqrt(ann)
    hl2 = np.log(d['high']/d['low'])**2
    park_var = hl2.rolling(k).sum()/(4*k*math.log(2))
    d[f'hvPark_f{k}'] = np.sqrt(park_var).shift(-k+1) * math.sqrt(ann)
    return d

def build_xy(f: pd.DataFrame, y: pd.Series, L: int):
    X = np.stack([f.iloc[i-L+1:i+1].values for i in range(L-1, len(f))])
    return X, y.iloc[L-1:].values

def train_stage1(X, y, cfg, device):
    N = len(X)
    X_all, y_all = X[:-1], y[:-1]
    split = int(len(X_all) * 0.9)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:], y_all[split:]
    loader = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                      torch.tensor(y_tr, dtype=torch.float32)),
                        batch_size=cfg['batch'], shuffle=True)
    model = HVGRU(X.shape[2], cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)
    best_loss, wait, best_ep = float('inf'), 0, 0
    for ep in range(1, cfg['epochs']+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
        model.eval()
        with torch.no_grad(): v_loss = loss_fn(model(X_va_t), y_va_t).item()
        if v_loss < best_loss - 1e-6:
            best_loss, wait, best_ep = v_loss, 0, ep
        else:
            wait += 1
            if wait >= cfg['early_stop_patience']: break
    return best_ep

def train_stage2(X, y, cfg, device, epochs):
    X_all, y_all = X[:-1], y[:-1]
    loader = DataLoader(TensorDataset(torch.tensor(X_all, dtype=torch.float32),
                                      torch.tensor(y_all, dtype=torch.float32)),
                        batch_size=cfg['batch'], shuffle=True)
    model = HVGRU(X.shape[2], cfg).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    return model

if __name__ == '__main__':
    df = pd.read_csv(r"D:\prediction_market_data\crypto_data\merged_btc_data.csv",
                     parse_dates=['date']).sort_values('date')
    cfg = default_cfg()
    H = cfg['horizon']
    warm, look, L = PRICE_WINDOW, cfg['train_window_days'], cfg['seq_len']

    # 特征工程
    df = future_hv(df, H)
    feat = base_features(df, H)
    roll_cols = [f'hv{H}_lag', 'hl_pct_log', 'fear_index', 'oc_ret', 'vol_spike_10']
    for c in roll_cols:
        feat[f'{c}_z'] = roll_z_per_row(feat[c], PRICE_WINDOW)
    static = ['iv', f'skew{H}', f'kurt{H}']
    raw = ['ret1_z', 'ret3_z']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 两个目标，循环处理
    for label, target_col in [("Close-to-Close", f"hvCC_f{H}"), ("Parkinson", f"hvPark_f{H}")]:
        feat_last = feat.iloc[-(warm + look):].copy()
        mu_s = feat_last[static].mean()
        sd_s = feat_last[static].std().add(1e-12)
        feat_last[static] = (feat_last[static] - mu_s) / sd_s
        mat = feat_last[[f'{c}_z' for c in roll_cols] + static + raw].dropna()
        y = df[target_col].reindex(mat.index)
        mask = y.notna()
        mat = mat[mask]
        y = y[mask]
        if len(mat) < L:
            raise ValueError(f"{label} 样本数不足 L={L}, 实际={len(mat)}")
        X, y = build_xy(mat, y, L)
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        assert np.isfinite(X).all() and np.isfinite(y).all(), "特征或目标存在无效值"

        best_ep = train_stage1(X, y, cfg, device)
        model = train_stage2(X, y, cfg, device, best_ep + 5)
        model.eval()
        x_pred = torch.tensor(X[-1][None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(x_pred).item()
        print(f"最新数据的未来{H}天{label}波动率预测输出: {pred:.6f}")


