import math, random, warnings, os
import numpy as np, pandas as pd, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

warnings.filterwarnings("ignore")
SEED = 2025
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ───────── CONFIG ─────────
PRICE_WINDOW = 60  # rolling-Z 窗口
def default_cfg():
    return dict(
        seq_len=60, batch=64, hidden=256, dropout=0.2,
        horizon=5, train_window_days=960,
        lr=3e-4, epochs=300, early_stop_patience=12
    )

# ───────── MODEL ─────────
class HVGRU(nn.Module):
    def __init__(self, n_feat, cfg):
        super().__init__()
        self.gru = nn.GRU(
            n_feat, cfg['hidden'],
            num_layers=2, batch_first=True,
            dropout=cfg['dropout']
        )
        self.head = nn.Sequential(
            nn.LayerNorm(cfg['hidden']),
            nn.Linear(cfg['hidden'], cfg['hidden']//2), nn.ReLU(),
            nn.Linear(cfg['hidden']//2, 1)
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1]).squeeze(-1)

# ───────── FEATURES ─────────
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
    ln_ret          = np.log(out['close']).diff()
    out['ret1_z']   = roll_z_fixed(out['close'].pct_change(), 20)
    out['ret3_z']   = roll_z_fixed(out['close'].pct_change(3), 20)
    out['iv']       = out['iv'] / 100
    out[f'hv{H}_lag'] = ln_ret.rolling(H, H).std(ddof=1).shift(1) * math.sqrt(365)
    out['hl_pct_log'] = np.log1p(((out['high'] - out['low'])/out['close']).clip(lower=0))
    vol10 = out['volume usdt'].rolling(10,5).mean().add(1e-12)
    out['vol_spike_10'] = np.log1p(out['volume usdt']/vol10)
    out[f'skew{H}']    = ln_ret.rolling(H,H).skew().shift(1).fillna(0)
    out[f'kurt{H}']    = ln_ret.rolling(H,H).kurt().shift(1).fillna(0)
    return out[['iv', f'hv{H}_lag', 'hl_pct_log',
                'oc_ret', 'vol_spike_10', 'fear_index',
                f'skew{H}', f'kurt{H}',
                'ret1_z', 'ret3_z']]

def future_hv(df, k, ann=365):
    d = df.copy(); d.columns = d.columns.str.lower()
    log_ret = np.log(d['close']).diff()
    d[f'hvCC_f{k}']   = (log_ret.rolling(k).std(ddof=1).shift(-k+1) * math.sqrt(ann))
    hl2               = np.log(d['high']/d['low'])**2
    park_var          = hl2.rolling(k).sum()/(4*k*math.log(2))
    d[f'hvPark_f{k}'] = (np.sqrt(park_var).shift(-k+1) * math.sqrt(ann))
    return d

def build_xy(f: pd.DataFrame, y: pd.Series, L: int):
    X = np.stack([f.iloc[i-L+1:i+1].values for i in range(L-1, len(f))])
    return X, y.iloc[L-1:].values

# ───────── TRAINING 阶段一：带 Early Stopping ─────────
def train_stage1(X, y, cfg, device, static_idx):
    N = len(X)
    X_pred, y_pred = X[-1], y[-1]
    X_all, y_all   = X[:N-1], y[:N-1]

    split = int(len(X_all) * 0.9)
    X_tr, y_tr = X_all[:split], y_all[:split]
    X_va, y_va = X_all[split:], y_all[split:]

    # --- 静态归一化：仅基于训练集
    mu_s = X_tr[:, -1, static_idx].mean(axis=0)
    sd_s = X_tr[:, -1, static_idx].std(axis=0) + 1e-12

    X_tr[:, -1, static_idx] = (X_tr[:, -1, static_idx] - mu_s) / sd_s
    X_va[:, -1, static_idx] = (X_va[:, -1, static_idx] - mu_s) / sd_s

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_tr, dtype=torch.float32),
            torch.tensor(y_tr, dtype=torch.float32)
        ),
        shuffle=True, batch_size=cfg['batch']
    )

    model = HVGRU(X.shape[2], cfg).to(device)
    opt   = optim.AdamW(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.SmoothL1Loss(beta=0.1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=10, verbose=False, min_lr=1e-6
    )

    X_va_t = torch.tensor(X_va, dtype=torch.float32, device=device)
    y_va_t = torch.tensor(y_va, dtype=torch.float32, device=device)

    best_loss, wait, best_epoch = float('inf'), 0, 0
    for epoch in range(1, cfg['epochs']+1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_va_t), y_va_t).item()
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss, wait, best_epoch = val_loss, 0, epoch
        else:
            wait += 1
            if wait >= cfg['early_stop_patience']:
                break

    # 返回 best_epoch 和归一化参数
    return best_epoch, mu_s, sd_s


# ───────── TRAINING 阶段二：全量重训 ─────────
def train_stage2(X, y, cfg, device, epochs, static_idx, mu_s, sd_s):
    X_all, y_all = X[:-1], y[:-1]

    # 使用阶段一训练集的归一化参数
    X_all[:, -1, static_idx] = (X_all[:, -1, static_idx] - mu_s) / sd_s

    loader = DataLoader(
        TensorDataset(
            torch.tensor(X_all, dtype=torch.float32),
            torch.tensor(y_all, dtype=torch.float32)
        ),
        shuffle=True, batch_size=cfg['batch']
    )

    model = HVGRU(X.shape[2], cfg).to(device)
    opt   = optim.AdamW(model.parameters(), lr=cfg['lr'])
    loss_fn = nn.SmoothL1Loss(beta=0.1)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

    return model

# ───────── BACKTEST ─────────
def backtest(path, cfg, out_path):
    H   = cfg['horizon']
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    warm = PRICE_WINDOW
    look = cfg['train_window_days']
    L    = cfg['seq_len']

    df = (pd.read_csv(path, parse_dates=['date'])
            .rename(columns=str.lower)
            .set_index('date')
            .sort_index())
    df   = future_hv(df, H)
    feat = base_features(df, H)
    roll_cols = [f'hv{H}_lag', 'hl_pct_log', 'fear_index', 'oc_ret', 'vol_spike_10']
    for c in roll_cols:
        feat[f'{c}_z'] = roll_z_per_row(feat[c], PRICE_WINDOW)
    static = ['iv', f'skew{H}', f'kurt{H}']
    raw    = ['ret1_z', 'ret3_z']

    err, rec = {'cc': [], 'park': []}, []

    start = warm + look - 1
    end   = len(df) - 1 - H

    for i in trange(start, end+1, desc='Rolling', ncols=70):
        seg_full = feat.iloc[i-(warm+look)+1 : i+1].copy()
        seg = seg_full.iloc[warm:].dropna(subset=[f'{c}_z' for c in roll_cols])
        mat = seg[[f'{c}_z' for c in roll_cols] + static + raw].dropna()

        for k, col in {'cc': f'hvCC_f{H}', 'park': f'hvPark_f{H}'}.items():
            df_y = pd.concat([mat, df[col].reindex(mat.index)], axis=1)\
                     .dropna().rename(columns={col: 'y'})
            if len(df_y) < L+32:
                continue
            # 确定 static 特征索引位置（只需一次）
            feat_cols = df_y.drop(columns='y').columns.tolist()
            static_idx = [feat_cols.index(col) for col in static]

            X, y = build_xy(df_y.drop(columns='y'), df_y['y'], L)
            if len(X) < 32:
                continue

            # ——— 阶段一：早停，选出 best_epoch ———
            best_epoch, mu_s, sd_s = train_stage1(X, y, cfg, dev, static_idx)

            # ——— 阶段二：全量重训 best_epoch 次 ———

            epochs_stage2 = best_epoch + 5
            print(f"Stage2 共训练了 {epochs_stage2} 轮")  # 这里直接输出
            model = train_stage2(X, y, cfg, dev, epochs_stage2, static_idx, mu_s, sd_s)

            # 预测
            X_pred = torch.tensor(X[-1][None], dtype=torch.float32, device=dev)
            pred = model(X_pred).item()
            y_true = y[-1]
            print(f"[{k}] Predicting: {df_y.index[-1].strftime('%Y-%m-%d')},  Pred = {pred:.5f},  True = {y_true:.5f}")

            del model; torch.cuda.empty_cache()

            err[k].append(abs(pred - y_true))
            rec.append({'date': df_y.index[-1], 'type': k,
                        'pred': pred, 'true': y_true})

    # 汇总
    print("\n──────── SUMMARY ────────")
    for k, name in [('cc', 'Close-to-Close'), ('park', 'Parkinson')]:
        err_list = err[k]
        rec_type = [r for r in rec if r['type'] == k]
        if not err_list:
            print(f"{name:<17}:  No predictions")
        else:
            abs_pct_err = [
                abs(r['pred'] - r['true']) / (abs(r['true']) + 1e-8)
                for r in rec_type
            ]
            mape = 100 * np.mean(abs_pct_err)
            print(
                f"{name:<17}:  MAE = {np.mean(err_list):.6f} | MedAE = {np.median(err_list):.6f} | MAPE = {mape:.2f}%")

    pd.DataFrame(rec).to_csv(out_path, index=False)
    print(f"\nSaved to → {os.path.abspath(out_path)}")

# ───────── ENTRY ─────────
if __name__ == "__main__":
    cfg = default_cfg()
    backtest(
        r"D:\prediction_market_data\crypto_data\merged_btc_data.csv",
        cfg,
        r"C:\Users\26876\Desktop\RA_Summer\Result_Vol\Result_Vol_2.csv"
    )
