"""
HV Interval‑Engine ★ Backtest Edition – Quantile GRU (rev. Indep‑Tail CQR)
--------------------------------------------------------------------------
• 独立尾部 CQR 校准：左右误差各按 (1‑conf_lvl) 分位 → 区间不过宽。
• 分位点范围收窄为 [0.10, 0.90]。
• 回测循环实时打印：date | true | [lo, hi] | mid。
"""

import math, random, warnings, numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange

warnings.filterwarnings("ignore")
SEED = 2025
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

PRICE_WINDOW = 60  # rolling‑Z 窗口

# ───────────────────────── Config ─────────────────────────

def default_cfg():
    return dict(
        train_frac=0.8, seq_len=60, epochs=300, batch=64,
        hidden=256, dropout=0.2, lr=1e-3,
        early_stop_patience=15, lr_reduce_patience=7, lr_reduce_factor=0.5,
        horizon=5, train_window_days=720,
        quant_lo=0.05, quant_hi=0.95, quant_step=0.05,
        conf_lvl=0.90,
    )

# ───────────────────────── Model ─────────────────────────

class PinballGRU(nn.Module):
    def __init__(self, n_feat: int, n_q: int, cfg):
        super().__init__()
        self.gru = nn.GRU(n_feat, cfg['hidden'], num_layers=2, batch_first=True, dropout=cfg['dropout'])
        self.head = nn.Sequential(
            nn.LayerNorm(cfg['hidden']),
            nn.Linear(cfg['hidden'], cfg['hidden'] // 2), nn.ReLU(),
            nn.Linear(cfg['hidden'] // 2, n_q)
        )
    def forward(self, x):
        return self.head(self.gru(x)[0][:, -1])

# ───────────────────────── Quantile utils ─────────────────────────

def build_quants(cfg):
    return np.round(np.arange(cfg['quant_lo'], cfg['quant_hi'] + 1e-9, cfg['quant_step']), 5)

def qloss(pred: torch.Tensor, targ: torch.Tensor, quants: np.ndarray):
    q = torch.as_tensor(quants, device=pred.device)
    e = targ.unsqueeze(1) - pred
    return torch.mean(torch.maximum((q - 1) * e, q * e))

# ───────────────────────── Feature engineering ─────────────────────────

def roll_z(s: pd.Series, w: int):
    mu = s.rolling(w, w).mean(); sd = s.rolling(w, w).std().add(1e-12)
    return (s - mu) / sd

def base_features(df: pd.DataFrame, H: int) -> pd.DataFrame:
    df = df.copy(); df.columns = df.columns.str.lower()
    out = df.copy()
    out['oc_ret'] = (out['close'] - out['open']) / out['open']
    ln_ret = np.log(out['close']).diff()
    out['ret1_raw'] = out['close'].pct_change()
    out['ret3_raw'] = out['close'].pct_change(3)
    out['ret1_z'] = roll_z(out['ret1_raw'], 20)
    out['ret3_z'] = roll_z(out['ret3_raw'], 20)
    out['iv'] = out['iv'] / 100
    out[f'hv{H}_lag'] = ln_ret.rolling(H, H).std(ddof=1).shift(1) * math.sqrt(365)
    out['hl_pct_log'] = np.log1p(((out['high'] - out['low']) / out['close']).clip(lower=0))
    vol_mean10 = out['volume usdt'].rolling(10, 10).mean().add(1e-12)
    out['vol_spike_10'] = np.log1p(out['volume usdt'] / vol_mean10)
    out[f'skew{H}'] = ln_ret.rolling(H, H).skew().shift(1).fillna(0)
    out[f'kurt{H}'] = ln_ret.rolling(H, H).kurt().shift(1).fillna(0)
    return out[['iv', f'hv{H}_lag', 'hl_pct_log', 'oc_ret', 'vol_spike_10', 'fear_index', f'skew{H}', f'kurt{H}', 'ret1_z', 'ret3_z']]

# ───────────────────────── Label ─────────────────────────

def future_hv(df: pd.DataFrame, k: int, ann: int = 365):
    df = df.copy(); df.columns = df.columns.str.lower()
    log_ret = np.log(df['close']).diff()
    df[f'hvcc_f{k}'] = log_ret.rolling(k).std(ddof=1) * math.sqrt(ann)
    df[f'hvcc_f{k}'] = df[f'hvcc_f{k}'].shift(-k + 1)
    hl2 = np.log(df['high'] / df['low']) ** 2
    park_var = hl2.rolling(k).sum() / (4 * k * math.log(2))
    df[f'hvpark_f{k}'] = np.sqrt(park_var) * math.sqrt(ann)
    df[f'hvpark_f{k}'] = df[f'hvpark_f{k}'].shift(-k + 1)
    return df

# ───────────────────────── Helper ─────────────────────────

def build_xy(f: pd.DataFrame, y: pd.Series, L: int):
    X = np.stack([f.iloc[i - L:i].values for i in range(L, len(f))])
    return X, y.iloc[L:].values

# ───────────────────────── Training ─────────────────────────

def train_gru_q(X_tr, y_tr, X_val, y_val, cfg, device):
    quants = build_quants(cfg)
    dl = DataLoader(TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.float32)), shuffle=True, batch_size=cfg['batch'])
    model = PinballGRU(X_tr.shape[2], len(quants), cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg['lr'])
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', factor=cfg['lr_reduce_factor'], patience=cfg['lr_reduce_patience'])
    best = float('inf'); wait = 0
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)
    for _ in range(cfg['epochs']):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = qloss(model(xb), yb, quants); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval(); vl = qloss(model(Xv), yv, quants).item(); sch.step(vl)
        if vl < best * 0.999:
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}; best = vl; wait = 0
        else:
            wait += 1
            if wait >= cfg['early_stop_patience']:
                break
    model.load_state_dict(best_state)
    return model

# ───────────────────────── Indep‑Tail CQR ─────────────────────────

def calibrate_cqr(model, X_cal, y_cal, quants, conf_lvl, device):
    """
    独立尾部 CQR 校准：对左右误差各取 (1-conf_lvl) 分位数
    输入：
      - model: 已训练好的 PinballGRU 模型
      - X_cal, y_cal: 用于校准的特征和标签 numpy 数组
      - quants: 分位点列表
      - conf_lvl: 目标置信水平 (e.g. 0.90)
      - device: 'cpu' or 'cuda'
    返回：
      - δ_lo, δ_hi: 下/上修正量
    """
    model.eval()
    # 1) 预测校准集的所有分位点
    with torch.no_grad():
        preds = model(torch.tensor(X_cal, dtype=torch.float32, device=device)).cpu().numpy()
    # 2) 单调修正
    preds = np.maximum.accumulate(preds, axis=1)
    # 3) 计算左右误差
    err_lo = np.maximum(y_cal - preds[:, 0], 0)
    err_hi = np.maximum(preds[:, -1] - y_cal, 0)
    # 4) 取 (1-conf_lvl) 分位数
    alpha = 1 - conf_lvl
    δ_lo = np.quantile(err_lo, alpha)
    δ_hi = np.quantile(err_hi, alpha)
    return δ_lo, δ_hi


def infer_interval(q_pred, δ_lo, δ_hi):
    q_pred = np.maximum.accumulate(q_pred)
    lo = q_pred[0] - δ_lo
    hi = q_pred[-1] + δ_hi
    lo = max(lo, 0.0)    # 保证不为负
    return lo, hi


# ───────────────────────── Backtest ─────────────────────────

# ───────────────────────── Backtest ─────────────────────────
def backtest(path: str, cfg):
    H      = cfg['horizon']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取数据 & 构造特征/标签
    df_raw = (
        pd.read_csv(path, parse_dates=['date'])
          .rename(columns=str.lower)
          .set_index('date')
          .sort_index()
    )
    df_raw = future_hv(df_raw, H)
    feat   = base_features(df_raw, H)

    rolling_cols = [f'hv{H}_lag', 'hl_pct_log']
    static_cols  = ['iv', 'oc_ret', 'vol_spike_10', 'fear_index', f'skew{H}', f'kurt{H}']
    raw_cols     = ['ret1_z', 'ret3_z']

    L    = cfg['seq_len']
    look = cfg['train_window_days'] + PRICE_WINDOW + L + H
    beg, end = look - 1, len(df_raw) - 1 - H

    quants = build_quants(cfg)

    cover_cnt = total_cnt = 0
    width_sum = 0.0
    err_abs   = {'cc': [], 'park': []}
    err_pct   = {'cc': [], 'park': []}

    for i in trange(beg, end + 1, ncols=70, desc="Rolling"):
        date_pt = feat.index[i]

        seg  = feat.iloc[i - look + 1 : i + 1].copy()
        base = seg.iloc[:-H]

        # rolling-Z
        mu = base[rolling_cols].rolling(PRICE_WINDOW, PRICE_WINDOW).mean()
        sd = base[rolling_cols].rolling(PRICE_WINDOW, PRICE_WINDOW).std().add(1e-12)
        for c in rolling_cols:
            seg[f'{c}_z'] = (seg[c] - mu[c]) / sd[c]
        seg[[f'{c}_z' for c in rolling_cols]] = seg[[f'{c}_z' for c in rolling_cols]].ffill()

        mat = seg[[f'{c}_z' for c in rolling_cols] + static_cols + raw_cols].dropna()

        for tag, col in {'cc': f'hvcc_f{H}', 'park': f'hvpark_f{H}'}.items():
            df_y = (
                pd.concat([mat, df_raw[col].reindex(mat.index)], axis=1)
                  .dropna()
                  .rename(columns={col: 'y'})
            )
            if len(df_y) < L + 20:
                continue

            # 静态特征归一化（仅用训练段）
            n_all = len(df_y)
            n_tr_main = int(n_all * cfg['train_frac'])
            mu_s = df_y[static_cols].iloc[:n_tr_main].mean()
            sd_s = df_y[static_cols].iloc[:n_tr_main].std().add(1e-12)
            df_y[static_cols] = (df_y[static_cols] - mu_s) / sd_s

            # 构建序列
            X, y = build_xy(df_y.drop(columns='y'), df_y['y'], L)
            # 留最后一个样本用于测试
            X_pred, y_true = X[-1], y[-1]
            X_hist, y_hist = X[:-1], y[:-1]

            # —— 按 train_frac 划分：训练集 vs 校准集 ——
            n_hist = len(X_hist)
            n_tr = int(n_hist * cfg['train_frac'])
            X_tr, y_tr = X_hist[:n_tr], y_hist[:n_tr]
            X_cal, y_cal = X_hist[n_tr:], y_hist[n_tr:]
            # 校准集过小时取尾部
            if len(X_cal) < max(10, len(quants)):
                tail = max(30, len(quants))
                X_cal, y_cal = X_tr[-tail:], y_tr[-tail:]

            # —— 训练（整个训练集）+ 验证（整个校准集） ——
            model = train_gru_q(X_tr, y_tr, X_cal, y_cal, cfg, device)

            # —— 用同一校准集做 CQR 校准 ——
            δ_lo, δ_hi = calibrate_cqr(model, X_cal, y_cal, quants, cfg['conf_lvl'], device)

            # —— 预测 & 打印 ——
            model.eval()
            with torch.no_grad():
                q_pred = model(torch.tensor(X_pred[None], dtype=torch.float32, device=device)).cpu().numpy().flatten()
            lo, hi = infer_interval(q_pred, δ_lo, δ_hi)
            mid    = np.interp(0.5, quants, np.maximum.accumulate(q_pred))

            print(f"{date_pt.date()} | true={y_true:.4f} | [{lo:.4f}, {hi:.4f}] | mid={mid:.4f}")

            # —— 统计 ——
            cover_cnt += (lo <= y_true <= hi)
            width_sum += (hi - lo)
            total_cnt += 1

            e = abs(mid - y_true)
            err_abs[tag].append(e)
            if abs(y_true) > 1e-12:
                err_pct[tag].append(e/abs(y_true))

    # 汇总
    print("\n──────── SUMMARY ────────")
    if total_cnt == 0:
        print("No valid predictions.")
        return
    cov = cover_cnt/total_cnt
    print(f"Coverage @ {cfg['conf_lvl']*100:.1f}% : {cover_cnt}/{total_cnt} = {cov:.2%}")
    print(f"Avg. Interval Width       : {width_sum/total_cnt:.6f}\n")
    for tag, name in [('cc','Close-Close'),('park','Parkinson')]:
        if not err_abs[tag]:
            print(f"{name:<15}: No data")
            continue
        mae, medae = np.mean(err_abs[tag]), np.median(err_abs[tag])
        mape = np.mean(err_pct[tag])*100 if err_pct[tag] else float('nan')
        print(f"{name:<15}: MAE={mae:.6f}  |  MedAE={medae:.6f}  |  MAPE={mape:.2f}%")

# ───────────────────────── Main ─────────────────────────
if __name__ == "__main__":
    cfg = default_cfg()
    backtest(r"D:\prediction_market_data\crypto_data\merged_btc_data.csv", cfg)
