# Realized_Vol_pred
🧠 **HVGRU Volatility Prediction Engine**

🎯 **Objective**
This engine employs a two-stage training GRU neural network (HVGRU) model to dynamically forecast future volatility measures for cryptocurrencies such as Bitcoin. It specifically predicts volatility metrics using Close-to-Close (CC) and Parkinson estimators over a defined forecast horizon.

📁 **Input Data Format**
The input CSV file must contain these columns:

* `date`: Trading date (YYYY-MM-DD)
* `open`, `high`, `low`, `close`: Daily OHLC prices
* `volume usdt`: Trading volume in USDT
* `iv`: Implied volatility (scaled to \[0,1])
* `fear_index`: Market fear index

🔍 **Preprocessing**

* Datetime Parsing: Convert `date` to datetime index and sort in ascending order.
* Volume Spike Transformation: Calculate the 10-day rolling mean and apply log-transformation.

🏗️ Feature Engineering

Rolling Standardization (Z-score)

Features:

hvH_lag: Historical volatility lagged by one day.

hl_pct_log: Intraday price volatility (log high-low percentage).

fear_index: Market sentiment indicator.

oc_ret: Open-to-close daily return.

vol_spike_10: Log-transformed volume spike ratio (current vs. 10-day average).

Window: PRICE_WINDOW = 60 days

Static Standardization

Features:

iv: Implied volatility from market expectations.

skewH: Skewness of returns over recent horizon.

kurtH: Kurtosis of returns over recent horizon.

Normalization parameters calculated solely from the training subset within each rolling window.

Raw Features

Features:

ret1_z: Z-score standardized daily return.

ret3_z: Z-score standardized 3-day cumulative return.

Included without additional transformations.

🪟 **Sliding Window & Sample Construction**

* **Key Parameters**

  * `train_window_days = 960`: Historical lookback period
  * `seq_len = 60`: GRU input sequence length
  * `PRICE_WINDOW = 60`: Rolling normalization window
  * `horizon = 5`: Predict future 5-day volatility


🧪 **Dataset Splitting (per window)**

* Initial split (Stage 1):

  * Training set: First 90% of samples
  * Validation set: Last 10% for EarlyStopping
  * 
* Final training (Stage 2): Entire window re-trained using optimal epoch determined in Stage 1.

🧠 **Model Training**

* **Architecture**

  * Two-layer GRU
  * LayerNorm and two-layer MLP head
  * Single-value output (volatility forecast)

* **Loss Function**

  * Smooth L1 Loss (Huber Loss)

* **Optimization**

  * Optimizer: AdamW (learning rate = 3e-4)
  * LR Scheduler: ReduceLROnPlateau, patience = 10
  * EarlyStopping: patience = 12 epochs

🔮 **Prediction & Evaluation**

* **Inference**

  * Model generates single-point forecasts for the future volatility
* **Metrics**

  * Mean Absolute Error (MAE)
  * Median Absolute Error (MedAE)
  * Mean Absolute Percentage Error (MAPE)

⚙️ **Default Configuration**

```python
def default_cfg():
    return dict(
        seq_len=60,
        batch=64,
        hidden=256,
        dropout=0.2,
        horizon=5,
        train_window_days=960,
        lr=3e-4,
        epochs=300,
        early_stop_patience=12
    )
```

📌 **Execution**
Run the script with your input CSV path and desired output path for results.

```python
if __name__ == "__main__":
    cfg = default_cfg()
    backtest(
        r"path/to/your/input.csv",
        cfg,
        r"path/to/save/results.csv"
    )
```
