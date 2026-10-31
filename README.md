# HVGRU Volatility Prediction Engine

## Objective
This engine employs a two-stage GRU neural network (HVGRU) with rolling-window backtesting to forecast future volatility (5-day horizon) for cryptocurrencies. It specifically predicts volatility metrics using two estimators: Close-to-Close (CC) and Parkinson.

## Input Data Format
The input CSV file must contain these columns:

* `date`: Trading date (YYYY-MM-DD)
* `open`, `high`, `low`, `close`: OHLC prices
* `volume usdt`: Trading volume in USDT
* `iv`: Implied volatility (scaled 0-1)
* `fear_index`: Market fear index

## Preprocessing

* Datetime Parsing: Convert `date` to datetime index and sort ascendingly.
* Volume Spike Transformation: Compute 10-day rolling mean and apply log transformation.

## Feature Engineering

* **Rolling Standardization (Z-score)**

  * Features:

    * `hvH_lag`: Historical volatility lagged by one day.
    * `hl_pct_log`: Intraday price volatility (log high-low percentage).
    * `fear_index`: Market sentiment indicator.
    * `oc_ret`: Open-to-close daily return.
    * `vol_spike_10`: Log-transformed volume spike ratio (current vs. 10-day average).
  * Window: `PRICE_WINDOW = 60` days

* **Static Standardization**

  * Features:

    * `iv`: Implied volatility from market expectations.
    * `skewH`: Skewness of returns over recent horizon.
    * `kurtH`: Kurtosis of returns over recent horizon.
  * Normalization parameters calculated solely from the training subset within each rolling window.

* **Raw Features**

  * Features:

    * `ret1_z`: Z-score standardized daily return.
    * `ret3_z`: Z-score standardized 3-day cumulative return.
  * Included without additional transformations.

## Sliding Window & Sample Construction

* **Parameters:**

  * Training window (`train_window_days`): 960 days
  * GRU input sequence length (`seq_len`): 60 days
  * Prediction horizon (`horizon`): 5 days


## Dataset Splitting (per window)

* Stage 1:

  * Training: 90%, Validation: 10% (Early Stopping)
* Stage 2:

  * Retrain on entire dataset using best epoch from Stage 1 + 5 additional epochs.

## Model Training

* **Architecture**:

  * Two-layer GRU → LayerNorm → two-layer MLP → single-value output
* **Loss Function**:

  * Smooth L1 Loss (Huber Loss)
* **Optimization**:

  * AdamW optimizer (LR = 3e-4)
  * ReduceLROnPlateau scheduler, patience = 10
  * EarlyStopping, patience = 12 epochs

## Prediction & Evaluation

* **Metrics:**

  * Mean Absolute Error (MAE)
  * Median Absolute Error (MedAE)
  * Mean Absolute Percentage Error (MAPE)

## Default Configuration

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

## Execution
Run the script with your input CSV path and desired output path:

```python
if __name__ == "__main__":
    cfg = default_cfg()
    backtest(
        r"path/to/your/input.csv",
        cfg,
        r"path/to/save/results.csv"
    )
```
