# Market Neural Network - Stock Market Prediction System

An end-to-end pipeline that tests whether neural networks can predict next-day stock direction from engineered technical features: data loading, 4 model architectures (Dense, LSTM, Attention/Transformer, Ensemble), a backtested trading strategy with risk management, and a full performance report against a buy-and-hold benchmark.

**Tech stack:** Python · TensorFlow/Keras · scikit-learn · pandas · yfinance

## Results

Trained on AAPL daily data (2022-01-01 to 2024-01-01), 56 engineered features (technical indicators, price/volume stats, lagged and time-based features), evaluated on a held-out 20% test split:

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|----------|-----------|--------|-----|---------|
| Dense | 0.491 | 0.588 | 0.317 | 0.412 | 0.520 |
| LSTM | 0.446 | 0.522 | 0.190 | 0.279 | 0.497 |
| Attention | 0.411 | 0.333 | 0.048 | 0.083 | 0.500 |
| Ensemble | 0.464 | 0.571 | 0.190 | 0.286 | 0.511 |

**None of the models beat chance-level prediction (AUC ≈ 0.50) on held-out data**, despite training accuracy climbing to 100% for several of them — a clear overfitting signal given ~340 training samples and 56 features. Backtesting the ensemble's trading signals produced a +0.77% return over the test window versus **+2.38% for simple buy-and-hold** — the strategy underperformed the market it was trying to beat.

This isn't a surprising result — daily-direction prediction from technical indicators alone is a genuinely hard (arguably close to impossible) problem, and the negative result here is more informative than a cherry-picked positive one would be. If I revisited this, I'd prioritize:
- A longer history and more tickers/sectors to shrink the overfitting gap
- Walk-forward validation instead of a single train/test split, given how noisy financial time series are
- Regressing toward simpler baselines (e.g. logistic regression on a handful of features) before adding architectural complexity that the data can't support

## Project Structure

- `main.py` — orchestrates data loading, training, evaluation, backtesting, and reporting
- `src/data/data_loader.py` — yfinance download + feature engineering + time-series splitting
- `src/models/neural_networks.py` — Dense, LSTM, Attention, and Ensemble model classes
- `src/strategies/trading_strategy.py` — signal-based trading strategy with stop loss/take profit
- `src/utils/metrics.py` — classification and portfolio performance metrics
- `src/visualization/plots.py` — stock, training, and strategy plots

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alexseveringhaus/Market-Neural-Network.git
   cd Market-Neural-Network
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Try It

Run the complete demonstration:

```bash
python main.py
```

This will:
1. Download and process stock data for AAPL
2. Engineer features using technical indicators
3. Train multiple neural network architectures
4. Create an ensemble model
5. Execute a trading strategy
6. Generate performance reports
7. Create visualizations for stock data and trading results
