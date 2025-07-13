# Market Neural Network - Stock Market Prediction System

A machine learning system for predicting stock market trends using multiple neural network architectures, feature engineering, and a simulated trading strategy. The project demonstrates loading and processing stock data, training 3 neural network models, combining their predictions in an ensemble, and simulating a trading strategy based on those predictions. Performance is evaluated using common financial metrics, and results are visualized.

## Features

### Neural Network Architectures
- **Dense Neural Networks** with regularization and dropout
- **LSTM Networks** for time series modeling
- **Attention Mechanisms** with transformer blocks
- **Ensemble Models** combining multiple architectures

### Feature Engineering
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, ATR
- **Price-based Features**: Momentum, volatility, skewness, kurtosis
- **Volume Analysis**: Volume ratios, moving averages
- **Time-based Features**: Day of week, month, quarter effects
- **Lag Features**: Multiple time-lagged variables

### Trading Strategy
- **Machine Learning-based Signals** with confidence thresholds
- **Risk Management**: Fixed risk per trade, stop loss, and take profit
- **Performance Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, Max Drawdown

### Evaluation
- **Performance Analysis**: Alpha, Beta, Information ratio, Upside/Downside capture
- **Trade Analysis**: Win rate, profit factor, consecutive wins/losses
- **Risk Metrics**: Maximum drawdown

### Visualizations
- **Stock and Technical Indicator Plots**
- **Trading Strategy Performance Plots**
- **Trade Analysis Visualizations**

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

## Quick Start

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

## Technical Details

### Neural Network Architectures

1. **Dense Neural Network**
   - Multiple hidden layers with ReLU activation
   - Batch normalization and dropout for regularization
   - L2 regularization to prevent overfitting

2. **LSTM Network**
   - Long Short-Term Memory layers for temporal dependencies
   - Dense layers for final classification

3. **Attention Network**
   - Multi-head attention mechanisms
   - Transformer blocks for pattern learning
   - Global average pooling for feature aggregation

### Feature Engineering

The system creates features including:
- **Price-based**: Returns, momentum, volatility measures
- **Technical**: RSI, MACD, Bollinger Bands, Stochastic
- **Volume**: Volume ratios, moving averages
- **Statistical**: Rolling mean, std, skewness, kurtosis
- **Temporal**: Day of week, month, quarter effects

### Risk Management

- **Position Sizing**: Fixed risk per trade
- **Portfolio Limits**: Maximum concurrent positions
- **Stop Loss**: Automatic position closure at loss threshold
- **Take Profit**: Automatic position closure at profit target