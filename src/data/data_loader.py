"""
Data Loading and Preprocessing Module

This module handles data acquisition from multiple sources, preprocessing,
and feature engineering for stock market prediction.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
import ta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

class DataLoader:
    """
    Data loader for stock market data with multiple sources and preprocessing.
    """
    
    def __init__(self, cache_data: bool = True):
        """
        Initialize the data loader.
        
        Args:
            cache_data: Whether to cache downloaded data
        """
        self.cache_data = cache_data
        self.cached_data = {}
        self.scaler = None
        
    def get_stock_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Download stock data from Yahoo Finance with error handling.
        
        Args:
            symbol: Stock symbol
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            interval: Data interval (1d, 1h, etc.)
            
        Returns:
            DataFrame with stock data
        """
        try:
            cache_key = f"{symbol}_{start_date}_{end_date}_{interval}"
            
            if self.cache_data and cache_key in self.cached_data:
                print(f"Using cached data for {symbol}")
                return self.cached_data[cache_key]
            
            print(f"Downloading data for {symbol} from {start_date} to {end_date}")
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            
            if data is None:
                raise ValueError(f"No data found for symbol {symbol}")
                
            # Basic data validation
            data = self._validate_and_clean_data(data)
            
            if self.cache_data:
                self.cached_data[cache_key] = data
                
            return data
            
        except Exception as e:
            print(f"Error downloading data for {symbol}: {str(e)}")
            raise
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean the downloaded data.
        
        Args:
            data: Raw stock data
            
        Returns:
            Cleaned DataFrame
        """
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join([str(i) for i in col if i]) for col in data.columns.values]
            # Rename columns to remove ticker suffix (e.g., 'Close_AAPL' -> 'Close')
            data.columns = [col.split('_')[0] for col in data.columns]
        
        # Remove rows with missing values
        data = data.dropna()
        
        # Ensure we have required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Add basic derived features
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        return data

    def engineer_features(self, data: pd.DataFrame, window_sizes: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Engineer features for machine learning.
        
        Args:
            data: Stock data DataFrame
            window_sizes: List of window sizes for rolling features
            
        Returns:
            DataFrame with engineered features
        """
        if window_sizes is None:
            window_sizes = [5, 10, 20, 50]
            
        df = data.copy()
        
        # Price-based features
        df['Price_Range'] = (df['High'] - df['Low']) / df['Close']
        df['Price_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Technical indicators using ta library
        df = self._add_technical_indicators(df)
        
        # Rolling statistics
        for window in window_sizes:
            df[f'Returns_MA_{window}'] = df['Returns'].rolling(window=window).mean()
            df[f'Returns_Std_{window}'] = df['Returns'].rolling(window=window).std()
            df[f'Returns_Skew_{window}'] = df['Returns'].rolling(window=window).skew()
            df[f'Returns_Kurt_{window}'] = df['Returns'].rolling(window=window).kurt()
            
            # Price momentum
            df[f'Price_Momentum_{window}'] = df['Close'] / df['Close'].shift(window) - 1
            
            # Volatility
            df[f'Volatility_{window}'] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'Returns_Lag_{lag}'] = df['Returns'].shift(lag)
            df[f'Volume_Lag_{lag}'] = df['Volume'].shift(lag)
        
        # Time-based features
        dt_index = pd.to_datetime(df.index)
        df['Day_of_Week'] = dt_index.weekday
        df['Month'] = dt_index.month
        df['Quarter'] = dt_index.quarter

        return df
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators using the ta library.
        
        Args:
            df: Stock data DataFrame
            
        Returns:
            DataFrame with technical indicators
        """
        # Trend indicators
        from ta.trend import SMAIndicator, EMAIndicator, MACD
        from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
        from ta.volatility import BollingerBands, AverageTrueRange

        close_series = pd.Series(df['Close'])
        df['SMA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
        df['SMA_50'] = SMAIndicator(close=close_series, window=50).sma_indicator()
        df['EMA_12'] = EMAIndicator(close=close_series, window=12).ema_indicator()
        df['EMA_26'] = EMAIndicator(close=close_series, window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=close_series)
        df['MACD'] = macd.macd_diff()
        df['MACD_Signal'] = macd.macd_signal()
        
        # RSI
        df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
        
        # Bollinger Bands
        bb = BollingerBands(close=close_series)
        df['BB_Upper'] = bb.bollinger_hband()
        df['BB_Lower'] = bb.bollinger_lband()
        df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / df['BB_Width']
        
        # Stochastic
        high_series = pd.Series(df['High'])
        low_series = pd.Series(df['Low'])
        stoch = StochasticOscillator(high=high_series, low=low_series, close=close_series)
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        # Williams %R
        df['Williams_R'] = WilliamsRIndicator(high=high_series, low=low_series, close=close_series).williams_r()
        
        # ATR (Average True Range)
        df['ATR'] = AverageTrueRange(high=high_series, low=low_series, close=close_series).average_true_range()
        
        return df
    
    def prepare_features(
        self, 
        data: pd.DataFrame, 
        target_column: str = 'Returns',
        prediction_horizon: int = 1,
        scale_features: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
        """
        Prepare features and target for machine learning.
        
        Args:
            data: DataFrame with engineered features
            target_column: Column to use as target
            prediction_horizon: Number of periods ahead to predict
            scale_features: Whether to scale features
            
        Returns:
            Tuple of (X, y, scaler)
        """
        # Remove rows with NaN values
        data_clean = data.dropna()
        
        # Create target variable (future returns)
        y = np.where(data_clean[target_column].shift(-prediction_horizon) > 0, 1, 0)
        
        # Remove the last rows where we don't have future data
        y = y[:-prediction_horizon]
        data_clean = data_clean[:-prediction_horizon]
        
        # Select feature columns (exclude target and date-related columns)
        exclude_columns = [
            target_column, 'Log_Returns', 'Returns_Lag_1', 'Returns_Lag_2', 
            'Returns_Lag_3', 'Returns_Lag_5', 'Returns_Lag_10'
        ]
        
        feature_columns = [col for col in data_clean.columns 
                          if col not in exclude_columns and not col.startswith('Returns_Lag_')]
        
        X = np.array(data_clean[feature_columns])
        
        if scale_features:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        
        return X, y, self.scaler
    
    def create_time_series_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        n_splits: int = 5
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create time series cross-validation splits.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_splits: Number of splits
            
        Returns:
            List of (X_train, X_test, y_train, y_test) tuples
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits = []
        
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            splits.append((X_train, X_test, y_train, y_test))
        
        return splits 