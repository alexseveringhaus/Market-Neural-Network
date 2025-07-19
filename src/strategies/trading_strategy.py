"""
Trading Strategy Module

This module implements various trading strategies using machine learning predictions,
including position sizing, risk management, and performance evaluation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class Trade:
    """Represents a single trade."""
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position_size: float
    side: str  # 'long' or 'short'
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None


@dataclass
class StrategyResult:
    """Results of a trading strategy."""
    trades: List[Trade]
    portfolio_values: pd.Series
    returns: pd.Series
    metrics: Dict[str, float]


class BaseTradingStrategy:
    """Base class for trading strategies."""
    
    def __init__(self, initial_capital: float = 100000):
        """
        Initialize the trading strategy.
        
        Args:
            initial_capital: Initial capital to trade with
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.positions = {}
        
    def generate_signals(self, predictions: np.ndarray, confidence_threshold: float = 0.6) -> np.ndarray:
        """
        Generate trading signals from predictions.
        
        Args:
            predictions: Model predictions (probabilities)
            confidence_threshold: Minimum confidence for taking a position
            
        Returns:
            Array of signals: 1 (long), -1 (short), 0 (no position)
        """
        signals = np.zeros_like(predictions)
        
        # Balanced signal generation with moderate thresholds
        # Long signals for bullish predictions
        long_mask = predictions > 0.53  # 53% threshold
        signals[long_mask] = 1
        
        # Short signals for bearish predictions
        short_mask = predictions < 0.47  # 47% threshold
        signals[short_mask] = -1
        
        # Simplified trend confirmation - less restrictive
        for i in range(3, len(signals)):
            if signals[i] != 0:
                # Check if signal aligns with recent trend (last 3 days)
                recent_trend = 1 if predictions[i-3:i].mean() > 0.5 else -1
                # Only cancel if strongly against trend (not just slightly)
                if abs(predictions[i] - 0.5) < 0.05 and signals[i] != recent_trend:
                    signals[i] = 0
        
        return signals
        
    def calculate_position_size(
        self, 
        signal: float, 
        price: float, 
        volatility: float,
        risk_per_trade: float = 0.02
    ) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            signal: Trading signal (-1, 0, 1)
            price: Current price
            volatility: Current volatility
            risk_per_trade: Maximum risk per trade as fraction of capital
            
        Returns:
            Position size in number of shares
        """
        if signal == 0:
            return 0
            
        # Conservative position sizing with volatility adjustment
        # Base position size on risk per trade
        risk_amount = self.current_capital * risk_per_trade
        
        # Volatility adjustment - less restrictive
        volatility_adjustment = max(0.6, 1 / (1 + volatility * 5))  # Minimum 60% of normal size (was 30%)
        
        # Market condition adjustment - less restrictive
        recent_trades = [t for t in self.trades[-3:] if t.exit_date is not None]
        if recent_trades:
            recent_losses = sum(1 for t in recent_trades if t.pnl < 0)
            loss_adjustment = max(0.7, 1 - (recent_losses / len(recent_trades)) * 0.3)  # Minimum 70% (was 50%)
        else:
            loss_adjustment = 1.0
        
        # Calculate final position size
        position_value = risk_amount * volatility_adjustment * loss_adjustment
        
        return position_value / price
        
    def execute_trade(
        self, 
        date: pd.Timestamp, 
        price: float, 
        signal: float, 
        position_size: float
    ):
        """
        Execute a trade.
        
        Args:
            date: Trade date
            price: Trade price
            signal: Trading signal
            position_size: Position size
        """
        if signal != 0 and position_size > 0:
            trade = Trade(
                entry_date=date,
                exit_date=None,
                entry_price=price,
                exit_price=None,
                position_size=position_size,
                side='long' if signal > 0 else 'short'
            )
            self.trades.append(trade)
            
            # Update capital
            trade_value = position_size * price
            self.current_capital -= trade_value
            
    def close_position(self, date: pd.Timestamp, price: float, trade: Trade):
        """
        Close a position.
        
        Args:
            date: Close date
            price: Close price
            trade: Trade to close
        """
        trade.exit_date = date
        trade.exit_price = price
        
        # Calculate P&L
        if trade.side == 'long':
            trade.pnl = (price - trade.entry_price) * trade.position_size
        else:  # short
            trade.pnl = (trade.entry_price - price) * trade.position_size
            
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.position_size)
        
        # Update capital
        self.current_capital += trade.position_size * price + trade.pnl


class MLTradingStrategy(BaseTradingStrategy):
    """Machine learning-based trading strategy."""
    
    def __init__(
        self, 
        initial_capital: float = 100000,
        confidence_threshold: float = 0.6,
        risk_per_trade: float = 0.02,
        max_positions: int = 5,
        stop_loss: float = 0.05,
        take_profit: float = 0.10
    ):
        """
        Initialize ML trading strategy.
        
        Args:
            initial_capital: Initial capital
            confidence_threshold: Minimum confidence for trades
            risk_per_trade: Maximum risk per trade
            max_positions: Maximum number of concurrent positions
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        super().__init__(initial_capital)
        self.confidence_threshold = confidence_threshold
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.portfolio_values = []
        self.dates = []
        
    def run_strategy(
        self, 
        data: pd.DataFrame, 
        predictions: np.ndarray,
        volatility_window: int = 20
    ) -> StrategyResult:
        """
        Run the trading strategy.
        
        Args:
            data: Market data
            predictions: Model predictions
            volatility_window: Window for volatility calculation
            
        Returns:
            Strategy results
        """
        # Calculate volatility
        data['Volatility'] = data['Returns'].rolling(window=volatility_window).std()
        
        # Generate signals
        signals = self.generate_signals(predictions, self.confidence_threshold)
        
        # Initialize tracking
        self.portfolio_values = [self.initial_capital]
        self.dates = [data.index[0]]
        
        for i, (date, row) in enumerate(data.iterrows()):
            # Check for stop loss and take profit on existing positions
            self._check_exit_conditions(date, row['Close'])
            
            # Generate new signals if we have capacity
            if len([t for t in self.trades if t.exit_date is None]) < self.max_positions:
                signal = signals[i] if i < len(signals) else 0
                
                if signal != 0:
                    volatility = row['Volatility'] if not pd.isna(row['Volatility']) else 0.02
                    
                    # Market condition filter - less restrictive volatility filter
                    if volatility > 0.08:  # Skip trades if volatility > 8% (was 5%)
                        continue
                    
                    # Check for recent losses - pause trading after consecutive losses
                    recent_trades = [t for t in self.trades[-5:] if t.exit_date is not None]
                    if len(recent_trades) >= 5:
                        recent_losses = sum(1 for t in recent_trades if t.pnl < 0)
                        if recent_losses >= 5:  # Pause after 5 consecutive losses (was 3)
                            continue
                    
                    position_size = self.calculate_position_size(
                        signal, row['Close'], volatility, self.risk_per_trade
                    )
                    
                    if position_size > 0:
                        self.execute_trade(date, row['Close'], signal, position_size)
            
            # Update portfolio value
            current_value = self._calculate_portfolio_value(row['Close'])
            self.portfolio_values.append(current_value)
            self.dates.append(date)
        
        # Close any remaining positions
        self._close_all_positions(data.index[-1], data.iloc[-1]['Close'])
        
        # Calculate final portfolio value
        final_value = self._calculate_portfolio_value(data.iloc[-1]['Close'])
        self.portfolio_values.append(final_value)
        self.dates.append(data.index[-1])
        
        # Create results
        portfolio_series = pd.Series(self.portfolio_values, index=self.dates)
        returns = portfolio_series.pct_change().dropna()
        
        metrics = self._calculate_metrics(portfolio_series, returns, data)
        
        return StrategyResult(
            trades=self.trades,
            portfolio_values=portfolio_series,
            returns=returns,
            metrics=metrics
        )
        
    def _check_exit_conditions(self, date: pd.Timestamp, current_price: float):
        """Check stop loss and take profit conditions with dynamic adjustments."""
        for trade in self.trades:
            if trade.exit_date is None:
                # Calculate dynamic stop loss based on volatility
                # Get recent volatility for this trade
                trade_duration = (date - trade.entry_date).days
                if trade_duration > 0:
                    # Adjust stop loss based on time in trade and volatility
                    time_adjustment = min(1.5, 1 + trade_duration * 0.02)  # Increase stop loss over time
                    dynamic_stop_loss = self.stop_loss * time_adjustment
                else:
                    dynamic_stop_loss = self.stop_loss
                
                if trade.side == 'long':
                    # Stop loss
                    if current_price <= trade.entry_price * (1 - dynamic_stop_loss):
                        self.close_position(date, current_price, trade)
                    # Take profit
                    elif current_price >= trade.entry_price * (1 + self.take_profit):
                        self.close_position(date, current_price, trade)
                    # Time-based exit - close if trade is open too long
                    elif trade_duration > 60:  # Close after 60 days (was 30)
                        self.close_position(date, current_price, trade)
                else:  # short
                    # Stop loss
                    if current_price >= trade.entry_price * (1 + dynamic_stop_loss):
                        self.close_position(date, current_price, trade)
                    # Take profit
                    elif current_price <= trade.entry_price * (1 - self.take_profit):
                        self.close_position(date, current_price, trade)
                    # Time-based exit - close if trade is open too long
                    elif trade_duration > 60:  # Close after 60 days (was 30)
                        self.close_position(date, current_price, trade)
                        
    def _close_all_positions(self, date: pd.Timestamp, price: float):
        """Close all open positions."""
        for trade in self.trades:
            if trade.exit_date is None:
                self.close_position(date, price, trade)
                
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        portfolio_value = self.current_capital
        
        for trade in self.trades:
            if trade.exit_date is None:
                if trade.side == 'long':
                    portfolio_value += trade.position_size * current_price
                else:  # short
                    portfolio_value += trade.position_size * (2 * trade.entry_price - current_price)
                    
        return portfolio_value
        
    def _calculate_metrics(
        self, 
        portfolio_values: pd.Series, 
        returns: pd.Series, 
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate performance metrics."""
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate and average trade
        completed_trades = [t for t in self.trades if t.exit_date is not None]
        if completed_trades:
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            win_rate = len(winning_trades) / len(completed_trades)
            avg_trade = np.mean([t.pnl for t in completed_trades])
            avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t.pnl for t in completed_trades if t.pnl < 0]) if any(t.pnl < 0 for t in completed_trades) else 0
        else:
            win_rate = avg_trade = avg_win = avg_loss = 0
            
        # Market comparison
        market_returns = market_data['Returns'].dropna()
        market_total_return = (1 + market_returns).prod() - 1
        market_annualized = (1 + market_total_return) ** (252 / len(market_returns)) - 1
        excess_return = annualized_return - market_annualized
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_trade': avg_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'total_trades': len(completed_trades),
            'excess_return': excess_return,
            'market_return': market_annualized
        }


class PortfolioOptimizer:
    """Portfolio optimization using modern portfolio theory."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize portfolio optimizer.
        
        Args:
            risk_free_rate: Risk-free rate
        """
        self.risk_free_rate = risk_free_rate
        
    def optimize_weights(
        self, 
        returns: pd.DataFrame, 
        method: str = 'sharpe'
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Optimize portfolio weights.
        
        Args:
            returns: Returns DataFrame
            method: Optimization method ('sharpe', 'min_variance', 'equal_weight')
            
        Returns:
            Tuple of (weights, metrics)
        """
        if method == 'equal_weight':
            n_assets = len(returns.columns)
            weights = np.ones(n_assets) / n_assets
        else:
            # Calculate covariance matrix
            cov_matrix = returns.cov() * 252  # Annualized
            
            if method == 'min_variance':
                weights = self._min_variance_weights(cov_matrix)
            else:  # sharpe
                weights = self._max_sharpe_weights(returns, cov_matrix)
                
        # Calculate portfolio metrics
        portfolio_return = (returns * weights).sum(axis=1)
        metrics = self._calculate_portfolio_metrics(portfolio_return)
        
        return weights, metrics
        
    def _min_variance_weights(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Calculate minimum variance weights."""
        n_assets = len(cov_matrix)
        
        # Objective: minimize portfolio variance
        from scipy.optimize import minimize
        
        def objective(weights):
            return weights.T @ cov_matrix.values @ weights
            
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (long-only)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
        
    def _max_sharpe_weights(
        self, 
        returns: pd.DataFrame, 
        cov_matrix: pd.DataFrame
    ) -> np.ndarray:
        """Calculate maximum Sharpe ratio weights."""
        n_assets = len(returns.columns)
        
        # Calculate expected returns
        expected_returns = returns.mean() * 252
        
        from scipy.optimize import minimize
        
        def objective(weights):
            portfolio_return = np.sum(weights * expected_returns)
            portfolio_vol = np.sqrt(weights.T @ cov_matrix.values @ weights)
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimize negative Sharpe ratio
            
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (long-only)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess
        initial_weights = np.ones(n_assets) / n_assets
        
        result = minimize(objective, initial_weights, method='SLSQP',
                         bounds=bounds, constraints=constraints)
        
        return result.x
        
    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        total_return = (1 + returns).prod() - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        } 