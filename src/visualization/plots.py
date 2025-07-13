"""
Visualization Module

This module provides plotting and visualization capabilities
for stock market analysis and trading strategy evaluation.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class StockVisualizer:
    """Stock market visualization class."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_stock_data(
        self, 
        data: pd.DataFrame, 
        title: str = "Stock Price Analysis",
        show_volume: bool = True,
        show_indicators: bool = True
    ):
        """
        Create a stock price chart.
        
        Args:
            data: Stock data DataFrame
            title: Chart title
            show_volume: Whether to show volume
            show_indicators: Whether to show technical indicators
        """
        fig, axes = plt.subplots(3 if show_volume else 2, 1, figsize=self.figsize, 
                                gridspec_kw={'height_ratios': [3, 1, 1] if show_volume else [3, 1]})
        
        if show_volume:
            price_ax, volume_ax, indicator_ax = axes
        else:
            price_ax, indicator_ax = axes
            
        # Price chart
        price_ax.plot(data.index, data['Close'], label='Close Price', linewidth=2)
        
        if show_indicators:
            if 'SMA_20' in data.columns:
                price_ax.plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
            if 'SMA_50' in data.columns:
                price_ax.plot(data.index, data['SMA_50'], label='SMA 50', alpha=0.7)
            if 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                price_ax.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], 
                                    alpha=0.2, label='Bollinger Bands')
        
        price_ax.set_title(title, fontsize=16, fontweight='bold')
        price_ax.set_ylabel('Price ($)', fontsize=12)
        price_ax.legend()
        price_ax.grid(True, alpha=0.3)
        
        # Volume chart
        if show_volume:
            volume_ax.bar(data.index, data['Volume'], alpha=0.7, color='blue')
            volume_ax.set_ylabel('Volume', fontsize=12)
            volume_ax.grid(True, alpha=0.3)
            
        # Technical indicators
        if show_indicators:
            if 'RSI' in data.columns:
                indicator_ax.plot(data.index, data['RSI'], label='RSI', color='purple')
                indicator_ax.axhline(y=70, color='r', linestyle='--', alpha=0.7)
                indicator_ax.axhline(y=30, color='g', linestyle='--', alpha=0.7)
                indicator_ax.set_ylabel('RSI', fontsize=12)
                indicator_ax.set_ylim(0, 100)
                indicator_ax.legend()
                indicator_ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def plot_returns_distribution(self, returns: pd.Series, title: str = "Returns Distribution"):
        """
        Plot returns distribution with statistical information.
        
        Args:
            returns: Returns series
            title: Plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # Histogram
        axes[0, 0].hist(returns, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Returns Histogram')
        axes[0, 0].set_xlabel('Returns')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(returns.dropna(), dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Box plot
        axes[1, 0].boxplot(returns.dropna())
        axes[1, 0].set_title('Returns Box Plot')
        axes[1, 0].set_ylabel('Returns')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Rolling volatility
        rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
        axes[1, 1].plot(rolling_vol.index, rolling_vol.values, color='red')
        axes[1, 1].set_title('Rolling Volatility (20-day)')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Volatility')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_correlation_matrix(self, data: pd.DataFrame, title: str = "Feature Correlation Matrix"):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data: DataFrame with features
            title: Plot title
        """
        # Calculate correlation matrix
        corr_matrix = data.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        plt.figure(figsize=self.figsize)
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_feature_importance(self, feature_names: List[str], importance_scores: np.ndarray, 
                               title: str = "Feature Importance"):
        """
        Plot feature importance scores.
        
        Args:
            feature_names: List of feature names
            importance_scores: Feature importance scores
            title: Plot title
        """
        # Sort features by importance
        sorted_idx = np.argsort(importance_scores)[::-1]
        sorted_names = [feature_names[i] for i in sorted_idx]
        sorted_scores = importance_scores[sorted_idx]
        
        # Plot top 20 features
        top_n = min(20, len(sorted_names))
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(top_n), sorted_scores[:top_n], color='skyblue')
        plt.yticks(range(top_n), sorted_names[:top_n])
        plt.xlabel('Importance Score')
        plt.title(title, fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for i, bar in enumerate(bars):
            width = bar.get_width()
            plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left', va='center')
        
        plt.tight_layout()
        plt.show()


class TradingVisualizer:
    """Visualization class for trading strategies."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the trading visualizer.
        
        Args:
            figsize: Default figure size
        """
        self.figsize = figsize
        
    def plot_strategy_performance(
        self, 
        strategy_result: Any,  # StrategyResult type
        market_data: pd.DataFrame,
        title: str = "Trading Strategy Performance"
    ):
        """
        Plot strategy performance analysis.
        
        Args:
            strategy_result: Results from trading strategy
            market_data: Market data for comparison
            title: Plot title
        """
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # Portfolio value vs market
        market_cumulative = (1 + market_data['Returns']).cumprod()
        strategy_cumulative = strategy_result.portfolio_values / strategy_result.portfolio_values.iloc[0]
        
        axes[0, 0].plot(strategy_cumulative.index, strategy_cumulative.values, 
                       label='Strategy', linewidth=2)
        axes[0, 0].plot(market_cumulative.index, market_cumulative.values, 
                       label='Market', linewidth=2, alpha=0.7)
        axes[0, 0].set_title('Cumulative Returns')
        axes[0, 0].set_ylabel('Cumulative Return')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Drawdown
        strategy_returns = strategy_result.returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        axes[0, 1].fill_between(drawdown.index, drawdown.values, 0, 
                               color='red', alpha=0.3)
        axes[0, 1].plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        axes[0, 1].set_title('Drawdown')
        axes[0, 1].set_ylabel('Drawdown')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Monthly returns heatmap
        strategy_returns_df = pd.DataFrame({'Returns': strategy_returns})
        strategy_returns_df.index = pd.to_datetime(strategy_returns_df.index)
        monthly_returns = strategy_returns_df.groupby([strategy_returns_df.index.year, 
                                                     strategy_returns_df.index.month])['Returns'].sum()
        
        monthly_returns_pivot = monthly_returns.unstack()
        
        sns.heatmap(monthly_returns_pivot, annot=True, fmt='.2%', cmap='RdYlGn', 
                   center=0, ax=axes[1, 0])
        axes[1, 0].set_title('Monthly Returns Heatmap')
        
        # Trade analysis
        completed_trades = [t for t in strategy_result.trades if t.exit_date is not None]
        if completed_trades:
            trade_pnls = [t.pnl for t in completed_trades]
            trade_durations = [(t.exit_date - t.entry_date).days for t in completed_trades]
            
            axes[1, 1].scatter(trade_durations, trade_pnls, alpha=0.6)
            axes[1, 1].set_xlabel('Trade Duration (days)')
            axes[1, 1].set_ylabel('Trade P&L ($)')
            axes[1, 1].set_title('Trade Duration vs P&L')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        rolling_sharpe = strategy_returns.rolling(window=60).mean() / strategy_returns.rolling(window=60).std() * np.sqrt(252)
        axes[2, 0].plot(rolling_sharpe.index, rolling_sharpe.values, color='green')
        axes[2, 0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2, 0].set_title('Rolling Sharpe Ratio (60-day)')
        axes[2, 0].set_ylabel('Sharpe Ratio')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Performance metrics table
        metrics = strategy_result.metrics
        metric_names = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        metric_values = [
            f"{metrics['total_return']:.2%}",
            f"{metrics['annualized_return']:.2%}",
            f"{metrics['sharpe_ratio']:.2f}",
            f"{metrics['max_drawdown']:.2%}",
            f"{metrics['win_rate']:.2%}"
        ]
        
        axes[2, 1].axis('tight')
        axes[2, 1].axis('off')
        table = axes[2, 1].table(cellText=list(zip(metric_names, metric_values)),
                                colLabels=['Metric', 'Value'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.5)
        axes[2, 1].set_title('Performance Metrics')
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_trade_analysis(self, trades: List[Any], title: str = "Trade Analysis"):  # List[Trade] type
        """
        Trade analysis visualization.
        
        Args:
            trades: List of completed trades
            title: Plot title
        """
        if not trades:
            print("No completed trades to analyze")
            return
            
        completed_trades = [t for t in trades if t.exit_date is not None]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # P&L distribution
        pnls = [t.pnl for t in completed_trades]
        axes[0, 0].hist(pnls, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[0, 0].set_title('P&L Distribution')
        axes[0, 0].set_xlabel('P&L ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Cumulative P&L
        cumulative_pnl = np.cumsum(pnls)
        axes[0, 1].plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2, color='green')
        axes[0, 1].set_title('Cumulative P&L')
        axes[0, 1].set_xlabel('Trade Number')
        axes[0, 1].set_ylabel('Cumulative P&L ($)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Win/Loss ratio over time
        wins = [1 if t.pnl > 0 else 0 for t in completed_trades]
        win_rate_rolling = pd.Series(wins).rolling(window=20).mean()
        axes[0, 2].plot(range(len(win_rate_rolling)), win_rate_rolling.values, color='blue')
        axes[0, 2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        axes[0, 2].set_title('Rolling Win Rate (20 trades)')
        axes[0, 2].set_xlabel('Trade Number')
        axes[0, 2].set_ylabel('Win Rate')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Trade duration analysis
        durations = [(t.exit_date - t.entry_date).days for t in completed_trades]
        axes[1, 0].hist(durations, bins=15, alpha=0.7, color='orange', edgecolor='black')
        axes[1, 0].set_title('Trade Duration Distribution')
        axes[1, 0].set_xlabel('Duration (days)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # P&L vs Duration scatter
        axes[1, 1].scatter(durations, pnls, alpha=0.6, color='purple')
        axes[1, 1].set_xlabel('Duration (days)')
        axes[1, 1].set_ylabel('P&L ($)')
        axes[1, 1].set_title('P&L vs Trade Duration')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Monthly performance
        trade_df = pd.DataFrame({
            'exit_date': [t.exit_date for t in completed_trades],
            'pnl': pnls
        })
        trade_df['exit_date'] = pd.to_datetime(trade_df['exit_date'])
        monthly_pnl = trade_df.groupby([trade_df['exit_date'].dt.year, 
                                       trade_df['exit_date'].dt.month])['pnl'].sum()
        
        monthly_pnl.plot(kind='bar', ax=axes[1, 2], color='lightgreen')
        axes[1, 2].set_title('Monthly P&L')
        axes[1, 2].set_xlabel('Year-Month')
        axes[1, 2].set_ylabel('P&L ($)')
        axes[1, 2].tick_params(axis='x', rotation=45)
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show() 