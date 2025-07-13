"""
Metrics and Evaluation Module

This module provides evaluation metrics for trading strategies
and machine learning models.
"""

# Standard library imports
from typing import List, Tuple, Dict, Optional, Any
import warnings

# Third-party imports
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')


class TradingMetrics:
    """Trading strategy evaluation metrics."""
    
    @staticmethod
    def calculate_returns(prices: pd.Series) -> pd.Series:
        """Calculate simple returns."""
        return prices.pct_change().dropna()
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate log returns."""
        return np.log(prices / prices.shift(1)).dropna()
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod()
    
    @staticmethod
    def calculate_annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized return."""
        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        return (1 + total_return) ** (1 / years) - 1
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate annualized volatility."""
        return returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, 
                              periods_per_year: int = 252) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate / periods_per_year
        return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                               periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio."""
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
        return excess_returns.mean() / downside_deviation * np.sqrt(periods_per_year)
    
    @staticmethod
    def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Timestamp, pd.Timestamp]:
        """Calculate maximum drawdown and its duration."""
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find the peak before the maximum drawdown
        peak_idx = cumulative_returns[:max_dd_idx].idxmax()
        
        return max_dd, peak_idx, max_dd_idx
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio."""
        annualized_return = TradingMetrics.calculate_annualized_return(returns, periods_per_year)
        max_dd, _, _ = TradingMetrics.calculate_max_drawdown(returns)
        return annualized_return / abs(max_dd) if max_dd != 0 else 0
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk."""
        return float(np.percentile(returns, confidence_level * 100))
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        var = TradingMetrics.calculate_var(returns, confidence_level)
        return float(returns[returns <= var].mean())
    
    @staticmethod
    def calculate_win_rate(trades: List[Dict]) -> float:
        """Calculate win rate from trades."""
        if not trades:
            return 0.0
        winning_trades = [t for t in trades if t.get('pnl', 0) is not None and float(t.get('pnl', 0)) > 0]
        return float(len(winning_trades)) / float(len(trades)) if trades else 0.0
    
    @staticmethod
    def calculate_profit_factor(trades: List[Dict]) -> float:
        """Calculate profit factor."""
        if not trades:
            return 0.0
        gross_profit = float(sum(float(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) is not None and float(t.get('pnl', 0)) > 0))
        gross_loss = abs(float(sum(float(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) is not None and float(t.get('pnl', 0)) < 0)))
        return gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    @staticmethod
    def calculate_average_trade(trades: List[Dict]) -> float:
        """Calculate average trade P&L."""
        if not trades:
            return 0.0
        valid_pnls = [float(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) is not None]
        return float(np.mean(valid_pnls)) if valid_pnls else 0.0
    
    @staticmethod
    def calculate_largest_win(trades: List[Dict]) -> float:
        """Calculate largest winning trade."""
        if not trades:
            return 0.0
        valid_pnls = [float(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) is not None]
        return float(max(valid_pnls)) if valid_pnls else 0.0
    
    @staticmethod
    def calculate_largest_loss(trades: List[Dict]) -> float:
        """Calculate largest losing trade."""
        if not trades:
            return 0.0
        valid_pnls = [float(t.get('pnl', 0)) for t in trades if t.get('pnl', 0) is not None]
        return float(min(valid_pnls)) if valid_pnls else 0.0
    
    @staticmethod
    def calculate_consecutive_wins(trades: List[Dict]) -> float:
        """Calculate maximum consecutive wins."""
        if not trades:
            return 0.0
        max_consecutive = 0
        current_consecutive = 0
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl is not None and float(pnl) > 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return float(max_consecutive)
    
    @staticmethod
    def calculate_consecutive_losses(trades: List[Dict]) -> float:
        """Calculate maximum consecutive losses."""
        if not trades:
            return 0.0
        max_consecutive = 0
        current_consecutive = 0
        for trade in trades:
            pnl = trade.get('pnl', 0)
            if pnl is not None and float(pnl) < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        return float(max_consecutive)


class ModelMetrics:
    """Machine learning model evaluation metrics."""
    
    @staticmethod
    def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                       y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Calculate classification metrics."""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division="warn")),
            'recall': float(recall_score(y_true, y_pred, zero_division="warn")),
            'f1_score': float(f1_score(y_true, y_pred, zero_division="warn"))
        }
        
        if y_prob is not None:
            metrics['auc_roc'] = float(roc_auc_score(y_true, y_prob))
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Confusion Matrix"):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'])
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                      title: str = "ROC Curve"):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.7, label='Random')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def calculate_feature_importance(model, feature_names: List[str]) -> Dict[str, float]:
        """Calculate feature importance for tree-based models."""
        if hasattr(model, 'feature_importances_'):
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        else:
            return {}





class PerformanceReport:
    """Generate performance reports."""
    
    def __init__(self, strategy_returns: pd.Series, market_returns: pd.Series, 
                 trades: Optional[List[Dict]] = None):
        """
        Initialize performance report.
        
        Args:
            strategy_returns: Strategy returns
            market_returns: Market returns
            trades: List of trades (optional)
        """
        # Ensure all returns are numeric
        self.strategy_returns = pd.Series(pd.to_numeric(strategy_returns, errors='coerce')).dropna()
        self.market_returns = pd.Series(pd.to_numeric(market_returns, errors='coerce')).dropna()
        # Ensure all trade pnls are float
        if trades is not None:
            for t in trades:
                if 'pnl' in t:
                    try:
                        t['pnl'] = float(t['pnl'])
                    except Exception:
                        t['pnl'] = 0.0
        self.trades = trades or []
        
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report."""
        report = {
            'strategy_metrics': self._calculate_strategy_metrics(),
            'market_metrics': self._calculate_market_metrics(),
            'comparison_metrics': self._calculate_comparison_metrics(),
            'risk_metrics': self._calculate_risk_metrics(),
            'trade_metrics': self._calculate_trade_metrics() if self.trades else {}
        }
        
        return report
    
    def _calculate_strategy_metrics(self) -> Dict[str, float]:
        """Calculate strategy performance metrics."""
        # Ensure returns are numeric
        strategy_returns_numeric = pd.Series(pd.to_numeric(self.strategy_returns, errors='coerce')).dropna()
        
        if len(strategy_returns_numeric) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'calmar_ratio': 0.0,
                'max_drawdown': 0.0,
                'var_95': 0.0,
                'cvar_95': 0.0
            }
        
        return {
            'total_return': (1 + strategy_returns_numeric).prod() - 1,
            'annualized_return': TradingMetrics.calculate_annualized_return(strategy_returns_numeric),
            'volatility': TradingMetrics.calculate_volatility(strategy_returns_numeric),
            'sharpe_ratio': TradingMetrics.calculate_sharpe_ratio(strategy_returns_numeric),
            'sortino_ratio': TradingMetrics.calculate_sortino_ratio(strategy_returns_numeric),
            'calmar_ratio': TradingMetrics.calculate_calmar_ratio(strategy_returns_numeric),
            'max_drawdown': TradingMetrics.calculate_max_drawdown(strategy_returns_numeric)[0],
            'var_95': TradingMetrics.calculate_var(strategy_returns_numeric, 0.05),
            'cvar_95': TradingMetrics.calculate_cvar(strategy_returns_numeric, 0.05)
        }
    
    def _calculate_market_metrics(self) -> Dict[str, float]:
        """Calculate market performance metrics."""
        # Ensure returns are numeric
        market_returns_numeric = pd.to_numeric(self.market_returns, errors='coerce').dropna()
        
        if len(market_returns_numeric) == 0:
            return {
                'total_return': 0.0,
                'annualized_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
        return {
            'total_return': (1 + market_returns_numeric).prod() - 1,  # type: ignore
            'annualized_return': TradingMetrics.calculate_annualized_return(market_returns_numeric),  # type: ignore
            'volatility': TradingMetrics.calculate_volatility(market_returns_numeric),  # type: ignore
            'sharpe_ratio': TradingMetrics.calculate_sharpe_ratio(market_returns_numeric),  # type: ignore
            'max_drawdown': TradingMetrics.calculate_max_drawdown(market_returns_numeric)[0]  # type: ignore
        }
    
    def _calculate_comparison_metrics(self) -> Dict[str, float]:
        """Calculate comparison metrics."""
        strategy_metrics = self._calculate_strategy_metrics()
        market_metrics = self._calculate_market_metrics()
        
        return {
            'excess_return': strategy_metrics['annualized_return'] - market_metrics['annualized_return'],
            'information_ratio': (strategy_metrics['annualized_return'] - market_metrics['annualized_return']) / 
                                strategy_metrics['volatility'],
            'beta': self._calculate_beta(),
            'alpha': self._calculate_alpha()
        }
    
    def _calculate_risk_metrics(self) -> Dict[str, float]:
        """Calculate risk metrics."""
        # Ensure returns are numeric
        strategy_returns_numeric = pd.to_numeric(self.strategy_returns, errors='coerce').dropna()
        market_returns_numeric = pd.to_numeric(self.market_returns, errors='coerce').dropna()
        
        if len(strategy_returns_numeric) == 0:
            return {
                'downside_deviation': 0.0,
                'upside_capture': 0.0,
                'downside_capture': 0.0,
                'skewness': 0.0,
                'kurtosis': 0.0
            }
        
        return {
            'downside_deviation': strategy_returns_numeric[strategy_returns_numeric < 0].std() * np.sqrt(252) if len(strategy_returns_numeric[strategy_returns_numeric < 0]) > 0 else 0.0,  # type: ignore
            'upside_capture': self._calculate_upside_capture(),
            'downside_capture': self._calculate_downside_capture(),
            'skewness': float(strategy_returns_numeric.skew()),  # type: ignore
            'kurtosis': float(strategy_returns_numeric.kurtosis())  # type: ignore
        }
    
    def _calculate_trade_metrics(self) -> Dict[str, float]:
        """Calculate trade-based metrics."""
        return {
            'win_rate': TradingMetrics.calculate_win_rate(self.trades),
            'profit_factor': TradingMetrics.calculate_profit_factor(self.trades),
            'average_trade': TradingMetrics.calculate_average_trade(self.trades),
            'largest_win': TradingMetrics.calculate_largest_win(self.trades),
            'largest_loss': TradingMetrics.calculate_largest_loss(self.trades),
            'consecutive_wins': TradingMetrics.calculate_consecutive_wins(self.trades),
            'consecutive_losses': TradingMetrics.calculate_consecutive_losses(self.trades),
            'total_trades': len(self.trades)
        }
    

    
    def _calculate_beta(self) -> float:
        """Calculate beta relative to market."""
        covariance = np.cov(self.strategy_returns, self.market_returns)[0, 1]
        market_variance = np.var(self.market_returns)
        return covariance / market_variance if market_variance != 0 else 0
    
    def _calculate_alpha(self) -> float:
        """Calculate alpha (excess return adjusted for beta)."""
        beta = self._calculate_beta()
        strategy_return = TradingMetrics.calculate_annualized_return(self.strategy_returns)
        market_return = TradingMetrics.calculate_annualized_return(self.market_returns)
        risk_free_rate = 0.02
        
        return strategy_return - (risk_free_rate + beta * (market_return - risk_free_rate))
    
    def _calculate_upside_capture(self) -> float:
        """Calculate upside capture ratio."""
        # Ensure returns are numeric
        strategy_returns_numeric = pd.to_numeric(self.strategy_returns, errors='coerce').dropna()
        market_returns_numeric = pd.to_numeric(self.market_returns, errors='coerce').dropna()
        
        # Align the series
        aligned_data = pd.concat([pd.Series(strategy_returns_numeric), pd.Series(market_returns_numeric)], axis=1).dropna()  # type: ignore
        if len(aligned_data) == 0:
            return 0.0
            
        strategy_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        market_positive = market_aligned[market_aligned > 0]
        strategy_positive = strategy_aligned[market_aligned > 0]
        
        if len(market_positive) == 0:
            return 0.0
        
        return strategy_positive.mean() / market_positive.mean() if market_positive.mean() != 0 else 0
    
    def _calculate_downside_capture(self) -> float:
        """Calculate downside capture ratio."""
        # Ensure returns are numeric
        strategy_returns_numeric = pd.to_numeric(self.strategy_returns, errors='coerce').dropna()
        market_returns_numeric = pd.to_numeric(self.market_returns, errors='coerce').dropna()
        
        # Align the series
        aligned_data = pd.concat([pd.Series(strategy_returns_numeric), pd.Series(market_returns_numeric)], axis=1).dropna()  # type: ignore
        if len(aligned_data) == 0:
            return 0.0
            
        strategy_aligned = aligned_data.iloc[:, 0]
        market_aligned = aligned_data.iloc[:, 1]
        
        market_negative = market_aligned[market_aligned < 0]
        strategy_negative = strategy_aligned[market_aligned < 0]
        
        if len(market_negative) == 0:
            return 0.0
        
        return strategy_negative.mean() / market_negative.mean() if market_negative.mean() != 0 else 0
    
    def print_report(self):
        """Print formatted performance report."""
        report = self.generate_report()
        
        print("\nSTRATEGY METRICS:")
        print("-" * 30)
        for metric, value in report['strategy_metrics'].items():
            if 'return' in metric or 'drawdown' in metric or 'var' in metric or 'cvar' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        print("\nMARKET METRICS:")
        print("-" * 30)
        for metric, value in report['market_metrics'].items():
            if 'return' in metric or 'drawdown' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        print("\nCOMPARISON METRICS:")
        print("-" * 30)
        for metric, value in report['comparison_metrics'].items():
            if 'return' in metric or 'alpha' in metric:
                print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
            else:
                print(f"{metric.replace('_', ' ').title()}: {value:.3f}")
        
        if self.trades:
            print("\nTRADE METRICS:")
            print("-" * 30)
            for metric, value in report['trade_metrics'].items():
                if 'rate' in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2%}")
                elif 'trades' in metric:
                    print(f"{metric.replace('_', ' ').title()}: {value:.0f}")
                else:
                    print(f"{metric.replace('_', ' ').title()}: {value:.2f}")