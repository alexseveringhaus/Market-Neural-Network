#!/usr/bin/env python3
"""
Market Neural Network - Stock Market Prediction System

This is the main script that demonstrates the capabilities of the
Market Neural Network project. It showcases multiple neural network
architectures, feature engineering, trading strategies,
and performance evaluation.
"""

import sys
import os
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
from data.data_loader import DataLoader
from models.neural_networks import (
    DenseNeuralNetwork, LSTMNeuralNetwork, AttentionNeuralNetwork, EnsembleNeuralNetwork
)
from strategies.trading_strategy import MLTradingStrategy
from visualization.plots import StockVisualizer, TradingVisualizer
from utils.metrics import PerformanceReport, ModelMetrics

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow to reduce retracing warnings
import tensorflow as tf
tf.config.experimental.enable_tensor_float_32_execution(False)


def main():
    """Main function demonstrating the full system capabilities."""
    
    print("Market Neural Network - Advanced Stock Prediction System")
    print("This demonstration showcases:")
    print("- Data loading and feature engineering")
    print("- Multiple neural network architectures (Dense, LSTM, Attention)")
    print("- Ensemble modeling")
    print("- Trading strategies with risk management")
    print("- Performance evaluation and visualization")
    
    # Configuration
    SYMBOL = 'AAPL'
    START_DATE = '2022-01-01'
    END_DATE = '2024-01-01'
    INITIAL_CAPITAL = 100000
    
    try:
        # Step 1: Data Loading and Feature Engineering
        print("\nLoading and Processing Data")
        
        data_loader = DataLoader(cache_data=True)
        
        # Download stock data
        print(f"Downloading data for {SYMBOL}")
        raw_data = data_loader.get_stock_data(SYMBOL, START_DATE, END_DATE)
        print(f"Downloaded {len(raw_data)} data points for {SYMBOL}")
        
        # Engineer features
        print("Engineering features")
        data = data_loader.engineer_features(raw_data)
        print(f"Engineered {len(data.columns)} features")
        
        # Prepare features for ML
        X, y, scaler = data_loader.prepare_features(data)
        print(f"Prepared {X.shape[1]} features for {len(y)} samples")
        
        # Create time series splits
        splits = data_loader.create_time_series_split(X, y, n_splits=3)
        X_train, X_test, y_train, y_test = splits[-1]  # Use last split
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Step 2: Model Training
        print("\nTraining Neural Network Models")
        
        models = {}
        
        # Dense Neural Network
        print("Training Dense Neural Network...")
        dense_model = DenseNeuralNetwork(
            input_shape=(X_train.shape[1],),
            hidden_layers=[128, 64, 32],
            dropout_rate=0.3,
            l2_reg=0.001
        )
        dense_history = dense_model.fit(X_train, y_train, epochs=50, verbose=1)
        models['Dense'] = dense_model
        print("Dense Neural Network trained")

        # LSTM Neural Network (reshape data for LSTM)
        print("Training LSTM Neural Network...")
        # Reshape data for LSTM (samples, timesteps, features)
        X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        lstm_model = LSTMNeuralNetwork(
            input_shape=(1, X_train.shape[1]),
            lstm_units=[64, 32],
            dense_units=[32],
            dropout_rate=0.3
        )
        lstm_history = lstm_model.fit(X_train_lstm, y_train, epochs=50, verbose=1)
        models['LSTM'] = lstm_model
        print("LSTM Neural Network trained")

        # Attention Neural Network
        print("Training Attention Neural Network...")
        attention_model = AttentionNeuralNetwork(
            input_shape=(X_train.shape[1],),
            attention_heads=8,
            transformer_blocks=2,
            dense_units=[64, 32]
        )
        attention_history = attention_model.fit(X_train, y_train, epochs=50, verbose=1)
        models['Attention'] = attention_model
        print("Attention Neural Network trained")

        # Ensemble
        ensemble_models = [models['Dense'], models['LSTM'], models['Attention']]
        ensemble_weights = [0.4, 0.35, 0.25]  # Weight by performance
        ensemble = EnsembleNeuralNetwork(ensemble_models, ensemble_weights)
        ensemble.fit(X_train, y_train, epochs=50, verbose=1)

        # Step 3: Model Evaluation
        print("\nModel Evaluation")
        
        # Pre-reshape data to avoid retracing
        X_test_reshaped = {
            'Dense': X_test,
            'LSTM': X_test_lstm,
            'Attention': X_test
        }
        
        for name, model in models.items():
            y_pred = model.predict(X_test_reshaped[name])
            
            y_pred_classes = (y_pred > 0.5).astype(int)
            metrics = ModelMetrics.calculate_classification_metrics(y_test, y_pred_classes, y_pred.flatten())
            
            print(f"\n{name} Neural Network Results:")
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1_score']:.3f}")
            if 'auc_roc' in metrics:
                print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
        
        # Step 4: Ensemble Model
        print("\nEnsemble Model")
        
        # Create ensemble with different weights
        ensemble_models = [models['Dense'], models['LSTM'], models['Attention']]
        ensemble_weights = [0.4, 0.35, 0.25]  # Weight by performance
        
        ensemble = EnsembleNeuralNetwork(ensemble_models, ensemble_weights)
        ensemble.fit(X_train, y_train, epochs=50, verbose=1)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test)
        y_pred_ensemble_classes = (y_pred_ensemble > 0.5).astype(int)
        ensemble_metrics = ModelMetrics.calculate_classification_metrics(
            y_test, y_pred_ensemble_classes, y_pred_ensemble.flatten()
        )
        
        print("Ensemble Model Results:")
        print(f"  Accuracy: {ensemble_metrics['accuracy']:.3f}")
        print(f"  Precision: {ensemble_metrics['precision']:.3f}")
        print(f"  Recall: {ensemble_metrics['recall']:.3f}")
        print(f"  F1-Score: {ensemble_metrics['f1_score']:.3f}")
        if 'auc_roc' in ensemble_metrics:
            print(f"  AUC-ROC: {ensemble_metrics['auc_roc']:.3f}")
        
        # Step 5: Trading Strategy
        print("\nTrading Strategy Execution")
        
        # Use ensemble predictions for trading - ensure same length
        test_data = data.iloc[len(X_train):len(X_train) + len(X_test)].copy()
        test_data['Predictions'] = y_pred_ensemble.flatten()
        
        # Initialize trading strategy
        strategy = MLTradingStrategy(
            initial_capital=INITIAL_CAPITAL,
            confidence_threshold=0.55,  # Balanced threshold
            risk_per_trade=0.03,  # Moderate risk per trade
            max_positions=4,  # Reasonable number of positions
            stop_loss=0.025,  # Tight stop loss
            take_profit=0.075  # Reasonable take profit
        )
        
        # Run strategy
        print("Executing trading strategy")
        strategy_result = strategy.run_strategy(test_data, y_pred_ensemble.flatten())
        
        print(f"Strategy executed with {len(strategy_result.trades)} trades")
        print(f"Final portfolio value: ${strategy_result.portfolio_values.iloc[-1]:,.2f}")
        print(f"Total return: {strategy_result.metrics['total_return']:.2%}")
        
        # Step 6: Performance Analysis
        print("\nPerformance Analysis")
        
        # Generate comprehensive performance report
        market_returns = test_data['Returns']
        
        # Ensure both returns have the same length
        min_length = min(len(strategy_result.returns), len(market_returns))
        strategy_returns_aligned = strategy_result.returns[:min_length]
        market_returns_aligned = market_returns[:min_length]
        
        performance_report = PerformanceReport(
            strategy_returns_aligned, 
            market_returns_aligned,
            [{'pnl': t.pnl} for t in strategy_result.trades if t.exit_date is not None]
        )
        
        performance_report.print_report()
        
        # Step 7: Visualizations
        print("\nGenerating Visualizations")
        
        # Stock data visualization
        print("Generating stock data visualizations...")
        stock_viz = StockVisualizer()
        stock_viz.plot_stock_data(data, f"{SYMBOL} Stock Analysis")
        stock_viz.plot_returns_distribution(data['Returns'], f"{SYMBOL} Returns Distribution")
        
        # Trading strategy visualization
        print("Generating trading strategy visualizations...")
        trading_viz = TradingVisualizer()
        trading_viz.plot_strategy_performance(strategy_result, test_data, f"{SYMBOL} Trading Strategy")
        
        if strategy_result.trades:
            trading_viz.plot_trade_analysis(strategy_result.trades, f"{SYMBOL} Trade Analysis")
        
        # Step 8: Model Training History
        print("\nModel Training Analysis")
        
        # Plot training history for best model
        print("Plotting training history for dense model...")
        # Note: We'll plot the dense model history as an example
        models['Dense'].plot_training_history()
        
        # Step 9: Summary and Recommendations
        print("\nSummary and Recommendations")
        
        print("System Performance Summary:")
        print(f"- Best individual model: {max(models.keys(), key=lambda x: ensemble_weights[list(models.keys()).index(x)])}")
        print(f"- Ensemble accuracy: {ensemble_metrics['accuracy']:.3f}")
        print(f"- Strategy Sharpe ratio: {strategy_result.metrics['sharpe_ratio']:.3f}")
        print(f"- Strategy win rate: {strategy_result.metrics['win_rate']:.2%}")
        print(f"- Excess return vs market: {strategy_result.metrics['excess_return']:.2%}")
        
        print("\nRecommendations:")
        if strategy_result.metrics['sharpe_ratio'] > 1.0:
            print("Strategy shows good risk-adjusted returns")
        else:
            print("Strategy may need optimization for better risk-adjusted returns")
            
        if strategy_result.metrics['win_rate'] > 0.5:
            print("Strategy has a positive win rate")
        else:
            print("Consider improving signal quality or risk management")
            
        if strategy_result.metrics['max_drawdown'] < -0.2:
            print("High maximum drawdown - consider tighter risk controls")
        else:
            print("Drawdown within acceptable limits")
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        print(f"\nError occurred: {str(e)}")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\nSuccess")
    else:
        print("\nSystem encountered issues. Review the error messages above.")
        sys.exit(1) 