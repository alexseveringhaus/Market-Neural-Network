"""
Neural Network Models Module

This module contains various neural network architectures for stock market prediction,
including LSTM, GRU, and hybrid models.
"""

import numpy as np
from tensorflow.keras import layers, optimizers, regularizers, callbacks  # type: ignore
from tensorflow.keras.models import Sequential, Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Dense, LSTM, Dropout, BatchNormalization, 
    Input, MultiHeadAttention,
    LayerNormalization, GlobalAveragePooling1D
)
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List


class BaseModel:
    """Base class for all neural network models."""
    
    def __init__(self, input_shape: Tuple[int, ...], **kwargs):
        """
        Initialize the base model.
        
        Args:
            input_shape: Shape of input features
            **kwargs: Additional arguments
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.is_fitted = False
        
    def build_model(self) -> Model:
        """Build the neural network model. To be implemented by subclasses."""
        raise NotImplementedError
        
    def compile_model(self, learning_rate: float = 0.001, **kwargs):
        """Compile the model with optimizer and loss function."""
        if self.model is None:
            self.build_model()
            
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(  # type: ignore
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'auc']
        )
        
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
            **kwargs: Additional arguments
            
        Returns:
            Training history
        """
        if self.model is None:
            self.compile_model()
            
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        # Training
        self.history = self.model.fit(  # type: ignore
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose,
            **kwargs
        )
        
        self.is_fitted = True
        return self.history.history
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)  # type: ignore
        
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)  # type: ignore
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
            
        y_pred = self.predict(X)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred_classes),
            'precision': precision_score(y, y_pred_classes, zero_division="warn"),
            'recall': recall_score(y, y_pred_classes, zero_division="warn"),
            'f1_score': f1_score(y, y_pred_classes, zero_division="warn")
        }
        
        return metrics
        
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 5)):
        """Plot training history."""
        if self.history is None:
            raise ValueError("No training history available")
            
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Loss
        axes[0].plot(self.history.history['loss'], label='Training Loss')
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        # Accuracy
        axes[1].plot(self.history.history['accuracy'], label='Training Accuracy')
        if 'val_accuracy' in self.history.history:
            axes[1].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[1].set_title('Model Accuracy')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()


class DenseNeuralNetwork(BaseModel):
    """Dense neural network for stock prediction."""
    
    def __init__(
        self, 
        input_shape: Tuple[int, ...],
        hidden_layers: List[int] = [128, 64, 32],
        dropout_rate: float = 0.3,
        l2_reg: float = 0.001,
        **kwargs
    ):
        """
        Initialize dense neural network.
        
        Args:
            input_shape: Shape of input features
            hidden_layers: List of hidden layer sizes
            dropout_rate: Dropout rate
            l2_reg: L2 regularization strength
        """
        super().__init__(input_shape, **kwargs)
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        
    def build_model(self) -> Model:
        """Build the dense neural network."""
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.hidden_layers[0],
            input_shape=self.input_shape,
            activation='relu',
            kernel_regularizer=regularizers.l2(self.l2_reg)
        ))
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Hidden layers
        for units in self.hidden_layers[1:]:
            model.add(Dense(
                units,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg)
            ))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        self.model = model
        return model


class LSTMNeuralNetwork(BaseModel):
    """LSTM neural network for time series prediction."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        lstm_units: List[int] = [64, 32],
        dense_units: List[int] = [32],
        dropout_rate: float = 0.3,
        recurrent_dropout: float = 0.2,
        **kwargs
    ):
        """
        Initialize LSTM neural network.
        
        Args:
            input_shape: Shape of input features (timesteps, features)
            lstm_units: List of LSTM layer sizes
            dense_units: List of dense layer sizes
            dropout_rate: Dropout rate
            recurrent_dropout: Recurrent dropout rate
        """
        super().__init__(input_shape, **kwargs)
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        
    def build_model(self) -> Model:
        """Build the LSTM neural network."""
        model = Sequential()
        
        # LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                kernel_regularizer=regularizers.l2(0.001)
            ))
            model.add(BatchNormalization())
        
        # Dense layers
        for units in self.dense_units:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        self.model = model
        return model


class AttentionNeuralNetwork(BaseModel):
    """Neural network with attention mechanism."""
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        attention_heads: int = 8,
        transformer_blocks: int = 2,
        dense_units: List[int] = [64, 32],
        dropout_rate: float = 0.3,
        **kwargs
    ):
        """
        Initialize attention neural network.
        
        Args:
            input_shape: Shape of input features
            attention_heads: Number of attention heads
            transformer_blocks: Number of transformer blocks
            dense_units: List of dense layer sizes
            dropout_rate: Dropout rate
        """
        super().__init__(input_shape, **kwargs)
        self.attention_heads = attention_heads
        self.transformer_blocks = transformer_blocks
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        
    def build_model(self) -> Model:
        """Build the attention neural network."""
        inputs = Input(shape=self.input_shape)
        
        # Reshape for attention (add sequence dimension if needed)
        if len(self.input_shape) == 1:
            x = layers.Reshape((1, self.input_shape[0]))(inputs)
        else:
            x = inputs
        
        # Transformer blocks
        for _ in range(self.transformer_blocks):
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=self.attention_heads,
                key_dim=32
            )(x, x)
            x = LayerNormalization(epsilon=1e-6)(x + attention_output)
            
            # Feed-forward network
            ffn = Sequential([
                Dense(128, activation='relu'),
                Dense(x.shape[-1])
            ])
            ffn_output = ffn(x)
            x = LayerNormalization(epsilon=1e-6)(x + ffn_output)
        
        # Global average pooling
        x = GlobalAveragePooling1D()(x)
        
        # Dense layers
        for units in self.dense_units:
            x = Dense(units, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs)
        return self.model


class EnsembleNeuralNetwork:
    """Ensemble of multiple neural network models."""
    
    def __init__(self, models: List[BaseModel], weights: Optional[List[float]] = None):
        """
        Initialize ensemble model.
        
        Args:
            models: List of neural network models
            weights: Weights for each model (optional)
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0] * len(models)
        self.is_fitted = False
        
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> List[Dict[str, List[float]]]:
        """Train all models in the ensemble."""
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}")
            
            # Handle different input shapes for different model types
            if isinstance(model, LSTMNeuralNetwork):
                # Reshape data for LSTM (samples, timesteps, features)
                X_train_reshaped = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
                X_val_reshaped = X_val.reshape(X_val.shape[0], 1, X_val.shape[1]) if X_val is not None else None
                history = model.fit(X_train_reshaped, y_train, X_val_reshaped, y_val, **kwargs)
            else:
                # Use original shape for Dense and Attention models
                history = model.fit(X_train, y_train, X_val, y_val, **kwargs)
            
            histories.append(history)
            
        self.is_fitted = True
        return histories
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
            
        predictions = []
        for model in self.models:
            # Handle different input shapes for different model types
            if isinstance(model, LSTMNeuralNetwork):
                # Reshape data for LSTM (samples, timesteps, features)
                X_reshaped = X.reshape(X.shape[0], 1, X.shape[1])
                pred = model.predict_proba(X_reshaped)
            else:
                # Use original shape for Dense and Attention models
                pred = model.predict_proba(X)
            
            predictions.append(pred)
            
        # Weighted average
        weighted_pred = np.zeros_like(predictions[0])
        for pred, weight in zip(predictions, self.weights):
            weighted_pred += weight * pred
            
        return weighted_pred / sum(self.weights)
        
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the ensemble model."""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before evaluation")
            
        y_pred = self.predict(X)
        y_pred_classes = (y_pred > 0.5).astype(int)
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred_classes),
            'precision': precision_score(y, y_pred_classes, zero_division="warn"),
            'recall': recall_score(y, y_pred_classes, zero_division="warn"),
            'f1_score': f1_score(y, y_pred_classes, zero_division="warn")
        }
        
        return metrics 