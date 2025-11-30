"""
LSTM Model Trainer for Sequential Recommendation
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from ..utils.config import ModelConfig
from ..utils.metrics import RecommendationMetrics
from ..utils.io import save_json

logger = logging.getLogger(__name__)

class LSTMTrainer:
    """LSTM model trainer for sequential recommendation"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.history = None
        self.training_metrics = {}
        
        # Enable mixed precision for better performance
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
    def build_model(self, vocab_size: int, max_seq_length: int) -> keras.Model:
        """Build LSTM model architecture"""
        
        # Input layer
        inputs = layers.Input(shape=(max_seq_length,), name='sequence_input')
        
        # Embedding layer
        embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=self.config.embedding_dim,
            input_length=max_seq_length,
            mask_zero=True,  # Mask padding tokens
            name='embedding'
        )(inputs)
        
        # LSTM layers
        x = embedding
        for i, units in enumerate(self.config.lstm_units):
            return_sequences = i < len(self.config.lstm_units) - 1
            x = layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=self.config.dropout,
                recurrent_dropout=self.config.dropout,
                name=f'lstm_{i+1}'
            )(x)
        
        # Dense layers
        x = layers.Dense(
            self.config.dense_units,
            activation='relu',
            name='dense_hidden'
        )(x)
        x = layers.Dropout(self.config.dropout)(x)
        
        # Output layer
        outputs = layers.Dense(
            vocab_size,
            activation='softmax',
            dtype='float32',  # Use float32 for output layer in mixed precision
            name='predictions'
        )(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs, name='lstm_recommender')
        
        return model
    
    def compile_model(self, model: keras.Model):
        """Compile the model"""
        
        optimizer = keras.optimizers.Adam(learning_rate=self.config.learning_rate)
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
        )
        
        return model
    
    def train(self,
             sequences: np.ndarray,
             targets: np.ndarray,
             vocab_size: int,
             validation_split: Optional[float] = None) -> Dict[str, Any]:
        """
        Train LSTM model
        
        Args:
            sequences: Input sequences (batch_size, max_seq_length)
            targets: Target product IDs (batch_size,)
            vocab_size: Size of product vocabulary
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting LSTM training...")
        
        if validation_split is None:
            validation_split = self.config.validation_split
        
        # Build and compile model
        max_seq_length = sequences.shape[1]
        self.model = self.build_model(vocab_size, max_seq_length)
        self.model = self.compile_model(self.model)
        
        logger.info(f"Model built with vocab_size={vocab_size}, seq_length={max_seq_length}")
        logger.info(f"Total parameters: {self.model.count_params():,}")
        
        # Prepare callbacks
        callbacks = self._prepare_callbacks()
        
        # Train model
        self.history = self.model.fit(
            sequences,
            targets,
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate final metrics
        self.training_metrics = self._calculate_training_metrics()
        
        logger.info(f"Training completed - Val Loss: {self.training_metrics['val_loss']:.4f}")
        
        return self.training_metrics
    
    def _prepare_callbacks(self) -> List[keras.callbacks.Callback]:
        """Prepare training callbacks"""
        
        callbacks = []
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.config.reduce_lr_factor,
            patience=self.config.reduce_lr_patience,
            min_lr=self.config.reduce_lr_min_lr,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        return callbacks
    
    def _calculate_training_metrics(self) -> Dict[str, Any]:
        """Calculate final training metrics"""
        
        if self.history is None:
            return {}
        
        history_dict = self.history.history
        
        # Get final epoch metrics
        final_epoch = len(history_dict['loss']) - 1
        
        metrics = {
            'train_loss': float(history_dict['loss'][final_epoch]),
            'val_loss': float(history_dict['val_loss'][final_epoch]),
            'train_accuracy': float(history_dict['accuracy'][final_epoch]),
            'val_accuracy': float(history_dict['val_accuracy'][final_epoch]),
            'final_epoch': final_epoch + 1,
            'total_epochs': len(history_dict['loss'])
        }
        
        # Add top-k accuracy if available
        if 'sparse_top_k_categorical_accuracy' in history_dict:
            metrics['train_top_k_acc'] = float(history_dict['sparse_top_k_categorical_accuracy'][final_epoch])
            metrics['val_top_k_acc'] = float(history_dict['val_sparse_top_k_categorical_accuracy'][final_epoch])
        
        return metrics
    
    def predict_top_k(self,
                     sequences: np.ndarray,
                     k: int = 10,
                     exclude_seen: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict top-k products for sequences
        
        Args:
            sequences: Input sequences
            k: Number of top products to return
            exclude_seen: Whether to exclude products seen in input sequence
            
        Returns:
            Tuple of (top_k_products, top_k_scores)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Get predictions
        predictions = self.model.predict(sequences, batch_size=self.config.batch_size)
        
        top_k_products = []
        top_k_scores = []
        
        for i, pred in enumerate(predictions):
            if exclude_seen:
                # Mask products already seen in sequence
                seen_products = set(sequences[i][sequences[i] > 0])  # Exclude padding (0)
                for product_id in seen_products:
                    pred[product_id] = 0
            
            # Get top-k predictions
            top_k_indices = np.argsort(pred)[-k:][::-1]
            top_k_probs = pred[top_k_indices]
            
            top_k_products.append(top_k_indices)
            top_k_scores.append(top_k_probs)
        
        return np.array(top_k_products), np.array(top_k_scores)
    
    def evaluate_on_sequences(self,
                            test_sequences: np.ndarray,
                            test_targets: np.ndarray,
                            k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """Evaluate model on test sequences"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating LSTM model on test sequences...")
        
        results = {}
        
        # Basic loss and accuracy
        test_loss, test_acc = self.model.evaluate(
            test_sequences, test_targets, 
            batch_size=self.config.batch_size,
            verbose=0
        )[:2]
        
        results['test_loss'] = test_loss
        results['test_accuracy'] = test_acc
        
        # Top-k recommendation metrics
        for k in k_values:
            top_k_products, top_k_scores = self.predict_top_k(test_sequences, k=k)
            
            # Calculate hit rate
            hits = 0
            for i, target in enumerate(test_targets):
                if target in top_k_products[i]:
                    hits += 1
            
            hit_rate = hits / len(test_targets)
            results[f'hit_rate@{k}'] = hit_rate
        
        logger.info(f"Evaluation completed - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        
        return results
    
    def save_model(self, model_path: Path, timestamp: Optional[str] = None) -> Path:
        """Save trained model and metadata"""
        
        if self.model is None:
            raise ValueError("No model to save")
        
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_path = Path(model_path)
        model_path.mkdir(exist_ok=True, parents=True)
        
        # Save model
        model_file = model_path / f"lstm_model_{timestamp}.h5"
        self.model.save(model_file)
        
        # Save training history
        if self.history is not None:
            history_file = model_path / f"lstm_training_history_{timestamp}.csv"
            history_df = pd.DataFrame(self.history.history)
            history_df.to_csv(history_file, index=False)
        
        # Save training results
        results_file = model_path / f"lstm_training_results_{timestamp}.json"
        save_json(self.training_metrics, results_file)
        
        logger.info(f"LSTM model saved to {model_file}")
        
        return model_file
    
    def load_model(self, model_path: Path):
        """Load trained model"""
        
        self.model = keras.models.load_model(model_path)
        logger.info(f"LSTM model loaded from {model_path}")