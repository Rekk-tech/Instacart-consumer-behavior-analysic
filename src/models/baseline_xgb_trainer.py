"""
XGBoost Baseline Model Trainer
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
import joblib
from datetime import datetime

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from ..utils.config import ModelConfig
from ..utils.metrics import RecommendationMetrics
from ..utils.io import save_json, save_model

logger = logging.getLogger(__name__)

class XGBTrainer:
    """XGBoost model trainer for next purchase prediction"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = None
        self.feature_importance = None
        self.training_metrics = {}
        
    def train(self,
             features: pd.DataFrame,
             target: pd.Series,
             feature_cols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Train XGBoost model
        
        Args:
            features: Feature dataframe
            target: Target series (reorder probability)
            feature_cols: List of feature columns to use
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting XGBoost training...")
        
        # Prepare features
        if feature_cols is None:
            # Exclude ID columns and target
            exclude_cols = ['user_id', 'product_id', 'order_id']
            feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols]
        y = target
        
        logger.info(f"Training with {len(feature_cols)} features on {len(X)} samples")
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Training parameters
        params = self.config.xgb_params.copy()
        
        # Train model with early stopping
        eval_list = [(dtrain, 'train'), (dval, 'eval')]
        
        self.model = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=params.get('n_estimators', 100),
            evals=eval_list,
            early_stopping_rounds=20,
            verbose_eval=10
        )
        
        # Get predictions
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        # Calculate metrics
        train_auc = roc_auc_score(y_train, train_pred)
        val_auc = roc_auc_score(y_val, val_pred)
        
        # PR AUC
        train_precision, train_recall, _ = precision_recall_curve(y_train, train_pred)
        val_precision, val_recall, _ = precision_recall_curve(y_val, val_pred)
        
        train_pr_auc = auc(train_recall, train_precision)
        val_pr_auc = auc(val_recall, val_precision)
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.model.get_score(importance_type='gain').values()
        }).sort_values('importance', ascending=False)
        
        # Store training metrics
        self.training_metrics = {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'train_pr_auc': train_pr_auc,
            'val_pr_auc': val_pr_auc,
            'best_iteration': self.model.best_iteration,
            'num_features': len(feature_cols),
            'training_samples': len(X_train),
            'validation_samples': len(X_val)
        }
        
        logger.info(f"Training completed - Val AUC: {val_auc:.4f}, Val PR-AUC: {val_pr_auc:.4f}")
        
        return self.training_metrics
    
    def predict(self, features: pd.DataFrame, feature_cols: Optional[List[str]] = None) -> np.ndarray:
        """Make predictions"""
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if feature_cols is None:
            # Use same features as training
            exclude_cols = ['user_id', 'product_id', 'order_id']
            feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        X = features[feature_cols]
        dtest = xgb.DMatrix(X)
        
        predictions = self.model.predict(dtest)
        return predictions
    
    def evaluate_recommendations(self,
                                features: pd.DataFrame,
                                target: pd.Series,
                                user_col: str = 'user_id',
                                item_col: str = 'product_id',
                                k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
        """
        Evaluate model as recommendation system
        
        Args:
            features: Feature dataframe
            target: Target series
            user_col: User column name
            item_col: Item column name
            k_values: Values of k for top-k evaluation
            
        Returns:
            Recommendation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        logger.info("Evaluating recommendation performance...")
        
        # Get predictions
        predictions = self.predict(features)
        
        # Create evaluation dataframe
        eval_df = features[[user_col, item_col]].copy()
        eval_df['prediction'] = predictions
        eval_df['actual'] = target.values
        
        # Calculate recommendation metrics
        metrics_calculator = RecommendationMetrics()
        
        results = {}
        for k in k_values:
            # Get top-k recommendations per user
            top_k = eval_df.groupby(user_col).apply(
                lambda x: x.nlargest(k, 'prediction')
            ).reset_index(drop=True)
            
            # Calculate metrics
            precision_k = metrics_calculator.precision_at_k(
                top_k['actual'].values, k
            )
            recall_k = metrics_calculator.recall_at_k(
                top_k['actual'].values, 
                eval_df.groupby(user_col)['actual'].sum().values, 
                k
            )
            
            results[f'precision@{k}'] = precision_k
            results[f'recall@{k}'] = recall_k
            results[f'f1@{k}'] = 2 * precision_k * recall_k / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0
        
        logger.info(f"Recommendation evaluation completed")
        
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
        model_file = model_path / f"xgboost_baseline_{timestamp}.joblib"
        joblib.dump(self.model, model_file)
        
        # Save feature importance
        importance_file = model_path / f"xgb_feature_importance_{timestamp}.csv"
        self.feature_importance.to_csv(importance_file, index=False)
        
        # Save training results
        results_file = model_path / f"xgb_training_results_{timestamp}.json"
        save_json(self.training_metrics, results_file)
        
        logger.info(f"Model saved to {model_file}")
        
        return model_file
    
    def load_model(self, model_path: Path):
        """Load trained model"""
        
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top feature importance"""
        
        if self.feature_importance is None:
            raise ValueError("No feature importance available")
        
        return self.feature_importance.head(top_n)