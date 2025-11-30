"""
Inference Engine for Product Recommendations
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import time
from datetime import datetime

from .model_loader import ModelLoader
from ..features import UserFeatureBuilder, ItemFeatureBuilder, UserItemFeatureBuilder, SequenceBuilder
from ..utils.config import ServingConfig
from ..utils.io import load_parquet

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Main inference engine for recommendations"""
    
    def __init__(self, config: ServingConfig, model_loader: ModelLoader):
        self.config = config
        self.model_loader = model_loader
        
        # Cache for frequently accessed data
        self.user_features_cache = {}
        self.item_features_cache = {}
        self.sequence_cache = {}
        
        # Load static data
        self.products_df = None
        self.popular_products = None
        
    def load_static_data(self, data_path: Path):
        """Load static product data for recommendations"""
        
        try:
            data_path = Path(data_path)
            
            # Load products
            products_file = data_path / "products.parquet"
            if products_file.exists():
                self.products_df = load_parquet(products_file)
                logger.info(f"Loaded {len(self.products_df)} products")
            
            # Load popular products for cold start
            features_path = data_path / "features"
            item_features_file = features_path / "item_features.parquet"
            if item_features_file.exists():
                item_features = load_parquet(item_features_file)
                self.popular_products = item_features.nlargest(
                    100, 'total_orders'
                )['product_id'].tolist()
                logger.info(f"Loaded {len(self.popular_products)} popular products")
            
        except Exception as e:
            logger.error(f"Failed to load static data: {str(e)}")
    
    def predict_xgb(self, 
                    user_id: int,
                    candidate_products: Optional[List[int]] = None,
                    top_k: int = 10) -> List[Dict[str, Any]]:
        """Generate recommendations using XGBoost model"""
        
        try:
            model = self.model_loader.get_model('xgb')
            
            # If no candidates provided, use popular products
            if candidate_products is None:
                candidate_products = self.popular_products[:100] if self.popular_products else []
            
            if not candidate_products:
                return []
            
            # Create user-item pairs for prediction
            user_item_pairs = pd.DataFrame({
                'user_id': [user_id] * len(candidate_products),
                'product_id': candidate_products
            })
            
            # Get features (this would need actual feature pipeline)
            # For now, return dummy predictions based on popularity
            predictions = np.random.random(len(candidate_products))
            
            # Sort by prediction score
            user_item_pairs['score'] = predictions
            top_recommendations = user_item_pairs.nlargest(top_k, 'score')
            
            # Format output
            recommendations = []
            for _, row in top_recommendations.iterrows():
                recommendations.append({
                    'product_id': int(row['product_id']),
                    'score': float(row['score']),
                    'model': 'xgboost'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"XGB prediction failed: {str(e)}")
            return self._fallback_recommendations(top_k)
    
    def predict_lstm(self,
                    user_id: int,
                    user_sequence: Optional[List[int]] = None,
                    top_k: int = 10) -> List[Dict[str, Any]]:
        """Generate recommendations using LSTM model"""
        
        try:
            model = self.model_loader.get_model('lstm')
            
            # Get or create user sequence
            if user_sequence is None:
                user_sequence = self._get_user_sequence(user_id)
            
            if not user_sequence:
                return self._fallback_recommendations(top_k)
            
            # Prepare sequence for prediction
            sequence_array = self._prepare_sequence(user_sequence)
            
            # Get predictions
            predictions = model.predict(sequence_array, verbose=0)[0]  # Get first (and only) prediction
            
            # Get top-k products (exclude padding token 0)
            top_k_indices = np.argsort(predictions)[-top_k-1:][::-1]  # Get one extra in case of padding
            top_k_indices = [idx for idx in top_k_indices if idx > 0][:top_k]  # Remove padding, keep top_k
            
            # Format output
            recommendations = []
            for idx in top_k_indices:
                recommendations.append({
                    'product_id': int(idx),  # Note: this would need proper decoding
                    'score': float(predictions[idx]),
                    'model': 'lstm'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"LSTM prediction failed: {str(e)}")
            return self._fallback_recommendations(top_k)
    
    def predict_tcn(self,
                   user_id: int,
                   user_sequence: Optional[List[int]] = None,
                   top_k: int = 10) -> List[Dict[str, Any]]:
        """Generate recommendations using TCN model"""
        
        try:
            model = self.model_loader.get_model('tcn')
            
            # Get or create user sequence
            if user_sequence is None:
                user_sequence = self._get_user_sequence(user_id)
            
            if not user_sequence:
                return self._fallback_recommendations(top_k)
            
            # Prepare sequence for prediction
            sequence_array = self._prepare_sequence(user_sequence)
            
            # Get predictions
            predictions = model.predict(sequence_array, verbose=0)[0]
            
            # Get top-k products (exclude padding token 0)
            top_k_indices = np.argsort(predictions)[-top_k-1:][::-1]
            top_k_indices = [idx for idx in top_k_indices if idx > 0][:top_k]
            
            # Format output
            recommendations = []
            for idx in top_k_indices:
                recommendations.append({
                    'product_id': int(idx),
                    'score': float(predictions[idx]),
                    'model': 'tcn'
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"TCN prediction failed: {str(e)}")
            return self._fallback_recommendations(top_k)
    
    def predict_ensemble(self,
                        user_id: int,
                        models: List[str] = ['xgb', 'lstm', 'tcn'],
                        weights: Optional[List[float]] = None,
                        top_k: int = 10) -> List[Dict[str, Any]]:
        """Generate ensemble recommendations"""
        
        try:
            if weights is None:
                weights = [1.0] * len(models)
            
            all_predictions = {}
            
            # Get predictions from each model
            for model_name, weight in zip(models, weights):
                if model_name == 'xgb':
                    preds = self.predict_xgb(user_id, top_k=top_k*2)  # Get more candidates
                elif model_name == 'lstm':
                    preds = self.predict_lstm(user_id, top_k=top_k*2)
                elif model_name == 'tcn':
                    preds = self.predict_tcn(user_id, top_k=top_k*2)
                else:
                    continue
                
                # Aggregate predictions
                for pred in preds:
                    product_id = pred['product_id']
                    score = pred['score'] * weight
                    
                    if product_id in all_predictions:
                        all_predictions[product_id]['score'] += score
                        all_predictions[product_id]['models'].append(model_name)
                    else:
                        all_predictions[product_id] = {
                            'product_id': product_id,
                            'score': score,
                            'models': [model_name]
                        }
            
            # Sort by ensemble score
            ensemble_recommendations = list(all_predictions.values())
            ensemble_recommendations.sort(key=lambda x: x['score'], reverse=True)
            
            # Format output
            final_recommendations = []
            for rec in ensemble_recommendations[:top_k]:
                final_recommendations.append({
                    'product_id': rec['product_id'],
                    'score': rec['score'],
                    'model': 'ensemble',
                    'contributing_models': rec['models']
                })
            
            return final_recommendations
            
        except Exception as e:
            logger.error(f"Ensemble prediction failed: {str(e)}")
            return self._fallback_recommendations(top_k)
    
    def _get_user_sequence(self, user_id: int) -> List[int]:
        """Get user's purchase sequence (cached or from database)"""
        
        if user_id in self.sequence_cache:
            return self.sequence_cache[user_id]
        
        # In production, this would query the database
        # For now, return empty sequence
        sequence = []
        
        # Cache the sequence
        if self.config.enable_cache:
            self.sequence_cache[user_id] = sequence
        
        return sequence
    
    def _prepare_sequence(self, sequence: List[int]) -> np.ndarray:
        """Prepare sequence for model input"""
        
        max_length = 20  # This should come from model config
        
        if len(sequence) >= max_length:
            prepared = sequence[-max_length:]
        else:
            prepared = [0] * (max_length - len(sequence)) + sequence
        
        return np.array([prepared])  # Add batch dimension
    
    def _fallback_recommendations(self, top_k: int) -> List[Dict[str, Any]]:
        """Fallback to popular products when models fail"""
        
        if not self.popular_products:
            return []
        
        recommendations = []
        for i, product_id in enumerate(self.popular_products[:top_k]):
            recommendations.append({
                'product_id': product_id,
                'score': 1.0 - (i * 0.1),  # Decreasing scores
                'model': 'fallback'
            })
        
        return recommendations
    
    def get_batch_recommendations(self,
                                 user_ids: List[int],
                                 model_name: str = 'ensemble',
                                 top_k: int = 10) -> Dict[int, List[Dict[str, Any]]]:
        """Get recommendations for multiple users"""
        
        start_time = time.time()
        results = {}
        
        for user_id in user_ids:
            if model_name == 'ensemble':
                results[user_id] = self.predict_ensemble(user_id, top_k=top_k)
            elif model_name == 'xgb':
                results[user_id] = self.predict_xgb(user_id, top_k=top_k)
            elif model_name == 'lstm':
                results[user_id] = self.predict_lstm(user_id, top_k=top_k)
            elif model_name == 'tcn':
                results[user_id] = self.predict_tcn(user_id, top_k=top_k)
            else:
                results[user_id] = self._fallback_recommendations(top_k)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Generated recommendations for {len(user_ids)} users in {elapsed_time:.2f}s")
        
        return results
    
    def clear_cache(self):
        """Clear all caches"""
        
        self.user_features_cache.clear()
        self.item_features_cache.clear()
        self.sequence_cache.clear()
        
        logger.info("Inference caches cleared")