"""
Model Loading and Management for Serving
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
import joblib
import pickle
from datetime import datetime

import tensorflow as tf
from tensorflow import keras

from ..utils.config import ServingConfig
from ..utils.io import load_json, load_parquet

logger = logging.getLogger(__name__)

class ModelLoader:
    """Load and manage trained models for serving"""
    
    def __init__(self, config: ServingConfig):
        self.config = config
        self.models = {}
        self.model_metadata = {}
        self.feature_encoders = {}
        
    def load_xgb_model(self, model_path: Path, model_name: str = "xgb") -> bool:
        """Load XGBoost model"""
        
        try:
            model_path = Path(model_path)
            
            # Load model
            if model_path.suffix == '.joblib':
                model = joblib.load(model_path)
            else:
                # Try to find the latest model
                model_files = list(model_path.glob("xgboost_baseline_*.joblib"))
                if not model_files:
                    raise FileNotFoundError(f"No XGBoost model found in {model_path}")
                
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                model = joblib.load(latest_model)
                logger.info(f"Loaded latest XGB model: {latest_model}")
            
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'type': 'xgboost',
                'path': str(model_path),
                'loaded_at': datetime.now().isoformat()
            }
            
            logger.info(f"XGBoost model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {str(e)}")
            return False
    
    def load_lstm_model(self, model_path: Path, model_name: str = "lstm") -> bool:
        """Load LSTM model"""
        
        try:
            model_path = Path(model_path)
            
            # Load model
            if model_path.suffix == '.h5':
                model = keras.models.load_model(model_path)
            else:
                # Try to find the latest model
                model_files = list(model_path.glob("lstm_model_*.h5"))
                if not model_files:
                    raise FileNotFoundError(f"No LSTM model found in {model_path}")
                
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                model = keras.models.load_model(latest_model)
                logger.info(f"Loaded latest LSTM model: {latest_model}")
            
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'type': 'lstm',
                'path': str(model_path),
                'loaded_at': datetime.now().isoformat(),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
            
            logger.info(f"LSTM model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {str(e)}")
            return False
    
    def load_tcn_model(self, model_path: Path, model_name: str = "tcn") -> bool:
        """Load TCN model"""
        
        try:
            model_path = Path(model_path)
            
            # Load model
            if model_path.suffix == '.h5':
                try:
                    model = keras.models.load_model(model_path)
                except Exception as e:
                    logger.warning(f"Failed to load TCN model directly: {str(e)}")
                    # Try loading weights + config
                    return self._load_tcn_weights_config(model_path, model_name)
            else:
                # Try to find the latest model
                model_files = list(model_path.glob("tcn_model_*.h5"))
                weights_files = list(model_path.glob("tcn_weights_*.h5"))
                
                if model_files:
                    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                    model = keras.models.load_model(latest_model)
                    logger.info(f"Loaded latest TCN model: {latest_model}")
                elif weights_files:
                    latest_weights = max(weights_files, key=lambda x: x.stat().st_mtime)
                    return self._load_tcn_weights_config(latest_weights, model_name)
                else:
                    raise FileNotFoundError(f"No TCN model found in {model_path}")
            
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'type': 'tcn',
                'path': str(model_path),
                'loaded_at': datetime.now().isoformat(),
                'input_shape': model.input_shape,
                'output_shape': model.output_shape
            }
            
            logger.info(f"TCN model '{model_name}' loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TCN model: {str(e)}")
            return False
    
    def _load_tcn_weights_config(self, weights_path: Path, model_name: str) -> bool:
        """Load TCN model from weights and config files"""
        
        try:
            weights_path = Path(weights_path)
            timestamp = weights_path.stem.split('_')[-1]
            
            # Find corresponding config file
            config_file = weights_path.parent / f"tcn_config_{timestamp}.json"
            if not config_file.exists():
                logger.error(f"Config file not found: {config_file}")
                return False
            
            # Load config and rebuild model
            model_config = load_json(config_file)
            model = keras.Model.from_config(model_config)
            
            # Load weights
            model.load_weights(weights_path)
            
            self.models[model_name] = model
            self.model_metadata[model_name] = {
                'type': 'tcn',
                'path': str(weights_path),
                'loaded_at': datetime.now().isoformat(),
                'loaded_from_weights': True
            }
            
            logger.info(f"TCN model '{model_name}' loaded from weights and config")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load TCN from weights/config: {str(e)}")
            return False
    
    def load_feature_encoders(self, encoders_path: Path) -> bool:
        """Load feature encoders and preprocessors"""
        
        try:
            encoders_path = Path(encoders_path)
            
            # Load different types of encoders
            encoder_files = {
                'product_encoder': list(encoders_path.glob("*product_encoder*.pkl")),
                'user_scaler': list(encoders_path.glob("*user_scaler*.pkl")),
                'item_scaler': list(encoders_path.glob("*item_scaler*.pkl"))
            }
            
            for encoder_name, files in encoder_files.items():
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    with open(latest_file, 'rb') as f:
                        encoder = pickle.load(f)
                    
                    self.feature_encoders[encoder_name] = encoder
                    logger.info(f"Loaded {encoder_name} from {latest_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load feature encoders: {str(e)}")
            return False
    
    def get_model(self, model_name: str):
        """Get loaded model"""
        
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        return self.models[model_name]
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model metadata"""
        
        if model_name not in self.model_metadata:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        return self.model_metadata[model_name]
    
    def list_loaded_models(self) -> List[str]:
        """List all loaded models"""
        
        return list(self.models.keys())
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a model to free memory"""
        
        if model_name in self.models:
            del self.models[model_name]
            
        if model_name in self.model_metadata:
            del self.model_metadata[model_name]
        
        logger.info(f"Model '{model_name}' unloaded")
        return True
    
    def reload_model(self, model_name: str, model_path: Path) -> bool:
        """Reload a model"""
        
        # Unload existing model
        if model_name in self.models:
            self.unload_model(model_name)
        
        # Determine model type and reload
        if 'xgb' in model_name.lower():
            return self.load_xgb_model(model_path, model_name)
        elif 'lstm' in model_name.lower():
            return self.load_lstm_model(model_path, model_name)
        elif 'tcn' in model_name.lower():
            return self.load_tcn_model(model_path, model_name)
        else:
            logger.error(f"Unknown model type for '{model_name}'")
            return False