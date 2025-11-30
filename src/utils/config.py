"""
Configuration management for Instacart recommendation system.
"""
import os
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Data pipeline configuration."""
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed" 
    features_data_path: str = "data/features"
    splits_data_path: str = "data/splits"
    
    # ETL settings
    test_order_percentage: float = 0.1
    min_orders_per_user: int = 4
    min_products_per_user: int = 10
    
    # RFM settings
    rfm_quantiles: int = 5
    
@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # User features
    user_feature_categories: List[str] = None
    
    # Item features  
    item_feature_categories: List[str] = None
    
    # Sequence settings
    max_sequence_length: int = 20
    vocab_size_limit: int = 10000
    
    def __post_init__(self):
        if self.user_feature_categories is None:
            self.user_feature_categories = [
                'order_patterns', 'basket_patterns', 'product_patterns', 
                'department_aisle', 'rfm_metrics'
            ]
        if self.item_feature_categories is None:
            self.item_feature_categories = [
                'popularity', 'reorder_patterns', 'department_aisle'
            ]

@dataclass  
class ModelConfig:
    """Model training configuration."""
    model_type: str = "xgb"  # xgb, lstm, tcn
    
    # Training settings
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    
    # XGBoost settings
    xgb_params: Dict[str, Any] = None
    
    # Deep learning settings
    lstm_params: Dict[str, Any] = None
    tcn_params: Dict[str, Any] = None
    
    # Training params
    epochs: int = 50
    batch_size: int = 256
    patience: int = 5
    learning_rate: float = 1e-3
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        
        if self.lstm_params is None:
            self.lstm_params = {
                'embed_dim': 128,
                'lstm_units': [64, 32],
                'dense_units': [128, 64],
                'dropout_rates': [0.2, 0.3, 0.2]
            }
            
        if self.tcn_params is None:
            self.tcn_params = {
                'embed_dim': 128,
                'nb_filters': 64,
                'kernel_size': 3,
                'dilations': [1, 2, 4, 8],
                'dense_units': [128, 64],
                'dropout_rates': [0.2, 0.3, 0.2]
            }

@dataclass
class ServingConfig:
    """Model serving configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "models/archived"
    max_recommendations: int = 10
    batch_inference_size: int = 1000
    
    # Cache settings
    enable_cache: bool = True
    cache_ttl: int = 3600  # 1 hour
    
@dataclass
class Config:
    """Main configuration class."""
    project_root: str = "."
    data: DataConfig = None
    features: FeatureConfig = None  
    model: ModelConfig = None
    serving: ServingConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.features is None:
            self.features = FeatureConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.serving is None:
            self.serving = ServingConfig()

def load_config(config_path: str) -> Config:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        
        # Convert nested dicts to dataclass instances
        data_config = DataConfig(**config_dict.get('data', {}))
        features_config = FeatureConfig(**config_dict.get('features', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        serving_config = ServingConfig(**config_dict.get('serving', {}))
        
        return Config(
            project_root=config_dict.get('project_root', '.'),
            data=data_config,
            features=features_config,
            model=model_config,
            serving=serving_config
        )
        
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        logger.info("Using default configuration")
        return Config()

def get_abs_path(relative_path: str, project_root: str = ".") -> str:
    """Convert relative path to absolute path."""
    return str(Path(project_root) / relative_path)

def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not."""
    Path(path).mkdir(parents=True, exist_ok=True)