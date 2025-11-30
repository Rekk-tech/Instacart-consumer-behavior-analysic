"""
I/O utilities for Instacart recommendation system.
"""
import pandas as pd
import numpy as np
import pickle
import joblib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)

def load_csv(file_path: str, **kwargs) -> pd.DataFrame:
    """Load CSV file with error handling."""
    try:
        df = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded CSV: {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading CSV {file_path}: {e}")
        raise

def save_csv(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """Save DataFrame to CSV with error handling."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(file_path, **kwargs)
        logger.info(f"Saved CSV: {file_path} with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error saving CSV {file_path}: {e}")
        raise

def load_parquet(file_path: str, **kwargs) -> pd.DataFrame:
    """Load Parquet file with error handling."""
    try:
        df = pd.read_parquet(file_path, **kwargs)
        logger.info(f"Loaded Parquet: {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading Parquet {file_path}: {e}")
        raise

def save_parquet(df: pd.DataFrame, file_path: str, **kwargs) -> None:
    """Save DataFrame to Parquet with error handling."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        df.to_parquet(file_path, **kwargs)
        logger.info(f"Saved Parquet: {file_path} with shape {df.shape}")
    except Exception as e:
        logger.error(f"Error saving Parquet {file_path}: {e}")
        raise

def load_pickle(file_path: str) -> Any:
    """Load pickle file with error handling."""
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Loaded pickle: {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading pickle {file_path}: {e}")
        raise

def save_pickle(obj: Any, file_path: str) -> None:
    """Save object to pickle with error handling."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Saved pickle: {file_path}")
    except Exception as e:
        logger.error(f"Error saving pickle {file_path}: {e}")
        raise

def load_json(file_path: str) -> Dict[str, Any]:
    """Load JSON file with error handling."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded JSON: {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON {file_path}: {e}")
        raise

def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Save data to JSON with error handling."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Saved JSON: {file_path}")
    except Exception as e:
        logger.error(f"Error saving JSON {file_path}: {e}")
        raise

def load_model(file_path: str) -> Any:
    """Load model with automatic format detection."""
    try:
        file_path = Path(file_path)
        
        if file_path.suffix == '.joblib':
            model = joblib.load(file_path)
        elif file_path.suffix in ['.pkl', '.pickle']:
            model = load_pickle(str(file_path))
        elif file_path.suffix in ['.h5', '.keras']:
            # TensorFlow/Keras model
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(file_path)
            except ImportError:
                logger.error("TensorFlow not installed, cannot load .h5/.keras model")
                raise
        else:
            # Try joblib first, then pickle
            try:
                model = joblib.load(file_path)
            except:
                model = load_pickle(str(file_path))
        
        logger.info(f"Loaded model: {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model {file_path}: {e}")
        raise

def save_model(model: Any, file_path: str) -> None:
    """Save model with automatic format detection."""
    try:
        file_path = Path(file_path)
        
        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_path.suffix == '.joblib':
            joblib.dump(model, file_path)
        elif file_path.suffix in ['.pkl', '.pickle']:
            save_pickle(model, str(file_path))
        elif file_path.suffix in ['.h5', '.keras']:
            # TensorFlow/Keras model
            model.save(file_path)
        else:
            # Default to joblib
            joblib.dump(model, file_path)
        
        logger.info(f"Saved model: {file_path}")
    except Exception as e:
        logger.error(f"Error saving model {file_path}: {e}")
        raise

def ensure_directories(*paths: str) -> None:
    """Ensure all directories exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory: {path}")

def list_files(directory: str, pattern: str = "*", recursive: bool = False) -> List[str]:
    """List files in directory matching pattern."""
    try:
        path = Path(directory)
        if recursive:
            files = list(path.rglob(pattern))
        else:
            files = list(path.glob(pattern))
        
        file_paths = [str(f) for f in files if f.is_file()]
        logger.info(f"Found {len(file_paths)} files in {directory} matching '{pattern}'")
        return file_paths
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        return []

def get_file_info(file_path: str) -> Dict[str, Any]:
    """Get file information."""
    try:
        path = Path(file_path)
        if not path.exists():
            return {"exists": False}
        
        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / (1024 * 1024),
            "modified_time": stat.st_mtime,
            "is_file": path.is_file(),
            "is_directory": path.is_dir(),
            "suffix": path.suffix
        }
    except Exception as e:
        logger.error(f"Error getting file info for {file_path}: {e}")
        return {"exists": False, "error": str(e)}

def check_data_integrity(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """Check data integrity and quality."""
    
    report = {
        "shape": df.shape,
        "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
        "column_count": len(df.columns),
        "missing_columns": [],
        "column_stats": {}
    }
    
    # Check for missing required columns
    missing_columns = set(required_columns) - set(df.columns)
    report["missing_columns"] = list(missing_columns)
    
    # Column statistics
    for col in df.columns:
        col_stats = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isnull().sum()),
            "null_percentage": float(df[col].isnull().mean() * 100),
            "unique_count": int(df[col].nunique())
        }
        
        if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            col_stats.update({
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "min": float(df[col].min()), 
                "max": float(df[col].max())
            })
        
        report["column_stats"][col] = col_stats
    
    return report

class DataFrameValidator:
    """Validator class for DataFrames."""
    
    def __init__(self, required_columns: List[str]):
        self.required_columns = required_columns
    
    def validate(self, df: pd.DataFrame, name: str = "DataFrame") -> bool:
        """Validate DataFrame meets requirements."""
        
        logger.info(f"Validating {name} with shape {df.shape}")
        
        # Check for required columns
        missing_columns = set(self.required_columns) - set(df.columns)
        if missing_columns:
            logger.error(f"Missing required columns in {name}: {missing_columns}")
            return False
        
        # Check for empty DataFrame
        if df.empty:
            logger.error(f"{name} is empty")
            return False
        
        # Check for all-null columns
        null_columns = df.columns[df.isnull().all()].tolist()
        if null_columns:
            logger.warning(f"All-null columns in {name}: {null_columns}")
        
        logger.info(f"✅ {name} validation passed")
        return True