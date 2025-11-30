"""
Data ingestion module for Instacart recommendation system.
Copy logic from notebook 01 (EDA) for loading raw CSV files.
"""
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
import logging

from src.utils import get_logger, load_csv, save_parquet, ensure_directories

logger = get_logger(__name__)

class DataIngestor:
    """Raw data ingestion and basic validation."""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Ensure directories exist
        ensure_directories(str(self.processed_data_path))
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Load all raw CSV files."""
        
        logger.info("Loading raw data files...")
        
        # Expected raw files
        raw_files = {
            'orders': 'orders.csv',
            'order_products_train': 'order_products__train.csv', 
            'order_products_prior': 'order_products__prior.csv',
            'products': 'products.csv',
            'aisles': 'aisles.csv',
            'departments': 'departments.csv'
        }
        
        data = {}
        
        for name, filename in raw_files.items():
            file_path = self.raw_data_path / filename
            
            if file_path.exists():
                df = load_csv(str(file_path))
                data[name] = df
                logger.info(f"Loaded {name}: {df.shape}")
            else:
                logger.error(f"Raw file not found: {file_path}")
                raise FileNotFoundError(f"Required raw file missing: {filename}")
        
        return data
    
    def validate_raw_data(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate raw data integrity."""
        
        logger.info("Validating raw data...")
        
        # Expected schemas
        expected_schemas = {
            'orders': ['order_id', 'user_id', 'eval_set', 'order_number', 'order_dow', 'order_hour_of_day', 'days_since_prior_order'],
            'order_products_train': ['order_id', 'product_id', 'add_to_cart_order', 'reordered'],
            'order_products_prior': ['order_id', 'product_id', 'add_to_cart_order', 'reordered'],
            'products': ['product_id', 'product_name', 'aisle_id', 'department_id'],
            'aisles': ['aisle_id', 'aisle'],
            'departments': ['department_id', 'department']
        }
        
        validation_passed = True
        
        for table_name, expected_columns in expected_schemas.items():
            if table_name not in data:
                logger.error(f"Missing table: {table_name}")
                validation_passed = False
                continue
            
            df = data[table_name]
            missing_columns = set(expected_columns) - set(df.columns)
            
            if missing_columns:
                logger.error(f"Missing columns in {table_name}: {missing_columns}")
                validation_passed = False
            else:
                logger.info(f"✅ {table_name} schema validation passed")
        
        # Additional validation checks
        if validation_passed:
            validation_passed = self._validate_data_consistency(data)
        
        return validation_passed
    
    def _validate_data_consistency(self, data: Dict[str, pd.DataFrame]) -> bool:
        """Validate data consistency across tables."""
        
        logger.info("Validating data consistency...")
        
        # Check foreign key relationships
        checks_passed = True
        
        # Products should have valid aisle_id and department_id
        product_aisles = set(data['products']['aisle_id'].unique())
        valid_aisles = set(data['aisles']['aisle_id'].unique())
        invalid_aisles = product_aisles - valid_aisles
        
        if invalid_aisles:
            logger.warning(f"Products with invalid aisle_id: {len(invalid_aisles)} unique values")
        
        product_depts = set(data['products']['department_id'].unique()) 
        valid_depts = set(data['departments']['department_id'].unique())
        invalid_depts = product_depts - valid_depts
        
        if invalid_depts:
            logger.warning(f"Products with invalid department_id: {len(invalid_depts)} unique values")
        
        # Order products should reference valid orders and products
        if 'order_products_prior' in data:
            order_ids_in_products = set(data['order_products_prior']['order_id'].unique())
            valid_order_ids = set(data['orders']['order_id'].unique())
            invalid_orders = order_ids_in_products - valid_order_ids
            
            if invalid_orders:
                logger.warning(f"Order products with invalid order_id: {len(invalid_orders)} unique values")
        
        logger.info("✅ Data consistency validation completed")
        return checks_passed
    
    def save_processed_data(self, data: Dict[str, pd.DataFrame]) -> None:
        """Save processed data to parquet format."""
        
        logger.info("Saving processed data to parquet...")
        
        for name, df in data.items():
            output_path = self.processed_data_path / f"{name}.parquet"
            save_parquet(df, str(output_path), index=False)
            logger.info(f"Saved {name} to {output_path}")
    
    def ingest(self) -> Dict[str, pd.DataFrame]:
        """Main ingestion pipeline."""
        
        logger.info("Starting data ingestion pipeline...")
        
        # Load raw data
        data = self.load_raw_data()
        
        # Validate data
        if not self.validate_raw_data(data):
            raise ValueError("Raw data validation failed")
        
        # Basic preprocessing
        data = self._basic_preprocessing(data)
        
        # Save processed data
        self.save_processed_data(data)
        
        logger.info("✅ Data ingestion completed successfully")
        return data
    
    def _basic_preprocessing(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Basic preprocessing steps."""
        
        logger.info("Applying basic preprocessing...")
        
        processed_data = {}
        
        for name, df in data.items():
            df_processed = df.copy()
            
            # Handle missing values
            if name == 'orders':
                # days_since_prior_order is NaN for first orders
                df_processed['days_since_prior_order'] = df_processed['days_since_prior_order'].fillna(0)
            
            # Data type optimization
            df_processed = self._optimize_dtypes(df_processed)
            
            processed_data[name] = df_processed
        
        logger.info("Basic preprocessing completed")
        return processed_data
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency."""
        
        df_optimized = df.copy()
        
        # Convert object columns that are actually integers
        for col in df_optimized.select_dtypes(include=['object']):
            if df_optimized[col].str.isnumeric().all():
                df_optimized[col] = df_optimized[col].astype('int32')
        
        # Downcast integer columns
        for col in df_optimized.select_dtypes(include=['int64']):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
        
        # Downcast float columns
        for col in df_optimized.select_dtypes(include=['float64']):
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
        
        return df_optimized

def ingest_data(raw_data_path: str, processed_data_path: str) -> Dict[str, pd.DataFrame]:
    """Main function to ingest raw data."""
    
    ingestor = DataIngestor(raw_data_path, processed_data_path)
    return ingestor.ingest()

if __name__ == "__main__":
    # Example usage
    from src.utils import setup_logging
    
    setup_logging()
    
    # Ingest data
    data = ingest_data("data/raw", "data/processed")
    
    # Print summary
    for name, df in data.items():
        print(f"{name}: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        print()