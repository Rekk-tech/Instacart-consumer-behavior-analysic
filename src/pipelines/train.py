"""
Training Pipeline - Orchestrates complete ML pipeline
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime
import argparse

from ..utils.config import load_config, DataConfig, FeatureConfig, ModelConfig
from ..utils.logging import setup_logging
from ..utils.metrics import RecommendationMetrics
from ..utils.io import save_json, load_parquet

# Data pipeline components
from ..data.ingest import DataIngestor
from ..data.etl_rfm import RFMProcessor

# Feature engineering components  
from ..features import UserFeatureBuilder, ItemFeatureBuilder, UserItemFeatureBuilder, SequenceBuilder

# Model training components
from ..models import XGBTrainer, LSTMTrainer, TCNTrainer

logger = logging.getLogger(__name__)

class TrainingPipeline:
    """Complete training pipeline for recommendation models"""
    
    def __init__(self, config_path: str):
        """Initialize training pipeline with config"""
        
        self.config = load_config(config_path)
        self.results = {}
        self.artifacts = {}
        
        # Setup logging
        setup_logging()
        logger.info(f"Initialized training pipeline with config: {config_path}")
        
    def run_full_pipeline(self) -> Dict[str, Any]:
        """Run complete training pipeline"""
        
        logger.info("Starting full training pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Data Ingestion
            logger.info("Step 1: Data Ingestion")
            raw_data = self.run_data_ingestion()
            
            # Step 2: ETL Processing
            logger.info("Step 2: ETL Processing")  
            processed_data = self.run_etl_processing(raw_data)
            
            # Step 3: Feature Engineering
            logger.info("Step 3: Feature Engineering")
            features = self.run_feature_engineering(processed_data)
            
            # Step 4: Model Training
            logger.info("Step 4: Model Training")
            model_results = self.run_model_training(features)
            
            # Step 5: Model Evaluation
            logger.info("Step 5: Model Evaluation")
            evaluation_results = self.run_model_evaluation(model_results, features)
            
            # Compile final results
            total_time = (datetime.now() - start_time).total_seconds()
            
            final_results = {
                'pipeline_config': self.config.model.model_type,
                'total_runtime_seconds': total_time,
                'data_stats': self.results.get('data_stats', {}),
                'feature_stats': self.results.get('feature_stats', {}),
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'artifacts_saved': list(self.artifacts.keys()),
                'completed_at': datetime.now().isoformat()
            }
            
            # Save pipeline results
            self._save_pipeline_results(final_results)
            
            logger.info(f"Training pipeline completed successfully in {total_time:.2f}s")
            return final_results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise
    
    def run_data_ingestion(self) -> Dict[str, pd.DataFrame]:
        """Run data ingestion step"""
        
        ingester = DataIngestor(
            self.config.data.raw_data_path,
            self.config.data.processed_data_path
        )
        
        # Load raw data
        raw_data = ingester.load_raw_data()
        
        # Store data statistics
        data_stats = {}
        for name, df in raw_data.items():
            data_stats[name] = {
                'rows': len(df),
                'columns': len(df.columns),
                'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
            }
        
        self.results['data_stats'] = data_stats
        logger.info(f"Data ingestion completed: {len(raw_data)} datasets loaded")
        
        return raw_data
    
    def run_etl_processing(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Run ETL processing step"""
        
        # RFM Analysis
        rfm_analyzer = RFMProcessor(self.config.data)
        
        # Process orders data
        processed_orders = rfm_analyzer.process_orders(
            raw_data['orders'],
            raw_data['order_products__prior']
        )
        
        # Generate RFM features
        rfm_features = rfm_analyzer.compute_rfm_features(processed_orders)
        
        processed_data = raw_data.copy()
        processed_data['processed_orders'] = processed_orders
        processed_data['rfm_features'] = rfm_features
        
        logger.info("ETL processing completed")
        return processed_data
    
    def run_feature_engineering(self, processed_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Run feature engineering step"""
        
        features = {}
        
        # User features
        user_builder = UserFeatureBuilder(self.config.features)
        user_features = user_builder.build_features(
            processed_data['orders'],
            processed_data['order_products__prior'],
            processed_data['products'],
            processed_data['aisles'], 
            processed_data['departments']
        )
        features['user_features'] = user_features
        
        # Item features
        item_builder = ItemFeatureBuilder(self.config.features)
        item_features = item_builder.build_features(
            processed_data['orders'],
            processed_data['order_products__prior'],
            processed_data['products'],
            processed_data['aisles'],
            processed_data['departments']
        )
        features['item_features'] = item_features
        
        # User-Item features (for XGBoost)
        if self.config.model.model_type == 'xgb':
            user_item_builder = UserItemFeatureBuilder(self.config.features)
            user_item_features = user_item_builder.build_features(
                processed_data['orders'],
                processed_data['order_products__prior'],
                user_features,
                item_features
            )
            features['user_item_features'] = user_item_features
        
        # Sequences (for LSTM/TCN)
        if self.config.model.model_type in ['lstm', 'tcn']:
            sequence_builder = SequenceBuilder(self.config.features)
            sequences, targets, metadata = sequence_builder.build_sequences(
                processed_data['orders'],
                processed_data['order_products__prior']
            )
            features['sequences'] = sequences
            features['targets'] = targets
            features['sequence_metadata'] = metadata
        
        # Store feature statistics
        feature_stats = {}
        for name, feature_data in features.items():
            if isinstance(feature_data, pd.DataFrame):
                feature_stats[name] = {
                    'rows': len(feature_data),
                    'columns': len(feature_data.columns)
                }
            elif isinstance(feature_data, np.ndarray):
                feature_stats[name] = {
                    'shape': feature_data.shape,
                    'dtype': str(feature_data.dtype)
                }
        
        self.results['feature_stats'] = feature_stats
        logger.info("Feature engineering completed")
        
        return features
    
    def run_model_training(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run model training step"""
        
        model_type = self.config.model.model_type
        
        if model_type == 'xgb':
            return self._train_xgb_model(features)
        elif model_type == 'lstm':
            return self._train_lstm_model(features)
        elif model_type == 'tcn':
            return self._train_tcn_model(features)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _train_xgb_model(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train XGBoost model"""
        
        trainer = XGBTrainer(self.config.model)
        
        # Prepare data
        X = features['user_item_features']
        # Create dummy target (in practice, this would be from order_products__train)
        y = pd.Series(np.random.binomial(1, 0.1, len(X)), name='reordered')
        
        # Train model
        training_results = trainer.train(X, y)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = trainer.save_model(Path("models"), timestamp)
        
        self.artifacts['xgb_model'] = str(model_path)
        
        return training_results
    
    def _train_lstm_model(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train LSTM model"""
        
        trainer = LSTMTrainer(self.config.model)
        
        # Get sequence data
        sequences = features['sequences']
        targets = features['targets']
        metadata = features['sequence_metadata']
        
        # Train model
        training_results = trainer.train(
            sequences, 
            targets,
            metadata['vocab_size']
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = trainer.save_model(Path("models"), timestamp)
        
        self.artifacts['lstm_model'] = str(model_path)
        
        return training_results
    
    def _train_tcn_model(self, features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Train TCN model"""
        
        trainer = TCNTrainer(self.config.model)
        
        # Get sequence data
        sequences = features['sequences']
        targets = features['targets']
        metadata = features['sequence_metadata']
        
        # Train model
        training_results = trainer.train(
            sequences,
            targets, 
            metadata['vocab_size']
        )
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = trainer.save_model(Path("models"), timestamp)
        
        self.artifacts['tcn_model'] = str(model_path)
        
        return training_results
    
    def run_model_evaluation(self, 
                           model_results: Dict[str, Any],
                           features: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Run model evaluation step"""
        
        # Placeholder evaluation - in practice would use test set
        evaluation_results = {
            'training_metrics': model_results,
            'test_metrics': {
                'placeholder': 'Would evaluate on test set'
            }
        }
        
        return evaluation_results
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save pipeline results to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.config.model.model_type
        
        results_file = Path("models") / f"training_results_{model_type}_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True, parents=True)
        
        save_json(results, results_file)
        
        self.artifacts['pipeline_results'] = str(results_file)
        logger.info(f"Pipeline results saved to {results_file}")

def main():
    """Main training script"""
    
    parser = argparse.ArgumentParser(description="Run training pipeline")
    parser.add_argument("--config", type=str, required=True, 
                       help="Path to configuration file")
    parser.add_argument("--steps", nargs="*", 
                       choices=["data", "etl", "features", "train", "evaluate"],
                       help="Specific steps to run (default: all)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrainingPipeline(args.config)
    
    try:
        if args.steps:
            logger.info(f"Running specific steps: {args.steps}")
            # TODO: Implement step-by-step execution
            results = pipeline.run_full_pipeline()
        else:
            # Run full pipeline
            results = pipeline.run_full_pipeline()
        
        print("\n" + "="*60)
        print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")  
        print("="*60)
        print(f"Model Type: {results['pipeline_config']}")
        print(f"Total Runtime: {results['total_runtime_seconds']:.2f}s")
        print(f"Artifacts: {len(results['artifacts_saved'])}")
        print("="*60)
        
    except Exception as e:
        print(f"\nTraining pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()