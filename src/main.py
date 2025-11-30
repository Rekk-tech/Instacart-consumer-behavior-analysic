"""
Main Entry Point for Instacart Recommendation System
"""

import argparse
import sys
from pathlib import Path

def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Instacart Recommendation System - Production CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost model
  python -m src.main train --config configs/train_xgb.yaml
  
  # Train LSTM model  
  python -m src.main train --config configs/train_lstm.yaml
  
  # Start API server
  python -m src.main serve --config configs/train_xgb.yaml --port 8000
  
  # Run data processing only
  python -m src.main pipeline --steps data etl --config configs/train_xgb.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train ML models')
    train_parser.add_argument('--config', type=str, required=True,
                             help='Path to configuration file')
    train_parser.add_argument('--steps', nargs='*',
                             choices=['data', 'etl', 'features', 'train', 'evaluate'],
                             help='Specific pipeline steps to run')
    
    # Serving command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--config', type=str, 
                             help='Path to configuration file')
    serve_parser.add_argument('--host', type=str, default='0.0.0.0',
                             help='Host to bind to')
    serve_parser.add_argument('--port', type=int, default=8000,
                             help='Port to bind to')
    serve_parser.add_argument('--reload', action='store_true',
                             help='Enable auto-reload for development')
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run data pipeline')
    pipeline_parser.add_argument('--config', type=str, required=True,
                                help='Path to configuration file')
    pipeline_parser.add_argument('--steps', nargs='+', required=True,
                                choices=['data', 'etl', 'features'],
                                help='Pipeline steps to run')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute commands
    if args.command == 'train':
        from .pipelines.train import main as train_main
        
        # Temporarily modify sys.argv for argparse in train module
        original_argv = sys.argv[:]
        sys.argv = ['train.py', '--config', args.config]
        if args.steps:
            sys.argv.extend(['--steps'] + args.steps)
        
        try:
            train_main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'serve':
        from .serving.api import main as serve_main
        
        # Temporarily modify sys.argv for argparse in serve module
        original_argv = sys.argv[:]
        sys.argv = ['api.py']
        if args.config:
            sys.argv.extend(['--config', args.config])
        sys.argv.extend(['--host', args.host, '--port', str(args.port)])
        if args.reload:
            sys.argv.append('--reload')
        
        try:
            serve_main()
        finally:
            sys.argv = original_argv
    
    elif args.command == 'pipeline':
        from .pipelines.train import TrainingPipeline
        from .utils.logging import setup_logging
        
        setup_logging()
        
        pipeline = TrainingPipeline(args.config)
        
        print(f"Running pipeline steps: {args.steps}")
        
        # Run specific steps
        raw_data = None
        processed_data = None
        
        if 'data' in args.steps:
            print("Running data ingestion...")
            raw_data = pipeline.run_data_ingestion()
            print("Data ingestion completed")
        
        if 'etl' in args.steps:
            print("Running ETL processing...")
            if raw_data is None:
                raw_data = pipeline.run_data_ingestion()
            processed_data = pipeline.run_etl_processing(raw_data)
            print("ETL processing completed")
        
        if 'features' in args.steps:
            print("Running feature engineering...")
            if raw_data is None:
                raw_data = pipeline.run_data_ingestion()
            if processed_data is None:
                processed_data = pipeline.run_etl_processing(raw_data)
            features = pipeline.run_feature_engineering(processed_data)
            print("Feature engineering completed")
        
        print("Pipeline steps completed successfully!")

if __name__ == '__main__':
    main()