"""
Logging configuration for Instacart recommendation system.
"""
import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    """Setup logging configuration."""
    
    # Create logs directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"instacart_recommender_{timestamp}.log"
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': str(log_file),
                'mode': 'a'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            },
            'urllib3': {
                'level': 'WARNING'
            },
            'requests': {
                'level': 'WARNING'
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging setup complete. Log file: {log_file}")

def get_logger(name: str) -> logging.Logger:
    """Get logger instance."""
    return logging.getLogger(name)