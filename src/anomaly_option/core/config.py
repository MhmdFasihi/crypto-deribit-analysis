"""
Configuration management for the Deribit analysis system.
Handles environment variables, default settings, and configuration validation.
"""

import os
from pathlib import Path
from datetime import datetime, timedelta
import logging
import logging.config
from typing import Dict, Any, Optional, List, Union
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Define constants
DEFAULT_CACHE_DIR = BASE_DIR / "options_cache"
DEFAULT_RESULTS_DIR = BASE_DIR / "results"
DEFAULT_LOG_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
DEFAULT_CACHE_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_RESULTS_DIR.mkdir(exist_ok=True, parents=True)
DEFAULT_LOG_DIR.mkdir(exist_ok=True, parents=True)


class Config:
    """Configuration class for managing settings."""

    # API settings
    API_KEY = os.getenv('DERIBIT_API_KEY', '')
    API_SECRET = os.getenv('DERIBIT_API_SECRET', '')
    API_URL = os.getenv('DERIBIT_API_URL', 'https://www.deribit.com')
    API_WS_URL = os.getenv('DERIBIT_WS_URL', 'wss://www.deribit.com/ws/api/v2')
    API_TEST_URL = os.getenv('DERIBIT_API_TEST_URL', 'https://test.deribit.com')
    API_TEST_WS_URL = os.getenv('DERIBIT_WS_TEST_URL', 'wss://test.deribit.com/ws/api/v2')
    USE_TEST_ENV = os.getenv('USE_TEST_ENV', 'False').lower() in ('true', '1', 't')

    # Rate limiting
    MAX_REQUESTS_PER_SECOND = int(os.getenv('MAX_REQUESTS_PER_SECOND', '5'))
    MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY = int(os.getenv('RETRY_DELAY', '2'))
    TIMEOUT = int(os.getenv('TIMEOUT', '30'))

    # Data fetching
    MAX_WORKERS = int(os.getenv('MAX_WORKERS', '4'))
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', '1000'))
    DEFAULT_SYMBOLS = os.getenv('DEFAULT_SYMBOLS', 'BTC-USD,ETH-USD').split(',')
    
    # Analysis parameters
    WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '30'))
    Z_THRESHOLD = float(os.getenv('Z_THRESHOLD', '3.0'))
    VOLATILITY_WINDOW = int(os.getenv('VOLATILITY_WINDOW', '21'))
    ANNUALIZATION_FACTOR = int(os.getenv('ANNUALIZATION_FACTOR', '365'))
    RISK_FREE_RATE = float(os.getenv('RISK_FREE_RATE', '0.05'))
    
    # Paths
    CACHE_DIR = Path(os.getenv('CACHE_DIR', str(DEFAULT_CACHE_DIR)))
    RESULTS_DIR = Path(os.getenv('RESULTS_DIR', str(DEFAULT_RESULTS_DIR)))
    LOG_DIR = Path(os.getenv('LOG_DIR', str(DEFAULT_LOG_DIR)))
    
    # Logging level
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_api_url(cls) -> str:
        """Get the API URL based on environment configuration."""
        return cls.API_TEST_URL if cls.USE_TEST_ENV else cls.API_URL
    
    @classmethod
    def get_ws_url(cls) -> str:
        """Get the WebSocket URL based on environment configuration."""
        return cls.API_TEST_WS_URL if cls.USE_TEST_ENV else cls.API_WS_URL
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate the configuration and return a list of issues."""
        issues = []
        
        # Only warn about API credentials if explicitly needed
        # Allow access without credentials for price data fetching or when using test environment
        if not cls.USE_TEST_ENV and (not cls.API_KEY or not cls.API_SECRET):
            issues.append("API credentials missing - will only be able to fetch public data")
        
        # Check rate limiting settings
        if cls.MAX_REQUESTS_PER_SECOND <= 0:
            issues.append("MAX_REQUESTS_PER_SECOND must be positive.")
        
        if cls.MAX_RETRIES < 0:
            issues.append("MAX_RETRIES must be non-negative.")
        
        if cls.RETRY_DELAY <= 0:
            issues.append("RETRY_DELAY must be positive.")
        
        if cls.TIMEOUT <= 0:
            issues.append("TIMEOUT must be positive.")
        
        # Check analysis parameters
        if cls.WINDOW_SIZE < 2:
            issues.append("WINDOW_SIZE must be at least 2.")
        
        if cls.Z_THRESHOLD <= 0:
            issues.append("Z_THRESHOLD must be positive.")
        
        if cls.VOLATILITY_WINDOW < 2:
            issues.append("VOLATILITY_WINDOW must be at least 2.")
        
        if cls.ANNUALIZATION_FACTOR <= 0:
            issues.append("ANNUALIZATION_FACTOR must be positive.")
        
        if cls.RISK_FREE_RATE < 0:
            issues.append("RISK_FREE_RATE must be non-negative.")
        
        return issues

    @classmethod
    def from_args(cls, args: Any) -> 'Config':
        """
        Create a Config instance from command line arguments.
        
        Args:
            args: Command line arguments
            
        Returns:
            Config instance
        """
        config = cls()
        
        # Update configuration from arguments
        if hasattr(args, 'symbols') and args.symbols:
            config.DEFAULT_SYMBOLS = args.symbols
            
        if hasattr(args, 'start_date') and args.start_date:
            config.DEFAULT_START_DATE = args.start_date
            
        if hasattr(args, 'end_date') and args.end_date:
            config.DEFAULT_END_DATE = args.end_date
            
        if hasattr(args, 'output_dir') and args.output_dir:
            config.RESULTS_DIR = Path(args.output_dir)
            
        if hasattr(args, 'alpha_vantage_key') and args.alpha_vantage_key:
            config.ALPHA_VANTAGE_API_KEY = args.alpha_vantage_key
            
        if hasattr(args, 'coinmarketcap_key') and args.coinmarketcap_key:
            config.COINMARKETCAP_API_KEY = args.coinmarketcap_key
            
        if hasattr(args, 'max_workers') and args.max_workers:
            config.MAX_WORKERS = args.max_workers
            
        if hasattr(args, 'batch_size') and args.batch_size:
            config.BATCH_SIZE = args.batch_size
            
        if hasattr(args, 'window_size') and args.window_size:
            config.WINDOW_SIZE = args.window_size
            
        if hasattr(args, 'z_threshold') and args.z_threshold:
            config.Z_THRESHOLD = args.z_threshold
            
        if hasattr(args, 'volatility_window') and args.volatility_window:
            config.VOLATILITY_WINDOW = args.volatility_window
            
        if hasattr(args, 'chunk_size') and args.chunk_size:
            config.CHUNK_SIZE = args.chunk_size
            
        if hasattr(args, 'debug') and args.debug:
            config.LOG_LEVEL = 'DEBUG'
            
        # Create necessary directories
        config._create_directories()
        
        return config
    
    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.CACHE_DIR.mkdir(exist_ok=True)
        self.RESULTS_DIR.mkdir(exist_ok=True)
        self.LOG_DIR.mkdir(exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.
        
        Returns:
            Dictionary with configuration values
        """
        return {
            'api': {
                'alpha_vantage_key': self.ALPHA_VANTAGE_API_KEY,
                'coinmarketcap_key': self.COINMARKETCAP_API_KEY
            },
            'data': {
                'default_symbols': self.DEFAULT_SYMBOLS,
                'default_start_date': self.DEFAULT_START_DATE,
                'default_end_date': self.DEFAULT_END_DATE,
                'data_dir': str(self.CACHE_DIR),
                'cache_dir': str(self.CACHE_DIR),
                'results_dir': str(self.RESULTS_DIR)
            },
            'analysis': {
                'window_size': self.WINDOW_SIZE,
                'z_threshold': self.Z_THRESHOLD,
                'volatility_window': self.VOLATILITY_WINDOW,
                'annualization_factor': self.ANNUALIZATION_FACTOR,
                'chunk_size': self.CHUNK_SIZE
            },
            'performance': {
                'max_workers': self.MAX_WORKERS,
                'batch_size': self.BATCH_SIZE,
                'cache_expiry': self.CACHE_EXPIRY
            },
            'logging': {
                'log_level': self.LOG_LEVEL,
                'log_format': self.LOG_FORMAT,
                'log_dir': str(self.LOG_DIR)
            }
        }


# Configure logging
def setup_logging():
    """Set up logging configuration."""
    log_file = Config.LOG_DIR / f"crypto_analysis_{datetime.now().strftime('%Y%m%d')}.log"
    
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
            'detailed': {
                'format': '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d: %(message)s'
            }
        },
        'handlers': {
            'console': {
                'level': Config.LOG_LEVEL,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'detailed',
                'filename': str(log_file),
                'maxBytes': 10485760,  # 10 MB
                'backupCount': 5
            }
        },
        'loggers': {
            '': {  # Root logger
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    return logging.getLogger('crypto_analysis')


# Create logger
logger = setup_logging()


# Validate configuration on import
issues = Config.validate()
if issues:
    for issue in issues:
        logger.warning(f"Configuration issue: {issue}")