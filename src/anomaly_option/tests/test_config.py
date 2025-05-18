"""
Tests for the configuration module.
"""

import os
import pytest
from pathlib import Path
from ..core.config import Config

def test_config_defaults():
    """Test default configuration values."""
    assert Config.MAX_REQUESTS_PER_SECOND > 0
    assert Config.MAX_RETRIES >= 0
    assert Config.RETRY_DELAY > 0
    assert Config.TIMEOUT > 0
    assert Config.WINDOW_SIZE >= 2
    assert Config.Z_THRESHOLD > 0
    assert Config.VOLATILITY_WINDOW >= 2
    assert Config.ANNUALIZATION_FACTOR > 0
    assert Config.RISK_FREE_RATE >= 0

def test_config_paths():
    """Test configuration paths."""
    assert isinstance(Config.CACHE_DIR, Path)
    assert isinstance(Config.RESULTS_DIR, Path)
    assert isinstance(Config.LOG_DIR, Path)
    
    # Test directory creation
    assert Config.CACHE_DIR.exists()
    assert Config.RESULTS_DIR.exists()
    assert Config.LOG_DIR.exists()

def test_api_urls():
    """Test API URL configuration."""
    # Test production URLs
    Config.USE_TEST_ENV = False
    assert "deribit.com" in Config.get_api_url()
    assert "deribit.com" in Config.get_ws_url()
    
    # Test test environment URLs
    Config.USE_TEST_ENV = True
    assert "test.deribit.com" in Config.get_api_url()
    assert "test.deribit.com" in Config.get_ws_url()

def test_config_validation():
    """Test configuration validation."""
    # Test with valid configuration
    issues = Config.validate()
    assert isinstance(issues, list)
    
    # Test with invalid configuration
    original_max_requests = Config.MAX_REQUESTS_PER_SECOND
    Config.MAX_REQUESTS_PER_SECOND = 0
    issues = Config.validate()
    assert any("MAX_REQUESTS_PER_SECOND" in issue for issue in issues)
    Config.MAX_REQUESTS_PER_SECOND = original_max_requests

def test_logging_setup():
    """Test logging configuration."""
    logger = Config.setup_logging()
    assert logger.name == "crypto_analysis"
    assert logger.level <= logging.INFO  # Should be INFO or lower 