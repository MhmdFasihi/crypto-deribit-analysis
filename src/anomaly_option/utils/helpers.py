"""
Utility functions for the anomaly option analysis system.
"""

import logging
from datetime import datetime, date
from typing import Union, Optional, List, Dict, Any
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def parse_date(date_str: Optional[Union[str, date]]) -> Optional[date]:
    """
    Parse a date string or date object into a date object.
    
    Args:
        date_str: Date string or date object
        
    Returns:
        Parsed date object or None if invalid
    """
    if date_str is None:
        return None
    
    if isinstance(date_str, date):
        return date_str
    
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        logger.error(f"Invalid date format: {date_str}")
        return None

def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series
        
    Returns:
        Returns series
    """
    return prices.pct_change().dropna()

def calculate_volatility(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculate rolling volatility from returns.
    
    Args:
        returns: Returns series
        window: Rolling window size
        
    Returns:
        Volatility series
    """
    return returns.rolling(window=window).std() * np.sqrt(252)

def save_to_csv(data: pd.DataFrame, filepath: Union[str, Path], **kwargs) -> bool:
    """
    Save DataFrame to CSV file.
    
    Args:
        data: DataFrame to save
        filepath: Output file path
        **kwargs: Additional arguments for pd.to_csv
        
    Returns:
        True if successful, False otherwise
    """
    try:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(filepath, **kwargs)
        return True
    except Exception as e:
        logger.error(f"Error saving to CSV: {str(e)}")
        return False

def load_from_csv(filepath: Union[str, Path], **kwargs) -> Optional[pd.DataFrame]:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Input file path
        **kwargs: Additional arguments for pd.read_csv
        
    Returns:
        Loaded DataFrame or None if failed
    """
    try:
        return pd.read_csv(filepath, **kwargs)
    except Exception as e:
        logger.error(f"Error loading from CSV: {str(e)}")
        return None

def format_number(value: float, decimals: int = 2) -> str:
    """
    Format number with specified decimal places.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        
    Returns:
        Formatted string
    """
    return f"{value:,.{decimals}f}"

def safe_divide(a: float, b: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, handling division by zero.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value if division by zero
        
    Returns:
        Result of division or default value
    """
    try:
        return a / b if b != 0 else default
    except Exception:
        return default 