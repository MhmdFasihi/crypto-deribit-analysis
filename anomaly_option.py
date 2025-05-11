"""
Comprehensive Cryptocurrency Volatility and Options Analysis System
Combines anomaly detection with options data analysis for advanced trading insights
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date, timedelta
from typing import Dict, Tuple, List, Union, Any, Optional, Callable
from scipy import stats
from scipy.stats import norm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from tqdm import tqdm
import time
import os
import json
import warnings
import gc
import shutil
import sys
import hmac
import hashlib
import requests
from urllib.parse import urlencode
from visualizer import CryptoVolatilityOptionsVisualizer

warnings.filterwarnings('ignore')

# Import all necessary classes
from crypto_data_fetcher import CryptoDataFetcher
from volatility_analyzer import CryptoVolatilityAnalyzer
from options_analyzer import OptionsAnalyzer
from analysis_system import VolatilityOptionsAnalysisSystem


def datetime_to_timestamp(datetime_obj: datetime) -> int:
    """Convert datetime to millisecond timestamp"""
    return int(datetime.timestamp(datetime_obj) * 1000)


class CryptoDataFetcher:
    """
    Unified data fetcher for both price and options data
    """
    
    def __init__(
        self, 
        symbols: List[str],
        start_date: date,
        end_date: date,
        fetch_options: bool = True,
        options_cache_dir: str = "options_cache",
        max_workers: int = 5,
        max_retries: int = 3,
        timeout: int = 30
    ) -> None:
        """Initialize the data fetcher for both price and options data."""
        self.symbols = [s.upper() for s in symbols]
        self.start_date = start_date
        self.end_date = end_date
        self.fetch_options = fetch_options
        self.price_data = {}
        self.options_data = {}
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Options-specific parameters
        if fetch_options:
            self.options_cache_dir = Path(options_cache_dir)
            self.options_cache_dir.mkdir(exist_ok=True)
            self.max_workers = max_workers
            self.session = requests.Session()
            self._validate_options_inputs()
    
    def _validate_options_inputs(self) -> None:
        """Validate options-related inputs."""
        if not self.symbols:
            raise ValueError("No symbols provided")
            
        for symbol in self.symbols:
            # Strip -USD suffix if present for options data fetching
            currency = symbol.replace('-USD', '')
            if currency not in ['BTC', 'ETH']:
                print(f"Warning: Options data for {currency} may not be available. Only BTC and ETH are fully supported.")
        
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
        
        if (self.end_date - self.start_date).days > 365:
            print("Warning: Date range exceeds 365 days, options data might be limited")
            
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
            
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1")
            
        if self.timeout < 1:
            raise ValueError("timeout must be at least 1 second")
    
    def _get_cache_filename(self, symbol: str) -> Path:
        """Generate cache filename for options data."""
        # Strip -USD suffix if present
        currency = symbol.replace('-USD', '')
        return self.options_cache_dir / f"{currency}_options_{self.start_date}_{self.end_date}.csv"
    
    def _make_api_request(self, url: str, params: Dict, retry_count: int = 0) -> Dict:
        """Make API request with retry logic and error handling."""
        try:
            response = self.session.get(url, params=params, timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if retry_count < self.max_retries:
                time.sleep(2 ** retry_count)  # Exponential backoff
                return self._make_api_request(url, params, retry_count + 1)
            raise Exception(f"API request failed after {self.max_retries} retries: {str(e)}")
    
    def fetch_price_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch cryptocurrency price data from Yahoo Finance."""
        start_date_str = self.start_date.strftime('%Y-%m-%d')
        end_date_str = self.end_date.strftime('%Y-%m-%d')
        
        print(f"Fetching price data for {', '.join(self.symbols)}...")
        
        for symbol in self.symbols:
            try:
                df = yf.download(symbol, start=start_date_str, end=end_date_str)
                if not df.empty:
                    # Validate required columns
                    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                    missing_cols = [col for col in required_cols if col not in df.columns]
                    
                    if missing_cols:
                        print(f"Warning: Missing columns {missing_cols} for {symbol}")
                        # Add missing columns with NaN values
                        for col in missing_cols:
                            df[col] = np.nan
                    
                    # Validate data quality
                    if df['Close'].isnull().any():
                        print(f"Warning: Missing close prices for {symbol}")
                    
                    if (df['Close'] <= 0).any():
                        print(f"Warning: Non-positive close prices found for {symbol}")
                    
                    self.price_data[symbol] = df
                    print(f"Successfully fetched {len(df)} days of price data for {symbol}")
                else:
                    print(f"No price data available for {symbol}")
            except Exception as e:
                print(f"Error fetching price data for {symbol}: {e}")
                self.price_data[symbol] = pd.DataFrame()
        
        return self.price_data
    
    def _fetch_options_chunk(self, currency: str, start_ts: int, end_ts: int) -> List[Dict]:
        """Fetch a chunk of options trading data from Deribit."""
        params = {
            "currency": currency,
            "kind": "option",
            "count": 10000,
            "include_old": True,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts
        }
        
        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'
        
        for attempt in range(5):
            try:
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                if "result" in data and "trades" in data["result"]:
                    return data["result"]["trades"]
                return []
            except Exception as e:
                if attempt == 4:
                    print(f"Failed to get options data for {currency} from {start_ts} to {end_ts}: {e}")
                    return []
                time.sleep(0.5 * (2 ** attempt))
        return []
    
    def _parse_instrument_name(self, name: str) -> Tuple[str, float, str]:
        """Parse Deribit instrument name to extract maturity, strike, and option type."""
        try:
            parts = name.split('-')
            maturity = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            return maturity, strike, option_type
        except Exception as e:
            print(f"Error parsing instrument name {name}: {e}")
            return None, None, None
    
    def _process_options_chunk(self, chunk: List[Dict]) -> pd.DataFrame:
        """Process a chunk of options data into a DataFrame."""
        if not chunk:
            return pd.DataFrame()
            
        df = pd.DataFrame(chunk)
        
        # Parse instrument names
        instrument_details = [self._parse_instrument_name(name) for name in df['instrument_name']]
        df['maturity_date'] = [details[0] for details in instrument_details]
        df['strike_price'] = [details[1] for details in instrument_details]
        df['option_type'] = [details[2] for details in instrument_details]
        
        # Convert timestamp and maturity date
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['maturity_date'] = pd.to_datetime(df['maturity_date'], format='%d%b%y')
        
        # Create derived columns
        df['date'] = df['date_time'].dt.date
        df['time_to_maturity'] = (df['maturity_date'] - df['date_time']).dt.total_seconds() / 86400
        df['moneyness'] = df['index_price'] / df['strike_price']
        df['iv'] = df['iv'] / 100
        df['is_call'] = df['option_type'] == 'call'
        
        # Calculate volume metrics
        df['volume_btc'] = df['price'] * df['contracts']
        df['volume_usd'] = df['volume_btc'] * df['index_price']
        
        return df
    
    def fetch_options_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch options data for all symbols from Deribit."""
        if not self.fetch_options:
            print("Options data fetching is disabled")
            return {}
        
        for symbol in self.symbols:
            # Strip -USD suffix if present
            currency = symbol.replace('-USD', '')
            
            if currency not in ['BTC', 'ETH']:
                print(f"Warning: Options data fetching not supported for {currency}. Skipping.")
                continue
            
            cache_file = self._get_cache_filename(currency)
            
            if cache_file.exists():
                print(f"Loading {currency} options data from cache...")
                try:
                    self.options_data[currency] = pd.read_csv(
                        cache_file, 
                        parse_dates=['date_time', 'maturity_date']
                    )
                    continue
                except Exception as e:
                    print(f"Error loading cached options data for {currency}: {e}")
            
            print(f"Fetching {currency} options data from API...")
            
            # Create date chunks for parallel processing
            date_chunks = []
            current_date = self.start_date
            while current_date < self.end_date:
                next_date = min(current_date + timedelta(days=1), self.end_date)
                date_chunks.append((
                    datetime_to_timestamp(datetime.combine(current_date, datetime.min.time())),
                    datetime_to_timestamp(datetime.combine(next_date, datetime.max.time()))
                ))
                current_date = next_date
                
            all_trades = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(self._fetch_options_chunk, currency, start_ts, end_ts)
                    for start_ts, end_ts in date_chunks
                ]
                
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Fetching {currency} options"):
                    trades = future.result()
                    if trades:
                        all_trades.extend(trades)
                        
            if not all_trades:
                print(f"No options data retrieved for {currency}")
                self.options_data[currency] = pd.DataFrame()
                continue
                
            df = self._process_options_chunk(all_trades)
            df.to_csv(cache_file, index=False)
            self.options_data[currency] = df
            print(f"Successfully fetched and processed {len(df)} options trades for {currency}")
        
        return self.options_data
    
    def fetch_all_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Fetch both price and options data."""
        price_data = self.fetch_price_data()
        options_data = {}
        
        if self.fetch_options:
            options_data = self.fetch_options_data()
        
        return price_data, options_data


class CryptoVolatilityAnalyzer:
    """
    Volatility analysis with anomaly detection capabilities
    """
    
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        window_size: int = 30,
        z_threshold: float = 3.0,
        volatility_window: int = 21,
        annualization_factor: int = 365,
        chunk_size: int = 1000  # Number of rows to process at once
    ) -> None:
        """Initialize volatility analyzer with price data."""
        self.price_data = price_data
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.volatility_window = volatility_window
        self.annualization_factor = annualization_factor
        self.chunk_size = chunk_size
        self.results = {}
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.price_data:
            raise ValueError("No price data provided")
            
        if self.window_size < 2:
            raise ValueError("window_size must be at least 2")
            
        if self.z_threshold <= 0:
            raise ValueError("z_threshold must be positive")
            
        if self.volatility_window < 2:
            raise ValueError("volatility_window must be at least 2")
            
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive")
            
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
    
    def _process_chunk(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        """Process a chunk of data for volatility analysis."""
        chunk = df.iloc[start_idx:end_idx].copy()
        
        # Calculate returns
        chunk['Returns'] = chunk['Close'].pct_change()
        chunk['Log_Returns'] = np.log(chunk['Close'] / chunk['Close'].shift(1))
        
        # Calculate realized volatility
        chunk['RV_Close'] = self.calculate_realized_volatility(chunk['Returns'])
        chunk['RV_Parkinson'] = self.calculate_parkinson_volatility(chunk)
        chunk['RV_Garman_Klass'] = self.calculate_garman_klass_volatility(chunk)
        
        # Calculate composite realized volatility
        volatility_cols = ['RV_Close', 'RV_Parkinson', 'RV_Garman_Klass']
        chunk['RV_Composite'] = chunk[volatility_cols].mean(axis=1, skipna=True)
        
        # Calculate Z-scores
        price_rolling_mean, price_rolling_std, price_z_score = self.calculate_z_score(chunk['Close'])
        return_rolling_mean, return_rolling_std, return_z_score = self.calculate_z_score(chunk['Returns'])
        vol_rolling_mean, vol_rolling_std, vol_z_score = self.calculate_z_score(chunk['RV_Composite'])
        
        # Add to DataFrame
        chunk['Price_Rolling_Mean'] = price_rolling_mean
        chunk['Price_Rolling_Std'] = price_rolling_std
        chunk['Price_Z_Score'] = price_z_score
        chunk['Return_Z_Score'] = return_z_score
        chunk['Volatility_Z_Score'] = vol_z_score
        
        # Identify anomalies
        chunk['Price_Anomaly'] = (np.abs(price_z_score) > self.z_threshold).astype(int)
        chunk['Return_Anomaly'] = (np.abs(return_z_score) > self.z_threshold).astype(int)
        chunk['Volatility_Anomaly'] = (np.abs(vol_z_score) > self.z_threshold).astype(int)
        chunk['Combined_Anomaly'] = ((chunk['Price_Anomaly'] == 1) | 
                                   (chunk['Return_Anomaly'] == 1) | 
                                   (chunk['Volatility_Anomaly'] == 1)).astype(int)
        
        # Categorize anomalies
        chunk['Anomaly_Direction'] = np.where(
            chunk['Combined_Anomaly'] == 1,
            np.where(chunk['Returns'] > 0, 1, -1),
            0
        )
        
        return chunk
    
    def calculate_z_score(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate rolling Z-score for a data series."""
        rolling_mean = data.rolling(window=self.window_size).mean()
        rolling_std = data.rolling(window=self.window_size).std()
        z_score = (data - rolling_mean) / rolling_std
        return rolling_mean, rolling_std, z_score
    
    def calculate_realized_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate realized volatility using rolling standard deviation."""
        return returns.rolling(window=self.volatility_window).std() * np.sqrt(self.annualization_factor)
    
    def calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility estimator based on high-low range."""
        if 'High' not in df.columns or 'Low' not in df.columns:
            return pd.Series(index=df.index)
        
        log_hl = np.log(df['High'] / df['Low'])
        estimator = log_hl ** 2 / (4 * np.log(2))
        return np.sqrt(estimator.rolling(window=self.volatility_window).mean() * self.annualization_factor)
    
    def calculate_garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility estimator."""
        if 'Open' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns or 'Close' not in df.columns:
            return pd.Series(index=df.index)
        
        log_hl = np.log(df['High'] / df['Low']) ** 2
        log_co = np.log(df['Close'] / df['Open']) ** 2
        estimator = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        return np.sqrt(estimator.rolling(window=self.volatility_window).mean() * self.annualization_factor)
    
    def analyze_volatility(self, symbol: str) -> pd.DataFrame:
        """Analyze volatility and detect anomalies for a given symbol."""
        if symbol not in self.price_data:
            raise ValueError(f"Price data not available for {symbol}")
        
        # Extract price data
        df = self.price_data[symbol].copy()
        
        if df.empty:
            print(f"Empty price data for {symbol}")
            return pd.DataFrame()
        
        # Process data in chunks
        chunks = []
        for start_idx in range(0, len(df), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(df))
            chunk = self._process_chunk(df, start_idx, end_idx)
            chunks.append(chunk)
            
            # Clear memory
            del chunk
        
        # Combine chunks
        result = pd.concat(chunks, axis=0)
        
        # Clear memory
        del chunks
        
        # Calculate other market indicators
        result['VWAP'] = (result['Close'] * result['Volume']).cumsum() / result['Volume'].cumsum()
        result['Momentum_5D'] = result['Close'].pct_change(periods=5)
        result['Momentum_20D'] = result['Close'].pct_change(periods=20)
        
        # Remove NaN values
        result = result.dropna()
        
        # Store results
        self.results[symbol] = result
        
        return result
    
    def analyze_all_symbols(self) -> Dict[str, pd.DataFrame]:
        """Analyze volatility for all symbols."""
        for symbol in self.price_data.keys():
            try:
                self.analyze_volatility(symbol)
                print(f"Completed volatility analysis for {symbol}")
            except Exception as e:
                print(f"Error analyzing volatility for {symbol}: {e}")
                self.results[symbol] = pd.DataFrame()
        
        return self.results
    
    def cleanup(self) -> None:
        """Clean up resources and memory."""
        self.price_data.clear()
        self.results.clear()
        gc.collect()  # Force garbage collection


class OptionsAnalyzer:
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        risk_free_rate: float = 0.02,
        min_volume: int = 100,
        min_open_interest: int = 50,
        min_days_to_expiry: int = 1,
        max_days_to_expiry: int = 365,
        min_strike_distance: float = 0.05,  # 5% from current price
        max_strike_distance: float = 0.50,  # 50% from current price
        min_implied_volatility: float = 0.05,  # 5% minimum IV
        max_implied_volatility: float = 5.00,  # 500% maximum IV
        min_option_price: float = 0.0001,  # Minimum option price
        max_option_price: float = 1000000.0,  # Maximum option price
        numerical_precision: int = 8  # Number of decimal places for calculations
    ) -> None:
        """Initialize options analyzer with price data."""
        self.price_data = price_data
        self.risk_free_rate = risk_free_rate
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry
        self.min_strike_distance = min_strike_distance
        self.max_strike_distance = max_strike_distance
        self.min_implied_volatility = min_implied_volatility
        self.max_implied_volatility = max_implied_volatility
        self.min_option_price = min_option_price
        self.max_option_price = max_option_price
        self.numerical_precision = numerical_precision
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.price_data:
            raise ValueError("No price data provided")
            
        if self.risk_free_rate < 0:
            raise ValueError("Risk-free rate must be non-negative")
            
        if self.min_volume < 0:
            raise ValueError("Minimum volume must be non-negative")
            
        if self.min_open_interest < 0:
            raise ValueError("Minimum open interest must be non-negative")
            
        if self.min_days_to_expiry < 0:
            raise ValueError("Minimum days to expiry must be non-negative")
            
        if self.max_days_to_expiry <= self.min_days_to_expiry:
            raise ValueError("Maximum days to expiry must be greater than minimum days to expiry")
            
        if self.min_strike_distance < 0 or self.min_strike_distance >= self.max_strike_distance:
            raise ValueError("Invalid strike distance range")
            
        if self.min_implied_volatility < 0 or self.min_implied_volatility >= self.max_implied_volatility:
            raise ValueError("Invalid implied volatility range")
            
        if self.min_option_price < 0 or self.min_option_price >= self.max_option_price:
            raise ValueError("Invalid option price range")
            
        if self.numerical_precision < 0:
            raise ValueError("Numerical precision must be non-negative")
    
    def _validate_option_data(self, option_data: pd.DataFrame) -> bool:
        """Validate option data for analysis."""
        if option_data.empty:
            return False
            
        required_columns = ['strike', 'expiry', 'type', 'price', 'volume', 'open_interest']
        if not all(col in option_data.columns for col in required_columns):
            return False
            
        # Check for invalid values
        if (option_data['price'] <= 0).any() or (option_data['volume'] < 0).any() or (option_data['open_interest'] < 0).any():
            return False
            
        # Check for missing values
        if option_data[required_columns].isnull().any().any():
            return False
            
        return True
    
    def _calculate_black_scholes(
        self,
        S: float,  # Current price
        K: float,  # Strike price
        T: float,  # Time to expiry (in years)
        r: float,  # Risk-free rate
        sigma: float,  # Volatility
        option_type: str  # 'call' or 'put'
    ) -> Tuple[float, Dict[str, float]]:
        """Calculate option price and Greeks using Black-Scholes model with improved accuracy."""
        try:
            # Input validation
            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
                raise ValueError("Invalid input parameters")
            
            # Calculate d1 and d2 with improved numerical stability
            d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate option price
            if option_type.lower() == 'call':
                price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
            # Calculate Greeks
            greeks = {
                'delta': norm.cdf(d1) if option_type.lower() == 'call' else norm.cdf(d1) - 1,
                'gamma': norm.pdf(d1) / (S * sigma * np.sqrt(T)),
                'vega': S * np.sqrt(T) * norm.pdf(d1) / 100,  # Divided by 100 for percentage
                'theta': (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2) if option_type.lower() == 'call' else
                        (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
            }
            
            # Round results to specified precision
            price = round(price, self.numerical_precision)
            greeks = {k: round(v, self.numerical_precision) for k, v in greeks.items()}
            
            return price, greeks
            
        except Exception as e:
            print(f"Error in Black-Scholes calculation: {e}")
            return None, None
    
    def _calculate_implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str,
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method with improved accuracy."""
        try:
            # Input validation
            if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
                raise ValueError("Invalid input parameters")
            
            # Initial guess for volatility
            sigma = 0.3  # 30% initial guess
            
            for i in range(max_iterations):
                # Calculate option price and vega
                price, greeks = self._calculate_black_scholes(S, K, T, r, sigma, option_type)
                
                if price is None or greeks is None:
                    raise ValueError("Failed to calculate option price or Greeks")
                
                # Calculate price difference
                diff = price - market_price
                
                # Check convergence
                if abs(diff) < tolerance:
                    return sigma
                
                # Update volatility using Newton-Raphson
                vega = greeks['vega'] * 100  # Convert back to decimal
                if abs(vega) < tolerance:
                    raise ValueError("Vega too small for numerical stability")
                
                sigma = sigma - diff / vega
                
                # Ensure volatility is within valid range
                sigma = max(self.min_implied_volatility, min(sigma, self.max_implied_volatility))
            
            raise ValueError("Failed to converge to implied volatility")
            
        except Exception as e:
            print(f"Error calculating implied volatility: {e}")
            return None
    
    def analyze_options(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Analyze options data for a given symbol."""
        try:
            if symbol not in self.price_data:
                raise ValueError(f"Price data not available for {symbol}")
            
            # Get price data
            price_data = self.price_data[symbol]
            
            if price_data.empty:
                raise ValueError(f"Empty price data for {symbol}")
            
            # Fetch options data
            options_data = self._fetch_options_data(symbol, start_date, end_date)
            
            if not self._validate_option_data(options_data):
                raise ValueError(f"Invalid options data for {symbol}")
            
            # Filter options data
            filtered_data = self._filter_options_data(options_data, price_data)
            
            if filtered_data.empty:
                raise ValueError(f"No valid options data after filtering for {symbol}")
            
            # Calculate metrics
            volume_analysis = self._analyze_volumes(filtered_data)
            iv_analysis = self._analyze_implied_volatility(filtered_data, price_data)
            greeks_analysis = self._analyze_greeks(filtered_data, price_data)
            
            return {
                'volume_analysis': volume_analysis,
                'iv_analysis': iv_analysis,
                'greeks_analysis': greeks_analysis
            }
            
        except Exception as e:
            print(f"Error analyzing options for {symbol}: {e}")
            return None
    
    def _filter_options_data(self, options_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Filter options data based on criteria."""
        try:
            # Calculate days to expiry
            options_data['days_to_expiry'] = (pd.to_datetime(options_data['expiry']) - pd.to_datetime(price_data.index[0])).dt.days
            
            # Filter based on criteria
            mask = (
                (options_data['volume'] >= self.min_volume) &
                (options_data['open_interest'] >= self.min_open_interest) &
                (options_data['days_to_expiry'] >= self.min_days_to_expiry) &
                (options_data['days_to_expiry'] <= self.max_days_to_expiry) &
                (options_data['price'] >= self.min_option_price) &
                (options_data['price'] <= self.max_option_price)
            )
            
            # Filter strike prices
            current_price = price_data['Close'].iloc[-1]
            strike_range = (
                (options_data['strike'] >= current_price * (1 - self.max_strike_distance)) &
                (options_data['strike'] <= current_price * (1 + self.max_strike_distance))
            )
            mask = mask & strike_range
            
            return options_data[mask].copy()
            
        except Exception as e:
            print(f"Error filtering options data: {e}")
            return pd.DataFrame()
    
    def _analyze_volumes(self, options_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze options trading volumes."""
        try:
            # Group by option type
            call_data = options_data[options_data['type'] == 'call']
            put_data = options_data[options_data['type'] == 'put']
            
            # Calculate volume metrics
            volume_analysis = {
                'total_volume': int(options_data['volume'].sum()),
                'call_volume': int(call_data['volume'].sum()),
                'put_volume': int(put_data['volume'].sum()),
                'put_call_ratio': float(put_data['volume'].sum() / call_data['volume'].sum() if call_data['volume'].sum() > 0 else 0),
                'volume_by_expiry': options_data.groupby('expiry')['volume'].sum().to_dict(),
                'volume_by_strike': options_data.groupby('strike')['volume'].sum().to_dict()
            }
            
            return volume_analysis
            
        except Exception as e:
            print(f"Error analyzing volumes: {e}")
            return {}
    
    def _analyze_implied_volatility(self, options_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze implied volatility patterns."""
        try:
            # Calculate implied volatility for each option
            current_price = price_data['Close'].iloc[-1]
            iv_data = []
            
            for _, row in options_data.iterrows():
                T = row['days_to_expiry'] / 365.0
                iv = self._calculate_implied_volatility(
                    row['price'],
                    current_price,
                    row['strike'],
                    T,
                    self.risk_free_rate,
                    row['type']
                )
                
                if iv is not None:
                    iv_data.append({
                        'strike': row['strike'],
                        'expiry': row['expiry'],
                        'type': row['type'],
                        'iv': iv
                    })
            
            iv_df = pd.DataFrame(iv_data)
            
            if iv_df.empty:
                raise ValueError("No valid implied volatility data")
            
            # Calculate IV metrics
            iv_analysis = {
                'mean_iv': float(iv_df['iv'].mean()),
                'std_iv': float(iv_df['iv'].std()),
                'iv_by_expiry': iv_df.groupby('expiry')['iv'].mean().to_dict(),
                'iv_by_strike': iv_df.groupby('strike')['iv'].mean().to_dict(),
                'iv_skew': float(iv_df[iv_df['type'] == 'put']['iv'].mean() - iv_df[iv_df['type'] == 'call']['iv'].mean())
            }
            
            return iv_analysis
            
        except Exception as e:
            print(f"Error analyzing implied volatility: {e}")
            return {}
    
    def _analyze_greeks(self, options_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze option Greeks."""
        try:
            # Calculate Greeks for each option
            current_price = price_data['Close'].iloc[-1]
            greeks_data = []
            
            for _, row in options_data.iterrows():
                T = row['days_to_expiry'] / 365.0
                iv = self._calculate_implied_volatility(
                    row['price'],
                    current_price,
                    row['strike'],
                    T,
                    self.risk_free_rate,
                    row['type']
                )
                
                if iv is not None:
                    _, greeks = self._calculate_black_scholes(
                        current_price,
                        row['strike'],
                        T,
                        self.risk_free_rate,
                        iv,
                        row['type']
                    )
                    
                    if greeks is not None:
                        greeks_data.append({
                            'strike': row['strike'],
                            'expiry': row['expiry'],
                            'type': row['type'],
                            **greeks
                        })
            
            greeks_df = pd.DataFrame(greeks_data)
            
            if greeks_df.empty:
                raise ValueError("No valid Greeks data")
            
            # Calculate Greeks metrics
            greeks_analysis = {
                'mean_delta': float(greeks_df['delta'].mean()),
                'mean_gamma': float(greeks_df['gamma'].mean()),
                'mean_vega': float(greeks_df['vega'].mean()),
                'mean_theta': float(greeks_df['theta'].mean()),
                'greeks_by_expiry': {
                    'delta': greeks_df.groupby('expiry')['delta'].mean().to_dict(),
                    'gamma': greeks_df.groupby('expiry')['gamma'].mean().to_dict(),
                    'vega': greeks_df.groupby('expiry')['vega'].mean().to_dict(),
                    'theta': greeks_df.groupby('expiry')['theta'].mean().to_dict()
                }
            }
            
            return greeks_analysis
            
        except Exception as e:
            print(f"Error analyzing Greeks: {e}")
            return {}


class CryptoOptionsAnalysis:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        max_workers: int = 4,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 5
    ) -> None:
        """Initialize the options analysis system."""
        self.data_fetcher = CryptoDataFetcher(api_key, api_secret, max_retries, timeout)
        self.volatility_analyzer = None
        self.options_analyzer = None
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._validate_credentials()
    
    def _validate_credentials(self) -> None:
        """Validate API credentials."""
        if not self.data_fetcher.api_key or not self.data_fetcher.api_secret:
            raise ValueError("API credentials are required")
        
        try:
            # Test API connection
            self.data_fetcher._make_api_request("GET", "/api/v2/public/test")
        except Exception as e:
            raise ValueError(f"Invalid API credentials: {e}")
    
    def _execute_with_timeout(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a function with timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except TimeoutError:
                print(f"Operation timed out after {self.timeout} seconds")
                return None
            except Exception as e:
                print(f"Error executing operation: {e}")
                return None
    
    def _retry_operation(self, func: Callable, *args, **kwargs) -> Any:
        """Retry an operation with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return self._execute_with_timeout(func, *args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Operation failed after {self.max_retries} attempts: {e}")
                    return None
                delay = self.retry_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed, retrying in {delay} seconds...")
                time.sleep(delay)
    
    def analyze_market(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, Any]:
        """Analyze market data for given symbols."""
        try:
            # Fetch price data with retry
            price_data = self._retry_operation(
                self.data_fetcher.fetch_price_data,
                symbols=symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            if not price_data:
                raise ValueError("Failed to fetch price data")
            
            # Initialize analyzers
            self.volatility_analyzer = CryptoVolatilityAnalyzer(price_data)
            self.options_analyzer = OptionsAnalyzer(price_data)
            
            # Analyze volatility with timeout
            volatility_results = self._execute_with_timeout(
                self.volatility_analyzer.analyze_all_symbols
            )
            
            if not volatility_results:
                raise ValueError("Failed to analyze volatility")
            
            # Analyze options with timeout
            options_results = {}
            for symbol in symbols:
                result = self._execute_with_timeout(
                    self.options_analyzer.analyze_options,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                if result:
                    options_results[symbol] = result
            
            return {
                'volatility': volatility_results,
                'options': options_results
            }
            
        except Exception as e:
            print(f"Error analyzing market: {e}")
            return {}
        finally:
            # Cleanup
            if self.volatility_analyzer:
                self.volatility_analyzer.cleanup()
    
    def generate_report(self, analysis_results: Dict[str, Any], output_path: str) -> bool:
        """Generate analysis report."""
        try:
            if not analysis_results:
                raise ValueError("No analysis results to report")
            
            # Create report directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Generate report with timeout
            report_data = self._execute_with_timeout(
                self._generate_report_content,
                analysis_results
            )
            
            if not report_data:
                raise ValueError("Failed to generate report content")
            
            # Write report with retry
            return self._retry_operation(
                self._write_report,
                report_data=report_data,
                output_path=output_path
            )
            
        except Exception as e:
            print(f"Error generating report: {e}")
            return False
    
    def _generate_report_content(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report content."""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'volatility_analysis': {},
                'options_analysis': {},
                'summary': {}
            }
            
            # Process volatility results
            for symbol, data in analysis_results['volatility'].items():
                if not data.empty:
                    report_data['volatility_analysis'][symbol] = {
                        'anomalies': {
                            'price': int(data['Price_Anomaly'].sum()),
                            'return': int(data['Return_Anomaly'].sum()),
                            'volatility': int(data['Volatility_Anomaly'].sum())
                        },
                        'volatility_metrics': {
                            'mean': float(data['RV_Composite'].mean()),
                            'std': float(data['RV_Composite'].std()),
                            'max': float(data['RV_Composite'].max()),
                            'min': float(data['RV_Composite'].min())
                        }
                    }
            
            # Process options results
            for symbol, data in analysis_results['options'].items():
                if data:
                    report_data['options_analysis'][symbol] = {
                        'volume_analysis': data.get('volume_analysis', {}),
                        'iv_analysis': data.get('iv_analysis', {}),
                        'greeks_analysis': data.get('greeks_analysis', {})
                    }
            
            # Generate summary
            report_data['summary'] = self._generate_summary(report_data)
            
            return report_data
            
        except Exception as e:
            print(f"Error generating report content: {e}")
            return None
    
    def _write_report(self, report_data: Dict[str, Any], output_path: str) -> bool:
        """Write report to file."""
        try:
            # Create backup of existing report if it exists
            if os.path.exists(output_path):
                backup_path = f"{output_path}.bak"
                shutil.copy2(output_path, backup_path)
            
            # Write new report
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=4)
            
            return True
            
        except Exception as e:
            print(f"Error writing report: {e}")
            return False
    
    def _generate_summary(self, report_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of analysis results."""
        try:
            summary = {
                'total_symbols': len(report_data['volatility_analysis']),
                'total_anomalies': 0,
                'high_volatility_symbols': [],
                'high_volume_symbols': []
            }
            
            # Count total anomalies
            for symbol_data in report_data['volatility_analysis'].values():
                summary['total_anomalies'] += sum(symbol_data['anomalies'].values())
            
            # Identify high volatility symbols
            for symbol, data in report_data['volatility_analysis'].items():
                if data['volatility_metrics']['mean'] > data['volatility_metrics']['std']:
                    summary['high_volatility_symbols'].append(symbol)
            
            # Identify high volume symbols
            for symbol, data in report_data['options_analysis'].items():
                if data['volume_analysis'].get('total_volume', 0) > 1000000:  # Example threshold
                    summary['high_volume_symbols'].append(symbol)
            
            return summary
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            return {}


def main():
    """Main function to run the analysis."""
    # Set up analysis parameters
    symbols = ['BTC', 'ETH']  # Add more symbols as needed
    end_date = date.today()
    start_date = end_date - timedelta(days=60)
    next_window = 30
    
    print(f"Analysis Period: {start_date} to {end_date}")
    print(f"Next Analysis Window: {next_window} days from {end_date}")
    
    try:
        # Initialize the analysis system
        # Note: Replace with your actual API credentials
        api_key = "YOUR_API_KEY"
        api_secret = "YOUR_API_SECRET"
        
        analysis_system = VolatilityOptionsAnalysisSystem(
            api_key=api_key,
            api_secret=api_secret,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            next_window=next_window
        )
        
        # Run the analysis
        volatility_results, options_results, report_path = analysis_system.run_analysis()
        
        if report_path:
            print(f"\nAnalysis completed successfully!")
            print(f"Report generated: {report_path}")
        else:
            print("\nAnalysis completed with errors. Check the output for details.")
        
    except Exception as e:
        print(f"Error in main function: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()