"""
Module for fetching cryptocurrency price and options data from various sources.
Handles API requests, rate limiting, caching, and data processing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import yfinance as yf
import requests
from pathlib import Path
import json
import time
import hmac
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import websockets
from urllib.parse import urlencode
import logging
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm
import traceback

from config import Config, logger

class APIError(Exception):
    """Exception raised for API errors."""
    pass

class AuthenticationError(APIError):
    """Exception raised for authentication errors."""
    pass

class RateLimitError(APIError):
    """Exception raised for rate limit errors."""
    pass

class NetworkError(APIError):
    """Exception raised for network errors."""
    pass

class DataProcessingError(Exception):
    """Exception raised for data processing errors."""
    pass


class CryptoDataFetcher:
    """
    Unified data fetcher for both price and options data
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        max_retries: Optional[int] = None,
        timeout: Optional[int] = None,
        use_test_env: Optional[bool] = None
    ) -> None:
        """Initialize data fetcher with the specified parameters or from config."""
        self.api_key = api_key or Config.API_KEY
        self.api_secret = api_secret or Config.API_SECRET
        self.max_retries = max_retries or Config.MAX_RETRIES
        self.timeout = timeout or Config.TIMEOUT
        
        # Force test environment if no credentials provided
        if not self.api_key or not self.api_secret:
            logger.warning("No API credentials provided, forcing test environment")
            self.use_test_env = True
        else:
            self.use_test_env = use_test_env if use_test_env is not None else Config.USE_TEST_ENV
        
        # Initialize session
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
        
        # Set up cache directory
        self.options_cache_dir = Config.CACHE_DIR
        self.options_cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up base URLs based on environment
        self.base_url = Config.get_api_url() if not self.use_test_env else Config.API_TEST_URL
        self.ws_url = Config.get_ws_url() if not self.use_test_env else Config.API_TEST_WS_URL
        
        logger.info(f"Initialized CryptoDataFetcher with {'test' if self.use_test_env else 'production'} environment")

    def _get_cache_filename(self, symbol: str, start_date: date, end_date: date) -> Path:
        """Generate cache filename for options data."""
        # Strip -USD suffix if present
        currency = symbol.replace('-USD', '')
        return self.options_cache_dir / f"{currency}_options_{start_date}_{end_date}.csv"
    
    def _generate_signature(self, method: str, uri: str, params: Dict = None) -> Dict:
        """Generate signature for authenticated Deribit API requests."""
        if not self.api_key or not self.api_secret:
            raise AuthenticationError("API key and secret are required for authenticated requests")
        
        params = params or {}
        nonce = str(int(time.time() * 1000))
        timestamp = str(int(time.time() * 1000))
        
        # Prepare the message string
        if method.upper() == "GET":
            params_str = ''
            if params:
                params_str = '?' + urlencode(params)
            message = timestamp + "\n" + nonce + "\n" + method.upper() + "\n" + uri + params_str + "\n"
        else:
            body = json.dumps(params) if params else ''
            message = timestamp + "\n" + nonce + "\n" + method.upper() + "\n" + uri + "\n" + body + "\n"
        
        # Create signature using HMAC-SHA256
        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()
        
        # Return authentication headers
        return {
            'Authorization': f'deri-hmac-sha256 id={self.api_key},ts={timestamp},sig={signature},nonce={nonce}'
        }

    @sleep_and_retry
    @limits(calls=Config.MAX_REQUESTS_PER_SECOND, period=1)
    def _make_api_request(
        self, 
        method: str, 
        endpoint: str, 
        params: Dict = None, 
        data: Dict = None,
        authenticated: bool = False
    ) -> Dict:
        """
        Make an API request with rate limiting and error handling.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (without base URL)
            params: Query parameters
            data: Request body for POST/PUT
            authenticated: Whether to authenticate the request
            
        Returns:
            API response as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        headers = {}
        
        # Add authentication headers if needed
        if authenticated:
            uri = endpoint
            headers.update(self._generate_signature(method, uri, params if method.upper() == "GET" else data))
        
        for attempt in range(self.max_retries + 1):
            try:
                if method.upper() == "GET":
                    response = self.session.get(
                        url, 
                        params=params, 
                        headers=headers, 
                        timeout=self.timeout
                    )
                elif method.upper() == "POST":
                    response = self.session.post(
                        url, 
                        params=params, 
                        json=data, 
                        headers=headers, 
                        timeout=self.timeout
                    )
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                result = response.json()
                
                # Check for API errors
                if "error" in result and result["error"]:
                    error_code = result["error"].get("code", "unknown")
                    error_message = result["error"].get("message", "Unknown API error")
                    
                    if error_code in [10009, 10010, 10011]:  # Authentication errors
                        raise AuthenticationError(f"Authentication error ({error_code}): {error_message}")
                    elif error_code in [10008, 10029, 10043]:  # Rate limit errors
                        raise RateLimitError(f"Rate limit error ({error_code}): {error_message}")
                    else:
                        raise APIError(f"API error ({error_code}): {error_message}")
                
                return result
                
            except RateLimitError as e:
                logger.warning(f"Rate limit error: {e}. Retrying in {2 ** attempt} seconds...")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
                    
            except AuthenticationError:
                # Don't retry authentication errors
                raise
                
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:  # Too Many Requests
                    logger.warning(f"Rate limit exceeded. Retrying in {2 ** attempt} seconds...")
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise RateLimitError(f"Rate limit exceeded after {self.max_retries} retries")
                elif e.response.status_code in [401, 403]:  # Unauthorized, Forbidden
                    raise AuthenticationError(f"Authentication error: {str(e)}")
                else:
                    logger.warning(f"HTTP error: {e}. Retrying in {2 ** attempt} seconds...")
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                    else:
                        raise APIError(f"HTTP error after {self.max_retries} retries: {str(e)}")
                        
            except requests.exceptions.RequestException as e:
                logger.warning(f"Network error: {e}. Retrying in {2 ** attempt} seconds...")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise NetworkError(f"Network error after {self.max_retries} retries: {str(e)}")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON response. Retrying in {2 ** attempt} seconds...")
                if attempt < self.max_retries:
                    time.sleep(2 ** attempt)
                else:
                    raise APIError(f"Invalid JSON response after {self.max_retries} retries: {str(e)}")
    
    def fetch_price_data(
        self,
        symbols: List[str],
        start_date: Union[str, date],
        end_date: Union[str, date]
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical price data for given symbols using yfinance.
        
        Args:
            symbols: List of symbols to fetch (e.g., ['BTC-USD', 'ETH-USD'])
            start_date: Start date as string (YYYY-MM-DD) or date object
            end_date: End date as string (YYYY-MM-DD) or date object
            
        Returns:
            Dictionary mapping symbols to price DataFrames
        """
        # Convert dates to strings if they are date objects
        if isinstance(start_date, date):
            start_date = start_date.strftime('%Y-%m-%d')
        if isinstance(end_date, date):
            end_date = end_date.strftime('%Y-%m-%d')
            
        logger.info(f"Fetching price data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        price_data = {}
        
        for symbol in symbols:
            try:
                logger.debug(f"Fetching data for {symbol}")
                
                # Ensure symbol has correct format for yfinance
                yf_symbol = symbol
                if not symbol.endswith('-USD') and not symbol.endswith('-USDT'):
                    yf_symbol = f"{symbol}-USD"
                
                # Fetch data using yfinance
                df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
                
                if not df.empty:
                    # Ensure all required columns are present
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        if col not in df.columns:
                            logger.warning(f"Missing column {col} for {symbol}, adding NaN values")
                            df[col] = np.nan
                    
                    # Filter to required columns and check data quality - avoiding any direct series comparison
                    df = df[required_columns]
                    
                    # Check for missing data safely
                    missing_check = df.isnull()
                    missing_rows = missing_check.any(axis=1)
                    missing_count = missing_rows.sum()
                    
                    if missing_count > 0:
                        missing_pct = (missing_count / len(df)) * 100
                        logger.warning(f"{symbol}: {missing_count} rows ({missing_pct:.2f}%) have missing data")
                    
                    # Check for suspicious values safely
                    if not df.empty and 'Close' in df.columns:
                        # Use .any() with parentheses to avoid ambiguity
                        non_positive_check = df['Close'] <= 0
                        if non_positive_check.any():
                            logger.warning(f"{symbol}: Contains non-positive close prices")
                    
                    # Log sample of the data for debugging
                    logger.debug(f"Sample of data for {symbol}:")
                    logger.debug(f"Shape: {df.shape}")
                    logger.debug(f"Columns: {df.columns.tolist()}")
                    if not df.empty:
                        logger.debug(f"First row: {df.iloc[0].to_dict() if len(df) > 0 else 'No rows'}")
                    
                    # Store the data
                    price_data[symbol] = df
                    logger.info(f"Successfully fetched {len(df)} days of price data for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")
                    price_data[symbol] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
                
            except Exception as e:
                logger.error(f"Error fetching price data for {symbol}: {str(e)}")
                logger.debug(traceback.format_exc())
                price_data[symbol] = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
        
        return price_data
    
    def _parse_instrument_name(self, name: str) -> Tuple[str, float, str]:
        """
        Parse Deribit instrument name to extract maturity, strike, and option type.
        
        Args:
            name: Instrument name (e.g., 'BTC-24JUN22-30000-C')
            
        Returns:
            Tuple of (maturity_date, strike_price, option_type)
        """
        try:
            parts = name.split('-')
            
            currency = parts[0]
            maturity = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            
            return maturity, strike, option_type
            
        except Exception as e:
            logger.error(f"Error parsing instrument name {name}: {e}")
            raise DataProcessingError(f"Failed to parse instrument name {name}: {str(e)}")
    
    def get_instruments(self, currency: str) -> List[Dict]:
        """
        Get available instruments for a currency.
        
        Args:
            currency: Currency code (e.g., 'BTC', 'ETH')
            
        Returns:
            List of instrument details
        """
        endpoint = "/api/v2/public/get_instruments"
        params = {
            "currency": currency,
            "kind": "option",
            "expired": False
        }
        
        try:
            response = self._make_api_request("GET", endpoint, params=params)
            
            if "result" in response:
                logger.info(f"Retrieved {len(response['result'])} instruments for {currency}")
                return response["result"]
            else:
                logger.warning(f"Unexpected response format for {currency} instruments")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching instruments for {currency}: {str(e)}")
            return []
    
    def get_ticker(self, instrument_name: str) -> Dict:
        """
        Get ticker information for an instrument.
        
        Args:
            instrument_name: Name of the instrument
            
        Returns:
            Ticker information
        """
        endpoint = "/api/v2/public/ticker"
        params = {"instrument_name": instrument_name}
        
        try:
            response = self._make_api_request("GET", endpoint, params=params)
            
            if "result" in response:
                return response["result"]
            else:
                logger.warning(f"Unexpected response format for ticker {instrument_name}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching ticker for {instrument_name}: {str(e)}")
            return {}
    
    def fetch_options_data(
        self, 
        symbols: List[str], 
        start_date: Union[str, date], 
        end_date: Union[str, date],
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch options data for given symbols.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary mapping symbols to options DataFrames
        """
        # Convert dates to date objects if they are strings
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            
        logger.info(f"Fetching options data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        options_data = {}
        
        for symbol in symbols:
            try:
                # Strip -USD suffix if present
                currency = symbol.replace('-USD', '')
                
                # Check supported currencies
                if currency not in ['BTC', 'ETH', 'SOL']:
                    logger.warning(f"Options data fetching not fully supported for {currency}")
                
                # Check cache
                cache_file = self._get_cache_filename(currency, start_date, end_date)
                
                if use_cache and cache_file.exists():
                    logger.info(f"Loading {currency} options data from cache: {cache_file}")
                    try:
                        options_data[symbol] = pd.read_csv(
                            cache_file, 
                            parse_dates=['expiration']
                        )
                        continue
                    except Exception as e:
                        logger.warning(f"Error loading cached options data for {currency}: {e}")
                
                logger.info(f"Fetching options chain for {currency} from Deribit")
                
                # Fetch available instruments
                instruments = self.get_instruments(currency)
                if not instruments:
                    logger.warning(f"No instruments found for {currency}")
                    options_data[symbol] = pd.DataFrame()
                    continue
                
                logger.info(f"Found {len(instruments)} instruments for {currency}")
                
                # Filter instruments by expiration date
                filtered_instruments = []
                for inst in instruments:
                    try:
                        maturity, strike, option_type = self._parse_instrument_name(inst['instrument_name'])
                        expiry_date = datetime.strptime(maturity, '%d%b%y').date()
                        
                        # Log the first few expiry dates for debugging
                        if len(filtered_instruments) < 10:
                            logger.debug(f"Instrument: {inst['instrument_name']}, Expiry: {expiry_date}, Start: {start_date}, End: {end_date}")
                        
                        # Check if expiry is within or shortly after date range (add 7 days buffer to end_date)
                        # This ensures we get options expiring just after our analysis period
                        end_date_with_buffer = end_date + timedelta(days=7)
                        if start_date <= expiry_date <= end_date_with_buffer:
                            filtered_instruments.append(inst)
                    except Exception as e:
                        logger.warning(f"Error processing instrument {inst.get('instrument_name', 'unknown')}: {e}")
                
                # Log first 10 expiry dates of filtered instruments
                if filtered_instruments:
                    expiry_dates = []
                    for inst in filtered_instruments[:10]:
                        try:
                            maturity, _, _ = self._parse_instrument_name(inst['instrument_name'])
                            expiry_date = datetime.strptime(maturity, '%d%b%y').date().strftime('%Y-%m-%d')
                            expiry_dates.append(expiry_date)
                        except:
                            expiry_dates.append("Unknown")
                    
                    logger.info(f"First 10 expiry dates for filtered instruments: {expiry_dates}")
                
                logger.info(f"Filtered to {len(filtered_instruments)} instruments within date range")
                
                # Fetch option data for filtered instruments
                option_data = []
                with tqdm(total=len(filtered_instruments), desc=f"Fetching {currency} options") as pbar:
                    for inst in filtered_instruments:
                        try:
                            ticker = self.get_ticker(inst['instrument_name'])
                            if not ticker:
                                continue
                                
                            maturity, strike, option_type = self._parse_instrument_name(inst['instrument_name'])
                            expiry_date = datetime.strptime(maturity, '%d%b%y').date()
                            
                            option_data.append({
                                'instrument_name': inst['instrument_name'],
                                'expiration': expiry_date,
                                'strike': strike,
                                'option_type': option_type,
                                'last_price': ticker.get('last_price', np.nan),
                                'mark_price': ticker.get('mark_price', np.nan),
                                'underlying_price': ticker.get('underlying_price', np.nan),
                                'bid': ticker.get('best_bid_price', np.nan),
                                'ask': ticker.get('best_ask_price', np.nan),
                                'iv': ticker.get('mark_iv', np.nan) / 100 if ticker.get('mark_iv') is not None else np.nan,
                                'volume': ticker.get('stats', {}).get('volume', np.nan),
                                'open_interest': ticker.get('open_interest', np.nan),
                                'underlying_index': ticker.get('underlying_index', ''),
                                'quotes_quantity': ticker.get('quotes_quantity', np.nan)
                            })
                            
                        except Exception as e:
                            logger.warning(f"Error processing ticker for {inst.get('instrument_name', 'unknown')}: {e}")
                        
                        finally:
                            pbar.update(1)
                
                # Create DataFrame
                if option_data:
                    df = pd.DataFrame(option_data)
                    
                    # Calculate time to expiry and moneyness
                    df['time_to_expiry'] = (df['expiration'] - datetime.now().date()).dt.days / 365.0
                    df['is_call'] = df['option_type'] == 'call'
                    
                    # Cache the data
                    df.to_csv(cache_file, index=False)
                    
                    options_data[symbol] = df
                    logger.info(f"Successfully fetched data for {len(df)} options for {currency}")
                else:
                    logger.warning(f"No valid options data retrieved for {currency}")
                    options_data[symbol] = pd.DataFrame()
                
            except Exception as e:
                logger.error(f"Error fetching options data for {symbol}: {str(e)}")
                logger.debug(traceback.format_exc())
                options_data[symbol] = pd.DataFrame()
        
        return options_data
    
    async def fetch_data_ws(self, request: Dict) -> Dict:
        """
        Fetch data using WebSocket connection.
        
        Args:
            request: WebSocket request payload
            
        Returns:
            WebSocket response
        """
        try:
            async with websockets.connect(self.ws_url, ping_interval=None) as websocket:
                await websocket.send(json.dumps(request))
                response = await websocket.recv()
                return json.loads(response)
        except Exception as e:
            logger.error(f"WebSocket error: {str(e)}")
            raise NetworkError(f"WebSocket error: {str(e)}")
    
    async def fetch_trades_ws(
        self,
        currency: str,
        start_ts: int,
        end_ts: int,
        count: int = 1000
    ) -> List[Dict]:
        """
        Fetch trades using WebSocket connection.
        
        Args:
            currency: Currency code
            start_ts: Start timestamp in milliseconds
            end_ts: End timestamp in milliseconds
            count: Maximum number of trades to fetch
            
        Returns:
            List of trades
        """
        request = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000),
            "method": "public/get_last_trades_by_currency_and_time",
            "params": {
                "currency": currency,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "count": count,
                "include_old": True
            }
        }
        
        try:
            response = await self.fetch_data_ws(request)
            
            if "result" in response and "trades" in response["result"]:
                return response["result"]["trades"]
            elif "error" in response:
                error_message = response["error"].get("message", "Unknown WebSocket error")
                error_code = response["error"].get("code", "unknown")
                raise APIError(f"WebSocket API error ({error_code}): {error_message}")
            else:
                raise APIError("Unexpected WebSocket response format")
                
        except Exception as e:
            if isinstance(e, APIError):
                raise
            logger.error(f"Error fetching trades via WebSocket: {str(e)}")
            raise NetworkError(f"WebSocket error: {str(e)}")
    
    def fetch_all_data(
        self,
        symbols: List[str],
        start_date: Union[str, date],
        end_date: Union[str, date],
        fetch_options: bool = True,
        use_cache: bool = True
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Fetch both price and options data.
        
        Args:
            symbols: List of symbols to fetch
            start_date: Start date
            end_date: End date
            fetch_options: Whether to fetch options data
            use_cache: Whether to use cached data
            
        Returns:
            Tuple of (price_data, options_data)
        """
        logger.info(f"Fetching all data for {len(symbols)} symbols")
        
        # Fetch price data
        price_data = self.fetch_price_data(symbols, start_date, end_date)
        
        # Fetch options data if requested
        options_data = {}
        if fetch_options:
            options_data = self.fetch_options_data(symbols, start_date, end_date, use_cache)
        
        return price_data, options_data