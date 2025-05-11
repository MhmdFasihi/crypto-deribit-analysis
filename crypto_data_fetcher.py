import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
import requests
from pathlib import Path
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import traceback
import asyncio
import websockets

class CryptoDataFetcher:
    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        max_retries: int = 3,
        timeout: int = 30
    ) -> None:
        """Initialize data fetcher."""
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = requests.Session()
        self.options_cache_dir = Path("options_cache")
        self.options_cache_dir.mkdir(exist_ok=True)
    
    def fetch_price_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Fetch historical price data for given symbols using yfinance."""
        price_data = {}
        
        for symbol in symbols:
            try:
                # Convert symbol to yfinance format (e.g., BTC-USD -> BTC-USD)
                yf_symbol = symbol
                
                # Fetch data using yfinance
                ticker = yf.Ticker(yf_symbol)
                df = ticker.history(start=start_date, end=end_date)
                
                if not df.empty:
                    # Ensure all required columns are present
                    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    for col in required_columns:
                        if col not in df.columns:
                            df[col] = np.nan
                    
                    # Select only required columns
                    df = df[required_columns]
                    
                    # Convert index to datetime if it's not already
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index)
                    
                    price_data[symbol] = df
                else:
                    print(f"No data found for {symbol}")
                    price_data[symbol] = pd.DataFrame()
                
            except Exception as e:
                print(f"Error fetching price data for {symbol}: {e}")
                price_data[symbol] = pd.DataFrame()
        
        return price_data
    
    def _get_cache_filename(self, symbol: str, start_date: str, end_date: str) -> Path:
        """Generate cache filename for options data."""
        # Strip -USD suffix if present
        currency = symbol.replace('-USD', '')
        return self.options_cache_dir / f"{currency}_options_{start_date}_{end_date}.csv"
    
    def _parse_instrument_name(self, name: str) -> Tuple[str, float, str]:
        """Parse Deribit instrument name to get maturity, strike, and option type."""
        try:
            parts = name.split('-')
            maturity = parts[1]
            strike = float(parts[2])
            option_type = 'call' if parts[3] == 'C' else 'put'
            return maturity, strike, option_type
        except Exception as e:
            print(f"Error parsing instrument name {name}: {e}")
            return None, None, None
    
    def _fetch_trades_chunk(self, currency: str, start_ts: int, end_ts: int) -> List[Dict]:
        """Fetch a chunk of trades from Deribit API."""
        params = {
            "currency": currency,
            "kind": "option",
            "count": 10000,
            "include_old": True,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts
        }
        
        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                if "result" in data and "trades" in data["result"]:
                    return data["result"]["trades"]
                return []
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Failed to get data for {currency} from {start_ts} to {end_ts}: {e}")
                    return []
                time.sleep(0.5 * (2 ** attempt))
        return []
    
    def _process_chunk(self, chunk: List[Dict]) -> pd.DataFrame:
        """Process a chunk of options data into a DataFrame."""
        if not chunk:
            return pd.DataFrame()
            
        df = pd.DataFrame(chunk)
        
        # Parse instrument names
        instrument_details = [self._parse_instrument_name(name) for name in df['instrument_name']]
        df['expiration'] = [details[0] for details in instrument_details]
        df['strike'] = [details[1] for details in instrument_details]
        df['option_type'] = [details[2] for details in instrument_details]
        
        # Convert timestamp and expiration date
        df['date_time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['expiration'] = pd.to_datetime(df['expiration'], format='%d%b%y')
        
        # Create derived columns
        df['date'] = df['date_time'].dt.date
        df['time_to_maturity'] = (df['expiration'] - df['date_time']).dt.total_seconds() / 86400
        df['moneyness'] = df['index_price'] / df['strike']
        df['iv'] = df['iv'] / 100  # Convert from percentage to decimal
        df['is_call'] = df['option_type'] == 'call'
        
        # Calculate volume metrics
        df['volume_btc'] = df['price'] * df['contracts']
        df['volume_usd'] = df['volume_btc'] * df['index_price']
        
        # Select and rename required columns
        required_columns = ['strike', 'expiration', 'option_type', 'last_price', 'iv', 'volume_btc', 'volume_usd']
        df = df.rename(columns={'price': 'last_price'})
        
        # Ensure all required columns exist
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        
        return df[required_columns]
    
    def fetch_options_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        options_data = {}
        for symbol in symbols:
            currency = symbol.replace('-USD', '')
            print(f"Fetching options chain for {currency} from Deribit...")
            try:
                instruments = self.get_instruments(currency)
                print(f"Fetched {len(instruments)} instruments for {currency}")
                for inst in instruments[:5]:
                    print(inst)
                rows = []
                for inst in tqdm(instruments, desc=f"{currency} options"):
                    # Parse instrument name for expiry, strike, type
                    maturity, strike, option_type = self._parse_instrument_name(inst['instrument_name'])
                    print(f"Instrument: {inst['instrument_name']}, Maturity: {maturity}, Strike: {strike}, Type: {option_type}")
                    print(f"Expiration: {pd.to_datetime(maturity, format='%d%b%y')}, In range: {pd.to_datetime(start_date) <= pd.to_datetime(maturity, format='%d%b%y') <= pd.to_datetime(end_date)}")
                    # Filter by date
                    if not (pd.to_datetime(start_date) <= pd.to_datetime(maturity, format='%d%b%y') <= pd.to_datetime(end_date)):
                        continue
                    try:
                        ticker = self.get_ticker(inst['instrument_name'])
                    except Exception as e:
                        print(f"  Error fetching ticker for {inst['instrument_name']}: {e}")
                        continue
                    row = {
                        'instrument_name': inst['instrument_name'],
                        'expiration': pd.to_datetime(maturity, format='%d%b%y'),
                        'strike': strike,
                        'option_type': option_type,
                        'last_price': ticker.get('last_price', np.nan),
                        'iv': ticker.get('iv', np.nan),
                        'bid': ticker.get('best_bid_price', np.nan),
                        'ask': ticker.get('best_ask_price', np.nan),
                        'volume': ticker.get('volume', np.nan),
                        'open_interest': ticker.get('open_interest', np.nan),
                    }
                    rows.append(row)
                    time.sleep(0.05)  # Be nice to the API
                df = pd.DataFrame(rows)
                # Ensure all required columns exist
                required_columns = ['strike', 'expiration', 'option_type', 'last_price', 'iv', 'bid', 'ask', 'volume', 'open_interest']
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = np.nan
                options_data[symbol] = df
                print(f"{symbol}: {len(df)} options contracts fetched.")
                print("\nOptions data summary:")
                for symbol, df in options_data.items():
                    print(f"{symbol}: {len(df)} rows, columns: {list(df.columns)}")
                    print(df.head())
                print(f"After filtering: {len(df)} rows for {symbol}")
            except Exception as e:
                print(f"Error fetching options for {symbol}: {e}")
                options_data[symbol] = pd.DataFrame()
        return options_data
    
    def get_instruments(self, currency: str) -> list:
        url = "https://www.deribit.com/api/v2/public/get_instruments"
        params = {
            "currency": currency,
            "kind": "option",
            "expired": False
        }
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        return data["result"]
    
    def get_ticker(self, instrument_name: str) -> dict:
        url = "https://www.deribit.com/api/v2/public/ticker"
        params = {"instrument_name": instrument_name}
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["result"]

    async def fetch_trades_ws(self, currency, start_ts, end_ts, count=1000):
        msg = {
            "jsonrpc": "2.0",
            "id": 1469,
            "method": "public/get_last_trades_by_currency_and_time",
            "params": {
                "currency": currency,
                "start_timestamp": start_ts,
                "end_timestamp": end_ts,
                "count": count
            }
        }
        async with websockets.connect('wss://www.deribit.com/ws/api/v2') as websocket:
            await websocket.send(json.dumps(msg))
            response = await websocket.recv()
            data = json.loads(response)
            return data['result']['trades'] 