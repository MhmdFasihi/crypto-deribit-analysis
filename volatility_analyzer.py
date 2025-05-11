import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any
import gc

class CryptoVolatilityAnalyzer:
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
        """Calculate rolling mean, standard deviation, and Z-score."""
        rolling_mean = data.rolling(window=self.window_size).mean()
        rolling_std = data.rolling(window=self.window_size).std()
        z_score = (data - rolling_mean) / rolling_std
        return rolling_mean, rolling_std, z_score
    
    def calculate_realized_volatility(self, returns: pd.Series) -> pd.Series:
        """Calculate realized volatility using close-to-close returns."""
        return returns.rolling(window=self.volatility_window).std() * np.sqrt(self.annualization_factor)
    
    def calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Parkinson volatility using high-low range."""
        hl_ratio = np.log(df['High'] / df['Low']) ** 2
        return np.sqrt(hl_ratio.rolling(window=self.volatility_window).mean() / (4 * np.log(2))) * np.sqrt(self.annualization_factor)
    
    def calculate_garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Garman-Klass volatility using OHLC data."""
        log_hl = np.log(df['High'] / df['Low']) ** 2
        log_co = np.log(df['Close'] / df['Open']) ** 2
        vol = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_co)
        return vol.rolling(window=self.volatility_window).mean() * np.sqrt(self.annualization_factor)
    
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