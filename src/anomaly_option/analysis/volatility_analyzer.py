"""
Volatility analysis module for cryptocurrency price data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional, Union, Any
from scipy import stats
import logging
import traceback
from pathlib import Path

from ..core.config import Config, logger
from ..utils.helpers import calculate_returns, calculate_volatility, save_to_csv

class VolatilityAnalysisError(Exception):
    """Exception raised for errors in the volatility analysis process."""
    pass


class CryptoVolatilityAnalyzer:
    """
    Volatility analyzer with advanced analysis and anomaly detection capabilities.
    """
    
    def __init__(
        self,
        price_data: Dict[str, pd.DataFrame],
        window_size: Optional[int] = None,
        z_threshold: Optional[float] = None,
        volatility_window: Optional[int] = None,
        annualization_factor: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> None:
        """
        Initialize volatility analyzer with price data and parameters.
        
        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            window_size: Window size for rolling calculations
            z_threshold: Z-score threshold for anomaly detection
            volatility_window: Window size for volatility calculations
            annualization_factor: Factor to annualize volatility
            chunk_size: Number of rows to process at once
        """
        self.price_data = price_data
        self.window_size = window_size or Config.WINDOW_SIZE
        self.z_threshold = z_threshold or Config.Z_THRESHOLD
        self.volatility_window = volatility_window or Config.VOLATILITY_WINDOW
        self.annualization_factor = annualization_factor or Config.ANNUALIZATION_FACTOR
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        
        # Results storage
        self.results = {}
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info(f"Initialized CryptoVolatilityAnalyzer with {len(price_data)} symbols")
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.price_data:
            raise ValueError("No price data provided")
        
        for symbol, df in self.price_data.items():
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Price data for {symbol} is not a DataFrame")
            
            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Price data for {symbol} is missing required columns: {missing_columns}")
            
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
        
        logger.debug("Input validation completed successfully")
    
    def _process_chunk(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> pd.DataFrame:
        """
        Process a chunk of data for volatility analysis.
        
        Args:
            df: Price DataFrame
            start_idx: Start index of the chunk
            end_idx: End index of the chunk
            
        Returns:
            Processed DataFrame chunk with volatility and anomaly metrics
        """
        try:
            chunk = df.iloc[start_idx:end_idx].copy()
            
            # Basic returns
            chunk['Returns'] = chunk['Close'].pct_change()
            chunk['Log_Returns'] = np.log(chunk['Close'] / chunk['Close'].shift(1))
            
            # Calculate volatility metrics
            chunk['RV_Close'] = self.calculate_realized_volatility(chunk['Returns'])
            
            # Only calculate Parkinson and Garman-Klass if we have OHLC data
            has_ohlc = all(col in chunk.columns for col in ['Open', 'High', 'Low', 'Close'])
            
            if has_ohlc:
                chunk['RV_Parkinson'] = self.calculate_parkinson_volatility(chunk)
                chunk['RV_Garman_Klass'] = self.calculate_garman_klass_volatility(chunk)
                
                vol_columns = ['RV_Close', 'RV_Parkinson', 'RV_Garman_Klass']
                chunk['RV_Composite'] = chunk[vol_columns].mean(axis=1, skipna=True)
            
            else:
                logger.warning("Missing OHLC data, using close-to-close volatility only")
                chunk['RV_Parkinson'] = np.nan
                chunk['RV_Garman_Klass'] = np.nan
                chunk['RV_Composite'] = chunk['RV_Close']
            
            # Z-scores for anomaly detection
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
                np.where(chunk['Returns'] > 0, 1, -1),  # 1 for positive, -1 for negative
                0  # 0 for no anomaly
            )
            
            # Calculate anomaly magnitude
            chunk['Anomaly_Magnitude'] = np.where(
                chunk['Combined_Anomaly'] == 1,
                np.abs(chunk['Returns']),
                0
            )
            
            return chunk
            
        except Exception as e:
            logger.error(f"Error processing chunk: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VolatilityAnalysisError(f"Failed to process chunk: {str(e)}")
    
    def calculate_z_score(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate rolling Z-score for a data series.
        
        Args:
            data: Input data series
            
        Returns:
            Tuple of (rolling_mean, rolling_std, z_score)
        """
        rolling_mean = data.rolling(window=self.window_size, min_periods=2).mean()
        rolling_std = data.rolling(window=self.window_size, min_periods=2).std()
        
        # Handle division by zero or NaN
        z_score = np.zeros_like(data, dtype=float)
        valid_mask = (rolling_std > 0) & ~rolling_std.isna()
        z_score[valid_mask] = (data[valid_mask] - rolling_mean[valid_mask]) / rolling_std[valid_mask]
        
        return rolling_mean, rolling_std, pd.Series(z_score, index=data.index)
    
    def calculate_realized_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Calculate realized volatility using close-to-close returns.
        
        Args:
            returns: Return series
            
        Returns:
            Annualized volatility series
        """
        # Use min_periods to handle startup effects
        return returns.rolling(window=self.volatility_window, min_periods=2).std() * np.sqrt(self.annualization_factor)
    
    def calculate_parkinson_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Parkinson volatility using high-low range.
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Parkinson volatility series
        """
        # Handle zero or negative prices
        high = df['High'].replace(0, np.nan)
        low = df['Low'].replace(0, np.nan)
        
        # Calculate log of high/low ratio
        with np.errstate(divide='ignore', invalid='ignore'):
            log_hl = np.log(high / low)
        
        # Square and scale
        hl_square = log_hl ** 2
        estimator = hl_square / (4 * np.log(2))
        
        # Rolling window calculation with min_periods
        parkinson = np.sqrt(
            estimator.rolling(window=self.volatility_window, min_periods=2).mean() * 
            self.annualization_factor
        )
        
        return parkinson
    
    def calculate_garman_klass_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Garman-Klass volatility using OHLC data.
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Garman-Klass volatility series
        """
        # Handle zero or negative prices
        open_price = df['Open'].replace(0, np.nan)
        high = df['High'].replace(0, np.nan)
        low = df['Low'].replace(0, np.nan)
        close = df['Close'].replace(0, np.nan)
        
        # Calculate log price ratios with error handling
        with np.errstate(divide='ignore', invalid='ignore'):
            log_hl = np.log(high / low) ** 2
            log_co = np.log(close / open_price) ** 2
        
        # Garman-Klass formula
        estimator = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
        
        # Rolling window calculation with min_periods
        gk_vol = np.sqrt(
            estimator.rolling(window=self.volatility_window, min_periods=2).mean() * 
            self.annualization_factor
        )
        
        return gk_vol
    
    def calculate_yang_zhang_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate Yang-Zhang volatility using OHLC data.
        
        Args:
            df: OHLC DataFrame
            
        Returns:
            Yang-Zhang volatility series
        """
        # Handle zero or negative prices
        open_price = df['Open'].replace(0, np.nan)
        high = df['High'].replace(0, np.nan)
        low = df['Low'].replace(0, np.nan)
        close = df['Close'].replace(0, np.nan)
        prev_close = close.shift(1)
        
        # Calculate overnight volatility (close to open)
        with np.errstate(divide='ignore', invalid='ignore'):
            overnight_returns = np.log(open_price / prev_close)
            overnight_vol = overnight_returns.rolling(window=self.volatility_window, min_periods=2).var()
            
            # Calculate open to close volatility
            open_close_returns = np.log(close / open_price)
            open_close_vol = open_close_returns.rolling(window=self.volatility_window, min_periods=2).var()
            
            # Calculate Rogers-Satchell volatility
            rs_vol = (
                np.log(high / close) * np.log(high / open_price) + 
                np.log(low / close) * np.log(low / open_price)
            ).rolling(window=self.volatility_window, min_periods=2).mean()
        
        # Yang-Zhang formula (k=0.34 typically)
        k = 0.34
        yang_zhang = np.sqrt(
            (overnight_vol + k * open_close_vol + (1 - k) * rs_vol) * self.annualization_factor
        )
        
        return yang_zhang
    
    def analyze_volatility(
        self, 
        symbol: str,
        additional_metrics: bool = True
    ) -> pd.DataFrame:
        """
        Analyze volatility and detect anomalies for a given symbol.
        
        Args:
            symbol: Symbol to analyze
            additional_metrics: Whether to calculate additional metrics
            
        Returns:
            DataFrame with volatility and anomaly metrics
        """
        if symbol not in self.price_data:
            logger.error(f"Price data not available for {symbol}")
            raise KeyError(f"Price data not available for {symbol}")
        
        try:
            # Extract price data
            df = self.price_data[symbol].copy()
            
            if df.empty:
                logger.warning(f"Empty price data for {symbol}")
                return pd.DataFrame()
            
            logger.info(f"Analyzing volatility for {symbol} with {len(df)} data points")
            
            # Process data in chunks to limit memory usage
            chunks = []
            start_time = datetime.now()
            
            for start_idx in range(0, len(df), self.chunk_size):
                end_idx = min(start_idx + self.chunk_size, len(df))
                chunk = self._process_chunk(df, start_idx, end_idx)
                chunks.append(chunk)
                
                # Clear memory
                del chunk
                gc.collect()
                
                # Progress logging
                if (start_idx + self.chunk_size) % (self.chunk_size * 10) == 0:
                    progress = min(100, int((start_idx + self.chunk_size) / len(df) * 100))
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.debug(f"Progress: {progress}% ({start_idx + self.chunk_size}/{len(df)}) in {elapsed:.2f}s")
            
            # Combine chunks
            result = pd.concat(chunks, axis=0)
            
            # Clear memory
            del chunks
            gc.collect()
            
            logger.debug(f"Basic volatility analysis completed for {symbol}")
            
            # Calculate additional market indicators if requested
            if additional_metrics:
                logger.debug(f"Calculating additional metrics for {symbol}")
                
                # Volume-weighted metrics (if Volume is available)
                if 'Volume' in result.columns and not result['Volume'].isnull().all():
                    # VWAP (Volume Weighted Average Price)
                    result['VWAP'] = (result['Close'] * result['Volume']).cumsum() / result['Volume'].cumsum()
                    
                    # Volume volatility
                    result['Volume_Change'] = result['Volume'].pct_change()
                    result['Volume_Z_Score'] = self.calculate_z_score(result['Volume'])[2]
                
                # Momentum indicators
                result['Momentum_5D'] = result['Close'].pct_change(periods=5)
                result['Momentum_20D'] = result['Close'].pct_change(periods=20)
                
                # Relative Strength Index (RSI)
                delta = result['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    rs = gain / loss
                    result['RSI'] = 100 - (100 / (1 + rs))
                
                # Add forward returns for backtesting
                for days in [1, 5, 10, 20]:
                    result[f'Forward_{days}d_Return'] = result['Close'].pct_change(periods=days).shift(-days)
            
            # Remove NaN values only at the beginning
            first_valid_idx = result['RV_Composite'].first_valid_index()
            if first_valid_idx is not None:
                result = result.loc[first_valid_idx:]
            
            # Log results
            anomalies_count = result['Combined_Anomaly'].sum()
            avg_volatility = result['RV_Composite'].mean()
            logger.info(f"Volatility analysis for {symbol} completed: {anomalies_count} anomalies detected, average volatility: {avg_volatility:.2%}")
            
            # Store results
            self.results[symbol] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing volatility for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VolatilityAnalysisError(f"Failed to analyze volatility for {symbol}: {str(e)}")
    
    def analyze_all_symbols(self, additional_metrics: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Analyze volatility for all symbols.
        
        Args:
            additional_metrics: Whether to calculate additional metrics
            
        Returns:
            Dictionary mapping symbols to result DataFrames
        """
        logger.info(f"Analyzing volatility for {len(self.price_data)} symbols")
        
        for symbol in self.price_data.keys():
            try:
                self.analyze_volatility(symbol, additional_metrics)
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                self.results[symbol] = pd.DataFrame()
        
        return self.results
    
    def get_anomalies(self, symbol: str) -> pd.DataFrame:
        """
        Get anomalies for a symbol.
        
        Args:
            symbol: Symbol to get anomalies for
            
        Returns:
            DataFrame with anomaly data
        """
        if symbol not in self.results:
            logger.warning(f"No results available for {symbol}, analyzing now")
            self.analyze_volatility(symbol)
        
        if symbol not in self.results or self.results[symbol].empty:
            logger.warning(f"No results available for {symbol}")
            return pd.DataFrame()
        
        # Filter to anomaly days
        anomalies = self.results[symbol][self.results[symbol]['Combined_Anomaly'] == 1].copy()
        
        if anomalies.empty:
            logger.info(f"No anomalies detected for {symbol}")
        else:
            logger.info(f"Found {len(anomalies)} anomalies for {symbol}")
            
        return anomalies
    
    def analyze_anomaly_impact(
        self, 
        symbol: str, 
        forward_days: List[int] = [1, 5, 10, 20]
    ) -> Dict[str, Any]:
        """
        Analyze the impact of anomalies on future returns.
        
        Args:
            symbol: Symbol to analyze
            forward_days: List of forward days to analyze
            
        Returns:
            Dictionary with anomaly impact analysis
        """
        if symbol not in self.results:
            logger.warning(f"No results available for {symbol}, analyzing now")
            self.analyze_volatility(symbol, additional_metrics=True)
        
        if symbol not in self.results or self.results[symbol].empty:
            logger.warning(f"No results available for {symbol}")
            return {}
        
        try:
            data = self.results[symbol]
            anomalies = data[data['Combined_Anomaly'] == 1]
            
            if anomalies.empty:
                logger.info(f"No anomalies detected for {symbol}")
                return {
                    'symbol': symbol,
                    'anomaly_count': 0,
                    'impact': {}
                }
            
            # Calculate forward returns if not already done
            for days in forward_days:
                col_name = f'Forward_{days}d_Return'
                if col_name not in data.columns:
                    data[col_name] = data['Close'].pct_change(periods=days).shift(-days)
            
            # Analyze impact by anomaly type
            impact = {}
            
            for days in forward_days:
                col_name = f'Forward_{days}d_Return'
                
                # All anomalies
                all_mean = anomalies[col_name].mean()
                all_median = anomalies[col_name].median()
                all_std = anomalies[col_name].std()
                
                # Positive anomalies (price above threshold)
                pos_anomalies = anomalies[anomalies['Anomaly_Direction'] == 1]
                pos_mean = pos_anomalies[col_name].mean() if not pos_anomalies.empty else np.nan
                pos_median = pos_anomalies[col_name].median() if not pos_anomalies.empty else np.nan
                
                # Negative anomalies (price below threshold)
                neg_anomalies = anomalies[anomalies['Anomaly_Direction'] == -1]
                neg_mean = neg_anomalies[col_name].mean() if not neg_anomalies.empty else np.nan
                neg_median = neg_anomalies[col_name].median() if not neg_anomalies.empty else np.nan
                
                # Store results
                impact[days] = {
                    'all': {
                        'mean': all_mean,
                        'median': all_median,
                        'std': all_std,
                        'count': len(anomalies)
                    },
                    'positive': {
                        'mean': pos_mean,
                        'median': pos_median,
                        'count': len(pos_anomalies)
                    },
                    'negative': {
                        'mean': neg_mean,
                        'median': neg_median,
                        'count': len(neg_anomalies)
                    }
                }
            
            # Statistical significance testing
            for days in forward_days:
                col_name = f'Forward_{days}d_Return'
                
                # All anomalies vs all data
                normal_returns = data[~data['Combined_Anomaly'].astype(bool)][col_name].dropna()
                anomaly_returns = anomalies[col_name].dropna()
                
                if len(normal_returns) > 0 and len(anomaly_returns) > 0:
                    t_stat, p_value = stats.ttest_ind(
                        anomaly_returns, 
                        normal_returns,
                        equal_var=False  # Use Welch's t-test for unequal variances
                    )
                    impact[days]['significance'] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
                else:
                    impact[days]['significance'] = {
                        't_statistic': np.nan,
                        'p_value': np.nan,
                        'significant': False
                    }
            
            return {
                'symbol': symbol,
                'anomaly_count': len(anomalies),
                'anomaly_rate': len(anomalies) / len(data),
                'impact': impact
            }
            
        except Exception as e:
            logger.error(f"Error analyzing anomaly impact for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'symbol': symbol,
                'error': str(e)
            }
    
    def filter_anomalies_by_date(
        self, 
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Filter anomalies by date range.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping symbols to filtered anomalies
        """
        # Convert dates if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        filtered_anomalies = {}
        
        for symbol, df in self.results.items():
            if df.empty:
                continue
            
            # Get all anomalies
            anomalies = df[df['Combined_Anomaly'] == 1].copy()
            
            if anomalies.empty:
                continue
            
            # Filter by date if specified
            if start_date:
                anomalies = anomalies[anomalies.index.date >= start_date]
            if end_date:
                anomalies = anomalies[anomalies.index.date <= end_date]
            
            if not anomalies.empty:
                filtered_anomalies[symbol] = anomalies
        
        logger.info(f"Found anomalies for {len(filtered_anomalies)} symbols in date range")
        return filtered_anomalies
    
    def get_summary_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Get summary statistics for all analyzed symbols.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {}
        
        for symbol, df in self.results.items():
            if df.empty:
                continue
            
            anomalies = df[df['Combined_Anomaly'] == 1]
            
            summary[symbol] = {
                'data_points': len(df),
                'start_date': df.index.min().date(),
                'end_date': df.index.max().date(),
                'volatility': {
                    'mean': df['RV_Composite'].mean(),
                    'median': df['RV_Composite'].median(),
                    'min': df['RV_Composite'].min(),
                    'max': df['RV_Composite'].max(),
                    'std': df['RV_Composite'].std()
                },
                'returns': {
                    'mean': df['Returns'].mean(),
                    'median': df['Returns'].median(),
                    'min': df['Returns'].min(),
                    'max': df['Returns'].max(),
                    'std': df['Returns'].std()
                },
                'anomalies': {
                    'count': len(anomalies),
                    'rate': len(anomalies) / len(df) if len(df) > 0 else 0,
                    'price_anomalies': df['Price_Anomaly'].sum(),
                    'return_anomalies': df['Return_Anomaly'].sum(),
                    'volatility_anomalies': df['Volatility_Anomaly'].sum(),
                    'positive_anomalies': len(anomalies[anomalies['Anomaly_Direction'] == 1]),
                    'negative_anomalies': len(anomalies[anomalies['Anomaly_Direction'] == -1])
                }
            }
        
        return summary
    
    def cleanup(self) -> None:
        """Clean up resources and memory."""
        logger.debug("Cleaning up resources")
        self.price_data.clear()
        self.results.clear()
        gc.collect()  # Force garbage collection