"""
Options analysis module for cryptocurrency options data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from scipy.stats import norm
import scipy.optimize as optimize
import warnings
import logging
import traceback
import gc
from functools import lru_cache
from pathlib import Path

from ..core.config import Config, logger
from ..utils.helpers import save_to_csv, load_from_csv, safe_divide

class OptionsAnalysisError(Exception):
    """Exception raised for errors in options analysis."""
    pass

class BlackScholesError(OptionsAnalysisError):
    """Exception raised for errors in Black-Scholes calculations."""
    pass

class ImpliedVolatilityError(OptionsAnalysisError):
    """Exception raised for errors in implied volatility calculations."""
    pass


class OptionsAnalyzer:
    """
    Advanced options analyzer for cryptocurrency derivatives.
    """
    
    def __init__(
        self,
        options_data: Dict[str, pd.DataFrame],
        price_data: Dict[str, pd.DataFrame],
        risk_free_rate: Optional[float] = None,
        min_volume: int = 1,
        min_open_interest: int = 1,
        min_days_to_expiry: int = 1,
        max_days_to_expiry: int = 365,
        min_strike_distance: float = 0.05,  # 5% from current price
        max_strike_distance: float = 2.0,  # 200% from current price
        min_implied_volatility: float = 0.01,  # 1% minimum IV
        max_implied_volatility: float = 5.0,  # 500% maximum IV
        numerical_precision: int = 6
    ) -> None:
        """
        Initialize options analyzer with options and price data.
        
        Args:
            options_data: Dictionary mapping symbols to options DataFrames
            price_data: Dictionary mapping symbols to price DataFrames
            risk_free_rate: Risk-free rate for pricing
            min_volume: Minimum trading volume filter
            min_open_interest: Minimum open interest filter
            min_days_to_expiry: Minimum days to expiry filter
            max_days_to_expiry: Maximum days to expiry filter
            min_strike_distance: Minimum distance from ATM as fraction
            max_strike_distance: Maximum distance from ATM as fraction
            min_implied_volatility: Minimum acceptable IV
            max_implied_volatility: Maximum acceptable IV
            numerical_precision: Decimal precision for calculations
        """
        self.options_data = options_data
        self.price_data = price_data
        self.risk_free_rate = risk_free_rate or Config.RISK_FREE_RATE
        self.min_volume = min_volume
        self.min_open_interest = min_open_interest
        self.min_days_to_expiry = min_days_to_expiry
        self.max_days_to_expiry = max_days_to_expiry
        self.min_strike_distance = min_strike_distance
        self.max_strike_distance = max_strike_distance
        self.min_implied_volatility = min_implied_volatility
        self.max_implied_volatility = max_implied_volatility
        self.numerical_precision = numerical_precision
        
        # Results storage
        self.results = {}
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info(f"Initialized OptionsAnalyzer with {len(options_data)} symbols")
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.options_data:
            raise ValueError("No options data provided")
            
        if not self.price_data:
            raise ValueError("No price data provided")
            
        if self.risk_free_rate < 0:
            raise ValueError("risk_free_rate must be non-negative")
            
        if self.min_volume < 0:
            raise ValueError("min_volume must be non-negative")
            
        if self.min_open_interest < 0:
            raise ValueError("min_open_interest must be non-negative")
            
        if self.min_days_to_expiry < 0:
            raise ValueError("min_days_to_expiry must be non-negative")
            
        if self.min_days_to_expiry >= self.max_days_to_expiry:
            raise ValueError("min_days_to_expiry must be less than max_days_to_expiry")
            
        if self.min_strike_distance < 0 or self.min_strike_distance >= self.max_strike_distance:
            raise ValueError("min_strike_distance must be non-negative and less than max_strike_distance")
            
        if self.min_implied_volatility < 0 or self.min_implied_volatility >= self.max_implied_volatility:
            raise ValueError("min_implied_volatility must be non-negative and less than max_implied_volatility")
            
        if self.numerical_precision < 0:
            raise ValueError("numerical_precision must be non-negative")
        
        logger.debug("Input validation completed successfully")
    
    @lru_cache(maxsize=1024)
    def _calculate_black_scholes(
        self,
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration in years
        sigma: float,  # Volatility
        r: float,  # Risk-free rate
        option_type: str = 'call',
        dividend_yield: float = 0.0  # Dividend yield
    ) -> Dict[str, float]:
        """
        Calculate option price and Greeks using Black-Scholes model.
        
        Args:
            S: Current price
            K: Strike price
            T: Time to expiry in years
            sigma: Volatility
            r: Risk-free rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            
        Returns:
            Dictionary with price and Greeks
        """
        try:
            # Input validation
            if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or r < 0:
                raise BlackScholesError(
                    f"Invalid Black-Scholes inputs: S={S}, K={K}, T={T}, sigma={sigma}, r={r}"
                )
            
            # Avoid numerical issues with very small T
            T = max(T, 1e-8)
            
            # Calculate d1 and d2 with improved numerical stability
            d1 = (np.log(S/K) + (r - dividend_yield + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            
            # Calculate option price
            if option_type.lower() == 'call':
                price = S * np.exp(-dividend_yield * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                delta = np.exp(-dividend_yield * T) * norm.cdf(d1)
                theta = (
                    -np.exp(-dividend_yield * T) * S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) - 
                    r * K * np.exp(-r * T) * norm.cdf(d2) + 
                    dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(d1)
                )
            else:  # put
                price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-dividend_yield * T) * norm.cdf(-d1)
                delta = np.exp(-dividend_yield * T) * (norm.cdf(d1) - 1)
                theta = (
                    -np.exp(-dividend_yield * T) * S * sigma * norm.pdf(d1) / (2 * np.sqrt(T)) + 
                    r * K * np.exp(-r * T) * norm.cdf(-d2) - 
                    dividend_yield * S * np.exp(-dividend_yield * T) * norm.cdf(-d1)
                )
            
            # Greeks common to both calls and puts
            gamma = norm.pdf(d1) * np.exp(-dividend_yield * T) / (S * sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1) * np.exp(-dividend_yield * T) / 100  # Divided by 100 for percentage
            rho = K * T * np.exp(-r * T) * norm.cdf(d2 if option_type.lower() == 'call' else -d2) / 100
            
            # Round results to specified precision
            return {
                'price': round(price, self.numerical_precision),
                'delta': round(delta, self.numerical_precision),
                'gamma': round(gamma, self.numerical_precision),
                'vega': round(vega, self.numerical_precision),
                'theta': round(theta, self.numerical_precision),
                'rho': round(rho, self.numerical_precision)
            }
            
        except Exception as e:
            logger.error(f"Black-Scholes calculation error: {str(e)}")
            logger.debug(traceback.format_exc())
            raise BlackScholesError(f"Failed to calculate Black-Scholes: {str(e)}")
    
    def _calculate_implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        dividend_yield: float = 0.0,
        initial_guess: float = 0.3,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            market_price: Market price of the option
            S: Current price
            K: Strike price
            T: Time to expiry in years
            r: Risk-free rate
            option_type: 'call' or 'put'
            dividend_yield: Dividend yield
            initial_guess: Initial guess for volatility
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Implied volatility
        """
        try:
            # Input validation
            if market_price <= 0 or S <= 0 or K <= 0 or T <= 0 or r < 0:
                raise ImpliedVolatilityError(
                    f"Invalid inputs for IV calculation: price={market_price}, S={S}, K={K}, T={T}, r={r}"
                )
            
            # Avoid numerical issues with very small T
            T = max(T, 1e-8)
            
            # Check if market price exceeds theoretical limits
            intrinsic_value = max(0, S - K) if option_type.lower() == 'call' else max(0, K - S)
            if market_price < intrinsic_value:
                logger.warning(f"Market price ({market_price}) less than intrinsic value ({intrinsic_value})")
                return self.min_implied_volatility
            
            # Define the objective function to minimize
            def objective(sigma):
                try:
                    option_price = self._calculate_black_scholes(
                        S, K, T, sigma, r, option_type, dividend_yield
                    )['price']
                    return option_price - market_price
                except Exception as e:
                    logger.warning(f"Error in objective function: {str(e)}")
                    return float('inf')
            
            # Try Newton-Raphson method first
            sigma = initial_guess
            for i in range(max_iterations):
                # Calculate option price and vega
                try:
                    bs_result = self._calculate_black_scholes(S, K, T, sigma, r, option_type, dividend_yield)
                    price_diff = bs_result['price'] - market_price
                    vega = bs_result['vega']
                    
                    # Check convergence
                    if abs(price_diff) < tolerance:
                        return round(sigma, self.numerical_precision)
                    
                    # Update sigma using Newton-Raphson
                    if abs(vega) < 1e-8:  # Avoid division by near-zero
                        break  # Switch to bisection method
                        
                    new_sigma = sigma - price_diff / (vega * 100)  # Multiply by 100 because vega is in percentage terms
                    
                    # Ensure sigma stays within bounds and check for convergence
                    new_sigma = max(self.min_implied_volatility, min(new_sigma, self.max_implied_volatility))
                    
                    # If change is very small, return the current value
                    if abs(new_sigma - sigma) < tolerance:
                        return round(sigma, self.numerical_precision)
                    
                    sigma = new_sigma
                    
                except Exception:
                    logger.warning(f"Newton-Raphson iteration failed, switching to bisection")
                    break
            
            # If Newton-Raphson fails, try bisection method
            try:
                return round(
                    optimize.brentq(
                        objective,
                        self.min_implied_volatility,
                        self.max_implied_volatility,
                        xtol=tolerance,
                        maxiter=max_iterations
                    ),
                    self.numerical_precision
                )
            except Exception as e:
                logger.error(f"Implied volatility calculation failed: {str(e)}")
                # Return NaN for failed calculations
                return np.nan
                
        except Exception as e:
            logger.error(f"Implied volatility calculation error: {str(e)}")
            logger.debug(traceback.format_exc())
            return np.nan
    
    def _filter_options_data(
        self, 
        options_df: pd.DataFrame, 
        current_price: float
    ) -> pd.DataFrame:
        """
        Filter and clean options data based on criteria.
        
        Args:
            options_df: Options DataFrame
            current_price: Current price of the underlying
            
        Returns:
            Filtered options DataFrame
        """
        if options_df.empty:
            return pd.DataFrame()
        
        try:
            # Make a copy to avoid modifying original data
            df = options_df.copy()
            
            logger.debug(f"Filtering options data: {len(df)} rows initially")
            
            # Ensure required columns exist
            required_columns = ['strike', 'expiration', 'option_type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # Calculate time to expiry in years
            if 'time_to_expiry' not in df.columns:
                if isinstance(df['expiration'].iloc[0], str):
                    df['expiration'] = pd.to_datetime(df['expiration'])
                
                df['time_to_expiry'] = (df['expiration'] - datetime.now()).dt.total_seconds() / (365.25 * 24 * 3600)
            
            # Calculate moneyness
            df['moneyness'] = df['strike'] / current_price
            
            # Apply filters
            original_len = len(df)
            
            # Filter by volume/open interest if columns exist
            if 'volume' in df.columns:
                df = df[df['volume'] >= self.min_volume]
                logger.debug(f"After volume filter: {len(df)}/{original_len} rows remain")
            
            if 'open_interest' in df.columns:
                df = df[df['open_interest'] >= self.min_open_interest]
                logger.debug(f"After open interest filter: {len(df)}/{original_len} rows remain")
            
            # Filter by time to expiry
            df = df[
                (df['time_to_expiry'] >= self.min_days_to_expiry / 365.25) & 
                (df['time_to_expiry'] <= self.max_days_to_expiry / 365.25)
            ]
            logger.debug(f"After time to expiry filter: {len(df)}/{original_len} rows remain")
            
            # Filter by strike distance
            df = df[
                (df['moneyness'] >= 1 - self.max_strike_distance) & 
                (df['moneyness'] <= 1 + self.max_strike_distance)
            ]
            logger.debug(f"After strike distance filter: {len(df)}/{original_len} rows remain")
            
            # Add metadata columns
            df['days_to_expiry'] = df['time_to_expiry'] * 365.25
            df['is_call'] = df['option_type'].str.lower() == 'call'
            
            return df
            
        except Exception as e:
            logger.error(f"Error filtering options data: {str(e)}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def _calculate_implied_volatility_and_greeks(
        self,
        df: pd.DataFrame,
        current_price: float
    ) -> pd.DataFrame:
        """
        Calculate implied volatility and Greeks for options.
        
        Args:
            df: Options DataFrame
            current_price: Current price of the underlying
            
        Returns:
            DataFrame with IV and Greeks added
        """
        if df.empty:
            return df
        
        try:
            # Make a copy to avoid modifying original data
            result = df.copy()
            
            # Check which price column to use
            price_col = None
            for col in ['last_price', 'mark_price', 'mid_price', 'price']:
                if col in result.columns and not result[col].isnull().all():
                    price_col = col
                    break
            
            if price_col is None:
                logger.error("No valid price column found in options data")
                return df
            
            # Calculate IV and Greeks row by row
            for idx, row in result.iterrows():
                try:
                    # Skip rows with missing data
                    if pd.isna(row[price_col]) or pd.isna(row['strike']) or pd.isna(row['time_to_expiry']):
                        continue
                    
                    # Calculate IV
                    if 'iv' not in result.columns or pd.isna(row['iv']):
                        iv = self._calculate_implied_volatility(
                            market_price=row[price_col],
                            S=current_price,
                            K=row['strike'],
                            T=row['time_to_expiry'],
                            r=self.risk_free_rate,
                            option_type=row['option_type']
                        )
                        result.at[idx, 'iv'] = iv
                    else:
                        iv = row['iv']
                    
                    # Skip if IV calculation failed
                    if pd.isna(iv):
                        continue
                    
                    # Check if IV is within acceptable range
                    if iv < self.min_implied_volatility or iv > self.max_implied_volatility:
                        logger.debug(f"IV out of range: {iv}")
                        continue
                    
                    # Calculate Greeks
                    greeks = self._calculate_black_scholes(
                        S=current_price,
                        K=row['strike'],
                        T=row['time_to_expiry'],
                        sigma=iv,
                        r=self.risk_free_rate,
                        option_type=row['option_type']
                    )
                    
                    # Add Greeks to DataFrame
                    for greek, value in greeks.items():
                        if greek != 'price':  # Skip price to avoid overwriting market price
                            result.at[idx, greek] = value
                            
                except Exception as e:
                    logger.warning(f"Error calculating IV/Greeks for option {idx}: {str(e)}")
                    continue
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating IV and Greeks: {str(e)}")
            logger.debug(traceback.format_exc())
            return df
    
    def analyze_options(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze options data for a given symbol.
        
        Args:
            symbol: Symbol to analyze
            
        Returns:
            Dictionary with analysis results
        """
        if symbol not in self.options_data or symbol not in self.price_data:
            logger.error(f"Data not available for {symbol}")
            raise KeyError(f"Data not available for {symbol}")
        
        try:
            # Get options and price data
            options_df = self.options_data[symbol]
            price_df = self.price_data[symbol]
            
            if options_df.empty or price_df.empty:
                logger.warning(f"Empty data for {symbol}")
                return {}
            
            # Get current price (last close price)
            current_price = price_df['Close'].iloc[-1]
            logger.info(f"Analyzing options for {symbol} with current price {current_price}")
            
            # Filter and process options data
            filtered_df = self._filter_options_data(options_df, current_price)
            
            if filtered_df.empty:
                logger.warning(f"No valid options data after filtering for {symbol}")
                return {}
            
            # Calculate implied volatility and Greeks
            processed_df = self._calculate_implied_volatility_and_greeks(filtered_df, current_price)
            
            # Analyze volumes
            volume_analysis = self._analyze_volumes(processed_df)
            
            # Analyze implied volatility
            iv_analysis = self._analyze_implied_volatility(processed_df)
            
            # Analyze Greeks
            greeks_analysis = self._analyze_greeks(processed_df)
            
            # Store results
            results = {
                'options_data': processed_df,
                'volume_analysis': volume_analysis,
                'iv_analysis': iv_analysis,
                'greeks_analysis': greeks_analysis,
                'current_price': current_price,
                'analysis_date': datetime.now().strftime('%Y-%m-%d')
            }
            
            self.results[symbol] = results
            
            logger.info(f"Options analysis completed for {symbol}")
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing options for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise OptionsAnalysisError(f"Failed to analyze options for {symbol}: {str(e)}")
    
    def _analyze_volumes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze options trading volumes.
        
        Args:
            df: Processed options DataFrame
            
        Returns:
            Dictionary with volume analysis results
        """
        try:
            # Check if volume data is available
            if 'volume' not in df.columns or df['volume'].isnull().all():
                logger.warning("No volume data available for analysis")
                return {
                    'total_volume': 0,
                    'call_volume': 0,
                    'put_volume': 0,
                    'volume_ratio': 0
                }
            
            # Split by option type
            calls = df[df['option_type'].str.lower() == 'call']
            puts = df[df['option_type'].str.lower() == 'put']
            
            # Calculate volume metrics
            total_volume = df['volume'].sum()
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            
            # Avoid division by zero
            volume_ratio = call_volume / put_volume if put_volume > 0 else np.nan
            
            # Volume by expiry
            expiry_groups = df.groupby('expiration')['volume'].sum()
            expiry_volume = expiry_groups.to_dict()
            
            # Volume by strike
            strike_groups = df.groupby('strike')['volume'].sum()
            strike_volume = strike_groups.to_dict()
            
            # Volume by moneyness
            # Group into buckets: Deep ITM, ITM, ATM, OTM, Deep OTM
            def get_moneyness_category(row):
                if row['option_type'].lower() == 'call':
                    if row['moneyness'] < 0.8:
                        return 'Deep ITM'
                    elif row['moneyness'] < 0.95:
                        return 'ITM'
                    elif row['moneyness'] < 1.05:
                        return 'ATM'
                    elif row['moneyness'] < 1.2:
                        return 'OTM'
                    else:
                        return 'Deep OTM'
                else:  # put
                    if row['moneyness'] > 1.2:
                        return 'Deep ITM'
                    elif row['moneyness'] > 1.05:
                        return 'ITM'
                    elif row['moneyness'] > 0.95:
                        return 'ATM'
                    elif row['moneyness'] > 0.8:
                        return 'OTM'
                    else:
                        return 'Deep OTM'
            
            df['moneyness_category'] = df.apply(get_moneyness_category, axis=1)
            moneyness_groups = df.groupby('moneyness_category')['volume'].sum()
            moneyness_volume = moneyness_groups.to_dict()
            
            # Most active options
            most_active = df.nlargest(10, 'volume')[['strike', 'expiration', 'option_type', 'volume']]
            most_active_dict = most_active.to_dict(orient='records')
            
            return {
                'total_volume': float(total_volume),
                'call_volume': float(call_volume),
                'put_volume': float(put_volume),
                'volume_ratio': float(volume_ratio),
                'volume_by_expiry': expiry_volume,
                'volume_by_strike': {str(k): float(v) for k, v in strike_volume.items()},
                'volume_by_moneyness': {str(k): float(v) for k, v in moneyness_volume.items()},
                'most_active_options': most_active_dict
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volumes: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'total_volume': 0,
                'call_volume': 0,
                'put_volume': 0,
                'error': str(e)
            }
    
    def _analyze_implied_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze implied volatility patterns.
        
        Args:
            df: Processed options DataFrame
            
        Returns:
            Dictionary with IV analysis results
        """
        try:
            # Check if IV data is available
            if 'iv' not in df.columns or df['iv'].isnull().all():
                logger.warning("No implied volatility data available for analysis")
                return {
                    'mean_iv': np.nan,
                    'median_iv': np.nan,
                    'min_iv': np.nan,
                    'max_iv': np.nan,
                    'std_iv': np.nan
                }
            
            # Filter out invalid IV values
            valid_iv = df[
                (df['iv'] >= self.min_implied_volatility) & 
                (df['iv'] <= self.max_implied_volatility)
            ]
            
            if valid_iv.empty:
                logger.warning("No valid implied volatility data after filtering")
                return {
                    'mean_iv': np.nan,
                    'median_iv': np.nan,
                    'min_iv': np.nan,
                    'max_iv': np.nan,
                    'std_iv': np.nan
                }
            
            # Basic IV statistics
            iv_stats = {
                'mean_iv': float(valid_iv['iv'].mean()),
                'median_iv': float(valid_iv['iv'].median()),
                'min_iv': float(valid_iv['iv'].min()),
                'max_iv': float(valid_iv['iv'].max()),
                'std_iv': float(valid_iv['iv'].std())
            }
            
            # IV by option type
            call_iv = valid_iv[valid_iv['option_type'].str.lower() == 'call']['iv']
            put_iv = valid_iv[valid_iv['option_type'].str.lower() == 'put']['iv']
            
            iv_by_type = {
                'call_mean_iv': float(call_iv.mean()) if not call_iv.empty else np.nan,
                'put_mean_iv': float(put_iv.mean()) if not put_iv.empty else np.nan,
                'iv_skew': float(put_iv.mean() - call_iv.mean()) if not (call_iv.empty or put_iv.empty) else np.nan
            }
            
            # IV term structure (IV by expiry)
            iv_term_structure = valid_iv.groupby('expiration')['iv'].mean().to_dict()
            iv_term_structure = {str(k): float(v) for k, v in iv_term_structure.items()}
            
            # IV smile (IV by moneyness)
            # Group by moneyness buckets
            moneyness_buckets = np.linspace(0.5, 2.0, 16)  # 15 buckets from 0.5 to 2.0
            valid_iv['moneyness_bucket'] = pd.cut(valid_iv['moneyness'], moneyness_buckets)
            iv_smile = valid_iv.groupby('moneyness_bucket')['iv'].mean().to_dict()
            
            # Convert to serializable format
            iv_smile_serializable = {}
            for k, v in iv_smile.items():
                if pd.isna(v):
                    continue
                # Convert pandas Interval to string representation
                bucket_str = f"{k.left:.2f}-{k.right:.2f}"
                iv_smile_serializable[bucket_str] = float(v)
            
            # IV surface (IV by moneyness and expiry)
            iv_surface = valid_iv.pivot_table(
                values='iv',
                index='moneyness_bucket',
                columns='expiration',
                aggfunc='mean'
            )
            
            # Convert to serializable format (nested dict)
            iv_surface_serializable = {}
            for moneyness_bucket in iv_surface.index:
                if pd.isna(moneyness_bucket):
                    continue
                bucket_str = f"{moneyness_bucket.left:.2f}-{moneyness_bucket.right:.2f}"
                iv_surface_serializable[bucket_str] = {}
                
                for expiry in iv_surface.columns:
                    iv_value = iv_surface.loc[moneyness_bucket, expiry]
                    if not pd.isna(iv_value):
                        iv_surface_serializable[bucket_str][str(expiry)] = float(iv_value)
            
            return {
                **iv_stats,
                **iv_by_type,
                'iv_term_structure': iv_term_structure,
                'iv_smile': iv_smile_serializable,
                'iv_surface': iv_surface_serializable
            }
            
        except Exception as e:
            logger.error(f"Error analyzing implied volatility: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'mean_iv': np.nan,
                'median_iv': np.nan,
                'error': str(e)
            }
    
    def _analyze_greeks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze option Greeks.
        
        Args:
            df: Processed options DataFrame
            
        Returns:
            Dictionary with Greeks analysis results
        """
        try:
            # Check if Greeks data is available
            greek_columns = ['delta', 'gamma', 'vega', 'theta', 'rho']
            available_greeks = [col for col in greek_columns if col in df.columns]
            
            if not available_greeks:
                logger.warning("No Greeks data available for analysis")
                return {
                    'available_greeks': [],
                    'greeks_stats': {}
                }
            
            # Filter out invalid values
            valid_greeks = df.dropna(subset=available_greeks)
            
            if valid_greeks.empty:
                logger.warning("No valid Greeks data after filtering")
                return {
                    'available_greeks': available_greeks,
                    'greeks_stats': {}
                }
            
            # Basic Greeks statistics
            greeks_stats = {}
            for greek in available_greeks:
                greeks_stats[f"{greek}_mean"] = float(valid_greeks[greek].mean())
                greeks_stats[f"{greek}_median"] = float(valid_greeks[greek].median())
                greeks_stats[f"{greek}_min"] = float(valid_greeks[greek].min())
                greeks_stats[f"{greek}_max"] = float(valid_greeks[greek].max())
                greeks_stats[f"{greek}_std"] = float(valid_greeks[greek].std())
            
            # Greeks by option type
            greeks_by_type = {}
            for greek in available_greeks:
                call_greek = valid_greeks[valid_greeks['option_type'].str.lower() == 'call'][greek]
                put_greek = valid_greeks[valid_greeks['option_type'].str.lower() == 'put'][greek]
                
                greeks_by_type[f"call_{greek}_mean"] = float(call_greek.mean()) if not call_greek.empty else np.nan
                greeks_by_type[f"put_{greek}_mean"] = float(put_greek.mean()) if not put_greek.empty else np.nan
            
            # Greeks by moneyness
            # Group by moneyness buckets
            moneyness_buckets = np.linspace(0.5, 2.0, 16)  # 15 buckets from 0.5 to 2.0
            valid_greeks['moneyness_bucket'] = pd.cut(valid_greeks['moneyness'], moneyness_buckets)
            
            greeks_by_moneyness = {}
            for greek in available_greeks:
                greek_by_moneyness = valid_greeks.groupby('moneyness_bucket')[greek].mean().to_dict()
                
                # Convert to serializable format
                greek_by_moneyness_serializable = {}
                for k, v in greek_by_moneyness.items():
                    if pd.isna(v):
                        continue
                    # Convert pandas Interval to string representation
                    bucket_str = f"{k.left:.2f}-{k.right:.2f}"
                    greek_by_moneyness_serializable[bucket_str] = float(v)
                
                greeks_by_moneyness[greek] = greek_by_moneyness_serializable
            
            # Greeks by expiry
            greeks_by_expiry = {}
            for greek in available_greeks:
                greek_by_expiry = valid_greeks.groupby('expiration')[greek].mean().to_dict()
                greeks_by_expiry[greek] = {str(k): float(v) for k, v in greek_by_expiry.items()}
            
            # Total portfolio Greeks (weighted by volume if available)
            portfolio_greeks = {}
            weight_col = 'volume' if 'volume' in valid_greeks.columns else None
            
            if weight_col is not None and not valid_greeks[weight_col].isnull().all():
                for greek in available_greeks:
                    portfolio_greeks[f"portfolio_{greek}"] = float(
                        (valid_greeks[greek] * valid_greeks[weight_col]).sum() / valid_greeks[weight_col].sum()
                    )
            else:
                for greek in available_greeks:
                    portfolio_greeks[f"portfolio_{greek}"] = float(valid_greeks[greek].mean())
            
            return {
                'available_greeks': available_greeks,
                'greeks_stats': greeks_stats,
                'greeks_by_type': greeks_by_type,
                'greeks_by_moneyness': greeks_by_moneyness,
                'greeks_by_expiry': greeks_by_expiry,
                'portfolio_greeks': portfolio_greeks
            }
            
        except Exception as e:
            logger.error(f"Error analyzing Greeks: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                'available_greeks': [],
                'greeks_stats': {},
                'error': str(e)
            }
    
    def analyze_all_symbols(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze options for all symbols.
        
        Returns:
            Dictionary mapping symbols to analysis results
        """
        logger.info(f"Analyzing options for {len(self.options_data)} symbols")
        
        for symbol in self.options_data.keys():
            try:
                if symbol not in self.price_data:
                    logger.warning(f"Missing price data for {symbol}, skipping")
                    continue
                    
                self.analyze_options(symbol)
                
            except Exception as e:
                logger.error(f"Error analyzing options for {symbol}: {str(e)}")
                self.results[symbol] = {}
        
        return self.results
    
    def get_option_chain(
        self, 
        symbol: str,
        expiry_date: Optional[Union[str, date]] = None,
        moneyness_range: Optional[Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Get option chain for a specific symbol and expiry date.
        
        Args:
            symbol: Symbol to get option chain for
            expiry_date: Specific expiry date to filter
            moneyness_range: Moneyness range to filter (min, max)
            
        Returns:
            Option chain as DataFrame
        """
        if symbol not in self.results:
            logger.warning(f"No analysis results for {symbol}, analyzing now")
            self.analyze_options(symbol)
        
        if symbol not in self.results or 'options_data' not in self.results[symbol]:
            logger.warning(f"No options data available for {symbol}")
            return pd.DataFrame()
        
        options_data = self.results[symbol]['options_data']
        
        # Filter by expiry date if specified
        if expiry_date is not None:
            if isinstance(expiry_date, str):
                expiry_date = pd.to_datetime(expiry_date).date()
                
            if isinstance(options_data['expiration'].iloc[0], pd.Timestamp):
                options_data = options_data[options_data['expiration'].dt.date == expiry_date]
            else:
                options_data = options_data[options_data['expiration'] == expiry_date]
        
        # Filter by moneyness range if specified
        if moneyness_range is not None:
            min_moneyness, max_moneyness = moneyness_range
            options_data = options_data[
                (options_data['moneyness'] >= min_moneyness) & 
                (options_data['moneyness'] <= max_moneyness)
            ]
        
        # Sort by strike price
        if not options_data.empty:
            options_data = options_data.sort_values(by=['strike'])
        
        return options_data
    
    def calculate_strategy_metrics(
        self,
        symbol: str,
        legs: List[Dict[str, Any]],
        current_price: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate metrics for an options strategy.
        
        Args:
            symbol: Symbol for the strategy
            legs: List of strategy legs (e.g., [{'option_id': 1, 'quantity': 1}])
            current_price: Current price (uses latest from price_data if None)
            
        Returns:
            Dictionary with strategy metrics
        """
        if symbol not in self.results:
            logger.warning(f"No analysis results for {symbol}, analyzing now")
            self.analyze_options(symbol)
        
        if symbol not in self.results or 'options_data' not in self.results[symbol]:
            logger.warning(f"No options data available for {symbol}")
            return {}
        
        options_data = self.results[symbol]['options_data']
        
        if current_price is None:
            if symbol not in self.price_data or self.price_data[symbol].empty:
                logger.error(f"No price data available for {symbol}")
                return {}
            current_price = self.price_data[symbol]['Close'].iloc[-1]
        
        strategy_metrics = {
            'symbol': symbol,
            'current_price': current_price,
            'legs': [],
            'total_cost': 0.0,
            'total_delta': 0.0,
            'total_gamma': 0.0,
            'total_vega': 0.0,
            'total_theta': 0.0
        }
        
        for leg in legs:
            option_id = leg.get('option_id')
            strike = leg.get('strike')
            expiry = leg.get('expiry')
            option_type = leg.get('option_type')
            quantity = leg.get('quantity', 1)
            
            # Find matching option
            if option_id is not None:
                option = options_data.iloc[option_id].to_dict() if option_id < len(options_data) else None
            elif strike is not None and expiry is not None and option_type is not None:
                # Match by strike, expiry, and type
                matches = options_data[
                    (options_data['strike'] == strike) & 
                    (options_data['expiration'] == pd.to_datetime(expiry)) & 
                    (options_data['option_type'].str.lower() == option_type.lower())
                ]
                option = matches.iloc[0].to_dict() if not matches.empty else None
            else:
                logger.error(f"Insufficient details to identify option: {leg}")
                continue
            
            if option is None:
                logger.error(f"No matching option found for leg: {leg}")
                continue
            
            # Include all leg details
            leg_details = {
                'strike': option['strike'],
                'expiry': option['expiration'].strftime('%Y-%m-%d') if isinstance(option['expiration'], pd.Timestamp) else str(option['expiration']),
                'option_type': option['option_type'],
                'quantity': quantity,
                'price': option.get('last_price') or option.get('mark_price') or 0.0,
                'iv': option.get('iv', np.nan),
                'delta': option.get('delta', np.nan),
                'gamma': option.get('gamma', np.nan),
                'vega': option.get('vega', np.nan),
                'theta': option.get('theta', np.nan)
            }
            
            # Add to strategy totals
            sign = quantity if option['option_type'].lower() == 'call' else -quantity
            strategy_metrics['total_cost'] += leg_details['price'] * abs(quantity)
            
            for greek in ['delta', 'gamma', 'vega', 'theta']:
                if greek in option and not pd.isna(option[greek]):
                    strategy_metrics[f'total_{greek}'] += option[greek] * sign
            
            strategy_metrics['legs'].append(leg_details)
        
        # Calculate potential payoffs at various price levels
        price_range = np.linspace(current_price * 0.7, current_price * 1.3, 61)
        payoffs = []
        
        for price in price_range:
            payoff = -strategy_metrics['total_cost']  # Initial cost
            
            for leg in strategy_metrics['legs']:
                quantity = leg['quantity']
                strike = leg['strike']
                option_type = leg['option_type'].lower()
                
                # Calculate intrinsic value at expiry
                if option_type == 'call':
                    intrinsic = max(0, price - strike)
                else:  # put
                    intrinsic = max(0, strike - price)
                
                payoff += intrinsic * quantity
            
            payoffs.append({
                'price': float(price),
                'payoff': float(payoff)
            })
        
        strategy_metrics['payoff_curve'] = payoffs
        
        # Calculate breakeven points
        if payoffs[0]['payoff'] > 0 and payoffs[-1]['payoff'] > 0:
            # Always profitable
            strategy_metrics['breakeven_points'] = ["Always profitable"]
        elif payoffs[0]['payoff'] < 0 and payoffs[-1]['payoff'] < 0:
            # Always unprofitable
            strategy_metrics['breakeven_points'] = ["Never profitable"]
        else:
            # Find where payoff crosses zero
            breakeven_points = []
            for i in range(1, len(payoffs)):
                if (payoffs[i-1]['payoff'] <= 0 and payoffs[i]['payoff'] >= 0) or \
                   (payoffs[i-1]['payoff'] >= 0 and payoffs[i]['payoff'] <= 0):
                    # Linear interpolation to find breakeven
                    p1 = payoffs[i-1]
                    p2 = payoffs[i]
                    
                    # Avoid division by zero
                    if p2['payoff'] == p1['payoff']:
                        continue
                    
                    # Calculate breakeven price
                    breakeven = p1['price'] + (p2['price'] - p1['price']) * (0 - p1['payoff']) / (p2['payoff'] - p1['payoff'])
                    breakeven_points.append(float(breakeven))
            
            strategy_metrics['breakeven_points'] = breakeven_points
        
        # Calculate risk/reward metrics
        if payoffs:
            strategy_metrics['max_profit'] = float(max(p['payoff'] for p in payoffs))
            strategy_metrics['max_loss'] = float(min(p['payoff'] for p in payoffs))
            
            # Avoid division by zero
            if strategy_metrics['max_loss'] != 0:
                strategy_metrics['risk_reward_ratio'] = abs(strategy_metrics['max_profit'] / strategy_metrics['max_loss'])
            else:
                strategy_metrics['risk_reward_ratio'] = float('inf')
        
        return strategy_metrics
    
    def cleanup(self) -> None:
        """Clean up resources and memory."""
        logger.debug("Cleaning up resources")
        self.options_data.clear()
        self.results.clear()
        gc.collect()  # Force garbage collection