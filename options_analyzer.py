import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import norm
import warnings

class OptionsAnalyzer:
    def __init__(
        self,
        options_data: Dict[str, pd.DataFrame],
        price_data: Dict[str, pd.DataFrame],
        risk_free_rate: float = 0.05,
        numerical_precision: int = 6
    ) -> None:
        """Initialize options analyzer with options and price data."""
        self.options_data = options_data
        self.price_data = price_data
        self.risk_free_rate = risk_free_rate
        self.numerical_precision = numerical_precision
        self.results = {}
        self._validate_inputs()
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.options_data:
            raise ValueError("No options data provided")
            
        if not self.price_data:
            raise ValueError("No price data provided")
            
        if self.risk_free_rate < 0:
            raise ValueError("risk_free_rate must be non-negative")
            
        if self.numerical_precision < 0:
            raise ValueError("numerical_precision must be non-negative")
    
    def _calculate_black_scholes(
        self,
        S: float,  # Current stock price
        K: float,  # Strike price
        T: float,  # Time to expiration in years
        sigma: float,  # Volatility
        r: float,  # Risk-free rate
        option_type: str = 'call'
    ) -> Dict[str, float]:
        """Calculate option price and Greeks using Black-Scholes model."""
        # Input validation
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or r < 0:
            raise ValueError("Invalid input parameters for Black-Scholes calculation")
        
        # Calculate d1 and d2
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Calculate option price
        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1) / 100  # Divided by 100 to express in terms of 1% change
            theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            vega = S * np.sqrt(T) * norm.pdf(d1) / 100
            theta = (-S * sigma * norm.pdf(d1)) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
        
        # Round results to specified precision
        return {
            'price': round(price, self.numerical_precision),
            'delta': round(delta, self.numerical_precision),
            'gamma': round(gamma, self.numerical_precision),
            'vega': round(vega, self.numerical_precision),
            'theta': round(theta, self.numerical_precision)
        }
    
    def _calculate_implied_volatility(
        self,
        market_price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: str = 'call',
        initial_guess: float = 0.5,
        max_iterations: int = 100,
        tolerance: float = 1e-5
    ) -> float:
        """Calculate implied volatility using Newton-Raphson method."""
        # Input validation
        if market_price <= 0 or S <= 0 or K <= 0 or T <= 0 or r < 0:
            raise ValueError("Invalid input parameters for implied volatility calculation")
        
        sigma = initial_guess
        for i in range(max_iterations):
            # Calculate option price and vega
            bs_result = self._calculate_black_scholes(S, K, T, sigma, r, option_type)
            price = bs_result['price']
            vega = bs_result['vega']
            
            # Calculate price difference
            price_diff = price - market_price
            
            # Check convergence
            if abs(price_diff) < tolerance:
                return round(sigma, self.numerical_precision)
            
            # Update sigma using Newton-Raphson
            sigma = sigma - price_diff / (vega * 100)  # Multiply by 100 because vega is in terms of 1%
            
            # Ensure sigma stays within reasonable bounds
            sigma = max(0.0001, min(sigma, 5.0))
        
        warnings.warn(f"Implied volatility calculation did not converge after {max_iterations} iterations")
        return round(sigma, self.numerical_precision)
    
    def analyze_options(self, symbol: str) -> pd.DataFrame:
        """Analyze options data for a given symbol."""
        if symbol not in self.options_data or symbol not in self.price_data:
            raise ValueError(f"Data not available for {symbol}")
        
        # Get current price
        current_price = self.price_data[symbol]['Close'].iloc[-1]
        
        # Filter and process options data
        options_df = self._filter_options_data(self.options_data[symbol], current_price)
        
        if options_df.empty:
            print(f"No valid options data for {symbol}")
            return pd.DataFrame()
        
        # Calculate implied volatility and Greeks
        options_df = self._calculate_implied_volatility_and_greeks(options_df, current_price)
        
        # Analyze volumes
        volume_analysis = self._analyze_volumes(options_df)
        
        # Analyze implied volatility
        iv_analysis = self._analyze_implied_volatility(options_df)
        
        # Analyze Greeks
        greeks_analysis = self._analyze_greeks(options_df)
        
        # Store results
        self.results[symbol] = {
            'options_data': options_df,
            'volume_analysis': volume_analysis,
            'iv_analysis': iv_analysis,
            'greeks_analysis': greeks_analysis
        }
        
        return options_df
    
    def _filter_options_data(self, df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """Filter and clean options data."""
        # Remove invalid data
        df = df.dropna(subset=['strike', 'expiration', 'option_type', 'last_price'])
        
        # Convert expiration to datetime
        df['expiration'] = pd.to_datetime(df['expiration'])
        
        # Calculate time to expiration in years
        df['time_to_expiry'] = (df['expiration'] - pd.Timestamp.now()).dt.total_seconds() / (365 * 24 * 3600)
        
        # Filter out expired options
        df = df[df['time_to_expiry'] > 0]
        
        # Filter out options with invalid prices
        df = df[df['last_price'] > 0]
        
        # Filter out options with invalid strikes
        df = df[df['strike'] > 0]
        
        # Add moneyness
        df['moneyness'] = df['strike'] / current_price
        
        return df
    
    def _calculate_implied_volatility_and_greeks(self, df: pd.DataFrame, current_price: float) -> pd.DataFrame:
        """Calculate implied volatility and Greeks for options."""
        for idx, row in df.iterrows():
            try:
                # Calculate implied volatility
                iv = self._calculate_implied_volatility(
                    market_price=row['last_price'],
                    S=current_price,
                    K=row['strike'],
                    T=row['time_to_expiry'],
                    r=self.risk_free_rate,
                    option_type=row['option_type']
                )
                
                # Calculate Greeks
                greeks = self._calculate_black_scholes(
                    S=current_price,
                    K=row['strike'],
                    T=row['time_to_expiry'],
                    sigma=iv,
                    r=self.risk_free_rate,
                    option_type=row['option_type']
                )
                
                # Update DataFrame
                df.at[idx, 'implied_volatility'] = iv
                df.at[idx, 'delta'] = greeks['delta']
                df.at[idx, 'gamma'] = greeks['gamma']
                df.at[idx, 'vega'] = greeks['vega']
                df.at[idx, 'theta'] = greeks['theta']
                
            except Exception as e:
                print(f"Error calculating Greeks for option {idx}: {e}")
                continue
        
        return df
    
    def _analyze_volumes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze trading volumes."""
        volume_analysis = {
            'total_volume': df['volume'].sum(),
            'call_volume': df[df['option_type'] == 'call']['volume'].sum(),
            'put_volume': df[df['option_type'] == 'put']['volume'].sum(),
            'volume_ratio': df[df['option_type'] == 'call']['volume'].sum() / 
                          df[df['option_type'] == 'put']['volume'].sum(),
            'most_active_strikes': df.groupby('strike')['volume'].sum().nlargest(5).to_dict(),
            'most_active_expirations': df.groupby('expiration')['volume'].sum().nlargest(5).to_dict()
        }
        
        return volume_analysis
    
    def _analyze_implied_volatility(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze implied volatility patterns."""
        iv_analysis = {
            'mean_iv': df['implied_volatility'].mean(),
            'std_iv': df['implied_volatility'].std(),
            'iv_skew': self._calculate_iv_skew(df),
            'iv_term_structure': self._calculate_iv_term_structure(df),
            'iv_by_moneyness': self._calculate_iv_by_moneyness(df)
        }
        
        return iv_analysis
    
    def _analyze_greeks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Greeks patterns."""
        greeks_analysis = {
            'mean_delta': df['delta'].mean(),
            'mean_gamma': df['gamma'].mean(),
            'mean_vega': df['vega'].mean(),
            'mean_theta': df['theta'].mean(),
            'greeks_by_moneyness': self._calculate_greeks_by_moneyness(df),
            'greeks_by_expiry': self._calculate_greeks_by_expiry(df)
        }
        
        return greeks_analysis
    
    def _calculate_iv_skew(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate implied volatility skew."""
        # Group by moneyness and calculate mean IV
        iv_by_moneyness = df.groupby(pd.qcut(df['moneyness'], 5))['implied_volatility'].mean()
        return iv_by_moneyness.to_dict()
    
    def _calculate_iv_term_structure(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate implied volatility term structure."""
        # Group by expiration and calculate mean IV
        iv_by_expiry = df.groupby('expiration')['implied_volatility'].mean()
        return iv_by_expiry.to_dict()
    
    def _calculate_iv_by_moneyness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate implied volatility by moneyness."""
        # Group by moneyness ranges and calculate mean IV
        moneyness_ranges = pd.qcut(df['moneyness'], 5)
        iv_by_moneyness = df.groupby(moneyness_ranges)['implied_volatility'].mean()
        return iv_by_moneyness.to_dict()
    
    def _calculate_greeks_by_moneyness(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate Greeks by moneyness."""
        # Group by moneyness ranges and calculate mean Greeks
        moneyness_ranges = pd.qcut(df['moneyness'], 5)
        greeks_by_moneyness = df.groupby(moneyness_ranges)[['delta', 'gamma', 'vega', 'theta']].mean()
        return greeks_by_moneyness.to_dict()
    
    def _calculate_greeks_by_expiry(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Calculate Greeks by expiration."""
        # Group by expiration and calculate mean Greeks
        greeks_by_expiry = df.groupby('expiration')[['delta', 'gamma', 'vega', 'theta']].mean()
        return greeks_by_expiry.to_dict()
    
    def analyze_all_symbols(self) -> Dict[str, Dict[str, Any]]:
        """Analyze options for all symbols."""
        for symbol in self.options_data.keys():
            try:
                self.analyze_options(symbol)
                print(f"Completed options analysis for {symbol}")
            except Exception as e:
                print(f"Error analyzing options for {symbol}: {e}")
                self.results[symbol] = {}
        
        return self.results 