"""
Visualization module for crypto volatility and options analysis.
Creates interactive charts and visualization dashboards for analysis results.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import os
import json
import logging
import traceback
import gc

from config import Config, logger

class VisualizationError(Exception):
    """Exception raised for errors in the visualization process."""
    pass


class CryptoVolatilityOptionsVisualizer:
    """
    Comprehensive visualization system for crypto volatility and options analysis.
    """
    
    def __init__(
        self,
        price_results: Dict[str, pd.DataFrame] = None,
        options_results: Dict[str, Dict[str, Any]] = None,
        term_structures: Dict[str, Dict[str, float]] = None,
        volatility_cones: Dict[str, Dict[str, float]] = None,
        output_dir: Optional[str] = None,
        color_scheme: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Initialize visualizer with analysis results.
        
        Args:
            price_results: Volatility analysis results from CryptoVolatilityAnalyzer
            options_results: Options analysis results from OptionsAnalyzer
            term_structures: Term structure data for options
            volatility_cones: Volatility cone data
            output_dir: Directory for saving visualizations
            color_scheme: Custom color scheme for visualizations
        """
        self.price_results = price_results or {}
        self.options_results = options_results or {}
        self.term_structures = term_structures or {}
        self.volatility_cones = volatility_cones or {}
        self.output_dir = Path(output_dir) if output_dir else Config.RESULTS_DIR
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up color scheme
        self.color_scheme = color_scheme or {
            'primary': '#1f77b4',    # Blue
            'secondary': '#ff7f0e',  # Orange
            'tertiary': '#2ca02c',   # Green
            'quaternary': '#d62728', # Red
            'positive': '#2ca02c',   # Green
            'negative': '#d62728',   # Red
            'neutral': '#7f7f7f',    # Gray
            'background': '#f8f9fa', # Light gray
            'grid': '#e6e6e6',       # Light gray
            'text': '#333333'        # Dark gray
        }
        
        # Set up figure templates
        self.plotly_template = go.layout.Template(
            layout=go.Layout(
                font=dict(family="Arial, sans-serif", size=12, color=self.color_scheme['text']),
                paper_bgcolor=self.color_scheme['background'],
                plot_bgcolor=self.color_scheme['background'],
                xaxis=dict(
                    gridcolor=self.color_scheme['grid'],
                    linecolor=self.color_scheme['grid'],
                    zerolinecolor=self.color_scheme['grid']
                ),
                yaxis=dict(
                    gridcolor=self.color_scheme['grid'],
                    linecolor=self.color_scheme['grid'],
                    zerolinecolor=self.color_scheme['grid']
                ),
                legend=dict(
                    bgcolor=self.color_scheme['background'],
                    bordercolor=self.color_scheme['grid'],
                    borderwidth=1
                )
            )
        )
        
        logger.info(f"Initialized CryptoVolatilityOptionsVisualizer with output directory: {self.output_dir}")
    
    def plot_price_volatility(
        self,
        symbol: str,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        show_anomalies: bool = True,
        height: int = 600,
        width: int = 1000
    ) -> go.Figure:
        """
        Create a plot of price and volatility for a symbol.
        
        Args:
            symbol: Symbol to plot
            start_date: Start date for the plot
            end_date: End date for the plot
            show_anomalies: Whether to highlight anomalies
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        try:
            if symbol not in self.price_results:
                logger.error(f"No price results available for {symbol}")
                raise VisualizationError(f"No price results available for {symbol}")
            
            df = self.price_results[symbol]
            
            if df.empty:
                logger.error(f"Empty price results for {symbol}")
                raise VisualizationError(f"Empty price results for {symbol}")
            
            # Filter by date if specified
            if start_date is not None or end_date is not None:
                df = self._filter_dataframe_by_date(df, start_date, end_date)
                if df.empty:
                    logger.error(f"No data available in date range for {symbol}")
                    raise VisualizationError(f"No data available in date range for {symbol}")
            
            # Create figure with secondary y-axis
            fig = make_subplots(
                rows=2, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                subplot_titles=(f"{symbol} Price", f"{symbol} Volatility"),
                row_heights=[0.7, 0.3]
            )
            
            # Add price trace
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color=self.color_scheme['primary'], width=1.5)
                ),
                row=1, col=1
            )
            
            # Add price rolling mean
            if 'Price_Rolling_Mean' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['Price_Rolling_Mean'],
                        mode='lines',
                        name='Price MA',
                        line=dict(color=self.color_scheme['secondary'], width=1, dash='dash')
                    ),
                    row=1, col=1
                )
            
            # Add volatility trace
            if 'RV_Composite' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df['RV_Composite'] * 100,  # Convert to percentage
                        mode='lines',
                        name='Volatility (%)',
                        line=dict(color=self.color_scheme['tertiary'], width=1.5)
                    ),
                    row=2, col=1
                )
            
            # Highlight anomalies if requested
            if show_anomalies and 'Combined_Anomaly' in df.columns:
                anomalies = df[df['Combined_Anomaly'] == 1]
                
                if not anomalies.empty:
                    # Price anomalies
                    fig.add_trace(
                        go.Scatter(
                            x=anomalies.index,
                            y=anomalies['Close'],
                            mode='markers',
                            name='Price Anomalies',
                            marker=dict(
                                color=self.color_scheme['quaternary'],
                                size=8,
                                symbol='circle',
                                line=dict(width=1, color=self.color_scheme['background'])
                            )
                        ),
                        row=1, col=1
                    )
                    
                    # Volatility anomalies
                    if 'RV_Composite' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=anomalies.index,
                                y=anomalies['RV_Composite'] * 100,  # Convert to percentage
                                mode='markers',
                                name='Volatility Anomalies',
                                marker=dict(
                                    color=self.color_scheme['quaternary'],
                                    size=8,
                                    symbol='circle',
                                    line=dict(width=1, color=self.color_scheme['background'])
                                )
                            ),
                            row=2, col=1
                        )
            
            # Update layout
            fig.update_layout(
                template=self.plotly_template,
                height=height,
                width=width,
                title=dict(
                    text=f"{symbol} Price and Volatility Analysis",
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Update axes
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price volatility plot for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create price volatility plot for {symbol}: {str(e)}")
    
    def plot_options_volume(
        self,
        symbol: str,
        height: int = 600,
        width: int = 1000
    ) -> go.Figure:
        """
        Create a plot of options trading volume.
        
        Args:
            symbol: Symbol to plot
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        try:
            if symbol not in self.options_results:
                logger.error(f"No options results available for {symbol}")
                raise VisualizationError(f"No options results available for {symbol}")
            
            results = self.options_results[symbol]
            
            if 'volume_analysis' not in results:
                logger.error(f"No volume analysis available for {symbol}")
                raise VisualizationError(f"No volume analysis available for {symbol}")
            
            volume_analysis = results['volume_analysis']
            
            # Create figure with subplots
            fig = make_subplots(
                rows=2, 
                cols=2,
                subplot_titles=(
                    f"{symbol} Call/Put Volume", 
                    "Volume by Expiry", 
                    "Volume by Strike", 
                    "Volume by Moneyness Category"
                ),
                specs=[
                    [{"type": "bar"}, {"type": "bar"}],
                    [{"type": "bar"}, {"type": "bar"}]
                ]
            )
            
            # Add call/put volume
            call_volume = volume_analysis.get('call_volume', 0)
            put_volume = volume_analysis.get('put_volume', 0)
            
            fig.add_trace(
                go.Bar(
                    x=['Calls', 'Puts'],
                    y=[call_volume, put_volume],
                    marker_color=[self.color_scheme['positive'], self.color_scheme['negative']],
                    text=[f"{call_volume:,.0f}", f"{put_volume:,.0f}"],
                    textposition='auto',
                    name='Volume'
                ),
                row=1, col=1
            )
            
            # Add volume by expiry
            volume_by_expiry = volume_analysis.get('volume_by_expiry', {})
            
            if volume_by_expiry:
                expiries = []
                volumes = []
                
                # Convert to lists and sort by expiry
                for expiry, volume in sorted(volume_by_expiry.items()):
                    if isinstance(expiry, str):
                        try:
                            # Try to parse date string
                            expiry_date = pd.to_datetime(expiry).strftime('%Y-%m-%d')
                        except:
                            expiry_date = expiry
                    else:
                        expiry_date = str(expiry)
                    
                    expiries.append(expiry_date)
                    volumes.append(volume)
                
                fig.add_trace(
                    go.Bar(
                        x=expiries,
                        y=volumes,
                        marker_color=self.color_scheme['secondary'],
                        name='Volume by Expiry'
                    ),
                    row=1, col=2
                )
            
            # Add volume by strike
            volume_by_strike = volume_analysis.get('volume_by_strike', {})
            
            if volume_by_strike:
                strikes = []
                volumes = []
                
                # Convert to lists and sort by strike
                for strike, volume in sorted(volume_by_strike.items(), key=lambda x: float(x[0]) if isinstance(x[0], str) else x[0]):
                    strikes.append(strike)
                    volumes.append(volume)
                
                fig.add_trace(
                    go.Bar(
                        x=strikes,
                        y=volumes,
                        marker_color=self.color_scheme['tertiary'],
                        name='Volume by Strike'
                    ),
                    row=2, col=1
                )
            
            # Add volume by moneyness
            volume_by_moneyness = volume_analysis.get('volume_by_moneyness', {})
            
            if volume_by_moneyness:
                categories = []
                volumes = []
                
                # Define the order of moneyness categories
                order = ['Deep ITM', 'ITM', 'ATM', 'OTM', 'Deep OTM']
                
                # Filter to only categories in our defined order
                for category in order:
                    if category in volume_by_moneyness:
                        categories.append(category)
                        volumes.append(volume_by_moneyness[category])
                
                # Add any other categories not in our defined order
                for category, volume in volume_by_moneyness.items():
                    if category not in order:
                        categories.append(category)
                        volumes.append(volume)
                
                fig.add_trace(
                    go.Bar(
                        x=categories,
                        y=volumes,
                        marker_color=self.color_scheme['primary'],
                        name='Volume by Moneyness'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                template=self.plotly_template,
                height=height,
                width=width,
                title=dict(
                    text=f"{symbol} Options Volume Analysis",
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=False
            )
            
            # Update x-axes
            fig.update_xaxes(title_text="Option Type", row=1, col=1)
            fig.update_xaxes(title_text="Expiry Date", row=1, col=2)
            fig.update_xaxes(title_text="Strike Price", row=2, col=1)
            fig.update_xaxes(title_text="Moneyness Category", row=2, col=2)
            
            # Update y-axes
            fig.update_yaxes(title_text="Volume", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=1, col=2)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=2)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating options volume plot for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create options volume plot for {symbol}: {str(e)}")
    
    def plot_iv_surface(
        self,
        symbol: str,
        height: int = 700,
        width: int = 1000
    ) -> go.Figure:
        """
        Create a 3D plot of the implied volatility surface.
        
        Args:
            symbol: Symbol to plot
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        try:
            if symbol not in self.options_results:
                logger.error(f"No options results available for {symbol}")
                raise VisualizationError(f"No options results available for {symbol}")
            
            results = self.options_results[symbol]
            
            if 'iv_analysis' not in results:
                logger.error(f"No IV analysis available for {symbol}")
                raise VisualizationError(f"No IV analysis available for {symbol}")
            
            iv_analysis = results['iv_analysis']
            
            if 'iv_surface' not in iv_analysis:
                logger.error(f"No IV surface data available for {symbol}")
                raise VisualizationError(f"No IV surface data available for {symbol}")
            
            iv_surface = iv_analysis['iv_surface']
            
            # Check if we have data
            if not iv_surface:
                logger.error(f"Empty IV surface data for {symbol}")
                raise VisualizationError(f"Empty IV surface data for {symbol}")
            
            # Create 3D surface plot
            fig = go.Figure()
            
            # Process IV surface data
            x_vals = []  # Moneyness
            y_vals = []  # Expiry
            z_vals = []  # IV
            
            for moneyness_bucket, expiries in iv_surface.items():
                for expiry, iv in expiries.items():
                    # Parse moneyness bucket
                    moneyness_parts = moneyness_bucket.split('-')
                    if len(moneyness_parts) == 2:
                        moneyness = (float(moneyness_parts[0]) + float(moneyness_parts[1])) / 2
                    else:
                        try:
                            moneyness = float(moneyness_bucket)
                        except:
                            moneyness = 1.0  # Default
                    
                    # Parse expiry
                    try:
                        expiry_date = pd.to_datetime(expiry)
                        days_to_expiry = (expiry_date - datetime.now()).days
                    except:
                        try:
                            days_to_expiry = float(expiry)
                        except:
                            days_to_expiry = 30  # Default
                    
                    x_vals.append(moneyness)
                    y_vals.append(days_to_expiry)
                    z_vals.append(iv * 100)  # Convert to percentage
            
            # Create scatter3d plot
            fig.add_trace(
                go.Scatter3d(
                    x=x_vals,
                    y=y_vals,
                    z=z_vals,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=z_vals,
                        colorscale='Viridis',
                        opacity=0.8,
                        showscale=True,
                        colorbar=dict(
                            title='IV (%)',
                            titleside='right'
                        )
                    ),
                    name='IV Surface'
                )
            )
            
            # Create a mesh3d surface if we have enough data points
            if len(x_vals) >= 6:
                try:
                    # Create a grid for the surface
                    from scipy.interpolate import griddata
                    
                    x_grid = np.linspace(min(x_vals), max(x_vals), 20)
                    y_grid = np.linspace(min(y_vals), max(y_vals), 20)
                    X, Y = np.meshgrid(x_grid, y_grid)
                    
                    # Interpolate the IV values
                    Z = griddata((x_vals, y_vals), z_vals, (X, Y), method='cubic')
                    
                    # Add surface
                    fig.add_trace(
                        go.Surface(
                            x=x_grid,
                            y=y_grid,
                            z=Z,
                            opacity=0.7,
                            colorscale='Viridis',
                            showscale=False,
                            name='IV Surface'
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not create surface mesh for {symbol}: {str(e)}")
            
            # Update layout
            fig.update_layout(
                template=self.plotly_template,
                height=height,
                width=width,
                title=dict(
                    text=f"{symbol} Implied Volatility Surface",
                    x=0.5,
                    xanchor='center'
                ),
                scene=dict(
                    xaxis_title='Moneyness (Strike/Price)',
                    yaxis_title='Days to Expiry',
                    zaxis_title='Implied Volatility (%)',
                    xaxis=dict(gridcolor=self.color_scheme['grid'], showbackground=True),
                    yaxis=dict(gridcolor=self.color_scheme['grid'], showbackground=True),
                    zaxis=dict(gridcolor=self.color_scheme['grid'], showbackground=True)
                )
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IV surface plot for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create IV surface plot for {symbol}: {str(e)}")
    
    def plot_iv_skew(
        self,
        symbol: str,
        height: int = 500,
        width: int = 800
    ) -> go.Figure:
        """
        Create a plot of implied volatility skew.
        
        Args:
            symbol: Symbol to plot
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        try:
            if symbol not in self.options_results:
                logger.error(f"No options results available for {symbol}")
                raise VisualizationError(f"No options results available for {symbol}")
            
            results = self.options_results[symbol]
            
            if 'iv_analysis' not in results or 'options_data' not in results:
                logger.error(f"No IV analysis or options data available for {symbol}")
                raise VisualizationError(f"No IV analysis or options data available for {symbol}")
            
            options_data = results['options_data']
            
            if options_data.empty or 'iv' not in options_data.columns:
                logger.error(f"No IV data available for {symbol}")
                raise VisualizationError(f"No IV data available for {symbol}")
            
            # Filter out invalid IVs
            df = options_data[~options_data['iv'].isna()].copy()
            
            if df.empty:
                logger.error(f"No valid IV data for {symbol}")
                raise VisualizationError(f"No valid IV data for {symbol}")
            
            # Create figure
            fig = go.Figure()
            
            # Group by expiry
            expiries = df['expiration'].unique()
            
            # Sort expiries
            if isinstance(expiries[0], pd.Timestamp):
                expiries = sorted(expiries)
            else:
                try:
                    expiries = sorted(pd.to_datetime(expiries))
                except:
                    # If we can't convert to datetime, use as is
                    expiries = sorted(expiries)
            
            # Take at most 5 expiries to avoid overcrowding
            if len(expiries) > 5:
                # Take first, two middle, and last expiry
                step = len(expiries) // 4
                expiries = [expiries[0], expiries[step], expiries[2*step], expiries[3*step], expiries[-1]]
            
            # Color map for different expiries
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm
            
            colormap = cm.get_cmap('viridis', len(expiries))
            colors = [mcolors.rgb2hex(colormap(i)) for i in range(len(expiries))]
            
            for i, expiry in enumerate(expiries):
                # Filter to expiry
                expiry_df = df[df['expiration'] == expiry]
                
                if expiry_df.empty:
                    continue
                
                # Sort by moneyness
                expiry_df = expiry_df.sort_values(by=['moneyness'])
                
                # Format the expiry for display
                if isinstance(expiry, pd.Timestamp):
                    expiry_str = expiry.strftime('%Y-%m-%d')
                else:
                    expiry_str = str(expiry)
                
                # Add trace for call options
                calls = expiry_df[expiry_df['option_type'].str.lower() == 'call']
                if not calls.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=calls['moneyness'],
                            y=calls['iv'] * 100,  # Convert to percentage
                            mode='lines+markers',
                            line=dict(color=colors[i]),
                            marker=dict(symbol='circle', size=6),
                            name=f"Calls - {expiry_str}"
                        )
                    )
                
                # Add trace for put options
                puts = expiry_df[expiry_df['option_type'].str.lower() == 'put']
                if not puts.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=puts['moneyness'],
                            y=puts['iv'] * 100,  # Convert to percentage
                            mode='lines+markers',
                            line=dict(color=colors[i], dash='dash'),
                            marker=dict(symbol='x', size=6),
                            name=f"Puts - {expiry_str}"
                        )
                    )
            
            # Update layout
            fig.update_layout(
                template=self.plotly_template,
                height=height,
                width=width,
                title=dict(
                    text=f"{symbol} Implied Volatility Skew",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title='Moneyness (Strike/Price)',
                    tickformat='.2f',
                    dtick=0.1
                ),
                yaxis=dict(
                    title='Implied Volatility (%)',
                    tickformat='.1f'
                ),
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add vertical line at moneyness = 1 (at-the-money)
            fig.add_shape(
                type='line',
                x0=1, y0=0,
                x1=1, y1=1,
                yref='paper',
                line=dict(
                    color=self.color_scheme['grid'],
                    width=1,
                    dash='dash'
                )
            )
            
            # Add annotation for ATM
            fig.add_annotation(
                x=1,
                y=1,
                yref='paper',
                text='ATM',
                showarrow=False,
                yanchor='bottom',
                bgcolor=self.color_scheme['background'],
                bordercolor=self.color_scheme['grid'],
                borderwidth=1,
                borderpad=4
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating IV skew plot for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create IV skew plot for {symbol}: {str(e)}")
    
    def plot_greeks_by_moneyness(
        self,
        symbol: str,
        greek: str = 'delta',
        height: int = 500,
        width: int = 800
    ) -> go.Figure:
        """
        Create a plot of option Greeks by moneyness.
        
        Args:
            symbol: Symbol to plot
            greek: Greek to plot ('delta', 'gamma', 'vega', 'theta', 'rho')
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        try:
            if symbol not in self.options_results:
                logger.error(f"No options results available for {symbol}")
                raise VisualizationError(f"No options results available for {symbol}")
            
            results = self.options_results[symbol]
            
            if 'options_data' not in results:
                logger.error(f"No options data available for {symbol}")
                raise VisualizationError(f"No options data available for {symbol}")
            
            options_data = results['options_data']
            
            greek = greek.lower()
            
            if greek not in ['delta', 'gamma', 'vega', 'theta', 'rho']:
                logger.error(f"Invalid Greek: {greek}")
                raise VisualizationError(f"Invalid Greek: {greek}")
            
            if greek not in options_data.columns:
                logger.error(f"No {greek} data available for {symbol}")
                raise VisualizationError(f"No {greek} data available for {symbol}")
            
            # Filter out invalid values
            df = options_data[~options_data[greek].isna()].copy()
            
            if df.empty:
                logger.error(f"No valid {greek} data for {symbol}")
                raise VisualizationError(f"No valid {greek} data for {symbol}")
            
            # Create figure
            fig = go.Figure()
            
            # Group by expiry
            expiries = df['expiration'].unique()
            
            # Sort expiries
            if isinstance(expiries[0], pd.Timestamp):
                expiries = sorted(expiries)
            else:
                try:
                    expiries = sorted(pd.to_datetime(expiries))
                except:
                    # If we can't convert to datetime, use as is
                    expiries = sorted(expiries)
            
            # Take at most 5 expiries to avoid overcrowding
            if len(expiries) > 5:
                # Take first, two middle, and last expiry
                step = len(expiries) // 4
                expiries = [expiries[0], expiries[step], expiries[2*step], expiries[3*step], expiries[-1]]
            
            # Color map for different expiries
            import matplotlib.colors as mcolors
            import matplotlib.cm as cm
            
            colormap = cm.get_cmap('viridis', len(expiries))
            colors = [mcolors.rgb2hex(colormap(i)) for i in range(len(expiries))]
            
            for i, expiry in enumerate(expiries):
                # Filter to expiry
                expiry_df = df[df['expiration'] == expiry]
                
                if expiry_df.empty:
                    continue
                
                # Sort by moneyness
                expiry_df = expiry_df.sort_values(by=['moneyness'])
                
                # Format the expiry for display
                if isinstance(expiry, pd.Timestamp):
                    expiry_str = expiry.strftime('%Y-%m-%d')
                else:
                    expiry_str = str(expiry)
                
                # Add trace for call options
                calls = expiry_df[expiry_df['option_type'].str.lower() == 'call']
                if not calls.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=calls['moneyness'],
                            y=calls[greek],
                            mode='lines+markers',
                            line=dict(color=colors[i]),
                            marker=dict(symbol='circle', size=6),
                            name=f"Calls - {expiry_str}"
                        )
                    )
                
                # Add trace for put options
                puts = expiry_df[expiry_df['option_type'].str.lower() == 'put']
                if not puts.empty:
                    fig.add_trace(
                        go.Scatter(
                            x=puts['moneyness'],
                            y=puts[greek],
                            mode='lines+markers',
                            line=dict(color=colors[i], dash='dash'),
                            marker=dict(symbol='x', size=6),
                            name=f"Puts - {expiry_str}"
                        )
                    )
            
            # Update layout
            fig.update_layout(
                template=self.plotly_template,
                height=height,
                width=width,
                title=dict(
                    text=f"{symbol} {greek.capitalize()} by Moneyness",
                    x=0.5,
                    xanchor='center'
                ),
                xaxis=dict(
                    title='Moneyness (Strike/Price)',
                    tickformat='.2f',
                    dtick=0.1
                ),
                yaxis=dict(
                    title=greek.capitalize()
                ),
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Add vertical line at moneyness = 1 (at-the-money)
            fig.add_shape(
                type='line',
                x0=1, y0=0,
                x1=1, y1=1,
                yref='paper',
                line=dict(
                    color=self.color_scheme['grid'],
                    width=1,
                    dash='dash'
                )
            )
            
            # Add annotation for ATM
            fig.add_annotation(
                x=1,
                y=1,
                yref='paper',
                text='ATM',
                showarrow=False,
                yanchor='bottom',
                bgcolor=self.color_scheme['background'],
                bordercolor=self.color_scheme['grid'],
                borderwidth=1,
                borderpad=4
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating {greek} plot for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create {greek} plot for {symbol}: {str(e)}")
    
    def plot_options_chain(
        self,
        symbol: str,
        expiry: Optional[Union[str, date]] = None,
        height: int = 600,
        width: int = 1000
    ) -> go.Figure:
        """
        Create a plot of the options chain for a specific expiry.
        
        Args:
            symbol: Symbol to plot
            expiry: Expiry date to filter
            height: Plot height
            width: Plot width
            
        Returns:
            Plotly figure
        """
        try:
            if symbol not in self.options_results:
                logger.error(f"No options results available for {symbol}")
                raise VisualizationError(f"No options results available for {symbol}")
            
            results = self.options_results[symbol]
            
            if 'options_data' not in results:
                logger.error(f"No options data available for {symbol}")
                raise VisualizationError(f"No options data available for {symbol}")
            
            options_data = results['options_data']
            
            if options_data.empty:
                logger.error(f"Empty options data for {symbol}")
                raise VisualizationError(f"Empty options data for {symbol}")
            
            # Filter by expiry if specified
            if expiry is not None:
                # Convert to datetime if it's a string
                if isinstance(expiry, str):
                    expiry = pd.to_datetime(expiry)
                
                # Filter
                if isinstance(options_data['expiration'].iloc[0], pd.Timestamp):
                    options_data = options_data[options_data['expiration'] == expiry]
                else:
                    # Try to convert expiration to datetime
                    try:
                        options_data['expiration'] = pd.to_datetime(options_data['expiration'])
                        options_data = options_data[options_data['expiration'] == expiry]
                    except:
                        logger.warning(f"Could not convert expiration to datetime for {symbol}")
                        # Try string comparison
                        options_data = options_data[options_data['expiration'].astype(str) == str(expiry)]
                
                if options_data.empty:
                    logger.error(f"No options data available for {symbol} with expiry {expiry}")
                    raise VisualizationError(f"No options data available for {symbol} with expiry {expiry}")
            else:
                # If no expiry specified, use the closest one
                if isinstance(options_data['expiration'].iloc[0], pd.Timestamp):
                    expiries = sorted(options_data['expiration'].unique())
                else:
                    # Try to convert expiration to datetime
                    try:
                        options_data['expiration'] = pd.to_datetime(options_data['expiration'])
                        expiries = sorted(options_data['expiration'].unique())
                    except:
                        logger.warning(f"Could not convert expiration to datetime for {symbol}")
                        # Use as is
                        expiries = sorted(options_data['expiration'].unique())
                
                # Use the closest expiry
                today = pd.Timestamp.now()
                future_expiries = [exp for exp in expiries if exp > today]
                
                if future_expiries:
                    expiry = future_expiries[0]
                    options_data = options_data[options_data['expiration'] == expiry]
                else:
                    # If no future expiries, use the latest one
                    expiry = expiries[-1]
                    options_data = options_data[options_data['expiration'] == expiry]
            
            # Format expiry for display
            if isinstance(expiry, pd.Timestamp):
                expiry_str = expiry.strftime('%Y-%m-%d')
            else:
                expiry_str = str(expiry)
            
            # Create figure with subplots
            fig = make_subplots(
                rows=3, 
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f"{symbol} Option Prices - {expiry_str}",
                    f"Implied Volatility",
                    f"Open Interest & Volume"
                ),
                row_heights=[0.4, 0.3, 0.3]
            )
            
            # Split by option type
            calls = options_data[options_data['option_type'].str.lower() == 'call']
            puts = options_data[options_data['option_type'].str.lower() == 'put']
            
            # Ensure they're sorted by strike
            calls = calls.sort_values(by=['strike'])
            puts = puts.sort_values(by=['strike'])
            
            # Option prices
            price_col = None
            for col in ['last_price', 'mark_price', 'mid_price', 'price']:
                if col in options_data.columns and not options_data[col].isnull().all():
                    price_col = col
                    break
            
            if price_col:
                fig.add_trace(
                    go.Scatter(
                        x=calls['strike'],
                        y=calls[price_col],
                        mode='lines+markers',
                        name='Call Price',
                        line=dict(color=self.color_scheme['primary']),
                        marker=dict(symbol='circle', size=6)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=puts['strike'],
                        y=puts[price_col],
                        mode='lines+markers',
                        name='Put Price',
                        line=dict(color=self.color_scheme['quaternary']),
                        marker=dict(symbol='x', size=6)
                    ),
                    row=1, col=1
                )
            
            # Implied volatility
            if 'iv' in options_data.columns and not options_data['iv'].isnull().all():
                fig.add_trace(
                    go.Scatter(
                        x=calls['strike'],
                        y=calls['iv'] * 100,  # Convert to percentage
                        mode='lines+markers',
                        name='Call IV',
                        line=dict(color=self.color_scheme['secondary']),
                        marker=dict(symbol='circle', size=6)
                    ),
                    row=2, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=puts['strike'],
                        y=puts['iv'] * 100,  # Convert to percentage
                        mode='lines+markers',
                        name='Put IV',
                        line=dict(color=self.color_scheme['tertiary']),
                        marker=dict(symbol='x', size=6)
                    ),
                    row=2, col=1
                )
            
            # Open interest and volume
            if 'open_interest' in options_data.columns and not options_data['open_interest'].isnull().all():
                fig.add_trace(
                    go.Bar(
                        x=calls['strike'],
                        y=calls['open_interest'],
                        name='Call OI',
                        marker_color=self.color_scheme['primary'],
                        opacity=0.7,
                        width=options_data['strike'].iloc[1] - options_data['strike'].iloc[0] if len(options_data['strike']) > 1 else 1
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=puts['strike'],
                        y=puts['open_interest'],
                        name='Put OI',
                        marker_color=self.color_scheme['quaternary'],
                        opacity=0.7,
                        width=options_data['strike'].iloc[1] - options_data['strike'].iloc[0] if len(options_data['strike']) > 1 else 1
                    ),
                    row=3, col=1
                )
            
            if 'volume' in options_data.columns and not options_data['volume'].isnull().all():
                fig.add_trace(
                    go.Scatter(
                        x=calls['strike'],
                        y=calls['volume'],
                        mode='lines',
                        name='Call Volume',
                        line=dict(color=self.color_scheme['primary'], dash='dash'),
                        opacity=0.9
                    ),
                    row=3, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=puts['strike'],
                        y=puts['volume'],
                        mode='lines',
                        name='Put Volume',
                        line=dict(color=self.color_scheme['quaternary'], dash='dash'),
                        opacity=0.9
                    ),
                    row=3, col=1
                )
            
            # Add current price line if available
            if 'current_price' in results:
                current_price = results['current_price']
                
                for row in range(1, 4):
                    fig.add_shape(
                        type='line',
                        x0=current_price, y0=0,
                        x1=current_price, y1=1,
                        yref='paper',
                        xref='x',
                        line=dict(
                            color=self.color_scheme['tertiary'],
                            width=1,
                            dash='dash'
                        ),
                        row=row, col=1
                    )
                
                # Add annotation for current price
                fig.add_annotation(
                    x=current_price,
                    y=1,
                    text=f"Current: {current_price:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=self.color_scheme['tertiary'],
                    arrowwidth=1,
                    arrowsize=1,
                    bgcolor=self.color_scheme['background'],
                    bordercolor=self.color_scheme['grid'],
                    borderwidth=1,
                    borderpad=4,
                    row=1, col=1
                )
            
            # Update layout
            fig.update_layout(
                template=self.plotly_template,
                height=height,
                width=width,
                title=dict(
                    text=f"{symbol} Options Chain - {expiry_str}",
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                barmode='group'
            )
            
            # Update axes
            fig.update_xaxes(title_text="Strike Price", row=3, col=1)
            fig.update_yaxes(title_text="Option Price", row=1, col=1)
            fig.update_yaxes(title_text="IV (%)", row=2, col=1)
            fig.update_yaxes(title_text="OI & Volume", row=3, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating options chain plot for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create options chain plot for {symbol}: {str(e)}")
    
    def create_comprehensive_dashboard(
        self,
        symbol: str,
        height: int = 1200,
        width: int = 1000,
        save_html: bool = True
    ) -> go.Figure:
        """
        Create a comprehensive dashboard for a symbol.
        
        Args:
            symbol: Symbol to create dashboard for
            height: Dashboard height
            width: Dashboard width
            save_html: Whether to save the dashboard as HTML
            
        Returns:
            Plotly figure
        """
        try:
            # Check if data is available
            has_price_data = symbol in self.price_results
            has_options_data = symbol in self.options_results
            
            if not has_price_data and not has_options_data:
                logger.error(f"No data available for {symbol}")
                raise VisualizationError(f"No data available for {symbol}")
            
            # Determine number of rows and subplots
            num_rows = 0
            row_heights = []
            specs = []
            titles = []
            
            # Price and volatility
            if has_price_data:
                num_rows += 2
                row_heights.extend([0.3, 0.15])
                specs.extend([{"type": "xy"}, {"type": "xy"}])
                titles.extend([f"{symbol} Price", f"{symbol} Volatility"])
            
            # Options data
            if has_options_data:
                # Options volume
                num_rows += 1
                row_heights.append(0.15)
                specs.append({"type": "xy"})
                titles.append(f"{symbol} Options Volume")
                
                # IV skew
                num_rows += 1
                row_heights.append(0.2)
                specs.append({"type": "xy"})
                titles.append(f"{symbol} IV Skew")
                
                # Greeks
                num_rows += 1
                row_heights.append(0.2)
                specs.append({"type": "xy"})
                titles.append(f"{symbol} Delta by Moneyness")
            
            # Create figure with subplots
            fig = make_subplots(
                rows=num_rows,
                cols=1,
                shared_xaxes=False,
                vertical_spacing=0.04,
                subplot_titles=titles,
                specs=specs,
                row_heights=row_heights
            )
            
            row = 1
            
            # Add price and volatility plots if available
            if has_price_data:
                # Get price data
                df = self.price_results[symbol]
                
                if not df.empty:
                    # Add price trace
                    fig.add_trace(
                        go.Scatter(
                            x=df.index,
                            y=df['Close'],
                            mode='lines',
                            name='Price',
                            line=dict(color=self.color_scheme['primary'], width=1.5)
                        ),
                        row=row, col=1
                    )
                    
                    # Add price rolling mean if available
                    if 'Price_Rolling_Mean' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['Price_Rolling_Mean'],
                                mode='lines',
                                name='Price MA',
                                line=dict(color=self.color_scheme['secondary'], width=1, dash='dash')
                            ),
                            row=row, col=1
                        )
                    
                    # Highlight anomalies if available
                    if 'Combined_Anomaly' in df.columns:
                        anomalies = df[df['Combined_Anomaly'] == 1]
                        
                        if not anomalies.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=anomalies.index,
                                    y=anomalies['Close'],
                                    mode='markers',
                                    name='Price Anomalies',
                                    marker=dict(
                                        color=self.color_scheme['quaternary'],
                                        size=8,
                                        symbol='circle',
                                        line=dict(width=1, color=self.color_scheme['background'])
                                    )
                                ),
                                row=row, col=1
                            )
                    
                    row += 1
                    
                    # Add volatility trace if available
                    if 'RV_Composite' in df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=df.index,
                                y=df['RV_Composite'] * 100,  # Convert to percentage
                                mode='lines',
                                name='Volatility (%)',
                                line=dict(color=self.color_scheme['tertiary'], width=1.5)
                            ),
                            row=row, col=1
                        )
                        
                        # Highlight volatility anomalies if available
                        if 'Volatility_Anomaly' in df.columns:
                            vol_anomalies = df[df['Volatility_Anomaly'] == 1]
                            
                            if not vol_anomalies.empty:
                                fig.add_trace(
                                    go.Scatter(
                                        x=vol_anomalies.index,
                                        y=vol_anomalies['RV_Composite'] * 100,  # Convert to percentage
                                        mode='markers',
                                        name='Volatility Anomalies',
                                        marker=dict(
                                            color=self.color_scheme['quaternary'],
                                            size=8,
                                            symbol='circle',
                                            line=dict(width=1, color=self.color_scheme['background'])
                                        )
                                    ),
                                    row=row, col=1
                                )
                    
                    row += 1
            
            # Add options data if available
            if has_options_data:
                results = self.options_results[symbol]
                
                # Options volume
                if 'volume_analysis' in results:
                    volume_analysis = results['volume_analysis']
                    
                    call_volume = volume_analysis.get('call_volume', 0)
                    put_volume = volume_analysis.get('put_volume', 0)
                    
                    fig.add_trace(
                        go.Bar(
                            x=['Calls', 'Puts'],
                            y=[call_volume, put_volume],
                            marker_color=[self.color_scheme['positive'], self.color_scheme['negative']],
                            text=[f"{call_volume:,.0f}", f"{put_volume:,.0f}"],
                            textposition='auto',
                            name='Volume'
                        ),
                        row=row, col=1
                    )
                    
                    # Add volume ratio
                    if 'volume_ratio' in volume_analysis:
                        volume_ratio = volume_analysis['volume_ratio']
                        fig.add_annotation(
                            x=0.5,
                            y=0.9,
                            xref='paper',
                            yref='paper',
                            text=f"Call/Put Ratio: {volume_ratio:.2f}",
                            showarrow=False,
                            font=dict(size=12),
                            bgcolor=self.color_scheme['background'],
                            bordercolor=self.color_scheme['grid'],
                            borderwidth=1,
                            borderpad=4,
                            row=row, col=1
                        )
                    
                    row += 1
                
                # IV skew
                if 'options_data' in results and 'iv' in results['options_data'].columns:
                    options_data = results['options_data']
                    
                    # Filter out invalid IVs
                    df = options_data[~options_data['iv'].isna()].copy()
                    
                    if not df.empty:
                        # Group by option type and moneyness
                        calls = df[df['option_type'].str.lower() == 'call']
                        puts = df[df['option_type'].str.lower() == 'put']
                        
                        # Sort by moneyness
                        calls = calls.sort_values(by=['moneyness'])
                        puts = puts.sort_values(by=['moneyness'])
                        
                        # Add traces
                        if not calls.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=calls['moneyness'],
                                    y=calls['iv'] * 100,  # Convert to percentage
                                    mode='lines+markers',
                                    line=dict(color=self.color_scheme['primary']),
                                    marker=dict(symbol='circle', size=6),
                                    name='Call IV'
                                ),
                                row=row, col=1
                            )
                        
                        if not puts.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=puts['moneyness'],
                                    y=puts['iv'] * 100,  # Convert to percentage
                                    mode='lines+markers',
                                    line=dict(color=self.color_scheme['quaternary']),
                                    marker=dict(symbol='x', size=6),
                                    name='Put IV'
                                ),
                                row=row, col=1
                            )
                        
                        # Add vertical line at moneyness = 1 (at-the-money)
                        fig.add_shape(
                            type='line',
                            x0=1, y0=0,
                            x1=1, y1=1,
                            yref='paper',
                            line=dict(
                                color=self.color_scheme['grid'],
                                width=1,
                                dash='dash'
                            ),
                            row=row, col=1
                        )
                        
                        # Add annotation for ATM
                        fig.add_annotation(
                            x=1,
                            y=1,
                            yref='paper',
                            text='ATM',
                            showarrow=False,
                            yanchor='bottom',
                            bgcolor=self.color_scheme['background'],
                            bordercolor=self.color_scheme['grid'],
                            borderwidth=1,
                            borderpad=4,
                            row=row, col=1
                        )
                    
                    row += 1
                
                # Delta by moneyness
                if 'options_data' in results and 'delta' in results['options_data'].columns:
                    options_data = results['options_data']
                    
                    # Filter out invalid deltas
                    df = options_data[~options_data['delta'].isna()].copy()
                    
                    if not df.empty:
                        # Group by option type and moneyness
                        calls = df[df['option_type'].str.lower() == 'call']
                        puts = df[df['option_type'].str.lower() == 'put']
                        
                        # Sort by moneyness
                        calls = calls.sort_values(by=['moneyness'])
                        puts = puts.sort_values(by=['moneyness'])
                        
                        # Add traces
                        if not calls.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=calls['moneyness'],
                                    y=calls['delta'],
                                    mode='lines+markers',
                                    line=dict(color=self.color_scheme['primary']),
                                    marker=dict(symbol='circle', size=6),
                                    name='Call Delta'
                                ),
                                row=row, col=1
                            )
                        
                        if not puts.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=puts['moneyness'],
                                    y=puts['delta'],
                                    mode='lines+markers',
                                    line=dict(color=self.color_scheme['quaternary']),
                                    marker=dict(symbol='x', size=6),
                                    name='Put Delta'
                                ),
                                row=row, col=1
                            )
                        
                        # Add vertical line at moneyness = 1 (at-the-money)
                        fig.add_shape(
                            type='line',
                            x0=1, y0=0,
                            x1=1, y1=1,
                            yref='paper',
                            line=dict(
                                color=self.color_scheme['grid'],
                                width=1,
                                dash='dash'
                            ),
                            row=row, col=1
                        )
                        
                        # Add horizontal line at delta = 0
                        fig.add_shape(
                            type='line',
                            x0=0, y0=0,
                            x1=1, y1=0,
                            xref='paper',
                            line=dict(
                                color=self.color_scheme['grid'],
                                width=1,
                                dash='dash'
                            ),
                            row=row, col=1
                        )
            
            # Update layout
            fig.update_layout(
                template=self.plotly_template,
                height=height,
                width=width,
                title=dict(
                    text=f"{symbol} Comprehensive Analysis Dashboard",
                    x=0.5,
                    xanchor='center',
                    font=dict(size=20)
                ),
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                margin=dict(t=100, b=50, l=50, r=50)
            )
            
            # Add timestamp
            fig.add_annotation(
                x=0.01,
                y=0.99,
                xref='paper',
                yref='paper',
                text=f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                showarrow=False,
                font=dict(size=10),
                align='left',
                bgcolor=self.color_scheme['background'],
                bordercolor=self.color_scheme['grid'],
                borderwidth=1,
                borderpad=4
            )
            
            # Save to HTML if requested
            if save_html:
                output_path = self.output_dir / f"{symbol}_dashboard.html"
                fig.write_html(str(output_path))
                logger.info(f"Saved dashboard to {output_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating comprehensive dashboard for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create comprehensive dashboard for {symbol}: {str(e)}")
    
    def create_all_dashboards(self, save_html: bool = True) -> Dict[str, go.Figure]:
        """
        Create dashboards for all symbols with data.
        
        Args:
            save_html: Whether to save the dashboards as HTML
            
        Returns:
            Dictionary mapping symbols to dashboard figures
        """
        # Get all symbols with data
        symbols = set(self.price_results.keys()) | set(self.options_results.keys())
        logger.info(f"Creating dashboards for {len(symbols)} symbols")
        
        dashboards = {}
        
        for symbol in symbols:
            try:
                dashboards[symbol] = self.create_comprehensive_dashboard(symbol, save_html=save_html)
                logger.info(f"Created dashboard for {symbol}")
            except Exception as e:
                logger.error(f"Error creating dashboard for {symbol}: {str(e)}")
        
        return dashboards
    
    def create_html_report(
        self,
        price_summary: Dict[str, Dict[str, Any]] = None,
        options_summary: Dict[str, Dict[str, Any]] = None,
        anomaly_impact: Dict[str, Dict[str, Any]] = None
    ) -> str:
        """
        Create an HTML report with analysis results.
        
        Args:
            price_summary: Summary of price analysis
            options_summary: Summary of options analysis
            anomaly_impact: Summary of anomaly impact
            
        Returns:
            Path to the generated HTML report
        """
        try:
            report_path = self.output_dir / "analysis_report.html"
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Crypto Analysis Report</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        margin: 20px;
                        color: {self.color_scheme['text']};
                        background-color: {self.color_scheme['background']};
                    }}
                    h1, h2, h3, h4 {{
                        color: {self.color_scheme['primary']};
                    }}
                    .dashboard-links {{
                        margin: 20px 0;
                    }}
                    .dashboard-links a {{
                        display: inline-block;
                        margin: 5px;
                        padding: 10px;
                        background-color: {self.color_scheme['secondary']};
                        color: white;
                        text-decoration: none;
                        border-radius: 5px;
                    }}
                    .dashboard-links a:hover {{
                        background-color: {self.color_scheme['primary']};
                    }}
                    table {{
                        border-collapse: collapse;
                        width: 100%;
                        margin: 20px 0;
                    }}
                    th, td {{
                        border: 1px solid {self.color_scheme['grid']};
                        padding: 8px;
                        text-align: left;
                    }}
                    th {{
                        background-color: {self.color_scheme['secondary']};
                        color: white;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
                    .positive {{
                        color: {self.color_scheme['positive']};
                    }}
                    .negative {{
                        color: {self.color_scheme['negative']};
                    }}
                    .timestamp {{
                        font-size: 12px;
                        color: {self.color_scheme['neutral']};
                        margin-top: 30px;
                    }}
                </style>
            </head>
            <body>
                <h1>Crypto Volatility and Options Analysis Report</h1>
                <p>This report summarizes the results of volatility and options analysis for cryptocurrency symbols.</p>
                
                <div class="dashboard-links">
                    <h2>Interactive Dashboards</h2>
            """
            
            # Add links to dashboards
            symbols = set(self.price_results.keys()) | set(self.options_results.keys())
            
            for symbol in sorted(symbols):
                dashboard_path = f"{symbol}_dashboard.html"
                if (self.output_dir / dashboard_path).exists():
                    html_content += f'<a href="{dashboard_path}" target="_blank">{symbol} Dashboard</a>\n'
            
            html_content += """
                </div>
            """
            
            # Add price analysis summary
            if price_summary:
                html_content += """
                <h2>Price and Volatility Analysis</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Avg. Volatility</th>
                        <th>Anomalies</th>
                        <th>Anomaly Rate</th>
                        <th>Avg. Return</th>
                    </tr>
                """
                
                for symbol, summary in sorted(price_summary.items()):
                    vol_mean = summary.get('volatility', {}).get('mean', 0) * 100
                    anomaly_count = summary.get('anomalies', {}).get('count', 0)
                    anomaly_rate = summary.get('anomalies', {}).get('rate', 0) * 100
                    return_mean = summary.get('returns', {}).get('mean', 0) * 100
                    
                    return_class = 'positive' if return_mean >= 0 else 'negative'
                    
                    html_content += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{vol_mean:.2f}%</td>
                        <td>{anomaly_count}</td>
                        <td>{anomaly_rate:.2f}%</td>
                        <td class="{return_class}">{return_mean:.2f}%</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            # Add options analysis summary
            if options_summary:
                html_content += """
                <h2>Options Analysis</h2>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Call Volume</th>
                        <th>Put Volume</th>
                        <th>C/P Ratio</th>
                        <th>Avg. IV</th>
                        <th>IV Skew</th>
                    </tr>
                """
                
                for symbol, summary in sorted(options_summary.items()):
                    vol_analysis = summary.get('volume_analysis', {})
                    iv_analysis = summary.get('iv_analysis', {})
                    
                    call_volume = vol_analysis.get('call_volume', 0)
                    put_volume = vol_analysis.get('put_volume', 0)
                    volume_ratio = vol_analysis.get('volume_ratio', 0)
                    mean_iv = iv_analysis.get('mean_iv', 0) * 100
                    iv_skew = iv_analysis.get('iv_skew', 0) * 100
                    
                    html_content += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{call_volume:,.0f}</td>
                        <td>{put_volume:,.0f}</td>
                        <td>{volume_ratio:.2f}</td>
                        <td>{mean_iv:.2f}%</td>
                        <td>{iv_skew:.2f}%</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            # Add anomaly impact analysis
            if anomaly_impact:
                html_content += """
                <h2>Anomaly Impact Analysis</h2>
                <p>This table shows the average returns following anomalous price and volatility events.</p>
                <table>
                    <tr>
                        <th>Symbol</th>
                        <th>Anomalies</th>
                        <th>1-Day Return</th>
                        <th>5-Day Return</th>
                        <th>10-Day Return</th>
                        <th>20-Day Return</th>
                        <th>Significant?</th>
                    </tr>
                """
                
                for symbol, impact in sorted(anomaly_impact.items()):
                    anomaly_count = impact.get('anomaly_count', 0)
                    
                    # Get returns for different horizons
                    returns = {}
                    significance = {}
                    
                    for days in [1, 5, 10, 20]:
                        if days in impact.get('impact', {}):
                            day_impact = impact['impact'][days]
                            returns[days] = day_impact.get('all', {}).get('mean', 0) * 100
                            significance[days] = day_impact.get('significance', {}).get('significant', False)
                    
                    html_content += f"""
                    <tr>
                        <td>{symbol}</td>
                        <td>{anomaly_count}</td>
                    """
                    
                    for days in [1, 5, 10, 20]:
                        return_val = returns.get(days, 0)
                        is_significant = significance.get(days, False)
                        
                        return_class = 'positive' if return_val >= 0 else 'negative'
                        significant_str = '✓' if is_significant else ''
                        
                        html_content += f"""
                        <td class="{return_class}">{return_val:.2f}%</td>
                        """
                    
                    # Add significance column
                    is_any_significant = any(significance.values())
                    
                    html_content += f"""
                        <td>{is_any_significant}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
            
            # Add timestamp
            html_content += f"""
                <div class="timestamp">
                    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Error creating HTML report: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to create HTML report: {str(e)}")
    
    def _filter_dataframe_by_date(
        self,
        df: pd.DataFrame,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None
    ) -> pd.DataFrame:
        """
        Filter a DataFrame by date range.
        
        Args:
            df: DataFrame to filter
            start_date: Start date
            end_date: End date
            
        Returns:
            Filtered DataFrame
        """
        # Make a copy to avoid modifying the original
        filtered_df = df.copy()
        
        # Convert dates to datetime if they are strings
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Filter by date
        if start_date:
            filtered_df = filtered_df[filtered_df.index.date >= start_date]
        
        if end_date:
            filtered_df = filtered_df[filtered_df.index.date <= end_date]
        
        return filtered_df
    
    def generate_all_plots(
        self,
        symbol: str,
        output_format: str = 'html',
        save_plots: bool = True
    ) -> Dict[str, go.Figure]:
        """
        Generate all available plots for a symbol.
        
        Args:
            symbol: Symbol to generate plots for
            output_format: Output format ('html', 'png', 'pdf', 'svg')
            save_plots: Whether to save the plots
            
        Returns:
            Dictionary mapping plot types to figures
        """
        plots = {}
        
        # Check if data is available
        has_price_data = symbol in self.price_results
        has_options_data = symbol in self.options_results
        
        try:
            # Price and volatility plot
            if has_price_data:
                plots['price_volatility'] = self.plot_price_volatility(symbol)
                
                if save_plots:
                    output_path = self.output_dir / f"{symbol}_price_volatility.{output_format}"
                    if output_format == 'html':
                        plots['price_volatility'].write_html(str(output_path))
                    else:
                        plots['price_volatility'].write_image(str(output_path))
                    logger.info(f"Saved price volatility plot to {output_path}")
            
            # Options plots
            if has_options_data:
                # Options volume plot
                plots['options_volume'] = self.plot_options_volume(symbol)
                
                if save_plots:
                    output_path = self.output_dir / f"{symbol}_options_volume.{output_format}"
                    if output_format == 'html':
                        plots['options_volume'].write_html(str(output_path))
                    else:
                        plots['options_volume'].write_image(str(output_path))
                    logger.info(f"Saved options volume plot to {output_path}")
                
                # IV skew plot
                plots['iv_skew'] = self.plot_iv_skew(symbol)
                
                if save_plots:
                    output_path = self.output_dir / f"{symbol}_iv_skew.{output_format}"
                    if output_format == 'html':
                        plots['iv_skew'].write_html(str(output_path))
                    else:
                        plots['iv_skew'].write_image(str(output_path))
                    logger.info(f"Saved IV skew plot to {output_path}")
                
                # Try to create IV surface plot
                try:
                    plots['iv_surface'] = self.plot_iv_surface(symbol)
                    
                    if save_plots:
                        output_path = self.output_dir / f"{symbol}_iv_surface.{output_format}"
                        if output_format == 'html':
                            plots['iv_surface'].write_html(str(output_path))
                        else:
                            plots['iv_surface'].write_image(str(output_path))
                        logger.info(f"Saved IV surface plot to {output_path}")
                except Exception as e:
                    logger.warning(f"Could not create IV surface plot for {symbol}: {str(e)}")
                
                # Greeks plots for different Greeks
                for greek in ['delta', 'gamma', 'vega', 'theta']:
                    try:
                        plots[f'{greek}_plot'] = self.plot_greeks_by_moneyness(symbol, greek)
                        
                        if save_plots:
                            output_path = self.output_dir / f"{symbol}_{greek}.{output_format}"
                            if output_format == 'html':
                                plots[f'{greek}_plot'].write_html(str(output_path))
                            else:
                                plots[f'{greek}_plot'].write_image(str(output_path))
                            logger.info(f"Saved {greek} plot to {output_path}")
                    except Exception as e:
                        logger.warning(f"Could not create {greek} plot for {symbol}: {str(e)}")
                
                # Options chain plot
                plots['options_chain'] = self.plot_options_chain(symbol)
                
                if save_plots:
                    output_path = self.output_dir / f"{symbol}_options_chain.{output_format}"
                    if output_format == 'html':
                        plots['options_chain'].write_html(str(output_path))
                    else:
                        plots['options_chain'].write_image(str(output_path))
                    logger.info(f"Saved options chain plot to {output_path}")
            
            # Create comprehensive dashboard
            plots['dashboard'] = self.create_comprehensive_dashboard(symbol, save_html=save_plots)
            
            logger.info(f"Generated all plots for {symbol}")
            return plots
            
        except Exception as e:
            logger.error(f"Error generating plots for {symbol}: {str(e)}")
            logger.debug(traceback.format_exc())
            raise VisualizationError(f"Failed to generate plots for {symbol}: {str(e)}")