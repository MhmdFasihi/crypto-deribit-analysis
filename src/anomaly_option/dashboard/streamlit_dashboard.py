"""
Streamlit dashboard for Crypto-Deribit-Analysis
Interactive visualization of cryptocurrency price volatility and options data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, date, timedelta
import os
import sys
import json
from pathlib import Path

# Add parent directory to path to import package
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from anomaly_option.core.config import Config
from anomaly_option.core.analysis_system import VolatilityOptionsAnalysisSystem
from anomaly_option.data.crypto_data_fetcher import CryptoDataFetcher
from anomaly_option.analysis.volatility_analyzer import CryptoVolatilityAnalyzer
from anomaly_option.analysis.options_analyzer import OptionsAnalyzer
from anomaly_option.visualization.visualizer import CryptoVolatilityOptionsVisualizer


# Set page configuration
st.set_page_config(
    page_title="Crypto Volatility & Options Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create a container for header
header = st.container()
with header:
    st.title("📈 Crypto Volatility & Options Analysis")
    st.markdown("""
        This dashboard provides an interactive analysis of cryptocurrency price volatility and options data.
        Select options in the sidebar to customize the analysis.
    """)
    st.divider()


# --------------------------------------------------------
# Sidebar configuration
# --------------------------------------------------------
st.sidebar.header("Analysis Parameters")

# Symbol selection
DEFAULT_SYMBOLS = ["BTC-USD", "ETH-USD"]
selected_symbols = st.sidebar.multiselect(
    "Select Cryptocurrencies",
    options=["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "XRP-USD"],
    default=DEFAULT_SYMBOLS
)

# Date range selection
today = date.today()
default_start_date = today - timedelta(days=60)
default_end_date = today

col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=default_start_date)
with col2:
    end_date = st.date_input("End Date", value=default_end_date)

# Analysis parameters
st.sidebar.subheader("Volatility Parameters")
window_size = st.sidebar.slider("Window Size", min_value=5, max_value=60, value=30, 
                                help="Window size for rolling calculations")
z_threshold = st.sidebar.slider("Z-Score Threshold", min_value=1.0, max_value=5.0, value=3.0, step=0.1,
                               help="Z-score threshold for anomaly detection")
volatility_window = st.sidebar.slider("Volatility Window", min_value=5, max_value=60, value=21,
                                     help="Window size for volatility calculations")

# Options data toggle
include_options = st.sidebar.checkbox("Include Options Data", value=True)

# Run analysis button
run_analysis = st.sidebar.button("Run Analysis", type="primary")

# Cache directory
cache_dir = st.sidebar.text_input("Cache Directory", value="data/cache",
                                 help="Directory for caching data")

# --------------------------------------------------------
# Helper functions
# --------------------------------------------------------

@st.cache_data(ttl=3600)  # Cache data for 1 hour
def run_volatility_analysis(symbols, start_date, end_date, window_size, z_threshold, volatility_window, include_options):
    """Run volatility and options analysis and return results"""
    try:
        # Create output directory if it doesn't exist
        Path("results").mkdir(exist_ok=True)
        
        # Initialize analysis system
        system = VolatilityOptionsAnalysisSystem(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            max_workers=4,
            output_dir="results",
            fetch_options=include_options
        )
        
        # Update configuration
        Config.WINDOW_SIZE = window_size
        Config.Z_THRESHOLD = z_threshold
        Config.VOLATILITY_WINDOW = volatility_window
        
        # Run analysis
        results = system.run_analysis()
        
        return {
            "success": True,
            "results": results,
            "price_data": system.price_data,
            "options_data": system.options_data,
            "volatility_results": system.volatility_results,
            "options_results": system.options_results
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def load_sample_data():
    """Load sample data if available or create dummy data"""
    price_data = {}
    volatility_results = {}
    options_results = {}
    
    # Create sample data for BTC-USD
    today = pd.Timestamp.today()
    dates = pd.date_range(end=today, periods=100)
    
    # Sample price data
    for symbol in ["BTC-USD", "ETH-USD"]:
        np.random.seed(42)  # For reproducibility
        close = np.random.normal(loc=1, scale=0.01, size=100).cumprod() * (50000 if symbol == "BTC-USD" else 2000)
        high = close * (1 + abs(np.random.normal(loc=0, scale=0.01, size=100)))
        low = close * (1 - abs(np.random.normal(loc=0, scale=0.01, size=100)))
        open_price = low + (high - low) * np.random.random(size=100)
        volume = np.random.normal(loc=1000, scale=100, size=100) * (10 if symbol == "BTC-USD" else 100)
        
        df = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }, index=dates)
        
        price_data[symbol] = df
        
        # Sample volatility results
        vol_df = df.copy()
        vol_df['Returns'] = vol_df['Close'].pct_change()
        vol_df['Log_Returns'] = np.log(vol_df['Close'] / vol_df['Close'].shift(1))
        vol_df['RV_Close'] = vol_df['Returns'].rolling(window=21).std() * np.sqrt(252)
        vol_df['RV_Parkinson'] = vol_df['RV_Close'] * 0.9  # Simplified
        vol_df['RV_Garman_Klass'] = vol_df['RV_Close'] * 0.95  # Simplified
        vol_df['RV_Composite'] = vol_df[['RV_Close', 'RV_Parkinson', 'RV_Garman_Klass']].mean(axis=1)
        
        vol_df['Price_Rolling_Mean'] = vol_df['Close'].rolling(window=30).mean()
        vol_df['Price_Rolling_Std'] = vol_df['Close'].rolling(window=30).std()
        vol_df['Price_Z_Score'] = (vol_df['Close'] - vol_df['Price_Rolling_Mean']) / vol_df['Price_Rolling_Std']
        vol_df['Return_Z_Score'] = (vol_df['Returns'] - vol_df['Returns'].rolling(window=30).mean()) / vol_df['Returns'].rolling(window=30).std()
        vol_df['Volatility_Z_Score'] = (vol_df['RV_Composite'] - vol_df['RV_Composite'].rolling(window=30).mean()) / vol_df['RV_Composite'].rolling(window=30).std()
        
        # Generate some anomalies
        vol_df['Price_Anomaly'] = 0
        vol_df['Return_Anomaly'] = 0
        vol_df['Volatility_Anomaly'] = 0
        
        # Random anomalies
        np.random.seed(42)
        anomaly_idx = np.random.choice(len(vol_df), 5, replace=False)
        vol_df.iloc[anomaly_idx, vol_df.columns.get_indexer(['Price_Anomaly'])] = 1
        
        anomaly_idx = np.random.choice(len(vol_df), 5, replace=False)
        vol_df.iloc[anomaly_idx, vol_df.columns.get_indexer(['Return_Anomaly'])] = 1
        
        anomaly_idx = np.random.choice(len(vol_df), 5, replace=False)
        vol_df.iloc[anomaly_idx, vol_df.columns.get_indexer(['Volatility_Anomaly'])] = 1
        
        vol_df['Combined_Anomaly'] = ((vol_df['Price_Anomaly'] == 1) | 
                                     (vol_df['Return_Anomaly'] == 1) | 
                                     (vol_df['Volatility_Anomaly'] == 1)).astype(int)
        
        vol_df['Anomaly_Direction'] = np.where(
            vol_df['Combined_Anomaly'] == 1,
            np.where(vol_df['Returns'] > 0, 1, -1),
            0
        )
        
        vol_df['Anomaly_Magnitude'] = np.where(
            vol_df['Combined_Anomaly'] == 1,
            np.abs(vol_df['Returns']),
            0
        )
        
        vol_df.fillna(0, inplace=True)
        volatility_results[symbol] = vol_df
        
        # Sample options data
        if symbol == "BTC-USD":
            options_results[symbol] = {
                'current_price': close[-1],
                'analysis_date': today.strftime('%Y-%m-%d'),
                'volume_analysis': {
                    'total_volume': 15000,
                    'call_volume': 9000,
                    'put_volume': 6000,
                    'volume_ratio': 1.5,
                    'volume_by_expiry': {
                        f"{(today + pd.Timedelta(days=30)).strftime('%Y-%m-%d')}": 5000,
                        f"{(today + pd.Timedelta(days=60)).strftime('%Y-%m-%d')}": 4000,
                        f"{(today + pd.Timedelta(days=90)).strftime('%Y-%m-%d')}": 3000,
                        f"{(today + pd.Timedelta(days=180)).strftime('%Y-%m-%d')}": 2000,
                        f"{(today + pd.Timedelta(days=365)).strftime('%Y-%m-%d')}": 1000
                    },
                    'volume_by_moneyness': {
                        'Deep ITM': 1000,
                        'ITM': 3000,
                        'ATM': 5000,
                        'OTM': 4000,
                        'Deep OTM': 2000
                    }
                },
                'iv_analysis': {
                    'mean_iv': 0.75,
                    'median_iv': 0.72,
                    'min_iv': 0.45,
                    'max_iv': 1.2,
                    'std_iv': 0.15,
                    'call_mean_iv': 0.7,
                    'put_mean_iv': 0.8,
                    'iv_skew': 0.1
                }
            }
    
    return {
        "price_data": price_data,
        "volatility_results": volatility_results,
        "options_results": options_results
    }


def create_price_volatility_plot(df, symbol, height=500):
    """Create a price and volatility plot for a symbol"""
    if df.empty:
        return None
        
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
            line=dict(color='blue', width=1.5)
        ),
        row=1, col=1
    )
    
    # Add price rolling mean if available
    if 'Price_Rolling_Mean' in df.columns and not df['Price_Rolling_Mean'].isnull().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Price_Rolling_Mean'],
                mode='lines',
                name='Price MA',
                line=dict(color='orange', width=1, dash='dash')
            ),
            row=1, col=1
        )
    
    # Add volatility trace
    if 'RV_Composite' in df.columns and not df['RV_Composite'].isnull().all():
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RV_Composite'] * 100,  # Convert to percentage
                mode='lines',
                name='Volatility (%)',
                line=dict(color='green', width=1.5)
            ),
            row=2, col=1
        )
    
    # Highlight anomalies if available
    if 'Combined_Anomaly' in df.columns:
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
                        color='red',
                        size=8,
                        symbol='circle',
                        line=dict(width=1, color='white')
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
                            color='red',
                            size=8,
                            symbol='circle',
                            line=dict(width=1, color='white')
                        )
                    ),
                    row=2, col=1
                )
    
    # Update layout
    fig.update_layout(
        height=height,
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    # Update axes
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volatility (%)", row=2, col=1)
    
    return fig

def create_options_volume_chart(volume_analysis, symbol, height=400):
    """Create options volume chart"""
    if not volume_analysis:
        return None
        
    # Create figure with subplots
    fig = make_subplots(
        rows=1, 
        cols=2,
        subplot_titles=(f"{symbol} Call/Put Volume", "Volume by Moneyness"),
        specs=[[{"type": "bar"}, {"type": "bar"}]]
    )
    
    # Add call/put volume
    call_volume = volume_analysis.get('call_volume', 0)
    put_volume = volume_analysis.get('put_volume', 0)
    
    fig.add_trace(
        go.Bar(
            x=['Calls', 'Puts'],
            y=[call_volume, put_volume],
            marker_color=['green', 'red'],
            text=[f"{call_volume:,.0f}", f"{put_volume:,.0f}"],
            textposition='auto',
            name='Volume'
        ),
        row=1, col=1
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
        
        fig.add_trace(
            go.Bar(
                x=categories,
                y=volumes,
                marker_color='blue',
                name='Volume by Moneyness'
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_layout(
        height=height,
        template='plotly_white',
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


def create_iv_skew_chart(options_data, symbol, height=400):
    """Create IV skew chart based on options data"""
    if 'iv_analysis' not in options_data:
        return None
    
    iv_analysis = options_data['iv_analysis']
    
    # Create figure
    fig = go.Figure()
    
    # Add a horizontal bar for skew visualization
    skew_value = iv_analysis.get('iv_skew', 0) * 100  # Convert to percentage
    
    fig.add_trace(
        go.Bar(
            x=['IV Skew'],
            y=[skew_value],
            marker_color='blue' if skew_value >= 0 else 'red',
            text=[f"{skew_value:.2f}%"],
            textposition='auto'
        )
    )
    
    # Add IV statistics as a table
    mean_iv = iv_analysis.get('mean_iv', 0) * 100
    median_iv = iv_analysis.get('median_iv', 0) * 100
    min_iv = iv_analysis.get('min_iv', 0) * 100
    max_iv = iv_analysis.get('max_iv', 0) * 100
    std_iv = iv_analysis.get('std_iv', 0) * 100
    call_mean_iv = iv_analysis.get('call_mean_iv', 0) * 100
    put_mean_iv = iv_analysis.get('put_mean_iv', 0) * 100
    
    stats_text = (
        f"<b>{symbol} IV Statistics</b><br>"
        f"Mean IV: {mean_iv:.2f}%<br>"
        f"Median IV: {median_iv:.2f}%<br>"
        f"Min IV: {min_iv:.2f}%<br>"
        f"Max IV: {max_iv:.2f}%<br>"
        f"Std Dev: {std_iv:.2f}%<br>"
        f"Call Mean IV: {call_mean_iv:.2f}%<br>"
        f"Put Mean IV: {put_mean_iv:.2f}%<br>"
        f"IV Skew: {skew_value:.2f}%"
    )
    
    fig.add_annotation(
        x=0.9,
        y=0.5,
        xref="paper",
        yref="paper",
        text=stats_text,
        showarrow=False,
        align="left",
        bordercolor="gray",
        borderwidth=1,
        borderpad=5,
        bgcolor="white",
        opacity=0.8
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Implied Volatility Skew",
        height=height,
        template='plotly_white',
        xaxis=dict(
            title='',
            showticklabels=True
        ),
        yaxis=dict(
            title='IV Skew (%)',
            range=[min(-5, skew_value * 1.2) if skew_value < 0 else -5, 
                   max(5, skew_value * 1.2) if skew_value > 0 else 5]
        ),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig

def create_anomaly_stats_chart(volatility_results, symbol, height=400):
    """Create anomaly statistics chart"""
    if symbol not in volatility_results or volatility_results[symbol].empty:
        return None
        
    df = volatility_results[symbol]
    
    if 'Combined_Anomaly' not in df.columns:
        return None
    
    # Calculate anomaly statistics
    anomalies = df[df['Combined_Anomaly'] == 1]
    
    if anomalies.empty:
        return None
    
    total_anomalies = len(anomalies)
    price_anomalies = df['Price_Anomaly'].sum()
    return_anomalies = df['Return_Anomaly'].sum()
    volatility_anomalies = df['Volatility_Anomaly'].sum() if 'Volatility_Anomaly' in df.columns else 0
    
    positive_anomalies = len(anomalies[anomalies['Anomaly_Direction'] == 1])
    negative_anomalies = len(anomalies[anomalies['Anomaly_Direction'] == -1])
    
    # Create data for the bar chart
    categories = ['Total', 'Price', 'Return', 'Volatility', 'Positive', 'Negative']
    values = [total_anomalies, price_anomalies, return_anomalies, volatility_anomalies, positive_anomalies, negative_anomalies]
    colors = ['blue', 'orange', 'green', 'red', 'green', 'red']
    
    # Create figure
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v}" for v in values],
            textposition='auto'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Anomaly Statistics",
        height=height,
        template='plotly_white',
        xaxis=dict(title='Anomaly Type'),
        yaxis=dict(title='Count'),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    return fig


# --------------------------------------------------------
# Main dashboard content
# --------------------------------------------------------

# Check if analysis should be run
if run_analysis:
    if not selected_symbols:
        st.error("Please select at least one cryptocurrency")
    else:
        with st.spinner("Running analysis..."):
            analysis_results = run_volatility_analysis(
                selected_symbols, 
                start_date, 
                end_date, 
                window_size, 
                z_threshold, 
                volatility_window,
                include_options
            )
            
            if not analysis_results["success"]:
                st.error(f"Analysis failed: {analysis_results['error']}")
            else:
                st.success("Analysis completed successfully!")
else:
    # Load sample data if available
    analysis_results = {"success": True}
    sample_data = load_sample_data()
    
    analysis_results["price_data"] = sample_data["price_data"]
    analysis_results["volatility_results"] = sample_data["volatility_results"]
    analysis_results["options_results"] = sample_data["options_results"]


# Display tabs for different analyses
if "success" in analysis_results and analysis_results["success"]:
    tabs = st.tabs(["Price & Volatility", "Anomalies", "Options Analysis"])
    
    # Tab 1: Price & Volatility
    with tabs[0]:
        for symbol in selected_symbols:
            if symbol in analysis_results["volatility_results"]:
                st.subheader(f"{symbol} Price & Volatility Analysis")
                
                vol_df = analysis_results["volatility_results"][symbol]
                fig = create_price_volatility_plot(vol_df, symbol)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"No price data available for {symbol}")
            else:
                st.warning(f"No analysis results available for {symbol}")
    
    # Tab 2: Anomalies
    with tabs[1]:
        for symbol in selected_symbols:
            if symbol in analysis_results["volatility_results"]:
                st.subheader(f"{symbol} Anomaly Analysis")
                
                # Statistics
                fig = create_anomaly_stats_chart(analysis_results["volatility_results"], symbol)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # List anomalies
                vol_df = analysis_results["volatility_results"][symbol]
                
                if 'Combined_Anomaly' in vol_df.columns:
                    anomalies = vol_df[vol_df['Combined_Anomaly'] == 1]
                    
                    if not anomalies.empty:
                        st.markdown(f"### {symbol} Anomalies ({len(anomalies)} total)")
                        
                        # Prepare data for display
                        display_df = anomalies[['Close', 'Returns', 'RV_Composite', 'Price_Z_Score', 
                                                'Return_Z_Score', 'Volatility_Z_Score']].copy()
                        
                        # Format for better display
                        display_df['Returns'] = display_df['Returns'] * 100
                        display_df['RV_Composite'] = display_df['RV_Composite'] * 100
                        
                        # Rename columns
                        display_df.columns = ['Price', 'Returns (%)', 'Volatility (%)', 
                                             'Price Z-Score', 'Return Z-Score', 'Volatility Z-Score']
                        
                        # Display table
                        st.dataframe(display_df.style.format({
                            'Price': '{:.2f}',
                            'Returns (%)': '{:.2f}%',
                            'Volatility (%)': '{:.2f}%',
                            'Price Z-Score': '{:.2f}',
                            'Return Z-Score': '{:.2f}',
                            'Volatility Z-Score': '{:.2f}'
                        }), use_container_width=True)
                    else:
                        st.info(f"No anomalies detected for {symbol} with current parameters")
                else:
                    st.warning(f"No anomaly data available for {symbol}")
            else:
                st.warning(f"No analysis results available for {symbol}")
    
    # Tab 3: Options Analysis
    with tabs[2]:
        if include_options:
            for symbol in selected_symbols:
                if symbol in analysis_results["options_results"]:
                    st.subheader(f"{symbol} Options Analysis")
                    
                    options_data = analysis_results["options_results"][symbol]
                    
                    if 'volume_analysis' in options_data:
                        fig = create_options_volume_chart(options_data['volume_analysis'], symbol)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # IV Skew
                        fig = create_iv_skew_chart(options_data, symbol)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning(f"No options volume data available for {symbol}")
                else:
                    st.warning(f"No options analysis results available for {symbol}")
        else:
            st.info("Options data analysis is disabled. Enable it in the sidebar to see options analysis.")

# Add information about the system
with st.expander("About the Analysis System"):
    st.markdown("""
    ### Crypto Volatility & Options Analysis System
    
    This dashboard provides interactive visualization of cryptocurrency price volatility and options data. 
    The analysis includes:
    
    - **Price Analysis**: Historical cryptocurrency price data visualization.
    - **Volatility Analysis**: Calculation of various volatility metrics (Close-to-Close, Parkinson, Garman-Klass).
    - **Anomaly Detection**: Identification of anomalous price and volatility patterns using statistical methods.
    - **Options Analysis**: Analysis of cryptocurrency options data and implied volatility patterns.
    
    The system uses Z-score based anomaly detection with configurable parameters. Anomalies are detected 
    when a value exceeds the specified Z-score threshold compared to its rolling average.
    
    #### Parameters:
    - **Window Size**: The number of periods used for rolling calculations (e.g., moving average).
    - **Z-Score Threshold**: The number of standard deviations a value must deviate to be considered an anomaly.
    - **Volatility Window**: The number of periods used specifically for volatility calculations.
    
    The dashboard provides sample data visualization when no analysis is run, and can perform live analysis 
    on selected cryptocurrencies with custom parameters.
    """)

# Footer
st.divider()
st.markdown(f"""
<div style="text-align: center; color: gray; font-size: 0.8em;">
    Crypto Volatility & Options Analysis Dashboard | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)
