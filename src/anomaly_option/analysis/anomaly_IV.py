"""
Enhanced Cryptocurrency Anomaly Detection System
Implements Z-score based anomaly detection with advanced analysis features
"""

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Tuple, Any
from scipy import stats


class EnhancedCryptoAnomalyDetector:
    """
    Enhanced anomaly detector with advanced analytical features.
    """

    def __init__(
        self,
        window_size: int = 7,
        z_threshold: float = 3.0,
        volatility_window: int = 21,
        annualization_factor: int = 252
    ) -> None:
        """Initialize detector with configurable parameters."""
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.data: Dict[str, pd.DataFrame] = {}
        self.anomalies: Dict[str, pd.DataFrame] = {}
        self.analysis: Dict[str, pd.DataFrame] = {}

    def fetch_data(
        self,
        symbols: list[str],
        start_date: str = None,
        end_date: str = None
    ) -> None:
        """Fetch cryptocurrency data from Yahoo Finance."""
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if not start_date:
            start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')

        for symbol in symbols:
            df = yf.download(symbol, start=start_date, end=end_date)
            # Add volume if available
            if 'Volume' in df.columns:
                self.data[symbol] = df
            else:
                self.data[symbol] = df

    def calculate_z_score(self, data: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate rolling Z-score."""
        rolling_mean = data.rolling(window=self.window_size).mean()
        rolling_std = data.rolling(window=self.window_size).std()
        z_score = (data - rolling_mean) / rolling_std
        return rolling_mean, rolling_std, z_score

    def detect_anomalies(self, symbol: str) -> pd.DataFrame:
        """Enhanced anomaly detection with additional metrics."""
        if symbol not in self.data:
            raise ValueError(f"Data not available for {symbol}")

        # Extract data
        df = self.data[symbol]
        closing_prices = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']

        # Calculate price returns
        returns = closing_prices.pct_change()
        log_returns = np.log(closing_prices / closing_prices.shift(1))

        # Calculate volatility
        volatility = returns.rolling(window=self.volatility_window).std() * np.sqrt(self.annualization_factor)  # Annualized

        # Calculate Z-scores for both price and returns
        price_rolling_mean, price_rolling_std, price_z_score = self.calculate_z_score(closing_prices)
        return_rolling_mean, return_rolling_std, return_z_score = self.calculate_z_score(returns)

        # Identify anomalies
        price_anomaly = (np.abs(price_z_score) > self.z_threshold).astype(int)
        return_anomaly = (np.abs(return_z_score) > self.z_threshold).astype(int)
        combined_anomaly = ((price_anomaly == 1) | (return_anomaly == 1)).astype(int)

        # Price momentum
        momentum = closing_prices.pct_change(periods=5)  # 5-day momentum

        # Categorize anomalies
        anomaly_direction = np.where(
            combined_anomaly == 1,
            np.where(returns > 0, 1, -1),  # 1 for positive, -1 for negative
            0
        )

        # Volume analysis (if available)
        if 'Volume' in df.columns:
            volume = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
            volume_ma = volume.rolling(window=self.window_size).mean()
            volume_ratio = volume / volume_ma
        else:
            volume_ratio = pd.Series(1, index=closing_prices.index)

        # Store results
        results = pd.DataFrame({
            'Close': closing_prices,
            'Returns': returns,
            'Log_Returns': log_returns,
            'Rolling_Mean': price_rolling_mean,
            'Rolling_Std': price_rolling_std,
            'Price_Z_Score': price_z_score,
            'Return_Z_Score': return_z_score,
            'Volatility': volatility,
            'Momentum': momentum,
            'Volume_Ratio': volume_ratio,
            'Price_Anomaly': price_anomaly,
            'Return_Anomaly': return_anomaly,
            'Combined_Anomaly': combined_anomaly,
            'Anomaly_Direction': anomaly_direction
        })

        # Calculate anomaly magnitude
        results['Anomaly_Magnitude'] = np.where(
            results['Combined_Anomaly'] == 1,
            np.abs(results['Returns']),
            0
        )

        # Remove NaN values
        results = results.dropna()

        self.anomalies[symbol] = results
        return results

    def analyze_anomaly_impact(self, symbol: str, forward_days: int = 5) -> pd.DataFrame:
        """Analyze the impact of anomalies on future price movements."""
        if symbol not in self.anomalies:
            raise ValueError(f"No anomaly data for {symbol}")

        data = self.anomalies[symbol].copy()

        # Calculate forward returns
        for i in range(1, forward_days + 1):
            data[f'Forward_{i}d_Return'] = data['Close'].pct_change(periods=i).shift(-i)
            data[f'Forward_{i}d_Cumulative'] = data['Close'].shift(-i) / data['Close'] - 1

        # Filter anomaly days
        anomaly_data = data[data['Combined_Anomaly'] == 1].copy()

        self.analysis[symbol] = anomaly_data
        return anomaly_data

    def create_enhanced_dashboard(self) -> go.Figure:
        """Create an enhanced interactive dashboard."""
        if not self.anomalies:
            raise ValueError("No anomaly detection results available.")

        n_cryptos = len(self.anomalies)

        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=n_cryptos * 3,
            cols=1,
            subplot_titles=[
                title for symbol in self.anomalies.keys()
                for title in [
                    f"{symbol} - Price & Returns",
                    f"{symbol} - Z-Score & Volatility Analysis",
                    f"{symbol} - Anomaly Impact Analysis"
                ]
            ],
            vertical_spacing=0.03,
            specs=[[{"secondary_y": True}]] * (n_cryptos * 3)
        )

        colors = {'BTC-USD': 'blue', 'ETH-USD': 'purple'}

        for idx, (symbol, data) in enumerate(self.anomalies.items()):
            row_price = idx * 3 + 1
            row_zscore = idx * 3 + 2
            row_impact = idx * 3 + 3
            color = colors.get(symbol, 'blue')

            # Plot 1: Price & Returns
            # Normalize price for better visualization
            normalized_price = (data['Close'] - data['Close'].min()) / (data['Close'].max() - data['Close'].min())

            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=normalized_price,
                    mode='lines',
                    name=f"{symbol} Normalized Price",
                    line=dict(color=color, width=1.5)
                ),
                row=row_price,
                col=1,
                secondary_y=False
            )

            # Add returns on secondary axis
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Returns'] * 100,  # Convert to percentage
                    mode='lines',
                    name=f"{symbol} Returns (%)",
                    line=dict(color='gray', width=1),
                    opacity=0.7
                ),
                row=row_price,
                col=1,
                secondary_y=True
            )

            # Highlight anomalies with different colors
            positive_anomalies = data[(data['Combined_Anomaly'] == 1) & (data['Anomaly_Direction'] == 1)]
            negative_anomalies = data[(data['Combined_Anomaly'] == 1) & (data['Anomaly_Direction'] == -1)]

            fig.add_trace(
                go.Scatter(
                    x=positive_anomalies.index,
                    y=positive_anomalies['Returns'] * 100,
                    mode='markers',
                    name=f"{symbol} Positive Anomalies",
                    marker=dict(color='green', size=10, symbol='triangle-up')
                ),
                row=row_price,
                col=1,
                secondary_y=True
            )

            fig.add_trace(
                go.Scatter(
                    x=negative_anomalies.index,
                    y=negative_anomalies['Returns'] * 100,
                    mode='markers',
                    name=f"{symbol} Negative Anomalies",
                    marker=dict(color='red', size=10, symbol='triangle-down')
                ),
                row=row_price,
                col=1,
                secondary_y=True
            )

            # Plot 2: Z-Score & Volatility
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Price_Z_Score'],
                    mode='lines',
                    name=f"{symbol} Price Z-Score",
                    line=dict(color=color, width=1.5)
                ),
                row=row_zscore,
                col=1,
                secondary_y=False
            )

            # Add volatility on secondary axis
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Volatility'] * 100,  # Convert to percentage
                    mode='lines',
                    name=f"{symbol} Volatility (%)",
                    line=dict(color='orange', width=1.5),
                    opacity=0.7
                ),
                row=row_zscore,
                col=1,
                secondary_y=True
            )

            # Add threshold lines for Z-score
            fig.add_hline(y=self.z_threshold, row=row_zscore, col=1,
                         line=dict(color='red', width=1, dash='dash'),
                         annotation_text="Upper Threshold")
            fig.add_hline(y=-self.z_threshold, row=row_zscore, col=1,
                         line=dict(color='red', width=1, dash='dash'),
                         annotation_text="Lower Threshold")

            # Plot 3: Anomaly Impact
            anomaly_days = data[data['Combined_Anomaly'] == 1]
            if len(anomaly_days) > 0:
                # Calculate average returns after anomalies
                forward_days = 5
                future_returns = []

                for i in range(1, forward_days + 1):
                    future_return = data['Close'].pct_change(periods=i).shift(-i)
                    avg_return = future_return[data['Combined_Anomaly'] == 1].mean() * 100
                    future_returns.append(avg_return)

                fig.add_trace(
                    go.Bar(
                        x=list(range(1, forward_days + 1)),
                        y=future_returns,
                        name=f"{symbol} Avg Return After Anomaly (%)",
                        marker_color=color
                    ),
                    row=row_impact,
                    col=1
                )

        # Update layout
        fig.update_layout(
            height=350 * n_cryptos * 3,
            title={
                'text': 'Enhanced Cryptocurrency Anomaly Detection Dashboard',
                'y':0.99,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            showlegend=True,
            template='plotly_white'
        )

        # Update axes labels
        for i in range(n_cryptos):
            row_price = i * 3 + 1
            row_zscore = i * 3 + 2
            row_impact = i * 3 + 3

            fig.update_yaxes(title_text="Normalized Price", row=row_price, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Returns (%)", row=row_price, col=1, secondary_y=True)

            fig.update_yaxes(title_text="Z-Score", row=row_zscore, col=1, secondary_y=False)
            fig.update_yaxes(title_text="Volatility (%)", row=row_zscore, col=1, secondary_y=True)

            fig.update_yaxes(title_text="Average Return (%)", row=row_impact, col=1)
            fig.update_xaxes(title_text="Days After Anomaly", row=row_impact, col=1)

        return fig

    def get_enhanced_statistics(self) -> Dict[str, Dict[str, Any]]:
        """Get enhanced statistics about anomalies."""
        stats = {}

        for symbol, data in self.anomalies.items():
            anomaly_data = data[data['Combined_Anomaly'] == 1]
            positive_anomalies = anomaly_data[anomaly_data['Anomaly_Direction'] == 1]
            negative_anomalies = anomaly_data[anomaly_data['Anomaly_Direction'] == -1]

            # Calculate forward returns statistics
            forward_returns = []
            for i in range(1, 6):  # 1-5 days forward
                future_return = data['Close'].pct_change(periods=i).shift(-i)
                avg_return = future_return[data['Combined_Anomaly'] == 1].mean() * 100
                forward_returns.append(avg_return)

            stats[symbol] = {
                'total_data_points': len(data),
                'total_anomalies': len(anomaly_data),
                'positive_anomalies': len(positive_anomalies),
                'negative_anomalies': len(negative_anomalies),
                'anomaly_percentage': (len(anomaly_data) / len(data)) * 100,
                'avg_anomaly_magnitude': anomaly_data['Anomaly_Magnitude'].mean() * 100,
                'max_anomaly_magnitude': anomaly_data['Anomaly_Magnitude'].max() * 100,
                'avg_volatility_at_anomaly': anomaly_data['Volatility'].mean() * 100,
                'avg_forward_returns': forward_returns
            }

        return stats


def main():
    """Main function demonstrating the enhanced anomaly detection system."""
    # Initialize detector
    detector = EnhancedCryptoAnomalyDetector(window_size=30, z_threshold=1.5)

    # Fetch data for BTC and ETH
    print("Fetching cryptocurrency data...")
    detector.fetch_data(['BTC-USD', 'ETH-USD'])

    # Detect anomalies
    print("Detecting anomalies...")
    for symbol in ['BTC-USD', 'ETH-USD']:
        detector.detect_anomalies(symbol)
        detector.analyze_anomaly_impact(symbol)

    # Get statistics
    stats = detector.get_enhanced_statistics()

    # Print summary
    print("\nEnhanced Anomaly Detection Summary:")
    print("-" * 60)
    for symbol, stat in stats.items():
        print(f"\n{symbol}:")
        print(f"  Total data points: {stat['total_data_points']}")
        print(f"  Total anomalies: {stat['total_anomalies']}")
        print(f"  Positive anomalies: {stat['positive_anomalies']}")
        print(f"  Negative anomalies: {stat['negative_anomalies']}")
        print(f"  Anomaly percentage: {stat['anomaly_percentage']:.2f}%")
        print(f"  Average anomaly magnitude: {stat['avg_anomaly_magnitude']:.2f}%")
        print(f"  Max anomaly magnitude: {stat['max_anomaly_magnitude']:.2f}%")
        print(f"  Average volatility at anomaly: {stat['avg_volatility_at_anomaly']:.2f}%")
        print(f"  Average forward returns after anomaly:")
        for i, ret in enumerate(stat['avg_forward_returns'], 1):
            print(f"    {i} day(s): {ret:.2f}%")

    # Create and show dashboard
    print("\nCreating enhanced dashboard...")
    fig = detector.create_enhanced_dashboard()
    fig.show()

    # Save dashboard as HTML
    fig.write_html("enhanced_crypto_anomaly_dashboard.html")
    print("\nDashboard saved as 'enhanced_crypto_anomaly_dashboard.html'")


if __name__ == "__main__":
    main()