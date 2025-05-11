import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import shutil
from pathlib import Path
import time

from crypto_data_fetcher import CryptoDataFetcher
from volatility_analyzer import CryptoVolatilityAnalyzer
from options_analyzer import OptionsAnalyzer
from visualizer import CryptoVolatilityOptionsVisualizer

class VolatilityOptionsAnalysisSystem:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: List[str],
        start_date: date,
        end_date: date,
        next_window: int = 30,
        max_workers: int = 4,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: int = 1,
        output_dir: str = "analysis_results"
    ) -> None:
        """Initialize the analysis system."""
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.next_window = next_window
        self.max_workers = max_workers
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.output_dir = output_dir
        
        # Initialize components
        self.data_fetcher = CryptoDataFetcher(api_key, api_secret, max_retries, timeout)
        self.volatility_analyzer = None
        self.options_analyzer = None
        
        # Create output directory
        self._create_output_dir()
    
    def _create_output_dir(self) -> None:
        """Create output directory if it doesn't exist."""
        os.makedirs(self.output_dir, exist_ok=True)
    
    def _execute_with_timeout(self, func: callable, *args, **kwargs) -> Any:
        """Execute a function with timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except TimeoutError:
                print(f"Function {func.__name__} timed out after {self.timeout} seconds")
                return None
    
    def _retry_operation(self, func: callable, *args, **kwargs) -> Any:
        """Retry an operation with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                wait_time = self.retry_delay * (2 ** attempt)
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def fetch_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Fetch price and options data for all symbols."""
        print("Fetching price data...")
        price_data = self._retry_operation(
            self.data_fetcher.fetch_price_data,
            self.symbols,
            self.start_date,
            self.end_date
        )
        
        print("Fetching options data...")
        options_data = self._retry_operation(
            self.data_fetcher.fetch_options_data,
            self.symbols,
            self.start_date,
            self.end_date
        )
        
        return price_data, options_data
    
    def analyze_volatility(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Analyze volatility for all symbols."""
        print("Initializing volatility analyzer...")
        self.volatility_analyzer = CryptoVolatilityAnalyzer(price_data)
        
        print("Analyzing volatility...")
        return self._execute_with_timeout(self.volatility_analyzer.analyze_all_symbols)
    
    def analyze_options(self, options_data: Dict[str, pd.DataFrame], price_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Analyze options for all symbols."""
        print("Initializing options analyzer...")
        self.options_analyzer = OptionsAnalyzer(options_data, price_data)
        
        print("Analyzing options...")
        return self._execute_with_timeout(self.options_analyzer.analyze_all_symbols)
    
    def generate_plots(self, volatility_results: Dict[str, pd.DataFrame], options_results: Dict[str, Dict[str, Any]]) -> None:
        """Generate plots for analysis results."""
        print("Generating plots...")
        
        for symbol in self.symbols:
            try:
                # Create subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=(
                        f"{symbol} Price and Volatility",
                        f"{symbol} Options Volume Analysis",
                        f"{symbol} Implied Volatility Surface"
                    ),
                    vertical_spacing=0.1
                )
                
                # Add price and volatility plot
                if symbol in volatility_results:
                    df = volatility_results[symbol]
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['Close'], name="Price"),
                        row=1, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=df['RV_Composite'], name="Volatility"),
                        row=1, col=1
                    )
                
                # Add options volume plot
                if symbol in options_results:
                    volume_data = options_results[symbol]['volume_analysis']
                    fig.add_trace(
                        go.Bar(
                            x=['Call Volume', 'Put Volume'],
                            y=[volume_data['call_volume'], volume_data['put_volume']],
                            name="Options Volume"
                        ),
                        row=2, col=1
                    )
                
                # Add implied volatility surface plot
                if symbol in options_results:
                    iv_data = options_results[symbol]['iv_analysis']
                    fig.add_trace(
                        go.Scatter(
                            x=list(iv_data['iv_term_structure'].keys()),
                            y=list(iv_data['iv_term_structure'].values()),
                            name="IV Term Structure"
                        ),
                        row=3, col=1
                    )
                
                # Update layout
                fig.update_layout(
                    height=1200,
                    showlegend=True,
                    title_text=f"{symbol} Analysis Results"
                )
                
                # Save plot
                plot_path = os.path.join(self.output_dir, f"{symbol}_analysis.html")
                fig.write_html(plot_path)
                print(f"Saved plot for {symbol} to {plot_path}")
                
            except Exception as e:
                print(f"Error generating plots for {symbol}: {e}")
    
    def generate_report(self, volatility_results: Dict[str, pd.DataFrame], options_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate analysis report."""
        print("Generating report...")
        
        report_path = os.path.join(self.output_dir, "analysis_report.html")
        
        # Create backup of existing report
        if os.path.exists(report_path):
            backup_path = f"{report_path}.bak"
            shutil.copy2(report_path, backup_path)
        
        try:
            # Generate report content
            content = self._generate_report_content(volatility_results, options_results)
            
            # Write report to file
            with open(report_path, 'w') as f:
                f.write(content)
            
            print(f"Report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            print(f"Error generating report: {e}")
            # Restore from backup if available
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, report_path)
                print("Restored report from backup")
            return None
    
    def _generate_report_content(self, volatility_results: Dict[str, pd.DataFrame], options_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate HTML content for the report."""
        content = f"""
        <html>
        <head>
            <title>Crypto Volatility and Options Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #333; }}
                .section {{ margin: 20px 0; padding: 20px; border: 1px solid #ddd; }}
                .summary {{ background-color: #f9f9f9; padding: 15px; }}
                .plot {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Crypto Volatility and Options Analysis Report</h1>
            <div class="summary">
                <h2>Analysis Summary</h2>
                <p>Analysis Period: {self.start_date} to {self.end_date}</p>
                <p>Next Analysis Window: {self.next_window} days from {self.end_date}</p>
                <p>Symbols Analyzed: {', '.join(self.symbols)}</p>
            </div>
        """
        
        # Add volatility analysis section
        content += """
            <div class="section">
                <h2>Volatility Analysis</h2>
        """
        
        for symbol in self.symbols:
            if symbol in volatility_results:
                df = volatility_results[symbol]
                content += f"""
                    <h3>{symbol}</h3>
                    <p>Average Volatility: {df['RV_Composite'].mean():.2%}</p>
                    <p>Volatility Range: {df['RV_Composite'].min():.2%} to {df['RV_Composite'].max():.2%}</p>
                    <p>Number of Anomalies: {df['Combined_Anomaly'].sum()}</p>
                    <div class="plot">
                        <iframe src="{symbol}_analysis.html" width="100%" height="400px" frameborder="0"></iframe>
                    </div>
                """
        
        content += """
            </div>
        """
        
        # Add options analysis section
        content += """
            <div class="section">
                <h2>Options Analysis</h2>
        """
        
        for symbol in self.symbols:
            if symbol in options_results:
                results = options_results[symbol]
                content += f"""
                    <h3>{symbol}</h3>
                    <p>Total Options Volume: {results['volume_analysis']['total_volume']:,}</p>
                    <p>Call/Put Ratio: {results['volume_analysis']['volume_ratio']:.2f}</p>
                    <p>Average Implied Volatility: {results['iv_analysis']['mean_iv']:.2%}</p>
                    <p>IV Range: {results['iv_analysis']['mean_iv'] - results['iv_analysis']['std_iv']:.2%} to {results['iv_analysis']['mean_iv'] + results['iv_analysis']['std_iv']:.2%}</p>
                """
        
        content += """
            </div>
        """
        
        # Add recommendations section
        content += """
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    <li>Monitor high volatility periods for potential trading opportunities</li>
                    <li>Watch for unusual options volume patterns that may indicate significant price movements</li>
                    <li>Consider implied volatility skew when selecting options strategies</li>
                    <li>Use the next analysis window ({self.next_window} days) to validate current trends</li>
                </ul>
            </div>
        """
        
        content += """
            </body>
        </html>
        """
        
        return content
    
    def run_analysis(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, Any]], str]:
        """Run the complete analysis pipeline."""
        try:
            # Fetch data
            price_data, options_data = self.fetch_data()
            
            # Analyze volatility
            volatility_results = self.analyze_volatility(price_data)
            
            # Analyze options
            options_results = self.analyze_options(options_data, price_data)
            
            # Generate plots
            self.generate_plots(volatility_results, options_results)
            
            # Generate report
            report_path = self.generate_report(volatility_results, options_results)
            
            return volatility_results, options_results, report_path
            
        except Exception as e:
            print(f"Error in analysis pipeline: {e}")
            return {}, {}, None
        
        finally:
            # Cleanup
            if self.volatility_analyzer:
                self.volatility_analyzer.cleanup()
            if self.options_analyzer:
                self.options_analyzer.cleanup() 