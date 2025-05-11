"""
Unified analysis system module for cryptocurrency volatility and options analysis.
Orchestrates the complete analysis pipeline from data fetching to visualization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
import time
import os
import json
import gc
import logging
import traceback
from pathlib import Path

from config import Config, logger
from crypto_data_fetcher import CryptoDataFetcher
from volatility_analyzer import CryptoVolatilityAnalyzer
from options_analyzer import OptionsAnalyzer
from visualizer import CryptoVolatilityOptionsVisualizer

class AnalysisSystemError(Exception):
    """Exception raised for errors in the analysis system."""
    pass

class VolatilityOptionsAnalysisSystem:
    """
    Unified system for cryptocurrency volatility and options analysis.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        next_window: int = 30,
        max_workers: Optional[int] = None,
        timeout: Optional[int] = None,
        max_retries: Optional[int] = None,
        retry_delay: Optional[int] = None,
        output_dir: Optional[str] = None,
        use_test_env: Optional[bool] = None,
        fetch_options: bool = True
    ) -> None:
        """
        Initialize the analysis system.
        
        Args:
            api_key: Deribit API key (optional, defaults to config)
            api_secret: Deribit API secret (optional, defaults to config)
            symbols: List of symbols to analyze (optional, defaults to config)
            start_date: Start date (optional, defaults to 30 days ago)
            end_date: End date (optional, defaults to today)
            next_window: Number of days for the next analysis window
            max_workers: Maximum number of parallel workers
            timeout: Timeout for operations in seconds
            max_retries: Maximum number of retries
            retry_delay: Delay between retries in seconds
            output_dir: Directory for output files
            use_test_env: Whether to use the test environment
            fetch_options: Whether to fetch options data
        """
        # Set parameters from arguments or config
        self.api_key = api_key or Config.API_KEY
        self.api_secret = api_secret or Config.API_SECRET
        self.symbols = symbols or Config.DEFAULT_SYMBOLS
        
        # Handle dates
        if end_date is None:
            self.end_date = date.today()
        elif isinstance(end_date, str):
            self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        else:
            self.end_date = end_date
            
        if start_date is None:
            self.start_date = self.end_date - timedelta(days=30)
        elif isinstance(start_date, str):
            self.start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        else:
            self.start_date = start_date
        
        self.next_window = next_window
        self.max_workers = max_workers or Config.MAX_WORKERS
        self.timeout = timeout or Config.TIMEOUT
        self.max_retries = max_retries or Config.MAX_RETRIES
        self.retry_delay = retry_delay or Config.RETRY_DELAY
        self.output_dir = Path(output_dir) if output_dir else Config.RESULTS_DIR
        self.use_test_env = use_test_env if use_test_env is not None else Config.USE_TEST_ENV
        self.fetch_options = fetch_options
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize components
        self.data_fetcher = None
        self.volatility_analyzer = None
        self.options_analyzer = None
        self.visualizer = None
        
        # Results storage
        self.price_data = {}
        self.options_data = {}
        self.volatility_results = {}
        self.options_results = {}
        
        # Validate inputs
        self._validate_inputs()
        
        logger.info(
            f"Initialized VolatilityOptionsAnalysisSystem for {len(self.symbols)} symbols "
            f"from {self.start_date} to {self.end_date}"
        )
    
    def _validate_inputs(self) -> None:
        """Validate input parameters."""
        if not self.symbols:
            raise ValueError("No symbols specified")
        
        if self.start_date > self.end_date:
            raise ValueError("Start date must be before end date")
        
        if self.next_window <= 0:
            raise ValueError("Next window must be positive")
        
        if self.max_workers <= 0:
            raise ValueError("Max workers must be positive")
        
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")
        
        if self.max_retries < 0:
            raise ValueError("Max retries must be non-negative")
        
        if self.retry_delay <= 0:
            raise ValueError("Retry delay must be positive")
    
    def _execute_with_timeout(self, func: callable, *args, **kwargs) -> Any:
        """
        Execute a function with timeout.
        
        Args:
            func: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result or None if timed out
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.timeout)
            except TimeoutError:
                logger.error(f"Function {func.__name__} timed out after {self.timeout} seconds")
                return None
            except Exception as e:
                logger.error(f"Error executing {func.__name__}: {str(e)}")
                logger.debug(traceback.format_exc())
                return None
    
    def _retry_operation(self, func: callable, *args, **kwargs) -> Any:
        """
        Retry an operation with exponential backoff.
        
        Args:
            func: Function to retry
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self._execute_with_timeout(func, *args, **kwargs)
            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"Operation {func.__name__} failed after {self.max_retries} retries: {str(e)}")
                    raise
                
                wait_time = self.retry_delay * (2 ** attempt)
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
    
    def fetch_data(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Fetch price and options data for all symbols.
        
        Returns:
            Tuple of (price_data, options_data)
        """
        try:
            logger.info("Initializing data fetcher")
            self.data_fetcher = CryptoDataFetcher(
                api_key=self.api_key,
                api_secret=self.api_secret,
                max_retries=self.max_retries,
                timeout=self.timeout,
                use_test_env=self.use_test_env
            )
            
            logger.info(f"Fetching data for {len(self.symbols)} symbols")
            
            # Use retry operation for data fetching
            price_data, options_data = self._retry_operation(
                self.data_fetcher.fetch_all_data,
                symbols=self.symbols,
                start_date=self.start_date,
                end_date=self.end_date,
                fetch_options=self.fetch_options
            )
            
            # Store the results
            self.price_data = price_data
            self.options_data = options_data
            
            # Log summary
            symbols_with_price = [symbol for symbol, df in price_data.items() if not df.empty]
            symbols_with_options = [symbol for symbol, df in options_data.items() if not df.empty]
            
            logger.info(f"Fetched price data for {len(symbols_with_price)}/{len(self.symbols)} symbols")
            if self.fetch_options:
                logger.info(f"Fetched options data for {len(symbols_with_options)}/{len(self.symbols)} symbols")
            
            return price_data, options_data
            
        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisSystemError(f"Failed to fetch data: {str(e)}")
    
    def analyze_volatility(self) -> Dict[str, pd.DataFrame]:
        """
        Analyze volatility for all symbols.
        
        Returns:
            Dictionary mapping symbols to volatility results
        """
        try:
            if not self.price_data:
                logger.error("No price data available, fetch data first")
                raise AnalysisSystemError("No price data available, fetch data first")
            
            logger.info("Initializing volatility analyzer")
            self.volatility_analyzer = CryptoVolatilityAnalyzer(
                price_data=self.price_data,
                window_size=Config.WINDOW_SIZE,
                z_threshold=Config.Z_THRESHOLD,
                volatility_window=Config.VOLATILITY_WINDOW,
                annualization_factor=Config.ANNUALIZATION_FACTOR,
                chunk_size=Config.CHUNK_SIZE
            )
            
            logger.info("Analyzing volatility for all symbols")
            
            # Use timeout for volatility analysis
            volatility_results = self._execute_with_timeout(
                self.volatility_analyzer.analyze_all_symbols
            )
            
            if volatility_results is None:
                logger.error("Volatility analysis timed out")
                raise AnalysisSystemError("Volatility analysis timed out")
            
            # Store the results
            self.volatility_results = volatility_results
            
            # Log summary
            symbols_with_results = [symbol for symbol, df in volatility_results.items() if not df.empty]
            logger.info(f"Completed volatility analysis for {len(symbols_with_results)}/{len(self.symbols)} symbols")
            
            return volatility_results
            
        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisSystemError(f"Failed to analyze volatility: {str(e)}")
    
    def analyze_options(self) -> Dict[str, Dict[str, Any]]:
        """
        Analyze options for all symbols.
        
        Returns:
            Dictionary mapping symbols to options results
        """
        try:
            if not self.options_data:
                if not self.fetch_options:
                    logger.info("Options data fetching is disabled, skipping options analysis")
                    return {}
                else:
                    logger.error("No options data available, fetch data first")
                    raise AnalysisSystemError("No options data available, fetch data first")
            
            if not self.price_data:
                logger.error("No price data available, fetch data first")
                raise AnalysisSystemError("No price data available, fetch data first")
            
            logger.info("Initializing options analyzer")
            self.options_analyzer = OptionsAnalyzer(
                options_data=self.options_data,
                price_data=self.price_data,
                risk_free_rate=Config.RISK_FREE_RATE
            )
            
            logger.info("Analyzing options for all symbols")
            
            # Use timeout for options analysis
            options_results = self._execute_with_timeout(
                self.options_analyzer.analyze_all_symbols
            )
            
            if options_results is None:
                logger.error("Options analysis timed out")
                raise AnalysisSystemError("Options analysis timed out")
            
            # Store the results
            self.options_results = options_results
            
            # Log summary
            symbols_with_results = [symbol for symbol, result in options_results.items() if result]
            logger.info(f"Completed options analysis for {len(symbols_with_results)}/{len(self.symbols)} symbols")
            
            return options_results
            
        except Exception as e:
            logger.error(f"Error analyzing options: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisSystemError(f"Failed to analyze options: {str(e)}")
    
    def generate_visualizations(self) -> Dict[str, Dict[str, Any]]:
        """
        Generate visualizations for all symbols.
        
        Returns:
            Dictionary mapping symbols to visualization results
        """
        try:
            if not self.volatility_results and not self.options_results:
                logger.error("No analysis results available, perform analysis first")
                raise AnalysisSystemError("No analysis results available, perform analysis first")
            
            logger.info("Initializing visualizer")
            self.visualizer = CryptoVolatilityOptionsVisualizer(
                price_results=self.volatility_results,
                options_results=self.options_results,
                output_dir=self.output_dir
            )
            
            logger.info("Generating dashboards for all symbols")
            
            # Create dashboards for all symbols
            dashboards = self.visualizer.create_all_dashboards(save_html=True)
            
            # Log summary
            logger.info(f"Created dashboards for {len(dashboards)} symbols")
            
            # Generate additional plots for each symbol
            visualization_results = {}
            
            for symbol in self.symbols:
                if symbol in self.volatility_results or symbol in self.options_results:
                    try:
                        visualization_results[symbol] = self.visualizer.generate_all_plots(symbol, save_plots=True)
                        logger.info(f"Generated all plots for {symbol}")
                    except Exception as e:
                        logger.error(f"Error generating plots for {symbol}: {str(e)}")
                        visualization_results[symbol] = {"error": str(e)}
            
            return visualization_results
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisSystemError(f"Failed to generate visualizations: {str(e)}")
    
    def generate_report(self) -> str:
        """
        Generate analysis report.
        
        Returns:
            Path to the generated report
        """
        try:
            if not self.volatility_results and not self.options_results:
                logger.error("No analysis results available, perform analysis first")
                raise AnalysisSystemError("No analysis results available, perform analysis first")
            
            # Prepare report data
            price_summary = None
            options_summary = None
            anomaly_impact = None
            
            # Get volatility summary if available
            if self.volatility_analyzer and self.volatility_results:
                price_summary = self.volatility_analyzer.get_summary_statistics()
                
                # Get anomaly impact analysis
                anomaly_impact = {}
                for symbol in self.symbols:
                    if symbol in self.volatility_results and not self.volatility_results[symbol].empty:
                        try:
                            anomaly_impact[symbol] = self.volatility_analyzer.analyze_anomaly_impact(symbol)
                        except Exception as e:
                            logger.error(f"Error analyzing anomaly impact for {symbol}: {str(e)}")
            
            # Get options summary if available
            if self.options_analyzer and self.options_results:
                options_summary = self.options_results
            
            # Create report
            if not self.visualizer:
                logger.info("Initializing visualizer")
                self.visualizer = CryptoVolatilityOptionsVisualizer(
                    price_results=self.volatility_results,
                    options_results=self.options_results,
                    output_dir=self.output_dir
                )
            
            logger.info("Generating analysis report")
            report_path = self.visualizer.create_html_report(
                price_summary=price_summary,
                options_summary=options_summary,
                anomaly_impact=anomaly_impact
            )
            
            logger.info(f"Generated analysis report: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            logger.debug(traceback.format_exc())
            raise AnalysisSystemError(f"Failed to generate report: {str(e)}")
    
    def save_analysis_results(self) -> Dict[str, str]:
        """
        Save analysis results to files.
        
        Returns:
            Dictionary mapping result types to file paths
        """
        try:
            logger.info("Saving analysis results")
            
            file_paths = {}
            
            # Create results directory if it doesn't exist
            results_dir = self.output_dir / "data"
            results_dir.mkdir(exist_ok=True, parents=True)
            
            # Save volatility results
            if self.volatility_results:
                logger.info("Saving volatility results")
                
                for symbol, df in self.volatility_results.items():
                    if not df.empty:
                        file_path = results_dir / f"{symbol}_volatility.csv"
                        df.to_csv(file_path)
                        logger.debug(f"Saved volatility results for {symbol} to {file_path}")
                
                file_paths['volatility'] = str(results_dir)
            
            # Save options results
            if self.options_results:
                logger.info("Saving options results")
                options_dir = results_dir / "options"
                options_dir.mkdir(exist_ok=True, parents=True)
                
                for symbol, results in self.options_results.items():
                    if not results:
                        continue
                    
                    # Save options data if available
                    if 'options_data' in results and not results['options_data'].empty:
                        file_path = options_dir / f"{symbol}_options_data.csv"
                        results['options_data'].to_csv(file_path)
                        logger.debug(f"Saved options data for {symbol} to {file_path}")
                    
                    # Save other results as JSON
                    results_without_df = {k: v for k, v in results.items() if k != 'options_data'}
                    if results_without_df:
                        try:
                            file_path = options_dir / f"{symbol}_options_analysis.json"
                            with open(file_path, 'w') as f:
                                json.dump(results_without_df, f, indent=2, default=str)
                            logger.debug(f"Saved options analysis for {symbol} to {file_path}")
                        except Exception as e:
                            logger.error(f"Error saving options analysis for {symbol}: {str(e)}")
                
                file_paths['options'] = str(options_dir)
            
            # Save system metadata
            metadata = {
                "analysis_time": datetime.now().isoformat(),
                "symbols": self.symbols,
                "start_date": self.start_date.isoformat(),
                "end_date": self.end_date.isoformat(),
                "next_window": self.next_window,
                "fetch_options": self.fetch_options,
                "parameters": {
                    "window_size": Config.WINDOW_SIZE,
                    "z_threshold": Config.Z_THRESHOLD,
                    "volatility_window": Config.VOLATILITY_WINDOW,
                    "annualization_factor": Config.ANNUALIZATION_FACTOR,
                    "risk_free_rate": Config.RISK_FREE_RATE
                }
            }
            
            metadata_path = self.output_dir / "analysis_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            file_paths['metadata'] = str(metadata_path)
            
            logger.info(f"Saved analysis results to {self.output_dir}")
            return file_paths
            
        except Exception as e:
            logger.error(f"Error saving analysis results: {str(e)}")
            logger.debug(traceback.format_exc())
            return {"error": str(e)}
    
    def load_analysis_results(self, input_dir: Optional[str] = None) -> bool:
        """
        Load analysis results from files.
        
        Args:
            input_dir: Directory containing the analysis results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            input_dir = Path(input_dir) if input_dir else self.output_dir
            logger.info(f"Loading analysis results from {input_dir}")
            
            # Check if metadata file exists
            metadata_path = input_dir / "analysis_metadata.json"
            if not metadata_path.exists():
                logger.error(f"Metadata file not found: {metadata_path}")
                return False
            
            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Update system parameters
            self.symbols = metadata.get('symbols', self.symbols)
            
            if 'start_date' in metadata:
                self.start_date = datetime.fromisoformat(metadata['start_date']).date()
            
            if 'end_date' in metadata:
                self.end_date = datetime.fromisoformat(metadata['end_date']).date()
            
            self.next_window = metadata.get('next_window', self.next_window)
            self.fetch_options = metadata.get('fetch_options', self.fetch_options)
            
            # Load volatility results
            results_dir = input_dir / "data"
            self.volatility_results = {}
            
            if results_dir.exists():
                logger.info("Loading volatility results")
                
                for symbol in self.symbols:
                    file_path = results_dir / f"{symbol}_volatility.csv"
                    if file_path.exists():
                        try:
                            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                            self.volatility_results[symbol] = df
                            logger.debug(f"Loaded volatility results for {symbol}")
                        except Exception as e:
                            logger.error(f"Error loading volatility results for {symbol}: {str(e)}")
            
            # Load options results
            options_dir = results_dir / "options"
            self.options_results = {}
            
            if options_dir.exists():
                logger.info("Loading options results")
                
                for symbol in self.symbols:
                    data_path = options_dir / f"{symbol}_options_data.csv"
                    analysis_path = options_dir / f"{symbol}_options_analysis.json"
                    
                    if not data_path.exists() and not analysis_path.exists():
                        continue
                    
                    self.options_results[symbol] = {}
                    
                    # Load options data
                    if data_path.exists():
                        try:
                            df = pd.read_csv(data_path, index_col=0)
                            self.options_results[symbol]['options_data'] = df
                            logger.debug(f"Loaded options data for {symbol}")
                        except Exception as e:
                            logger.error(f"Error loading options data for {symbol}: {str(e)}")
                    
                    # Load options analysis
                    if analysis_path.exists():
                        try:
                            with open(analysis_path, 'r') as f:
                                analysis = json.load(f)
                            self.options_results[symbol].update(analysis)
                            logger.debug(f"Loaded options analysis for {symbol}")
                        except Exception as e:
                            logger.error(f"Error loading options analysis for {symbol}: {str(e)}")
            
            logger.info(
                f"Loaded analysis results for {len(self.volatility_results)} volatility and "
                f"{len(self.options_results)} options symbols"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error loading analysis results: {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Returns:
            Dictionary with analysis results and paths
        """
        try:
            logger.info("Starting analysis pipeline")
            start_time = time.time()
            
            # Step 1: Fetch data
            logger.info("Step 1: Fetching data")
            self.price_data, self.options_data = self.fetch_data()
            
            # Step 2: Analyze volatility
            logger.info("Step 2: Analyzing volatility")
            self.volatility_results = self.analyze_volatility()
            
            # Step 3: Analyze options (if options data was fetched)
            if self.fetch_options:
                logger.info("Step 3: Analyzing options")
                self.options_results = self.analyze_options()
            else:
                logger.info("Step 3: Skipping options analysis (options data not fetched)")
                self.options_results = {}
            
            # Step 4: Generate visualizations
            logger.info("Step 4: Generating visualizations")
            visualization_results = self.generate_visualizations()
            
            # Step 5: Generate report
            logger.info("Step 5: Generating report")
            report_path = self.generate_report()
            
            # Step 6: Save results
            logger.info("Step 6: Saving results")
            file_paths = self.save_analysis_results()
            
            # Calculate elapsed time
            elapsed_time = time.time() - start_time
            logger.info(f"Analysis pipeline completed in {elapsed_time:.2f} seconds")
            
            # Prepare summary
            summary = {
                "symbols_analyzed": self.symbols,
                "period": {
                    "start_date": self.start_date.isoformat(),
                    "end_date": self.end_date.isoformat(),
                    "next_window": self.next_window
                },
                "results": {
                    "volatility": {
                        "symbols": list(self.volatility_results.keys()),
                        "count": len(self.volatility_results)
                    },
                    "options": {
                        "symbols": list(self.options_results.keys()),
                        "count": len(self.options_results)
                    }
                },
                "output": {
                    "report": report_path,
                    "file_paths": file_paths
                },
                "elapsed_time": elapsed_time
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        
        finally:
            # Cleanup
            self._cleanup()
    
    def _cleanup(self) -> None:
        """Clean up resources and memory."""
        logger.debug("Cleaning up resources")
        
        # Clean up components
        if self.volatility_analyzer:
            try:
                self.volatility_analyzer.cleanup()
            except:
                pass
            
        if self.options_analyzer:
            try:
                self.options_analyzer.cleanup()
            except:
                pass
            
        # Clear raw data (keep results)
        self.price_data = {}
        self.options_data = {}
        
        # Force garbage collection
        gc.collect()