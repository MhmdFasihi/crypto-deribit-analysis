#!/usr/bin/env python3
"""
Command-line interface for the Anomaly Option Analysis System.
"""

import argparse
import sys
from datetime import datetime, date, timedelta
from pathlib import Path

from anomaly_option.core.config import Config, logger
from anomaly_option.core.analysis_system import VolatilityOptionsAnalysisSystem

def parse_date(date_str):
    """Parse date string including relative dates like '7d'."""
    if not date_str:
        return None
    
    if isinstance(date_str, date):
        return date_str
        
    if date_str.lower() == 'today':
        return date.today()
    
    # Handle relative dates
    if date_str.endswith('d'):
        try:
            days = int(date_str[:-1])
            return date.today() - timedelta(days=days)
        except ValueError:
            pass
    
    # Try standard format
    try:
        return datetime.strptime(date_str, '%Y-%m-%d').date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Cryptocurrency options analysis and anomaly detection system"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--symbols",
        "-s",
        nargs="+",
        help="Symbols to analyze (e.g., BTC-USD ETH-USD)",
    )
    parser.add_argument(
        "--start-date",
        help="Start date (format: YYYY-MM-DD or relative like '7d')",
    )
    parser.add_argument(
        "--end-date",
        help="End date (format: YYYY-MM-DD or 'today')",
    )
    parser.add_argument(
        "--next-window",
        type=int,
        help="Number of days for the next analysis window",
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir",
        "-o",
        help="Directory for output files",
    )
    parser.add_argument(
        "--load-results",
        "-l",
        action="store_true",
        help="Load results from output directory instead of running analysis",
    )
    parser.add_argument(
        "--skip-visualizations",
        action="store_true",
        help="Skip generating visualizations",
    )
    
    # API settings
    parser.add_argument(
        "--api-key",
        help="Deribit API key (overrides environment variable)",
    )
    parser.add_argument(
        "--api-secret",
        help="Deribit API secret (overrides environment variable)",
    )
    parser.add_argument(
        "--test-env",
        action="store_true",
        help="Use Deribit test environment",
    )
    
    # Analysis options
    parser.add_argument(
        "--skip-options",
        action="store_true",
        help="Skip options data fetching and analysis",
    )
    
    # Performance settings
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout for operations in seconds",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of retries",
    )
    
    # Debug options
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--config",
        help="Path to custom configuration file",
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    try:
        # Initialize configuration
        config = Config.from_args(args)
        
        # Create output directories
        Path("logs").mkdir(exist_ok=True)
        Path(config.RESULTS_DIR).mkdir(exist_ok=True)
        
        # Initialize and run analysis system
        system = VolatilityOptionsAnalysisSystem(
            api_key=args.api_key,
            api_secret=args.api_secret,
            symbols=args.symbols,
            start_date=parse_date(args.start_date),
            end_date=parse_date(args.end_date),
            next_window=args.next_window,
            max_workers=args.max_workers,
            timeout=args.timeout,
            max_retries=args.max_retries,
            output_dir=args.output_dir,
            use_test_env=args.test_env,
            fetch_options=not args.skip_options
        )
        
        if args.load_results:
            system.load_results()
        else:
            system.run_analysis()
        
        if not args.skip_visualizations:
            system.generate_visualizations()
        
        return 0
        
    except Exception as e:
        logger.error(f"Error running analysis: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
