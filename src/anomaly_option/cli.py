#!/usr/bin/env python3
"""
Command-line interface for the anomaly_option package.
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

from .core.config import Config
from .core.analysis_system import AnalysisSystem

def setup_logging(debug: bool = False) -> None:
    """Set up logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/anomaly_option.log"),
        ],
    )

def parse_args() -> argparse.Namespace:
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

def main() -> int:
    """Main entry point for the CLI."""
    args = parse_args()
    setup_logging(args.debug)
    
    try:
        # Initialize configuration
        config = Config.from_args(args)
        
        # Create output directories
        Path("logs").mkdir(exist_ok=True)
        Path(config.results_dir).mkdir(exist_ok=True)
        
        # Initialize and run analysis system
        system = AnalysisSystem(config)
        
        if args.load_results:
            system.load_results()
        else:
            system.run_analysis()
        
        if not args.skip_visualizations:
            system.generate_visualizations()
        
        return 0
        
    except Exception as e:
        logging.error(f"Error running analysis: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main()) 