#!/usr/bin/env python3
"""
Main script to run the crypto-deribit-analysis system.
Handles command-line arguments and launches the analysis pipeline.
"""

import argparse
import sys
import os
from datetime import date, datetime, timedelta
import json
import traceback
from pathlib import Path

# Add the project directory to the Python path to allow imports
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

# Import the configuration first to set up logging
from config import Config, logger

# Import analysis components
from analysis_system import VolatilityOptionsAnalysisSystem


def parse_date(date_str: str) -> date:
    """Parse date string in various formats."""
    try:
        # Try different date formats
        for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%m/%d/%Y', '%d-%m-%Y'):
            try:
                return datetime.strptime(date_str, fmt).date()
            except ValueError:
                continue
        
        # If it's a relative date like "today", "yesterday", etc.
        today = date.today()
        if date_str.lower() == 'today':
            return today
        elif date_str.lower() == 'yesterday':
            return today - timedelta(days=1)
        elif date_str.lower() == 'tomorrow':
            return today + timedelta(days=1)
        elif date_str.lower().endswith('d') or date_str.lower().endswith('days'):
            # Handle "7d" or "7days" format
            days = int(date_str.lower().replace('days', '').replace('d', ''))
            return today - timedelta(days=days)
        
        raise ValueError(f"Unrecognized date format: {date_str}")
    except Exception as e:
        raise ValueError(f"Error parsing date '{date_str}': {str(e)}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Crypto Volatility and Options Analysis System')
    
    # Basic arguments
    parser.add_argument('--symbols', '-s', type=str, nargs='+', 
                        help='Symbols to analyze (e.g., BTC-USD ETH-USD)')
    parser.add_argument('--start-date', type=str, 
                        help='Start date (format: YYYY-MM-DD or relative like "7d")')
    parser.add_argument('--end-date', type=str, 
                        help='End date (format: YYYY-MM-DD or "today")')
    parser.add_argument('--next-window', type=int, default=30,
                        help='Number of days for the next analysis window')
    
    # Output options
    parser.add_argument('--output-dir', '-o', type=str,
                        help='Directory for output files')
    parser.add_argument('--load-results', '-l', action='store_true',
                        help='Load results from output directory instead of running analysis')
    parser.add_argument('--skip-visualizations', action='store_true',
                        help='Skip generating visualizations')
    
    # API and environment options
    parser.add_argument('--api-key', type=str,
                        help='Deribit API key (overrides environment variable)')
    parser.add_argument('--api-secret', type=str,
                        help='Deribit API secret (overrides environment variable)')
    parser.add_argument('--test-env', action='store_true',
                        help='Use Deribit test environment')
    parser.add_argument('--skip-options', action='store_true',
                        help='Skip options data fetching and analysis')
    
    # Performance options
    parser.add_argument('--max-workers', type=int,
                        help='Maximum number of parallel workers')
    parser.add_argument('--timeout', type=int,
                        help='Timeout for operations in seconds')
    parser.add_argument('--max-retries', type=int,
                        help='Maximum number of retries')
    
    # Debug options
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--config', type=str,
                        help='Path to custom configuration file')
    
    return parser.parse_args()


def main():
    """Main function to run the analysis pipeline."""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Set log level based on debug flag
    if args.debug:
        logger.setLevel('DEBUG')
        logger.debug("Debug logging enabled")
    
    # Log startup information
    logger.info("Starting Crypto Volatility and Options Analysis System")
    
    try:
        # Process dates
        start_date = None
        if args.start_date:
            start_date = parse_date(args.start_date)
        
        end_date = None
        if args.end_date:
            end_date = parse_date(args.end_date)
        
        # Process symbols
        symbols = args.symbols
        if not symbols:
            logger.info(f"No symbols specified, using default: {Config.DEFAULT_SYMBOLS}")
            symbols = Config.DEFAULT_SYMBOLS
        
        # Initialize the analysis system
        system = VolatilityOptionsAnalysisSystem(
            api_key=args.api_key,
            api_secret=args.api_secret,
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            next_window=args.next_window,
            max_workers=args.max_workers,
            timeout=args.timeout,
            max_retries=args.max_retries,
            output_dir=args.output_dir,
            use_test_env=args.test_env,
            fetch_options=not args.skip_options
        )
        
        # Run the analysis or load existing results
        if args.load_results:
            logger.info("Loading existing analysis results")
            success = system.load_analysis_results()
            
            if not success:
                logger.error("Failed to load analysis results")
                return 1
            
            logger.info("Successfully loaded analysis results")
            
            # Generate visualizations if not skipped
            if not args.skip_visualizations:
                logger.info("Generating visualizations from loaded results")
                system.generate_visualizations()
                system.generate_report()
            
        else:
            # Run the full analysis pipeline
            logger.info("Running full analysis pipeline")
            summary = system.run_analysis()
            
            if 'error' in summary:
                logger.error(f"Analysis pipeline failed: {summary['error']}")
                return 1
            
            # Print summary
            logger.info("Analysis pipeline completed successfully")
            logger.info(f"Analyzed {len(summary['symbols_analyzed'])} symbols from {summary['period']['start_date']} to {summary['period']['end_date']}")
            logger.info(f"Generated {summary['results']['volatility']['count']} volatility and {summary['results']['options']['count']} options analyses")
            logger.info(f"Report saved to {summary['output']['report']}")
            logger.info(f"Elapsed time: {summary['elapsed_time']:.2f} seconds")
            
            # Save summary to file
            summary_path = os.path.join(system.output_dir, "analysis_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"Summary saved to {summary_path}")
        
        logger.info("Analysis system completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)