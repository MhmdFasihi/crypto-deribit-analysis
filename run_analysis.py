from datetime import date, timedelta
from crypto_data_fetcher import CryptoDataFetcher
from volatility_analyzer import CryptoVolatilityAnalyzer
from options_analyzer import OptionsAnalyzer
from visualizer import CryptoVolatilityOptionsVisualizer
import traceback

def main():
    # Set up date range (last 30 days)
    end_date = date.today()
    start_date = end_date - timedelta(days=30)
    
    # Initialize data fetcher
    symbols = ['BTC-USD', 'ETH-USD']
    fetcher = CryptoDataFetcher()  # Initialize without parameters
    
    # Fetch price data
    print("Fetching data...")
    print(f"Fetching price data for {', '.join(symbols)}...")
    price_data = fetcher.fetch_price_data(
        symbols=symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Check if price data is valid
    if not price_data or all(df.empty for df in price_data.values()):
        print("Error: No price data fetched")
        return

    # Fetch options data
    print("\nFetching options data...")
    options_data = fetcher.fetch_options_data(
        symbols=symbols,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    # Initialize analyzers
    print("\nInitializing analyzers...")
    volatility_analyzer = CryptoVolatilityAnalyzer(price_data)
    options_analyzer = OptionsAnalyzer(options_data, price_data)
    
    # Analyze volatility
    print("\nAnalyzing volatility...")
    volatility_results = volatility_analyzer.analyze_all_symbols()
    
    # Analyze options
    print("\nAnalyzing options...")
    options_results = {}
    for symbol in symbols:
        try:
            if options_data[symbol].empty:
                print(f"Warning: No options data available for {symbol}")
                continue
                
            print(f"Analyzing options for {symbol}...")
            options_results[symbol] = options_analyzer.analyze_options(symbol=symbol)
            print(f"Completed options analysis for {symbol}")
            
        except Exception as e:
            print(f"Error analyzing options for {symbol}:")
            traceback.print_exc()
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualizer = CryptoVolatilityOptionsVisualizer(
        price_results=volatility_results,
        options_results=options_results,
        term_structures={},
        volatility_cones={}
    )
    
    # Generate visualizations for each symbol
    for symbol in symbols:
        try:
            print(f"\nGenerating visualizations for {symbol}...")
            # Add your visualization calls here
            # For example:
            # fig = visualizer.plot_price_volatility(symbol)
            # fig.write_html(f"results/{symbol}_price_volatility.html")
        except Exception as e:
            print(f"Error generating visualizations for {symbol}:")
            traceback.print_exc()
    
    print("\nAnalysis complete! Check the 'results' directory for visualizations.")

if __name__ == "__main__":
    main() 