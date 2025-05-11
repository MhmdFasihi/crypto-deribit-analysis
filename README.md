# Crypto-Deribit-Analysis

A comprehensive system for analyzing cryptocurrency price volatility and options data from the Deribit exchange.

## Features

- **Price Data Analysis**: Fetch and analyze historical cryptocurrency price data
- **Volatility Analysis**: Calculate various volatility metrics (Close-to-Close, Parkinson, Garman-Klass)
- **Anomaly Detection**: Identify anomalous price and volatility patterns using statistical methods
- **Options Data Analysis**: Fetch and analyze cryptocurrency options data from Deribit
- **Options Greeks**: Calculate and analyze option Greeks (Delta, Gamma, Vega, Theta)
- **IV Surface**: Generate and visualize implied volatility surfaces and skews
- **Interactive Visualizations**: Create interactive dashboards and plots for analysis results
- **Report Generation**: Automatically generate comprehensive HTML reports

## Installation

### Prerequisites

- Python 3.10 or higher
- Conda or pip package manager

### Setup with Conda (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/MhmdFasihi/crypto-deribit-analysis.git
cd crypto-deribit-analysis
```

2. Create and activate a conda environment:
```bash
conda env create -f environment.yml
conda activate crypto_deribit
```

3. Create a `.env` file with your Deribit API credentials (copy from `.env.example`):
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

### Alternative Setup with pip

1. Clone the repository:
```bash
git clone https://github.com/MhmdFasihi/crypto-deribit-analysis.git
cd crypto-deribit-analysis
```

2. Set up a virtual environment:
```bash
python -m venv .mh_env
source .mh_env/bin/activate  # On Windows, use: .mh_env\Scripts\activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your Deribit API credentials (copy from `.env.example`):
```bash
cp .env.example .env
# Edit .env with your API keys and settings
```

## Usage

### Basic Usage

Run a basic analysis on default symbols:

```bash
python run_analysis.py
```

### Advanced Usage

Specify symbols and date range:

```bash
python run_analysis.py --symbols BTC-USD ETH-USD --start-date 2023-01-01 --end-date today
```

Use Deribit test environment:

```bash
python run_analysis.py --test-env
```

Skip options data analysis (price/volatility analysis only):

```bash
python run_analysis.py --skip-options
```

Load existing results and regenerate visualizations:

```bash
python run_analysis.py --load-results
```

### Command-Line Arguments

| Argument | Description |
|----------|-------------|
| `--symbols`, `-s` | Symbols to analyze (e.g., BTC-USD ETH-USD) |
| `--start-date` | Start date (format: YYYY-MM-DD or relative like "7d") |
| `--end-date` | End date (format: YYYY-MM-DD or "today") |
| `--next-window` | Number of days for the next analysis window |
| `--output-dir`, `-o` | Directory for output files |
| `--load-results`, `-l` | Load results from output directory instead of running analysis |
| `--skip-visualizations` | Skip generating visualizations |
| `--api-key` | Deribit API key (overrides environment variable) |
| `--api-secret` | Deribit API secret (overrides environment variable) |
| `--test-env` | Use Deribit test environment |
| `--skip-options` | Skip options data fetching and analysis |
| `--max-workers` | Maximum number of parallel workers |
| `--timeout` | Timeout for operations in seconds |
| `--max-retries` | Maximum number of retries |
| `--debug` | Enable debug logging |
| `--config` | Path to custom configuration file |

## Configuration

The system can be configured using environment variables (in the `.env` file) or command-line arguments. The following configuration options are available:

### API Settings

- `DERIBIT_API_KEY`: Your Deribit API key
- `DERIBIT_API_SECRET`: Your Deribit API secret
- `USE_TEST_ENV`: Whether to use the Deribit test environment (True/False)

### Analysis Parameters

- `DEFAULT_SYMBOLS`: Default symbols to analyze (comma-separated)
- `WINDOW_SIZE`: Window size for rolling calculations
- `Z_THRESHOLD`: Z-score threshold for anomaly detection
- `VOLATILITY_WINDOW`: Window size for volatility calculations
- `ANNUALIZATION_FACTOR`: Factor to annualize volatility
- `RISK_FREE_RATE`: Risk-free rate for options pricing

### Performance Settings

- `MAX_REQUESTS_PER_SECOND`: Maximum API requests per second
- `MAX_RETRIES`: Maximum number of retries for API requests
- `RETRY_DELAY`: Delay between retries (in seconds)
- `TIMEOUT`: Timeout for operations (in seconds)
- `MAX_WORKERS`: Maximum number of parallel workers
- `CHUNK_SIZE`: Number of rows to process at once

### Output Settings

- `CACHE_DIR`: Directory for caching options data
- `RESULTS_DIR`: Directory for output files
- `LOG_DIR`: Directory for log files
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Project Structure

```
crypto-deribit-analysis/
├── analysis_system.py       # Main analysis system
├── config.py                # Configuration management
├── crypto_data_fetcher.py   # Data fetching module
├── options_analyzer.py      # Options analysis module
├── requirements.txt         # Project dependencies (pip)
├── environment.yml          # Conda environment specification
├── run_analysis.py          # Command-line interface
├── visualizer.py            # Visualization module
├── volatility_analyzer.py   # Volatility analysis module
├── .env.example             # Example environment variables
├── .gitignore               # Git ignore file
└── README.md                # This file
```

## Output Files

The system generates the following output files:

- **HTML Dashboards**: Interactive dashboards for each symbol
- **CSV Files**: Raw data and analysis results
- **JSON Files**: Options analysis results and metadata
- **HTML Report**: Comprehensive analysis report

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

- Deribit Exchange for providing the API
- yfinance for price data access
- Plotly for visualization capabilities