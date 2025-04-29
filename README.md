# Hybrid Stock Scanner
My favorite day trading scanner


A high-performance stock scanning tool designed for day traders looking to identify both volatility and trend-based trading opportunities across different market capitalization tiers.

## Overview

This scanner implements a hybrid approach to stock screening by:

1. Categorizing stocks into strategic tiers based on market cap and trading characteristics
2. Analyzing stocks for volatility patterns (suitable for breakout strategies)
3. Identifying trend-following candidates with stable momentum
4. Providing automatic selection of the best candidates for each strategy type

## Features

- **Tiered Stock Classification**: Pre-categorizes stocks into 4 distinct tiers:
  - Tier 1: Small Cap Volatility ($0.2-$16)
  - Tier 2: Mid Cap Volatility ($14-$40)
  - Tier 3: Large Cap Trend ($25-$100)
  - Tier 4: Mega Cap Trend ($40-$1000)

- **Volatility Analysis**: Identifies high-ATR stocks with consistent price ranges and volume spikes

- **Trend Analysis**: Evaluates moving average relationships, price direction consistency, and volume stability

- **Smart Caching**: Implements API response caching to avoid rate limits and improve performance

- **Batch Processing**: Processes stock data in chunks to optimize API usage

- **Detailed Reporting**: Generates JSON and CSV reports with scan results and assignment reasons

## Requirements

- Python 3.6+
- Dependencies:
  - pandas
  - numpy
  - alpaca-trade-api (or other broker API)
  - logging
  - datetime
  - json
  - csv

## Setup

1. Clone this repository
```bash
git clone https://github.com/yourusername/hybrid-stock-scanner.git](https://github.com/AngelaGibson1/day_trading_scanner
cd hybrid-stock-scanner
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a `config.py` file with your API keys and settings
```python
# Example config.py structure
ALPACA_API_KEY = 'your_api_key'
ALPACA_SECRET_KEY = 'your_secret_key'
ALPACA_BASE_URL = 'https://api.alpaca.markets'

# Timeframe settings
tf_1min = '1Min'  
tf_5min = '5Min'
tf_15min = '15Min'
tf_1h = '1H'
tf_1d = '1D'

# Other configuration variables as needed
```

## Usage

Run the scanner to get trading candidates for both strategies:

```bash
python hybrid_scanner.py
```

This will:
1. Analyze pre-selected tier lists
2. Select the top 10 candidates for volatility-based trading
3. Select the top 10 candidates for trend-following strategies
4. Save results to the `cache` directory as both JSON and CSV

## Advanced Configuration

You can customize the tier criteria and stock lists by modifying the following variables in the script:

- `TOP_TICKERS`: High-volume stocks for quick scanning
- `TIER1_VOLATILITY_TICKERS`: Pre-screened small cap volatility stocks
- `TIER2_VOLATILITY_TICKERS`: Pre-screened mid cap volatility stocks
- `TIER3_TREND_TICKERS`: Pre-screened large cap trend stocks
- `TIER4_TREND_TICKERS`: Pre-screened mega cap trend stocks

The tier criteria can be adjusted in the `default_tiers` dictionary in the `StockScanner` class.

## Output

After running, the script will print configuration directives for your trading bot:

```python
# For trend-following bot
SYMBOLS_TREND = ['AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN', 'SPY', 'GOOG', 'NFLX']

# For volatility breakout bot
SYMBOLS_VOLATILITY = ['PDYN', 'AVTE', 'LTBR', 'RDW', 'RAY', 'THTX', 'FNA', 'BLBX', 'AAOI', 'XLO']
```

These can be directly copied into your trading bot configuration.

## Notes

- The scanner uses a cache directory to store API responses for 30 minutes to reduce API calls
- Adjust the tier criteria to match your trading style and risk tolerance
- Modify the scanning parameters in the `scan_for_trends` and `scan_for_volatility` methods
- The script assumes you have a `TradingBot` class implementation with `_load_data_with_cache` method

## License

[MIT License](LICENSE)

## Disclaimer

This software is for educational and informational purposes only. It is not intended to be financial advice or a recommendation to trade securities. Always do your own research and consider your risk tolerance before trading.
