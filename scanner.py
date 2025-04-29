### You will need to create a config file to call your API keys 
## If you need help with building a config file feel free to reach out  

from trading_bot import TradingBot
from config import *  # Load API keys and SYMBOLS
import pandas as pd
import numpy as np
import datetime as dt
import logging
import time
import json
import csv
import os
from typing import Dict, List, Tuple, Optional, Union
from functools import lru_cache

# Logging config
logging.basicConfig(
    level=logging.INFO,  # Set to INFO by default, can be overridden with --debug flag
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("hybrid_scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HybridScanner")

# Top volume tickers for quick scan
TOP_TICKERS = [
    'AAPL', 'MSFT', 'TSLA', 'NVDA', 'AMD', 'META', 'AMZN', 'SPY', 'GOOG', 'NFLX',
    'BAC', 'DIS', 'INTC', 'JPM', 'QCOM', 'PYPL', 'V', 'MA', 'SOFI', 'NIO',
    'PLTR', 'COIN', 'UBER', 'CSCO', 'F', 'PFE', 'HOOD', 'WMT', 'BABA', 'KO'
]

# Adding pre-screened stocks from tier data
# Tier 1 - Volatility Small Cap
TIER1_VOLATILITY_TICKERS = [
    'PDYN', 'AVTE', 'LTBR', 'RDW', 'RAY', 'THTX', 'FNA', 'BLBX', 'AAOI', 'XLO',
    'CRNC', 'NAOV', 'ARBB', 'SPWH', 'PSTV', 'SOPA', 'OST', 'LIPO', 'WLGS', 'QNTM',
    'SPRC', 'JYD', 'STAI', 'PTN', 'KITT'
]

# Tier 2 - Volatility Mid Cap
TIER2_VOLATILITY_TICKERS = [
    'ARLP', 'HCC', 'BTU', 'ARNC', 'X', 'CLF', 'SM', 'PBF', 'CNX', 'AR',
    'XOM', 'CF', 'SWN', 'APA', 'MOS', 'VAL', 'ZIM', 'MTDR', 'CEIX', 'DAR',
    'NUE', 'HUN', 'DDS', 'NWL', 'F'
]

# Tier 3 - Trend Large Cap
TIER3_TREND_TICKERS = [
    'MPC', 'LMT', 'COP', 'HAL', 'DAR', 'DVN', 'AAL', 'MOS', 'MAR', 'CAT',
    'ABBV', 'TRV', 'MET', 'DE', 'WM', 'LNG', 'DHI', 'PFE', 'MS', 'BMY',
    'GS', 'JPM', 'HON', 'CSCO', 'INTC'
]

# Tier 4 - Trend Mega Cap
TIER4_TREND_TICKERS = [
    'OXY', 'CVX', 'XOM', 'LNG', 'NOC', 'ADM', 'MRK', 'ABBV', 'HES', 'SLB',
    'BA', 'SCHW', 'UNH', 'LRCX', 'CAT', 'BRK.B', 'CVS', 'MSFT', 'AAPL', 'ETN',
    'GS', 'JPM', 'WMT', 'PG'
]

# Cache directory for API responses
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_result(func):
    """Decorator to cache API results to disk"""
    def wrapper(*args, **kwargs):
        # Create a cache key based on function name and arguments
        cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
        cache_file = os.path.join(CACHE_DIR, f"{hash(cache_key)}.json")
        
        # Check if cache exists and is less than 30 minutes old
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < 1800:  # 30 minutes
                try:
                    with open(cache_file, 'r') as f:
                        logger.debug(f"Using cached result for {func.__name__}")
                        return json.load(f)
                except (json.JSONDecodeError, IOError):
                    logger.debug(f"Cache file corrupted for {func.__name__}, refreshing")
        
        # Call the function and save results to cache
        result = func(*args, **kwargs)
        try:
            with open(cache_file, 'w') as f:
                json.dump(result, f)
        except (IOError, TypeError):
            logger.warning(f"Could not cache result for {func.__name__}")
        
        return result
    return wrapper

class DataManager:
    """Class to manage data retrieval and caching"""
    
    def __init__(self, bot: object):
        self.bot = bot
        self._batch_data = {}  # Store batch data
    
    def batch_get_bars(self, symbols: List[str], timeframe: str = "1Min", limit: int = 5) -> Dict[str, pd.DataFrame]:
        """Batch request for bars data"""
        if not symbols:
            return {}
            
        # Check if we've already requested this batch
        batch_key = f"{'-'.join(sorted(symbols))}-{timeframe}-{limit}"
        if batch_key in self._batch_data:
            return self._batch_data[batch_key]
            
        # Split into chunks to avoid API limits
        chunk_size = 10  # Smaller chunk size to be safe
        results = {}
        
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            
            # Process each symbol individually using the bot's methods
            for symbol in chunk:
                try:
                    # Use _load_data_with_cache method from TradingBot
                    bars = self.bot._load_data_with_cache(symbol, timeframe)
                    if not bars.empty:
                        results[symbol] = bars
                except Exception as e:
                    logger.debug(f"Failed to get bars for {symbol}: {e}")
                    continue
                    
            # Respect API rate limits
            if i + chunk_size < len(symbols):
                time.sleep(1)  # 1s pause to avoid rate limits
                
        # Cache the results
        self._batch_data[batch_key] = results
        return results


def volume_scanner(data_manager: DataManager, scan_symbols: List[str], timeframe: str = "1Min", bar_limit: int = 5) -> List[str]:
    """Scan tickers by volume using batch requests"""
    print(f"\n🔍 Scanning {len(scan_symbols)} tickers by volume...")
    
    # Get data in batch
    batch_bars = data_manager.batch_get_bars(scan_symbols, timeframe, bar_limit)
    
    volume_data = []
    for symbol, bars in batch_bars.items():
        if bars is None or bars.empty:
            continue
        total_volume = bars['volume'].sum()
        volume_data.append({'symbol': symbol, 'volume': total_volume})
    
    # Process any tickers that failed in the batch request individually
    missed_tickers = set(scan_symbols) - set(batch_bars.keys())
    if missed_tickers:
        logger.debug(f"Processing {len(missed_tickers)} tickers individually that failed in batch")
        for symbol in missed_tickers:
            try:
                # Use _load_data_with_cache from the bot
                bars = data_manager.bot._load_data_with_cache(symbol, timeframe)
                total_volume = bars['volume'].sum() if not bars.empty else 0
                volume_data.append({'symbol': symbol, 'volume': total_volume})
            except Exception as e:
                logger.debug(f" Skipped {symbol}: {e}")
                continue
    
    df_vol = pd.DataFrame(volume_data)
    df_vol = df_vol[df_vol['volume'] > 0]
    top = df_vol.sort_values('volume', ascending=False)
    
    print("\n📊 Top Volume Tickers:")
    print(top.to_string(index=False))
    
    return top['symbol'].tolist()


class StockScanner:
    def __init__(self, bot: object, config_path: Optional[str] = None):
        """
        Initialize scanner with trading bot and optional config path
        """
        self.bot = bot
        self.config_path = config_path
        
        # Updated tier configuration with much more relaxed criteria
        self.default_tiers = {
            1: {'name': 'Small Cap Volatility', 'price_min': 0.2, 'price_max': 16, 'volume': 250000, 'atr_percent': 0.005, 'max_spread': 0.005, 'float_min': 5_000_000, 'float_max': 100_000_000},
            2: {'name': 'Mid Cap Volatility', 'price_min': 14, 'price_max': 40, 'volume': 250000, 'atr_percent': 0.005, 'max_spread': 0.005, 'float_min': 30_000_000, 'float_max': 200_000_000},
            3: {'name': 'Large Cap Trend', 'price_min': 25, 'price_max': 100, 'volume': 250000, 'atr_percent': 0.005, 'max_spread': 0.005, 'float_min': 50_000_000, 'float_max': 500_000_000},
            4: {'name': 'Mega Cap Trend', 'price_min': 40, 'price_max': 1000, 'volume': 250000, 'atr_percent': 0.005, 'max_spread': 0.005, 'float_min': 100_000_000, 'float_max': 10_000_000_000},
        }
        
        # Load tiers from config if provided
        self.tiers = self._load_tier_config() if config_path else self.default_tiers
        
    def _load_tier_config(self) -> Dict:
        """Load tier configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
                logger.info(f"Loaded tier configuration from {self.config_path}")
                return config
        except (IOError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load tier config: {e}, using defaults")
            return self.default_tiers
            
    def save_tier_config(self, path: Optional[str] = None) -> bool:
        """Save current tier configuration to file"""
        save_path = path or self.config_path
        if not save_path:
            logger.warning("No config path specified")
            return False
            
        try:
            with open(save_path, 'w') as f:
                json.dump(self.tiers, f, indent=2)
            logger.info(f"Saved tier configuration to {save_path}")
            return True
        except IOError as e:
            logger.error(f"Failed to save tier config: {e}")
            return False

    def calculate_atr(self, df: pd.DataFrame, period: int = 30) -> float:
        """Calculate Average True Range (ATR)"""
        if df.empty or len(df) <= 1:
            return 0.0
        
        # Vectorized ATR calculation
        high, low, close = df['high'].values, df['low'].values, df['close'].values
        
        # Handle NaN values
        if np.isnan(high).any() or np.isnan(low).any() or np.isnan(close).any():
            high = np.nan_to_num(high)
            low = np.nan_to_num(low)
            close = np.nan_to_num(close)
            
        # Need at least 2 values for TR calculation
        if len(high) < 2:
            return 0.0
            
        # Calculate TR components
        tr1 = np.abs(high[1:] - low[1:])
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(close[:-1] - low[1:])
        
        # Find maximum of the three components
        tr = np.vstack([tr1, tr2, tr3]).max(axis=0)
        
        # Use last 'period' elements or all if less than period
        actual_period = min(period, len(tr))
        if actual_period == 0:
            return 0.0
            
        return np.mean(tr[-actual_period:])

    def scan_stock(self, symbol: str) -> Dict[int, bool]:
        """
        Scan a stock against all tier criteria
        Returns dictionary of tier numbers and whether the stock matches
        """
        try:
            # Get historical data using _load_data_with_cache from TradingBot
            price_data = self.bot._load_data_with_cache(symbol, self.bot.config.tf_1h)
            
            if price_data.empty:
                logger.debug(f"[SKIP] {symbol}: Missing price data.")
                return {}
            
            # Extract quote data from the last row of price data
            try:
                last_bar = price_data.iloc[-1]
                
                # Create a quote-like dictionary from the last price bar
                current_price = last_bar['close']
                
                # Estimate bid/ask using the last bar's OHLC data
                mid_price = (last_bar['high'] + last_bar['low']) / 2
                price_range = last_bar['high'] - last_bar['low']
                
                # Estimate bid/ask as half of the price range around the mid price
                bid_price = mid_price - (price_range * 0.25)
                ask_price = mid_price + (price_range * 0.25)
                
                # If price range is too small, use a default spread
                if price_range < 0.001 * current_price:
                    bid_price = current_price * 0.999
                    ask_price = current_price * 1.001
                
                spread = (ask_price - bid_price) / current_price
                avg_volume = price_data['volume'].tail(20).mean()
                atr = self.calculate_atr(price_data)
                atr_percent = atr / current_price if current_price > 0 else 0
                
                # Estimate float using average trading volume as a proxy
                # This is not accurate but gives a rough estimate for filtering
                # Typically shares outstanding is 5-20x daily volume
                estimated_float = avg_volume * 15  # arbitrary multiplier
                
                # Log all computed metrics
                logger.info(
                    f"[METRICS] {symbol} | Price: ${current_price:.2f}, ATR%: {atr_percent:.4f}, "
                    f"Volume: {avg_volume:,.0f}, Spread: {spread:.5f}, Est. Float: {estimated_float:,.0f}"
                )
                
                # Check against all tiers
                results = {}
                failed_criteria = {}
                
                for tier_num, criteria in self.tiers.items():
                    price_match = criteria['price_min'] <= current_price <= criteria['price_max']
                    volume_match = avg_volume >= criteria['volume']
                    atr_match = atr_percent >= criteria['atr_percent']
                    spread_match = spread <= criteria['max_spread']
                    float_match = criteria['float_min'] <= estimated_float <= criteria['float_max']
                    
                    # Track failed criteria for better debugging
                    if not price_match or not volume_match or not atr_match or not spread_match or not float_match:
                        failed_criteria[tier_num] = []
                        if not price_match:
                            failed_criteria[tier_num].append(f"Price: ${current_price:.2f} not in range ${criteria['price_min']}-${criteria['price_max']}")
                        if not volume_match:
                            failed_criteria[tier_num].append(f"Volume: {avg_volume:,.0f} < {criteria['volume']:,.0f}")
                        if not atr_match:
                            failed_criteria[tier_num].append(f"ATR%: {atr_percent:.4f} < {criteria['atr_percent']:.4f}")
                        if not spread_match:
                            failed_criteria[tier_num].append(f"Spread: {spread:.5f} > {criteria['max_spread']:.5f}")
                        if not float_match:
                            failed_criteria[tier_num].append(f"Float: {estimated_float:,.0f} not in range {criteria['float_min']:,.0f}-{criteria['float_max']:,.0f}")
                    
                    # Overall match if all criteria are met
                    match = price_match and volume_match and atr_match and spread_match and float_match
                    results[tier_num] = match
                
                # Log failed criteria for debugging
                if failed_criteria:
                    for tier_num, failures in failed_criteria.items():
                        logger.info(f"[TIER {tier_num}] {symbol} failed criteria: {', '.join(failures)}")
                
                return results
                
            except (KeyError, IndexError) as e:
                logger.warning(f"[ERROR] {symbol}: Error processing data: {e}")
                return {}
                
        except Exception as e:
            logger.warning(f" Error scanning {symbol}: {e}")
            return {}

    def batch_scan_stocks(self, symbols: List[str]) -> Dict[str, Dict[int, bool]]:
        """
        Scan multiple stocks efficiently.
        Returns dict mapping symbols to their tier match results,
        and saves the scan results to disk as a JSON file.
        """
        results = {}
        all_matched_symbols = set()
        
        # Process in chunks to avoid rate limiting
        chunk_size = 10
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            
            for symbol in chunk:
                logger.info(f"🧪 Scanning {symbol}...")
                matches = self.scan_stock(symbol)
                if matches:
                    results[symbol] = matches
                    if any(matches.values()):
                        all_matched_symbols.add(symbol)
                else:
                    logger.info(f"❌ No tier match for {symbol}")
            
            # Respect API rate limits
            if i + chunk_size < len(symbols):
                time.sleep(1)

        # Save results to disk
        timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        results_file = os.path.join(CACHE_DIR, f"scan_results_{timestamp}.json")

        try:
            with open(results_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'tier_results': {
                        symbol: {str(tier): bool(val) for tier, val in tiers.items()}
                        for symbol, tiers in results.items()
                    },
                    'all_symbols': sorted(list(all_matched_symbols)),
                }, f, indent=2)
            logger.info(f"📊 Scan results saved to {results_file}")
        except IOError as e:
            logger.warning(f"⚠️ Failed to save scan results: {e}")

        return results

    def scan_for_trends(self, symbols: List[str], long_period: int = 50, short_period: int = 20) -> List[Dict]:
        """
        Identify stocks with strong trend patterns suitable for trend-following strategies.
        Uses moving average relationships and consistent price direction.
        
        Args:
            symbols: List of stock symbols to scan
            long_period: Longer-term moving average period
            short_period: Shorter-term moving average period
            
        Returns:
            List of dictionaries with symbols and scores that exhibit strong trend characteristics
        """
        logger.info(f" Scanning for trend-following candidates...")
        trend_candidates = []
        
        # Process in chunks to avoid rate limiting
        chunk_size = 10
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            
            for symbol in chunk:
                try:
                    # Get historical hourly data instead of daily
                    # Using tf_1h which we know exists in the config
                    price_data = self.bot._load_data_with_cache(symbol, self.bot.config.tf_1h)
                    
                    if price_data.empty or len(price_data) < long_period + 10:
                        logger.debug(f"[SKIP TREND] {symbol}: Insufficient historical data")
                        continue
                    
                    # Calculate moving averages
                    price_data['ma_short'] = price_data['close'].rolling(window=short_period).mean()
                    price_data['ma_long'] = price_data['close'].rolling(window=long_period).mean()
                    
                    # Get recent data (last 10 periods)
                    recent_data = price_data.iloc[-10:].copy()
                    
                    # Calculate trend strength indicators
                    # 1. Direction consistency (percentage of days price moved in same direction)
                    price_changes = recent_data['close'].diff()
                    direction_consistency = abs(price_changes.sum()) / price_changes.abs().sum() if price_changes.abs().sum() > 0 else 0
                    
                    # 2. MA relationship (short MA above long MA for uptrend)
                    ma_diff = recent_data['ma_short'] - recent_data['ma_long']
                    ma_relationship = ma_diff.mean()
                    
                    # 3. Slope of long MA
                    long_ma_values = recent_data['ma_long'].values
                    if len(long_ma_values) > 1:
                        long_ma_slope = (long_ma_values[-1] - long_ma_values[0]) / len(long_ma_values)
                    else:
                        long_ma_slope = 0
                    
                    # Normalize slope by price
                    normalized_slope = long_ma_slope / recent_data['close'].mean() if recent_data['close'].mean() > 0 else 0
                    
                    # Combine metrics into a trend score
                    trend_score = (
                        direction_consistency * 0.4 +  # Direction consistency contributes 40%
                        (1 if ma_relationship > 0 else 0) * 0.3 +  # MA relationship contributes 30%
                        (1 if normalized_slope > 0 else 0) * 0.3    # MA slope contributes 30%
                    )
                    
                    # Check volume stability (volatility isn't good for trend following)
                    volume_data = recent_data['volume']
                    volume_stability = 1 - (volume_data.std() / volume_data.mean() if volume_data.mean() > 0 else 1)
                    
                    # Combine final trend score with volume stability
                    final_score = trend_score * 0.7 + volume_stability * 0.3
                    
                    logger.debug(
                        f"[TREND] {symbol} | Direction: {direction_consistency:.2f}, "
                        f"MA Rel: {ma_relationship:.4f}, Slope: {normalized_slope:.6f}, "
                        f"Vol Stability: {volume_stability:.2f}, Score: {final_score:.2f}"
                    )
                    
                    # Classify as trend candidate if score exceeds threshold
                    # Always add the score regardless of threshold for ranking purposes
                    trend_candidates.append({
                        'symbol': symbol,
                        'score': final_score,
                        'is_candidate': final_score > 0.6  # Threshold for trend candidates
                    })
                    
                    if final_score > 0.6:
                        logger.info(f" {symbol} identified as trend candidate (score: {final_score:.2f})")
                    
                except Exception as e:
                    logger.warning(f" Error scanning {symbol} for trends: {e}")
                    continue
            
            # Respect API rate limits
            if i + chunk_size < len(symbols):
                time.sleep(1)
        
        # Sort candidates by score
        sorted_candidates = sorted(trend_candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates

    def scan_for_volatility(self, symbols: List[str], period: int = 30) -> List[Dict]:
        """
        Identify stocks with high volatility suitable for breakout strategies.
        
        Args:
            symbols: List of stock symbols to scan
            period: Period for calculating volatility metrics
            
        Returns:
            List of dictionaries with symbols and scores that exhibit high volatility
        """
        logger.info(f" Scanning for volatility breakout candidates...")
        volatility_candidates = []
        
        # Process in chunks to avoid rate limiting
        chunk_size = 10
        for i in range(0, len(symbols), chunk_size):
            chunk = symbols[i:i+chunk_size]
            
            for symbol in chunk:
                try:
                    # Get historical hourly data
                    price_data = self.bot._load_data_with_cache(symbol, self.bot.config.tf_1h)
                    
                    if price_data.empty or len(price_data) < period:
                        logger.debug(f"[SKIP VOL] {symbol}: Insufficient historical data")
                        continue
                    
                    # Calculate volatility metrics
                    # 1. ATR percentage
                    atr = self.calculate_atr(price_data, period)
                    current_price = price_data['close'].iloc[-1]
                    atr_percent = atr / current_price if current_price > 0 else 0
                    
                    # 2. Price range (daily high-low) percentage
                    recent_data = price_data.tail(period).copy()
                    daily_ranges = (recent_data['high'] - recent_data['low']) / recent_data['close']
                    avg_daily_range = daily_ranges.mean()
                    
                    # 3. Standard deviation of returns
                    returns = recent_data['close'].pct_change().dropna()
                    returns_std = returns.std()
                    
                    # 4. Volume spikes (ratio of recent volume to longer-term average)
                    recent_volume_avg = recent_data['volume'].tail(5).mean()
                    longer_volume_avg = price_data['volume'].tail(20).mean() if len(price_data) >= 20 else recent_volume_avg
                    volume_ratio = recent_volume_avg / longer_volume_avg if longer_volume_avg > 0 else 1
                    
                    # Calculate weighted volatility score (higher is better for breakout strategies)
                    volatility_score = (
                        atr_percent * 0.35 +           # ATR% contributes 35%
                        avg_daily_range * 0.25 +       # Average daily range contributes 25%
                        returns_std * 100 * 0.25 +     # Standard deviation contributes 25%
                        min(2, volume_ratio) * 0.15    # Volume ratio contributes 15% (capped at 2)
                    )
                    
                    # Normalize score between 0 and 1 for better comparison
                    # Typical max range for this calculation is around 0.1-0.2
                    normalized_score = min(1.0, volatility_score / 0.2)
                    
                    logger.debug(
                        f"[VOL] {symbol} | ATR%: {atr_percent:.4f}, Range%: {avg_daily_range:.4f}, "
                        f"StdDev: {returns_std:.4f}, Vol Ratio: {volume_ratio:.2f}, "
                        f"Score: {normalized_score:.2f}"
                    )
                    
                    # Add to candidates list with score
                    volatility_candidates.append({
                        'symbol': symbol,
                        'score': normalized_score,
                        'is_candidate': normalized_score > 0.5  # Threshold for volatility candidates
                    })
                    
                    if normalized_score > 0.5:
                        logger.info(f" {symbol} identified as volatility candidate (score: {normalized_score:.2f})")
                    
                except Exception as e:
                    logger.warning(f" Error scanning {symbol} for volatility: {e}")
                    continue
            
            # Respect API rate limits -- ALPACA kept crashing on me so stay within this parameter
            if i + chunk_size < len(symbols):
                time.sleep(1)
        
        # Sort candidates by score
        sorted_candidates = sorted(volatility_candidates, key=lambda x: x['score'], reverse=True)
        return sorted_candidates

def main():
    logger.info("Launching full hybrid scan pipeline...")

    # Initialize bot and managers
    bot = TradingBot(config_module=__import__('config'))
    data_manager = DataManager(bot)
    scanner = StockScanner(bot=bot)

    # Verify tier lists and print their contents
    print("\n==== VERIFYING TIER LISTS ====")
    
    # Check that tier lists are not empty
    if not TIER1_VOLATILITY_TICKERS:
        print("WARNING: Tier 1 volatility tickers list is empty!")
    if not TIER2_VOLATILITY_TICKERS:
        print("WARNING: Tier 2 volatility tickers list is empty!")
    if not TIER3_TREND_TICKERS:
        print("WARNING: Tier 3 trend tickers list is empty!")
    if not TIER4_TREND_TICKERS:
        print("WARNING: Tier 4 trend tickers list is empty!")
    
    print(f"Tier 1 volatility tickers ({len(TIER1_VOLATILITY_TICKERS)}): {TIER1_VOLATILITY_TICKERS}")
    print(f"Tier 2 volatility tickers ({len(TIER2_VOLATILITY_TICKERS)}): {TIER2_VOLATILITY_TICKERS}")
    print(f"Tier 3 trend tickers ({len(TIER3_TREND_TICKERS)}): {TIER3_TREND_TICKERS}")
    print(f"Tier 4 trend tickers ({len(TIER4_TREND_TICKERS)}): {TIER4_TREND_TICKERS}")
    
    # Simple direct approach - skip scoring and complex filtering
    volatility_symbols = []
    trend_symbols = []
    assignment_reasons = {}
    
    print("\n==== DIRECT ASSIGNMENT ====")
    
    # Add Tier 1 volatility stocks (high priority)
    for symbol in TIER1_VOLATILITY_TICKERS:
        if len(volatility_symbols) < 10 and symbol not in volatility_symbols:
            volatility_symbols.append(symbol)
            assignment_reasons[symbol] = "Tier 1 volatility stock (small cap)"
            print(f"Added {symbol} to volatility list: {assignment_reasons[symbol]}")
    
    # Add Tier 2 volatility stocks
    for symbol in TIER2_VOLATILITY_TICKERS:
        if len(volatility_symbols) < 10 and symbol not in volatility_symbols:
            volatility_symbols.append(symbol)
            assignment_reasons[symbol] = "Tier 2 volatility stock (mid cap)"
            print(f"Added {symbol} to volatility list: {assignment_reasons[symbol]}")
    
    # Add Tier 3 trend stocks
    for symbol in TIER3_TREND_TICKERS:
        if len(trend_symbols) < 10 and symbol not in trend_symbols:
            trend_symbols.append(symbol)
            assignment_reasons[symbol] = "Tier 3 trend stock (large cap)"
            print(f"Added {symbol} to trend list: {assignment_reasons[symbol]}")
    
    # Add Tier 4 trend stocks  
    for symbol in TIER4_TREND_TICKERS:
        if len(trend_symbols) < 10 and symbol not in trend_symbols:
            trend_symbols.append(symbol)
            assignment_reasons[symbol] = "Tier 4 trend stock (mega cap)"
            print(f"Added {symbol} to trend list: {assignment_reasons[symbol]}")
    
    print(f"\nAfter tier assignment - Volatility: {len(volatility_symbols)}, Trend: {len(trend_symbols)}")
    
    # Check for overlap (should be none, but just in case)
    common = set(trend_symbols).intersection(set(volatility_symbols))
    if common:
        print(f"WARNING: Found overlap between lists: {common}")
        # Remove from volatility list
        for symbol in common:
            volatility_symbols.remove(symbol)
            print(f"Removed {symbol} from volatility list (duplicate)")
    
    # Fill remaining slots from TOP_TICKERS
    for symbol in TOP_TICKERS:
        if symbol not in volatility_symbols and symbol not in trend_symbols:
            if len(volatility_symbols) < 10:
                volatility_symbols.append(symbol)
                assignment_reasons[symbol] = "Top volume stock (volatility)"
                print(f"Added {symbol} to volatility list: {assignment_reasons[symbol]}")
            elif len(trend_symbols) < 10:
                trend_symbols.append(symbol)
                assignment_reasons[symbol] = "Top volume stock (trend)"
                print(f"Added {symbol} to trend list: {assignment_reasons[symbol]}")
        
        # Stop if both lists are full
        if len(volatility_symbols) >= 10 and len(trend_symbols) >= 10:
            break
    
    # Ensure lists have exactly 10 symbols
    volatility_symbols = volatility_symbols[:10]
    trend_symbols = trend_symbols[:10]
    
    print(f"\n==== FINAL LISTS ====")
    print(f"Volatility symbols ({len(volatility_symbols)}): {volatility_symbols}")
    print(f"Trend symbols ({len(trend_symbols)}): {trend_symbols}")
    
    # Print tier scan results (informational only)
    print("\n" + "="*80)
    print("COPY AND PASTE CONFIG OUTPUTS:")
    print("="*80)
    
    # Trend-following bot symbols output
    print("\n# For trend-following bot")
    print(f"SYMBOLS_TREND = {trend_symbols}")
    
    # Volatility breakout bot symbols output
    print("\n# For volatility breakout bot")
    print(f"SYMBOLS_VOLATILITY = {volatility_symbols}")
    
    print("\n" + "="*80)

    # Step 5: Save JSON results
    timestamp = dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_file = os.path.join(CACHE_DIR, f"scan_results_{timestamp}.json")

    try:
        with open(results_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'volatility_symbols': volatility_symbols,
                'trend_symbols': trend_symbols,
                'assignment_reasons': assignment_reasons
            }, f, indent=2)
        logger.info(f"📊 JSON scan results saved to {results_file}")
    except IOError as e:
        logger.warning(f"Failed to save JSON scan results: {e}")

    # Step 6: Save CSV results
    csv_file = os.path.join(CACHE_DIR, f"scan_results_{timestamp}.csv")
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['symbol', 'strategy', 'reason'])

            # Add trend symbols
            for symbol in trend_symbols:
                writer.writerow([
                    symbol, 
                    'trend', 
                    assignment_reasons.get(symbol, 'Unknown')
                ])
                
            # Add volatility symbols
            for symbol in volatility_symbols:
                writer.writerow([
                    symbol, 
                    'volatility', 
                    assignment_reasons.get(symbol, 'Unknown')
                ])

        logger.info(f"📁 CSV scan results saved to {csv_file}")
    except Exception as e:
        logger.warning(f" Failed to save CSV scan results: {e}")

# Execute the main function when the script is run
if __name__ == "__main__":
    main()
