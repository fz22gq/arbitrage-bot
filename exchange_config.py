"""
Exchange Configuration and Utility Functions
Handles exchange setup, API credentials, and common trading utilities.
"""

import os
import ssl

# Fix SSL certificate issues on macOS and other systems
# This MUST be done before importing requests/ccxt to ensure they use the correct certificates
try:
    import certifi
    # Use certifi's certificate bundle for SSL verification
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
except ImportError:
    # certifi not installed, will use system certificates
    pass
except Exception:
    # If there's any other issue, continue without setting env vars
    pass

# Now import other modules (they will use the SSL certificates we just configured)
import ccxt
import requests
import pytz
import datetime
import ast
from colorama import Style, Fore
from typing import Dict, List, Optional, Any

# ============================================================================
# FILE PATHS
# ============================================================================
LOGS_DIR = 'logs'
LOGS_FILE = os.path.join(LOGS_DIR, 'logs.txt')

# ============================================================================
# GENERAL CONFIGURATION
# ============================================================================
DEFAULT_RENEWAL = False
DEFAULT_DELTA_NEUTRAL = False
DEFAULT_TIMEZONE = 'Australia/Adelaide'
DEFAULT_PYTHON_COMMAND = 'python3'

renewal = True  # Enable renewal functionality for session restarts
delta_neutral = DEFAULT_DELTA_NEUTRAL
timezone = DEFAULT_TIMEZONE
python_command = DEFAULT_PYTHON_COMMAND

# ============================================================================
# EXCHANGE CONFIGURATION
# ============================================================================
# Exchange credentials - users should fill these in
# Note: Some exchanges may use different names in ccxt (e.g., 'gate' vs 'gateio')
exchanges = {
    'ascendex': {},
    'bequant': {},
    'bigone': {},
    'binance': {},
    'binanceus': {},
    'bingx': {},
    'bitfinex': {},
    'bitget': {},
    'bitflyer': {},
    'blofin': {},
    'btcmarkets': {},
    'bitmart': {},
    'bitmex': {},
    'bitopro': {},
    'bitrue': {},
    'bitstamp': {},
    'bitso': {},
    'bitvavo': {},
    'bybit': {},
    'cex': {},
    'coinbase': {},
    'coinbaseexchange': {},
    'coinex': {},
    'cryptocom': {},
    'cryptomus': {},
    'delta': {},
    'digifinex': {},
    'exmo': {},
    'fmfwio': {},
    'gate': {},  # Note: ccxt uses 'gate' not 'gateio'
    'gateio': {},  # Keep for backward compatibility
    'gemini': {},
    'hashkey': {},
    'hitbtc': {},
    'htx': {},  # Formerly Huobi
    'huobi': {},  # Keep for backward compatibility
    'hyperliquid': {},
    'indodax': {},
    'kraken': {},
    'kucoin': {},
    'latoken': {},
    'lbank': {},
    'luno': {},
    'mexc': {},
    'ndax': {},
    'okx': {},
    'onetrading': {},
    'oxfun': {},
    'p2b': {},
    'paradex': {},
    'phemex': {},
    'poloniex': {},
    'probit': {},
    'tokocrypto': {},
    'upbit': {},
    'woo': {},
    'woofipro': {},
    'xt': {},
    'zonda': {},
    # Add more exchanges as needed
    # 'another_exchange_here': {
    #     'apiKey': 'your_api_key_here',
    #     'secret': 'your_secret_here',
    # },
}

# ============================================================================
# TELEGRAM CONFIGURATION
# ============================================================================
telegram_sending = False
apiToken = 'your_telegram_bot_token_here'
chatID = 'your_telegram_chat_id_here'

# ============================================================================
# TRADING SETTINGS
# ============================================================================
# Trading criteria
criteria_pct = 0.05  # Minimum percentage difference (0.15% = require 0.15% price spread)
criteria_usd = 0.2  # Minimum USD profit (only trade if profit > $0.20 after fees)

# Trading constants
MIN_ORDER_VALUE_USD = 50.0  # Minimum order value in USD
LARGE_NUMBER = 10e13  # Used for unlimited max values
BTC_USDT_PAIR = 'BTC/USDT'  # Fallback reference pair for fees (when actual pair not found)

# Default investment amount (used if not specified via command line)
DEFAULT_TARGET_INVESTMENT_USD = 5000.0  # Default target investment in USD

# Timeout settings
first_orders_fill_timeout = 0  # Will be set to 3600 if 0

# ============================================================================
# DYNAMIC ORDER SIZING
# ============================================================================
DYNAMIC_ORDER_SIZING = True  # Enable smart order sizing based on spread and opportunity quality
MAX_ORDER_SIZE_PCT = 0.25  # Maximum order size as % of available balance (25% = risk management)
MIN_ORDER_SIZE_PCT = 0.01  # Minimum order size as % of available balance (1% = ensure meaningful trades)
SPREAD_MULTIPLIER_BASE = 1.0  # Base multiplier for order size based on spread
SPREAD_MULTIPLIER_MAX = 5.0  # Maximum multiplier for very large spreads (5x base size)
PROFIT_MULTIPLIER_BASE = 1.0  # Base multiplier for order size based on profit potential
PROFIT_MULTIPLIER_MAX = 3.0  # Maximum multiplier for very profitable opportunities (3x base size)

# ============================================================================
# RISK MANAGEMENT & REBALANCING
# ============================================================================
AUTO_REBALANCE_ON_EXIT = True  # Automatically rebalance on session exit
REBALANCE_LOSS_THRESHOLD = -0.02  # Maximum loss percentage allowed for rebalancing (-2% = allow up to 2% loss)
# If current price is more than 2% below entry price, rebalancing will be skipped with a warning

# ============================================================================
# RATE LIMITING & API OPTIMIZATION
# ============================================================================
ORDERBOOK_FETCH_DELAY = 0.1  # Minimum delay between orderbook fetches (seconds) - increase for high volatility pairs
BALANCE_CACHE_TTL = 5.0  # Cache balances for this many seconds to reduce API calls
MIN_ITERATION_DELAY = 0.05  # Minimum delay between monitoring iterations (seconds)
RATE_LIMIT_BACKOFF_BASE = 2.0  # Base multiplier for exponential backoff on rate limit errors
MAX_RATE_LIMIT_BACKOFF = 30.0  # Maximum backoff delay in seconds

# ============================================================================
# LOGGING SETTINGS
# ============================================================================
OPPORTUNITY_LOG_THROTTLE = 2.0  # Minimum seconds between logging the same opportunity
OPPORTUNITY_LOG_DEDUPE = True  # Only log when opportunity actually changes

# ============================================================================
# DEMO/SIMULATION SETTINGS
# ============================================================================
demo_fake_delay = False
demo_fake_delay_ms = 500

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================
def calculate_optimal_order_size(
    min_ask_price: float,
    max_bid_price: float,
    available_crypto: float,
    available_quote: float,
    buy_fee_rate: float,
    sell_fee_rate: float,
    min_order_value: float,
    base_order_size: float
) -> float:
    """
    Calculate optimal order size based on spread, profit potential, and available balances.
    
    Args:
        min_ask_price: Best ask price (buy price)
        max_bid_price: Best bid price (sell price)
        available_crypto: Available crypto balance on sell exchange
        available_quote: Available quote currency balance on buy exchange
        buy_fee_rate: Total fee rate for buying (base + quote fees)
        sell_fee_rate: Total fee rate for selling (base + quote fees)
        min_order_value: Minimum order value in USD
        base_order_size: Base order size to use as starting point
    
    Returns:
        Optimal order size in base currency
    """
    if not DYNAMIC_ORDER_SIZING:
        return base_order_size
    
    # Calculate spread percentage
    spread_pct = (max_bid_price - min_ask_price) / min_ask_price
    
    # Calculate profit per unit (after fees)
    profit_per_unit = max_bid_price * (1 - sell_fee_rate) - min_ask_price * (1 + buy_fee_rate)
    profit_pct = profit_per_unit / min_ask_price if min_ask_price > 0 else 0
    
    # Calculate spread multiplier (larger spreads = larger orders)
    # Normalize spread to 0-1 range (assuming spreads typically 0-5%)
    spread_normalized = min(spread_pct / 0.05, 1.0)  # Cap at 5% spread
    spread_multiplier = SPREAD_MULTIPLIER_BASE + (SPREAD_MULTIPLIER_MAX - SPREAD_MULTIPLIER_BASE) * spread_normalized
    
    # Calculate profit multiplier (higher profit = larger orders)
    # Normalize profit to 0-1 range (assuming profits typically 0-3%)
    profit_normalized = min(profit_pct / 0.03, 1.0) if profit_pct > 0 else 0  # Cap at 3% profit
    profit_multiplier = PROFIT_MULTIPLIER_BASE + (PROFIT_MULTIPLIER_MAX - PROFIT_MULTIPLIER_BASE) * profit_normalized
    
    # Combine multipliers (use the higher of the two to prioritize better opportunities)
    combined_multiplier = max(spread_multiplier, profit_multiplier)
    
    # Calculate base order size with multiplier
    optimal_size = base_order_size * combined_multiplier
    
    # Apply balance constraints
    # Maximum: use percentage of available balance
    max_from_crypto = available_crypto * MAX_ORDER_SIZE_PCT
    max_from_quote = (available_quote / min_ask_price) * MAX_ORDER_SIZE_PCT if min_ask_price > 0 else 0
    
    # Use the minimum of the two constraints (we need both crypto to sell and quote to buy)
    max_order_size = min(max_from_crypto, max_from_quote)
    
    # Minimum: ensure meaningful trade size
    min_order_size = base_order_size * MIN_ORDER_SIZE_PCT
    
    # Apply constraints
    optimal_size = max(min_order_size, min(optimal_size, max_order_size))
    
    # Ensure minimum order value is met
    order_value = optimal_size * min_ask_price
    if order_value < min_order_value and min_ask_price > 0:
        optimal_size = min_order_value / min_ask_price
    
    # Final constraint: don't exceed available balances
    optimal_size = min(optimal_size, available_crypto, available_quote / min_ask_price if min_ask_price > 0 else 0)
    
    return max(optimal_size, 0)  # Ensure non-negative

def calculate_average(values: List[float]) -> float:
    """Calculate the average of a list of values."""
    if not values:
        return 0.0
    return sum(values) / len(values)

def send_to_telegram(message: str) -> None:
    """Send a message to Telegram."""
    if not telegram_sending:
        return
    
    # Clean up formatting characters for Telegram
    clean_message = message.replace("[2m", "").replace("[0m", "").replace("[32m", "").replace("[31m", "")
    
    api_url = f'https://api.telegram.org/bot{apiToken}/sendMessage'
    
    try:
        response = requests.post(
            api_url, 
            json={'chat_id': chatID, 'text': clean_message},
            timeout=10
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to send Telegram message: {e}")

def append_list_to_file(filename: str, new_element: Any) -> None:
    """Append an element to a list stored in a file."""
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                data_list = ast.literal_eval(file.read())
        else:
            data_list = []
    except (FileNotFoundError, ValueError, SyntaxError):
        data_list = []

    data_list.append(new_element)

    try:
        with open(filename, 'w') as file:
            file.write(str(data_list))
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def append_new_line(file_name: str, text_to_append: str) -> None:
    """Append a new line to a text file, creating directories if necessary."""
    try:
        # Create directory if it doesn't exist
        dir_name = os.path.dirname(file_name)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        # Append to file
        with open(file_name, 'a+') as file_object:
            file_object.seek(0)
            data = file_object.read(100)
            if len(data) > 0:
                file_object.write('\n')
            file_object.write(text_to_append)
    except IOError as e:
        print(f"Error writing to log file {file_name}: {e}")

def printerror(**kwargs) -> None:
    """Print and log error messages with consistent formatting."""
    message = kwargs.get('m', 'Unknown error occurred')
    name_of_data = kwargs.get('name_of_data')
    data = kwargs.get('data')
    
    # Print to console
    print(f"{get_time()}{Fore.RED}{Style.BRIGHT}Error: {message}{Style.RESET_ALL}")
    
    # Log to file
    log_message = f"{get_time_blank()} ERROR: {message}"
    if name_of_data and data:
        log_message += f" | {name_of_data}: {data}"
    
    append_new_line(LOGS_FILE, log_message)

def get_balance(exchange_name: str, pair: str) -> float:
    """Get the free balance for a specific cryptocurrency on an exchange."""
    try:
        # Extract base currency from pair
        base_currency = pair.split('/')[0] if '/' in pair else pair.replace('/USDT', '')
        
        exchange_instance = get_exchange_instance(exchange_name)
        balance_info = exchange_instance.fetch_balance()
        return float(balance_info[base_currency]['free'])
    except (KeyError, ValueError, TypeError):
        return 0.0
    except Exception as e:
        printerror(m=f"Error fetching balance from {exchange_name}: {e}")
        return 0.0

def get_precision_min(pair: str, exchange_name: str) -> Optional[float]:
    """Get the minimum price precision for a trading pair on an exchange."""
    try:
        exchange_instance = get_exchange_instance(exchange_name)
        pair_info = exchange_instance.load_markets(pair)
        return pair_info[pair]['limits']['price']['min']
    except Exception as e:
        printerror(m=f"Error getting price precision for {pair} on {exchange_name}: {e}")
        return None

def get_time() -> str:
    """Get formatted timestamp with styling for console output."""
    tz = pytz.timezone(timezone)
    now = datetime.datetime.now(tz)
    timestamp = now.strftime("[%d/%m/%y %H:%M:%S]")
    return f"{Style.DIM}{timestamp}{Style.RESET_ALL} "

def get_time_blank() -> str:
    """Get formatted timestamp without styling for log files."""
    tz = pytz.timezone(timezone)
    now = datetime.datetime.now(tz)
    return now.strftime("[%d/%m/%y %H:%M:%S]")

def get_balance_usdt(exchange_list: List[str]) -> float:
    """Get total USDT balance across multiple exchanges."""
    total_balance = 0.0
    
    for exchange_name in exchange_list:
        try:
            exchange_instance = get_exchange_instance(exchange_name)
            balances = exchange_instance.fetchBalance()
            total_balance += float(balances['USDT']['free'])
        except Exception as e:
            printerror(m=f"Error fetching USDT balance from {exchange_name}: {e}")
    
    return total_balance

def cancel_all_orders(exchange_name: str, pair: str) -> bool:
    """Cancel all open orders for a pair on an exchange."""
    try:
        exchange_instance = get_exchange_instance(exchange_name)
        if not exchange_instance.has['cancelAllOrders']:
            return False
        
        open_orders = exchange_instance.fetchOpenOrders(pair)
        if not open_orders:
            return True
        
        exchange_instance.cancelAllOrders(pair)
        print(f"{get_time()}Successfully canceled all orders on {exchange_name}.")
        append_new_line(LOGS_FILE, f"{get_time_blank()} INFO: Successfully canceled all orders on {exchange_name}.")
        return True
    except Exception as e:
        printerror(m=f"Error canceling orders on {exchange_name}: {e}")
        return False

def emergency_convert_list(pair_to_sell: str, exchange_list: List[str]) -> None:
    """Emergency function to sell all cryptocurrency back to base currency."""
    for exchange_name in exchange_list:
        try:
            # Cancel any open orders first
            cancel_all_orders(exchange_name, pair_to_sell)
            
            # Get current balance
            balance = get_balance(exchange_name, pair_to_sell)
            if balance <= 0:
                print(f"{get_time()}No {pair_to_sell.split('/')[0]} balance on {exchange_name}.")
                continue
            
            # Get market info and current price
            exchange_instance = get_exchange_instance(exchange_name)
            markets = exchange_instance.load_markets()
            ticker = exchange_instance.fetch_ticker(pair_to_sell)
            current_price = float(ticker['last'])
            
            # Get trading limits
            pair_limits = markets[pair_to_sell]['limits']
            min_cost = pair_limits['cost']['min'] or 0
            min_amount = pair_limits['amount']['min'] or 0
            max_cost = pair_limits['cost']['max'] or LARGE_NUMBER
            max_amount = pair_limits['amount']['max'] or LARGE_NUMBER
            
            # Check if balance meets minimum requirements
            order_value = balance * current_price
            
            if (balance >= min_amount and order_value >= min_cost and 
                balance <= max_amount and order_value <= max_cost and
                order_value >= MIN_ORDER_VALUE_USD):
                
                # Execute market sell order
                exchange_instance.createMarketSellOrder(pair_to_sell, balance)
                
                base_currency = pair_to_sell.split('/')[0]
                print(f"{get_time()}Successfully sold {balance:.6f} {base_currency} on {exchange_name}.")
                append_new_line(LOGS_FILE, 
                    f"{get_time_blank()} INFO: Successfully sold {balance:.6f} {base_currency} on {exchange_name}."
                )
            else:
                print(f"{get_time()}Insufficient balance or order below minimum on {exchange_name}.")
                append_new_line(LOGS_FILE, 
                    f"{get_time_blank()} INFO: Insufficient balance on {exchange_name}."
                )
                
        except Exception as e:
            printerror(m=f"Error during emergency conversion on {exchange_name}: {e}")

def convert_crypto_to_quote(exchange_name: str, base_currency: str, quote_currency: str) -> bool:
    """Convert a specific cryptocurrency to quote currency on an exchange."""
    try:
        pair = f"{base_currency}/{quote_currency}"
        
        # Check if pair exists on exchange
        try:
            markets = ex[exchange_name].load_markets()
            if pair not in markets:
                return False
        except:
            return False
        
        # Cancel any open orders first
        cancel_all_orders(exchange_name, pair)
        
        # Get current balance
        balance = get_balance(exchange_name, pair)
        if balance <= 0:
            return True  # No balance to convert, consider it successful
        
        # Get market info and current price
        exchange_instance = get_exchange_instance(exchange_name)
        ticker = exchange_instance.fetch_ticker(pair)
        current_price = float(ticker['last'])
        
        # Get trading limits
        pair_limits = markets[pair]['limits']
        min_cost = pair_limits['cost']['min'] or 0
        min_amount = pair_limits['amount']['min'] or 0
        max_cost = pair_limits['cost']['max'] or LARGE_NUMBER
        max_amount = pair_limits['amount']['max'] or LARGE_NUMBER
        
        # Check if balance meets minimum requirements
        order_value = balance * current_price
        
        if (balance >= min_amount and order_value >= min_cost and 
            balance <= max_amount and order_value <= max_cost and
            order_value >= MIN_ORDER_VALUE_USD):
            
            # Execute market sell order
            exchange_instance.createMarketSellOrder(pair, balance)
            print(f"{get_time()}Converted {balance:.6f} {base_currency} to {quote_currency} on {exchange_name}.")
            append_new_line(LOGS_FILE, 
                f"{get_time_blank()} INFO: Converted {balance:.6f} {base_currency} to {quote_currency} on {exchange_name}."
            )
            return True
        else:
            print(f"{get_time()}Balance too small to convert {base_currency} on {exchange_name} (value: ${order_value:.2f}).")
            return False
            
    except Exception as e:
        printerror(m=f"Error converting {base_currency} to {quote_currency} on {exchange_name}: {e}")
        return False

def detect_and_convert_leftover_crypto(exchange_list: List[str], current_pair: str, quote_currency: str) -> None:
    """Detect and convert leftover cryptocurrency from different pairs to quote currency."""
    printandtelegram(f"{get_time()}Checking for leftover cryptocurrency from previous sessions...")
    
    # Common cryptocurrencies that might be leftover
    common_cryptos = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOT', 'MATIC', 'AVAX', 'LINK', 'UNI', 
                      'XRP', 'LTC', 'BCH', 'XLM', 'ALGO', 'ATOM', 'NEAR', 'FTM', 'EOS', 'FLR']
    
    converted_count = 0
    total_converted_value = 0.0
    
    for exchange_name in exchange_list:
        try:
            exchange_instance = get_exchange_instance(exchange_name)
            balance = exchange_instance.fetch_balance()
            
            # Check each common crypto
            for crypto in common_cryptos:
                # Skip if it's the current base currency or quote currency
                base_currency = current_pair.split('/')[0]
                if crypto == base_currency or crypto == quote_currency:
                    continue
                
                # Check if there's a balance of this crypto
                if crypto in balance and 'free' in balance[crypto]:
                    crypto_balance = float(balance[crypto]['free'])
                    if crypto_balance > 0:
                        # Try to convert it
                        if convert_crypto_to_quote(exchange_name, crypto, quote_currency):
                            # Calculate value converted
                            try:
                                ticker = exchange_instance.fetch_ticker(f"{crypto}/{quote_currency}")
                                value = crypto_balance * float(ticker['last'])
                                total_converted_value += value
                                converted_count += 1
                            except:
                                pass
                        
        except Exception as e:
            printerror(m=f"Error checking leftover crypto on {exchange_name}: {e}")
    
    if converted_count > 0:
        printandtelegram(f"{get_time()}Converted {converted_count} leftover cryptocurrency(s) worth ~${total_converted_value:.2f} to {quote_currency}.")
    else:
        printandtelegram(f"{get_time()}No leftover cryptocurrency found from previous sessions.")

def rebalance_to_quote_currency(pair: str, exchange_list: List[str], average_entry_price: float = 0.0, force: bool = False) -> bool:
    """
    Rebalance by selling all cryptocurrency back to quote currency across all exchanges.
    
    Args:
        pair: Trading pair (e.g., 'BTC/USDT')
        exchange_list: List of exchange names
        average_entry_price: Average entry price to check against current price (0.0 = skip price check)
        force: If True, rebalance even if it would result in a loss
    
    Returns:
        True if rebalancing was performed, False if skipped due to loss threshold
    """
    base_currency = pair.split('/')[0]
    quote_currency = pair.split('/')[1]
    
    # Get current market price to compare with entry price
    current_price = 0.0
    if average_entry_price > 0 and not force:
        try:
            # Get average price across exchanges
            prices = []
            for exchange_name in exchange_list:
                try:
                    exchange_instance = get_exchange_instance(exchange_name)
                    ticker = exchange_instance.fetch_ticker(pair)
                    prices.append(float(ticker['last']))
                except:
                    pass
            if prices:
                current_price = sum(prices) / len(prices)
                
                # Calculate loss percentage
                if current_price < average_entry_price:
                    loss_pct = (current_price - average_entry_price) / average_entry_price
                    
                    if loss_pct < REBALANCE_LOSS_THRESHOLD:
                        printandtelegram(f"{get_time()}⚠️  WARNING: Rebalancing would result in {loss_pct*100:.2f}% loss "
                                       f"(entry: ${average_entry_price:.2f}, current: ${current_price:.2f})")
                        printandtelegram(f"{get_time()}Skipping rebalancing to avoid locking in losses. "
                                       f"Threshold: {REBALANCE_LOSS_THRESHOLD*100:.2f}%")
                        printandtelegram(f"{get_time()}You can manually rebalance later when price recovers, "
                                       f"or set AUTO_REBALANCE_ON_EXIT=False to disable automatic rebalancing.")
                        append_new_line(LOGS_FILE, 
                            f"{get_time_blank()} WARNING: Rebalancing skipped due to loss threshold. "
                            f"Entry: ${average_entry_price:.2f}, Current: ${current_price:.2f}, "
                            f"Loss: {loss_pct*100:.2f}%")
                        return False
                    else:
                        printandtelegram(f"{get_time()}Current price is {loss_pct*100:.2f}% below entry, "
                                       f"but within acceptable threshold. Proceeding with rebalancing.")
        except Exception as e:
            printerror(m=f"Error checking price for rebalancing: {e}")
            # Continue with rebalancing if price check fails
    
    printandtelegram(f"{get_time()}Starting rebalancing: converting all {base_currency} to {quote_currency}...")
    
    exchanges_converted = 0
    total_amount_converted = 0.0
    
    for exchange_name in exchange_list:
        try:
            # Get balance before conversion
            balance_before = get_balance(exchange_name, pair)
            
            if balance_before > 0:
                if convert_crypto_to_quote(exchange_name, base_currency, quote_currency):
                    exchanges_converted += 1
                    total_amount_converted += balance_before
        except Exception as e:
            printerror(m=f"Error rebalancing on {exchange_name}: {e}")
    
    if exchanges_converted > 0:
        printandtelegram(f"{get_time()}Rebalancing complete: converted {total_amount_converted:.6f} {base_currency} to {quote_currency} on {exchanges_converted} exchange(s).")
        return True
    else:
        printandtelegram(f"{get_time()}Rebalancing complete: no {base_currency} found to convert.")
        return True

def check_balance_distribution(exchange_list: List[str], pair: str, target_investment_usd: float) -> Dict[str, Dict[str, float]]:
    """Check balance distribution across exchanges and return balance info."""
    base_currency = pair.split('/')[0]
    quote_currency = pair.split('/')[1]
    
    balances = {}
    total_quote = 0.0
    total_crypto = 0.0
    
    for exchange_name in exchange_list:
        try:
            exchange_instance = get_exchange_instance(exchange_name)
            balance = exchange_instance.fetch_balance()
            quote_bal = float(balance[quote_currency]['free'])
            crypto_bal = float(balance[base_currency]['free'])
            
            balances[exchange_name] = {
                'quote': quote_bal,
                'crypto': crypto_bal
            }
            
            total_quote += quote_bal
            total_crypto += crypto_bal
        except Exception as e:
            printerror(m=f"Error fetching balance from {exchange_name}: {e}")
            balances[exchange_name] = {'quote': 0.0, 'crypto': 0.0}
    
    return {
        'per_exchange': balances,
        'total_quote': total_quote,
        'total_crypto': total_crypto
    }

def should_rebalance_balances(balance_info: Dict, exchange_list: List[str], target_investment_usd: float, average_price: float) -> bool:
    """Determine if balances need rebalancing based on distribution."""
    if not balance_info or 'per_exchange' not in balance_info:
        return True
    
    per_exchange = balance_info['per_exchange']
    num_exchanges = len(exchange_list)
    
    if num_exchanges == 0:
        return False
    
    # Calculate target per exchange (in quote currency)
    target_per_exchange = target_investment_usd / num_exchanges
    
    # Check if distribution is too uneven
    # Consider rebalancing if any exchange has less than 20% or more than 200% of target
    rebalance_threshold_low = target_per_exchange * 0.2
    rebalance_threshold_high = target_per_exchange * 2.0
    
    needs_rebalance = False
    for exchange_name in exchange_list:
        if exchange_name not in per_exchange:
            needs_rebalance = True
            break
        
        exchange_total = per_exchange[exchange_name]['quote'] + (per_exchange[exchange_name]['crypto'] * average_price)
        
        if exchange_total < rebalance_threshold_low or exchange_total > rebalance_threshold_high:
            needs_rebalance = True
            break
    
    return needs_rebalance

def calculate_fees(exchange_names: List[str], pair: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate trading fees for each exchange using the actual trading pair.
    
    Args:
        exchange_names: List of exchange names to check
        pair: Trading pair (e.g., 'BTC/USDT')
    
    Returns:
        Dictionary mapping exchange names to fee structures {'base': float, 'quote': float}
    """
    fees = {}
    
    for exchange_name in exchange_names:
        try:
            try:
                exchange_instance = get_exchange_instance(exchange_name)
            except:
                fees[exchange_name] = {'base': 0, 'quote': 0.001}  # Default 0.1%
                continue
            
            markets = exchange_instance.load_markets()
            pair_info = markets.get(pair, {})
            
            # If the pair doesn't exist, fall back to BTC/USDT as reference
            if not pair_info:
                pair_info = markets.get('BTC/USDT', {})
            
            # Extract fee information
            fee_side = pair_info.get('feeSide')
            taker_fee = pair_info.get('taker', 0)
            
            # If taker fee is not available, try to get it from exchange defaults
            if taker_fee == 0:
                try:
                    if hasattr(exchange_instance, 'fees') and 'trading' in exchange_instance.fees:
                        taker_fee = exchange_instance.fees['trading'].get('taker', 0.001)
                    else:
                        taker_fee = 0.001  # Default 0.1%
                except:
                    taker_fee = 0.001  # Default 0.1%
            
            if fee_side:
                fees[exchange_name] = {
                    'base': taker_fee if fee_side == 'base' else 0,
                    'quote': 0 if fee_side == 'base' else taker_fee
                }
            else:
                fees[exchange_name] = {'base': 0, 'quote': taker_fee}
                
        except Exception as e:
            # Default to 0.1% fee if we can't determine
            fees[exchange_name] = {'base': 0, 'quote': 0.001}
    
    return fees

def printandtelegram(message: str) -> None:
    """Print message to console and send to Telegram."""
    print(message)
    send_to_telegram(message)

def check_all_exchange_fees(pair: str = 'BTC/USDT', favorable_threshold: float = 0.0015) -> Dict[str, Dict[str, Any]]:
    """
    Check fees for all configured exchanges and identify favorable ones for arbitrage.
    
    Args:
        pair: Trading pair to check fees for (default: BTC/USDT)
        favorable_threshold: Maximum total fee rate to be considered favorable (default: 0.15%)
    
    Returns:
        Dictionary with exchange fee information and favorable exchanges list
    """
    # Use the global ex dictionary that's already initialized
    
    exchange_fees = {}
    favorable_exchanges = []
    
    # Only check exchanges that are actually configured in the exchanges dict
    for exchange_name in sorted(exchanges.keys()):
        try:
            # Try to get exchange instance (lazy initialization)
            try:
                exchange_instance = get_exchange_instance(exchange_name)
            except (AttributeError, Exception) as e:
                # Exchange not supported or initialization failed
                exchange_fees[exchange_name] = {
                    'total_fee': None,
                    'status': 'not_supported',
                    'error': str(e)
                }
                continue
            
            # Try to get fees
            try:
                markets = exchange_instance.load_markets()
                pair_info = markets.get(pair, {})
                
                # Fallback to BTC/USDT if pair not found
                if not pair_info:
                    pair_info = markets.get('BTC/USDT', {})
                
                # Extract fee information
                fee_side = pair_info.get('feeSide')
                taker_fee = pair_info.get('taker', 0)
                
                # If taker fee not available, try exchange defaults
                if taker_fee == 0:
                    try:
                        if hasattr(exchange_instance, 'fees') and 'trading' in exchange_instance.fees:
                            taker_fee = exchange_instance.fees['trading'].get('taker', 0.001)
                        else:
                            taker_fee = 0.001  # Default 0.1%
                    except:
                        taker_fee = 0.001  # Default 0.1%
                
                total_fee_rate = taker_fee
                
                exchange_fees[exchange_name] = {
                    'total_fee': total_fee_rate,
                    'fee_side': fee_side or 'quote',
                    'status': 'success'
                }
                
                # Check if favorable (low fees are better for arbitrage)
                if total_fee_rate <= favorable_threshold:
                    favorable_exchanges.append({
                        'name': exchange_name,
                        'total_fee': total_fee_rate,
                        'fee_pct': total_fee_rate * 100
                    })
                    
            except Exception as e:
                exchange_fees[exchange_name] = {
                    'total_fee': None,
                    'status': 'error',
                    'error': str(e)[:100]  # Truncate long error messages
                }
                
        except Exception as e:
            exchange_fees[exchange_name] = {
                'total_fee': None,
                'status': 'error',
                'error': str(e)[:100]
            }
    
    # Sort favorable exchanges by fee (lowest first)
    favorable_exchanges.sort(key=lambda x: x['total_fee'])
    
    # Function returns data without printing (caller handles display/logging)
    
    return {
        'all_fees': exchange_fees,
        'favorable': favorable_exchanges,
        'favorable_threshold': favorable_threshold
    }

# Initialize exchange instances LAZILY (only when needed)
# This prevents initializing all 59 exchanges at startup, which can cause stalling
DISABLE_SSL_VERIFY = False  # Set to True only for testing if SSL issues occur

default_config = {
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',  # Use spot trading by default
    }
}

# Add SSL verification setting if disabled
if DISABLE_SSL_VERIFY:
    default_config['verify'] = False
    # Also configure requests to not verify SSL
    import ssl
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Exchange cache - only initialized exchanges are stored here
# We use a custom dict-like class to support lazy initialization
# This prevents initializing all 59 exchanges at startup, which was causing stalling
class LazyExchangeDict(dict):
    """
    Dictionary that initializes exchanges lazily on access.
    Only initializes exchanges when they're actually used, not all 59 at startup.
    This fixes the stalling issue where the bot tried to initialize all exchanges.
    """
    def __getitem__(self, exchange_name: str):
        if exchange_name not in self:
            # Only initialize if the exchange is in our configured list
            if exchange_name not in exchanges:
                raise KeyError(f"Exchange {exchange_name} is not in the configured exchanges list")
            
            merged_config = {**default_config, **exchanges[exchange_name]}
            try:
                # Initialize the exchange instance
                self[exchange_name] = getattr(ccxt, exchange_name)(merged_config)
            except Exception as e:
                error_msg = f"Error initializing exchange {exchange_name}: {type(e).__name__}: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()
                raise
        
        return super().__getitem__(exchange_name)
    
    def get(self, exchange_name: str, default=None):
        """Get exchange instance, returning default if not found."""
        try:
            return self[exchange_name]
        except (KeyError, ValueError):
            return default
    
    def keys(self):
        """Return keys of initialized exchanges only."""
        return super().keys()
    
    def __contains__(self, exchange_name: str):
        """Check if exchange is initialized (not if it's configured)."""
        return super().__contains__(exchange_name)

# Global exchange dictionary - now uses lazy initialization
# Only exchanges that are actually used will be initialized
ex = LazyExchangeDict()

def get_exchange_instance(exchange_name: str):
    """
    Get or create an exchange instance lazily.
    Only initializes exchanges when they're actually needed.
    This prevents initializing all 59 exchanges at startup.
    """
    return ex[exchange_name]

# Initialize exchanges that are explicitly requested at startup (if any)
# This allows pre-initialization for commonly used exchanges if needed
# For now, we'll keep it empty and initialize on-demand

# Legacy compatibility (for existing code that uses old function name)
moy = calculate_average
append_list_file = append_list_to_file
