#!/usr/bin/env python3
"""
Barbotine Arbitrage Bot - Real Money Mode
Executes real arbitrage trades with actual funds on cryptocurrency exchanges.
⚠️  WARNING: This bot trades with real money. Use at your own risk.
"""

import asyncio
import time
import sys
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass

import ccxt.pro
import ccxt
from colorama import Fore, Back, Style, init

# Initialize colorama
init()

# Import configuration
from exchange_config import (
    ex, python_command, first_orders_fill_timeout, criteria_pct, criteria_usd, 
    printerror, get_time, get_time_blank, append_new_line, append_list_to_file,
    printandtelegram, calculate_average, send_to_telegram, get_balance,
    emergency_convert_list, get_balance_usdt, cancel_all_orders, MIN_ORDER_VALUE_USD,
    detect_and_convert_leftover_crypto, rebalance_to_quote_currency,
    check_balance_distribution, should_rebalance_balances, AUTO_REBALANCE_ON_EXIT,
    ORDERBOOK_FETCH_DELAY, BALANCE_CACHE_TTL, MIN_ITERATION_DELAY,
    RATE_LIMIT_BACKOFF_BASE, MAX_RATE_LIMIT_BACKOFF, DEFAULT_TARGET_INVESTMENT_USD,
    OPPORTUNITY_LOG_THROTTLE, OPPORTUNITY_LOG_DEDUPE, calculate_optimal_order_size,
    DYNAMIC_ORDER_SIZING
)

# Constants
REQUIRED_ARGS = 6
DEFAULT_TIMEOUT_SECONDS = 3600
CONSOLE_CLEAR_LINES = 2

@dataclass
class TradingState:
    """Holds the current trading state."""
    bid_prices: Dict[str, float]
    ask_prices: Dict[str, float]
    total_profit_usd: float
    previous_ask_price: float
    previous_bid_price: float
    opportunity_count: int
    iteration_count: int
    stop_requested: bool
    average_entry_price: float  # Track average entry price for rebalancing decisions
    total_crypto_bought: float  # Track total crypto bought to calculate average
    total_cost_basis: float  # Track total cost basis (USDT spent)
    
    def __init__(self):
        self.bid_prices = {}
        self.ask_prices = {}
        self.total_profit_usd = 0.0
        self.previous_ask_price = 0.0
        self.previous_bid_price = 0.0
        self.opportunity_count = 0
        self.iteration_count = 0
        self.stop_requested = False
        self.average_entry_price = 0.0
        self.total_crypto_bought = 0.0
        self.total_cost_basis = 0.0

@dataclass
class TradingConfig:
    """Configuration for the trading session."""
    pair: str
    base_currency: str
    quote_currency: str
    total_investment_usd: float
    timeout_minutes: int
    session_title: str
    exchange_names: List[str]
    exchange_instances: List[object]
    
def validate_arguments() -> None:
    """Validate command line arguments."""
    if len(sys.argv) != REQUIRED_ARGS:
        print(f"\nIncorrect usage. Expected format:")
        print(f"{python_command} bot.py [pair] [total_usdt_investment] [timeout_minutes] [session_title] [exchange_list]")
        print(f"\nReceived arguments: {sys.argv}")
        sys.exit(1)

def setup_exchanges(exchange_list_str: str) -> List[str]:
    """Setup and validate exchange instances."""
    exchange_names = exchange_list_str.split(',')
    
    # Initialize missing exchanges with proper configuration
    default_config = {
        'enableRateLimit': True,
        'options': {
            'defaultType': 'spot',
        }
    }
    
    for exchange_name in exchange_names:
        if exchange_name not in ex:
            try:
                ex[exchange_name] = getattr(ccxt, exchange_name)(default_config)
            except AttributeError:
                printerror(m=f"Unsupported exchange: {exchange_name}")
                sys.exit(1)
            except Exception as e:
                error_type = type(e).__name__
                printerror(m=f"Error creating exchange {exchange_name}: {error_type}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)
    
    return exchange_names

def calculate_fees(exchange_names: List[str], pair: str) -> Dict[str, Dict[str, float]]:
    """Calculate trading fees for each exchange using the actual trading pair."""
    fees = {}
    
    for exchange_name in exchange_names:
        try:
            markets = ex[exchange_name].load_markets()
            # Use the actual trading pair instead of hardcoded BTC/USDT
            pair_info = markets.get(pair, {})
            
            # If the pair doesn't exist, fall back to BTC/USDT as reference
            if not pair_info:
                pair_info = markets.get('BTC/USDT', {})
                if pair_info:
                    printerror(m=f"Pair {pair} not found on {exchange_name}, using BTC/USDT fee structure")
            
            # Extract fee information
            fee_side = pair_info.get('feeSide')
            taker_fee = pair_info.get('taker', 0)
            
            # If taker fee is not available, try to get it from exchange defaults
            if taker_fee == 0:
                # Try to get default taker fee from exchange
                try:
                    exchange_instance = ex[exchange_name]
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
            printerror(m=f"Error calculating fees for {exchange_name}: {e}")
            # Default to quote-side fee (most common) - 0.1% taker fee
            fees[exchange_name] = {'base': 0, 'quote': 0.001}
    
    return fees

def listen_for_manual_exit(state: TradingState) -> None:
    """Listen for user input to request manual exit."""
    try:
        input("")  # Wait for any input
        state.stop_requested = True
    except:
        pass

def is_rate_limit_error(error: Exception) -> bool:
    """Check if an error is a rate limit error."""
    error_str = str(error).lower()
    rate_limit_indicators = [
        'rate limit', 'ratelimit', 'too many requests', '429',
        'rate_limit_exceeded', 'request limit', 'throttle',
        'nonce is behind', 'rate limit exceeded'
    ]
    return any(indicator in error_str for indicator in rate_limit_indicators)

async def fetch_orderbook_safe(exchange_instance, pair: str, backoff_delay: float = 0.0) -> Optional[dict]:
    """Safely fetch orderbook with error handling, retry logic, and rate limit detection."""
    # Apply backoff delay if provided
    if backoff_delay > 0:
        await asyncio.sleep(backoff_delay)
    
    try:
        # Add minimum delay to prevent overwhelming APIs
        if ORDERBOOK_FETCH_DELAY > 0:
            await asyncio.sleep(ORDERBOOK_FETCH_DELAY)
        
        orderbook = await exchange_instance.watch_order_book(pair)
        return orderbook, 0.0  # Return orderbook and reset backoff
    except Exception as e:
        error_msg = str(e)
        is_rate_limited = is_rate_limit_error(e)
        
        if is_rate_limited:
            printerror(m=f"Rate limit detected on {exchange_instance.id}: {error_msg}")
            append_new_line('logs/logs.txt', 
                f"{get_time_blank()} WARNING: Rate limit on {exchange_instance.id}, applying backoff")
        
        # Only clear console for rate limit errors (to avoid disrupting normal display)
        if is_rate_limited:
            for _ in range(CONSOLE_CLEAR_LINES):
                sys.stdout.write("\033[F\033[K")
        
        # Calculate backoff delay for rate limit errors
        if is_rate_limited:
            # Increase backoff exponentially on rate limit
            new_backoff = backoff_delay * RATE_LIMIT_BACKOFF_BASE if backoff_delay > 0 else RATE_LIMIT_BACKOFF_BASE
            new_backoff = min(new_backoff, MAX_RATE_LIMIT_BACKOFF)
        else:
            # Gradually reduce backoff when no rate limit (halve it each successful iteration)
            new_backoff = max(0.0, backoff_delay / 2.0)
        
        # Try to recreate the exchange instance
        try:
            exchange_id = exchange_instance.id
            await exchange_instance.close()
            new_instance = getattr(ccxt.pro, exchange_id)({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
            # Wait before retry if rate limited
            if is_rate_limited and new_backoff > 0:
                await asyncio.sleep(new_backoff)
            
            orderbook = await new_instance.watch_order_book(pair)
            return orderbook, 0.0  # Reset backoff on success
        except Exception as retry_error:
            if is_rate_limit_error(retry_error):
                printerror(m=f"Rate limit persists on {exchange_instance.id}, backing off for {new_backoff:.1f}s")
                return None, new_backoff
            else:
                printerror(m=f"Failed to retry orderbook fetch: {retry_error}")
                # Return with small backoff for non-rate-limit errors to prevent tight loop
                return None, min(2.0, backoff_delay + 0.5)

def clear_console_lines(count: int) -> None:
    """Clear specified number of lines from console."""
    for _ in range(count):
        sys.stdout.write("\033[F\033[K")

def calculate_theoretical_balances(
    usd_balances: Dict[str, float],
    crypto_balances: Dict[str, float],
    min_ask_exchange: str,
    max_bid_exchange: str,
    min_ask_price: float,
    max_bid_price: float,
    crypto_per_transaction: float,
    fees: Dict[str, Dict[str, float]]
) -> tuple:
    """Calculate theoretical balances after a trade."""
    
    # Get total taker fee for each exchange (base + quote fees)
    buy_fee_rate = fees[min_ask_exchange].get('base', 0) + fees[min_ask_exchange].get('quote', 0)
    sell_fee_rate = fees[max_bid_exchange].get('base', 0) + fees[max_bid_exchange].get('quote', 0)
    
    # Calculate cost of buying on min_ask exchange (price + fee)
    buy_cost = crypto_per_transaction * min_ask_price * (1 + buy_fee_rate)
    
    # Calculate proceeds from selling on max_bid exchange (price - fee)
    sell_proceeds = crypto_per_transaction * max_bid_price * (1 - sell_fee_rate)
    
    theoretical_min_ask_usd = usd_balances[min_ask_exchange] - buy_cost
    theoretical_max_bid_usd = usd_balances[max_bid_exchange] + sell_proceeds
    
    return theoretical_min_ask_usd, theoretical_max_bid_usd

def rank_exchanges_by_fees(fees: Dict[str, Dict[str, float]], exchange_names: List[str]) -> List[tuple]:
    """
    Rank exchanges by total fee rate (lower is better for arbitrage).
    Returns list of (exchange_name, total_fee_rate) tuples sorted by fee rate (ascending).
    """
    exchange_fees = []
    for exchange_name in exchange_names:
        fee_info = fees.get(exchange_name, {})
        total_fee_rate = fee_info.get('base', 0) + fee_info.get('quote', 0)
        exchange_fees.append((exchange_name, total_fee_rate))
    
    # Sort by total fee rate (ascending - lower fees first)
    exchange_fees.sort(key=lambda x: x[1])
    return exchange_fees

def format_exchange_balances(
    crypto_balances: Dict[str, float], 
    usd_balances: Dict[str, float],
    base_currency: str,
    quote_currency: str
) -> str:
    """Format exchange balances for display."""
    balance_lines = []
    for exchange_name in crypto_balances:
        crypto_amount = round(crypto_balances[exchange_name], 6)
        usd_amount = round(usd_balances[exchange_name], 2)
        balance_lines.append(f"➝ {exchange_name}: {crypto_amount} {base_currency} / {usd_amount} {quote_currency}")
    
    return "\n".join(balance_lines)

def execute_real_trades(
    min_ask_exchange: str,
    max_bid_exchange: str, 
    min_ask_price: float,
    max_bid_price: float,
    crypto_per_transaction: float,
    profit_usd: float,
    pair: str,
    base_currency: str,
    fees: Dict[str, Dict[str, float]],
    state: TradingState = None
) -> bool:
    """Execute real arbitrage trades on exchanges."""
    
    try:
        # Get market info for order size validation
        min_ask_markets = ex[min_ask_exchange].load_markets()
        max_bid_markets = ex[max_bid_exchange].load_markets()
        
        min_ask_market = min_ask_markets.get(pair, {})
        max_bid_market = max_bid_markets.get(pair, {})
        
        # Get trading limits
        min_ask_limits = min_ask_market.get('limits', {})
        max_bid_limits = max_bid_market.get('limits', {})
        
        min_amount_ask = min_ask_limits.get('amount', {}).get('min', 0)
        min_amount_bid = max_bid_limits.get('amount', {}).get('min', 0)
        min_cost_ask = min_ask_limits.get('cost', {}).get('min', 0)
        min_cost_bid = max_bid_limits.get('cost', {}).get('min', 0)
        
        # Validate order sizes
        buy_cost = crypto_per_transaction * min_ask_price
        sell_cost = crypto_per_transaction * max_bid_price
        
        if (crypto_per_transaction < max(min_amount_ask, min_amount_bid) or
            buy_cost < max(min_cost_ask, MIN_ORDER_VALUE_USD) or
            sell_cost < max(min_cost_bid, MIN_ORDER_VALUE_USD)):
            printerror(m=f"Order size too small for {pair} on exchanges")
            return False
        
        # Place sell order first (on max_bid exchange) - we need to have crypto to sell
        printandtelegram(f"{get_time()}Placing SELL market order on {max_bid_exchange}: "
                        f"{crypto_per_transaction:.6f} {base_currency} at ~{max_bid_price}")
        
        # Use createOrder with market type for better compatibility
        sell_order = ex[max_bid_exchange].create_order(
            pair,
            'market',
            'sell',
            crypto_per_transaction
        )
        
        if not sell_order:
            printerror(m=f"Failed to place sell order on {max_bid_exchange}")
            return False
        
        sell_order_id = sell_order.get('id') or sell_order.get('orderId')
        printandtelegram(f"{get_time()}Sell order placed on {max_bid_exchange}, order ID: {sell_order_id}")
        
        # Wait for sell order to fill
        sell_filled = False
        sell_fill_time = time.time()
        while time.time() - sell_fill_time < 30:  # 30 second timeout
            try:
                order_status = ex[max_bid_exchange].fetch_order(sell_order_id, pair)
                if order_status['status'] == 'closed' or order_status['filled'] > 0:
                    sell_filled = True
                    actual_sell_amount = order_status.get('filled', crypto_per_transaction)
                    actual_sell_price = order_status.get('average', max_bid_price)
                    printandtelegram(f"{get_time()}Sell order filled on {max_bid_exchange}: "
                                    f"{actual_sell_amount:.6f} {base_currency} at {actual_sell_price}")
                    break
                time.sleep(0.5)
            except Exception as e:
                printerror(m=f"Error checking sell order status: {e}")
                time.sleep(1)
        
        if not sell_filled:
            printerror(m=f"Sell order on {max_bid_exchange} did not fill in time")
            # Try to cancel the order
            try:
                ex[max_bid_exchange].cancel_order(sell_order_id, pair)
            except:
                pass
            return False
        
        # Get the actual amount received from the sell
        try:
            sell_order_final = ex[max_bid_exchange].fetch_order(sell_order_id, pair)
            quote_received = sell_order_final.get('cost', crypto_per_transaction * max_bid_price)
        except:
            quote_received = crypto_per_transaction * max_bid_price
        
        # Calculate how much crypto we can buy with the proceeds
        buy_amount = quote_received / min_ask_price
        
        # Place buy order (on min_ask exchange)
        printandtelegram(f"{get_time()}Placing BUY market order on {min_ask_exchange}: "
                        f"{buy_amount:.6f} {base_currency} at ~{min_ask_price}")
        
        # Use createOrder with market type for better compatibility
        buy_order = ex[min_ask_exchange].create_order(
            pair,
            'market',
            'buy',
            buy_amount
        )
        
        if not buy_order:
            printerror(m=f"Failed to place buy order on {min_ask_exchange}")
            return False
        
        buy_order_id = buy_order.get('id') or buy_order.get('orderId')
        printandtelegram(f"{get_time()}Buy order placed on {min_ask_exchange}, order ID: {buy_order_id}")
        
        # Wait for buy order to fill
        buy_filled = False
        buy_fill_time = time.time()
        actual_buy_price = min_ask_price  # Default to expected price
        actual_buy_amount = buy_amount  # Default to expected amount
        
        while time.time() - buy_fill_time < 30:  # 30 second timeout
            try:
                order_status = ex[min_ask_exchange].fetch_order(buy_order_id, pair)
                if order_status['status'] == 'closed' or order_status['filled'] > 0:
                    buy_filled = True
                    actual_buy_amount = order_status.get('filled', buy_amount)
                    actual_buy_price = order_status.get('average', min_ask_price)
                    printandtelegram(f"{get_time()}Buy order filled on {min_ask_exchange}: "
                                    f"{actual_buy_amount:.6f} {base_currency} at {actual_buy_price}")
                    break
                time.sleep(0.5)
            except Exception as e:
                printerror(m=f"Error checking buy order status: {e}")
                time.sleep(1)
        
        if not buy_filled:
            printerror(m=f"Buy order on {min_ask_exchange} did not fill in time")
            # Try to cancel the order
            try:
                ex[min_ask_exchange].cancel_order(buy_order_id, pair)
            except:
                pass
            return False
        
        # Update entry price tracking (weighted average)
        # This tracks the average cost basis for rebalancing decisions
        if state is not None:
            cost_of_this_buy = actual_buy_amount * actual_buy_price
            state.total_cost_basis += cost_of_this_buy
            state.total_crypto_bought += actual_buy_amount
            if state.total_crypto_bought > 0:
                state.average_entry_price = state.total_cost_basis / state.total_crypto_bought
        
        # Record the profit
        append_list_to_file('all_opportunities_profits.txt', profit_usd)
        
        return True
        
    except Exception as e:
        printerror(m=f"Error executing real trades: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

async def monitor_arbitrage_opportunities(
    exchange_instance,
    config: TradingConfig,
    state: TradingState,
    fees: Dict[str, Dict[str, float]],
    crypto_per_transaction: float,
    session_start_time: float,
    timeout_timestamp: float
) -> None:
    """Monitor for arbitrage opportunities on a single exchange."""
    
    # Balance cache to reduce API calls
    balance_cache = {}
    balance_cache_timestamp = {}
    backoff_delay = 0.0  # Track backoff delay for this exchange
    
    # Only the first exchange (alphabetically) should log opportunities to prevent duplicates
    # This is a simple way to ensure only one task logs without needing locks
    should_log = exchange_instance.id == min(config.exchange_names)
    
    # Track last logged opportunity for deduplication
    last_logged_opportunity = None
    last_log_time = 0.0
    
    while time.time() <= timeout_timestamp:
        if state.stop_requested:
            clear_console_lines(CONSOLE_CLEAR_LINES)
            print(f"{get_time()}Manual rebalance requested. Breaking.")
            append_new_line('logs/logs.txt', f"{get_time_blank()} INFO: Manual rebalance requested.")
            await exchange_instance.close()
            break
        
        # Fetch orderbook with backoff delay
        result = await fetch_orderbook_safe(exchange_instance, config.pair, backoff_delay)
        if result is None:
            # If fetch failed, wait a bit before retrying to avoid tight loop
            await asyncio.sleep(1.0)
            continue
        
        orderbook, new_backoff = result if isinstance(result, tuple) else (result, 0.0)
        backoff_delay = new_backoff  # Update backoff delay
        
        if not orderbook:
            # If no orderbook, wait before retrying
            await asyncio.sleep(0.5)
            continue
        
        # Validate orderbook has bids and asks before accessing
        if (not orderbook.get("bids") or len(orderbook["bids"]) == 0 or
            not orderbook.get("asks") or len(orderbook["asks"]) == 0):
            # Empty orderbook, wait before retrying
            await asyncio.sleep(0.5)
            continue
        
        # Update price data
        state.bid_prices[exchange_instance.id] = orderbook["bids"][0][0]
        state.ask_prices[exchange_instance.id] = orderbook["asks"][0][0]
        
        # Find best arbitrage opportunity by evaluating ALL exchange pairs
        # and selecting the one with highest profit after fees
        # Get all exchanges with valid prices
        valid_ask_exchanges = {ex: price for ex, price in state.ask_prices.items() if ex in config.exchange_names}
        valid_bid_exchanges = {ex: price for ex, price in state.bid_prices.items() if ex in config.exchange_names}
        
        if not valid_ask_exchanges or not valid_bid_exchanges:
            continue
        
        # Get real balances for validation (with caching to reduce API calls)
        exchange_balances = {}
        current_time = time.time()
        
        try:
            for exchange_name in config.exchange_names:
                # Check cache first
                cache_key = exchange_name
                if (cache_key in balance_cache and 
                    cache_key in balance_cache_timestamp and
                    current_time - balance_cache_timestamp[cache_key] < BALANCE_CACHE_TTL):
                    # Use cached balance
                    exchange_balances[exchange_name] = balance_cache[cache_key]
                else:
                    # Fetch fresh balance
                    try:
                        balance = ex[exchange_name].fetch_balance()
                        exchange_balances[exchange_name] = {
                            'crypto': float(balance[config.base_currency]['free']),
                            'quote': float(balance[config.quote_currency]['free'])
                        }
                        # Update cache
                        balance_cache[cache_key] = exchange_balances[exchange_name]
                        balance_cache_timestamp[cache_key] = current_time
                    except Exception as balance_error:
                        if is_rate_limit_error(balance_error):
                            # If rate limited, use cached balance if available
                            if cache_key in balance_cache:
                                exchange_balances[exchange_name] = balance_cache[cache_key]
                                printerror(m=f"Rate limited fetching balance from {exchange_name}, using cached value")
                            else:
                                printerror(m=f"Rate limited fetching balance from {exchange_name}, no cache available")
                                continue
                        else:
                            printerror(m=f"Error fetching balance from {exchange_name}: {balance_error}")
                            continue
        except Exception as e:
            printerror(m=f"Error processing balances: {e}")
            continue
        
        # Evaluate ALL possible exchange pairs to find the one with highest profit after fees
        best_profit = float('-inf')
        best_ask_exchange = None
        best_bid_exchange = None
        best_ask_price = 0.0
        best_bid_price = 0.0
        
        for ask_exchange, ask_price in valid_ask_exchanges.items():
            for bid_exchange, bid_price in valid_bid_exchanges.items():
                # Skip if same exchange
                if ask_exchange == bid_exchange:
                    continue
                
                # Check balance availability
                if exchange_balances.get(bid_exchange, {}).get('crypto', 0) < crypto_per_transaction:
                    continue
                buy_cost = crypto_per_transaction * ask_price
                if exchange_balances.get(ask_exchange, {}).get('quote', 0) < buy_cost:
                    continue
                
                # Calculate profit after fees for this pair
                buy_fee_rate = fees.get(ask_exchange, {}).get('base', 0) + fees.get(ask_exchange, {}).get('quote', 0)
                sell_fee_rate = fees.get(bid_exchange, {}).get('base', 0) + fees.get(bid_exchange, {}).get('quote', 0)
                
                # Cost to buy (including fees)
                buy_cost_with_fees = crypto_per_transaction * ask_price * (1 + buy_fee_rate)
                # Proceeds from selling (after fees)
                sell_proceeds_after_fees = crypto_per_transaction * bid_price * (1 - sell_fee_rate)
                
                # Profit = sell proceeds - buy cost
                profit = sell_proceeds_after_fees - buy_cost_with_fees
                
                # Track the best opportunity
                if profit > best_profit:
                    best_profit = profit
                    best_ask_exchange = ask_exchange
                    best_bid_exchange = bid_exchange
                    best_ask_price = ask_price
                    best_bid_price = bid_price
        
        # If no profitable opportunity found, continue
        if best_ask_exchange is None or best_bid_exchange is None:
            continue
        
        min_ask_exchange = best_ask_exchange
        max_bid_exchange = best_bid_exchange
        min_ask_price = best_ask_price
        max_bid_price = best_bid_price
        
        # Get fee rates
        buy_fee_rate = fees.get(min_ask_exchange, {}).get('base', 0) + fees.get(min_ask_exchange, {}).get('quote', 0)
        sell_fee_rate = fees.get(max_bid_exchange, {}).get('base', 0) + fees.get(max_bid_exchange, {}).get('quote', 0)
        
        # Check spread percentage first (doesn't depend on order size)
        price_diff_pct = abs(min_ask_price - max_bid_price) / ((max_bid_price + min_ask_price) / 2) * 100
        
        # Initialize trade_order_size
        trade_order_size = None
        profit_usd = 0.0
        
        # Basic checks that don't depend on order size
        if (max_bid_exchange != min_ask_exchange and
            price_diff_pct >= criteria_pct and
            state.previous_ask_price != min_ask_price and
            state.previous_bid_price != max_bid_price):
            
            # Calculate optimal order size FIRST (before checking profit threshold)
            # This allows us to check if a larger order would meet the profit criteria
            available_crypto = exchange_balances.get(max_bid_exchange, {}).get('crypto', 0)
            available_quote = exchange_balances.get(min_ask_exchange, {}).get('quote', 0)
            
            # Calculate optimal order size for this specific opportunity
            optimal_order_size = calculate_optimal_order_size(
                min_ask_price=min_ask_price,
                max_bid_price=max_bid_price,
                available_crypto=available_crypto,
                available_quote=available_quote,
                buy_fee_rate=buy_fee_rate,
                sell_fee_rate=sell_fee_rate,
                min_order_value=MIN_ORDER_VALUE_USD,
                base_order_size=crypto_per_transaction
            )
            
            # Calculate profit with optimal order size
            buy_cost_with_fees = optimal_order_size * min_ask_price * (1 + buy_fee_rate)
            sell_proceeds_after_fees = optimal_order_size * max_bid_price * (1 - sell_fee_rate)
            profit_usd = sell_proceeds_after_fees - buy_cost_with_fees
            
            # Now check if profit with optimal size meets criteria
            if profit_usd > float(criteria_usd):
                # Profitable with optimal size - proceed with trade
                state.opportunity_count += 1
                trade_order_size = optimal_order_size
        
        # Only execute if we have a valid trade size
        if trade_order_size is not None and trade_order_size > 0:
            
            # Total fees in USD (using actual trade order size)
            buy_fee_usd = trade_order_size * min_ask_price * buy_fee_rate
            sell_fee_usd = trade_order_size * max_bid_price * sell_fee_rate
            fees_usd = buy_fee_usd + sell_fee_usd
            
            # Fees in crypto (approximate)
            fees_crypto = trade_order_size * (buy_fee_rate + sell_fee_rate)
            
            # Clear console and display opportunity
            clear_console_lines(1)
            print("-----------------------------------------------------")
            
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - session_start_time))
            
            print(f"{Style.RESET_ALL}Opportunity #{state.opportunity_count} detected! "
                  f"({min_ask_exchange} {min_ask_price} -> {max_bid_price} {max_bid_exchange})\n")
            print(f"Expected profit: {Fore.GREEN}+{round(profit_usd, 4)} {config.quote_currency}{Style.RESET_ALL}")
            print(f"Session total profit: {Fore.GREEN}+{round(state.total_profit_usd, 4)} {config.quote_currency}{Style.RESET_ALL}")
            print(f"Fees paid: {Fore.RED}-{round(fees_usd, 4)} {config.quote_currency} -{round(fees_crypto, 4)} {config.base_currency}")
            print(f"Time elapsed: {elapsed_time}")
            print("-----------------------------------------------------\n")
            
            # Send Telegram notification
            telegram_msg = (f"[{config.session_title} Trade #{state.opportunity_count}]\n\n"
                          f"Opportunity detected!\n"
                          f"Expected profit: {round(profit_usd, 4)} {config.quote_currency}\n"
                          f"{min_ask_exchange} {min_ask_price} -> {max_bid_price} {max_bid_exchange}\n"
                          f"Time elapsed: {elapsed_time}\n"
                          f"Session total profit: {round(state.total_profit_usd, 4)} {config.quote_currency}\n"
                          f"Fees: {round(fees_usd, 4)} {config.quote_currency} {round(fees_crypto, 4)} {config.base_currency}")
            send_to_telegram(telegram_msg)
            
            # Log if dynamic sizing was used
            if DYNAMIC_ORDER_SIZING and abs(trade_order_size - crypto_per_transaction) > crypto_per_transaction * 0.1:
                size_change_pct = ((trade_order_size - crypto_per_transaction) / crypto_per_transaction) * 100
                printandtelegram(f"{get_time()}Dynamic sizing: Order size adjusted by {size_change_pct:+.1f}% "
                               f"({crypto_per_transaction:.6f} -> {trade_order_size:.6f} {config.base_currency}) "
                               f"based on {price_diff_pct:.3f}% spread")
            
            # Execute real trades with optimal order size
            trade_success = execute_real_trades(
                min_ask_exchange, max_bid_exchange, min_ask_price, max_bid_price,
                trade_order_size, profit_usd, config.pair, config.base_currency, fees, state
            )
            
            if trade_success:
                state.total_profit_usd += profit_usd
                printandtelegram(f"{get_time()}Trade executed successfully! Profit: {round(profit_usd, 4)} {config.quote_currency}")
            else:
                printerror(m="Trade execution failed")
            
            state.previous_ask_price = min_ask_price
            state.previous_bid_price = max_bid_price
        
        else:
            # Display current best opportunity (only log from one exchange to prevent duplicates)
            if should_log:
                current_time = time.time()
                current_opportunity = (min_ask_exchange, max_bid_exchange, round(min_ask_price, 2), round(max_bid_price, 2))
                
                # Check if we should log this opportunity
                should_log_this = True
                if OPPORTUNITY_LOG_DEDUPE:
                    # Only log if opportunity changed
                    if current_opportunity == last_logged_opportunity:
                        should_log_this = False
                
                # Throttle logging frequency
                if should_log_this and (current_time - last_log_time) < OPPORTUNITY_LOG_THROTTLE:
                    should_log_this = False
                
                if should_log_this:
                    # Calculate profit with base order size for display (so users can see opportunity)
                    buy_fee_rate = fees.get(min_ask_exchange, {}).get('base', 0) + fees.get(min_ask_exchange, {}).get('quote', 0)
                    sell_fee_rate = fees.get(max_bid_exchange, {}).get('base', 0) + fees.get(max_bid_exchange, {}).get('quote', 0)
                    buy_cost_with_fees = crypto_per_transaction * min_ask_price * (1 + buy_fee_rate)
                    sell_proceeds_after_fees = crypto_per_transaction * max_bid_price * (1 - sell_fee_rate)
                    display_profit_usd = sell_proceeds_after_fees - buy_cost_with_fees
                    
                    # Don't clear console - let it scroll naturally to show history
                    color = Fore.GREEN if display_profit_usd > 0 else Fore.RED if display_profit_usd < 0 else Fore.WHITE
                    
                    # Format prices with appropriate precision (more decimals for low-priced tokens)
                    price_precision = 8 if min_ask_price < 1.0 else 6 if min_ask_price < 100.0 else 4
                    
                    # Calculate price spread for clarity
                    price_spread = max_bid_price - min_ask_price
                    spread_pct = (price_spread / min_ask_price) * 100 if min_ask_price > 0 else 0
                    
                    # Calculate total fee in USD for display
                    buy_fee_usd = crypto_per_transaction * min_ask_price * buy_fee_rate
                    sell_fee_usd = crypto_per_transaction * max_bid_price * sell_fee_rate
                    total_fee_usd = buy_fee_usd + sell_fee_usd
                    
                    print(f"{get_time()}Best: {color}{round(display_profit_usd, 4)} {config.quote_currency}{Style.RESET_ALL} "
                          f"(fees: {round(total_fee_usd, 4)}) buy: {min_ask_exchange} at {min_ask_price:.{price_precision}f} "
                          f"sell: {max_bid_exchange} at {max_bid_price:.{price_precision}f} "
                          f"(spread: {spread_pct:+.4f}%)")
                    
                    # Update tracking
                    last_logged_opportunity = current_opportunity
                    last_log_time = current_time
        
        # Add minimum delay between iterations to prevent overwhelming APIs
        if MIN_ITERATION_DELAY > 0:
            await asyncio.sleep(MIN_ITERATION_DELAY)

def place_initial_orders(config: TradingConfig, average_price: float, total_crypto: float, state: TradingState = None) -> bool:
    """Place initial buy orders to distribute crypto across exchanges."""
    printandtelegram(f"{get_time()}Fetching the global average price for {config.pair}...")
    printandtelegram(f"{get_time()}Average {config.pair} price in {config.quote_currency}: {average_price}")
    
    # Calculate crypto per exchange
    crypto_per_exchange = total_crypto / len(config.exchange_names)
    
    orders_placed = []
    
    # Place buy limit orders on each exchange
    for exchange_name in config.exchange_names:
        try:
            # Get market info
            markets = ex[exchange_name].load_markets()
            market = markets.get(config.pair, {})
            limits = market.get('limits', {})
            
            min_amount = limits.get('amount', {}).get('min', 0)
            min_cost = limits.get('cost', {}).get('min', 0)
            
            # Validate order size
            order_cost = crypto_per_exchange * average_price
            if crypto_per_exchange < min_amount or order_cost < max(min_cost, MIN_ORDER_VALUE_USD):
                printerror(m=f"Order size too small for {exchange_name}, skipping")
                continue
            
            # Place limit buy order
            printandtelegram(f'{get_time()}Placing buy limit order of {round(crypto_per_exchange, 6)} '
                            f'{config.base_currency} at {average_price} on {exchange_name}.')
            
            # Use createOrder with limit type for better compatibility
            order = ex[exchange_name].create_order(
                config.pair,
                'limit',
                'buy',
                crypto_per_exchange,
                average_price
            )
            
            if order:
                orders_placed.append((exchange_name, order.get('id') or order.get('orderId')))
                printandtelegram(f"{get_time()}Order placed on {exchange_name}, ID: {order.get('id') or order.get('orderId')}")
            
            time.sleep(0.5)  # Rate limiting
            
        except Exception as e:
            printerror(m=f"Error placing initial order on {exchange_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not orders_placed:
        printerror(m="No initial orders were placed")
        return False
    
    printandtelegram(f"{get_time()}All orders sent. Waiting for fills...")
    
    # Wait for orders to fill
    start_time = time.time()
    timeout = first_orders_fill_timeout if first_orders_fill_timeout > 0 else DEFAULT_TIMEOUT_SECONDS
    
    while time.time() - start_time < timeout:
        all_filled = True
        for exchange_name, order_id in orders_placed:
            try:
                order_status = ex[exchange_name].fetch_order(order_id, config.pair)
                if order_status['status'] != 'closed' and order_status.get('filled', 0) == 0:
                    all_filled = False
                    break
                elif order_status.get('filled', 0) > 0 and order_status['status'] != 'closed':
                    # Partially filled, still waiting
                    all_filled = False
                    break
            except Exception as e:
                printerror(m=f"Error checking order status on {exchange_name}: {e}")
                all_filled = False
                break
        
        if all_filled:
            # Calculate average entry price from filled orders
            total_cost = 0.0
            total_crypto_bought = 0.0
            for exchange_name, order_id in orders_placed:
                try:
                    order_status = ex[exchange_name].fetch_order(order_id, config.pair)
                    filled = order_status.get('filled', 0)
                    avg_price = order_status.get('average', average_price)
                    if filled > 0:
                        total_cost += filled * avg_price
                        total_crypto_bought += filled
                except:
                    pass
            
            # Update state with entry price if state is provided
            if state is not None and total_crypto_bought > 0:
                state.average_entry_price = total_cost / total_crypto_bought
                state.total_cost_basis = total_cost
                state.total_crypto_bought = total_crypto_bought
                printandtelegram(f"{get_time()}Average entry price recorded: ${state.average_entry_price:.2f}")
            
            printandtelegram(f"{get_time()}All initial orders filled.")
            return True
        
        time.sleep(2)
    
    # Timeout - cancel remaining orders
    printerror(m=f"Initial orders did not fill within {timeout} seconds. Canceling remaining orders...")
    for exchange_name, order_id in orders_placed:
        try:
            order_status = ex[exchange_name].fetch_order(order_id, config.pair)
            if order_status['status'] != 'closed':
                ex[exchange_name].cancel_order(order_id, config.pair)
                printandtelegram(f"{get_time()}Canceled unfilled order on {exchange_name}")
        except Exception as e:
            printerror(m=f"Error canceling order on {exchange_name}: {e}")
    
    # Still update entry price with what was filled (in case of timeout)
    total_cost = 0.0
    total_crypto_bought = 0.0
    for exchange_name, order_id in orders_placed:
        try:
            order_status = ex[exchange_name].fetch_order(order_id, config.pair)
            filled = order_status.get('filled', 0)
            avg_price = order_status.get('average', average_price)
            if filled > 0:
                total_cost += filled * avg_price
                total_crypto_bought += filled
        except:
            pass
    
    if state is not None and total_crypto_bought > 0:
        state.average_entry_price = total_cost / total_crypto_bought
        state.total_cost_basis = total_cost
        state.total_crypto_bought = total_crypto_bought
    
    return False

async def run_arbitrage_session(config: TradingConfig) -> None:
    """Run the main arbitrage monitoring session."""
    state = TradingState()
    
    # Log exchange configuration at session start
    printandtelegram(f"{get_time()}=== Exchange Configuration ===")
    printandtelegram(f"{get_time()}Configured exchanges in exchange_config.py: {', '.join(sorted(ex.keys()))}")
    printandtelegram(f"{get_time()}Exchanges used in this session: {', '.join(sorted(config.exchange_names))}")
    printandtelegram(f"{get_time()}Total exchanges available: {len(ex)} | Active in session: {len(config.exchange_names)}")
    printandtelegram(f"{get_time()}================================\n")
    append_new_line('logs/logs.txt', f"{get_time_blank()} INFO: Real money session started with exchanges: {', '.join(sorted(config.exchange_names))}")
    
    # Step 1: Detect and convert leftover cryptocurrency from different pairs
    detect_and_convert_leftover_crypto(config.exchange_names, config.pair, config.quote_currency)
    time.sleep(2)  # Wait for conversions to complete
    
    # Step 2: Get real balances from exchanges
    printandtelegram(f"{get_time()}Fetching real balances from exchanges...")
    
    quote_balances = {}
    crypto_balances = {}
    
    for exchange_name in config.exchange_names:
        try:
            balance = ex[exchange_name].fetch_balance()
            quote_balances[exchange_name] = float(balance[config.quote_currency]['free'])
            crypto_balances[exchange_name] = float(balance[config.base_currency]['free'])
        except Exception as e:
            printerror(m=f"Error fetching balance from {exchange_name}: {e}")
            quote_balances[exchange_name] = 0.0
            crypto_balances[exchange_name] = 0.0
    
    total_quote = sum(quote_balances.values())
    total_crypto = sum(crypto_balances.values())
    
    printandtelegram(f"{get_time()}Total balances: {total_quote:.2f} {config.quote_currency}, "
                    f"{total_crypto:.6f} {config.base_currency}")
    
    # Step 3: Get average price and check balance distribution
    all_prices = []
    for exchange_instance in config.exchange_instances:
        try:
            ticker = exchange_instance.fetch_ticker(config.pair)
            all_prices.append(ticker['last'])
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            printerror(m=f"Error fetching ticker from {exchange_instance.id}: {error_type}: {error_msg}")
            import traceback
            traceback.print_exc()
    
    if not all_prices:
        printerror(m="Could not fetch any price data")
        return
    
    average_price = calculate_average(all_prices)
    
    # Step 4: Check balance distribution and rebalance if needed
    balance_info = check_balance_distribution(config.exchange_names, config.pair, config.total_investment_usd)
    needs_rebalance = should_rebalance_balances(balance_info, config.exchange_names, config.total_investment_usd, average_price)
    
    if needs_rebalance:
        printandtelegram(f"{get_time()}Balance distribution is uneven. Rebalancing to {config.quote_currency}...")
        rebalance_to_quote_currency(config.pair, config.exchange_names)
        time.sleep(3)  # Wait for rebalancing to complete
        
        # Refresh balances after rebalancing
        for exchange_name in config.exchange_names:
            try:
                balance = ex[exchange_name].fetch_balance()
                quote_balances[exchange_name] = float(balance[config.quote_currency]['free'])
                crypto_balances[exchange_name] = float(balance[config.base_currency]['free'])
            except Exception as e:
                printerror(m=f"Error refreshing balance from {exchange_name}: {e}")
        
        total_quote = sum(quote_balances.values())
        total_crypto = sum(crypto_balances.values())
    
    # Step 5: If we don't have enough crypto distributed, place initial orders
    if total_crypto < config.total_investment_usd / average_price * 0.5:
        # Need to buy crypto first
        investment_per_exchange = config.total_investment_usd / len(config.exchange_names)
        target_crypto_per_exchange = investment_per_exchange / average_price
        
        if place_initial_orders(config, average_price, target_crypto_per_exchange * len(config.exchange_names), state):
            # Wait a bit for balances to update
            time.sleep(3)
            # Refresh balances
            for exchange_name in config.exchange_names:
                try:
                    balance = ex[exchange_name].fetch_balance()
                    crypto_balances[exchange_name] = float(balance[config.base_currency]['free'])
                except Exception as e:
                    printerror(m=f"Error refreshing balance from {exchange_name}: {e}")
    
    time.sleep(1)
    printandtelegram(f"{get_time()}Starting arbitrage monitoring with parameters: {sys.argv}")
    
    # Calculate fees using the actual trading pair
    fees = calculate_fees(config.exchange_names, config.pair)
    
    # Log fees for sanity check
    printandtelegram(f"{get_time()}=== Fee Structure for {config.pair} ===")
    for exchange_name in config.exchange_names:
        fee_info = fees.get(exchange_name, {})
        base_fee = fee_info.get('base', 0)
        quote_fee = fee_info.get('quote', 0)
        total_fee_rate = base_fee + quote_fee
        fee_side = "base" if base_fee > 0 else "quote" if quote_fee > 0 else "unknown"
        printandtelegram(f"{get_time()}{exchange_name}: {total_fee_rate*100:.3f}% total fee "
                        f"(base: {base_fee*100:.3f}%, quote: {quote_fee*100:.3f}%, side: {fee_side})")
    printandtelegram(f"{get_time()}================================\n")
    
    # Rank exchanges by fees (for arbitrage suitability)
    fee_rankings = rank_exchanges_by_fees(fees, config.exchange_names)
    printandtelegram(f"{get_time()}=== Exchange Ranking by Fees (Best to Worst for Arbitrage) ===")
    for rank, (exchange_name, total_fee_rate) in enumerate(fee_rankings, 1):
        printandtelegram(f"{get_time()}#{rank}: {exchange_name} - {total_fee_rate*100:.3f}% total fee")
    printandtelegram(f"{get_time()}================================\n")
    append_new_line('logs/logs.txt', f"{get_time_blank()} INFO: Fee structure for {config.pair}: {fees}")
    append_new_line('logs/logs.txt', f"{get_time_blank()} INFO: Fee rankings: {fee_rankings}")
    
    # Calculate crypto per transaction based on available balances
    # Use the minimum available crypto across exchanges to ensure we can trade on all exchanges
    # This is more conservative but ensures we don't run into balance issues
    available_crypto_list = [crypto_balances[ex] for ex in config.exchange_names if crypto_balances[ex] > 0]
    
    if not available_crypto_list:
        printerror(m="No crypto available for trading on any exchange")
        return
    
    # Use the minimum available crypto, but ensure we have enough for meaningful trades
    min_available_crypto = min(available_crypto_list)
    total_available_crypto = sum(crypto_balances.values())
    
    # Calculate crypto per transaction: use minimum of:
    # 1. Minimum available on any exchange (to ensure we can trade everywhere)
    # 2. Average available across exchanges / 2 (conservative approach)
    # 3. Total available / number of exchanges / 2 (original approach)
    crypto_per_transaction = min(
        min_available_crypto * 0.8,  # Use 80% of minimum to be safe
        total_available_crypto / len(config.exchange_names) / 2,
        total_available_crypto / len(config.exchange_names) / 3  # More conservative
    )
    
    if crypto_per_transaction <= 0:
        printerror(m="Calculated crypto per transaction is too small")
        return
    
    printandtelegram(f"{get_time()}Crypto per transaction: {crypto_per_transaction:.6f} {config.base_currency} "
                    f"(min available: {min_available_crypto:.6f}, total: {total_available_crypto:.6f})")
    
    # Setup session parameters
    session_start_time = time.time()
    timeout_timestamp = time.time() + (config.timeout_minutes * 60)
    
    # Start listener thread for manual exit
    listener_thread = threading.Thread(target=listen_for_manual_exit, args=(state,))
    listener_thread.daemon = True
    listener_thread.start()
    
    # Create exchange monitoring tasks
    exchange_tasks = []
    for exchange_instance in config.exchange_instances:
        # Create ccxt.pro instance with proper configuration
        pro_exchange = getattr(ccxt.pro, exchange_instance.id)({
            'enableRateLimit': True,
            'options': {
                'defaultType': 'spot',
            }
        })
        task = monitor_arbitrage_opportunities(
            pro_exchange, config, state, fees, crypto_per_transaction,
            session_start_time, timeout_timestamp
        )
        exchange_tasks.append(task)
    
    # Run all monitoring tasks
    await asyncio.gather(*exchange_tasks)
    
    # Calculate final balances
    printandtelegram(f"{get_time()}Session finished. Calculating final balances...")
    total_final_balance = 0.0
    
    for exchange_name in config.exchange_names:
        try:
            balance = ex[exchange_name].fetch_balance()
            ticker = ex[exchange_name].fetch_ticker(config.pair)
            current_price = ticker['last']
            
            quote_bal = float(balance[config.quote_currency]['free'])
            crypto_bal = float(balance[config.base_currency]['free'])
            
            total_final_balance += quote_bal + (crypto_bal * current_price)
        except Exception as e:
            printerror(m=f"Error calculating final balance for {exchange_name}: {e}")
    
    # Update balance files
    try:
        with open('usable_balance.txt', 'r') as f:
            initial_balance = float(f.read().strip())
    except:
        initial_balance = config.total_investment_usd
    
    try:
        with open('usable_balance.txt', 'w') as f:
            f.write(str(total_final_balance))
        
        session_profit = total_final_balance - initial_balance
        printandtelegram(f"{get_time()}Session finished.")
        printandtelegram(f"{get_time()}Total session profit: {session_profit:.4f} {config.quote_currency}")
        
    except Exception as e:
        printerror(m=f"Error updating final balance: {e}")
    
    # Rebalance at session end: convert all crypto back to quote currency
    # This ensures clean state for next session or pair change
    # Only rebalance if enabled and if it won't result in significant losses
    if AUTO_REBALANCE_ON_EXIT:
        average_entry = state.average_entry_price if state is not None and state.average_entry_price > 0 else 0.0
        rebalanced = rebalance_to_quote_currency(config.pair, config.exchange_names, average_entry, force=False)
        if rebalanced:
            time.sleep(2)  # Wait for rebalancing to complete
        else:
            printandtelegram(f"{get_time()}Rebalancing skipped to avoid locking in losses. "
                           f"Crypto positions remain on exchanges.")
    else:
        printandtelegram(f"{get_time()}Automatic rebalancing disabled. Crypto positions remain on exchanges.")

def main():
    """Main entry point for the real money bot."""
    # Validate arguments
    validate_arguments()
    
    # Set timeout if not configured
    global first_orders_fill_timeout
    if first_orders_fill_timeout <= 0:
        first_orders_fill_timeout = DEFAULT_TIMEOUT_SECONDS
    
    # Parse arguments
    pair = str(sys.argv[1]).upper()
    # Use command line argument if provided, otherwise use default from config
    total_investment = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TARGET_INVESTMENT_USD
    timeout_minutes = int(sys.argv[3]) if len(sys.argv) > 3 else 525600  # Default: 1 year
    session_title = str(sys.argv[4]) if len(sys.argv) > 4 else f"Real-Money-{pair.replace('/', '')}"
    exchange_list_str = sys.argv[5] if len(sys.argv) > 5 else ','.join(ex.keys())
    
    # Setup configuration
    exchange_names = setup_exchanges(exchange_list_str)
    exchange_instances = [ex[name] for name in exchange_names]
    
    config = TradingConfig(
        pair=pair,
        base_currency=pair.split('/')[0],
        quote_currency=pair.split('/')[1],
        total_investment_usd=total_investment,
        timeout_minutes=timeout_minutes,
        session_title=session_title,
        exchange_names=exchange_names,
        exchange_instances=exchange_instances
    )
    
    print()  # Add spacing
    
    # Run the arbitrage session
    asyncio.run(run_arbitrage_session(config))

if __name__ == "__main__":
    main()

