#!/usr/bin/env python3
"""
Barbotine Arbitrage Bot - Fake Money Mode
Simulates arbitrage trading with fake money for testing and demonstration.
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
    ex, get_exchange_instance, python_command, first_orders_fill_timeout, demo_fake_delay, 
    demo_fake_delay_ms, criteria_pct, criteria_usd, printerror, 
    get_time, get_time_blank, append_new_line, append_list_to_file,
    printandtelegram, calculate_average, send_to_telegram, get_balance,
    emergency_convert_list, MIN_ORDER_VALUE_USD, AUTO_REBALANCE_ON_EXIT,
    REBALANCE_LOSS_THRESHOLD, ORDERBOOK_FETCH_DELAY, BALANCE_CACHE_TTL,
    MIN_ITERATION_DELAY, RATE_LIMIT_BACKOFF_BASE, MAX_RATE_LIMIT_BACKOFF,
    DEFAULT_TARGET_INVESTMENT_USD, OPPORTUNITY_LOG_THROTTLE, OPPORTUNITY_LOG_DEDUPE,
    calculate_optimal_order_size, DYNAMIC_ORDER_SIZING, MAX_ORDER_SIZE_PCT
)

# Constants
REQUIRED_ARGS = 6
# MIN_ORDER_VALUE_USD is imported from exchange_config
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
        print(f"{python_command} bot-fake-money.py [pair] [total_usdt_investment] [timeout_minutes] [session_title] [exchange_list]")
        print(f"\nReceived arguments: {sys.argv}")
        sys.exit(1)

def setup_exchanges(exchange_list_str: str) -> List[str]:
    """Setup and validate exchange instances (lazy initialization)."""
    exchange_names = [name.strip() for name in exchange_list_str.split(',')]
    
    # Validate and initialize exchanges on-demand
    for exchange_name in exchange_names:
        try:
            # This will initialize the exchange if needed
            get_exchange_instance(exchange_name)
        except ValueError as e:
            printerror(m=f"Exchange {exchange_name} not in configured list: {e}")
            sys.exit(1)
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
            exchange_instance = get_exchange_instance(exchange_name)
            markets = exchange_instance.load_markets()
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
        
        # NOTE: watch_order_book uses WebSocket connections that stay OPEN between calls.
        # This maintains a persistent connection - we do NOT reconnect on each iteration.
        # The connection is only recreated on errors (see error handling below).
        # This is efficient and doesn't waste rate limits.
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
        # NOTE: This is the ONLY place where we reconnect. Reconnections only happen on errors.
        # Normal operation maintains persistent WebSocket connections - no reconnections needed.
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

def simulate_order_delay(min_ask_exchange: str, max_bid_exchange: str, pair: str) -> tuple:
    """Simulate network delay and return updated prices."""
    if not demo_fake_delay:
        return None, None
    
    asyncio.run(asyncio.sleep(demo_fake_delay_ms / 1000))
    
    try:
        # Use lazy initialization to get exchange instances
        min_ask_ex = get_exchange_instance(min_ask_exchange)
        max_bid_ex = get_exchange_instance(max_bid_exchange)
        
        # Fetch orderbook synchronously (not using watch_order_book)
        min_ask_ob = min_ask_ex.fetch_order_book(pair)
        max_bid_ob = max_bid_ex.fetch_order_book(pair)
        
        if min_ask_ob and max_bid_ob and min_ask_ob.get('asks') and max_bid_ob.get('bids'):
            return min_ask_ob['asks'][0][0], max_bid_ob['bids'][0][0]
    except Exception as e:
        printerror(m=f"Error in delay simulation: {e}")
    
    return None, None

def execute_simulated_trades(
    min_ask_exchange: str,
    max_bid_exchange: str, 
    min_ask_price: float,
    max_bid_price: float,
    crypto_per_transaction: float,
    profit_usd: float,
    crypto_balances: Dict[str, float],
    usd_balances: Dict[str, float],
    fees: Dict[str, Dict[str, float]],
    base_currency: str,
    state: TradingState = None
) -> None:
    """Execute simulated arbitrage trades and update balances."""
    
    # Log the trades
    printandtelegram(f"{get_time()}Sell market order filled on {max_bid_exchange} "
                    f"for {crypto_per_transaction:.6f} {base_currency} at {max_bid_price}")
    printandtelegram(f"{get_time()}Buy market order filled on {min_ask_exchange} "
                    f"for {crypto_per_transaction:.6f} {base_currency} at {min_ask_price}")
    
    # Record the profit
    append_list_to_file('all_opportunities_profits.txt', profit_usd)
    
    # Update balances
    # Get fee rates
    buy_fee_rate = fees[min_ask_exchange].get('base', 0) + fees[min_ask_exchange].get('quote', 0)
    sell_fee_rate = fees[max_bid_exchange].get('base', 0) + fees[max_bid_exchange].get('quote', 0)
    
    # Buy on min_ask exchange
    crypto_balances[min_ask_exchange] += crypto_per_transaction
    buy_cost = crypto_per_transaction * min_ask_price * (1 + buy_fee_rate)
    usd_balances[min_ask_exchange] -= buy_cost
    
    # Sell on max_bid exchange
    crypto_balances[max_bid_exchange] -= crypto_per_transaction
    sell_proceeds = crypto_per_transaction * max_bid_price * (1 - sell_fee_rate)
    usd_balances[max_bid_exchange] += sell_proceeds
    
    # Update entry price tracking (weighted average)
    # This tracks the average cost basis for rebalancing decisions
    if state is not None:
        cost_of_this_buy = crypto_per_transaction * min_ask_price
        state.total_cost_basis += cost_of_this_buy
        state.total_crypto_bought += crypto_per_transaction
        if state.total_crypto_bought > 0:
            state.average_entry_price = state.total_cost_basis / state.total_crypto_bought

async def monitor_arbitrage_opportunities(
    exchange_instance,
    config: TradingConfig,
    state: TradingState,
    crypto_balances: Dict[str, float],
    usd_balances: Dict[str, float],
    fees: Dict[str, Dict[str, float]],
    crypto_per_transaction: float,
    session_start_time: float,
    timeout_timestamp: float
) -> None:
    """Monitor for arbitrage opportunities on a single exchange."""
    
    backoff_delay = 0.0  # Track backoff delay for this exchange
    
    # Only the first exchange (alphabetically) should log opportunities to prevent duplicates
    # This is a simple way to ensure only one task logs without needing locks
    should_log = exchange_instance.id == min(config.exchange_names)
    
    # Track last logged opportunity for deduplication
    last_logged_opportunity = None
    last_log_time = 0.0
    
    # Timing tracking for performance debugging
    iteration_count = 0
    last_timing_log = time.time()
    last_heartbeat_log = time.time()  # For periodic status updates
    
    while time.time() <= timeout_timestamp:
        iteration_start = time.time()
        iteration_count += 1
        fetch_time = 0.0
        calc_time = 0.0
        
        if state.stop_requested:
            clear_console_lines(CONSOLE_CLEAR_LINES)
            print(f"{get_time()}Manual rebalance requested. Breaking.")
            append_new_line('logs/logs.txt', f"{get_time_blank()} INFO: Manual rebalance requested.")
            await exchange_instance.close()
            break
        
        # Fetch orderbook with backoff delay
        fetch_start = time.time()
        result = await fetch_orderbook_safe(exchange_instance, config.pair, backoff_delay)
        fetch_time = time.time() - fetch_start
        
        if result is None:
            # If fetch failed, wait a bit before retrying to avoid tight loop
            if should_log and fetch_time > 0.5:
                print(f"{get_time()}[TIMING] {exchange_instance.id}: orderbook fetch failed after {fetch_time:.3f}s")
            # Log heartbeat even on errors
            if should_log:
                current_time = time.time()
                heartbeat_interval = 10.0
                should_heartbeat = (current_time - last_heartbeat_log) >= heartbeat_interval
                if should_heartbeat:
                    print(f"{get_time()}[MONITORING] {exchange_instance.id}: Recovering from orderbook fetch error...")
                    last_heartbeat_log = current_time
            await asyncio.sleep(1.0)
            continue
        
        orderbook, new_backoff = result if isinstance(result, tuple) else (result, 0.0)
        backoff_delay = new_backoff  # Update backoff delay
        
        if not orderbook:
            # If no orderbook, wait before retrying
            if should_log:
                current_time = time.time()
                heartbeat_interval = 10.0
                should_heartbeat = (current_time - last_heartbeat_log) >= heartbeat_interval
                if should_heartbeat:
                    print(f"{get_time()}[MONITORING] {exchange_instance.id}: No orderbook received, retrying...")
                    last_heartbeat_log = current_time
            await asyncio.sleep(0.5)
            continue
        
        # Validate orderbook has bids and asks before accessing
        if (not orderbook.get("bids") or len(orderbook["bids"]) == 0 or
            not orderbook.get("asks") or len(orderbook["asks"]) == 0):
            # Empty orderbook, wait before retrying
            if should_log:
                current_time = time.time()
                heartbeat_interval = 10.0
                should_heartbeat = (current_time - last_heartbeat_log) >= heartbeat_interval
                if should_heartbeat:
                    print(f"{get_time()}[MONITORING] {exchange_instance.id}: Empty orderbook, retrying...")
                    last_heartbeat_log = current_time
            await asyncio.sleep(0.5)
            continue
        
        # Update price data
        state.bid_prices[exchange_instance.id] = orderbook["bids"][0][0]
        state.ask_prices[exchange_instance.id] = orderbook["asks"][0][0]
        
        # Find best arbitrage opportunity by evaluating ALL exchange pairs
        # and selecting the one with highest profit after fees
        calc_start = time.time()
        
        # Get all exchanges with valid prices
        # Only require at least 2 exchanges (one for buy, one for sell) to proceed
        valid_ask_exchanges = {ex: price for ex, price in state.ask_prices.items() if ex in config.exchange_names}
        valid_bid_exchanges = {ex: price for ex, price in state.bid_prices.items() if ex in config.exchange_names}
        
        # Need at least one exchange for buying and one for selling (can be different)
        if len(valid_ask_exchanges) < 1 or len(valid_bid_exchanges) < 1:
            # Not enough exchanges with prices yet, continue monitoring
            # But still log heartbeat if needed
            if should_log:
                current_time = time.time()
                heartbeat_interval = 10.0
                should_heartbeat = (current_time - last_heartbeat_log) >= heartbeat_interval
                if should_heartbeat:
                    print(f"{get_time()}[MONITORING] Waiting for prices from exchanges... "
                          f"(have {len(valid_ask_exchanges)} ask, {len(valid_bid_exchanges)} bid)")
                    last_heartbeat_log = current_time
            continue
        
        # Evaluate ALL possible exchange pairs to find the one with highest profit after fees
        # We track both: best overall opportunity (for display) and best tradeable opportunity (for execution)
        best_profit = float('-inf')
        best_ask_exchange = None
        best_bid_exchange = None
        best_ask_price = 0.0
        best_bid_price = 0.0
        
        # Also track best opportunity regardless of balance constraints (for display)
        best_display_profit = float('-inf')
        best_display_ask_exchange = None
        best_display_bid_exchange = None
        best_display_ask_price = 0.0
        best_display_bid_price = 0.0
        
        for ask_exchange, ask_price in valid_ask_exchanges.items():
            for bid_exchange, bid_price in valid_bid_exchanges.items():
                # Skip if same exchange
                if ask_exchange == bid_exchange:
                    continue
                
                # Calculate profit after fees for this pair (always calculate for display)
                buy_fee_rate = fees.get(ask_exchange, {}).get('base', 0) + fees.get(ask_exchange, {}).get('quote', 0)
                sell_fee_rate = fees.get(bid_exchange, {}).get('base', 0) + fees.get(bid_exchange, {}).get('quote', 0)
                
                # Cost to buy (including fees)
                buy_cost_with_fees = crypto_per_transaction * ask_price * (1 + buy_fee_rate)
                # Proceeds from selling (after fees)
                sell_proceeds_after_fees = crypto_per_transaction * bid_price * (1 - sell_fee_rate)
                
                # Profit = sell proceeds - buy cost
                profit = sell_proceeds_after_fees - buy_cost_with_fees
                
                # Track the best opportunity for display (regardless of balance constraints)
                if profit > best_display_profit:
                    best_display_profit = profit
                    best_display_ask_exchange = ask_exchange
                    best_display_bid_exchange = bid_exchange
                    best_display_ask_price = ask_price
                    best_display_bid_price = bid_price
                
                # Check balance availability (only for tradeable opportunities)
                if crypto_balances.get(bid_exchange, 0) < crypto_per_transaction:
                    continue
                buy_cost = crypto_per_transaction * ask_price
                if usd_balances.get(ask_exchange, 0) < buy_cost:
                    continue
                
                # Track the best tradeable opportunity
                if profit > best_profit:
                    best_profit = profit
                    best_ask_exchange = ask_exchange
                    best_bid_exchange = bid_exchange
                    best_ask_price = ask_price
                    best_bid_price = bid_price
        
        calc_time = time.time() - calc_start
        
        # Use display opportunity if no tradeable opportunity found (for logging purposes)
        # This ensures we always show the best spread, even if we can't trade it
        if best_ask_exchange is None or best_bid_exchange is None:
            # Use display opportunity for logging
            if best_display_ask_exchange and best_display_bid_exchange:
                min_ask_exchange = best_display_ask_exchange
                max_bid_exchange = best_display_bid_exchange
                min_ask_price = best_display_ask_price
                max_bid_price = best_display_bid_price
            else:
                # No opportunities at all - log heartbeat and continue
                if should_log:
                    current_time = time.time()
                    heartbeat_interval = 10.0
                    should_heartbeat = (current_time - last_heartbeat_log) >= heartbeat_interval
                    if should_heartbeat:
                        print(f"{get_time()}[MONITORING] No opportunities found (waiting for exchange prices)")
                        last_heartbeat_log = current_time
                continue
        else:
            # We have a tradeable opportunity - use it
            min_ask_exchange = best_ask_exchange
            max_bid_exchange = best_bid_exchange
            min_ask_price = best_ask_price
            max_bid_price = best_bid_price
        
        # Validate prices before proceeding (guard against division by zero)
        # This prevents "float division by zero" errors when prices are invalid
        if (not min_ask_exchange or not max_bid_exchange or 
            min_ask_price is None or max_bid_price is None or
            min_ask_price <= 0 or max_bid_price <= 0):
            # Invalid prices or exchanges - skip this iteration
            continue
        
        # Calculate profit (we already calculated it above, but recalculate for consistency)
        buy_fee_rate = fees.get(min_ask_exchange, {}).get('base', 0) + fees.get(min_ask_exchange, {}).get('quote', 0)
        sell_fee_rate = fees.get(max_bid_exchange, {}).get('base', 0) + fees.get(max_bid_exchange, {}).get('quote', 0)
        
        # Check spread percentage first (doesn't depend on order size)
        # Guard against division by zero if prices are invalid
        avg_price = (max_bid_price + min_ask_price) / 2
        if avg_price > 0:
            price_diff_pct = abs(min_ask_price - max_bid_price) / avg_price * 100
        else:
            price_diff_pct = 0.0  # Invalid prices, skip this opportunity
            continue
        
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
            available_crypto = crypto_balances.get(max_bid_exchange, 0)
            available_quote = usd_balances.get(min_ask_exchange, 0)
            
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
            
            exchange_balances = format_exchange_balances(
                crypto_balances, usd_balances, config.base_currency, config.quote_currency
            )
            
            elapsed_time = time.strftime('%H:%M:%S', time.gmtime(time.time() - session_start_time))
            
            print(f"{Style.RESET_ALL}Opportunity #{state.opportunity_count} detected! "
                  f"({min_ask_exchange} {min_ask_price} -> {max_bid_price} {max_bid_exchange})\n")
            print(f"Expected profit: {Fore.GREEN}+{round(profit_usd, 4)} {config.quote_currency}{Style.RESET_ALL}")
            print(f"Session total profit: {Fore.GREEN}+{round(state.total_profit_usd, 4)} {config.quote_currency}{Style.RESET_ALL}")
            print(f"Fees paid: {Fore.RED}-{round(fees_usd, 4)} {config.quote_currency} -{round(fees_crypto, 4)} {config.base_currency}")
            print(f"{Style.DIM}{exchange_balances}")
            print(f"Time elapsed: {elapsed_time}")
            print("-----------------------------------------------------\n")
            
            # Send Telegram notification
            telegram_msg = (f"[{config.session_title} Trade #{state.opportunity_count}]\n\n"
                          f"Opportunity detected!\n"
                          f"Expected profit: {round(profit_usd, 4)} {config.quote_currency}\n"
                          f"{min_ask_exchange} {min_ask_price} -> {max_bid_price} {max_bid_exchange}\n"
                          f"Time elapsed: {elapsed_time}\n"
                          f"Session total profit: {round(state.total_profit_usd, 4)} {config.quote_currency}\n"
                          f"Fees: {round(fees_usd, 4)} {config.quote_currency} {round(fees_crypto, 4)} {config.base_currency}\n\n"
                          f"--------BALANCES--------\n{exchange_balances}")
            send_to_telegram(telegram_msg)
            
            # Simulate order delay if enabled
            if demo_fake_delay:
                timestamp = time.time()
                delayed_ask, delayed_bid = simulate_order_delay(min_ask_exchange, max_bid_exchange, config.pair)
                if delayed_ask and delayed_bid:
                    min_ask_price, max_bid_price = delayed_ask, delayed_bid
                    delay_ms = int(round(1000 * (time.time() - timestamp), 0))
                    printandtelegram(f"{get_time()}Calculated P&L with {delay_ms}ms simulated delay")
            
            # Execute simulated trades with optimal order size
            execute_simulated_trades(
                min_ask_exchange, max_bid_exchange, min_ask_price, max_bid_price,
                trade_order_size, profit_usd, crypto_balances, usd_balances,
                fees, config.base_currency, state
            )
            
            # Log if dynamic sizing was used
            if DYNAMIC_ORDER_SIZING and abs(trade_order_size - crypto_per_transaction) > crypto_per_transaction * 0.1:
                size_change_pct = ((trade_order_size - crypto_per_transaction) / crypto_per_transaction) * 100
                printandtelegram(f"{get_time()}Dynamic sizing: Order size adjusted by {size_change_pct:+.1f}% "
                               f"({crypto_per_transaction:.6f} -> {trade_order_size:.6f} {config.base_currency}) "
                               f"based on {price_diff_pct:.3f}% spread")
            
            state.total_profit_usd += profit_usd
            state.previous_ask_price = min_ask_price
            state.previous_bid_price = max_bid_price
            
            # Recalculate crypto per transaction based on minimum available
            # This prevents exchanges from running out of funds
            total_crypto = sum(crypto_balances.values())
            min_crypto = min(crypto_balances.values()) if crypto_balances.values() else 0
            # Use minimum to ensure all exchanges can participate, but don't go below 80% of minimum
            old_crypto_per_transaction = crypto_per_transaction
            crypto_per_transaction = min(
                min_crypto * 0.8 if min_crypto > 0 else 0,
                total_crypto / len(config.exchange_names) / 2  # Conservative fallback
            )
            # Log if order size was significantly reduced
            if should_log and old_crypto_per_transaction > 0:
                reduction_pct = ((crypto_per_transaction - old_crypto_per_transaction) / old_crypto_per_transaction) * 100
                if reduction_pct < -10:  # More than 10% reduction
                    print(f"{get_time()}[BALANCE] Order size reduced by {abs(reduction_pct):.1f}% "
                          f"({old_crypto_per_transaction:.6f} -> {crypto_per_transaction:.6f} {config.base_currency}) "
                          f"due to balance constraints (min: {min_crypto:.6f}, total: {total_crypto:.6f})")
        
        else:
            # Display current best opportunity (only log from one exchange to prevent duplicates)
            # Always show the best spread, even if it's not profitable
            if should_log:
                current_time = time.time()
                current_opportunity = (min_ask_exchange, max_bid_exchange, round(min_ask_price, 2), round(max_bid_price, 2))
                
                # Check if we should log this opportunity
                # Reduced throttling to show opportunities more frequently
                should_log_this = True
                if OPPORTUNITY_LOG_DEDUPE:
                    # Only log if opportunity changed
                    if current_opportunity == last_logged_opportunity:
                        should_log_this = False
                
                # Reduced throttle: show opportunities more frequently (1 second instead of 2)
                reduced_throttle = max(1.0, OPPORTUNITY_LOG_THROTTLE / 2.0)
                if should_log_this and (current_time - last_log_time) < reduced_throttle:
                    should_log_this = False
                
                # Periodic heartbeat: log status every 10 seconds even if opportunity hasn't changed
                # This shows the bot is still running and monitoring
                heartbeat_interval = 10.0  # Log heartbeat every 10 seconds
                should_heartbeat = (current_time - last_heartbeat_log) >= heartbeat_interval
                
                if should_log_this:
                    # Calculate profit with dynamic order sizing for display (if enabled)
                    # This shows what the profit would be with optimal sizing, not just base size
                    fee_calc_start = time.time()
                    buy_fee_rate = fees.get(min_ask_exchange, {}).get('base', 0) + fees.get(min_ask_exchange, {}).get('quote', 0)
                    sell_fee_rate = fees.get(max_bid_exchange, {}).get('base', 0) + fees.get(max_bid_exchange, {}).get('quote', 0)
                    
                    # Use dynamic order sizing if enabled, otherwise use base size
                    if DYNAMIC_ORDER_SIZING:
                        available_crypto = crypto_balances.get(max_bid_exchange, 0)
                        available_quote = usd_balances.get(min_ask_exchange, 0)
                        display_order_size = calculate_optimal_order_size(
                            min_ask_price=min_ask_price,
                            max_bid_price=max_bid_price,
                            available_crypto=available_crypto,
                            available_quote=available_quote,
                            buy_fee_rate=buy_fee_rate,
                            sell_fee_rate=sell_fee_rate,
                            min_order_value=MIN_ORDER_VALUE_USD,
                            base_order_size=crypto_per_transaction
                        )
                        # Log balance constraints if size was significantly reduced (for debugging)
                        # This helps understand why order sizes are reduced (e.g., -75% due to MAX_ORDER_SIZE_PCT = 25%)
                        if should_log and abs(display_order_size - crypto_per_transaction) > crypto_per_transaction * 0.2:
                            max_from_crypto = available_crypto * MAX_ORDER_SIZE_PCT
                            max_from_quote = (available_quote / min_ask_price) * MAX_ORDER_SIZE_PCT if min_ask_price > 0 else 0
                            limiting_factor = "crypto balance" if max_from_crypto < max_from_quote else "quote balance"
                            print(f"{get_time()}[SIZING] Base: {crypto_per_transaction:.6f} -> Optimal: {display_order_size:.6f} "
                                  f"(limited by {limiting_factor}: {min(max_from_crypto, max_from_quote):.6f} = 25% of available)")
                    else:
                        display_order_size = crypto_per_transaction
                    
                    # Calculate profit with display order size
                    buy_cost_with_fees = display_order_size * min_ask_price * (1 + buy_fee_rate)
                    sell_proceeds_after_fees = display_order_size * max_bid_price * (1 - sell_fee_rate)
                    display_profit_usd = sell_proceeds_after_fees - buy_cost_with_fees
                    
                    # Don't clear console - let it scroll naturally to show history
                    color = Fore.GREEN if display_profit_usd > 0 else Fore.RED if display_profit_usd < 0 else Fore.WHITE
                    
                    # Format prices with appropriate precision (more decimals for low-priced tokens)
                    price_precision = 8 if min_ask_price < 1.0 else 6 if min_ask_price < 100.0 else 4
                    
                    # Calculate price spread for clarity
                    price_spread = max_bid_price - min_ask_price
                    spread_pct = (price_spread / min_ask_price) * 100 if min_ask_price > 0 else 0
                    
                    # Calculate total fee in USD for display (using display order size)
                    buy_fee_usd = display_order_size * min_ask_price * buy_fee_rate
                    sell_fee_usd = display_order_size * max_bid_price * sell_fee_rate
                    total_fee_usd = buy_fee_usd + sell_fee_usd
                    fee_calc_time = time.time() - fee_calc_start
                    
                    # Calculate order size in USD for context
                    order_size_usd = display_order_size * min_ask_price
                    
                    # Show if dynamic sizing adjusted the order size
                    size_indicator = ""
                    if DYNAMIC_ORDER_SIZING and abs(display_order_size - crypto_per_transaction) > crypto_per_transaction * 0.05:
                        size_change_pct = ((display_order_size - crypto_per_transaction) / crypto_per_transaction) * 100
                        size_indicator = f" [size: {size_change_pct:+.1f}%]"
                    
                    print(f"{get_time()}Best: {color}{round(display_profit_usd, 4)} {config.quote_currency}{Style.RESET_ALL} "
                          f"(fees: {round(total_fee_usd, 4)}, order: {display_order_size:.6f} {config.base_currency} = ${order_size_usd:.2f}{size_indicator}) "
                          f"buy: {min_ask_exchange} at {min_ask_price:.{price_precision}f} "
                          f"sell: {max_bid_exchange} at {max_bid_price:.{price_precision}f} "
                          f"(spread: {spread_pct:+.4f}%)")
                    
                    # Update tracking
                    last_logged_opportunity = current_opportunity
                    last_log_time = current_time
                    last_heartbeat_log = current_time  # Reset heartbeat timer when we log
                    
                    # Log timing if fee calculation took significant time
                    if fee_calc_time > 0.001:  # More than 1ms
                        print(f"{get_time()}[TIMING] Fee calc took {fee_calc_time*1000:.2f}ms")
                
                elif should_heartbeat:
                    # Periodic heartbeat: show bot is still monitoring even if opportunity hasn't changed
                    # Calculate profit with dynamic order sizing for display (if enabled)
                    buy_fee_rate = fees.get(min_ask_exchange, {}).get('base', 0) + fees.get(min_ask_exchange, {}).get('quote', 0)
                    sell_fee_rate = fees.get(max_bid_exchange, {}).get('base', 0) + fees.get(max_bid_exchange, {}).get('quote', 0)
                    
                    # Use dynamic order sizing if enabled, otherwise use base size
                    if DYNAMIC_ORDER_SIZING:
                        available_crypto = crypto_balances.get(max_bid_exchange, 0)
                        available_quote = usd_balances.get(min_ask_exchange, 0)
                        display_order_size = calculate_optimal_order_size(
                            min_ask_price=min_ask_price,
                            max_bid_price=max_bid_price,
                            available_crypto=available_crypto,
                            available_quote=available_quote,
                            buy_fee_rate=buy_fee_rate,
                            sell_fee_rate=sell_fee_rate,
                            min_order_value=MIN_ORDER_VALUE_USD,
                            base_order_size=crypto_per_transaction
                        )
                    else:
                        display_order_size = crypto_per_transaction
                    
                    # Calculate profit with display order size
                    buy_cost_with_fees = display_order_size * min_ask_price * (1 + buy_fee_rate)
                    sell_proceeds_after_fees = display_order_size * max_bid_price * (1 - sell_fee_rate)
                    display_profit_usd = sell_proceeds_after_fees - buy_cost_with_fees
                    
                    color = Fore.GREEN if display_profit_usd > 0 else Fore.RED if display_profit_usd < 0 else Fore.WHITE
                    price_precision = 8 if min_ask_price < 1.0 else 6 if min_ask_price < 100.0 else 4
                    price_spread = max_bid_price - min_ask_price
                    # Guard against division by zero
                    if min_ask_price > 0:
                        spread_pct = (price_spread / min_ask_price) * 100
                    else:
                        spread_pct = 0.0
                    buy_fee_usd = display_order_size * min_ask_price * buy_fee_rate
                    sell_fee_usd = display_order_size * max_bid_price * sell_fee_rate
                    total_fee_usd = buy_fee_usd + sell_fee_usd
                    
                    # Calculate order size in USD for context
                    order_size_usd = display_order_size * min_ask_price
                    
                    # Show if dynamic sizing adjusted the order size
                    size_indicator = ""
                    if DYNAMIC_ORDER_SIZING and abs(display_order_size - crypto_per_transaction) > crypto_per_transaction * 0.05:
                        size_change_pct = ((display_order_size - crypto_per_transaction) / crypto_per_transaction) * 100
                        size_indicator = f" [size: {size_change_pct:+.1f}%]"
                    
                    print(f"{get_time()}[MONITORING] Best: {color}{round(display_profit_usd, 4)} {config.quote_currency}{Style.RESET_ALL} "
                          f"(fees: {round(total_fee_usd, 4)}, order: {display_order_size:.6f} {config.base_currency} = ${order_size_usd:.2f}{size_indicator}) "
                          f"buy: {min_ask_exchange} at {min_ask_price:.{price_precision}f} "
                          f"sell: {max_bid_exchange} at {max_bid_price:.{price_precision}f} "
                          f"(spread: {spread_pct:+.4f}%) - No change")
                    last_heartbeat_log = current_time
        
        # Calculate total iteration time
        iteration_time = time.time() - iteration_start
        
        # Always check heartbeat at end of iteration (fallback if we missed it earlier)
        # This ensures we log something every 10 seconds even if we hit early continues
        if should_log:
            current_time = time.time()
            heartbeat_interval = 10.0
            should_heartbeat = (current_time - last_heartbeat_log) >= heartbeat_interval
            if should_heartbeat:
                # Count how many exchanges have prices
                valid_asks = len([ex for ex, price in state.ask_prices.items() if ex in config.exchange_names])
                valid_bids = len([ex for ex, price in state.bid_prices.items() if ex in config.exchange_names])
                if valid_asks >= 1 and valid_bids >= 1:
                    # We have prices but didn't log - opportunity probably hasn't changed
                    print(f"{get_time()}[MONITORING] Bot running - monitoring {valid_asks} ask / {valid_bids} bid exchanges")
                else:
                    print(f"{get_time()}[MONITORING] Bot running - waiting for exchange prices... ({valid_asks} ask, {valid_bids} bid)")
                last_heartbeat_log = current_time
        
        # Log timing every 10 iterations or if iteration took > 1 second
        if should_log and (iteration_count % 10 == 0 or iteration_time > 1.0 or 
                          (time.time() - last_timing_log) > 5.0):
            print(f"{get_time()}[TIMING] {exchange_instance.id} iter #{iteration_count}: "
                  f"fetch={fetch_time*1000:.1f}ms, calc={calc_time*1000:.1f}ms, total={iteration_time*1000:.1f}ms")
            last_timing_log = time.time()
        
        # Add minimum delay between iterations (target ~2 seconds per update)
        # This ensures continuous monitoring without overwhelming APIs
        # Adjust delay to target ~2 seconds total per iteration
        # NOTE: This sleep is just a rate-limiting delay between iterations.
        # The WebSocket connection (watch_order_book) stays OPEN between iterations - we do NOT reconnect.
        # Reconnections only happen on errors (see fetch_orderbook_safe error handling).
        # This prevents wasting rate limits by maintaining persistent connections.
        target_iteration_time = 2.0  # Target 2 seconds per iteration
        elapsed_time = time.time() - iteration_start
        remaining_delay = max(0.0, target_iteration_time - elapsed_time)
        iteration_delay = max(MIN_ITERATION_DELAY, remaining_delay)
        
        await asyncio.sleep(iteration_delay)

def simulate_initial_orders(config: TradingConfig, average_price: float, total_crypto: float) -> None:
    """Simulate the initial order placement that would happen in real money mode."""
    printandtelegram(f"{get_time()}Fetching the global average price for {config.pair}...")
    printandtelegram(f"{get_time()}Average {config.pair} price in {config.quote_currency}: {average_price}")
    
    # Simulate placing initial buy orders
    crypto_per_exchange = total_crypto / len(config.exchange_names)
    
    orders_filled = 0
    target_orders = len(config.exchange_names)
    
    # Show order placement simulation
    for i, exchange_name in enumerate(config.exchange_names):
        time.sleep(0.7)  # Simulate network delay
        printandtelegram(f'{get_time()}Buy limit order of {round(crypto_per_exchange, 6)} '
                        f'{config.base_currency} at {average_price} sent to {exchange_name}.')
    
    printandtelegram(f"{get_time()}All orders sent.")
    
    # Simulate order filling process
    while orders_filled != target_orders:
        for exchange_name in config.exchange_names:
            if orders_filled >= target_orders:
                break
            time.sleep(2.1)  # Simulate order fill time
            printandtelegram(f"{get_time()}{exchange_name} order filled.")
            orders_filled += 1

async def run_arbitrage_session(config: TradingConfig) -> None:
    """Run the main arbitrage monitoring session."""
    state = TradingState()
    
    # Log exchange configuration at session start
    printandtelegram(f"{get_time()}=== Exchange Configuration ===")
    printandtelegram(f"{get_time()}Exchanges used in this session: {', '.join(sorted(config.exchange_names))}")
    printandtelegram(f"{get_time()}Active exchanges in session: {len(config.exchange_names)}")
    printandtelegram(f"{get_time()}Note: Only active exchanges are initialized (lazy loading)")
    printandtelegram(f"{get_time()}================================\n")
    append_new_line('logs/logs.txt', f"{get_time_blank()} INFO: Fake money session started with exchanges: {', '.join(sorted(config.exchange_names))}")
    
    # Setup initial balances
    usd_balances = {exchange: (config.total_investment_usd / 2) / len(config.exchange_names) 
                   for exchange in config.exchange_names}
    
    # Get average price and calculate initial crypto allocation
    all_prices = []
    for exchange_name in config.exchange_names:
        try:
            exchange_instance = get_exchange_instance(exchange_name)
            ticker = exchange_instance.fetch_ticker(config.pair)
            all_prices.append(ticker['last'])
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            printerror(m=f"Error fetching ticker from {exchange_name}: {error_type}: {error_msg}")
            # Print full traceback for debugging SSL/connection issues
            import traceback
            traceback.print_exc()
    
    if not all_prices:
        printerror(m="Could not fetch any price data")
        return
    
    average_price = calculate_average(all_prices)
    total_crypto = (config.total_investment_usd / 2) / average_price
    crypto_balances = {exchange: total_crypto / len(config.exchange_names) 
                      for exchange in config.exchange_names}
    
    # Track initial entry price
    if state is not None:
        state.average_entry_price = average_price
        state.total_cost_basis = (config.total_investment_usd / 2)
        state.total_crypto_bought = total_crypto
        printandtelegram(f"{get_time()}Average entry price recorded: ${state.average_entry_price:.2f}")
    
    # Simulate the initial order placement process (like real money mode would do)
    simulate_initial_orders(config, average_price, total_crypto)
    
    time.sleep(1)  # Brief pause before starting main session
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
    
    # Initial crypto per transaction
    # NOTE: This scales inversely with price to keep USD order size similar across different tokens.
    # For example: BTC at $87k -> ~0.004 BTC per order (~$350), XRP at $1.87 -> ~191 XRP per order (~$350)
    # This is why profit values in USD are similar - the order sizes are similar in USD value.
    crypto_per_transaction = total_crypto / len(config.exchange_names)
    order_size_usd_initial = crypto_per_transaction * average_price
    printandtelegram(f"{get_time()}Initial order size: {crypto_per_transaction:.6f} {config.base_currency} "
                    f"(~${order_size_usd_initial:.2f} per transaction)")
    printandtelegram(f"{get_time()}Initial balances: {total_crypto:.6f} {config.base_currency} total "
                    f"({crypto_per_transaction:.6f} per exchange), "
                    f"${config.total_investment_usd/2:.2f} {config.quote_currency} total "
                    f"(${config.total_investment_usd/2/len(config.exchange_names):.2f} per exchange)")
    if DYNAMIC_ORDER_SIZING:
        printandtelegram(f"{get_time()}Dynamic sizing: MAX_ORDER_SIZE_PCT={MAX_ORDER_SIZE_PCT*100:.0f}% "
                        f"(orders limited to {MAX_ORDER_SIZE_PCT*100:.0f}% of available balance per trade)")
    
    # Setup session parameters
    session_start_time = time.time()
    timeout_timestamp = time.time() + (config.timeout_minutes * 60)
    
    # Start listener thread for manual exit
    listener_thread = threading.Thread(target=listen_for_manual_exit, args=(state,))
    listener_thread.daemon = True
    listener_thread.start()
    
    # Create exchange monitoring tasks
    exchange_tasks = []
    for exchange_name in config.exchange_names:
        try:
            # Get the regular exchange instance first to validate it works
            exchange_instance = get_exchange_instance(exchange_name)
            
            # Create ccxt.pro instance with proper configuration for WebSocket monitoring
            pro_exchange = getattr(ccxt.pro, exchange_instance.id)({
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',
                }
            })
            task = monitor_arbitrage_opportunities(
                pro_exchange, config, state, crypto_balances, usd_balances,
                fees, crypto_per_transaction, session_start_time, timeout_timestamp
            )
            exchange_tasks.append(task)
            printandtelegram(f"{get_time()}Started monitoring task for {exchange_name}")
        except Exception as e:
            printerror(m=f"Failed to start monitoring for {exchange_name}: {e}")
            append_new_line('logs/logs.txt', 
                f"{get_time_blank()} ERROR: Failed to start monitoring for {exchange_name}: {e}")
            # Continue with other exchanges even if one fails
    
    if not exchange_tasks:
        printerror(m="No exchange monitoring tasks could be started. Exiting.")
        return
    
    printandtelegram(f"{get_time()}Started {len(exchange_tasks)} exchange monitoring task(s)")
    
    # Run all monitoring tasks with error handling
    # Use return_exceptions=True so one failing exchange doesn't stop others
    results = await asyncio.gather(*exchange_tasks, return_exceptions=True)
    
    # Log any exceptions that occurred
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            exchange_name = config.exchange_names[i] if i < len(config.exchange_names) else "unknown"
            printerror(m=f"Monitoring task for {exchange_name} failed: {result}")
            append_new_line('logs/logs.txt', 
                f"{get_time_blank()} ERROR: Monitoring task for {exchange_name} failed: {result}")
    
    # Calculate final balances
    # Check if we should rebalance (simulate loss protection)
    total_crypto_value = 0.0
    current_price = 0.0
    
    for exchange_name in config.exchange_names:
        try:
            exchange_instance = get_exchange_instance(exchange_name)
            ticker = exchange_instance.fetchTicker(config.pair)
            current_price = float(ticker['last'])
            total_crypto_value += crypto_balances[exchange_name] * current_price
        except Exception as e:
            printerror(m=f"Error fetching price for {exchange_name}: {e}")
    
    # Check loss threshold if entry price is tracked
    should_rebalance = True
    if state is not None and state.average_entry_price > 0 and current_price > 0:
        loss_pct = (current_price - state.average_entry_price) / state.average_entry_price
        
        if loss_pct < REBALANCE_LOSS_THRESHOLD:
            printandtelegram(f"{get_time()}⚠️  WARNING: Rebalancing would result in {loss_pct*100:.2f}% loss "
                           f"(entry: ${state.average_entry_price:.2f}, current: ${current_price:.2f})")
            printandtelegram(f"{get_time()}Skipping rebalancing to avoid locking in losses. "
                           f"Threshold: {REBALANCE_LOSS_THRESHOLD*100:.2f}%")
            printandtelegram(f"{get_time()}In simulation mode, crypto positions remain in balances.")
            should_rebalance = False
        elif loss_pct < 0:
            printandtelegram(f"{get_time()}Current price is {loss_pct*100:.2f}% below entry, "
                           f"but within acceptable threshold. Proceeding with rebalancing.")
    
    # Rebalance if enabled and within threshold
    total_final_balance = 0.0
    if AUTO_REBALANCE_ON_EXIT and should_rebalance:
        for exchange_name in config.exchange_names:
            try:
                exchange_instance = get_exchange_instance(exchange_name)
                ticker = exchange_instance.fetchTicker(config.pair)
                current_price = float(ticker['last'])
                usd_balances[exchange_name] += crypto_balances[exchange_name] * current_price
                crypto_balances[exchange_name] = 0
                total_final_balance += usd_balances[exchange_name]
            except Exception as e:
                printerror(m=f"Error calculating final balance for {exchange_name}: {e}")
    else:
        # Don't rebalance - keep crypto positions
        for exchange_name in config.exchange_names:
            try:
                exchange_instance = get_exchange_instance(exchange_name)
                ticker = exchange_instance.fetchTicker(config.pair)
                current_price = float(ticker['last'])
                # Calculate total value but don't convert
                total_final_balance += usd_balances[exchange_name] + (crypto_balances[exchange_name] * current_price)
            except Exception as e:
                printerror(m=f"Error calculating final balance for {exchange_name}: {e}")
    
    # Update balance file and display results
    try:
        with open('real_balance.txt', 'r+') as balance_file:
            initial_balance = float(balance_file.read())
            balance_file.seek(0)
            balance_file.write(str(total_final_balance))
        
        session_profit = total_final_balance - initial_balance
        printandtelegram(f"{get_time()}Session finished.")
        printandtelegram(f"{get_time()}Total session profit: {session_profit:.4f} {config.quote_currency}")
        
        if not should_rebalance and AUTO_REBALANCE_ON_EXIT:
            printandtelegram(f"{get_time()}Note: Crypto positions remain in simulation balances (not converted to {config.quote_currency}).")
        
    except Exception as e:
        printerror(m=f"Error updating final balance: {e}")

def main():
    """Main entry point for the fake money bot."""
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
    session_title = str(sys.argv[4]) if len(sys.argv) > 4 else f"Fake-Money-{pair.replace('/', '')}"
    exchange_list_str = sys.argv[5] if len(sys.argv) > 5 else ','.join(ex.keys())
    
    # Setup configuration
    exchange_names = setup_exchanges(exchange_list_str)
    # Initialize exchange instances lazily - only the ones we need
    exchange_instances = [get_exchange_instance(name) for name in exchange_names]
    
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
