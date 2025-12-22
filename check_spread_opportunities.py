#!/usr/bin/env python3
"""
Spread Opportunity Checker - Standalone Utility
Quickly checks all exchanges for arbitrage opportunities and recommends best exchange combinations.
Run this before starting the bot to identify which exchanges have the best spreads.
"""

import sys
import argparse
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Style, init
from exchange_config import (
    exchanges, get_time, ex, ccxt, calculate_fees
)
from typing import Dict, List, Any, Tuple, Optional

# Initialize colorama
init()

def fetch_ticker_price_sync(exchange_name: str, pair: str) -> tuple:
    """Fetch current price from an exchange using ticker (synchronous)."""
    try:
        if exchange_name not in ex:
            try:
                default_config = {
                    'enableRateLimit': True,
                    'options': {'defaultType': 'spot'}
                }
                ex[exchange_name] = getattr(ccxt, exchange_name)(default_config)
            except:
                return exchange_name, None, None, "Exchange not supported"
        
        ticker = ex[exchange_name].fetch_ticker(pair)
        bid = ticker.get('bid', 0)
        ask = ticker.get('ask', 0)
        return exchange_name, bid, ask, None
    except Exception as e:
        return exchange_name, None, None, str(e)

async def fetch_ticker_price_async(exchange_name: str, pair: str, executor: ThreadPoolExecutor) -> tuple:
    """Fetch current price asynchronously using thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(executor, fetch_ticker_price_sync, exchange_name, pair)

async def check_all_exchanges_spread(pair: str, min_spread_pct: float = 0.05) -> dict:
    """
    Check all exchanges for arbitrage opportunities.
    
    Args:
        pair: Trading pair to check (e.g., 'BTC/USDT')
        min_spread_pct: Minimum spread percentage to consider (default: 0.05%)
    
    Returns:
        Dictionary with opportunities and recommended exchanges
    """
    print(f"{get_time()}Fetching prices from all exchanges...\n")
    
    # Fetch prices from all exchanges concurrently with real-time logging
    exchange_prices = {}
    exchange_list = sorted(exchanges.keys())
    total_exchanges = len(exchange_list)
    
    # Use thread pool executor to run synchronous CCXT calls concurrently
    with ThreadPoolExecutor(max_workers=min(20, total_exchanges)) as executor:
        # Create tasks for all exchanges
        tasks = [fetch_ticker_price_async(exchange_name, pair, executor) for exchange_name in exchange_list]
        
        # Process results as they complete for real-time logging
        completed = 0
        for coro in asyncio.as_completed(tasks):
            exchange_name, bid, ask, error = await coro
            completed += 1
            
            if error:
                print(f"{get_time()}[{completed}/{total_exchanges}]   {exchange_name:20s} - ❌ Error: {error[:40]}", flush=True)
            elif bid and ask and bid > 0 and ask > 0:
                exchange_prices[exchange_name] = {'bid': bid, 'ask': ask}
                spread_pct = ((ask - bid) / bid) * 100 if bid > 0 else 0
                marker = "✓" if spread_pct >= min_spread_pct else " "
                print(f"{get_time()}[{completed}/{total_exchanges}] {marker} {exchange_name:20s} - "
                      f"Bid: {bid:.2f} | Ask: {ask:.2f} | Spread: {spread_pct:.4f}%", flush=True)
            else:
                print(f"{get_time()}[{completed}/{total_exchanges}]   {exchange_name:20s} - ❌ No price data", flush=True)
    
    print()
    
    if len(exchange_prices) < 2:
        print(f"{get_time()}Not enough exchanges with valid prices. Need at least 2 exchanges.")
        return {'opportunities': [], 'recommended_exchanges': []}
    
    # Calculate fees for exchanges with valid prices
    print(f"{get_time()}Calculating fees and finding opportunities...\n")
    fees = calculate_fees(list(exchange_prices.keys()), pair)
    
    # Find all arbitrage opportunities
    opportunities = []
    
    def safe_get_fee_rate(fee_info: dict) -> float:
        """Safely extract fee rate, handling None, lists, nested lists, and other types."""
        def extract_fee_value(value):
            """Recursively extract numeric fee value from various types."""
            if value is None:
                return 0
            
            # Handle list types (some exchanges return [maker, taker] or nested lists)
            if isinstance(value, list):
                if len(value) == 0:
                    return 0
                # Get last element (usually taker fee)
                last_elem = value[-1]
                # If it's still a list, recurse
                if isinstance(last_elem, list):
                    return extract_fee_value(last_elem)
                # Otherwise try to convert to float
                try:
                    return float(last_elem)
                except (ValueError, TypeError):
                    return 0
            
            # Convert to float if not already
            try:
                return float(value) if value else 0
            except (ValueError, TypeError):
                return 0
        
        base_fee = fee_info.get('base', 0)
        quote_fee = fee_info.get('quote', 0)
        
        base_fee_value = extract_fee_value(base_fee)
        quote_fee_value = extract_fee_value(quote_fee)
        
        return base_fee_value + quote_fee_value
    
    # Find all arbitrage opportunities: BUY on one exchange, SELL on another
    total_pairs_evaluated = 0
    profitable_pairs = 0
    
    for buy_exchange, buy_data in exchange_prices.items():
        buy_price = buy_data['ask']  # We BUY at ask price on buy_exchange
        buy_fee_info = fees.get(buy_exchange, {})
        buy_fee_rate = safe_get_fee_rate(buy_fee_info)
        
        for sell_exchange, sell_data in exchange_prices.items():
            if buy_exchange == sell_exchange:
                continue  # Skip same exchange pairs
            
            total_pairs_evaluated += 1
            
            sell_price = sell_data['bid']  # We SELL at bid price on sell_exchange
            sell_fee_info = fees.get(sell_exchange, {})
            sell_fee_rate = safe_get_fee_rate(sell_fee_info)
            
            # Calculate spread percentage
            spread_pct = ((sell_price - buy_price) / buy_price) * 100 if buy_price > 0 else 0
            
            # Calculate profit per unit (after fees)
            buy_cost_with_fees = buy_price * (1 + buy_fee_rate)
            sell_proceeds_after_fees = sell_price * (1 - sell_fee_rate)
            profit_per_unit = sell_proceeds_after_fees - buy_cost_with_fees
            profit_pct = (profit_per_unit / buy_price) * 100 if buy_price > 0 else 0
            
            # Only include profitable opportunities
            if profit_pct > 0:
                profitable_pairs += 1
                opportunities.append({
                    'buy_exchange': buy_exchange,
                    'sell_exchange': sell_exchange,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'spread_pct': spread_pct,
                    'profit_pct': profit_pct,
                    'total_fees_pct': (buy_fee_rate + sell_fee_rate) * 100,
                    'buy_fee': buy_fee_rate * 100,
                    'sell_fee': sell_fee_rate * 100
                })
    
    print(f"{get_time()}Evaluated {total_pairs_evaluated} exchange pairs")
    print(f"{get_time()}Found {profitable_pairs} profitable opportunities (after fees)\n")
    
    # Sort by profit percentage (highest first)
    opportunities.sort(key=lambda x: x['profit_pct'], reverse=True)
    
    # Extract unique exchanges from top opportunities
    recommended_exchanges = set()
    for opp in opportunities[:20]:  # Top 20 opportunities
        recommended_exchanges.add(opp['buy_exchange'])
        recommended_exchanges.add(opp['sell_exchange'])
    
    return {
        'opportunities': opportunities,
        'recommended_exchanges': sorted(list(recommended_exchanges)),
        'exchange_prices': exchange_prices,
        'fees': fees
    }

def main():
    """Main entry point for spread checking utility."""
    parser = argparse.ArgumentParser(
        description='Check all exchanges for arbitrage opportunities',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 check_spread_opportunities.py
  python3 check_spread_opportunities.py --pair ETH/USDT
  python3 check_spread_opportunities.py --min-spread 0.1
  python3 check_spread_opportunities.py --pair BTC/USDT --min-spread 0.05 --top 10
        """
    )
    
    parser.add_argument(
        '--pair',
        type=str,
        default='BTC/USDT',
        help='Trading pair to check (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--min-spread',
        type=float,
        default=0.05,
        help='Minimum spread percentage to display (default: 0.05%%)'
    )
    
    parser.add_argument(
        '--top',
        type=int,
        default=20,
        help='Number of top opportunities to show (default: 20)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"{'Spread Opportunity Checker - Standalone Utility':^70}")
    print(f"{'='*70}\n")
    print(f"{get_time()}Checking {len(exchanges)} exchanges for arbitrage opportunities")
    print(f"{get_time()}Trading pair: {args.pair}")
    print(f"{get_time()}Minimum spread: {args.min_spread:.2f}%%\n")
    
    try:
        # Check all exchanges
        results = asyncio.run(check_all_exchanges_spread(args.pair, args.min_spread))
        
        opportunities = results['opportunities']
        recommended = results['recommended_exchanges']
        
        if not opportunities:
            print(f"{get_time()}{Fore.YELLOW}No profitable opportunities found at this time.{Style.RESET_ALL}\n")
            print(f"{get_time()}This could mean:")
            print(f"{get_time()}  - Spreads are too small to cover fees")
            print(f"{get_time()}  - Markets are efficient (no arbitrage)")
            print(f"{get_time()}  - Try checking again during higher volatility\n")
            return 0
        
        # Filter opportunities by minimum spread
        filtered_opps = [opp for opp in opportunities if opp['spread_pct'] >= args.min_spread]
        
        print(f"{get_time()}{'='*70}")
        print(f"{get_time()}=== Top {min(args.top, len(filtered_opps))} Arbitrage Opportunities ===")
        
        for idx, opp in enumerate(filtered_opps[:args.top], 1):
            color = Fore.GREEN if opp['profit_pct'] > 0.1 else Fore.YELLOW if opp['profit_pct'] > 0.05 else Fore.WHITE
            print(f"{get_time()}#{idx:2d}: {color}{opp['profit_pct']:+.4f}%% profit{Style.RESET_ALL} "
                  f"(spread: {opp['spread_pct']:+.4f}%%, fees: {opp['total_fees_pct']:.3f}%%)")
            print(f"{get_time()}     BUY  on {opp['buy_exchange']:15s} @ {opp['buy_price']:.2f} "
                  f"(fee: {opp['buy_fee']:.3f}%%)")
            print(f"{get_time()}     SELL on {opp['sell_exchange']:15s} @ {opp['sell_price']:.2f} "
                  f"(fee: {opp['sell_fee']:.3f}%%)")
            print()
        
        print(f"{get_time()}{'='*70}\n")
        
        # Show recommended exchanges
        if recommended:
            print(f"{get_time()}{Fore.CYAN}💡 Recommended Exchanges for Arbitrage:{Style.RESET_ALL}")
            print(f"{get_time()}  {', '.join(recommended)}\n")
            print(f"{get_time()}{Fore.CYAN}💡 Example command to use recommended exchanges:{Style.RESET_ALL}")
            print(f"{get_time()}  python3 main.py fake-money 5000 {args.pair} {','.join(recommended[:6])}\n")
        
        # Summary
        print(f"{get_time()}Summary:")
        print(f"{get_time()}  Total opportunities found: {len(opportunities)}")
        print(f"{get_time()}  Opportunities with spread >= {args.min_spread:.2f}%%: {len(filtered_opps)}")
        print(f"{get_time()}  Recommended exchanges: {len(recommended)}\n")
        
        return 0
        
    except KeyboardInterrupt:
        print(f"\n{get_time()}Interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n{get_time()}Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

