#!/usr/bin/env python3
"""
Exchange Fee Checker - Standalone Utility
Checks fees for all configured exchanges and identifies favorable ones for arbitrage.
Run this independently to help decide which exchanges to use before starting the bot.
"""

import sys
import argparse
from colorama import Fore, Style, init
from exchange_config import (
    exchanges, get_time, ex, ccxt
)

# Initialize colorama
init()

def main():
    """Main entry point for fee checking utility."""
    parser = argparse.ArgumentParser(
        description='Check fees for all configured exchanges',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 check_exchange_fees.py
  python3 check_exchange_fees.py --pair ETH/USDT
  python3 check_exchange_fees.py --threshold 0.002
  python3 check_exchange_fees.py --pair BTC/USDT --threshold 0.001
        """
    )
    
    parser.add_argument(
        '--pair',
        type=str,
        default='BTC/USDT',
        help='Trading pair to check fees for (default: BTC/USDT)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0015,
        help='Maximum fee rate to be considered favorable (default: 0.0015 = 0.15%%)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"{'Exchange Fee Checker - Standalone Utility':^70}")
    print(f"{'='*70}\n")
    print(f"{get_time()}Checking fees for {len(exchanges)} exchanges")
    print(f"{get_time()}Trading pair: {args.pair}")
    print(f"{get_time()}Favorable threshold: {args.threshold*100:.2f}%%\n")
    
    try:
        # Check fees for all exchanges with real-time logging
        exchange_fees = {}
        favorable_exchanges = []
        
        from exchange_config import ex, ccxt
        
        print(f"{get_time()}Checking exchanges...\n")
        
        for idx, exchange_name in enumerate(sorted(exchanges.keys()), 1):
            print(f"{get_time()}Checking [{idx}/{len(exchanges)}] {exchange_name}...", end=' ', flush=True)
            
            try:
                # Try to initialize exchange if not already initialized
                if exchange_name not in ex:
                    try:
                        default_config = {
                            'enableRateLimit': True,
                            'options': {'defaultType': 'spot'}
                        }
                        ex[exchange_name] = getattr(ccxt, exchange_name)(default_config)
                    except (AttributeError, Exception) as e:
                        exchange_fees[exchange_name] = {
                            'total_fee': None,
                            'status': 'not_supported',
                            'error': str(e)
                        }
                        print(f"❌ Not supported ({str(e)[:30]})")
                        continue
                
                # Try to get fees
                try:
                    markets = ex[exchange_name].load_markets()
                    pair_info = markets.get(args.pair, {})
                    
                    # Fallback to BTC/USDT if pair not found
                    if not pair_info:
                        pair_info = markets.get('BTC/USDT', {})
                    
                    # Extract fee information
                    fee_side = pair_info.get('feeSide')
                    taker_fee = pair_info.get('taker', 0)
                    
                    # If taker fee not available, try exchange defaults
                    if taker_fee == 0:
                        try:
                            exchange_instance = ex[exchange_name]
                            if hasattr(exchange_instance, 'fees') and 'trading' in exchange_instance.fees:
                                taker_fee = exchange_instance.fees['trading'].get('taker', 0.001)
                            else:
                                taker_fee = 0.001  # Default 0.1%
                        except:
                            taker_fee = 0.001  # Default 0.1%
                    
                    total_fee_rate = taker_fee
                    fee_pct = total_fee_rate * 100
                    
                    exchange_fees[exchange_name] = {
                        'total_fee': total_fee_rate,
                        'fee_side': fee_side or 'quote',
                        'status': 'success'
                    }
                    
                    # Check if favorable
                    is_favorable = total_fee_rate <= args.threshold
                    marker = "✓" if is_favorable else " "
                    
                    print(f"{marker} {fee_pct:.3f}% ({fee_side or 'quote'})")
                    
                    if is_favorable:
                        favorable_exchanges.append({
                            'name': exchange_name,
                            'total_fee': total_fee_rate,
                            'fee_pct': fee_pct
                        })
                        
                except Exception as e:
                    exchange_fees[exchange_name] = {
                        'total_fee': None,
                        'status': 'error',
                        'error': str(e)[:100]
                    }
                    print(f"❌ Error: {str(e)[:50]}")
                    
            except Exception as e:
                exchange_fees[exchange_name] = {
                    'total_fee': None,
                    'status': 'error',
                    'error': str(e)[:100]
                }
                print(f"❌ Error: {str(e)[:50]}")
        
        print()  # Empty line after all checks
        
        # Sort favorable exchanges by fee (lowest first)
        favorable_exchanges.sort(key=lambda x: x['total_fee'])
        
        # Create results dict in same format as check_all_exchange_fees
        fee_results = {
            'all_fees': exchange_fees,
            'favorable': favorable_exchanges,
            'favorable_threshold': args.threshold
        }
        
        # Print all exchanges with their fees (sorted by fee, lowest first)
        successful_fees = [(name, info) for name, info in exchange_fees.items() if info.get('status') == 'success']
        successful_fees.sort(key=lambda x: x[1].get('total_fee', float('inf')))
        
        print(f"\n{get_time()}{'='*70}")
        print(f"{get_time()}=== All Exchanges Fee Summary ===")
        for exchange_name, fee_info in successful_fees:
            fee_pct = fee_info.get('total_fee', 0) * 100
            fee_side = fee_info.get('fee_side', 'quote')
            is_favorable = fee_info.get('total_fee', float('inf')) <= args.threshold
            marker = "✓" if is_favorable else " "
            print(f"{get_time()}{marker} {exchange_name:20s} - {fee_pct:6.3f}% ({fee_side})")
        
        # Show exchanges that failed or aren't supported
        failed_exchanges = [(name, info) for name, info in exchange_fees.items() if info.get('status') != 'success']
        if failed_exchanges:
            print(f"{get_time()}--- Exchanges Not Available ---")
            for exchange_name, fee_info in sorted(failed_exchanges):
                status = fee_info.get('status', 'unknown')
                error = fee_info.get('error', 'N/A')
                print(f"{get_time()}  {exchange_name:20s} - {status} ({error[:50]})")
        
        print(f"{get_time()}{'='*70}\n")
        
        # Print summary
        successful_count = sum(1 for f in exchange_fees.values() if f.get('status') == 'success')
        favorable_count = len(favorable_exchanges)
        
        print(f"{get_time()}Summary:")
        print(f"{get_time()}  Total exchanges checked: {len(exchanges)}")
        print(f"{get_time()}  Successfully retrieved: {successful_count}")
        print(f"{get_time()}  Favorable (<= {args.threshold*100:.2f}%%): {favorable_count}\n")
        
        if favorable_count > 0:
            print(f"{get_time()}{Fore.CYAN}💡 Top Favorable Exchanges for Arbitrage:{Style.RESET_ALL}")
            for idx, exc in enumerate(favorable_exchanges[:10], 1):  # Show top 10
                print(f"{get_time()}  #{idx}: {exc['name']} - {exc['fee_pct']:.3f}% total fee")
            print()
            print(f"{get_time()}{Fore.CYAN}💡 Example command to use favorable exchanges:{Style.RESET_ALL}")
            favorable_names = [exc['name'] for exc in favorable_exchanges[:5]]
            print(f"{get_time()}  python3 main.py fake-money 5000 {args.pair} {','.join(favorable_names)}\n")
        
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

