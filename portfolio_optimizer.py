import argparse
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data_provider import YFinanceProvider
from analyzers import PortfolioAnalyzer, EfficientFrontierAnalyzer
from reporting import Visualizer

class PortfolioOptimizer:
    """
    Orchestrator for portfolio optimization using Modern Portfolio Theory.

    Supports two modes:
    1. Optimization mode (default): Given tickers, compute optimal allocations.
    2. Review mode: Given tickers + current weights, show existing risk analysis
       alongside optimal suggestions.
    """
    def __init__(self):
        self.provider = YFinanceProvider()
        self.port_analyzer = PortfolioAnalyzer()
        self.frontier_analyzer = EfficientFrontierAnalyzer()
        self.visualizer = Visualizer()

    def _fetch_aligned_prices(self, tickers: List[str], period: str = "2y") -> pd.DataFrame:
        """
        Fetch historical close prices for all tickers and align them
        by date using inner join (only dates where all tickers traded).
        """
        price_series = {}
        for ticker in tickers:
            try:
                hist = self.provider.get_historical_data(ticker, period=period)
                price_series[ticker] = hist['Close']
            except Exception as e:
                print(f"⚠️  Failed to fetch history for {ticker}: {e}")

        if len(price_series) < 2:
            if len(price_series) == 1:
                # Single asset — still allow stats reporting
                return pd.DataFrame(price_series)
            raise ValueError("Need at least 1 ticker with valid price data.")

        # Inner join: only keep dates present for ALL tickers
        prices_df = pd.DataFrame(price_series).dropna()
        return prices_df

    def run(self, tickers: List[str],
            current_weights: Optional[List[float]] = None,
            risk_free_rate: float = 0.045,
            weight_bounds: Tuple[float, float] = (0.0, 1.0)):
        """
        Execute the portfolio optimization pipeline.

        :param tickers: List of stock tickers to optimize.
        :param current_weights: Optional list of current portfolio weights (same order as tickers).
        :param risk_free_rate: Annualized risk-free rate for Sharpe calculation.
        :param weight_bounds: (min, max) weight per asset. Default: long-only, no constraints.
        """
        print(f"--- Portfolio Optimizer: {', '.join(tickers)} ---")
        print(f"    Risk-free rate: {risk_free_rate:.2%}")
        print(f"    Weight bounds: [{weight_bounds[0]:.0%}, {weight_bounds[1]:.0%}]")

        # --- Step 1: Existing risk analysis (if current weights provided) ---
        current_portfolio_perf = None
        if current_weights is not None:
            if len(current_weights) != len(tickers):
                print(f"⚠️  Weight count ({len(current_weights)}) doesn't match ticker count ({len(tickers)}). Skipping current portfolio analysis.")
                current_weights = None
            elif abs(sum(current_weights) - 1.0) > 0.01:
                print(f"⚠️  Weights sum to {sum(current_weights):.4f}, not 1.0. Skipping current portfolio analysis.")
                current_weights = None

        if current_weights is not None:
            print("\n--- Current Portfolio Risk Analysis ---")
            portfolio_data = {}
            for ticker, weight in zip(tickers, current_weights):
                try:
                    info = self.provider.get_info(ticker)
                    portfolio_data[ticker] = {'weight': weight, 'info': info}
                except Exception as e:
                    print(f"  Failed to fetch info for {ticker}: {e}")

            if portfolio_data:
                risk_results = self.port_analyzer.run({'portfolio': portfolio_data})
                print(f"  Portfolio Beta: {risk_results.get('portfolio_beta', 0.0):.2f}")
                print("  Sector Allocation:")
                for sector, alloc in risk_results.get('sector_allocation', {}).items():
                    print(f"    - {sector}: {alloc * 100:.1f}%")
                risks = risk_results.get('risks', [])
                if risks:
                    print("  ⚠️ RISKS:")
                    for risk in risks:
                        print(f"    - {risk}")

        # --- Step 2: Fetch aligned price data ---
        print("\nFetching historical price data...")
        try:
            prices_df = self._fetch_aligned_prices(tickers)
        except ValueError as e:
            print(f"❌ {e}")
            return

        available_tickers = list(prices_df.columns)
        if len(available_tickers) < len(tickers):
            skipped = set(tickers) - set(available_tickers)
            print(f"  ⚠️  Skipped tickers (no data): {', '.join(skipped)}")
        print(f"  Using {len(available_tickers)} tickers with {len(prices_df)} aligned trading days.")

        # --- Step 3: Run optimization ---
        print("\nRunning Modern Portfolio Theory optimization...")
        ef_results = self.frontier_analyzer.run({
            'prices': prices_df,
            'risk_free_rate': risk_free_rate,
            'weight_bounds': weight_bounds,
        })

        if 'error' in ef_results:
            print(f"❌ Optimization failed: {ef_results['error']}")
            return

        # --- Step 4: Print results ---
        print("\n" + "=" * 60)
        print("  PORTFOLIO OPTIMIZATION REPORT")
        print("=" * 60)

        # Individual asset stats
        print("\n📊 Individual Asset Statistics:")
        print(f"  {'Ticker':<8} {'Ann. Return':>12} {'Ann. Volatility':>16}")
        print(f"  {'------':<8} {'-----------':>12} {'---------------':>16}")
        for ticker, stats in ef_results['individual_stats'].items():
            print(f"  {ticker:<8} {stats['annual_return']:>11.2%} {stats['annual_volatility']:>15.2%}")

        # Correlation matrix
        print("\n📈 Correlation Matrix:")
        corr = ef_results['correlation_matrix']
        header = "  " + " " * 8 + "".join(f"{t:>8}" for t in corr.columns)
        print(header)
        for idx in corr.index:
            row = f"  {idx:<8}" + "".join(f"{corr.loc[idx, col]:>8.3f}" for col in corr.columns)
            print(row)

        # Max Sharpe portfolio
        ms = ef_results['max_sharpe']
        print(f"\n🏆 MAX SHARPE RATIO PORTFOLIO (Optimal):")
        print(f"  Expected Annual Return: {ms['expected_return']:.2%}")
        print(f"  Annual Volatility:      {ms['volatility']:.2%}")
        print(f"  Sharpe Ratio:           {ms['sharpe_ratio']:.4f}")
        print(f"  Recommended Weights:")
        for ticker, weight in ms['weights'].items():
            bar = "█" * int(weight * 40)
            print(f"    {ticker:<8} {weight:>6.1%}  {bar}")

        # Min Variance portfolio
        mv = ef_results['min_variance']
        print(f"\n🛡️  MINIMUM VARIANCE PORTFOLIO (Conservative):")
        print(f"  Expected Annual Return: {mv['expected_return']:.2%}")
        print(f"  Annual Volatility:      {mv['volatility']:.2%}")
        print(f"  Sharpe Ratio:           {mv['sharpe_ratio']:.4f}")
        print(f"  Recommended Weights:")
        for ticker, weight in mv['weights'].items():
            bar = "█" * int(weight * 40)
            print(f"    {ticker:<8} {weight:>6.1%}  {bar}")

        # Current portfolio comparison
        if current_weights is not None:
            from mpt_engine import portfolio_performance, compute_annual_returns, compute_covariance_matrix
            import numpy as np
            ann_returns = compute_annual_returns(prices_df)
            cov_matrix = compute_covariance_matrix(prices_df)
            # Re-align weights to available tickers
            aligned_weights = []
            for t in available_tickers:
                idx = tickers.index(t) if t in tickers else -1
                if idx >= 0 and idx < len(current_weights):
                    aligned_weights.append(current_weights[idx])
                else:
                    aligned_weights.append(0.0)
            w_arr = np.array(aligned_weights)
            if abs(w_arr.sum() - 1.0) < 0.01:
                curr_ret, curr_vol, curr_sharpe = portfolio_performance(
                    w_arr, ann_returns.values, cov_matrix.values, risk_free_rate
                )
                current_portfolio_perf = {
                    'expected_return': curr_ret,
                    'volatility': curr_vol,
                    'sharpe_ratio': curr_sharpe,
                }
                print(f"\n📍 YOUR CURRENT PORTFOLIO:")
                print(f"  Expected Annual Return: {curr_ret:.2%}")
                print(f"  Annual Volatility:      {curr_vol:.2%}")
                print(f"  Sharpe Ratio:           {curr_sharpe:.4f}")

                sharpe_improvement = ms['sharpe_ratio'] - curr_sharpe
                if sharpe_improvement > 0:
                    print(f"\n  💡 Switching to the Max Sharpe portfolio would improve your")
                    print(f"     Sharpe Ratio by {sharpe_improvement:.4f}")

        print("\n" + "=" * 60)

        # --- Step 5: Render chart ---
        self.visualizer.plot_efficient_frontier(
            frontier_data=ef_results,
            current_portfolio=current_portfolio_perf,
            show=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Portfolio Optimizer — Modern Portfolio Theory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python portfolio_optimizer.py --tickers AAPL MSFT GOOGL AMZN
  python portfolio_optimizer.py --tickers AAPL MSFT --current-weights 0.6 0.4
  python portfolio_optimizer.py --tickers TSLA AAPL MSFT --risk-free-rate 0.04 --min-weight 0.05 --max-weight 0.50
        """
    )
    parser.add_argument("--tickers", type=str, nargs='+', required=True,
                        help="List of stock tickers to optimize")
    parser.add_argument("--current-weights", type=float, nargs='+', default=None,
                        help="Current portfolio weights (same order as tickers, must sum to 1.0)")
    parser.add_argument("--risk-free-rate", type=float, default=0.045,
                        help="Annualized risk-free rate (default: 0.045 = 4.5%%)")
    parser.add_argument("--min-weight", type=float, default=0.0,
                        help="Minimum weight per asset (default: 0.0)")
    parser.add_argument("--max-weight", type=float, default=1.0,
                        help="Maximum weight per asset (default: 1.0)")

    args = parser.parse_args()

    optimizer = PortfolioOptimizer()
    optimizer.run(
        tickers=args.tickers,
        current_weights=args.current_weights,
        risk_free_rate=args.risk_free_rate,
        weight_bounds=(args.min_weight, args.max_weight)
    )
