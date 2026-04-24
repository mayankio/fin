import argparse
from typing import Dict

from data_provider import YFinanceProvider
from analyzers import PortfolioAnalyzer

class PortfolioOptimizer:
    """
    Analyzes the overall portfolio, highlights risks and beta.
    Accepts dictionary of {ticker: weight (0.0 to 1.0)}
    """
    def __init__(self):
        self.provider = YFinanceProvider()
        self.port_analyzer = PortfolioAnalyzer()

    def run(self, portfolio: Dict[str, float]):
        print(f"--- Starting Portfolio Review ---")
        portfolio_data = {}
        for ticker, weight in portfolio.items():
            try:
                info = self.provider.get_info(ticker)
                portfolio_data[ticker] = {
                    'weight': weight,
                    'info': info
                }
            except Exception as e:
                print(f"Failed to fetch {ticker}: {e}")
                
        results = self.port_analyzer.run({'portfolio': portfolio_data})
        
        print("\n=== PORTFOLIO REVIEW REPORT ===")
        print(f"Total Portfolio Beta: {results.get('portfolio_beta', 0.0):.2f}")
        
        print("\nSector Allocation:")
        for sector, alloc in results.get('sector_allocation', {}).items():
            print(f"  - {sector}: {alloc * 100:.1f}%")
            
        risks = results.get('risks', [])
        if risks:
            print("\n⚠️ HIGHLIGHTED RISKS:")
            for risk in risks:
                print(f"  - {risk}")
            print("\nSuggested Action: Rebalance to ensure no single sector exceeds 40% allocation.")
        else:
            print("\nPortfolio is well diversified across sectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Portfolio Optimizer")
    # For simplicity, we just use a hardcoded default portfolio
    # Since parsing dictionaries via argparse can be messy.
    args = parser.parse_args()
    
    portfolio = {
        "TSLA": 0.45,
        "F": 0.05,
        "AAPL": 0.30,
        "MSFT": 0.20
    }
    
    optimizer = PortfolioOptimizer()
    optimizer.run(portfolio=portfolio)
