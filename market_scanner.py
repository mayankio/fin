import argparse
from typing import List

from data_provider import YFinanceProvider
from analyzers import TechnicalAnalyzer, FundamentalAnalyzer
from reporting import Visualizer, RecommendationEngine

class MarketScanner:
    """
    Scans a list of tickers, runs technical & fundamental analysis,
    and outputs the top recommended stocks with visualizations.
    """
    def __init__(self):
        self.provider = YFinanceProvider()
        self.tech_analyzer = TechnicalAnalyzer()
        self.fund_analyzer = FundamentalAnalyzer()
        self.visualizer = Visualizer()
        self.rec_engine = RecommendationEngine()

    def run(self, tickers: List[str]):
        print(f"--- Starting Market Scanner for: {', '.join(tickers)} ---")
        
        scored_stocks = []
        
        for ticker in tickers:
            try:
                print(f"Scanning {ticker}...")
                
                # Fetch Data
                history = self.provider.get_historical_data(ticker)
                info = self.provider.get_info(ticker)
                
                # Run Analysis
                tech_results = self.tech_analyzer.run({'history': history})
                fund_results = self.fund_analyzer.run({'info': info})
                
                # Generate Recommendation
                rec_report = self.rec_engine.generate_recommendation(ticker, tech_results, fund_results)
                
                # Parse out score for sorting (Hack for scanner)
                score_line = [line for line in rec_report.split('\n') if "Total Model Score" in line]
                score = 0
                if score_line:
                    score = int(score_line[0].split(': ')[1])
                    
                scored_stocks.append({
                    'ticker': ticker,
                    'score': score,
                    'report': rec_report,
                    'history': history
                })
            except Exception as e:
                print(f"Failed to scan {ticker}: {e}")
                
        # Sort by score descending and take top 2
        scored_stocks.sort(key=lambda x: x['score'], reverse=True)
        top_stocks = scored_stocks[:2]
        
        print("\n=== TOP 2 RECOMMENDED STOCKS ===")
        for stock in top_stocks:
            print("\n" + stock['report'])
            self.visualizer.plot_technical_analysis(stock['ticker'], stock['history'], show=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Scanner")
    parser.add_argument("--tickers", type=str, nargs='+', default=["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
                        help="List of tickers to scan")
    args = parser.parse_args()
    
    scanner = MarketScanner()
    scanner.run(tickers=args.tickers)
