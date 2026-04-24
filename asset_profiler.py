import argparse
from typing import List

from data_provider import YFinanceProvider
from social_provider import BrowserScraperProvider
from analyzers import TechnicalAnalyzer, FundamentalAnalyzer, CompetitorAnalyzer, SentimentAnalyzer
from reporting import Visualizer, RecommendationEngine

class AssetProfiler:
    """
    Runs TA, FA, and Competitor analysis for a single stock.
    """
    def __init__(self):
        self.provider = YFinanceProvider()
        self.social_provider = BrowserScraperProvider()
        self.tech_analyzer = TechnicalAnalyzer()
        self.fund_analyzer = FundamentalAnalyzer()
        self.comp_analyzer = CompetitorAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.visualizer = Visualizer()
        self.rec_engine = RecommendationEngine()

    def run(self, ticker: str, competitors: List[str], lookback_days: int = 7):
        print(f"--- Starting Deep Dive for: {ticker} ---")
        try:
            # Data Layer
            history = self.provider.get_historical_data(ticker)
            info = self.provider.get_info(ticker)
            
            # Fetch Social Data
            print(f"Fetching social data for {ticker} (lookback: {lookback_days} days)...")
            social_data = self.social_provider.get_social_chatter(ticker, lookback_days)
            
            comp_data = {}
            for comp in competitors:
                print(f"Fetching competitor data for {comp}...")
                try:
                    comp_data[comp] = self.provider.get_info(comp)
                except Exception as e:
                    print(f"Warning: Could not fetch {comp} - {e}")
            
            # Analysis Layer
            tech_results = self.tech_analyzer.run({'history': history})
            fund_results = self.fund_analyzer.run({'info': info})
            comp_results = self.comp_analyzer.run({'target_info': info, 'competitors': comp_data})
            sentiment_results = self.sentiment_analyzer.run({'chatter': social_data})
            
            # Presentation Layer
            print("\n=== DEEP DIVE REPORT ===")
            rec_report = self.rec_engine.generate_recommendation(ticker, tech_results, fund_results, sentiment_results)
            print(rec_report)
            
            print(f"\n--- Competitor Comparison ---")
            print(f"Target ({ticker}) P/E: {comp_results.get('target_PE')}, ROE: {comp_results.get('target_ROE')}")
            for comp, metrics in comp_results.get('competitors', {}).items():
                print(f"{comp} -> P/E: {metrics.get('PE_Ratio')}, ROE: {metrics.get('Return_on_Equity')}")
                
            self.visualizer.plot_technical_analysis(ticker, tech_results['df_with_indicators'], show=True)
            
        except Exception as e:
            print(f"Deep Dive failed for {ticker}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asset Profiler")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Target ticker")
    parser.add_argument("--competitors", type=str, nargs='+', default=["AMD", "INTC", "QCOM"], help="Competitor tickers")
    parser.add_argument("--lookback-days", type=int, default=7, help="Days to look back for social chatter")
    parser.add_argument("--authenticate", action="store_true", help="Launch browser to manually log into Reddit/X")
    args = parser.parse_args()
    
    profiler = AssetProfiler()
    if args.authenticate:
        profiler.social_provider.authenticate()
        
    profiler.run(ticker=args.ticker, competitors=args.competitors, lookback_days=args.lookback_days)
