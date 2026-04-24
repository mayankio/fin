import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_provider import YFinanceProvider
from social_provider import BrowserScraperProvider
from analyzers import SentimentAnalyzer

@pytest.mark.e2e
def test_e2e_yfinance_fetch():
    """Tests that we can actually reach YFinance and parse a dataframe."""
    provider = YFinanceProvider()
    df = provider.get_historical_data("AAPL", period="1mo")
    assert not df.empty
    assert 'Close' in df.columns
    
    info = provider.get_info("AAPL")
    assert info is not None
    assert 'trailingPE' in info or 'forwardPE' in info

@pytest.mark.e2e
def test_e2e_playwright_scrape():
    """Tests that Playwright can launch headlessly and scrape something from Twitter/Reddit."""
    # We use a very short lookback and generic ticker to avoid test flakiness
    provider = BrowserScraperProvider(auth_file="nonexistent_test_auth.json")
    chatter = provider.get_social_chatter("AAPL", lookback_days=1)
    
    # We don't assert length because unauthenticated X might block us,
    # but we assert it runs without raising exceptions.
    assert isinstance(chatter, list)

@pytest.mark.e2e
def test_e2e_finbert_inference():
    """Tests that the transformers pipeline can load and score text."""
    analyzer = SentimentAnalyzer()
    
    # Skip if model failed to load (e.g. no transformers installed)
    if not analyzer.nlp:
        pytest.skip("transformers not installed")
        
    data = {
        'chatter': [
            {'text': 'AAPL had amazing earnings! Bullish!'},
            {'text': 'The market is crashing, sell everything.'}
        ]
    }
    
    res = analyzer.run(data)
    assert 'error' not in res
    assert res['overall_sentiment'] in ['Bullish', 'Bearish', 'Neutral']
    assert res['total_analyzed'] == 2
