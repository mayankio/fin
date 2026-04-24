import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from asset_profiler import AssetProfiler
from market_scanner import MarketScanner

@patch('asset_profiler.YFinanceProvider')
@patch('asset_profiler.BrowserScraperProvider')
@patch('asset_profiler.Visualizer')
def test_asset_profiler_run(mock_vis, mock_social, mock_yfinance):
    # Setup mocks
    mock_yf_instance = MagicMock()
    mock_yf_instance.get_historical_data.return_value = MagicMock()
    mock_yf_instance.get_info.return_value = {'trailingPE': 15}
    mock_yfinance.return_value = mock_yf_instance

    mock_social_instance = MagicMock()
    mock_social_instance.get_social_chatter.return_value = []
    mock_social.return_value = mock_social_instance
    
    mock_vis_instance = MagicMock()
    mock_vis.return_value = mock_vis_instance
    
    # Mock analyzers
    with patch('asset_profiler.TechnicalAnalyzer') as mock_ta, \
         patch('asset_profiler.FundamentalAnalyzer') as mock_fa, \
         patch('asset_profiler.CompetitorAnalyzer') as mock_ca, \
         patch('asset_profiler.SentimentAnalyzer') as mock_sent:
             
        mock_ta.return_value.run.return_value = {'df_with_indicators': MagicMock()}
        mock_fa.return_value.run.return_value = {}
        mock_ca.return_value.run.return_value = {}
        mock_sent.return_value.run.return_value = {'overall_sentiment': 'Neutral'}
        
        profiler = AssetProfiler()
        profiler.run("AAPL", ["MSFT"])
        
        # Verify calls
        mock_yf_instance.get_historical_data.assert_called_with("AAPL")
        mock_social_instance.get_social_chatter.assert_called_with("AAPL", 7)
        mock_vis_instance.plot_technical_analysis.assert_called_once()

@patch('market_scanner.YFinanceProvider')
@patch('market_scanner.Visualizer')
def test_market_scanner_run(mock_vis, mock_yfinance):
    mock_yf_instance = MagicMock()
    mock_yf_instance.get_historical_data.return_value = MagicMock()
    mock_yf_instance.get_info.return_value = {'trailingPE': 15}
    mock_yfinance.return_value = mock_yf_instance
    
    with patch('market_scanner.TechnicalAnalyzer') as mock_ta, \
         patch('market_scanner.FundamentalAnalyzer') as mock_fa, \
         patch('market_scanner.SentimentAnalyzer') as mock_sent:
             
        mock_ta.return_value.run.return_value = {'summary': 'Uptrend'}
        mock_fa.return_value.run.return_value = {'summary': 'Good'}
        mock_sent.return_value.run.return_value = {'overall_sentiment': 'Neutral'}
        
        scanner = MarketScanner()
        # Run without sentiment
        scanner.run(["AAPL", "MSFT"])
        
        assert mock_yf_instance.get_historical_data.call_count == 2
        
        # Run with sentiment
        with patch('market_scanner.BrowserScraperProvider') as mock_social:
            mock_social_instance = MagicMock()
            mock_social_instance.get_social_chatter.return_value = []
            mock_social.return_value = mock_social_instance
            
            scanner = MarketScanner()
            scanner.run(["GOOGL"], enable_sentiment=True)
            mock_social_instance.get_social_chatter.assert_called_once()
