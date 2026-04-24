import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from analyzers import TechnicalAnalyzer, FundamentalAnalyzer, CompetitorAnalyzer, SentimentAnalyzer

def test_technical_analyzer(mock_history_df):
    analyzer = TechnicalAnalyzer()
    res = analyzer.run({'history': mock_history_df})
    
    assert 'error' not in res
    assert 'latest_RSI' in res
    assert 'latest_SMA_50' in res
    assert 'latest_SMA_200' in res
    assert 'latest_MACD' in res
    assert 'summary' in res
    assert not res['df_with_indicators'].empty

def test_technical_analyzer_empty():
    analyzer = TechnicalAnalyzer()
    res = analyzer.run({})
    assert "error" in res

def test_fundamental_analyzer(mock_info_dict):
    analyzer = FundamentalAnalyzer()
    res = analyzer.run({'info': mock_info_dict})
    
    assert res['PE_Ratio'] == 20.5
    assert res['Return_on_Equity'] == 0.25
    assert res['Debt_to_Equity'] == 1.2
    assert "P/E: 20.50 | D/E: 1.20 | ROE: 25.00%" in res['summary']

def test_competitor_analyzer():
    analyzer = CompetitorAnalyzer()
    data = {
        'target_info': {'trailingPE': 15, 'returnOnEquity': 0.2},
        'competitors': {
            'COMP1': {'trailingPE': 20, 'returnOnEquity': 0.15},
            'COMP2': {'trailingPE': 10, 'returnOnEquity': 0.25}
        }
    }
    res = analyzer.run(data)
    
    assert res['target_PE'] == 15
    assert res['competitors']['COMP1']['PE_Ratio'] == 20
    assert res['competitors']['COMP2']['Return_on_Equity'] == 0.25

@patch('analyzers.SentimentAnalyzer.__init__', return_value=None)
def test_sentiment_analyzer(mock_init, mock_social_data):
    # Mocking init to prevent pipeline download
    analyzer = SentimentAnalyzer()
    # Manually inject mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.return_value = [
        {'label': 'positive', 'score': 0.99},
        {'label': 'negative', 'score': 0.85},
        {'label': 'positive', 'score': 0.75}
    ]
    analyzer.nlp = mock_pipeline

    res = analyzer.run({'chatter': mock_social_data})

    assert res['total_analyzed'] == 3
    assert res['bullish_ratio'] == 2/3
    assert res['bearish_ratio'] == 1/3
    assert res['overall_sentiment'] == 'Bullish'
    assert 'This stock is going to the moon!' in res['top_bullish_headlines']
