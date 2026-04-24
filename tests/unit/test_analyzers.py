import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from analyzers import TechnicalAnalyzer, FundamentalAnalyzer, CompetitorAnalyzer, SentimentAnalyzer, EfficientFrontierAnalyzer

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


# ============================================================
# EfficientFrontierAnalyzer Tests
# ============================================================

@pytest.fixture
def mock_prices_df():
    """Synthetic price DataFrame for EF analyzer tests."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    return pd.DataFrame({
        'AAPL': 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, 252))),
        'MSFT': 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.012, 252))),
        'GOOGL': 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.018, 252))),
    }, index=dates)


def test_ef_analyzer_returns_expected_keys(mock_prices_df):
    """EFA-U01: Output dict has all required keys."""
    analyzer = EfficientFrontierAnalyzer()
    res = analyzer.run({'prices': mock_prices_df})
    assert 'error' not in res
    assert 'max_sharpe' in res
    assert 'min_variance' in res
    assert 'frontier_returns' in res
    assert 'frontier_volatilities' in res
    assert 'individual_stats' in res
    assert 'correlation_matrix' in res
    assert 'weights' in res['max_sharpe']
    assert 'sharpe_ratio' in res['max_sharpe']


def test_ef_analyzer_empty_prices():
    """EFA-U02: Empty DataFrame → error dict."""
    analyzer = EfficientFrontierAnalyzer()
    res = analyzer.run({'prices': pd.DataFrame()})
    assert 'error' in res


def test_ef_analyzer_single_stock():
    """EFA-U03: 1 stock → weights = {ticker: 1.0}."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    prices = pd.DataFrame({
        'SOLO': 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, 252)))
    }, index=dates)
    analyzer = EfficientFrontierAnalyzer()
    res = analyzer.run({'prices': prices})
    assert 'error' not in res
    assert res['max_sharpe']['weights']['SOLO'] == 1.0


def test_ef_analyzer_custom_risk_free_rate(mock_prices_df):
    """EFA-U04: Custom rf changes Sharpe values."""
    analyzer = EfficientFrontierAnalyzer()
    res_low = analyzer.run({'prices': mock_prices_df, 'risk_free_rate': 0.01})
    res_high = analyzer.run({'prices': mock_prices_df, 'risk_free_rate': 0.10})
    assert res_low['max_sharpe']['sharpe_ratio'] > res_high['max_sharpe']['sharpe_ratio']


def test_ef_analyzer_custom_bounds(mock_prices_df):
    """EFA-U05: Bounds (0.1, 0.5) → all weights in range."""
    analyzer = EfficientFrontierAnalyzer()
    res = analyzer.run({'prices': mock_prices_df, 'weight_bounds': (0.1, 0.5)})
    for ticker, weight in res['max_sharpe']['weights'].items():
        assert weight >= 0.1 - 1e-6
        assert weight <= 0.5 + 1e-6
