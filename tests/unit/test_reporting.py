import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from reporting import RecommendationEngine, Visualizer

def test_recommendation_engine_buy():
    engine = RecommendationEngine()
    tech_res = {'summary': 'Uptrend, Bullish MACD, Oversold'} # +3
    fund_res = {'PE_Ratio': 10, 'Return_on_Equity': 0.20, 'Debt_to_Equity': 0.5, 'summary': ''} # +3
    sent_res = {'overall_sentiment': 'Bullish', 'summary': ''} # +1
    
    # Total score should be 7
    report = engine.generate_recommendation("AAPL", tech_res, fund_res, sent_res)
    assert "Total Model Score: 7" in report
    assert "Final Verdict: BUY" in report

def test_recommendation_engine_sell():
    engine = RecommendationEngine()
    tech_res = {'summary': 'Downtrend, Bearish MACD, Overbought'} # -3
    fund_res = {'PE_Ratio': 50, 'Return_on_Equity': 0.02, 'Debt_to_Equity': 3.0, 'summary': ''} # -3
    sent_res = {'overall_sentiment': 'Bearish', 'summary': ''} # -1
    
    # Total score should be -7
    report = engine.generate_recommendation("AAPL", tech_res, fund_res, sent_res)
    assert "Total Model Score: -7" in report
    assert "Final Verdict: SELL" in report

@patch('reporting.make_subplots')
def test_visualizer(mock_make_subplots, mock_history_df):
    mock_fig = MagicMock()
    mock_make_subplots.return_value = mock_fig
    
    vis = Visualizer()
    vis.plot_technical_analysis("AAPL", mock_history_df, show=False)
    
    mock_make_subplots.assert_called_once()
    assert mock_fig.add_trace.call_count > 0


# ============================================================
# Visualizer.plot_efficient_frontier Tests
# ============================================================

import numpy as np

@pytest.fixture
def mock_frontier_data():
    """Mock EfficientFrontierAnalyzer output for visualization tests."""
    return {
        'frontier_volatilities': np.linspace(0.10, 0.30, 50),
        'frontier_returns': np.linspace(0.05, 0.25, 50),
        'max_sharpe': {
            'weights': {'AAPL': 0.5, 'MSFT': 0.3, 'GOOGL': 0.2},
            'expected_return': 0.20,
            'volatility': 0.18,
            'sharpe_ratio': 0.861,
        },
        'min_variance': {
            'weights': {'AAPL': 0.3, 'MSFT': 0.4, 'GOOGL': 0.3},
            'expected_return': 0.10,
            'volatility': 0.12,
            'sharpe_ratio': 0.458,
        },
        'individual_stats': {
            'AAPL': {'annual_return': 0.15, 'annual_volatility': 0.20},
            'MSFT': {'annual_return': 0.18, 'annual_volatility': 0.16},
            'GOOGL': {'annual_return': 0.12, 'annual_volatility': 0.22},
        },
    }


def test_plot_frontier_returns_figure(mock_frontier_data):
    """VIS-U01: Returns a go.Figure object."""
    import plotly.graph_objects as go
    vis = Visualizer()
    fig = vis.plot_efficient_frontier(mock_frontier_data, show=False)
    assert isinstance(fig, go.Figure)


def test_plot_frontier_has_traces(mock_frontier_data):
    """VIS-U02: Figure has traces for frontier, max sharpe, min var, and 3 assets."""
    vis = Visualizer()
    fig = vis.plot_efficient_frontier(mock_frontier_data, show=False)
    # 1 frontier + 1 max sharpe + 1 min var + 3 individual assets = 6
    assert len(fig.data) == 6


def test_plot_frontier_with_current_portfolio(mock_frontier_data):
    """VIS-U03: Current portfolio dot adds one more trace."""
    vis = Visualizer()
    current = {'expected_return': 0.13, 'volatility': 0.19, 'sharpe_ratio': 0.447}
    fig = vis.plot_efficient_frontier(mock_frontier_data, current_portfolio=current, show=False)
    # 6 standard traces + 1 current portfolio = 7
    assert len(fig.data) == 7


def test_plot_frontier_empty_data():
    """VIS-U04: Empty/error data returns None."""
    vis = Visualizer()
    assert vis.plot_efficient_frontier({}, show=False) is None
    assert vis.plot_efficient_frontier({'error': 'test'}, show=False) is None

