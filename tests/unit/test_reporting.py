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
