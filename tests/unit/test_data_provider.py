import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from data_provider import YFinanceProvider

@pytest.fixture
def provider():
    return YFinanceProvider()

@patch('yfinance.Ticker')
def test_get_historical_data_success(mock_ticker, provider, mock_history_df):
    mock_instance = MagicMock()
    mock_instance.history.return_value = mock_history_df
    mock_ticker.return_value = mock_instance

    df = provider.get_historical_data("AAPL")
    assert not df.empty
    assert len(df) == 250
    mock_instance.history.assert_called_once()

@patch('yfinance.Ticker')
def test_get_historical_data_empty(mock_ticker, provider):
    mock_instance = MagicMock()
    mock_instance.history.return_value = pd.DataFrame()
    mock_ticker.return_value = mock_instance

    with pytest.raises(Exception, match="Failed to fetch historical data for INVALID: No historical data found"):
        provider.get_historical_data("INVALID")

@patch('yfinance.Ticker')
def test_get_info_success(mock_ticker, provider, mock_info_dict):
    mock_instance = MagicMock()
    mock_instance.info = mock_info_dict
    mock_ticker.return_value = mock_instance

    info = provider.get_info("AAPL")
    assert info['trailingPE'] == 20.5

@patch('yfinance.Ticker')
def test_get_info_empty(mock_ticker, provider):
    mock_instance = MagicMock()
    mock_instance.info = {}
    mock_ticker.return_value = mock_instance

    with pytest.raises(Exception, match="Failed to fetch info"):
        provider.get_info("INVALID")
