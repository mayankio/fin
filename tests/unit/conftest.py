import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def mock_history_df():
    """Returns a mock DataFrame mimicking yfinance historical data with sufficient length for SMA 200."""
    dates = pd.date_range(start="2023-01-01", periods=250, freq="D")
    df = pd.DataFrame({
        "Open": np.linspace(100, 150, 250),
        "High": np.linspace(105, 155, 250),
        "Low": np.linspace(95, 145, 250),
        "Close": np.linspace(102, 152, 250),
        "Volume": np.random.randint(1000, 5000, 250)
    }, index=dates)
    return df

@pytest.fixture
def mock_info_dict():
    """Returns a mock dictionary mimicking yfinance info."""
    return {
        "trailingPE": 20.5,
        "forwardPE": 18.0,
        "returnOnEquity": 0.25,
        "debtToEquity": 1.2,
        "beta": 1.1,
        "sector": "Technology"
    }

@pytest.fixture
def mock_social_data():
    """Returns mock social data extracted by the scraper."""
    return [
        {"source": "X", "text": "This stock is going to the moon!", "timestamp": "2023-10-01T12:00:00"},
        {"source": "Reddit", "text": "Earnings were terrible, I'm selling.", "timestamp": "2023-10-01T13:00:00"},
        {"source": "Reddit", "text": "Looks like a solid long-term investment.", "timestamp": "2023-10-01T14:00:00"}
    ]
