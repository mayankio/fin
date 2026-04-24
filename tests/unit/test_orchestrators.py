import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from asset_profiler import AssetProfiler
from market_scanner import MarketScanner
from portfolio_optimizer import PortfolioOptimizer

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


# ============================================================
# PortfolioOptimizer Tests
# ============================================================

def _make_mock_history(n=252):
    """Create a mock DataFrame with Close column."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        'Open': np.linspace(100, 150, n),
        'High': np.linspace(105, 155, n),
        'Low': np.linspace(95, 145, n),
        'Close': 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n))),
        'Volume': np.random.randint(1000, 5000, n)
    }, index=dates)


@patch('portfolio_optimizer.YFinanceProvider')
@patch('portfolio_optimizer.Visualizer')
@patch('portfolio_optimizer.EfficientFrontierAnalyzer')
def test_optimizer_tickers_only_mode(mock_efa, mock_vis, mock_yfinance):
    """ORC-U01: Runs with just tickers — calls EfficientFrontierAnalyzer."""
    mock_yf_instance = MagicMock()
    mock_yf_instance.get_historical_data.return_value = _make_mock_history()
    mock_yfinance.return_value = mock_yf_instance

    mock_efa_instance = MagicMock()
    mock_efa_instance.run.return_value = {
        'max_sharpe': {
            'weights': {'AAPL': 0.6, 'MSFT': 0.4},
            'expected_return': 0.15, 'volatility': 0.18, 'sharpe_ratio': 0.58
        },
        'min_variance': {
            'weights': {'AAPL': 0.4, 'MSFT': 0.6},
            'expected_return': 0.10, 'volatility': 0.12, 'sharpe_ratio': 0.46
        },
        'frontier_returns': np.linspace(0.05, 0.20, 50),
        'frontier_volatilities': np.linspace(0.10, 0.25, 50),
        'individual_stats': {
            'AAPL': {'annual_return': 0.12, 'annual_volatility': 0.20},
            'MSFT': {'annual_return': 0.15, 'annual_volatility': 0.18},
        },
        'correlation_matrix': pd.DataFrame([[1, 0.5], [0.5, 1]], columns=['AAPL', 'MSFT'], index=['AAPL', 'MSFT']),
    }
    mock_efa.return_value = mock_efa_instance

    mock_vis_instance = MagicMock()
    mock_vis.return_value = mock_vis_instance

    optimizer = PortfolioOptimizer()
    optimizer.run(tickers=["AAPL", "MSFT"])

    mock_efa_instance.run.assert_called_once()
    mock_vis_instance.plot_efficient_frontier.assert_called_once()


@patch('portfolio_optimizer.YFinanceProvider')
@patch('portfolio_optimizer.Visualizer')
@patch('portfolio_optimizer.EfficientFrontierAnalyzer')
@patch('portfolio_optimizer.PortfolioAnalyzer')
def test_optimizer_with_current_weights(mock_pa, mock_efa, mock_vis, mock_yfinance):
    """ORC-U02: With current weights — calls both analyzers."""
    mock_yf_instance = MagicMock()
    mock_yf_instance.get_historical_data.return_value = _make_mock_history()
    mock_yf_instance.get_info.return_value = {'beta': 1.1, 'sector': 'Technology'}
    mock_yfinance.return_value = mock_yf_instance

    mock_pa_instance = MagicMock()
    mock_pa_instance.run.return_value = {
        'portfolio_beta': 1.1,
        'sector_allocation': {'Technology': 1.0},
        'risks': []
    }
    mock_pa.return_value = mock_pa_instance

    mock_efa_instance = MagicMock()
    mock_efa_instance.run.return_value = {
        'max_sharpe': {
            'weights': {'AAPL': 0.6, 'MSFT': 0.4},
            'expected_return': 0.15, 'volatility': 0.18, 'sharpe_ratio': 0.58
        },
        'min_variance': {
            'weights': {'AAPL': 0.4, 'MSFT': 0.6},
            'expected_return': 0.10, 'volatility': 0.12, 'sharpe_ratio': 0.46
        },
        'frontier_returns': np.linspace(0.05, 0.20, 50),
        'frontier_volatilities': np.linspace(0.10, 0.25, 50),
        'individual_stats': {
            'AAPL': {'annual_return': 0.12, 'annual_volatility': 0.20},
            'MSFT': {'annual_return': 0.15, 'annual_volatility': 0.18},
        },
        'correlation_matrix': pd.DataFrame([[1, 0.5], [0.5, 1]], columns=['AAPL', 'MSFT'], index=['AAPL', 'MSFT']),
    }
    mock_efa.return_value = mock_efa_instance

    mock_vis_instance = MagicMock()
    mock_vis.return_value = mock_vis_instance

    optimizer = PortfolioOptimizer()
    optimizer.run(tickers=["AAPL", "MSFT"], current_weights=[0.6, 0.4])

    # Both analyzers should be called
    mock_pa_instance.run.assert_called_once()
    mock_efa_instance.run.assert_called_once()


@patch('portfolio_optimizer.YFinanceProvider')
@patch('portfolio_optimizer.Visualizer')
@patch('portfolio_optimizer.EfficientFrontierAnalyzer')
def test_optimizer_fetch_failure_graceful(mock_efa, mock_vis, mock_yfinance):
    """ORC-U03: One ticker fails → skips it, optimizes remaining."""
    mock_yf_instance = MagicMock()

    def side_effect(ticker, period="2y"):
        if ticker == "BAD":
            raise Exception("Ticker not found")
        return _make_mock_history()

    mock_yf_instance.get_historical_data.side_effect = side_effect
    mock_yfinance.return_value = mock_yf_instance

    mock_efa_instance = MagicMock()
    mock_efa_instance.run.return_value = {
        'max_sharpe': {
            'weights': {'AAPL': 0.5, 'MSFT': 0.5},
            'expected_return': 0.15, 'volatility': 0.18, 'sharpe_ratio': 0.58
        },
        'min_variance': {
            'weights': {'AAPL': 0.5, 'MSFT': 0.5},
            'expected_return': 0.10, 'volatility': 0.12, 'sharpe_ratio': 0.46
        },
        'frontier_returns': np.linspace(0.05, 0.20, 50),
        'frontier_volatilities': np.linspace(0.10, 0.25, 50),
        'individual_stats': {
            'AAPL': {'annual_return': 0.12, 'annual_volatility': 0.20},
            'MSFT': {'annual_return': 0.15, 'annual_volatility': 0.18},
        },
        'correlation_matrix': pd.DataFrame([[1, 0.5], [0.5, 1]], columns=['AAPL', 'MSFT'], index=['AAPL', 'MSFT']),
    }
    mock_efa.return_value = mock_efa_instance

    mock_vis_instance = MagicMock()
    mock_vis.return_value = mock_vis_instance

    optimizer = PortfolioOptimizer()
    # BAD ticker should be skipped, AAPL and MSFT should proceed
    optimizer.run(tickers=["AAPL", "BAD", "MSFT"])

    mock_efa_instance.run.assert_called_once()


@patch('portfolio_optimizer.YFinanceProvider')
@patch('portfolio_optimizer.Visualizer')
@patch('portfolio_optimizer.EfficientFrontierAnalyzer')
def test_optimizer_single_ticker(mock_efa, mock_vis, mock_yfinance):
    """ORC-U04: Single ticker → reports stats, degenerate optimization."""
    mock_yf_instance = MagicMock()
    mock_yf_instance.get_historical_data.return_value = _make_mock_history()
    mock_yfinance.return_value = mock_yf_instance

    mock_efa_instance = MagicMock()
    mock_efa_instance.run.return_value = {
        'max_sharpe': {
            'weights': {'AAPL': 1.0},
            'expected_return': 0.12, 'volatility': 0.20, 'sharpe_ratio': 0.375
        },
        'min_variance': {
            'weights': {'AAPL': 1.0},
            'expected_return': 0.12, 'volatility': 0.20, 'sharpe_ratio': 0.375
        },
        'frontier_returns': np.array([0.12]),
        'frontier_volatilities': np.array([0.20]),
        'individual_stats': {
            'AAPL': {'annual_return': 0.12, 'annual_volatility': 0.20},
        },
        'correlation_matrix': pd.DataFrame([[1.0]], columns=['AAPL'], index=['AAPL']),
    }
    mock_efa.return_value = mock_efa_instance

    mock_vis_instance = MagicMock()
    mock_vis.return_value = mock_vis_instance

    optimizer = PortfolioOptimizer()
    optimizer.run(tickers=["AAPL"])

    mock_efa_instance.run.assert_called_once()
