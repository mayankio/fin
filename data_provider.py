import abc
import yfinance as yf
import pandas as pd
from typing import Dict, Any, Optional

class MarketDataProvider(abc.ABC):
    """
    Abstract Base Class representing a market data provider.
    Any data source used by the application should implement these methods.
    """
    
    @abc.abstractmethod
    def get_historical_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        """Fetch historical price data."""
        pass

    @abc.abstractmethod
    def get_financials(self, ticker: str) -> pd.DataFrame:
        """Fetch company financials (income statement)."""
        pass

    @abc.abstractmethod
    def get_balance_sheet(self, ticker: str) -> pd.DataFrame:
        """Fetch company balance sheet."""
        pass

    @abc.abstractmethod
    def get_major_holders(self, ticker: str) -> pd.DataFrame:
        """Fetch major holders information."""
        pass

    @abc.abstractmethod
    def get_info(self, ticker: str) -> Dict[str, Any]:
        """Fetch generic ticket info including static fundamental ratios."""
        pass

class YFinanceProvider(MarketDataProvider):
    """
    Concrete implementation of MarketDataProvider using yfinance.
    """
    
    def get_historical_data(self, ticker: str, period: str = "2y") -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                raise ValueError(f"No historical data found for {ticker}. The ticker might be delisted or incorrect.")
            return df
        except Exception as e:
            raise Exception(f"Failed to fetch historical data for {ticker}: {str(e)}")

    def get_info(self, ticker: str) -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            if not info:
                raise ValueError(f"No info found for {ticker}")
            return info
        except Exception as e:
            raise Exception(f"Failed to fetch info for {ticker}: {str(e)}")

    def get_financials(self, ticker: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            financials = stock.financials
            return financials
        except Exception as e:
            raise Exception(f"Failed to fetch financials for {ticker}: {str(e)}")

    def get_balance_sheet(self, ticker: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            bs = stock.balance_sheet
            return bs
        except Exception as e:
            raise Exception(f"Failed to fetch balance sheet for {ticker}: {str(e)}")
            
    def get_major_holders(self, ticker: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)
            holders = stock.major_holders
            if holders is None:
                return pd.DataFrame()
            return holders
        except Exception as e:
            raise Exception(f"Failed to fetch major holders for {ticker}: {str(e)}")
