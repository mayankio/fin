import abc
from typing import Dict, Any
import pandas as pd
import numpy as np
import ta

class BaseAnalyzer(abc.ABC):
    """
    Abstract Base Class for all analyzers.
    Follows the Strategy pattern for analyzing market data.
    """
    @abc.abstractmethod
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the analysis strategy.
        
        :param data: A dictionary containing data needed for analysis. 
                     Expected keys vary by analyzer (e.g., 'history', 'info', 'competitors').
        :return: A dictionary containing the analysis results.
        """
        pass

class TechnicalAnalyzer(BaseAnalyzer):
    """
    Computes Technical Indicators: RSI, 50/200 SMA, MACD.
    """
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        df = data.get('history')
        if df is None or df.empty:
            return {"error": "No historical data provided for TechnicalAnalyzer"}
        
        df = df.copy()
        
        # RSI
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        
        # SMAs
        df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['SMA_200'] = ta.trend.SMAIndicator(close=df['Close'], window=200).sma_indicator()
        
        # MACD
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_hist'] = macd.macd_diff()
        
        # Get the latest data point for summary
        latest = df.iloc[-1]
        
        summary_signals = []
        if pd.notna(latest['RSI']):
            if latest['RSI'] < 30:
                summary_signals.append("Oversold (RSI < 30)")
            elif latest['RSI'] > 70:
                summary_signals.append("Overbought (RSI > 70)")
            else:
                summary_signals.append("Neutral RSI")

        if pd.notna(latest['SMA_50']) and pd.notna(latest['SMA_200']):
            if latest['SMA_50'] > latest['SMA_200']:
                summary_signals.append("Uptrend (50 SMA > 200 SMA)")
            else:
                summary_signals.append("Downtrend (50 SMA < 200 SMA)")
                
        if pd.notna(latest['MACD_hist']):
            if latest['MACD_hist'] > 0:
                summary_signals.append("Bullish MACD")
            else:
                summary_signals.append("Bearish MACD")
                
        return {
            'latest_RSI': latest['RSI'],
            'latest_SMA_50': latest['SMA_50'],
            'latest_SMA_200': latest['SMA_200'],
            'latest_MACD': latest['MACD'],
            'latest_MACD_hist': latest['MACD_hist'],
            'summary': ", ".join(summary_signals),
            'df_with_indicators': df
        }

class FundamentalAnalyzer(BaseAnalyzer):
    """
    Evaluates P/E, Debt/Equity, Return on Equity based on info dictionary.
    """
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        info = data.get('info', {})
        if not info:
             return {"error": "No info data provided for FundamentalAnalyzer"}
             
        pe_ratio = info.get('trailingPE') or info.get('forwardPE', None)
        debt_to_equity = info.get('debtToEquity', None)
        roe = info.get('returnOnEquity', None)
        
        # Format for readability
        roe_str = f"{roe*100:.2f}%" if roe is not None else "N/A"
        de_str = f"{debt_to_equity:.2f}" if debt_to_equity is not None else "N/A"
        pe_str = f"{pe_ratio:.2f}" if pe_ratio is not None else "N/A"
        
        return {
            'PE_Ratio': pe_ratio,
            'Debt_to_Equity': debt_to_equity,
            'Return_on_Equity': roe,
            'summary': f"P/E: {pe_str} | D/E: {de_str} | ROE: {roe_str}"
        }

class CompetitorAnalyzer(BaseAnalyzer):
    """
    Compares the target stock's fundamentals against sector peers.
    Expected data: {'target_info': {...}, 'competitors': {'CMP1': {...}, 'CMP2': {...}}}
    """
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        target_info = data.get('target_info', {})
        competitors = data.get('competitors', {})
        
        if not target_info or not competitors:
             return {"error": "Missing target or competitor info for CompetitorAnalyzer"}
             
        target_pe = target_info.get('trailingPE')
        
        comparison_results = {}
        for comp_ticker, comp_info in competitors.items():
            comp_pe = comp_info.get('trailingPE')
            comp_roe = comp_info.get('returnOnEquity')
            comparison_results[comp_ticker] = {
                'PE_Ratio': comp_pe,
                'Return_on_Equity': comp_roe
            }
            
        return {
            'target_PE': target_pe,
            'target_ROE': target_info.get('returnOnEquity'),
            'competitors': comparison_results
        }

class PortfolioAnalyzer(BaseAnalyzer):
    """
    Calculates portfolio risk, sector allocation, and beta.
    Expected data: {'portfolio': {'AAPL': {'weight': 0.5, 'info': {...}}, ...}}
    """
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        portfolio = data.get('portfolio', {})
        if not portfolio:
             return {"error": "No portfolio data provided for PortfolioAnalyzer"}
             
        total_beta = 0.0
        sectors_allocation = {}
        
        for ticker, p_data in portfolio.items():
            weight = p_data.get('weight', 0.0)
            info = p_data.get('info', {})
            
            beta = info.get('beta', 1.0)
            sector = info.get('sector', 'Unknown Sector')
            
            if beta is not None:
                total_beta += beta * weight
                
            sectors_allocation[sector] = sectors_allocation.get(sector, 0.0) + weight
            
        # Identify risk if over 40% in one sector
        risks = []
        for sector, alloc in sectors_allocation.items():
            if alloc > 0.4:
                risks.append(f"Over-concentration risk in {sector} ({alloc*100:.1f}%)")
                
        return {
            'portfolio_beta': total_beta,
            'sector_allocation': sectors_allocation,
            'risks': risks
        }

class EfficientFrontierAnalyzer(BaseAnalyzer):
    """
    Computes the Efficient Frontier, Max Sharpe, and Min Variance portfolios
    using Modern Portfolio Theory.

    Expected data: {
        'prices': pd.DataFrame (columns=tickers, rows=dates, values=Close prices),
        'risk_free_rate': float (default 0.045),
        'weight_bounds': tuple (default (0.0, 1.0)),
        'num_frontier_points': int (default 100)
    }
    """
    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        prices = data.get('prices')
        if prices is None or prices.empty:
            return {"error": "No price data provided for EfficientFrontierAnalyzer"}

        risk_free_rate = data.get('risk_free_rate', 0.045)
        weight_bounds = data.get('weight_bounds', (0.0, 1.0))
        num_frontier_points = data.get('num_frontier_points', 100)

        try:
            from mpt_engine import run_optimization
            result = run_optimization(
                prices_df=prices,
                risk_free_rate=risk_free_rate,
                weight_bounds=weight_bounds,
                num_frontier_points=num_frontier_points
            )
        except ValueError as e:
            return {"error": f"Optimization failed: {str(e)}"}

        # Convert dataclass results into plain dict for consistency with other analyzers
        def _portfolio_to_dict(pr):
            return {
                'weights': {t: float(w) for t, w in zip(pr.ticker_labels, pr.weights)},
                'expected_return': pr.expected_return,
                'volatility': pr.volatility,
                'sharpe_ratio': pr.sharpe_ratio,
            }

        return {
            'max_sharpe': _portfolio_to_dict(result.max_sharpe),
            'min_variance': _portfolio_to_dict(result.min_variance),
            'frontier_returns': result.frontier_returns,
            'frontier_volatilities': result.frontier_volatilities,
            'individual_stats': result.individual_stats,
            'correlation_matrix': result.correlation_matrix,
            'covariance_matrix': result.covariance_matrix,
        }


class SentimentAnalyzer(BaseAnalyzer):
    """
    Evaluates market sentiment using HuggingFace's FinBERT model.
    Expected data: {'chatter': [{'text': '...', 'source': '...', 'timestamp': '...'}, ...]}
    """
    def __init__(self):
        try:
            from transformers import pipeline
            # FinBERT is specifically trained on financial text
            self.nlp = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except ImportError:
            print("Warning: transformers library not found. SentimentAnalyzer requires it.")
            self.nlp = None

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        chatter = data.get('chatter', [])
        if not chatter:
             return {"error": "No chatter data provided for SentimentAnalyzer"}
             
        if not self.nlp:
             return {"error": "NLP model not loaded. Is transformers installed?"}

        texts = [item['text'] for item in chatter if item.get('text')]
        if not texts:
            return {"error": "No valid text found in chatter."}

        try:
            # Batch process the texts
            # Some texts might be too long, we truncate them to 512 tokens implicitly by the pipeline
            results = self.nlp(texts, truncation=True, max_length=512)
        except Exception as e:
            return {"error": f"Error running FinBERT: {str(e)}"}

        bullish_count = 0
        bearish_count = 0
        neutral_count = 0

        # ProsusAI/finbert labels: 'positive', 'negative', 'neutral'
        scored_texts = []
        for text, res in zip(texts, results):
            label = res['label']
            score = res['score']
            
            if label == 'positive':
                bullish_count += 1
            elif label == 'negative':
                bearish_count += 1
            else:
                neutral_count += 1
                
            scored_texts.append({'text': text, 'label': label, 'score': score})

        total = len(texts)
        bullish_ratio = bullish_count / total
        bearish_ratio = bearish_count / total
        
        # Determine overall sentiment
        # We can use a simple threshold
        if bullish_ratio > bearish_ratio + 0.1:
            overall = "Bullish"
        elif bearish_ratio > bullish_ratio + 0.1:
            overall = "Bearish"
        else:
            overall = "Neutral"
            
        # Sentiment score from -1.0 to 1.0
        sentiment_score = bullish_ratio - bearish_ratio

        # Sort to find most impactful texts (highest confidence positive/negative)
        top_bullish = sorted([x for x in scored_texts if x['label'] == 'positive'], key=lambda x: x['score'], reverse=True)[:3]
        top_bearish = sorted([x for x in scored_texts if x['label'] == 'negative'], key=lambda x: x['score'], reverse=True)[:3]

        return {
            'overall_sentiment': overall,
            'sentiment_score': sentiment_score,
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'total_analyzed': total,
            'top_bullish_headlines': [x['text'] for x in top_bullish],
            'top_bearish_headlines': [x['text'] for x in top_bearish],
            'summary': f"Sentiment: {overall} (Score: {sentiment_score:.2f}) | Bullish: {bullish_ratio:.1%} | Bearish: {bearish_ratio:.1%}"
        }

