import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any

class Visualizer:
    """
    Handles generating modern, interactive charts using Plotly.
    """
    
    def plot_technical_analysis(self, ticker: str, df: pd.DataFrame, show: bool = True) -> go.Figure:
        """
        Creates a dual-panel chart:
        - Top panel: Candlestick price + 50 SMA + 200 SMA
        - Bottom panel: MACD line, Signal line, Histogram
        """
        if df is None or df.empty:
            print(f"No data available to plot for {ticker}")
            return None

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, 
            vertical_spacing=0.03, 
            subplot_titles=(f'{ticker} Price & SMAs', 'MACD'),
            row_width=[0.3, 0.7] # Bottom subplot is 30% height, top is 70%
        )

        # 1. Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], high=df['High'],
            low=df['Low'], close=df['Close'], name='Price'), 
            row=1, col=1)
                      
        # SMAs
        if 'SMA_50' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='50 SMA'), row=1, col=1)
        if 'SMA_200' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1), name='200 SMA'), row=1, col=1)

        # 2. MACD
        if 'MACD' in df.columns and 'MACD_signal' in df.columns and 'MACD_hist' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], line=dict(color='black', width=1.5), name='MACD'), row=2, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['MACD_signal'], line=dict(color='red', width=1.5), name='Signal'), row=2, col=1)
            
            # Histogram colors
            colors = ['green' if val >= 0 else 'red' for val in df['MACD_hist']]
            fig.add_trace(go.Bar(x=df.index, y=df['MACD_hist'], marker_color=colors, name='MACD Hist'), row=2, col=1)

        fig.update_layout(
            title_text=f"{ticker} Technical Analysis", 
            xaxis_rangeslider_visible=False, 
            template="plotly_white",
            height=700
        )
        
        if show:
            fig.show()
            
        return fig

class RecommendationEngine:
    """
    Aggregates analyzer outputs to generate a reasoned string with a BUY/HOLD/SELL verdict.
    """
    
    def generate_recommendation(self, ticker: str, tech_results: Dict[str, Any], fund_results: Dict[str, Any]) -> str:
        score = 0
        reasoning = [f"--- Recommendation Report for {ticker} ---"]
        
        # Technical Evaluation
        tech_summary = tech_results.get('summary', '')
        if "Uptrend" in tech_summary: score += 1
        if "Downtrend" in tech_summary: score -= 1
        if "Bullish MACD" in tech_summary: score += 1
        if "Bearish MACD" in tech_summary: score -= 1
        if "Oversold" in tech_summary: score += 1
        if "Overbought" in tech_summary: score -= 1
        
        reasoning.append(f"Technical Summary: {tech_summary}")
        
        # Fundamental Evaluation
        pe = fund_results.get('PE_Ratio')
        roe = fund_results.get('Return_on_Equity')
        de = fund_results.get('Debt_to_Equity')
        
        fund_notes = []
        if pe is not None:
            if pe < 15:
                score += 1
                fund_notes.append("Attractive P/E (<15)")
            elif pe > 30:
                score -= 1
                fund_notes.append("High P/E (>30)")
        
        if roe is not None:
            if roe > 0.15:
                score += 1
                fund_notes.append("Strong ROE (>15%)")
            elif roe < 0.05:
                score -= 1
                fund_notes.append("Weak ROE (<5%)")
                
        if de is not None:
            if de < 1.0:
                score += 1
                fund_notes.append("Low Debt/Equity (<1.0)")
            elif de > 2.0:
                score -= 1
                fund_notes.append("High Debt/Equity (>2.0)")
                
        reasoning.append(f"Fundamental Summary: {fund_results.get('summary')}")
        reasoning.append(f"Fundamental Notes: {', '.join(fund_notes) if fund_notes else 'None'}")
        
        # Final Verdict Logic
        verdict = "HOLD"
        # Minimum score to buy is 2, <= -1 is sell
        if score >= 2:
            verdict = "BUY"
        elif score <= -1:
            verdict = "SELL"
            
        reasoning.append(f"Total Model Score: {score}")
        reasoning.append(f"Final Verdict: {verdict}")
        reasoning.append("-" * 40)
        
        return "\n".join(reasoning)
