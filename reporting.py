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

    def plot_efficient_frontier(self, frontier_data: dict, current_portfolio: dict = None, show: bool = True) -> go.Figure:
        """
        Renders an interactive Efficient Frontier chart.

        Displays:
        - Efficient Frontier curve (volatility vs return)
        - Max Sharpe portfolio (star marker)
        - Min Variance portfolio (diamond marker)
        - Individual assets (circle markers)
        - Optional: Current portfolio position (X marker)

        :param frontier_data: Dict from EfficientFrontierAnalyzer with keys:
            frontier_volatilities, frontier_returns, max_sharpe, min_variance, individual_stats
        :param current_portfolio: Optional dict with keys: expected_return, volatility, sharpe_ratio
        :param show: Whether to display the figure.
        :return: Plotly Figure object, or None if data is invalid.
        """
        if not frontier_data or 'error' in frontier_data:
            print("No frontier data available to plot.")
            return None

        fig = go.Figure()

        # 1. Efficient Frontier curve
        fig.add_trace(go.Scatter(
            x=frontier_data['frontier_volatilities'],
            y=frontier_data['frontier_returns'],
            mode='lines',
            name='Efficient Frontier',
            line=dict(color='#636EFA', width=3),
            hovertemplate='Volatility: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
        ))

        # 2. Max Sharpe portfolio
        ms = frontier_data['max_sharpe']
        ms_weights_str = "<br>".join(f"  {t}: {w:.1%}" for t, w in ms['weights'].items())
        fig.add_trace(go.Scatter(
            x=[ms['volatility']],
            y=[ms['expected_return']],
            mode='markers',
            name=f"Max Sharpe (SR={ms['sharpe_ratio']:.3f})",
            marker=dict(color='gold', size=18, symbol='star', line=dict(color='black', width=1.5)),
            hovertemplate=(
                f"<b>Max Sharpe Portfolio</b><br>"
                f"Return: {ms['expected_return']:.2%}<br>"
                f"Volatility: {ms['volatility']:.2%}<br>"
                f"Sharpe: {ms['sharpe_ratio']:.4f}<br>"
                f"<br>Weights:<br>{ms_weights_str}<extra></extra>"
            )
        ))

        # 3. Min Variance portfolio
        mv = frontier_data['min_variance']
        mv_weights_str = "<br>".join(f"  {t}: {w:.1%}" for t, w in mv['weights'].items())
        fig.add_trace(go.Scatter(
            x=[mv['volatility']],
            y=[mv['expected_return']],
            mode='markers',
            name=f"Min Variance (SR={mv['sharpe_ratio']:.3f})",
            marker=dict(color='limegreen', size=16, symbol='diamond', line=dict(color='black', width=1.5)),
            hovertemplate=(
                f"<b>Min Variance Portfolio</b><br>"
                f"Return: {mv['expected_return']:.2%}<br>"
                f"Volatility: {mv['volatility']:.2%}<br>"
                f"Sharpe: {mv['sharpe_ratio']:.4f}<br>"
                f"<br>Weights:<br>{mv_weights_str}<extra></extra>"
            )
        ))

        # 4. Individual assets
        for ticker, stats in frontier_data.get('individual_stats', {}).items():
            fig.add_trace(go.Scatter(
                x=[stats['annual_volatility']],
                y=[stats['annual_return']],
                mode='markers+text',
                name=ticker,
                text=[ticker],
                textposition='top center',
                marker=dict(size=10, symbol='circle', line=dict(color='black', width=1)),
                hovertemplate=(
                    f"<b>{ticker}</b><br>"
                    f"Return: {stats['annual_return']:.2%}<br>"
                    f"Volatility: {stats['annual_volatility']:.2%}<extra></extra>"
                )
            ))

        # 5. Current portfolio (optional)
        if current_portfolio:
            fig.add_trace(go.Scatter(
                x=[current_portfolio['volatility']],
                y=[current_portfolio['expected_return']],
                mode='markers',
                name=f"Your Portfolio (SR={current_portfolio['sharpe_ratio']:.3f})",
                marker=dict(color='red', size=16, symbol='x', line=dict(color='black', width=1.5)),
                hovertemplate=(
                    f"<b>Current Portfolio</b><br>"
                    f"Return: {current_portfolio['expected_return']:.2%}<br>"
                    f"Volatility: {current_portfolio['volatility']:.2%}<br>"
                    f"Sharpe: {current_portfolio['sharpe_ratio']:.4f}<extra></extra>"
                )
            ))

        fig.update_layout(
            title_text="Efficient Frontier — Modern Portfolio Theory",
            xaxis_title="Annual Volatility (Risk)",
            yaxis_title="Expected Annual Return",
            template="plotly_white",
            height=650,
            width=900,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            xaxis=dict(tickformat='.1%'),
            yaxis=dict(tickformat='.1%'),
        )

        if show:
            fig.show()

        return fig


class RecommendationEngine:
    """
    Aggregates analyzer outputs to generate a reasoned string with a BUY/HOLD/SELL verdict.
    """
    
    def generate_recommendation(self, ticker: str, tech_results: Dict[str, Any], fund_results: Dict[str, Any], sentiment_results: Dict[str, Any] = None) -> str:
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

        # Sentiment Evaluation
        if sentiment_results and 'error' not in sentiment_results:
            sentiment_summary = sentiment_results.get('summary', '')
            overall_sentiment = sentiment_results.get('overall_sentiment', 'Neutral')
            
            if overall_sentiment == 'Bullish':
                score += 1
            elif overall_sentiment == 'Bearish':
                score -= 1
                
            reasoning.append(f"Sentiment Summary: {sentiment_summary}")
            
            top_bullish = sentiment_results.get('top_bullish_headlines', [])
            if top_bullish:
                reasoning.append(f"  + Top Bullish: {top_bullish[0]}")
                
            top_bearish = sentiment_results.get('top_bearish_headlines', [])
            if top_bearish:
                reasoning.append(f"  - Top Bearish: {top_bearish[0]}")
        elif sentiment_results and 'error' in sentiment_results:
            reasoning.append(f"Sentiment Analysis Error: {sentiment_results['error']}")
        
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
