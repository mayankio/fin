# Financial Analysis Tool

An automated, object-oriented Python application for stock market analysis utilizing market data via `yfinance`. The system is strictly typed and heavily documented.

## Architecture Overview

The system strictly adheres to SOLID principles and is structured into layers to cleanly separate concerns:

1. **Data Layer (`data_provider.py`)**: Abstracted via `MarketDataProvider` to decouple from the specific data source. The `YFinanceProvider` class handles fetching historical prices, financials, and company data.
2. **Analysis Layer (`analyzers.py`)**: Implements the **Strategy Pattern** via `BaseAnalyzer`. Multiple standalone modular analyzers perform specific logic (Technical, Fundamental, Competitor, Portfolio).
3. **Presentation Layer (`reporting.py`)**: Responsible for drawing interactive charts (`Visualizer`) via Plotly and generating structured text analysis reports (`RecommendationEngine`).
4. **Orchestration Layer (`market_scanner.py`, `asset_profiler.py`, `portfolio_optimizer.py`)**: Standalone scripts that execute specific workflows using the underlying layers.

## Setup Instructions

1. Ensure you have Python 3.9+ installed and optionally activate a virtual environment.
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Execution Use Cases

Run the corresponding standalone scripts directly:

### MarketScanner (`market_scanner.py`)
Scans a list of tickers (e.g. AAPL, MSFT, GOOGL, AMZN, META). Runs Technical and Fundamental analysis across all, evaluates a score, and displays the **Top 2 recommended stocks** along with visualizations.
```bash
python3 market_scanner.py --tickers AAPL MSFT GOOGL AMZN META
```

### AssetProfiler (`asset_profiler.py`)
Detailed analysis of a single stock target against sector peers. Specifically analyzes Technicals, Fundamentals, and comparative P/E against competitors.
```bash
python3 asset_profiler.py --ticker NVDA --competitors AMD INTC QCOM
```

### PortfolioOptimizer (`portfolio_optimizer.py`)
Reviews an arbitrary portfolio based on given weighted ticker allocation. Highlights any sector concentration risk (e.g. >40% in Tech) and aggregated portfolio Beta.
```bash
python3 portfolio_optimizer.py
```

## Extensibility

The application leverages the **Strategy Pattern** for the Analysis Layer, making it inherently extensible without needing to rewrite the core controller logic.

### How to add a new `MLForecastingAnalyzer` or `SentimentAnalyzer`:

1. **Create the Strategy Class**: In `analyzers.py`, create a new class inheriting from `BaseAnalyzer`:
   ```python
   class MLForecastingAnalyzer(BaseAnalyzer):
       def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
           # 1. Access required data dynamically
           history = data.get('history')
           
           # 2. Add your custom complex logic (e.g. Prophet, LSTM, or NLTK Sentiment)
           # prediction = model.predict(history)
           
           # 3. Return structured dictionary of results
           return {"prediction": "Bullish", "confidence": 0.85}
   ```

2. **Register the Strategy**: In your target orchestrator (e.g., `asset_profiler.py`), instantiate your new analyzer in the constructor:
   ```python
   self.ml_analyzer = MLForecastingAnalyzer()
   ```

3. **Invoke the Strategy in a Workflow**: Inside the app's `run()` method, populate necessary data and invoke the `.run()` method:
   ```python
   ml_results = self.ml_analyzer.run({'history': history})
   ```

4. **Update Recommendation Logic (Optional)**: If those ML or Sentiment results should impact the final BUY/HOLD/SELL verdict, update the `generate_recommendation` method signature inside `reporting.py` to accept `ml_results` and incorporate it into the scoring logic.
