# Low-Level Design (LLD): Financial Analysis Application

## 1. Class Diagrams & Interfaces

The lower-level implementation relies on strict Python typing (`typing.Dict`, `typing.List`, `typing.Any`) to enforce schema validation at runtime. 

### 1.1. Module: `data_provider.py`
**Abstract Base Class: `MarketDataProvider`**
- Interfacing baseline for any future data integration (e.g. Bloomberg, Alpaca).
- Methods defined: 
  - `get_historical_data(ticker, period) -> pd.DataFrame`
  - `get_financials(ticker) -> pd.DataFrame`
  - `get_balance_sheet(ticker) -> pd.DataFrame`
  - `get_major_holders(ticker) -> pd.DataFrame`
  - `get_info(ticker) -> Dict[str, Any]`

**Implementation: `YFinanceProvider`**
- Encapsulates the `yfinance` module.
- `get_historical_data` relies rigidly on `stock.history()`. If `.empty` is returned, it raises a `ValueError` protecting downstream pandas operations from `IndexError`.
- Leverages `.info` property to fetch critical fundamental statics.

### 1.2. Module: `analyzers.py`
**Abstract Base Class: `BaseAnalyzer`**
- `run(data: Dict[str, Any]) -> Dict[str, Any]` represents the uniform polymorphic method for the Strategy pattern.

**Implementation: `TechnicalAnalyzer`**
- Uses the `ta` library for deterministic mathematics.
- Generates `SMAIndicator` instances with windows `50` and `200`.
- Generates `MACD` instance using `.macd()` and `.macd_diff()`.
- Mutates a copy of the input DataFrame and calculates the ultimate summary string (e.g., matching Golden Cross logic: `latest['SMA_50'] > latest['SMA_200']`).

**Implementation: `FundamentalAnalyzer`**
- Operates primarily on trailing/forward elements provided by YFinance `info` dicts.
- Gracefully handles schema variations using `,get('...', None)` defaults to guard against missing `trailingPE` or `debtToEquity` endpoints.

**Implementation: `CompetitorAnalyzer`**
- Iterates over an array of peer payload dictionaries comparing target P/E and target ROE systematically against the aggregated sector benchmark.

**Implementation: `PortfolioAnalyzer`**
- Consumes dictionary formats modeled as `{'ticker': {'weight': float, 'info': dict}}`.
- Sum-products the weights into aggregated total portfolio Beta.

### 1.3. Module: `reporting.py`
**Class: `Visualizer`**
- Imports `plotly.graph_objects` (`go`).
- Uses `make_subplots` allocating row height ratio `[0.3, 0.7]` (30% bottom MACD height, 70% top Candlestick height).
- Leverages Python list comprehension to dynamically assign bar colors (`go.Bar`) based on MACD histogram zero-line crossover.

**Class: `RecommendationEngine`**
- Stateful heuristical integer scoring system: `score = 0`.
- Additive/Subtractive logic models:
  - Technical: `+1` for Uptrend, `+1` for Oversold, `-1` for Downtrend.
  - Fundamental: `+1` for `P/E < 15`, `-1` for `P/E > 30`.
- Verdict Engine logic limits:
  - `score >= 2` mapping to `VERDICT: BUY`
  - `score <= -1` mapping to `VERDICT: SELL`

## 2. Controllers Formulation

The three individual orchestrators utilize explicit command-line interfacing (CLI):

1. **`market_scanner.py`**
   - Ingests string lists of tickers via `argparse` `nargs='+'`.
   - Generates iterative try/except catches on individual assets. 
   - Dynamically parses the `RecommendationEngine` string payload output to pull out `Total Model Score` string integers enabling `.sort(key=...)` functionality for ranking.

2. **`asset_profiler.py`**
   - Maps singular `target` string and multi-string `competitors` logic.
   - Responsible for combining FA, TA, and Comparative Peer outputs into synchronous presentation via standard out.

3. **`portfolio_optimizer.py`**
   - Hardcoded dictionary configurations mapping ticker keys to weighted value bounds ensuring validation bounds (Sum == 1.0) internally.

## 3. Expected Exception Strategies

| Vector | Failure Mechanism | Layer Bubble Resolution |
|--------|-------------------|-------------------------|
| Ticker Does Not Exist | YFinance returns null/empty arrays | `data_provider.py` safely raises explicit `ValueError` bypassing pandas failures. Controller skips ticker. |
| Missing `trailingPE` Data | Asset lacks GAAP required EPS | `analyzers.py` defaults to `None`. Reporting engine checks `is not None` and appends `None` notes securely. |
| Disconnected Network | Failed `yfinance` socket bind | Exception bubbles to controller `except Exception` printing "Failed to fetch...". Does not exit runtime block loops (specifically for the Scanner array traversal). |
