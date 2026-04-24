"""
Modern Portfolio Theory (MPT) Engine.

Pure mathematical functions for portfolio optimization.
No I/O, no side effects — accepts numpy/pandas, returns numpy/pandas.
Uses scipy.optimize for constrained portfolio weight optimization.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# --- Constants ---
TRADING_DAYS_PER_YEAR = 252


# --- Data Structures ---

@dataclass
class PortfolioResult:
    """Represents a single portfolio allocation and its performance metrics."""
    weights: np.ndarray
    expected_return: float
    volatility: float
    sharpe_ratio: float
    ticker_labels: List[str] = field(default_factory=list)


@dataclass
class EfficientFrontierResult:
    """Complete output of an efficient frontier computation."""
    max_sharpe: PortfolioResult
    min_variance: PortfolioResult
    frontier_returns: np.ndarray
    frontier_volatilities: np.ndarray
    individual_stats: Dict[str, Dict[str, float]]
    correlation_matrix: pd.DataFrame
    covariance_matrix: pd.DataFrame


# --- Core Computation Functions ---

def compute_daily_returns(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily percentage returns from a price DataFrame.

    :param prices_df: DataFrame with columns=tickers, rows=dates, values=Close prices.
    :return: DataFrame of daily returns with first row dropped (NaN).
    """
    return prices_df.pct_change().dropna()


def compute_annual_returns(prices_df: pd.DataFrame) -> pd.Series:
    """
    Compute mean annualized return per asset from daily close prices.

    :param prices_df: DataFrame with columns=tickers, rows=dates, values=Close prices.
    :return: Series of annualized mean returns indexed by ticker.
    """
    daily_returns = compute_daily_returns(prices_df)
    return daily_returns.mean() * TRADING_DAYS_PER_YEAR


def compute_covariance_matrix(prices_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the annualized covariance matrix from daily close prices.

    :param prices_df: DataFrame with columns=tickers, rows=dates, values=Close prices.
    :return: DataFrame (n_assets × n_assets) annualized covariance matrix.
    """
    daily_returns = compute_daily_returns(prices_df)
    return daily_returns.cov() * TRADING_DAYS_PER_YEAR


def portfolio_performance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.045
) -> Tuple[float, float, float]:
    """
    Calculate portfolio expected return, volatility, and Sharpe ratio.

    :param weights: Array of portfolio weights, shape (n_assets,).
    :param mean_returns: Array of annualized mean returns per asset.
    :param cov_matrix: Annualized covariance matrix, shape (n_assets, n_assets).
    :param risk_free_rate: Annualized risk-free rate.
    :return: Tuple of (expected_return, volatility, sharpe_ratio).
    """
    expected_return = float(np.dot(weights, mean_returns))
    portfolio_variance = float(np.dot(weights.T, np.dot(cov_matrix, weights)))
    volatility = np.sqrt(portfolio_variance)

    if volatility == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (expected_return - risk_free_rate) / volatility

    return expected_return, volatility, sharpe_ratio


# --- Optimization Functions ---

def _negative_sharpe(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float
) -> float:
    """Objective function: negative Sharpe ratio (for minimization)."""
    _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
    return -sharpe


def _portfolio_variance(
    weights: np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray
) -> float:
    """Objective function: portfolio variance (for minimization)."""
    return float(np.dot(weights.T, np.dot(cov_matrix, weights)))


def _portfolio_return(
    weights: np.ndarray,
    mean_returns: np.ndarray
) -> float:
    """Helper: portfolio expected return."""
    return float(np.dot(weights, mean_returns))


def optimize_max_sharpe(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.045,
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Find the portfolio weights that maximize the Sharpe ratio.

    Uses SLSQP constrained optimization with sum-to-one equality constraint
    and per-asset weight bounds.

    :param mean_returns: Array of annualized mean returns, shape (n_assets,).
    :param cov_matrix: Annualized covariance matrix, shape (n_assets, n_assets).
    :param risk_free_rate: Annualized risk-free rate.
    :param weight_bounds: (min_weight, max_weight) per asset. Default long-only.
    :return: Array of optimal weights, shape (n_assets,).
    :raises ValueError: If optimization fails to converge.
    """
    n_assets = len(mean_returns)
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = tuple(weight_bounds for _ in range(n_assets))

    result = minimize(
        _negative_sharpe,
        initial_weights,
        args=(mean_returns, cov_matrix, risk_free_rate),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if not result.success:
        raise ValueError(f"Max Sharpe optimization failed to converge: {result.message}")

    return result.x


def optimize_min_variance(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
) -> np.ndarray:
    """
    Find the portfolio weights that minimize portfolio variance.

    :param mean_returns: Array of annualized mean returns, shape (n_assets,).
    :param cov_matrix: Annualized covariance matrix, shape (n_assets, n_assets).
    :param weight_bounds: (min_weight, max_weight) per asset. Default long-only.
    :return: Array of optimal weights, shape (n_assets,).
    :raises ValueError: If optimization fails to converge.
    """
    n_assets = len(mean_returns)
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = tuple(weight_bounds for _ in range(n_assets))

    result = minimize(
        _portfolio_variance,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if not result.success:
        raise ValueError(f"Min Variance optimization failed to converge: {result.message}")

    return result.x


def _optimize_for_target_return(
    target_return: float,
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
) -> Optional[np.ndarray]:
    """
    Find minimum-variance portfolio for a given target return.

    :param target_return: The desired portfolio expected return.
    :param mean_returns: Array of annualized mean returns.
    :param cov_matrix: Annualized covariance matrix.
    :param weight_bounds: Per-asset weight bounds.
    :return: Optimal weights or None if optimization fails.
    """
    n_assets = len(mean_returns)
    initial_weights = np.array([1.0 / n_assets] * n_assets)

    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: _portfolio_return(w, mean_returns) - target_return}
    ]
    bounds = tuple(weight_bounds for _ in range(n_assets))

    result = minimize(
        _portfolio_variance,
        initial_weights,
        args=(mean_returns, cov_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-12}
    )

    if result.success:
        return result.x
    return None


def compute_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.045,
    num_points: int = 100,
    weight_bounds: Tuple[float, float] = (0.0, 1.0)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the efficient frontier as arrays of (volatilities, returns).

    Sweeps target returns from the min-variance return to the max single-asset
    return, solving for minimum-variance at each target. Points where the
    optimizer fails to converge are silently skipped.

    :param mean_returns: Array of annualized mean returns.
    :param cov_matrix: Annualized covariance matrix.
    :param risk_free_rate: Annualized risk-free rate (unused directly, kept for API consistency).
    :param num_points: Number of points along the frontier.
    :param weight_bounds: Per-asset weight bounds.
    :return: Tuple of (frontier_volatilities, frontier_returns) arrays.
    """
    # Find the min-variance portfolio return as the lower bound
    min_var_weights = optimize_min_variance(mean_returns, cov_matrix, weight_bounds)
    min_var_return = _portfolio_return(min_var_weights, mean_returns)

    # Upper bound: maximum achievable return given the weight bounds
    max_return = float(np.max(mean_returns))
    if weight_bounds[1] < 1.0:
        # With upper bound constraints, max return is the weighted max
        # Use optimization to find the actual max achievable return
        n_assets = len(mean_returns)
        neg_return_result = minimize(
            lambda w: -_portfolio_return(w, mean_returns),
            np.array([1.0 / n_assets] * n_assets),
            method='SLSQP',
            bounds=tuple(weight_bounds for _ in range(n_assets)),
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        )
        if neg_return_result.success:
            max_return = -neg_return_result.fun

    target_returns = np.linspace(min_var_return, max_return, num_points)

    frontier_volatilities = []
    frontier_returns = []

    for target in target_returns:
        weights = _optimize_for_target_return(target, mean_returns, cov_matrix, weight_bounds)
        if weights is not None:
            ret, vol, _ = portfolio_performance(weights, mean_returns, cov_matrix, risk_free_rate)
            frontier_returns.append(ret)
            frontier_volatilities.append(vol)

    return np.array(frontier_volatilities), np.array(frontier_returns)


# --- High-Level Pipeline ---

def run_optimization(
    prices_df: pd.DataFrame,
    risk_free_rate: float = 0.045,
    weight_bounds: Tuple[float, float] = (0.0, 1.0),
    num_frontier_points: int = 100
) -> EfficientFrontierResult:
    """
    Full MPT optimization pipeline: computes returns, covariance,
    max-Sharpe portfolio, min-variance portfolio, and the efficient frontier.

    :param prices_df: DataFrame with columns=tickers, rows=dates, values=Close prices.
                      Must have at least 30 rows and 2+ columns.
    :param risk_free_rate: Annualized risk-free rate.
    :param weight_bounds: (min_weight, max_weight) per asset.
    :param num_frontier_points: Number of points to compute along the frontier.
    :return: EfficientFrontierResult containing all optimization outputs.
    :raises ValueError: If input data is insufficient or optimization fails.
    """
    if prices_df.shape[0] < 30:
        raise ValueError(
            f"Insufficient price history: {prices_df.shape[0]} days provided, "
            f"minimum 30 required for meaningful statistics."
        )

    if prices_df.shape[1] < 2:
        # Single asset — no optimization possible, just report stats
        tickers = list(prices_df.columns)
        ann_returns = compute_annual_returns(prices_df)
        cov_matrix = compute_covariance_matrix(prices_df)

        weights = np.array([1.0])
        ret, vol, sharpe = portfolio_performance(
            weights, ann_returns.values, cov_matrix.values, risk_free_rate
        )
        single_result = PortfolioResult(
            weights=weights,
            expected_return=ret,
            volatility=vol,
            sharpe_ratio=sharpe,
            ticker_labels=tickers
        )
        return EfficientFrontierResult(
            max_sharpe=single_result,
            min_variance=single_result,
            frontier_returns=np.array([ret]),
            frontier_volatilities=np.array([vol]),
            individual_stats={tickers[0]: {'annual_return': ret, 'annual_volatility': vol}},
            correlation_matrix=pd.DataFrame([[1.0]], columns=tickers, index=tickers),
            covariance_matrix=cov_matrix
        )

    tickers = list(prices_df.columns)
    ann_returns = compute_annual_returns(prices_df)
    cov_matrix = compute_covariance_matrix(prices_df)

    mean_ret_arr = ann_returns.values
    cov_arr = cov_matrix.values

    # Individual asset stats
    individual_stats = {}
    for i, ticker in enumerate(tickers):
        individual_stats[ticker] = {
            'annual_return': float(ann_returns.iloc[i]),
            'annual_volatility': float(np.sqrt(cov_matrix.iloc[i, i]))
        }

    # Optimize
    max_sharpe_weights = optimize_max_sharpe(mean_ret_arr, cov_arr, risk_free_rate, weight_bounds)
    ms_ret, ms_vol, ms_sharpe = portfolio_performance(
        max_sharpe_weights, mean_ret_arr, cov_arr, risk_free_rate
    )

    min_var_weights = optimize_min_variance(mean_ret_arr, cov_arr, weight_bounds)
    mv_ret, mv_vol, mv_sharpe = portfolio_performance(
        min_var_weights, mean_ret_arr, cov_arr, risk_free_rate
    )

    # Frontier
    frontier_vols, frontier_rets = compute_efficient_frontier(
        mean_ret_arr, cov_arr, risk_free_rate, num_frontier_points, weight_bounds
    )

    # Correlation matrix
    daily_returns = compute_daily_returns(prices_df)
    corr_matrix = daily_returns.corr()

    return EfficientFrontierResult(
        max_sharpe=PortfolioResult(
            weights=max_sharpe_weights,
            expected_return=ms_ret,
            volatility=ms_vol,
            sharpe_ratio=ms_sharpe,
            ticker_labels=tickers
        ),
        min_variance=PortfolioResult(
            weights=min_var_weights,
            expected_return=mv_ret,
            volatility=mv_vol,
            sharpe_ratio=mv_sharpe,
            ticker_labels=tickers
        ),
        frontier_returns=frontier_rets,
        frontier_volatilities=frontier_vols,
        individual_stats=individual_stats,
        correlation_matrix=corr_matrix,
        covariance_matrix=cov_matrix
    )
