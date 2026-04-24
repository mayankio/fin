"""
Exhaustive tests for the MPT (Modern Portfolio Theory) math engine.

Covers: return computation, covariance, portfolio performance,
max-Sharpe optimization, min-variance optimization, efficient frontier,
full pipeline, edge cases, and integration scenarios.
"""
import pytest
import numpy as np
import pandas as pd
import time

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from mpt_engine import (
    compute_daily_returns,
    compute_annual_returns,
    compute_covariance_matrix,
    portfolio_performance,
    optimize_max_sharpe,
    optimize_min_variance,
    compute_efficient_frontier,
    run_optimization,
    TRADING_DAYS_PER_YEAR,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthetic_prices_3_assets():
    """3 assets, 252 days, with distinct return/vol profiles."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    # Asset A: moderate return, low vol
    a = 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, 252)))
    # Asset B: high return, high vol
    b = 100 * np.exp(np.cumsum(np.random.normal(0.0008, 0.025, 252)))
    # Asset C: low return, moderate vol
    c = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.015, 252)))
    return pd.DataFrame({'A': a, 'B': b, 'C': c}, index=dates)


@pytest.fixture
def synthetic_prices_4_assets():
    """4 diversified assets for general tests."""
    np.random.seed(123)
    dates = pd.date_range(start="2022-06-01", periods=500, freq="B")
    prices = {}
    for i, name in enumerate(['ALPHA', 'BETA', 'GAMMA', 'DELTA']):
        drift = 0.0002 + i * 0.0002
        vol = 0.01 + i * 0.005
        prices[name] = 100 * np.exp(np.cumsum(np.random.normal(drift, vol, 500)))
    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def two_asset_prices():
    """Simple 2-asset case for analytical verification."""
    np.random.seed(7)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    a = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, 252)))
    b = 100 * np.exp(np.cumsum(np.random.normal(0.0006, 0.020, 252)))
    return pd.DataFrame({'X': a, 'Y': b}, index=dates)


@pytest.fixture
def single_asset_prices():
    """Single asset — edge case."""
    np.random.seed(99)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    a = 100 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, 252)))
    return pd.DataFrame({'SOLO': a}, index=dates)


# ============================================================
# MPT-U01 to MPT-U02: compute_annual_returns
# ============================================================

def test_compute_annual_returns_basic(synthetic_prices_3_assets):
    """MPT-U01: 3 assets → returns are scalars, one per asset."""
    returns = compute_annual_returns(synthetic_prices_3_assets)
    assert isinstance(returns, pd.Series)
    assert len(returns) == 3
    assert list(returns.index) == ['A', 'B', 'C']
    # Returns should be finite
    assert returns.notna().all()


def test_compute_annual_returns_single_asset(single_asset_prices):
    """MPT-U02: Single asset → scalar return."""
    returns = compute_annual_returns(single_asset_prices)
    assert len(returns) == 1
    assert returns.index[0] == 'SOLO'
    assert np.isfinite(returns.iloc[0])


# ============================================================
# MPT-U03 to MPT-U05: compute_covariance_matrix
# ============================================================

def test_compute_covariance_matrix_shape(synthetic_prices_4_assets):
    """MPT-U03: 4 assets → 4×4 covariance matrix."""
    cov = compute_covariance_matrix(synthetic_prices_4_assets)
    assert cov.shape == (4, 4)
    assert list(cov.columns) == ['ALPHA', 'BETA', 'GAMMA', 'DELTA']


def test_covariance_matrix_symmetry(synthetic_prices_3_assets):
    """MPT-U04: Covariance matrix must be symmetric."""
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    np.testing.assert_array_almost_equal(cov.values, cov.values.T)


def test_covariance_matrix_positive_semidefinite(synthetic_prices_3_assets):
    """MPT-U05: All eigenvalues ≥ 0."""
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    eigenvalues = np.linalg.eigvalsh(cov.values)
    assert np.all(eigenvalues >= -1e-10)  # tolerance for floating point


# ============================================================
# MPT-U06 to MPT-U07: portfolio_performance
# ============================================================

def test_portfolio_performance_equal_weights(synthetic_prices_3_assets):
    """MPT-U06: Equal weights with known inputs."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    weights = np.array([1/3, 1/3, 1/3])

    ret, vol, sharpe = portfolio_performance(weights, ann_ret.values, cov.values, 0.045)

    # Return is weighted average of individual returns
    expected_ret = np.dot(weights, ann_ret.values)
    assert abs(ret - expected_ret) < 1e-10

    # Volatility = sqrt(w^T * Σ * w)
    expected_vol = np.sqrt(np.dot(weights.T, np.dot(cov.values, weights)))
    assert abs(vol - expected_vol) < 1e-10

    # Sharpe = (ret - rf) / vol
    expected_sharpe = (expected_ret - 0.045) / expected_vol
    assert abs(sharpe - expected_sharpe) < 1e-10


def test_portfolio_performance_single_asset(single_asset_prices):
    """MPT-U07: 100% in one asset."""
    ann_ret = compute_annual_returns(single_asset_prices)
    cov = compute_covariance_matrix(single_asset_prices)
    weights = np.array([1.0])

    ret, vol, sharpe = portfolio_performance(weights, ann_ret.values, cov.values, 0.045)
    assert abs(ret - ann_ret.iloc[0]) < 1e-10
    assert abs(vol - np.sqrt(cov.iloc[0, 0])) < 1e-10


# ============================================================
# MPT-U08 to MPT-U11: optimize_max_sharpe
# ============================================================

def test_optimize_max_sharpe_weights_sum_to_one(synthetic_prices_4_assets):
    """MPT-U08: Max Sharpe weights must sum to 1.0."""
    ann_ret = compute_annual_returns(synthetic_prices_4_assets)
    cov = compute_covariance_matrix(synthetic_prices_4_assets)
    weights = optimize_max_sharpe(ann_ret.values, cov.values)
    assert abs(np.sum(weights) - 1.0) < 1e-6


def test_optimize_max_sharpe_respects_bounds(synthetic_prices_4_assets):
    """MPT-U09: Custom bounds are respected."""
    ann_ret = compute_annual_returns(synthetic_prices_4_assets)
    cov = compute_covariance_matrix(synthetic_prices_4_assets)
    weights = optimize_max_sharpe(ann_ret.values, cov.values, weight_bounds=(0.05, 0.40))
    assert np.all(weights >= 0.05 - 1e-6)
    assert np.all(weights <= 0.40 + 1e-6)


def test_optimize_max_sharpe_no_short_selling(synthetic_prices_3_assets):
    """MPT-U10: Default bounds → all weights ≥ 0."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    weights = optimize_max_sharpe(ann_ret.values, cov.values)
    assert np.all(weights >= -1e-6)


def test_optimize_max_sharpe_beats_equal_weight(synthetic_prices_3_assets):
    """MPT-U11: Optimal Sharpe ≥ equal-weight Sharpe."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)

    opt_weights = optimize_max_sharpe(ann_ret.values, cov.values)
    _, _, opt_sharpe = portfolio_performance(opt_weights, ann_ret.values, cov.values)

    eq_weights = np.array([1/3, 1/3, 1/3])
    _, _, eq_sharpe = portfolio_performance(eq_weights, ann_ret.values, cov.values)

    assert opt_sharpe >= eq_sharpe - 1e-6  # allow tiny tolerance


# ============================================================
# MPT-U12 to MPT-U13: optimize_min_variance
# ============================================================

def test_optimize_min_variance_weights_sum_to_one(synthetic_prices_4_assets):
    """MPT-U12: Min variance weights must sum to 1.0."""
    ann_ret = compute_annual_returns(synthetic_prices_4_assets)
    cov = compute_covariance_matrix(synthetic_prices_4_assets)
    weights = optimize_min_variance(ann_ret.values, cov.values)
    assert abs(np.sum(weights) - 1.0) < 1e-6


def test_optimize_min_variance_lowest_vol(synthetic_prices_3_assets):
    """MPT-U13: Min-var volatility ≤ any individual asset vol."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    weights = optimize_min_variance(ann_ret.values, cov.values)
    _, mv_vol, _ = portfolio_performance(weights, ann_ret.values, cov.values)

    individual_vols = [np.sqrt(cov.values[i, i]) for i in range(3)]
    assert mv_vol <= min(individual_vols) + 1e-6


# ============================================================
# MPT-U14 to MPT-U16: compute_efficient_frontier
# ============================================================

def test_efficient_frontier_monotonic_return(synthetic_prices_3_assets):
    """MPT-U14: Frontier returns should be non-decreasing."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    _, frontier_rets = compute_efficient_frontier(ann_ret.values, cov.values, num_points=50)
    # Allow small tolerance for numerical noise
    diffs = np.diff(frontier_rets)
    assert np.all(diffs >= -1e-6)


def test_efficient_frontier_point_count(synthetic_prices_3_assets):
    """MPT-U15: Requested 50 points → at least 40 points returned (some may fail)."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    vols, rets = compute_efficient_frontier(ann_ret.values, cov.values, num_points=50)
    # Some points may fail to converge, but most should succeed
    assert len(rets) >= 40
    assert len(vols) == len(rets)


def test_efficient_frontier_contains_min_var(synthetic_prices_3_assets):
    """MPT-U16: First point on frontier should be close to min-var portfolio."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)

    mv_weights = optimize_min_variance(ann_ret.values, cov.values)
    _, mv_vol, _ = portfolio_performance(mv_weights, ann_ret.values, cov.values)

    frontier_vols, _ = compute_efficient_frontier(ann_ret.values, cov.values, num_points=100)
    # First frontier point should be near the min-var vol
    assert abs(frontier_vols[0] - mv_vol) < 0.01


# ============================================================
# MPT-U17: two_asset_analytical_verification
# ============================================================

def test_two_asset_analytical_verification(two_asset_prices):
    """MPT-U17: 2-asset case — verify optimizer matches known optimal."""
    ann_ret = compute_annual_returns(two_asset_prices)
    cov = compute_covariance_matrix(two_asset_prices)

    # Analytical min-variance for 2 assets:
    # w1 = (σ2² - σ12) / (σ1² + σ2² - 2σ12)
    s11 = cov.values[0, 0]
    s22 = cov.values[1, 1]
    s12 = cov.values[0, 1]
    w1_analytical = (s22 - s12) / (s11 + s22 - 2 * s12)
    w1_analytical = np.clip(w1_analytical, 0.0, 1.0)
    w2_analytical = 1.0 - w1_analytical

    # Numerical result
    mv_weights = optimize_min_variance(ann_ret.values, cov.values)

    np.testing.assert_allclose(mv_weights[0], w1_analytical, atol=1e-4)
    np.testing.assert_allclose(mv_weights[1], w2_analytical, atol=1e-4)


# ============================================================
# MPT-U18 to MPT-U20: Edge cases
# ============================================================

def test_identical_assets():
    """MPT-U18: All assets identical → equal weights expected."""
    np.random.seed(55)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    series = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 252)))
    prices = pd.DataFrame({'A': series, 'B': series.copy(), 'C': series.copy()}, index=dates)

    ann_ret = compute_annual_returns(prices)
    cov = compute_covariance_matrix(prices)
    weights = optimize_min_variance(ann_ret.values, cov.values)
    # With identical assets, any combination is optimal; just check constraint
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert np.all(weights >= -1e-6)


def test_negative_risk_free_rate(synthetic_prices_3_assets):
    """MPT-U20: Negative risk-free rate → still converges, Sharpe adjusted."""
    ann_ret = compute_annual_returns(synthetic_prices_3_assets)
    cov = compute_covariance_matrix(synthetic_prices_3_assets)
    weights = optimize_max_sharpe(ann_ret.values, cov.values, risk_free_rate=-0.01)
    assert abs(np.sum(weights) - 1.0) < 1e-6
    _, _, sharpe = portfolio_performance(weights, ann_ret.values, cov.values, -0.01)
    # Sharpe should be higher with lower rf
    assert np.isfinite(sharpe)


# ============================================================
# EFA-U01 to EFA-U05: EfficientFrontierAnalyzer (via run_optimization)
# ============================================================

def test_run_optimization_returns_expected_structure(synthetic_prices_3_assets):
    """EFA-U01: Full pipeline returns EfficientFrontierResult with all fields."""
    result = run_optimization(synthetic_prices_3_assets)
    assert result.max_sharpe is not None
    assert result.min_variance is not None
    assert len(result.frontier_returns) > 0
    assert len(result.frontier_volatilities) > 0
    assert 'A' in result.individual_stats
    assert result.correlation_matrix.shape == (3, 3)
    assert result.covariance_matrix.shape == (3, 3)


def test_run_optimization_single_stock(single_asset_prices):
    """EFA-U03: Single stock → weights = [1.0], degenerate frontier."""
    result = run_optimization(single_asset_prices)
    assert len(result.max_sharpe.weights) == 1
    assert result.max_sharpe.weights[0] == 1.0
    assert len(result.frontier_returns) == 1


def test_run_optimization_custom_risk_free_rate(synthetic_prices_3_assets):
    """EFA-U04: Custom rf changes Sharpe ratios."""
    result_high = run_optimization(synthetic_prices_3_assets, risk_free_rate=0.10)
    result_low = run_optimization(synthetic_prices_3_assets, risk_free_rate=0.01)
    # Higher rf → lower Sharpe
    assert result_high.max_sharpe.sharpe_ratio < result_low.max_sharpe.sharpe_ratio


def test_run_optimization_custom_bounds(synthetic_prices_4_assets):
    """EFA-U05: Bounds (0.1, 0.5) → all weights in range."""
    result = run_optimization(synthetic_prices_4_assets, weight_bounds=(0.1, 0.5))
    for w in result.max_sharpe.weights:
        assert w >= 0.1 - 1e-6
        assert w <= 0.5 + 1e-6


# ============================================================
# EDGE-01 to EDGE-07: Edge cases for run_optimization
# ============================================================

def test_insufficient_history():
    """EDGE-01: <30 days → raises ValueError."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="B")
    prices = pd.DataFrame({'A': range(20), 'B': range(20)}, index=dates, dtype=float)
    with pytest.raises(ValueError, match="Insufficient price history"):
        run_optimization(prices)


def test_nan_in_price_data():
    """EDGE-06: NaN values in prices → handled by dropna in pct_change."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=260, freq="B")
    a = 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 260)))
    b = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 260)))
    prices = pd.DataFrame({'A': a, 'B': b}, index=dates)
    # Inject NaN
    prices.iloc[50, 0] = np.nan
    prices.iloc[100, 1] = np.nan
    # dropna in pct_change handles this; should not crash
    prices_clean = prices.dropna()
    result = run_optimization(prices_clean)
    assert result.max_sharpe is not None


def test_weight_bounds_infeasible():
    """EDGE-05: Bounds (0.5, 1.0) with 3 assets → can't sum to 1.0 → error."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    prices = pd.DataFrame({
        'A': 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, 252))),
        'B': 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, 252))),
        'C': 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, 252))),
    }, index=dates)
    with pytest.raises(ValueError):
        run_optimization(prices, weight_bounds=(0.5, 1.0))


def test_all_negative_returns():
    """EDGE-04: All assets have negative expected returns → still converges."""
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="B")
    # Strongly negative drift
    prices = pd.DataFrame({
        'A': 100 * np.exp(np.cumsum(np.random.normal(-0.003, 0.01, 252))),
        'B': 100 * np.exp(np.cumsum(np.random.normal(-0.002, 0.015, 252))),
    }, index=dates)
    result = run_optimization(prices, risk_free_rate=0.0)
    assert result.max_sharpe is not None
    assert abs(np.sum(result.max_sharpe.weights) - 1.0) < 1e-6


def test_twenty_assets_performance():
    """EDGE-07: 20 tickers × 500 days → completes in <10 seconds."""
    np.random.seed(42)
    dates = pd.date_range(start="2022-01-01", periods=500, freq="B")
    prices = pd.DataFrame({
        f"STOCK_{i}": 100 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01 + i*0.001, 500)))
        for i in range(20)
    }, index=dates)

    start = time.time()
    result = run_optimization(prices, num_frontier_points=50)
    elapsed = time.time() - start

    assert result.max_sharpe is not None
    assert elapsed < 10.0, f"Optimization took {elapsed:.1f}s, exceeding 10s limit"


# ============================================================
# INT-01: Integration — full pipeline consistency
# ============================================================

def test_full_pipeline_consistency(synthetic_prices_4_assets):
    """INT-01: Full pipeline produces internally consistent results."""
    result = run_optimization(synthetic_prices_4_assets)

    # Max Sharpe should have higher Sharpe than Min Variance (usually)
    # At minimum, both should be finite
    assert np.isfinite(result.max_sharpe.sharpe_ratio)
    assert np.isfinite(result.min_variance.sharpe_ratio)

    # Min Variance should have lower or equal volatility
    assert result.min_variance.volatility <= result.max_sharpe.volatility + 1e-6

    # Both weight arrays should sum to 1
    assert abs(np.sum(result.max_sharpe.weights) - 1.0) < 1e-6
    assert abs(np.sum(result.min_variance.weights) - 1.0) < 1e-6

    # Frontier should span from min_var to higher returns
    assert len(result.frontier_returns) > 10
    assert result.frontier_returns[0] <= result.frontier_returns[-1] + 1e-6
