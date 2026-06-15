"""
FundFL Chart Utilities
======================
Visualization helpers for fund risk analysis.
"""

import numpy as np
import pandas as pd


def generate_fund_returns(annual_return: float = 0.12, annual_vol: float = 0.24,
                          days: int = 252, seed: int = None) -> np.ndarray:
    """Generate synthetic daily returns."""
    if seed is not None:
        np.random.seed(seed)
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    return np.random.normal(daily_return, daily_vol, days)


def returns_to_dataframe(returns: np.ndarray, fund_name: str = "Fund") -> pd.DataFrame:
    """Convert returns to cumulative NAV DataFrame."""
    cum = np.cumprod(1 + returns)
    dates = pd.date_range(end="2026-04-29", periods=len(returns), freq="B")
    return pd.DataFrame({
        "date": dates,
        "NAV": cum,
        "daily_return": returns,
        "fund": fund_name,
    })


def compute_drawdown_series(returns: np.ndarray) -> np.ndarray:
    """Compute drawdown series."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return (cum - peak) / peak


def risk_radar_data(profile: dict) -> dict:
    """Prepare data for risk radar chart."""
    metrics = {
        "年化收益": abs(profile.get("annual_return", 0)) * 100,
        "年化波动": abs(profile.get("annual_volatility", 0)) * 100,
        "Sharpe": abs(profile.get("sharpe_ratio", 0)) * 20,
        "Sortino": abs(profile.get("sortino_ratio", 0)) * 20,
        "最大回撤": abs(profile.get("max_drawdown", 0)) * 100,
        "胜率": abs(profile.get("win_rate", 0)) * 100,
    }
    return metrics


def rolling_sharpe(returns: np.ndarray, window: int = 60) -> np.ndarray:
    """Compute rolling Sharpe ratio."""
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    return (rolling_mean * 252 - 0.02) / (rolling_std * np.sqrt(252) + 1e-8).values
