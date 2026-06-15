"""
FundFL Risk Analyzer
====================
16 risk metrics computation + DINOv2-style fund feature extraction.

Risk Metrics (16):
  1. Annualized Return
  2. Annualized Volatility
  3. Sharpe Ratio
  4. Sortino Ratio
  5. Jensen's Alpha
  6. Beta
  7. Maximum Drawdown
  8. Calmar Ratio
  9. VaR (95%)
  10. CVaR (95%)
  11. Information Ratio
  12. Tracking Error
  13. Skewness
  14. Kurtosis
  15. M² (Modigliani-Modigliani)
  16. Win Rate

FL Integration:
  - Risk feature vectors shared via FedAvg (not raw returns)
  - Cross-fund similarity search via HNSW
  - Privacy: individual fund returns never leave the institution
"""

import numpy as np
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class RiskProfile:
    """Complete risk profile for a fund."""
    fund_code: str
    fund_name: str
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    jensen_alpha: float
    beta: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    cvar_95: float
    information_ratio: float
    tracking_error: float
    skewness: float
    kurtosis: float
    m_squared: float
    win_rate: float

    def to_feature_vector(self) -> np.ndarray:
        """Convert to 16-dim feature vector for HNSW search."""
        return np.array([
            self.annual_return, self.annual_volatility, self.sharpe_ratio,
            self.sortino_ratio, self.jensen_alpha, self.beta,
            self.max_drawdown, self.calmar_ratio, self.var_95,
            self.cvar_95, self.information_ratio, self.tracking_error,
            self.skewness, self.kurtosis, self.m_squared, self.win_rate,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


class RiskAnalyzer:
    """Compute 16 risk metrics from fund return series."""

    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate

    def compute(self, returns: np.ndarray, benchmark: Optional[np.ndarray] = None,
                fund_code: str = "", fund_name: str = "") -> RiskProfile:
        """Compute full risk profile from daily returns."""
        n = len(returns)
        annual_factor = 252

        # Basic stats
        annual_return = float(np.mean(returns) * annual_factor)
        annual_vol = float(np.std(returns, ddof=1) * np.sqrt(annual_factor))

        # Sharpe
        sharpe = (annual_return - self.RISK_FREE_RATE) / max(annual_vol, 1e-8)

        # Sortino
        downside = returns[returns < 0]
        downside_vol = float(np.std(downside, ddof=1) * np.sqrt(annual_factor)) if len(downside) > 1 else 1e-8
        sortino = (annual_return - self.RISK_FREE_RATE) / downside_vol

        # Max Drawdown
        cum = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cum)
        drawdown = (cum - peak) / peak
        max_dd = float(np.min(drawdown))

        # Calmar
        calmar = annual_return / abs(max_dd) if abs(max_dd) > 1e-8 else 0.0

        # VaR & CVaR
        var_95 = float(np.percentile(returns, 5))
        cvar_95 = float(np.mean(returns[returns <= var_95]))

        # Skewness & Kurtosis
        skewness = float(self._skewness(returns))
        kurtosis = float(self._kurtosis(returns))

        # Win Rate
        win_rate = float(np.mean(returns > 0))

        # M²
        m_squared = self.RISK_FREE_RATE + sharpe * (0.15 - self.RISK_FREE_RATE)

        # Alpha & Beta (if benchmark provided)
        if benchmark is not None and len(benchmark) == n:
            beta = float(np.cov(returns, benchmark)[0, 1] / np.var(benchmark, ddof=1))
            benchmark_return = float(np.mean(benchmark) * annual_factor)
            jensen_alpha = annual_return - (self.RISK_FREE_RATE + beta * (benchmark_return - self.RISK_FREE_RATE))
            tracking_error = float(np.std(returns - benchmark, ddof=1) * np.sqrt(annual_factor))
            ir = (annual_return - benchmark_return) / tracking_error if tracking_error > 1e-8 else 0.0
        else:
            beta = jensen_alpha = tracking_error = ir = 0.0

        return RiskProfile(
            fund_code=fund_code, fund_name=fund_name,
            annual_return=annual_return, annual_volatility=annual_vol,
            sharpe_ratio=sharpe, sortino_ratio=sortino,
            jensen_alpha=jensen_alpha, beta=beta,
            max_drawdown=max_dd, calmar_ratio=calmar,
            var_95=var_95, cvar_95=cvar_95,
            information_ratio=ir, tracking_error=tracking_error,
            skewness=skewness, kurtosis=kurtosis,
            m_squared=m_squared, win_rate=win_rate,
        )

    @staticmethod
    def _skewness(x: np.ndarray) -> float:
        n = len(x)
        if n < 3: return 0.0
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-10: return 0.0
        return float(np.mean(((x - m) / s) ** 3))

    @staticmethod
    def _kurtosis(x: np.ndarray) -> float:
        n = len(x)
        if n < 4: return 0.0
        m = np.mean(x)
        s = np.std(x, ddof=1)
        if s < 1e-10: return 0.0
        return float(np.mean(((x - m) / s) ** 4) - 3)

    @staticmethod
    def similarity(profile_a: RiskProfile, profile_b: RiskProfile) -> float:
        """Cosine similarity between two risk profiles."""
        va = profile_a.to_feature_vector()
        vb = profile_b.to_feature_vector()
        dot = np.dot(va, vb)
        norm = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(dot / norm) if norm > 1e-8 else 0.0
