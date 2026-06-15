from __future__ import annotations
"""
FundFL Constants
================
"""

# Fund categories
FUND_CATEGORIES = {
    "equity": {"cn": "股票型", "emoji": "📈"},
    "bond": {"cn": "债券型", "emoji": "🏦"},
    "hybrid": {"cn": "混合型", "emoji": "⚖️"},
    "money_market": {"cn": "货币型", "emoji": "💰"},
    "index": {"cn": "指数型", "emoji": "📊"},
    "quant": {"cn": "量化型", "emoji": "🤖"},
    "fof": {"cn": "FOF", "emoji": "🎯"},
    "alternative": {"cn": "另类投资", "emoji": "🔮"},
}

# Risk level thresholds
RISK_LEVELS = {
    "low": {"label": "低风险", "max_vol": 0.05, "color": "#22c55e"},
    "medium": {"label": "中风险", "max_vol": 0.15, "color": "#f59e0b"},
    "high": {"label": "高风险", "max_vol": 0.30, "color": "#ef4444"},
    "extreme": {"label": "极高风险", "max_vol": float("inf"), "color": "#7c3aed"},
}

# Metric descriptions
METRIC_DESCRIPTIONS = {
    "annual_return": "基金年化收益率，衡量投资回报水平",
    "annual_volatility": "基金年化波动率，衡量收益的不确定性",
    "sharpe_ratio": "风险调整后收益，每承担一单位风险获得的超额收益",
    "sortino_ratio": "下行风险调整后收益，只惩罚下行波动",
    "jensen_alpha": "相对于CAPM的超额收益，衡量基金经理的选股能力",
    "beta": "相对于基准的系统性风险暴露",
    "max_drawdown": "历史最大回撤，从峰值到谷值的最大跌幅",
    "calmar_ratio": "年化收益与最大回撤的比值",
    "var_95": "95%置信度下的在险价值",
    "cvar_95": "95%置信度下的条件在险价值（期望损失）",
    "information_ratio": "相对于基准的信息比率",
    "tracking_error": "相对于基准的跟踪误差",
    "skewness": "收益分布的偏度，正值表示右偏",
    "kurtosis": "收益分布的峰度，正值表示尖峰厚尾",
    "m_squared": "M²指标，将基金风险调整至市场水平后的收益",
    "win_rate": "正收益日占比",
}

# Demo fund data
DEMO_FUNDS = [
    {"code": "PXSGX", "name": "鹏欣盛冠混合", "category": "hybrid", "annual_return": 0.15, "annual_vol": 0.22, "seed": 42},
    {"code": "HXTT01", "name": "华夏天天利货币", "category": "money_market", "annual_return": 0.025, "annual_vol": 0.005, "seed": 101},
    {"code": "EFT500", "name": "易方达沪深300ETF", "category": "index", "annual_return": 0.08, "annual_vol": 0.20, "seed": 202},
    {"code": "GFQH01", "name": "广发量化先锋", "category": "quant", "annual_return": 0.18, "annual_vol": 0.28, "seed": 303},
    {"code": "ZSFOF1", "name": "中金优选FOF", "category": "fof", "annual_return": 0.10, "annual_vol": 0.12, "seed": 404},
    {"code": "BOCZQ1", "name": "中银纯债", "category": "bond", "annual_return": 0.04, "annual_vol": 0.03, "seed": 505},
    {"code": "SFJJ01", "name": "上投摩根新兴动力", "category": "equity", "annual_return": 0.22, "annual_vol": 0.35, "seed": 606},
    {"code": "XTLC01", "name": "兴全趋势投资", "category": "hybrid", "annual_return": 0.13, "annual_vol": 0.18, "seed": 707},
    {"code": "HJQT01", "name": "华安黄金ETF", "category": "alternative", "annual_return": 0.06, "annual_vol": 0.15, "seed": 808},
    {"code": "YFZQ01", "name": "银华中小盘", "category": "equity", "annual_return": 0.19, "annual_vol": 0.32, "seed": 909},
]

# FL demo institutions
DEMO_INSTITUTIONS = [
    {"id": "INST_001", "name": "鹏华基金", "num_funds": 45},
    {"id": "INST_002", "name": "易方达基金", "num_funds": 62},
    {"id": "INST_003", "name": "华夏基金", "num_funds": 58},
    {"id": "INST_004", "name": "广发基金", "num_funds": 41},
    {"id": "INST_005", "name": "南方基金", "num_funds": 53},
]
