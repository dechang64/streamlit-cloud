"""
FundFL — 私募基金联邦学习分析平台
=====================================
Streamlit Cloud Demo

Features:
  - 16项风险指标计算
  - HNSW向量相似搜索
  - 联邦学习模拟 (FedAvg)
  - 审计链 (区块链式)
  - 交互式可视化
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import streamlit as st

from analysis.risk_analyzer import RiskAnalyzer, RiskProfile
from analysis.hnsw_index import HNSWIndex, SearchResult
from analysis.audit_chain import AuditChain
from analysis.fl_engine import FLEngine
from visualization.charts import (
    generate_fund_returns, returns_to_dataframe,
    compute_drawdown_series, risk_radar_data, rolling_sharpe,
)
from utils.constants import (
    FUND_CATEGORIES, RISK_LEVELS, METRIC_DESCRIPTIONS,
    DEMO_FUNDS, DEMO_INSTITUTIONS,
)

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="FundFL — 私募基金联邦学习分析平台",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-card .label {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #38bdf8;
        margin-top: 4px;
    }
    .metric-card .value.positive { color: #34d399; }
    .metric-card .value.negative { color: #f87171; }
    .risk-low { color: #22c55e; }
    .risk-medium { color: #f59e0b; }
    .risk-high { color: #ef4444; }
    .risk-extreme { color: #7c3aed; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        background: #1e293b;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: #0ea5e9 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ───────────────────────────────────────
if "analyzer" not in st.session_state:
    st.session_state.analyzer = RiskAnalyzer()
if "index" not in st.session_state:
    st.session_state.index = HNSWIndex()
if "audit_chain" not in st.session_state:
    st.session_state.audit_chain = AuditChain()
if "fl_engine" not in st.session_state:
    st.session_state.fl_engine = FLEngine()
if "fund_profiles" not in st.session_state:
    st.session_state.fund_profiles = {}
if "returns_cache" not in st.session_state:
    st.session_state.returns_cache = {}


def get_risk_level(volatility: float) -> dict:
    """Get risk level based on volatility."""
    for level, info in RISK_LEVELS.items():
        if volatility < info["max_vol"]:
            return {"level": level, "label": info["label"], "color": info["color"]}
    return {"level": "extreme", "label": "极高风险", "color": "#7c3aed"}


def load_demo_funds():
    """Load demo fund data into index and cache."""
    analyzer = st.session_state.analyzer
    index = st.session_state.index

    for fund in DEMO_FUNDS:
        if fund["code"] in st.session_state.fund_profiles:
            continue

        returns = generate_fund_returns(
            annual_return=fund["annual_return"],
            annual_vol=fund["annual_vol"],
            seed=fund["seed"],
        )
        benchmark = generate_fund_returns(annual_return=0.06, annual_vol=0.15, seed=0)

        profile = analyzer.compute(
            returns, benchmark=benchmark,
            fund_code=fund["code"],
            fund_name=fund["name"],
        )

        st.session_state.fund_profiles[fund["code"]] = profile
        st.session_state.returns_cache[fund["code"]] = returns

        index.add(
            code=fund["code"],
            name=fund["name"],
            vector=profile.to_feature_vector(),
            profile=profile.to_dict(),
        )

        # Audit
        st.session_state.audit_chain.add_block(
            fund_code=fund["code"],
            action="risk_computation",
            data=profile.to_dict(),
        )


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📈 FundFL")
    st.caption("私募基金联邦学习分析平台")
    st.divider()

    st.markdown("### 数据管理")
    if st.button("🔄 加载演示数据", use_container_width=True):
        with st.spinner("加载中..."):
            load_demo_funds()
        st.success(f"已加载 {len(st.session_state.fund_profiles)} 只基金")

    st.divider()
    st.markdown("### 平台状态")
    valid, reason = st.session_state.audit_chain.verify()
    st.metric("审计链长度", st.session_state.audit_chain.length)
    st.metric("向量索引", len(st.session_state.index))
    st.metric("基金数量", len(st.session_state.fund_profiles))
    st.metric("链完整性", "✅ 通过" if valid else "❌ 失败")

    st.divider()
    st.markdown("""
    <div style="font-size: 11px; color: #64748b;">
    FundFL v0.1.0<br>
    Rust + HNSW + gRPC<br>
    MIT License
    </div>
    """, unsafe_allow_html=True)


# ── Main Content ─────────────────────────────────────────────
st.title("📈 FundFL — 私募基金联邦学习分析平台")
st.caption("16项风险指标 · HNSW向量检索 · 联邦学习 · 审计链")

tab_risk, tab_search, tab_fl, tab_audit = st.tabs([
    "📊 风险分析", "🔍 相似搜索", "🤝 联邦学习", "⛓️ 审计链"
])

# ═══════════════════════════════════════════════════════════════
# Tab 1: Risk Analysis
# ═══════════════════════════════════════════════════════════════
with tab_risk:
    st.header("📊 基金风险分析")

    col_input, col_select = st.columns([1, 1])

    with col_input:
        st.subheader("自定义分析")
        fund_code = st.text_input("基金代码", placeholder="如 PXSGX")
        fund_name = st.text_input("基金名称", placeholder="如 鹏欣盛冠混合")
        annual_ret = st.slider("预期年化收益", -0.5, 1.0, 0.12, 0.01, format="%.0f%%")
        annual_vol = st.slider("预期年化波动", 0.01, 0.80, 0.24, 0.01, format="%.0f%%")
        days = st.slider("数据天数", 60, 756, 252, step=1)

        if st.button("🔬 计算风险指标", type="primary", use_container_width=True):
            returns = generate_fund_returns(annual_ret, annual_vol, days)
            benchmark = generate_fund_returns(0.06, 0.15, days, seed=0)
            profile = st.session_state.analyzer.compute(
                returns, benchmark=benchmark,
                fund_code=fund_code or "CUSTOM",
                fund_name=fund_name or "自定义基金",
            )
            st.session_state.fund_profiles[fund_code or "CUSTOM"] = profile
            st.session_state.returns_cache[fund_code or "CUSTOM"] = returns

            st.session_state.audit_chain.add_block(
                fund_code=fund_code or "CUSTOM",
                action="risk_computation",
                data=profile.to_dict(),
            )

            st.session_state.index.add(
                code=fund_code or "CUSTOM",
                name=fund_name or "自定义基金",
                vector=profile.to_feature_vector(),
                profile=profile.to_dict(),
            )
            st.rerun()

    with col_select:
        st.subheader("选择已有基金")
        if st.session_state.fund_profiles:
            selected = st.selectbox(
                "选择基金",
                options=list(st.session_state.fund_profiles.keys()),
                format_func=lambda x: f"{x} — {st.session_state.fund_profiles[x].fund_name}",
            )
            if selected:
                profile = st.session_state.fund_profiles[selected]
                risk = get_risk_level(profile.annual_volatility)
                st.markdown(f"### {profile.fund_name}")
                st.markdown(f"**{selected}** · <span class='risk-{risk['level']}'>{risk['label']}</span>", unsafe_allow_html=True)
        else:
            st.info("请先加载演示数据或计算自定义基金")

    # Display selected profile
    if st.session_state.fund_profiles:
        # Get the latest profile
        latest_code = list(st.session_state.fund_profiles.keys())[-1]
        profile = st.session_state.fund_profiles[latest_code]
        risk = get_risk_level(profile.annual_volatility)

        # Metrics grid
        st.divider()
        st.subheader(f"📋 {profile.fund_name} ({profile.fund_code}) 风险报告")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("年化收益", f"{profile.annual_return:.2%}")
        with m2:
            st.metric("年化波动", f"{profile.annual_volatility:.2%}")
        with m3:
            st.metric("Sharpe", f"{profile.sharpe_ratio:.4f}")
        with m4:
            st.metric("最大回撤", f"{profile.max_drawdown:.2%}")

        m5, m6, m7, m8 = st.columns(4)
        with m5:
            st.metric("Sortino", f"{profile.sortino_ratio:.4f}")
        with m6:
            st.metric("Jensen α", f"{profile.jensen_alpha:.4f}")
        with m7:
            st.metric("Beta", f"{profile.beta:.4f}")
        with m8:
            st.metric("Calmar", f"{profile.calmar_ratio:.4f}")

        m9, m10, m11, m12 = st.columns(4)
        with m9:
            st.metric("VaR(95%)", f"{profile.var_95:.4f}")
        with m10:
            st.metric("CVaR(95%)", f"{profile.cvar_95:.4f}")
        with m11:
            st.metric("M²", f"{profile.m_squared:.4f}")
        with m12:
            st.metric("胜率", f"{profile.win_rate:.1%}")

        # Charts
        if latest_code in st.session_state.returns_cache:
            returns = st.session_state.returns_cache[latest_code]
            df = returns_to_dataframe(returns, profile.fund_name)

            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.subheader("累计净值曲线")
                st.line_chart(df.set_index("date")["NAV"], use_container_width=True)

            with col_chart2:
                st.subheader("回撤曲线")
                dd = compute_drawdown_series(returns)
                dd_df = pd.DataFrame({"date": df["date"], "drawdown": dd * 100})
                st.area_chart(dd_df.set_index("date"), use_container_width=True)

            # Rolling Sharpe
            col_chart3, col_chart4 = st.columns(2)
            with col_chart3:
                st.subheader("滚动Sharpe (60日)")
                rs = rolling_sharpe(returns, 60)
                rs_df = pd.DataFrame({"date": df["date"][59:], "rolling_sharpe": rs[59:]})
                st.line_chart(rs_df.set_index("date"), use_container_width=True)

            with col_chart4:
                st.subheader("收益分布")
                st.histogram(pd.DataFrame({"daily_return": returns}), x="daily_return",
                             bins=50, use_container_width=True)

        # Metric explanations
        with st.expander("📖 指标说明"):
            for metric, desc in METRIC_DESCRIPTIONS.items():
                st.markdown(f"**{metric}**: {desc}")

        # All funds comparison
        if len(st.session_state.fund_profiles) > 1:
            st.divider()
            st.subheader("📊 基金对比")
            comp_data = []
            for code, p in st.session_state.fund_profiles.items():
                comp_data.append({
                    "代码": code,
                    "名称": p.fund_name,
                    "年化收益": f"{p.annual_return:.2%}",
                    "年化波动": f"{p.annual_volatility:.2%}",
                    "Sharpe": f"{p.sharpe_ratio:.4f}",
                    "Sortino": f"{p.sortino_ratio:.4f}",
                    "最大回撤": f"{p.max_drawdown:.2%}",
                    "胜率": f"{p.win_rate:.1%}",
                })
            st.dataframe(comp_data, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# Tab 2: Similar Search
# ═══════════════════════════════════════════════════════════════
with tab_search:
    st.header("🔍 HNSW 向量相似搜索")
    st.caption("基于16维风险特征向量的余弦相似度搜索")

    if not st.session_state.fund_profiles:
        st.info("请先在「风险分析」页面加载演示数据")
    else:
        col_search, col_params = st.columns([2, 1])

        with col_search:
            query_fund = st.selectbox(
                "查询基金",
                options=list(st.session_state.fund_profiles.keys()),
                format_func=lambda x: f"{x} — {st.session_state.fund_profiles[x].fund_name}",
            )

        with col_params:
            top_k = st.slider("返回数量", 1, 10, 5)
            ef_search = st.slider("ef_search", 10, 200, 50, step=10)

        if query_fund:
            profile = st.session_state.fund_profiles[query_fund]
            query_vec = profile.to_feature_vector()

            st.session_state.index.ef_search = ef_search
            results = st.session_state.index.search(query_vec, k=top_k + 1)  # +1 because self is included

            # Filter out self
            results = [r for r in results if r.fund_code != query_fund][:top_k]

            if results:
                st.subheader(f"与 {profile.fund_name} 最相似的基金")

                # Similarity table
                table_data = []
                for i, r in enumerate(results):
                    sim = 1.0 - r.distance
                    table_data.append({
                        "排名": i + 1,
                        "基金代码": r.fund_code,
                        "基金名称": r.fund_name,
                        "余弦相似度": f"{sim:.4f}",
                        "距离": f"{r.distance:.4f}",
                        "Sharpe": f"{r.sharpe:.4f}",
                        "年化收益": f"{r.annual_return:.2%}",
                    })
                st.dataframe(table_data, use_container_width=True, hide_index=True)

                # Visual comparison
                if len(results) >= 2:
                    st.subheader("风险特征对比")
                    query_profile = st.session_state.fund_profiles[query_fund]
                    metrics_to_compare = [
                        ("年化收益", "annual_return", "%"),
                        ("年化波动", "annual_volatility", "%"),
                        ("Sharpe", "sharpe_ratio", ""),
                        ("Sortino", "sortino_ratio", ""),
                        ("最大回撤", "max_drawdown", "%"),
                        ("胜率", "win_rate", "%"),
                    ]

                    compare_data = {"指标": [m[0] for m in metrics_to_compare]}
                    compare_data[query_fund] = []
                    for r in results[:3]:
                        compare_data[r.fund_code] = []

                    for metric_name, attr, fmt in metrics_to_compare:
                        val = getattr(query_profile, attr, 0)
                        compare_data[query_fund].append(f"{val:.2%}" if fmt == "%" else f"{val:.4f}")
                        for r in results[:3]:
                            rp = st.session_state.fund_profiles.get(r.fund_code)
                            if rp:
                                val = getattr(rp, attr, 0)
                                compare_data[r.fund_code].append(f"{val:.2%}" if fmt == "%" else f"{val:.4f}")
                            else:
                                compare_data[r.fund_code].append("—")

                    st.dataframe(pd.DataFrame(compare_data), use_container_width=True, hide_index=True)

                # Audit
                st.session_state.audit_chain.add_block(
                    fund_code=query_fund,
                    action="similarity_search",
                    data={"query": query_fund, "results": [r.fund_code for r in results], "k": top_k},
                )
            else:
                st.warning("未找到相似基金")

        # Index stats
        with st.expander("📊 索引统计"):
            stats = st.session_state.index.stats()
            st.json(stats)


# ═══════════════════════════════════════════════════════════════
# Tab 3: Federated Learning
# ═══════════════════════════════════════════════════════════════
with tab_fl:
    st.header("🤝 联邦学习模拟")
    st.caption("跨机构基金风险模型联邦训练 — 仅共享聚合风险特征，不泄露原始收益数据")

    col_config, col_run = st.columns([1, 2])

    with col_config:
        st.subheader("训练配置")
        num_rounds = st.slider("训练轮数", 1, 50, 10)
        lr = st.slider("学习率", 0.001, 0.1, 0.01, 0.001, format="%.3f")
        hidden_dim = st.selectbox("隐藏层维度", [16, 32, 64, 128], index=1)

        st.subheader("参与机构")
        selected_inst = st.multiselect(
            "选择机构",
            options=[inst["name"] for inst in DEMO_INSTITUTIONS],
            default=[inst["name"] for inst in DEMO_INSTITUTIONS[:3]],
        )

    with col_run:
        if st.button("🚀 开始联邦训练", type="primary", use_container_width=True):
            # Reset engine
            st.session_state.fl_engine = FLEngine(
                input_dim=16, hidden_dim=hidden_dim, learning_rate=lr,
            )

            # Add clients
            for inst in DEMO_INSTITUTIONS:
                if inst["name"] in selected_inst:
                    st.session_state.fl_engine.add_client(
                        client_id=inst["id"],
                        institution_name=inst["name"],
                        num_funds=inst["num_funds"],
                    )

            # Progress bar
            progress = st.progress(0, text="训练中...")
            status = st.empty()

            all_results = []
            for i in range(num_rounds):
                result = st.session_state.fl_engine.run_round()
                all_results.append(result)
                progress.progress((i + 1) / num_rounds, text=f"Round {i+1}/{num_rounds}")
                time.sleep(0.1)  # Visual feedback

            status.success(f"✅ 训练完成！{num_rounds}轮，{len(selected_inst)}个机构参与")

            st.session_state.fl_results = all_results

            # Audit
            st.session_state.audit_chain.add_block(
                fund_code="FL_GLOBAL",
                action="federated_training",
                data={
                    "rounds": num_rounds,
                    "clients": len(selected_inst),
                    "lr": lr,
                    "final_loss": all_results[-1].global_loss,
                },
            )

    # Display results
    if "fl_results" in st.session_state and st.session_state.fl_results:
        results = st.session_state.fl_results

        st.divider()
        st.subheader("📈 训练曲线")

        col_loss, conv = st.columns(2)
        with col_loss:
            loss_df = pd.DataFrame({
                "round": [r.round_num for r in results],
                "global_loss": [r.global_loss for r in results],
            })
            st.line_chart(loss_df.set_index("round"), use_container_width=True)
            st.caption("全局损失函数")

        with conv:
            conv_df = pd.DataFrame({
                "round": [r.round_num for r in results],
                "convergence": [r.convergence_delta for r in results],
            })
            st.line_chart(conv_df.set_index("round"), use_container_width=True)
            st.caption("收敛速度 (参数变化量)")

        # Client contributions
        st.subheader("🏛️ 机构贡献")
        if results:
            last = results[-1]
            contrib_data = []
            for c in last.client_contributions:
                contrib_data.append({
                    "机构": c["institution"],
                    "权重": c["weight"],
                    "本地更新范数": f"{c['norm_delta']:.6f}",
                })
            st.dataframe(contrib_data, use_container_width=True, hide_index=True)

        # Global model summary
        with st.expander("🧠 全局模型摘要"):
            summary = st.session_state.fl_engine.get_global_model_summary()
            st.json(summary)

        # Privacy note
        st.info("🔒 **隐私保护**: 各机构仅上传聚合后的风险特征向量，原始日收益率数据从未离开本地。联邦聚合采用加权平均 (FedAvg)，无法反推个体数据。")


# ═══════════════════════════════════════════════════════════════
# Tab 4: Audit Chain
# ═══════════════════════════════════════════════════════════════
with tab_audit:
    st.header("⛓️ 审计链")
    st.caption("区块链式不可篡改审计日志 — 每次风险计算、搜索、训练均上链")

    # Verify
    valid, reason = st.session_state.audit_chain.verify()
    if valid:
        st.success(f"✅ 链完整性验证通过 | 长度: {st.session_state.audit_chain.length}")
    else:
        st.error(f"❌ 验证失败: {reason}")

    # Filter
    col_filter, col_view = st.columns([1, 3])

    with col_filter:
        filter_fund = st.selectbox(
            "筛选基金",
            options=["全部"] + list(st.session_state.fund_profiles.keys()),
        )
        filter_action = st.selectbox(
            "筛选操作",
            options=["全部", "risk_computation", "similarity_search", "federated_training"],
        )

    with col_view:
        blocks = st.session_state.audit_chain.get_blocks(
            fund_code=filter_fund if filter_fund != "全部" else None,
        )
        if filter_action != "全部":
            blocks = [b for b in blocks if b.action == filter_action]

        if blocks:
            # Display as table
            chain_data = []
            for b in blocks:
                chain_data.append({
                    "#": b.index,
                    "时间": time.strftime("%m-%d %H:%M:%S", time.localtime(b.timestamp)),
                    "基金": b.fund_code,
                    "操作": b.action,
                    "数据哈希": b.data_hash[:16] + "...",
                    "区块哈希": b.block_hash[:16] + "...",
                    "前哈希": b.prev_hash[:16] + "...",
                })
            st.dataframe(chain_data, use_container_width=True, hide_index=True)

            # Block detail
            if blocks:
                selected_block = st.selectbox(
                    "查看区块详情",
                    options=range(len(blocks)),
                    format_func=lambda i: f"Block #{blocks[i].index}",
                )
                if selected_block is not None:
                    block = blocks[selected_block]
                    st.json(block.to_dict())
        else:
            st.info("暂无审计记录")

    # Chain visualization
    if len(blocks) > 1:
        st.subheader("🔗 链结构可视化")
        st.markdown("""
        <div style="display: flex; gap: 4px; overflow-x: auto; padding: 20px 0;">
        """ + "".join([
            f'<div style="min-width: 80px; background: #1e293b; border: 1px solid #334155; '
            f'border-radius: 8px; padding: 8px; text-align: center; font-size: 11px;">'
            f'<div style="color: #38bdf8;">#{b.index}</div>'
            f'<div style="color: #64748b;">{b.action[:8]}</div>'
            f'<div style="color: #475569;">{b.fund_code[:6]}</div>'
            f'</div>'
            f'<div style="display: flex; align-items: center; color: #334155;">→</div>'
            for b in blocks[-10:]  # Show last 10
        ]) + "</div>", unsafe_allow_html=True)
