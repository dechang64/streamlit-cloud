"""
TT-OPC / FedCtx Streamlit Hub
===============================
Landing page for all Streamlit Cloud deployments.

Each sub-project is deployed as a separate Streamlit Cloud app
pointing to its subfolder in this repo.
"""
import streamlit as st

st.set_page_config(
    page_title="FedCtx Streamlit Hub",
    page_icon="🚀",
    layout="wide",
)

st.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 32px; border-radius: 16px; color: white; margin-bottom: 24px;">
    <div style="font-size: 2.5rem; font-weight: 800;">🚀 FedCtx Streamlit Hub</div>
    <div style="opacity: 0.95; margin-top: 8px;">一个 Rust 后端，驱动 11 个 AI 项目</div>
</div>
""", unsafe_allow_html=True)

apps = [
    ("📖", "Reading-FL", "坐忘书房 · AI读书会", "reading-fl", "FedAvg + HNSW + KG + 审计链"),
    ("🔍", "Defect-FL", "PCB缺陷联邦检测", "defect-fl", "FedAvg + HNSW + 审计链"),
    ("🤖", "Embodied-FL", "具身智能联邦平台", "embodied-fl", "FedAvg + 审计链"),
    ("💰", "FundFL", "基金风险联邦分析", "fundfl", "HNSW + PageRank"),
    ("🧵", "Embroidery-Agent", "刺绣AI Agent", "embroidery-agent", "FedAvg + 审计链"),
    ("🧬", "Organoid-FL", "类器官智能分析", "organoid-fl", "FedAvg + HNSW + 审计链"),
    ("🏛️", "Mural-Guardian", "石窟壁画智能修复", "mural-guardian", "缺陷检测 + 修复引擎"),
    ("🧪", "AI-BioTech Forum", "AI生物技术论坛", "ai-biotech-forum", "会议网站"),
]

cols = st.columns(4)
for i, (icon, name, desc, folder, features) in enumerate(apps):
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background: var(--secondary-background-color); border-radius: 12px; padding: 16px; margin-bottom: 12px; border-left: 4px solid #8b5cf6;">
            <div style="font-size: 2rem;">{icon}</div>
            <div style="font-size: 1.1rem; font-weight: 700;">{name}</div>
            <div style="font-size: 0.85rem; color: #64748b;">{desc}</div>
            <div style="font-size: 0.75rem; color: #8b5cf6; margin-top: 6px;">{features}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🔧 FedCtx — Federated Semantic Infrastructure")
st.info("""
**FedCtx** 是一个 9.3MB 的 Rust 单二进制文件，提供：
- HNSW 向量搜索 | FedAvg/FedProx/EWA+DP 聚合 | SHA-256 审计链
- 知识图谱 + GraphRAG | 记忆存储 | 混合检索 + PageRank

三种协议接入：gRPC / REST / MCP

GitHub: [dechang64/unified-fl-backend](https://github.com/dechang64/unified-fl-backend)
""")
