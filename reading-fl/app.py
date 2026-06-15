"""
Reading-FL — 坐忘·阅读联邦平台
Streamlit Cloud App

Features:
    - Write reading reflections
    - AI emotion classification (6 emotions)
    - Reader matching via HNSW
    - Resonance detection (high-impact excerpts)
    - Federated learning simulation across campuses
    - Audit chain provenance verification
"""

import streamlit as st
import numpy as np
import time
import os
import hashlib

from core.fl_engine import ReadingFLEngine
from matching.reader_matcher import ReaderMatcher
from matching.resonance_detector import ResonanceDetector
from matching.hnsw_index import HNSWIndex
from audit.provenance import DataProvenance
from audit.chain import AuditChain
from data.reflection import (
    EMOTION_LABELS, EMOTION_LABEL_CN,
    Reflection, BookExcerpt, ReaderProfile,
)
from data.generator import SyntheticDataGenerator as ReadingDataGenerator, BOOK_CORPUS


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Reading-FL · 坐忘·阅读",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .main-header h1 {
        font-size: 2.5rem;
        color: #7c3aed;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #666;
    }
    .emotion-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin: 0.25rem;
        font-weight: 600;
    }
    .resonance-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f59e0b;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "matcher" not in st.session_state:
    st.session_state.matcher = ReaderMatcher(embedding_dim=64)
if "resonance" not in st.session_state:
    st.session_state.resonance = ResonanceDetector()
if "provenance" not in st.session_state:
    st.session_state.provenance = DataProvenance()
if "reflections" not in st.session_state:
    st.session_state.reflections = []
if "reader_profiles" not in st.session_state:
    st.session_state.reader_profiles = {}


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/open-book.png", width=64)
    st.title("坐忘·阅读")
    st.caption("Reading-FL v0.1")

    st.divider()

    st.subheader("👤 读者信息")
    reader_id = st.text_input("读者ID", value="reader_001", placeholder="e.g., reader_001")
    campus = st.selectbox("校区", ["北京大学", "清华大学", "复旦大学", "浙江大学", "南京大学"], index=0)

    st.divider()

    st.subheader("📊 平台统计")
    matcher_stats = st.session_state.matcher.get_stats()
    st.metric("注册读者", matcher_stats["n_readers"])
    st.metric("已确认匹配", matcher_stats["n_confirmed_matches"])
    st.metric("待处理请求", matcher_stats["n_pending_requests"])

    resonance_stats = st.session_state.resonance.get_stats()
    st.metric("追踪书摘", resonance_stats["n_excerpts_tracked"])


# ============================================================
# Main Page
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>📖 坐忘·阅读</h1>
    <p>基于联邦学习的跨校区阅读社区 — 感悟不出校，共鸣跨校园</p>
</div>
""", unsafe_allow_html=True)

tab_write, tab_match, tab_resonance, tab_fl, tab_audit, tab_about = st.tabs([
    "✍️ 写感悟", "👥 读者匹配", "💫 共鸣检测", "🤝 联邦学习", "🔒 审计链", "ℹ️ 关于"
])


# ============================================================
# Tab 1: Write Reflection
# ============================================================
with tab_write:
    st.header("写下你的阅读感悟")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📚 选择书摘")
        book_title = st.text_input("书名", value="百年孤独", placeholder="输入书名")
        author = st.text_input("作者", value="马尔克斯", placeholder="输入作者")
        excerpt = st.text_area(
            "书摘内容",
            value="多年以后，面对行刑队，奥雷里亚诺·布恩迪亚上校将会回想起父亲带他去见识冰块的那个遥远的下午。",
            height=100,
        )

        st.subheader("✍️ 你的感悟")
        reflection_text = st.text_area(
            "阅读感悟",
            placeholder="写下你读完这段话的感受...",
            height=150,
        )

        st.subheader("💭 情感标签")
        selected_emotions = st.multiselect(
            "选择情感（可多选）",
            EMOTION_LABELS,
            format_func=lambda e: f"{EMOTION_LABEL_CN[e]} ({e})",
            default=["moved"],
        )

        depth = st.slider("感悟深度", 0.0, 1.0, value=0.7, step=0.1)

        submit = st.button("📝 提交感悟", type="primary", use_container_width=True)

        if submit and reflection_text:
            # Create reflection
            refl = Reflection(
                reader_id=reader_id,
                excerpt=BookExcerpt(
                    book_id=hashlib.md5(book_title.encode()).hexdigest()[:8],
                    book_title=book_title,
                    author=author,
                    paragraph_id="p1",
                    text=excerpt,
                ),
                reflection_text=reflection_text,
                emotions=selected_emotions,
                reflection_depth=depth,
                campus=campus,
            )

            st.session_state.reflections.append(refl)

            # Update matcher
            embedding = np.random.RandomState(hash(reader_id) % 2**31).randn(64).astype(np.float32)
            st.session_state.matcher.add_reader(reader_id, embedding, metadata={"campus": campus})

            # Update resonance
            st.session_state.resonance.record_signal(
                excerpt_id=refl.excerpt.book_id,
                reader_id=reader_id,
                campus=campus,
                depth=depth,
                emotion=selected_emotions[0] if selected_emotions else "calm",
                reflection_length=len(reflection_text),
            )

            # Update profile
            if reader_id not in st.session_state.reader_profiles:
                st.session_state.reader_profiles[reader_id] = ReaderProfile(
                    reader_id=reader_id, campus=campus,
                )
            st.session_state.reader_profiles[reader_id].update(refl)

            st.success(f"✅ 感悟已提交！情感: {', '.join(EMOTION_LABEL_CN[e] for e in selected_emotions)}")

    with col2:
        st.subheader("📖 最近感悟")
        if st.session_state.reflections:
            for refl in reversed(st.session_state.reflections[-5:]):
                st.markdown(f"""
                <div style="background:#f8f9fa; padding:1rem; border-radius:8px; margin-bottom:0.5rem;">
                    <div style="font-weight:bold; color:#7c3aed;">{refl.excerpt.book_title}</div>
                    <div style="font-size:0.85rem; color:#666; margin:0.25rem 0;">{refl.excerpt.text[:50]}...</div>
                    <div style="font-size:0.9rem;">{refl.reflection_text[:80]}...</div>
                    <div style="margin-top:0.5rem;">
                        {' '.join(f'<span class="emotion-badge" style="background:#ede9fe; color:#7c3aed;">{EMOTION_LABEL_CN[e]}</span>' for e in refl.emotions)}
                    </div>
                    <div style="font-size:0.75rem; color:#999; margin-top:0.25rem;">{refl.reader_id} · {refl.campus}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("还没有感悟，写下第一条吧！")


# ============================================================
# Tab 2: Reader Matching
# ============================================================
with tab_match:
    st.header("读者匹配")
    st.caption("基于HNSW向量检索，找到阅读品味相似的读者（双盲匹配）")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 发现相似读者")
        k = st.slider("推荐数量", 1, 20, value=5)

        if st.button("🔎 搜索相似读者", use_container_width=True):
            if st.session_state.matcher.get_stats()["n_readers"] > 1:
                # Generate demo readers if needed
                rng = np.random.RandomState(42)
                for i in range(2, 12):
                    rid = f"reader_{i:03d}"
                    if rid not in st.session_state.matcher.profiles:
                        emb = rng.randn(64).astype(np.float32)
                        campus_list = ["北京大学", "清华大学", "复旦大学", "浙江大学", "南京大学"]
                        st.session_state.matcher.add_reader(
                            rid, emb,
                            metadata={"campus": campus_list[i % len(campus_list)]}
                        )

                query = st.session_state.matcher.profiles.get(reader_id)
                if query is not None:
                    results = st.session_state.matcher.find_similar(query, k=k)
                    for dist, rid, meta in results:
                        if rid == reader_id:
                            continue
                        campus_name = meta.get("campus", "未知")
                        similarity = max(0, 1 - dist)
                        st.markdown(f"""
                        <div style="background:#f0fdf4; padding:1rem; border-radius:8px; margin-bottom:0.5rem; border-left:4px solid #22c55e;">
                            <div style="font-weight:bold;">👤 {rid}</div>
                            <div style="font-size:0.85rem; color:#666;">📍 {campus_name}</div>
                            <div style="font-size:0.9rem;">相似度: <strong>{similarity:.1%}</strong></div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.warning("需要至少2位读者才能匹配。请先提交感悟。")

    with col2:
        st.subheader("🤝 匹配操作")
        target_reader = st.text_input("目标读者ID", value="reader_002")

        if st.button("💌 发送匹配请求", use_container_width=True):
            if target_reader:
                # Add target if not exists
                if target_reader not in st.session_state.matcher.profiles:
                    rng = np.random.RandomState(hash(target_reader) % 2**31)
                    emb = rng.randn(64).astype(np.float32)
                    st.session_state.matcher.add_reader(target_reader, emb, metadata={"campus": "其他校区"})

                result = st.session_state.matcher.send_match_request(reader_id, target_reader)
                if result:
                    st.success(f"🎉 与 {target_reader} 匹配成功！")
                else:
                    st.info(f"⏳ 已向 {target_reader} 发送匹配请求，等待对方确认...")

        st.divider()
        st.subheader("📬 待处理请求")
        pending = st.session_state.matcher.get_pending_requests(reader_id)
        if pending:
            for pid in pending:
                if st.button(f"✅ 接受 {pid}", key=f"accept_{pid}"):
                    st.session_state.matcher.confirm_match(pid, reader_id)
                    st.success(f"🎉 与 {pid} 匹配成功！")
                    st.rerun()
        else:
            st.info("暂无待处理请求")

        st.subheader("✅ 已确认匹配")
        confirmed = st.session_state.matcher.get_confirmed_matches(reader_id)
        if confirmed:
            for mid in confirmed:
                st.markdown(f"🤝 {mid}")
        else:
            st.info("暂无已确认匹配")


# ============================================================
# Tab 3: Resonance Detection
# ============================================================
with tab_resonance:
    st.header("共鸣检测")
    st.caption("发现那些打动最多人的书摘 — 坐忘·咖候选")

    # Generate demo data if needed
    if st.button("🎲 生成演示数据", use_container_width=True):
        gen = ReadingDataGenerator(seed=42)
        rng = np.random.RandomState(42)
        campuses = ["北京大学", "清华大学", "复旦大学"]
        for book in BOOK_CORPUS[:5]:
            for excerpt_text in book["excerpts"]:
                eid = hashlib.md5(excerpt_text.encode()).hexdigest()[:8]
                for campus in campuses:
                    for j in range(rng.randint(3, 8)):
                        rid = f"demo_{campus}_{j}"
                        depth = rng.uniform(0.2, 1.0)
                        emotion = rng.choice(EMOTION_LABELS)
                        st.session_state.resonance.record_signal(
                            excerpt_id=eid,
                            reader_id=rid,
                            campus=campus,
                            depth=depth,
                            emotion=emotion,
                            reflection_length=rng.randint(20, 500),
                        )
        st.success(f"✅ 已生成演示数据！追踪 {st.session_state.resonance.get_stats()['n_excerpts_tracked']} 条书摘")

    st.divider()

    # Top resonant excerpts
    st.subheader("💫 高共鸣书摘 TOP 10")
    top = st.session_state.resonance.get_top_resonant(k=10, min_reflections=1)
    if top:
        for i, (eid, score, stats) in enumerate(top):
            st.markdown(f"""
            <div class="resonance-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div style="font-size:1.2rem; font-weight:bold; color:#92400e;">#{i+1} 共鸣指数: {score:.2f}</div>
                        <div style="font-size:0.85rem; color:#78716c;">书摘ID: {eid} | 感悟数: {stats['n_reflections']} | 校区数: {stats['n_campuses']}</div>
                    </div>
                </div>
                <div style="margin-top:0.5rem; font-size:0.9rem;">
                    平均深度: {stats['avg_depth']:.2f} | 一致性: {stats['consistency']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无数据。点击「生成演示数据」开始体验。")

    st.divider()
    st.subheader("☕ 坐忘·咖候选")
    coffee = st.session_state.resonance.get_coffee_sleeve_candidates(k=5)
    if coffee:
        for i, (eid, score, stats) in enumerate(coffee):
            st.markdown(f"""
            <div style="background:linear-gradient(135deg, #fef3c7, #fde68a); padding:1rem; border-radius:12px; margin:0.5rem 0; border-left:4px solid #d97706;">
                <div style="font-weight:bold; color:#92400e;">☕ 候选 #{i+1} — 共鸣指数: {score:.2f}</div>
                <div style="font-size:0.85rem;">感悟数: {stats['n_reflections']} | 跨校区: {stats['n_campuses']} | 深度: {stats['avg_depth']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("暂无符合条件的候选（需跨2+校区共鸣且深度>0.4）")


# ============================================================
# Tab 4: Federated Learning
# ============================================================
with tab_fl:
    st.header("联邦学习模拟")
    st.caption("跨校区情感分类模型协同训练 — 感悟数据不出校")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("⚙️ 训练参数")
        n_rounds = st.number_input("联邦轮数", value=10, min_value=1, max_value=50)
        n_clients = st.number_input("校区数量", value=3, min_value=2, max_value=10)
        lr = st.number_input("学习率", value=0.001, min_value=0.0001, max_value=0.1, format="%.4f")
        local_epochs = st.number_input("本地轮数", value=2, min_value=1, max_value=10)
        n_samples = st.number_input("每校区样本数", value=300, min_value=50, max_value=5000, step=50)

        run_fl = st.button("🚀 开始训练", type="primary", use_container_width=True)

    with col2:
        if run_fl:
            with st.spinner("🤝 跨校区联邦训练中..."):
                engine = ReadingFLEngine(
                    input_dim=64,
                    num_emotions=6,
                    hidden_dim=128,
                    lr=lr,
                    local_epochs=local_epochs,
                )

                progress = st.empty()
                def on_progress(rnd, metrics):
                    progress.text(f"轮次 {rnd}/{n_rounds} — 验证准确率: {metrics['val_acc']:.1%}")

                # Generate synthetic data
                rng = np.random.RandomState(42)
                features = rng.randn(n_samples * n_clients, 64).astype(np.float32)
                labels = rng.randint(0, 6, size=n_samples * n_clients)

                history = engine.run(
                    features=features,
                    labels=labels,
                    n_clients=n_clients,
                    rounds=n_rounds,
                    progress_callback=on_progress,
                )

            st.success("✅ 训练完成！")

            final = history[-1]
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("最终训练准确率", f"{final['avg_train_acc']:.1%}")
            with m2:
                st.metric("最终验证准确率", f"{final['val_acc']:.1%}")
            with m3:
                st.metric("最终验证损失", f"{final['val_loss']:.3f}")
            with m4:
                st.metric("联邦轮数", n_rounds)

            # Convergence chart
            st.subheader("📈 收敛曲线")
            rounds = [h["round"] for h in history]
            chart_data = {
                "轮次": rounds,
                "训练准确率": [h["avg_train_acc"] for h in history],
                "验证准确率": [h["val_acc"] for h in history],
                "训练损失": [h["avg_train_loss"] for h in history],
                "验证损失": [h["val_loss"] for h in history],
            }
            st.line_chart(chart_data, x="轮次", y=["训练准确率", "验证准确率"])

            # Emotion distribution
            st.subheader("🎭 情感分类分布")
            preds, probs = engine.predict(features[:20])
            emotion_counts = {}
            for p in preds:
                emotion = ReadingFLEngine.EMOTIONS[p]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            for emotion, count in emotion_counts.items():
                cn = EMOTION_LABEL_CN.get(emotion, emotion)
                st.markdown(f"**{cn}** ({emotion}): {count}")
        else:
            st.info("配置参数后点击「开始训练」")


# ============================================================
# Tab 5: Audit Chain
# ============================================================
with tab_audit:
    st.header("审计链存证")
    st.caption("区块链式感悟真实性验证")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📝 添加存证")
        audit_text = st.text_area("感悟文本", height=100, placeholder="输入要存证的感悟...")
        audit_reader = st.text_input("读者ID", value=reader_id)

        if st.button("🔗 上链存证", type="primary", use_container_width=True):
            if audit_text:
                chain = st.session_state.provenance.chain
                block = chain.add_block(
                    data={"text": audit_text, "reader": audit_reader},
                    validator=campus,
                )
                st.success(f"✅ 已存证！区块 #{block.index}")
                st.json({
                    "区块索引": block.index,
                    "时间戳": block.timestamp,
                    "数据哈希": block.data_hash[:16] + "...",
                    "前块哈希": block.prev_hash[:16] + "...",
                    "区块哈希": block.block_hash[:16] + "...",
                    "验证者": block.validator,
                })

    with col2:
        st.subheader("🔍 验证链完整性")
        chain = st.session_state.provenance.chain
        is_valid = chain.verify_chain()
        stats = chain.get_stats()

        if is_valid:
            st.success("✅ 审计链完整，所有区块验证通过")
        else:
            st.error("❌ 审计链验证失败！")

        st.metric("链长度", stats["chain_length"])
        st.metric("已存证感悟", stats["n_reflections"])
        st.metric("参与校区", len(stats["validators"]))

        # Provenance check
        st.divider()
        st.subheader("🛡️ 真实性验证")
        if audit_text:
            check = st.session_state.provenance.verify(
                reflection_text=audit_text,
                reader_id=audit_reader,
            )
            st.metric("真实性评分", f"{check.score:.1%}")
            st.json({"检查项": check.checks, "详情": check.details})


# ============================================================
# Tab 6: About
# ============================================================
with tab_about:
    st.header("关于 坐忘·阅读")

    st.markdown("""
    ### 📖 项目简介

    **坐忘·阅读 (Reading-FL)** 是一个基于联邦学习的跨校区阅读社区平台。读者在各自校区写下阅读感悟，AI分析情感与共鸣，跨校区匹配阅读品味相似的读者。

    核心理念：**感悟不出校，共鸣跨校园**。

    ### 🔬 核心技术

    | 技术 | 用途 |
    |------|------|
    | FedAvg | 跨校区情感分类协同训练 |
    | HNSW | 读者品味相似检索 |
    | 共鸣检测 | 高影响力书摘发现 |
    | 审计链 | 感悟真实性存证 |
    | 多信号验证 | 灯行为+时序+文本+链上哈希 |

    ### 🎭 情感标签

    | 标签 | 含义 |
    |------|------|
    | 感动 (moved) | 深受触动 |
    | 思考 (thinking) | 引发深思 |
    | 共鸣 (resonance) | 强烈认同 |
    | 困惑 (confused) | 不太理解 |
    | 反对 (disagree) | 不同观点 |
    | 平静 (calm) | 理性客观 |

    ### ☕ 坐忘·咖

    高共鸣书摘自动推荐印制在咖啡杯套上，让好文字在校园间流动。

    ### 📄 开源协议

    Apache-2.0 | [GitHub](https://github.com/dechang64)
    """)
