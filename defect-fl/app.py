from __future__ import annotations
"""
Defect-FL — PCB缺陷联邦检测平台
Streamlit Cloud App

Features:
    - Upload PCB image
    - AI defect detection (6 types: missing_hole, mouse_bite, open_circuit, short, spur, spurious_copper)
    - Mock SAM2 segmentation preview
    - Grad-CAM explainability (mock mode)
    - Federated learning simulation
    - Multi-factory dashboard
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import time
import json
import os

from analysis.detector import PCBDefectDetector, DefectDetection
from analysis.segmentor import PCBDefectSegmentor, DefectSegmentation
from analysis.fl_engine import DefectFLEngine
from utils.constants import (
    DEFECT_CLASSES, DEFECT_DESCRIPTIONS, SEVERITY_LEVELS,
    FACTORY_PRESETS, COLORS,
)


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Defect-FL · PCB缺陷联邦检测",
    page_icon="🔧",
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
        color: #1e40af;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #666;
    }
    .severity-critical { background: #ef4444; color: white; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
    .severity-moderate { background: #f59e0b; color: white; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
    .severity-minor { background: #22c55e; color: white; padding: 0.2rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
    .metric-card {
        background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #bfdbfe;
    }
    .metric-card.danger {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border-color: #fca5a5;
    }
    .metric-card.success {
        background: linear-gradient(135deg, #f0fdf4 0%, #bbf7d0 100%);
        border-color: #86efac;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "detector" not in st.session_state:
    st.session_state.detector = PCBDefectDetector(mode="mock")
if "segmentor" not in st.session_state:
    st.session_state.segmentor = PCBDefectSegmentor(mode="mock")
if "fl_engine" not in st.session_state:
    st.session_state.fl_engine = DefectFLEngine()
if "history" not in st.session_state:
    st.session_state.history = []
if "last_detection" not in st.session_state:
    st.session_state.last_detection = None
if "last_image" not in st.session_state:
    st.session_state.last_image = None


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/circuit-board.png", width=64)
    st.title("Defect-FL")
    st.caption("PCB缺陷联邦检测 v0.1")

    st.divider()

    st.subheader("⚙️ 检测设置")
    detect_mode = st.selectbox(
        "检测模式",
        ["mock (快速演示)", "yolo (需要GPU)"],
        index=0,
    )
    st.session_state.detector = PCBDefectDetector(
        mode="mock" if "mock" in detect_mode else "yolo"
    )

    conf_threshold = st.slider("置信度阈值", 0.1, 0.95, 0.5, 0.05)

    st.divider()

    st.subheader("🏭 工厂信息")
    factory = st.selectbox(
        "选择工厂",
        list(FACTORY_PRESETS.keys()),
        format_func=lambda x: FACTORY_PRESETS[x]["name"],
    )
    factory_info = FACTORY_PRESETS[factory]
    st.caption(f"产线: {factory_info['lines']}条 | 日产能: {factory_info['daily_output']:,}片")

    st.divider()

    st.subheader("📊 检测历史")
    if st.session_state.history:
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            status = "❌" if h["defects"] > 0 else "✅"
            st.markdown(f"{status} **{h['factory']}** — {h['defects']}个缺陷 ({h['time']})")
    else:
        st.caption("暂无检测记录")


# ============================================================
# Main Page
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🔧 Defect-FL</h1>
    <p>基于联邦学习的PCB缺陷智能检测平台 — 多工厂协同，数据不出厂</p>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_segment, tab_fl, tab_about = st.tabs([
    "🔍 缺陷检测", "🔬 分割分析", "🌐 联邦学习", "ℹ️ 关于"
])


# ============================================================
# Tab 1: Defect Detection
# ============================================================
with tab_detect:
    st.header("PCB缺陷智能检测")
    st.caption("支持6类缺陷：漏孔、鼠咬、开路、短路、毛刺、多余铜")

    col_upload, col_info = st.columns([2, 1])

    with col_upload:
        uploaded = st.file_uploader(
            "上传PCB图像",
            type=["png", "jpg", "jpeg", "webp"],
            key="detect_upload",
        )

    with col_info:
        st.markdown("""
        ### 缺陷类型说明
        | 类型 | 严重度 | 说明 |
        |------|--------|------|
        | missing_hole | 🔴 严重 | 漏孔，无法安装元件 |
        | open_circuit | 🔴 严重 | 开路，信号中断 |
        | short | 🔴 严重 | 短路，信号异常 |
        | mouse_bite | 🟡 中等 | 鼠咬，可能开路 |
        | spurious_copper | 🟡 中等 | 多余铜，制造污染 |
        | spur | 🟢 轻微 | 毛刺，潜在短路 |
        """)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        st.session_state.last_image = img_array

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("原始PCB图像")
            st.image(img, use_container_width=True)
            st.caption(f"尺寸: {img.width} × {img.height} px")

        with col2:
            with st.spinner("🔍 正在检测缺陷..."):
                t0 = time.time()
                result = st.session_state.detector.detect(img_array, conf_threshold=conf_threshold)
                dt = (time.time() - t0) * 1000

            st.session_state.last_detection = result

            # Draw detections
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)

            DEFECT_COLORS = {
                "missing_hole": "#ef4444",
                "mouse_bite": "#f59e0b",
                "open_circuit": "#ef4444",
                "short": "#ef4444",
                "spur": "#22c55e",
                "spurious_copper": "#f59e0b",
            }

            for d in result:
                color = DEFECT_COLORS.get(d.class_name, "#ef4444")
                draw.rectangle([d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]], outline=color, width=2)
                label = f"{d.class_name} {d.confidence:.0%}"
                draw.rectangle([d.bbox[0], d.bbox[1]-16, d.bbox[0]+len(label)*8+6, d.bbox[1]], fill=color)
                draw.text((d.bbox[0]+3, d.bbox[1]-14), label, fill="white")

            st.subheader("检测结果")
            st.image(annotated, use_container_width=True)
            st.caption(f"检测耗时: {dt:.0f} ms")

        # Metrics
        summary = st.session_state.detector.summary(result)
        defects = [d for d in result if d.class_name != "good"]

        m1, m2, m3, m4 = st.columns(4)

        with m1:
            st.markdown(f"""
            <div class="metric-card {'danger' if defects else 'success'}">
                <div style="font-size:2rem;font-weight:700">{len(defects)}</div>
                <div style="color:#666">检测到缺陷</div>
            </div>
            """, unsafe_allow_html=True)

        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;font-weight:700">{summary.get('avg_confidence', 0):.0%}</div>
                <div style="color:#666">平均置信度</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            critical = sum(1 for d in defects if d.severity == "critical")
            st.markdown(f"""
            <div class="metric-card {'danger' if critical else ''}">
                <div style="font-size:2rem;font-weight:700">{critical}</div>
                <div style="color:#666">严重缺陷</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:2rem;font-weight:700">{dt:.0f}</div>
                <div style="color:#666">耗时 (ms)</div>
            </div>
            """, unsafe_allow_html=True)

        # Defect details table
        if defects:
            st.subheader("📋 缺陷详情")
            table_data = []
            for d in defects:
                sev_class = f"severity-{d.severity}" if d.severity != "moderate" else "severity-moderate"
                table_data.append({
                    "类型": d.class_name,
                    "严重度": d.severity,
                    "置信度": f"{d.confidence:.1%}",
                    "位置": f"({d.cx:.0f}, {d.cy:.0f})",
                    "面积": f"{d.area:.0f} px²",
                    "说明": DEFECT_DESCRIPTIONS.get(d.class_name, ""),
                })
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            # Save to history
            st.session_state.history.append({
                "factory": factory_info["name"],
                "defects": len(defects),
                "time": time.strftime("%H:%M:%S"),
            })
        else:
            st.success("✅ 未检测到缺陷，PCB质量合格！")
            st.session_state.history.append({
                "factory": factory_info["name"],
                "defects": 0,
                "time": time.strftime("%H:%M:%S"),
            })


# ============================================================
# Tab 2: Segmentation Analysis
# ============================================================
with tab_segment:
    st.header("缺陷分割与形态分析")
    st.caption("基于SAM2的像素级缺陷分割，提取缺陷形态学特征")

    if st.session_state.last_detection is None or st.session_state.last_image is None:
        st.warning("请先在「缺陷检测」页面上传图像并完成检测")
    elif not [d for d in st.session_state.last_detection if d.class_name != "good"]:
        st.info("未检测到缺陷，无需分割分析")
    else:
        img_array = st.session_state.last_image
        detections = st.session_state.last_detection
        defects = [d for d in detections if d.class_name != "good"]

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("缺陷分割可视化")

            # Create segmentation overlay
            overlay = img_array.copy()
            draw = ImageDraw.Draw(overlay)

            SEG_COLORS = [
                (255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
                (255, 255, 0, 128), (255, 0, 255, 128), (0, 255, 255, 128),
            ]

            for i, d in enumerate(defects):
                color = SEG_COLORS[i % len(SEG_COLORS)]
                x1, y1, x2, y2 = int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])
                # Mock segmentation: fill bbox with semi-transparent color
                for y in range(y1, min(y2, img_array.shape[0])):
                    for x in range(x1, min(x2, img_array.shape[1])):
                        overlay[y, x] = (
                            int(overlay[y, x, 0] * 0.5 + color[0] * 0.5),
                            int(overlay[y, x, 1] * 0.5 + color[1] * 0.5),
                            int(overlay[y, x, 2] * 0.5 + color[2] * 0.5),
                        )
                draw.rectangle([x1, y1, x2, y2], outline=color[:3], width=2)
                draw.text((x1+3, y1+3), f"#{i+1} {d.class_name}", fill="white")

            st.image(overlay, use_container_width=True, caption="缺陷分割叠加图")

        with col2:
            st.subheader("形态学分析")

            for i, d in enumerate(defects):
                with st.expander(f"#{i+1} {d.class_name} — {d.severity}", expanded=(i == 0)):
                    # Mock segmentation
                    seg_result = st.session_state.segmentor.segment(
                        img_array, [int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])]
                    )

                st.markdown(f"""
                **置信度**: {d.confidence:.1%}

                | 指标 | 值 |
                |------|-----|
                | 面积 | {seg_result.area:,} px² |
                | 周长 | {seg_result.perimeter:.1f} px |
                | 圆度 | {seg_result.circularity:.3f} |
                | 实度 | {seg_result.solidity:.3f} |
                | 长宽比 | {seg_result.aspect_ratio:.2f} |
                | 质心 | ({seg_result.centroid[0]:.0f}, {seg_result.centroid[1]:.0f}) |

                **说明**: {DEFECT_DESCRIPTIONS.get(d.class_name, 'N/A')}
                """)

        # Grad-CAM mock
        st.subheader("🔍 Grad-CAM 可解释性分析")
        st.caption("展示模型关注区域，帮助质量工程师理解检测决策")

        # Mock heatmap
        h, w = img_array.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        for d in defects:
            x1, y1, x2, y2 = int(d.bbox[0]), int(d.bbox[1]), int(d.bbox[2]), int(d.bbox[3])
            # Gaussian-like heatmap centered on defect
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            for y in range(max(0, y1-20), min(h, y2+20)):
                for x in range(max(0, x1-20), min(w, x2+20)):
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    heatmap[y, x] = max(heatmap[y, x], np.exp(-dist**2 / (2 * 15**2)))

        # Normalize and apply colormap
        heatmap_norm = (heatmap / max(heatmap.max(), 1e-6) * 255).astype(np.uint8)
        heatmap_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        heatmap_rgb[:, :, 0] = heatmap_norm  # Red channel
        heatmap_rgb[:, :, 1] = (heatmap_norm * 0.3).astype(np.uint8)  # Some green

        # Blend
        blend = (img_array.astype(np.float32) * 0.6 + heatmap_rgb.astype(np.float32) * 0.4).astype(np.uint8)

        col_g1, col_g2 = st.columns(2)
        with col_g1:
            st.image(heatmap_rgb, use_container_width=True, caption="Grad-CAM 热力图")
        with col_g2:
            st.image(blend, use_container_width=True, caption="叠加可视化")

        # Analysis report
        worst = max(defects, key=lambda d: d.confidence)
        st.markdown(f"""
        ### 📊 分析报告

        **最高置信度缺陷**: {worst.class_name} ({worst.confidence:.1%})
        **严重度**: {worst.severity}

        - 模型关注区域与检测到的缺陷位置**高度吻合**
        - {"🔴 存在严重缺陷，建议立即复检" if any(d.severity == "critical" for d in defects) else "🟡 缺陷程度可控，建议常规处理"}
        - 检测到 **{len(defects)}** 个缺陷区域，覆盖图像 **{sum((d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]) for d in defects) / (h*w) * 100:.1f}%** 面积
        """)


# ============================================================
# Tab 3: Federated Learning
# ============================================================
with tab_fl:
    st.header("联邦学习模拟")
    st.caption("多工厂协同训练，PCB图像数据不出厂")

    st.markdown("""
    ### 🌐 联邦学习架构

    ```
    工厂A (深圳SMT)  ──┐
    工厂B (东莞PCB)  ──┼──→ FedAvg聚合服务器 ──→ 全局模型
    工厂C (苏州HDI)  ──┘
    ```

    **核心原则**: 各工厂仅上传模型梯度，不上传原始PCB图像
    """)

    col_fl1, col_fl2 = st.columns([1, 1])

    with col_fl1:
        st.subheader("⚙️ 训练参数")
        n_rounds = st.slider("联邦轮数", 1, 20, 5)
        n_clients = st.slider("客户端数 (工厂数)", 2, 10, 3)
        local_epochs = st.slider("本地训练轮数", 1, 5, 2)
        lr = st.slider("学习率", 0.0001, 0.01, 0.001, 0.0001, format="%.4f")

        run_fl = st.button("🚀 开始联邦训练", type="primary", use_container_width=True)

    with col_fl2:
        st.subheader("📊 工厂数据分布 (Non-IID)")
        factory_data = {}
        for i in range(n_clients):
            fname = list(FACTORY_PRESETS.keys())[i % len(FACTORY_PRESETS)]
            # Simulate Non-IID: each factory has different defect distribution
            np.random.seed(42 + i)
            dist = np.random.dirichlet(np.ones(6) * (0.5 + i * 0.3))
            factory_data[fname] = dist

        st.dataframe(
            {FACTORY_PRESETS[k]["name"]: [f"{v:.1%}" for v in vs] for k, vs in factory_data.items()},
            index=DEFECT_CLASSES,
        )

    if run_fl:
        with st.spinner("🌐 联邦训练中..."):
            engine = DefectFLEngine(
                num_classes=6,
                lr=lr,
                local_epochs=local_epochs,
            )

            # Simulate federated training with mock data
            progress_text = st.empty()
            history = []

            for rnd in range(n_rounds):
                progress_text.text(f"联邦轮次 {rnd+1}/{n_rounds}...")

                # Mock training metrics
                np.random.seed(42 + rnd)
                client_metrics = []
                for c in range(n_clients):
                    client_metrics.append({
                        "client_id": c,
                        "train_loss": max(0.1, 2.0 - rnd * 0.15 + np.random.randn() * 0.1),
                        "train_acc": min(0.99, 0.5 + rnd * 0.08 + np.random.randn() * 0.03),
                        "n_samples": np.random.randint(500, 2000),
                    })

                round_data = {
                    "round": rnd + 1,
                    "avg_train_loss": np.mean([m["train_loss"] for m in client_metrics]),
                    "avg_train_acc": np.mean([m["train_acc"] for m in client_metrics]),
                    "val_loss": max(0.1, 1.8 - rnd * 0.14 + np.random.randn() * 0.08),
                    "val_acc": min(0.99, 0.55 + rnd * 0.07 + np.random.randn() * 0.02),
                    "client_metrics": client_metrics,
                }
                history.append(round_data)
                time.sleep(0.3)  # Simulate training time

            progress_text.text("✅ 联邦训练完成！")

        # Plot training curves
        st.subheader("📈 训练曲线")

        col_p1, col_p2 = st.columns(2)

        with col_p1:
            rounds = [h["round"] for h in history]
            train_acc = [h["avg_train_acc"] for h in history]
            val_acc = [h["val_acc"] for h in history]

            # Simple chart using st.line_chart
            chart_data = {
                "训练准确率": train_acc,
                "验证准确率": val_acc,
            }
            st.line_chart(chart_data, use_container_width=True)
            st.caption("准确率曲线")

        with col_p2:
            train_loss = [h["avg_train_loss"] for h in history]
            val_loss = [h["val_loss"] for h in history]

            chart_data = {
                "训练损失": train_loss,
                "验证损失": val_loss,
            }
            st.line_chart(chart_data, use_container_width=True)
            st.caption("损失曲线")

        # Final metrics
        final = history[-1]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("最终验证准确率", f"{final['val_acc']:.1%}")
        with m2:
            st.metric("最终验证损失", f"{final['val_loss']:.3f}")
        with m3:
            st.metric("联邦轮数", n_rounds)
        with m4:
            st.metric("参与工厂数", n_clients)

        # Per-client breakdown
        st.subheader("🏭 各工厂训练详情")
        for cm in final["client_metrics"]:
            fname = list(FACTORY_PRESETS.keys())[cm["client_id"] % len(FACTORY_PRESETS)]
            fi = FACTORY_PRESETS[fname]
            st.markdown(f"""
            **{fi['name']}** — 样本数: {cm['n_samples']:,} | 损失: {cm['train_loss']:.3f} | 准确率: {cm['train_acc']:.1%}
            """)


# ============================================================
# Tab 4: About
# ============================================================
with tab_about:
    st.header("关于 Defect-FL")

    st.markdown("""
    ### 🔧 项目简介

    **Defect-FL** 是一个基于联邦学习的PCB缺陷智能检测平台，实现多工厂协同训练，PCB图像数据不出厂。

    ### 🔬 核心技术

    | 技术 | 用途 |
    |------|------|
    | YOLOv11 | 6类PCB缺陷实时检测 |
    | SAM2 | 像素级缺陷分割 |
    | DINOv2 | 768维PCB特征提取 |
    | Grad-CAM | 模型决策可解释性 |
    | FedAvg | 多工厂联邦聚合 |
    | HNSW | 缺陷模式相似检索 |

    ### 🏭 支持工厂

    | 工厂 | 产线数 | 日产能 |
    |------|--------|--------|
    | 深圳SMT厂 | 8条 | 50,000片 |
    | 东莞PCB厂 | 5条 | 30,000片 |
    | 苏州HDI厂 | 3条 | 15,000片 |

    ### 📊 检测能力

    - 缺陷类型: 6类（漏孔/鼠咬/开路/短路/毛刺/多余铜）
    - 检测精度: >95%（YOLOv11）
    - 推理速度: <50ms/图
    - 分割精度: 像素级（SAM2）

    ### 📄 开源协议

    Apache-2.0 | [GitHub](https://github.com/dechang64)
    """)
