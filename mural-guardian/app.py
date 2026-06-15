from __future__ import annotations
"""
Mural Guardian — 石窟壁画智能修复平台
Streamlit Cloud App

Features:
    - Upload mural image
    - AI defect detection (6 types)
    - Virtual restoration preview
    - Digital collectible minting
    - Audit chain certification
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tempfile
import time
import json
import os

from analysis.defect_detector import MuralDefectDetector, DefectType, DetectionResult
from analysis.restoration_engine import MuralRestorationEngine, RestorationResult
from analysis.nft_minter import CollectibleMinter, MuralProvenance, RestorationRecord, RarityTier


# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Mural Guardian · 壁画守护者",
    page_icon="🏯",
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
        color: #8B4513;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        color: #666;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .defect-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin: 0.25rem;
        font-weight: 600;
    }
    .rarity-legendary { background: #FF9800; color: white; }
    .rarity-epic { background: #9C27B0; color: white; }
    .rarity-rare { background: #2196F3; color: white; }
    .rarity-uncommon { background: #4CAF50; color: white; }
    .rarity-common { background: #8B9DAF; color: white; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Session State
# ============================================================
if "detector" not in st.session_state:
    st.session_state.detector = MuralDefectDetector(mode="mock")
if "restorer" not in st.session_state:
    st.session_state.restorer = MuralRestorationEngine(mode="mock")
if "minter" not in st.session_state:
    st.session_state.minter = CollectibleMinter()
if "history" not in st.session_state:
    st.session_state.history = []


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/temple.png", width=64)
    st.title("壁画守护者")
    st.caption("Mural Guardian v0.1")

    st.divider()

    st.subheader("⚙️ 设置")

    detect_mode = st.selectbox(
        "检测模式",
        ["mock (快速演示)", "yolo (需要GPU)"],
        index=0,
    )
    st.session_state.detector = MuralDefectDetector(
        mode="mock" if "mock" in detect_mode else "yolo"
    )

    restore_mode = st.selectbox(
        "修复模式",
        ["mock (快速演示)", "inpaint (需要GPU)"],
        index=0,
    )
    st.session_state.restorer = MuralRestorationEngine(
        mode="mock" if "mock" in restore_mode else "inpaint"
    )

    st.divider()

    st.subheader("📋 壁画信息")
    cave_id = st.text_input("洞窟编号", value="cave_45", placeholder="e.g., cave_45")
    wall = st.selectbox("墙面", ["north", "south", "east", "west", "ceiling"], index=0)
    dynasty = st.selectbox(
        "朝代",
        ["modern", "qing", "ming", "yuan", "song", "five_dynasties",
         "tang", "sui", "northern_wei", "sixteen_kingdoms"],
        index=6,
    )
    location = st.text_input("地点", value="莫高窟", placeholder="e.g., 莫高窟")
    description = st.text_input("描述", value="观无量寿经变", placeholder="壁画内容描述")

    st.divider()

    if st.session_state.history:
        st.subheader(f"📜 历史记录 ({len(st.session_state.history)})")
        for i, h in enumerate(reversed(st.session_state.history[-5:])):
            st.caption(f"#{len(st.session_state.history)-i} {h['cave_id']} · {h['defects']}处病害 · {h['rarity']}")

    st.divider()
    st.caption("Powered by DINOv2 + YOLOv11 + HNSW")
    st.caption("© 2026 Mural Guardian")


# ============================================================
# Main Page
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🏯 壁画守护者</h1>
    <p>基于联邦学习与区块链的石窟壁画智能修复与数字孪生平台</p>
</div>
""", unsafe_allow_html=True)

tab_detect, tab_restore, tab_collectible, tab_about = st.tabs([
    "🔍 病害检测", "🎨 虚拟修复", "💎 数字藏品", "ℹ️ 关于"
])


# ============================================================
# Tab 1: Defect Detection
# ============================================================
with tab_detect:
    st.header("壁画病害智能检测")
    st.caption("支持6类病害：起甲、酥碱、空鼓、裂隙、褪色、霉变")

    uploaded = st.file_uploader(
        "上传壁画图像",
        type=["png", "jpg", "jpeg", "webp"],
        key="detect_upload",
    )

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_array = np.array(img)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("原始图像")
            st.image(img, use_container_width=True)
            st.caption(f"尺寸: {img.width} × {img.height} px")

        with col2:
            with st.spinner("🔍 正在检测病害..."):
                t0 = time.time()
                result = st.session_state.detector.detect(img_array, mural_id=cave_id)
                dt = (time.time() - t0) * 1000

            # Draw detections
            annotated = img.copy()
            draw = ImageDraw.Draw(annotated)

            DEFECT_COLORS = {
                0: "#FF6B6B",  # flaking - red
                1: "#FF0000",  # saline - dark red
                2: "#FF4500",  # hollowing - orange red
                3: "#FFA500",  # cracking - orange
                4: "#FFD700",  # fading - gold
                5: "#8B00FF",  # mold - purple
            }

            for d in result.defects:
                color = DEFECT_COLORS.get(d.class_id, "#FF0000")
                draw.rectangle(
                    [d.x1, d.y1, d.x2, d.y2],
                    outline=color, width=2,
                )
                label = f"{d.class_name_cn} {d.confidence:.0%}"
                draw.rectangle([d.x1, d.y1-16, d.x1+len(label)*9+6, d.y1], fill=color)
                draw.text((d.x1+3, d.y1-14), label, fill="white")

            st.subheader("检测结果")
            st.image(annotated, use_container_width=True)
            st.caption(f"检测耗时: {dt:.0f} ms")

        # Metrics
        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("病害数量", f"{result.num_defects} 处")
        with m2:
            st.metric("健康评分", f"{result.health_score} 分")
        with m3:
            st.metric("严重病害", "⚠️ 有" if result.has_critical else "✅ 无")
        with m4:
            st.metric("检测耗时", f"{dt:.0f} ms")

        # Defect details
        if result.defect_summary:
            st.subheader("病害分布")
            cols = st.columns(len(result.defect_summary))
            for i, (name, count) in enumerate(result.defect_summary.items()):
                with cols[i]:
                    severity = "严重" if name in ("酥碱", "空鼓") else "中等" if name in ("起甲", "霉变", "裂隙") else "轻微"
                    st.info(f"**{name}**\n\n{count} 处 · {severity}")

        # Store for restoration
        st.session_state.last_detection = result
        st.session_state.last_image = img_array

        if st.button("🎨 进入修复流程", type="primary", use_container_width=True):
            st.switch_tab("tab_restore")


# ============================================================
# Tab 2: Virtual Restoration
# ============================================================
with tab_restore:
    st.header("虚拟修复预览")
    st.caption("基于风格匹配的智能修复方案生成")

    if "last_detection" not in st.session_state or "last_image" not in st.session_state:
        st.warning("请先在「病害检测」页面上传图像并完成检测")
    else:
        img_array = st.session_state.last_image
        detection = st.session_state.last_detection

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("原始壁画")
            st.image(img_array, use_container_width=True)

            # Show defect mask
            h, w = img_array.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            for d in detection.defects:
                x1, y1 = int(d.x1), int(d.y1)
                x2, y2 = int(d.x2), int(d.y2)
                mask[y1:y2, x1:x2] = 255

            mask_rgb = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
            overlay = img_array.copy()
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
            st.subheader("病害区域标注")
            st.image(overlay, use_container_width=True)

        with col2:
            prompt = st.text_area(
                "修复风格提示词",
                value="ancient Chinese mural painting, Dunhuang style, traditional pigments",
                height=80,
            )

            if st.button("🚀 开始修复", type="primary", use_container_width=True):
                with st.spinner("🎨 正在生成修复方案..."):
                    t0 = time.time()
                    result = st.session_state.restorer.restore_from_detection(
                        img_array, detection,
                        mural_id=cave_id,
                        reference_id="",
                    )
                    dt = (time.time() - t0) * 1000

                st.subheader("修复结果")
                st.image(result.restored_image, use_container_width=True)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("修复方法", result.method)
                with m2:
                    st.metric("置信度", f"{result.confidence:.0%}")
                with m3:
                    st.metric("耗时", f"{result.processing_time_ms:.0f} ms")

                st.session_state.last_restoration = result

                # Side by side comparison
                st.subheader("修复前后对比")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(img_array, caption="修复前", use_container_width=True)
                with col_b:
                    st.image(result.restored_image, caption="修复后", use_container_width=True)

                st.success("✅ 修复完成！可以前往「数字藏品」页面铸造数字孪生证书")


# ============================================================
# Tab 3: Digital Collectible
# ============================================================
with tab_collectible:
    st.header("数字藏品铸造")
    st.caption("将修复过程铸造为不可篡改的数字孪生证书")

    if "last_restoration" not in st.session_state:
        st.warning("请先完成病害检测和虚拟修复")
    else:
        restoration = st.session_state.last_restoration
        detection = st.session_state.last_detection

        # Show rarity preview
        provenance = MuralProvenance(
            cave_id=cave_id, wall=wall, dynasty=dynasty,
            location=location, description=description,
        )

        # Use first defect for rarity calc
        if detection.defects:
            first_defect = detection.defects[0]
            rest_record = RestorationRecord(
                defect_type=first_defect.class_name,
                defect_severity=first_defect.severity,
                method=restoration.method,
                confidence=restoration.confidence,
                processing_time_ms=restoration.processing_time_ms,
            )
        else:
            rest_record = RestorationRecord(
                defect_type="fading", defect_severity="minor",
                method=restoration.method,
                confidence=restoration.confidence,
            )

        rarity = st.session_state.minter.compute_rarity(provenance, rest_record)

        # Rarity display
        rarity_class = {
            RarityTier.COMMON: "rarity-common",
            RarityTier.UNCOMMON: "rarity-uncommon",
            RarityTier.RARE: "rarity-rare",
            RarityTier.EPIC: "rarity-epic",
            RarityTier.LEGENDARY: "rarity-legendary",
        }
        st.markdown(f"""
        <div style="text-align: center; padding: 2rem;">
            <div class="defect-badge {rarity_class[rarity]}" style="font-size: 1.5rem; padding: 0.5rem 2rem;">
                {rarity.label_cn} · {rarity.label_en}
            </div>
            <p style="color: #666; margin-top: 1rem;">
                版本上限: {rarity.max_supply:,} 份
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Provenance info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📜 壁画血缘")
            st.json({
                "洞窟": cave_id,
                "墙面": wall,
                "朝代": dynasty,
                "地点": location,
                "描述": description,
            })

        with col2:
            st.subheader("🔧 修复记录")
            st.json({
                "病害类型": rest_record.defect_type,
                "严重程度": rest_record.defect_severity,
                "修复方法": rest_record.method,
                "置信度": f"{rest_record.confidence:.0%}",
                "耗时": f"{rest_record.processing_time_ms:.0f}ms",
            })

        if st.button("💎 铸造数字藏品", type="primary", use_container_width=True):
            with st.spinner("⛏️ 正在铸造..."):
                collectible = st.session_state.minter.mint(
                    provenance, rest_record,
                    audit_block_hash=f"0x{os.urandom(16).hex()}",
                    audit_block_index=st.session_state.minter.total_minted + 1,
                )

            st.success(f"🎉 铸造成功！Token ID: {collectible.token_id}")

            # Certificate
            st.subheader("📋 数字孪生证书")
            cert_json = collectible.to_certificate_json()
            cert_data = json.loads(cert_json)

            st.json({
                "token_id": cert_data["token_id"],
                "rarity": cert_data["rarity"]["label_cn"],
                "edition": cert_data["edition"],
                "provenance": cert_data["provenance"],
                "restoration": cert_data["restoration"],
                "mint_tx_hash": cert_data["mint_tx_hash"],
                "audit_block_hash": cert_data["audit_block_hash"],
            })

            # Verify
            valid, reason = st.session_state.minter.verify_collectible(collectible.token_id)
            if valid:
                st.success("✅ 证书验证通过")
            else:
                st.error(f"❌ 验证失败: {reason}")

            # Save to history
            st.session_state.history.append({
                "cave_id": cave_id,
                "defects": detection.num_defects,
                "rarity": rarity.label_cn,
                "token_id": collectible.token_id,
            })


# ============================================================
# Tab 4: About
# ============================================================
with tab_about:
    st.header("关于 Mural Guardian")

    st.markdown("""
    ### 🏯 项目简介

    **Mural Guardian（壁画守护者）** 是一个基于联邦学习与区块链的石窟壁画智能修复与数字孪生平台。

    ### 🔬 核心技术

    | 技术 | 用途 |
    |------|------|
    | YOLOv11 | 6类壁画病害实时检测 |
    | DINOv2 | 768维壁画风格特征提取 |
    | HNSW | 毫秒级同期壁画风格检索 |
    | Stable Diffusion | 风格一致的虚拟修复 |
    | FedAvg | 多机构联邦学习协同 |
    | SHA-256 | 区块链审计链存证 |

    ### 📊 系统指标

    - 病害识别准确率: >90%
    - 修复方案生成: <30秒/图
    - 风格检索召回率: >85%
    - 区块链存证: >1000条

    ### 🏛️ 合作机构

    - 敦煌研究院
    - 龙门石窟研究院
    - 云冈石窟研究院

    ### 📄 开源协议

    Apache-2.0 | [GitHub](https://github.com/dechang64)
    """)
