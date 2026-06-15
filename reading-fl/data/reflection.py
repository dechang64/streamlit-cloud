"""
Data structures for Reading-FL.

Core data types:
    - Reflection: 读者感悟（核心数据）
    - ReadingSession: 灯的阅读行为数据
    - BookExcerpt: 书摘
    - ReaderProfile: 读者画像
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import hashlib
import json


# 情感标签定义
EMOTION_LABELS = ["moved", "thinking", "resonance", "confused", "disagree", "calm"]
EMOTION_LABEL_CN = {
    "moved": "感动", "thinking": "思考", "resonance": "共鸣",
    "confused": "困惑", "disagree": "反对", "calm": "平静"
}


@dataclass
class BookExcerpt:
    """书摘：一段被读者标记的文字"""
    book_id: str
    book_title: str
    author: str
    paragraph_id: str
    text: str
    domain: str = "general"  # 文学/科技/哲学/历史/心理/社会

    def to_dict(self) -> dict:
        return {
            "book_id": self.book_id,
            "book_title": self.book_title,
            "author": self.author,
            "paragraph_id": self.paragraph_id,
            "text": self.text,
            "domain": self.domain,
        }


@dataclass
class Reflection:
    """
    读者感悟 — 坐忘系统的核心数据单元

    与微信读书的"标记"不同，感悟是读者主动写的文字，
    包含真实的情感反应和思考过程。
    """
    reader_id: str              # 匿名ID（SHA-256哈希）
    campus_id: str              # 校区ID（FL客户端标识）
    excerpt: BookExcerpt        # 关联的书摘
    reflection_text: str        # 读者感悟原文
    emotion_label: str          # 情感标签
    emotion_vector: list = field(default_factory=list)  # 情感连续向量
    reading_duration_sec: float = 0.0  # 灯亮时长
    lamp_id: str = ""           # 物理灯ID
    timestamp: str = ""
    authenticity_hash: str = "" # 区块链哈希

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.authenticity_hash:
            self._compute_hash()

    def _compute_hash(self):
        """计算感悟的完整性哈希"""
        content = json.dumps({
            "reader": self.reader_id,
            "campus": self.campus_id,
            "excerpt": self.excerpt.text[:50],
            "reflection": self.reflection_text,
            "duration": self.reading_duration_sec,
            "lamp": self.lamp_id,
            "ts": self.timestamp,
        }, sort_keys=True)
        self.authenticity_hash = hashlib.sha256(content.encode()).hexdigest()

    @property
    def reflection_depth(self) -> float:
        """
        感悟深度分 (0-1)

        基于三个信号:
        - 文本长度（长感悟通常更深）
        - 情感浓度（非"平静"的情感通常更深）
        - 阅读时长（读得久再写感悟，通常更深）
        """
        length_score = min(len(self.reflection_text) / 200.0, 1.0)
        # More nuanced emotion scoring — not all "calm" is shallow
        emotion_scores = {
            "moved": 0.8, "thinking": 0.85, "resonance": 0.75,
            "confused": 0.6, "disagree": 0.7, "calm": 0.4,
        }
        emotion_score = emotion_scores.get(self.emotion_label, 0.5)
        duration_score = min(self.reading_duration_sec / 300.0, 1.0)
        return 0.4 * length_score + 0.3 * emotion_score + 0.3 * duration_score

    def to_dict(self) -> dict:
        return {
            "reader_id": self.reader_id,
            "campus_id": self.campus_id,
            "excerpt": self.excerpt.to_dict(),
            "reflection_text": self.reflection_text,
            "emotion_label": self.emotion_label,
            "reading_duration_sec": self.reading_duration_sec,
            "lamp_id": self.lamp_id,
            "timestamp": self.timestamp,
            "authenticity_hash": self.authenticity_hash,
            "depth": self.reflection_depth,
        }


@dataclass
class ReadingEvent:
    """单次阅读事件（来自坐忘·灯）"""
    event_type: str   # "page_turn" | "pause" | "resume" | "finish" | "abandon"
    timestamp: float  # 相对于开始时间的秒数
    paragraph_id: str = ""


@dataclass
class ReadingSession:
    """
    阅读会话 — 坐忘·灯采集的行为数据

    灯亮 = 开始阅读，灯灭 = 结束阅读
    这是物理行为，比屏幕上的"停留时长"更难伪造
    """
    reader_id: str
    book_id: str
    lamp_id: str
    events: list = field(default_factory=list)
    total_duration_sec: float = 0.0
    start_time: str = ""

    def __post_init__(self):
        if not self.start_time:
            self.start_time = datetime.now().isoformat()

    @property
    def completion_rate(self) -> float:
        """完成率：有多少翻页事件（粗略估计）"""
        page_turns = sum(1 for e in self.events if e.event_type == "page_turn")
        return min(page_turns / 10.0, 1.0)  # 假设10页为完整阅读

    @property
    def pause_positions(self) -> list:
        """停顿位置序列（秒）"""
        return [e.timestamp for e in self.events if e.event_type == "pause"]

    @property
    def is_authentic(self) -> bool:
        """
        真实性快速检查

        规则:
        - 时长 > 10秒（排除秒刷）
        - 至少有1个翻页事件
        - 不是精确的整数时长（排除定时器）
        """
        if self.total_duration_sec < 10:
            return False
        has_interaction = any(
            e.event_type in ("page_turn", "pause") for e in self.events
        )
        if not has_interaction:
            return False
        # 排除精确整数（如恰好60秒、120秒）
        if abs(self.total_duration_sec - round(self.total_duration_sec)) < 0.01:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "reader_id": self.reader_id,
            "book_id": self.book_id,
            "lamp_id": self.lamp_id,
            "total_duration_sec": self.total_duration_sec,
            "completion_rate": self.completion_rate,
            "is_authentic": self.is_authentic,
            "events": [
                {"type": e.event_type, "ts": e.timestamp}
                for e in self.events
            ],
        }


@dataclass
class ReaderProfile:
    """
    读者画像 — 由阅读行为和感悟聚合而成

    用于书友匹配：画像向量距离近 = 阅读品味相似
    """
    reader_id: str
    campus_id: str
    embedding: list = field(default_factory=list)  # 64维画像向量
    books_read: list = field(default_factory=list)
    total_reflections: int = 0
    avg_depth: float = 0.0
    preferred_emotions: list = field(default_factory=list)
    preferred_domains: list = field(default_factory=list)
    quality_reputation: float = 0.5  # 0-1，持续产出高质量感悟会提升

    def update_from_reflection(self, reflection: Reflection, new_embedding: list, ema_alpha: float = 0.3):
        """用新感悟更新画像"""
        self.total_reflections += 1
        self.avg_depth = (
            (self.avg_depth * (self.total_reflections - 1) + reflection.reflection_depth)
            / self.total_reflections
        )
        if reflection.excerpt.book_id not in self.books_read:
            self.books_read.append(reflection.excerpt.book_id)
        if reflection.excerpt.domain not in self.preferred_domains:
            self.preferred_domains.append(reflection.excerpt.domain)
        if new_embedding:
            # Exponential moving average update for profile vector
            if self.embedding:
                self.embedding = [
                    ema_alpha * n + (1 - ema_alpha) * o
                    for n, o in zip(new_embedding, self.embedding)
                ]
            else:
                self.embedding = new_embedding
        # 质量信誉：高质量提升，低质量惩罚
        depth = reflection.reflection_depth
        if depth >= 0.5:
            self.quality_reputation = min(
                1.0, self.quality_reputation + 0.01 * depth
            )
        else:
            self.quality_reputation = max(
                0.0, self.quality_reputation - 0.005 * (1.0 - depth)
            )
