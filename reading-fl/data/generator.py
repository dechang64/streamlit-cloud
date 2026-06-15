from __future__ import annotations
"""
Synthetic data generator for Reading-FL.

Generates realistic reading community data for development and testing.
Each campus has a distinct "reading personality" (Non-IID distribution).
"""

import random
import hashlib
import numpy as np
from typing import Optional

from .reflection import (
    BookExcerpt, Reflection, ReadingSession, ReadingEvent, ReaderProfile,
    EMOTION_LABELS,
)


# ============================================================
# Book corpus — 20 books across 6 domains
# ============================================================

BOOK_CORPUS = [
    # 文学
    {"id": "b01", "title": "百年孤独", "author": "马尔克斯", "domain": "文学",
     "excerpts": [
         "多年以后，面对行刑队，奥雷里亚诺·布恩迪亚上校将会回想起父亲带他去见识冰块的那个遥远的下午。",
         "生命中曾经有过的所有灿烂，原来终究，都需要用寂寞来偿还。",
         "过去都是假的，回忆是一条没有归途的路。",
     ]},
    {"id": "b02", "title": "活着", "author": "余华", "domain": "文学",
     "excerpts": [
         "人是为活着本身而活着的，而不是为了活着之外的任何事物所活着。",
         "最初我们来到这个世界，是因为不得不来；最终我们离开这个世界，是因为不得不走。",
         "没有什么比时间更具有说服力了，因为时间无需通知我们就可以改变一切。",
     ]},
    {"id": "b03", "title": "红楼梦", "author": "曹雪芹", "domain": "文学",
     "excerpts": [
         "满纸荒唐言，一把辛酸泪。都云作者痴，谁解其中味。",
         "假作真时真亦假，无为有处有还无。",
         "质本洁来还洁去，强于污淖陷渠沟。",
     ]},
    # 科技
    {"id": "b04", "title": "人类简史", "author": "赫拉利", "domain": "科技",
     "excerpts": [
         "人类以为自己驯化了植物，实际上是植物驯化了人类。",
         "金钱是有史以来最成功的虚构故事。",
         "想象的秩序并非个人主观的想象，而是存在于千千万万人共同的想象之中。",
     ]},
    {"id": "b05", "title": "技术的本质", "author": "布莱恩·阿瑟", "domain": "科技",
     "excerpts": [
         "技术是对自然现象有目的的编程。",
         "所有技术都是从已有技术中诞生的。",
         "技术的进化不是随机的，而是沿着可行性的方向不断探索。",
     ]},
    {"id": "b06", "title": "失控", "author": "凯文·凯利", "domain": "科技",
     "excerpts": [
         "要成为上帝，你只需要控制一切。但真正的上帝从不控制。",
         "群体被看做是一种自适应系统，它能够自我进化。",
         "最伟大的发明不是某个具体的东西，而是发明本身这个过程。",
     ]},
    # 哲学
    {"id": "b07", "title": "苏菲的世界", "author": "乔斯坦·贾德", "domain": "哲学",
     "excerpts": [
         "你是谁？世界从哪里来？",
         "真正的智慧来自知道自己无知。",
         "哲学家们只是用不同的方式解释世界，而问题在于改变世界。",
     ]},
    {"id": "b08", "title": "存在与时间", "author": "海德格尔", "domain": "哲学",
     "excerpts": [
         "向死而生的意义是：当你无限接近死亡，才能深切体会生的意义。",
         "语言是存在的家。",
         "人是被抛入世界的存在。",
     ]},
    {"id": "b09", "title": "沉思录", "author": "马可·奥勒留", "domain": "哲学",
     "excerpts": [
         "你有力量控制自己的思想，而不是外部事件。认识到这一点，你就会找到力量。",
         "浪费在愤怒上的时间，就是浪费在敌人身上的时间。",
         "幸福取决于灵魂的品质。",
     ]},
    # 历史
    {"id": "b10", "title": "万历十五年", "author": "黄仁宇", "domain": "历史",
     "excerpts": [
         "大历史不会萎缩，也不会被遗忘，它只是以不同的方式重新开始。",
         "一个帝国的覆灭，往往不是因为外敌入侵，而是从内部开始腐烂。",
         "道德不是解决问题的工具，而是逃避问题的借口。",
     ]},
    {"id": "b11", "title": "枪炮、病菌与钢铁", "author": "戴蒙德", "domain": "历史",
     "excerpts": [
         "历史的走向从来不是由人种决定的，而是由地理和环境决定的。",
         "粮食生产是文明发展的前提，而不是结果。",
         "技术的传播比技术的发明更重要。",
     ]},
    {"id": "b12", "title": "全球通史", "author": "斯塔夫里阿诺斯", "domain": "历史",
     "excerpts": [
         "人类的每一个进步，都伴随着新的问题。",
         "历史告诉我们，没有任何一个文明可以永远持续。",
         "变革的阻力往往来自那些从旧秩序中获益最多的人。",
     ]},
    # 心理
    {"id": "b13", "title": "思考，快与慢", "author": "卡尼曼", "domain": "心理",
     "excerpts": [
         "我们对自己认为了解的事物过于自信。",
         "损失带来的痛苦是同等收益带来的快乐的两倍。",
         "直觉不是魔法，而是识别——这是专家的核心能力。",
     ]},
    {"id": "b14", "title": "被讨厌的勇气", "author": "岸见一郎", "domain": "心理",
     "excerpts": [
         "决定我们自身的不是过去的经历，而是我们自己赋予经历的意义。",
         "一切烦恼都是人际关系的烦恼。",
         "不要害怕被别人讨厌，因为那是你自由生活的证明。",
     ]},
    {"id": "b15", "title": "心流", "author": "契克森米哈赖", "domain": "心理",
     "excerpts": [
         "最优体验发生在一个人为了一个值得的目标而奋力拼搏的时候。",
         "控制意识最好的方法就是找到一个能让你全神贯注的活动。",
         "幸福不是终点，而是全身心投入一件事的过程。",
     ]},
    # 社会
    {"id": "b16", "title": "乡土中国", "author": "费孝通", "domain": "社会",
     "excerpts": [
         "中国社会是熟人社会，不是法治社会。",
         "从基层上看去，中国社会是乡土性的。",
         "文字的发生是在人和人传情达意的过程中受到了空间和时间的阻隔。",
     ]},
    {"id": "b17", "title": "乌合之众", "author": "勒庞", "domain": "社会",
     "excerpts": [
         "个人一旦成为群体的一员，他所作所为就不会再承担责任。",
         "群体只会干两种事——锦上添花或落井下石。",
         "数量即是正义。",
     ]},
    {"id": "b18", "title": "娱乐至死", "author": "尼尔·波兹曼", "domain": "社会",
     "excerpts": [
         "人类无言的死亡，是因为他们心甘情愿地成为娱乐的附庸。",
         "人们感到痛苦的不是他们用笑声代替了思考，而是他们不知道自己为什么笑以及为什么不再思考。",
         "我们毁掉的不是我们所憎恨的东西，而是我们所热爱的东西。",
     ]},
    # 补充
    {"id": "b19", "title": "小王子", "author": "圣埃克苏佩里", "domain": "文学",
     "excerpts": [
         "只有用心灵才能看清事物的本质，真正重要的东西是肉眼无法看见的。",
         "如果你驯服了我，我们就互相不可缺少了。",
         "所有的大人都曾经是小孩，虽然只有少数人记得。",
     ]},
    {"id": "b20", "title": "三体", "author": "刘慈欣", "domain": "科技",
     "excerpts": [
         "给岁月以文明，而不是给文明以岁月。",
         "弱小和无知不是生存的障碍，傲慢才是。",
         "在宇宙中，你再快都有比你快的，你再慢也有比你慢的。",
     ]},
]

# 每个校区的情感偏好分布（Non-IID）
CAMPUS_EMOTION_PROFILES = {
    "campus_a": {"moved": 0.10, "thinking": 0.35, "resonance": 0.15,
                 "confused": 0.25, "disagree": 0.10, "calm": 0.05},  # 理工科：偏思考/困惑
    "campus_b": {"moved": 0.30, "thinking": 0.15, "resonance": 0.30,
                 "confused": 0.05, "disagree": 0.05, "calm": 0.15},  # 文科：偏感动/共鸣
    "campus_c": {"moved": 0.20, "thinking": 0.25, "resonance": 0.20,
                 "confused": 0.15, "disagree": 0.10, "calm": 0.10},  # 综合：均匀
}

# 每个校区的领域偏好
CAMPUS_DOMAIN_PREFERENCES = {
    "campus_a": {"科技": 0.35, "哲学": 0.20, "心理": 0.15, "历史": 0.15, "文学": 0.10, "社会": 0.05},
    "campus_b": {"文学": 0.35, "哲学": 0.20, "心理": 0.15, "社会": 0.15, "历史": 0.10, "科技": 0.05},
    "campus_c": {"心理": 0.25, "文学": 0.20, "科技": 0.15, "哲学": 0.15, "社会": 0.15, "历史": 0.10},
}

# 感悟模板（按情感分类）
REFLECTION_TEMPLATES = {
    "moved": [
        "读到这段话的时候，眼眶突然就湿了。{trigger}",
        "这句话让我想起了{memory}，那种感觉很难用语言描述。",
        "原来真的有人能把我心里模糊的感受说清楚。{trigger}",
        "放下书，在窗边坐了很久。{trigger}",
        "这段文字有一种安静的力量，不声不响地击中了你。",
    ],
    "thinking": [
        "这个观点很有意思，但我觉得还可以从另一个角度想：{extension}",
        "作者在这里的逻辑链条似乎有个跳跃，{question}",
        "如果把这个框架应用到{scenario}，结论可能会完全不同。",
        "这让我想到了{reference}中类似的论述，但两者的出发点不同。",
        "我在想，这个论断在{context}的背景下还成立吗？",
    ],
    "resonance": [
        "这就是我一直在找的表述！{trigger}",
        "完全同意。我自己也一直在思考这个问题，但从来没说得这么清楚。",
        "这段话应该打印出来贴在墙上。{trigger}",
        "读完这段，立刻分享给了室友，他们也觉得说到了心坎上。",
        "这不就是我上周在课上讨论的观点吗？原来早有人想清楚了。",
    ],
    "confused": [
        "这里没太看懂，{question}",
        "作者的意思是{interpretation}吗？还是我理解偏了？",
        "这段和前文的逻辑似乎有矛盾，{conflict}",
        "查了一些资料，发现这个问题比书中说的复杂得多。",
        "我觉得作者在这里可能过于简化了，{nuance}",
    ],
    "disagree": [
        "不同意这个观点。{counter}",
        "这个论述忽略了一个重要因素：{factor}",
        "虽然理解作者的出发点，但{objection}",
        "这种二元对立的思维方式本身就有问题。",
        "如果按照这个逻辑推下去，会得出一个很荒谬的结论。",
    ],
    "calm": [
        "这段话让我安静了下来。",
        "读到这里，节奏慢了下来，像是在散步。",
        "没有什么特别强烈的感受，但觉得这段话是对的。",
        "像一杯温水，不烫不凉，刚好。",
        "这种平和的叙述方式反而更有力量。",
    ],
}

# 感悟中的触发词
TRIGGERS = {
    "moved": ["因为它让我意识到", "那种无力感", "关于离别", "关于孤独", "关于时间"],
    "thinking": ["从系统论的角度", "用博弈论来分析", "从进化心理学的视角", "这个假设", "反证法"],
    "resonance": ["关于成长的困惑", "关于选择", "关于自我认同", "关于理想", "关于坚持"],
    "confused": ["因果关系的方向", "隐含的前提", "样本偏差", "时间跨度", "控制变量"],
    "disagree": ["幸存者偏差", "相关性不等于因果性", "文化差异", "时代背景", "样本代表性"],
    "calm": ["窗外的雨", "午后的阳光", "安静的图书馆", "一杯茶", "翻页的声音"],
}

MEMORIES = [
    "小时候在外婆家的夏天", "高中毕业那天", "第一次独自旅行",
    "某个深夜的对话", "一场没有结果的考试", "离开家乡的那天早上",
]

SCENARIOS = [
    "人工智能教育", "远程办公", "城市规划", "医疗资源分配",
    "气候变化", "教育公平", "数字隐私", "社交媒体",
]

REFERENCES = [
    "《黑天鹅》", "《反脆弱》", "《自私的基因》", "《枪炮、病菌与钢铁》",
    "《思考，快与慢》", "《人类简史》", "《原则》", "《规模》",
]


class SyntheticDataGenerator:
    """生成合成阅读社区数据"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def generate_campus_data(
        self,
        campus_id: str,
        campus_name: str,
        reader_count: int,
        reflections_per_reader: int = 3,
    ) -> dict:
        """
        生成一个校区的完整数据

        Returns:
            {
                "campus_id": str,
                "campus_name": str,
                "readers": list[ReaderProfile],
                "reflections": list[Reflection],
                "sessions": list[ReadingSession],
                "excerpts": list[BookExcerpt],
            }
        """
        emotion_profile = CAMPUS_EMOTION_PROFILES.get(campus_id, CAMPUS_EMOTION_PROFILES["campus_c"])
        domain_prefs = CAMPUS_DOMAIN_PREFERENCES.get(campus_id, CAMPUS_DOMAIN_PREFERENCES["campus_c"])

        readers = []
        reflections = []
        sessions = []
        excerpts = []

        for i in range(reader_count):
            reader_id = hashlib.sha256(f"{campus_id}_reader_{i}".encode()).hexdigest()[:16]
            reader = ReaderProfile(reader_id=reader_id, campus_id=campus_id)
            readers.append(reader)

            for j in range(reflections_per_reader):
                # 按领域偏好选书
                domain = self._weighted_choice(domain_prefs)
                books_in_domain = [b for b in BOOK_CORPUS if b["domain"] == domain]
                if not books_in_domain:
                    books_in_domain = BOOK_CORPUS
                book = self.rng.choice(books_in_domain)
                excerpt_text = self.rng.choice(book["excerpts"])

                excerpt = BookExcerpt(
                    book_id=book["id"],
                    book_title=book["title"],
                    author=book["author"],
                    paragraph_id=f"{book['id']}_p{self.rng.randint(1, 20)}",
                    text=excerpt_text,
                    domain=domain,
                )
                excerpts.append(excerpt)

                # 按校区情感偏好选情感
                emotion = self._weighted_choice(emotion_profile)

                # 生成感悟文本
                reflection_text = self._generate_reflection(emotion, campus_id)

                # 生成阅读时长（30-600秒，带随机性）
                duration = self.rng.uniform(30, 600)
                if abs(duration - round(duration)) < 1e-9:
                    duration += self.rng.uniform(0.1, 0.9)  # 避免精确整数

                reflection = Reflection(
                    reader_id=reader_id,
                    campus_id=campus_id,
                    excerpt=excerpt,
                    reflection_text=reflection_text,
                    emotion_label=emotion,
                    reading_duration_sec=round(duration, 1),
                    lamp_id=f"LAMP_{campus_id[-1]}{self.rng.randint(100, 999)}",
                )
                reflections.append(reflection)

                # 生成阅读会话
                session = self._generate_session(reader_id, book["id"], reflection.lamp_id)
                sessions.append(session)

        return {
            "campus_id": campus_id,
            "campus_name": campus_name,
            "readers": readers,
            "reflections": reflections,
            "sessions": sessions,
            "excerpts": excerpts,
        }

    def generate_all_campuses(self, campuses: list) -> dict:
        """生成所有校区的数据"""
        all_data = {}
        for campus in campuses:
            data = self.generate_campus_data(
                campus_id=campus["id"],
                campus_name=campus["name"],
                reader_count=campus["reader_count"],
            )
            all_data[campus["id"]] = data
        return all_data

    def _weighted_choice(self, weights: dict) -> str:
        items = list(weights.keys())
        probs = list(weights.values())
        return self.rng.choices(items, weights=probs, k=1)[0]

    def _generate_reflection(self, emotion: str, campus_id: str) -> str:
        """生成一条感悟文本"""
        templates = REFLECTION_TEMPLATES[emotion]
        template = self.rng.choice(templates)

        triggers = TRIGGERS[emotion]
        trigger = self.rng.choice(triggers)

        if "{trigger}" in template:
            template = template.replace("{trigger}", trigger)
        elif "{memory}" in template:
            template = template.replace("{memory}", self.rng.choice(MEMORIES))
        elif "{extension}" in template:
            template = template.replace("{extension}", self.rng.choice(SCENARIOS))
        elif "{question}" in template:
            template = template.replace("{question}", f"这里的{self.rng.choice(['因果', '逻辑', '前提', '推论'])}是什么？")
        elif "{scenario}" in template:
            template = template.replace("{scenario}", self.rng.choice(SCENARIOS))
        elif "{reference}" in template:
            template = template.replace("{reference}", self.rng.choice(REFERENCES))
        elif "{counter}" in template:
            template = template.replace("{counter}", f"我认为{self.rng.choice(['实际情况更复杂', '这个结论过于绝对', '忽略了反面证据'])}")
        elif "{factor}" in template:
            template = template.replace("{factor}", self.rng.choice(["文化背景", "历史语境", "个体差异", "系统性因素"]))
        elif "{objection}" in template:
            template = template.replace("{objection}", f"现实中{self.rng.choice(['情况远比这复杂', '很多例外', '反例比比皆是'])}")
        elif "{conflict}" in template:
            template = template.replace("{conflict}", "前文说的是A，这里突然变成了B")
        elif "{nuance}" in template:
            template = template.replace("{nuance}", "中间地带往往才是最值得讨论的")
        elif "{interpretation}" in template:
            template = template.replace("{interpretation}", "作者可能在暗示某种深层结构")
        elif "{context}" in template:
            template = template.replace("{context}", self.rng.choice(["当代中国", "数字化时代", "后疫情时代", "全球化背景下"]))

        return template

    def _generate_session(self, reader_id: str, book_id: str, lamp_id: str) -> ReadingSession:
        """生成一次阅读会话"""
        duration = self.rng.uniform(60, 900)
        if abs(duration - round(duration)) < 1e-9:
            duration += self.rng.uniform(0.1, 0.9)

        n_page_turns = self.rng.randint(2, 15)
        n_pauses = self.rng.randint(0, 4)

        events = []
        current_time = 0.0

        for _ in range(n_page_turns):
            current_time += self.rng.uniform(5, 60)
            events.append(ReadingEvent(
                event_type="page_turn",
                timestamp=round(current_time, 1),
                paragraph_id=f"p{self.rng.randint(1, 20)}",
            ))

        for _ in range(n_pauses):
            events.append(ReadingEvent(
                event_type="pause",
                timestamp=round(self.rng.uniform(0, duration), 1),
            ))

        events.sort(key=lambda e: e.timestamp)

        # 最后一个事件
        if self.rng.random() > 0.3:
            events.append(ReadingEvent(
                event_type="finish" if self.rng.random() > 0.2 else "abandon",
                timestamp=round(duration, 1),
            ))

        return ReadingSession(
            reader_id=reader_id,
            book_id=book_id,
            lamp_id=lamp_id,
            events=events,
            total_duration_sec=round(duration, 1),
        )
