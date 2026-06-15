from __future__ import annotations
# ── python/utils/constants.py
DEFECT_CLASSES = ["missing_hole", "mouse_bite", "open_circuit", "short", "spur", "spurious_copper"]

DEFECT_DESCRIPTIONS = {
    "missing_hole": "Drill hole absent — prevents component mounting",
    "mouse_bite": "Copper nibble on trace edge — may cause open circuit",
    "open_circuit": "Broken copper trace — signal interruption",
    "short": "Unintended copper bridge — signal shorting",
    "spur": "Extra copper protrusion — potential short risk",
    "spurious_copper": "Unwanted copper deposit — manufacturing contamination",
}

SEVERITY_LEVELS = {
    "missing_hole": "critical",
    "open_circuit": "critical",
    "short": "critical",
    "mouse_bite": "moderate",
    "spur": "minor",
    "spurious_copper": "moderate",
}

FACTORY_PRESETS = {
    "shenzhen_smt": {"name": "深圳SMT厂", "lines": 8, "daily_output": 50000},
    "dongguan_pcb": {"name": "东莞PCB厂", "lines": 5, "daily_output": 30000},
    "suzhou_hdi": {"name": "苏州HDI厂", "lines": 3, "daily_output": 15000},
}

COLORS = {
    "primary": "#38bdf8", "secondary": "#8b5cf6", "accent": "#22c55e",
    "warning": "#f59e0b", "danger": "#ef4444",
    "bg_dark": "#0a0e1a", "bg_card": "#111827",
    "text": "#e2e8f0", "text_muted": "#64748b",
}
