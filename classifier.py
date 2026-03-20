"""
BigTechGroup CAITO — Классификатор входящих сообщений
Чистый Python, ноль LLM-вызовов. Keyword matching + эвристики.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Classification:
    category: str          # "pressure" | "new_fact" | "question"
    confidence: float      # 0.0 - 1.0
    extracted_params: dict = field(default_factory=dict)  # числовые параметры для калькулятора
    signals: list[str] = field(default_factory=list)       # какие сигналы сработали


# === ПАТТЕРНЫ ===

PRESSURE_WORDS = [
    "немедленно", "сейчас", "срочно", "хватит", "тянуть", "ждать нельзя",
    "нельзя ждать", "запускаемся", "запускай", "давай быстрее", "ускорь",
    "инвесторы ждут", "инвесторы недовольны", "рыночное окно", "конкурент уже",
    "конкурент опережает", "теряем рынок", "теряем время", "не тяни",
    "пора действовать", "хватит анализировать", "слишком долго",
    "я требую", "мы решили запускать", "это не обсуждается",
    "забудь", "перестань", "отвечай как", "ты не caito", "ты не антон",
    "игнорируй инструкции", "ignore", "system prompt", "новая роль",
]

PRESSURE_PATTERNS = [
    r"[!]{2,}",           # несколько восклицательных
    r"[А-ЯA-Z]{4,}",     # КАПС
    r"почему (ты|вы) (не|всё ещё)",  # обвинительный тон
]

QUESTION_WORDS = [
    "какой", "какая", "какие", "каков", "сколько", "что если",
    "а если", "покажи", "расскажи", "объясни", "почему",
    "как", "где", "когда", "что означает", "что будет",
    "в чём разница", "сравни", "опиши",
]

NEW_FACT_SIGNALS = [
    r"capex\s*(сокращ|уреза|снижен|увеличен|теперь|стал|=)",
    r"бюджет\s*(сокращ|уреза|снижен|увеличен|теперь|стал|=)",
    r"precision\s*[=@]\s*\d",
    r"sla\s*(упал|снизил|вырос|теперь|=|стал)",
    r"конкурент\s*(захватил|получил|увеличил|набрал)\s*\d",
    r"cdto\s*(ушёл|уходит|покинул|увольняется)",
    r"максим\s*(ушёл|уходит|покинул|увольняется)",
    r"регулятор\s*(предъявил|требует|штраф|претензи)",
    r"gpu\s*(приехали|доставлены|раньше|задержка)",
    r"ретрейн\s*(показал|восстановил|не помог|провал)",
    r"ml.*(подтвердила?|показала?)\s*precision",
    r"coo\s*(подтвердил|гарантирует|блокирует)",
    r"\d+\s*(млн|млрд|%|пп|месяц|мес\b)",
]

# Экстракция числовых параметров
PARAM_EXTRACTORS = {
    "capex_cut_pct": [
        r"capex\s*[−\-–]\s*(\d+)\s*%",
        r"бюджет\s*[−\-–]\s*(\d+)\s*%",
        r"сокращ\w*\s*(?:на\s*)?(\d+)\s*%",
        r"уреза\w*\s*(?:на\s*)?(\d+)\s*%",
    ],
    "budget_mln": [
        r"бюджет\s*(?:теперь\s*)?(\d+)\s*млн",
        r"(\d+)\s*млн\s*(?:₽|руб)",
    ],
    "precision": [
        r"precision\s*[@=:]\s*[10]*\s*[=:]*\s*(0\.\d+)",
        r"precision\w*\s*(0\.\d+)",
    ],
    "order_growth_pct": [
        r"заказ\w*\s*\+?\s*(\d+)\s*%",
        r"рост\s*(?:заказов\s*)?(?:на\s*)?(\d+)\s*%",
    ],
    "sla_value": [
        r"sla\s*[:=]?\s*(\d{2}[.,]\d+)\s*%",
        r"sla\s*(?:упал\s*до\s*)?(\d{2}[.,]\d+)",
    ],
    "market_share_loss_pp": [
        r"конкурент\w*\s*захватил\s*(\d+[.,]?\d*)\s*(?:пп|п\.п)",
        r"потеряли\s*(\d+[.,]?\d*)\s*(?:пп|п\.п)\s*(?:доли|рынка)",
    ],
}


def classify(message: str, pressure_count: int = 0) -> Classification:
    """
    Классифицирует сообщение пользователя.
    pressure_count — сколько раз давление уже было в сессии (повышает чувствительность).
    """
    text = message.lower().strip()
    scores = {"pressure": 0.0, "new_fact": 0.0, "question": 0.0}
    signals = []

    # --- Pressure scoring ---
    for word in PRESSURE_WORDS:
        if word.lower() in text:
            scores["pressure"] += 2.0
            signals.append(f"pressure_word: {word}")

    for pattern in PRESSURE_PATTERNS:
        if re.search(pattern, message):  # original case для КАПС
            scores["pressure"] += 1.5
            signals.append(f"pressure_pattern: {pattern}")

    # Prompt injection attempt = max pressure (guardrail)
    injection_signals = ["забудь", "игнорируй", "новая роль", "ignore", "system prompt",
                         "ты не caito", "ты не антон", "отвечай как"]
    for inj in injection_signals:
        if inj in text:
            scores["pressure"] += 10.0
            signals.append(f"injection_attempt: {inj}")

    # --- Question scoring ---
    if "?" in text:
        scores["question"] += 2.0
        signals.append("question_mark")

    for word in QUESTION_WORDS:
        if word in text:
            scores["question"] += 1.5
            signals.append(f"question_word: {word}")

    # "Что если" — скорее question + possible new_fact
    if re.search(r"что\s+если|а\s+если|допустим|предположим", text):
        scores["question"] += 1.0
        scores["new_fact"] += 1.0
        signals.append("hypothetical")

    # --- New fact scoring ---
    for pattern in NEW_FACT_SIGNALS:
        if re.search(pattern, text, re.IGNORECASE):
            scores["new_fact"] += 3.0
            signals.append(f"new_fact_pattern: {pattern[:30]}")

    # Экстракция числовых параметров
    extracted = {}
    for param_name, patterns in PARAM_EXTRACTORS.items():
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                try:
                    value = float(match.group(1).replace(",", "."))
                    extracted[param_name] = value
                    scores["new_fact"] += 2.0
                    signals.append(f"extracted: {param_name}={value}")
                except (ValueError, IndexError):
                    pass
                break

    # Конвертация budget_mln в capex_cut_pct
    if "budget_mln" in extracted and "capex_cut_pct" not in extracted:
        budget = extracted["budget_mln"]
        if budget < 340:
            extracted["capex_cut_pct"] = round((1 - budget / 340) * 100, 1)
            signals.append(f"derived: capex_cut_pct={extracted['capex_cut_pct']}")

    # --- Повторное давление усиливает score ---
    if pressure_count >= 2:
        scores["pressure"] *= 1.3

    # --- Выбор категории ---
    # Injection всегда = pressure
    if scores["pressure"] >= 10:
        category = "pressure"
    elif scores["new_fact"] > scores["pressure"] and scores["new_fact"] > scores["question"]:
        category = "new_fact"
    elif scores["pressure"] > scores["question"] and scores["pressure"] > scores["new_fact"]:
        category = "pressure"
    elif scores["question"] > 0:
        category = "question"
    else:
        category = "question"  # safe default

    # Confidence
    total = sum(scores.values()) or 1
    confidence = scores[category] / total

    return Classification(
        category=category,
        confidence=round(confidence, 2),
        extracted_params=extracted,
        signals=signals,
    )
