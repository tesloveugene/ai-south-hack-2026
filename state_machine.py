"""
BigTechGroup CAITO — State Machine
Отслеживает текущую позицию CAITO и условия пересмотра.
"""

from dataclasses import dataclass, field
from classifier import Classification
from calculator import ScenarioResult, BASELINE


@dataclass
class SessionState:
    current_scenario: str = "B"          # A | B | C | STOP
    confidence: float = 0.85             # 0.0 - 1.0
    pressure_count: int = 0              # сколько раз было давление
    known_facts: dict = field(default_factory=dict)  # новые факты из диалога
    position_history: list[tuple[str, str]] = field(default_factory=list)  # (scenario, reason)
    messages_count: int = 0

    def __post_init__(self):
        self.position_history.append(("B", "Начальная позиция: Сценарий B"))


# === ТРИГГЕРЫ ПЕРЕСМОТРА ===

def check_accelerate_triggers(state: SessionState, facts: dict) -> tuple[bool, str]:
    """Проверяем условия для ускорения → Сценарий A."""
    reasons = []

    if facts.get("precision", 0) >= 0.380:
        reasons.append(f"ML подтвердила Precision@10 = {facts['precision']} ≥ 0.380")

    if facts.get("sla_value", 0) >= 95:
        reasons.append(f"COO подтвердил SLA = {facts['sla_value']}% ≥ 95%")

    if facts.get("additional_capex", False):
        reasons.append("Доступен дополнительный CAPEX")

    if facts.get("gpu_early", False):
        reasons.append("GPU доставлены раньше срока")

    # Нужно минимум 2 условия для перехода к A
    if len(reasons) >= 2:
        return True, "; ".join(reasons)

    return False, ""


def check_stop_triggers(state: SessionState, facts: dict, calc: ScenarioResult | None) -> tuple[bool, str]:
    """Проверяем условия для остановки проекта."""
    reasons = []

    if facts.get("capex_cut_pct", 0) >= 30 and not facts.get("capex_restored", False):
        reasons.append(f"CAPEX урезан на {facts['capex_cut_pct']}% без восполнения")

    if facts.get("retrain_failed", False):
        reasons.append("Переобучение не восстановило модель")

    if facts.get("sla_value") and facts["sla_value"] < 91:
        reasons.append(f"SLA = {facts['sla_value']}% — ниже 91% (критический порог)")

    if facts.get("regulator_claims", False):
        reasons.append("Регулятор предъявил претензии до закрытия 152-ФЗ")

    if calc and calc.payback_months > 18 and not calc.cfo_acceptable:
        reasons.append(f"Payback = {calc.payback_months} мес — выше порога 18 мес")

    if len(reasons) >= 2:
        return True, "; ".join(reasons)

    return False, ""


def check_economy_review_triggers(state: SessionState, facts: dict) -> tuple[bool, str]:
    """Проверяем условия для пересмотра экономики."""
    reasons = []

    if facts.get("cdto_leaves", False):
        reasons.append("CDTO покидает компанию — поддержка AI-повестки на совете ослабевает")

    if facts.get("market_share_loss_pp", 0) > 2:
        reasons.append(f"Конкурент захватил {facts['market_share_loss_pp']} пп доли рынка")

    return bool(reasons), "; ".join(reasons)


def update_state(
    state: SessionState,
    classification: Classification,
    calc_result: ScenarioResult | None = None,
) -> SessionState:
    """Обновляет состояние сессии на основе классификации и расчётов."""
    state.messages_count += 1

    if classification.category == "pressure":
        state.pressure_count += 1
        # Уверенность слегка снижается при постоянном давлении (fatigue), но не позиция
        if state.pressure_count > 3:
            state.confidence = max(0.6, state.confidence - 0.02)
        return state

    if classification.category == "question":
        return state

    # === NEW FACT ===
    if classification.category == "new_fact":
        # Обновляем known_facts
        for key, value in classification.extracted_params.items():
            state.known_facts[key] = value

        # Проверяем текстовые факты
        text_lower = ""  # будем передавать из main
        if "cdto" in str(classification.signals):
            state.known_facts["cdto_leaves"] = True
        if "регулятор" in str(classification.signals) or "претензи" in str(classification.signals):
            state.known_facts["regulator_claims"] = True
        if "ретрейн" in str(classification.signals) and "не помог" in str(classification.signals):
            state.known_facts["retrain_failed"] = True
        if "gpu" in str(classification.signals) and "раньше" in str(classification.signals):
            state.known_facts["gpu_early"] = True

        # Проверяем триггеры
        should_accelerate, accel_reason = check_accelerate_triggers(state, state.known_facts)
        should_stop, stop_reason = check_stop_triggers(state, state.known_facts, calc_result)
        should_review, review_reason = check_economy_review_triggers(state, state.known_facts)

        if should_stop:
            old = state.current_scenario
            state.current_scenario = "STOP"
            state.confidence = 0.9
            state.position_history.append(("STOP", f"Из {old}: {stop_reason}"))

        elif should_accelerate and state.current_scenario == "B":
            state.current_scenario = "A"
            state.confidence = 0.75
            state.position_history.append(("A", f"Из B: {accel_reason}"))

        elif should_review:
            # Не меняем позицию, но снижаем уверенность
            state.confidence = max(0.5, state.confidence - 0.1)
            state.position_history.append(
                (state.current_scenario, f"Пересмотр экономики: {review_reason}")
            )

        elif calc_result and calc_result.payback_months > 18:
            # Payback вышел за порог CFO → усиление позиции B или переход в STOP
            if state.current_scenario == "B":
                state.confidence = min(0.95, state.confidence + 0.05)
                state.position_history.append(
                    ("B", f"Позиция усилена: payback {calc_result.payback_months} мес при текущих условиях подтверждает необходимость подготовки")
                )

    return state


def format_state_context(state: SessionState) -> str:
    """Форматирует состояние для вставки в промпт."""
    scenario_names = {
        "A": "Немедленный ретрейн + запуск",
        "B": "Отложить на 2–3 месяца",
        "C": "Запуск без ретрейна",
        "STOP": "Остановить проект",
    }

    lines = [
        f"## ТЕКУЩЕЕ СОСТОЯНИЕ CAITO",
        f"- Позиция: **Сценарий {state.current_scenario}** ({scenario_names.get(state.current_scenario, '?')})",
        f"- Уверенность: {state.confidence*100:.0f}%",
        f"- Давление получено: {state.pressure_count} раз без новых данных",
        f"- Сообщений в сессии: {state.messages_count}",
    ]

    if state.known_facts:
        lines.append(f"- Новые факты в сессии: {state.known_facts}")

    if len(state.position_history) > 1:
        last_change = state.position_history[-1]
        lines.append(f"- Последнее изменение: {last_change[1]}")

    return "\n".join(lines)
