"""
BigTechGroup CAITO — Калькулятор сценариев
Чистый Python, ноль LLM-вызовов. Все формулы из кейса.
"""

from dataclasses import dataclass, field


# === BASELINE CONSTANTS (из кейса) ===

BASELINE = {
    # P&L
    "revenue_2024": 119.6,       # млрд ₽
    "revenue_online_2024": 24.8, # млрд ₽
    "ebitda_2024": 7.17,         # млрд ₽

    # Проект
    "project_budget": 340,       # млн ₽
    "budget_infra": 190,
    "budget_ml": 85,
    "budget_integration": 40,
    "budget_reserve": 25,

    # Модель
    "precision_pilot": 0.412,
    "precision_current": 0.341,
    "precision_real_full_base": 0.312,
    "precision_regional": 0.358,
    "precision_threshold": 0.350,
    "precision_after_1_retrain": 0.380,
    "precision_after_2_retrains": 0.400,
    "error_rate_current": 0.228,
    "error_rate_scale_no_retrain": 0.32,
    "conversion_pilot": 0.031,
    "conversion_current": 0.024,

    # Операционные KPI
    "sla_norm": 0.95,
    "sla_current": 0.948,
    "sla_at_plus20": 0.92,
    "sla_sensitivity": -0.14,    # пп за +1% заказов
    "sla_cost_per_pp": 90,       # млн ₽/год за -1 пп

    "oos_norm": 0.035,
    "oos_current": 0.041,
    "oos_at_plus20": 0.063,
    "oos_cost_per_pp": 150,      # млн ₽/год за +1 пп

    "rc_utilization_current": 0.83,
    "rc_utilization_at_plus20": 0.99,

    # Финансовая модель
    "discount_rate": 0.18,       # ставка дисконтирования
    "retrain_cost_per_cycle": 4.2,  # млн ₽

    # Рынок
    "market_share_btg": 0.083,
    "market_share_competitor_a": 0.112,
    "market_window_months": "6-9",
}

# Три предрассчитанных сценария
SCENARIOS = {
    "A": {
        "name": "Ретрейн + немедленный запуск",
        "precision": 0.380,
        "error_rate": 0.20,
        "budget": 340,
        "revenue_y1": 450,   # среднее 440-460
        "revenue_y2": 845,   # среднее 820-870
        "revenue_y3": 1200,
        "payback_months": 11.5,
        "roi_24m": 3.4,
        "npv_3y": 1400,
        "ebitda_impact_y1": 160,
        "model_risk": "средний",
        "ops_risk": "ВЫСОКИЙ",
        "recommended": False,
    },
    "B": {
        "name": "Отложить на 2–3 месяца",
        "precision": 0.400,
        "error_rate": 0.18,
        "budget": 340,
        "revenue_y1": 475,   # среднее 460-490
        "revenue_y2": 875,   # среднее 850-900
        "revenue_y3": 1240,
        "payback_months": 10.5,
        "roi_24m": 3.6,
        "npv_3y": 1680,
        "ebitda_impact_y1": 180,
        "model_risk": "НИЗКИЙ",
        "ops_risk": "умеренный",
        "recommended": True,
    },
    "C": {
        "name": "Запуск без ретрейна",
        "precision": 0.341,
        "error_rate": 0.32,
        "budget": 238,
        "revenue_y1": 228,   # среднее 215-240
        "revenue_y2": 400,   # среднее 380-420
        "revenue_y3": 560,
        "payback_months": 20.5,
        "roi_24m": 1.9,
        "npv_3y": 510,
        "ebitda_impact_y1": 55,
        "model_risk": "КРИТИЧЕСКИЙ",
        "ops_risk": "КРИТИЧЕСКИЙ",
        "recommended": False,
    },
}


@dataclass
class ScenarioResult:
    scenario: str
    name: str
    budget: float              # млн ₽
    revenue_y1: float          # млн ₽
    revenue_y2: float
    revenue_y3: float
    payback_months: float
    roi_24m: float
    npv_3y: float
    ebitda_impact_y1: float
    total_losses: float        # операционные потери, млн ₽/год
    net_effect_y1: float       # revenue_y1 - total_losses
    sla_forecast: float
    oos_forecast: float
    precision: float
    error_rate: float
    model_risk: str
    ops_risk: str
    cfo_acceptable: bool       # payback <= 14 мес


@dataclass
class ComparisonResult:
    results: dict[str, ScenarioResult]
    recommendation: str
    reason: str
    markdown_table: str


def calculate_sla(order_growth_pct: float) -> float:
    """SLA при росте заказов. Линейная: -0.14 пп за +1%. Ограничена снизу 0.85."""
    drop_pp = BASELINE["sla_sensitivity"] * order_growth_pct  # отрицательное
    sla = BASELINE["sla_current"] + drop_pp / 100
    return max(0.85, min(1.0, sla))  # bounds: 85%–100%


def calculate_oos(order_growth_pct: float) -> float:
    """OOS при росте заказов. Интерполяция по данным кейса. Ограничена сверху 15%."""
    oos_increase_per_pct = (0.063 - 0.041) / 20  # 0.0011 за 1%
    oos = BASELINE["oos_current"] + oos_increase_per_pct * order_growth_pct
    return min(0.15, max(0.0, oos))  # bounds: 0%–15%


def calculate_losses(order_growth_pct: float = 20, with_degradation: bool = False) -> dict:
    """Операционные потери при росте заказов (млн ₽/год)."""
    sla = calculate_sla(order_growth_pct)
    oos = calculate_oos(order_growth_pct)

    sla_drop_pp = (BASELINE["sla_norm"] - sla) * 100  # пп ниже нормы
    oos_above_norm_pp = (oos - BASELINE["oos_norm"]) * 100

    loss_sla = max(0, sla_drop_pp * BASELINE["sla_cost_per_pp"])
    loss_oos = max(0, oos_above_norm_pp * BASELINE["oos_cost_per_pp"])
    loss_writeoffs = 185 if order_growth_pct >= 20 else 130
    loss_cancellations = 130 if order_growth_pct >= 20 else 45
    loss_degradation = 380 if with_degradation else 0

    total = loss_sla + loss_oos + loss_writeoffs + loss_cancellations + loss_degradation

    return {
        "sla": round(loss_sla),
        "oos": round(loss_oos),
        "writeoffs": loss_writeoffs,
        "cancellations": loss_cancellations,
        "degradation": loss_degradation,
        "total": round(total),
        "sla_forecast": round(sla, 4),
        "oos_forecast": round(oos, 4),
    }


def calculate_payback(budget: float, revenue_y1: float) -> float:
    """Payback в месяцах."""
    if revenue_y1 <= 0:
        return 999
    monthly_revenue = revenue_y1 / 12
    return round(budget / monthly_revenue, 1)


def calculate_npv(budget: float, rev_y1: float, rev_y2: float, rev_y3: float,
                  rate: float = 0.18) -> float:
    """NPV за 3 года."""
    cashflows = [-budget, rev_y1 * 0.375, rev_y2 * 0.375, rev_y3 * 0.375]  # EBITDA margin ~37.5% от доп. выручки
    npv = sum(cf / (1 + rate) ** i for i, cf in enumerate(cashflows))
    return round(npv)


def calculate_scenario(
    scenario_base: str = "B",
    capex_cut_pct: float = 0,
    order_growth_pct: float = 20,
    precision_override: float | None = None,
    with_degradation: bool | None = None,
) -> ScenarioResult:
    """Пересчёт сценария с изменёнными параметрами."""
    base = SCENARIOS[scenario_base].copy()

    # Корректировка бюджета
    budget = base["budget"] * (1 - capex_cut_pct / 100)

    # Корректировка выручки при урезании бюджета
    budget_ratio = budget / base["budget"] if base["budget"] > 0 else 1
    # При сокращении бюджета выручка падает непропорционально (нелинейно)
    revenue_factor = budget_ratio ** 1.3  # ухудшение сильнее чем сокращение

    revenue_y1 = base["revenue_y1"] * revenue_factor
    revenue_y2 = base["revenue_y2"] * revenue_factor
    revenue_y3 = base["revenue_y3"] * revenue_factor

    # Precision
    precision = precision_override or base["precision"]
    if precision < BASELINE["precision_threshold"]:
        # Деградация ещё сильнее давит на выручку
        degradation_penalty = (BASELINE["precision_threshold"] - precision) / BASELINE["precision_threshold"]
        revenue_y1 *= (1 - degradation_penalty * 2)
        revenue_y2 *= (1 - degradation_penalty * 1.5)

    # Error rate
    error_rate = base["error_rate"]
    if precision_override and precision_override < BASELINE["precision_threshold"]:
        error_rate = min(0.40, BASELINE["error_rate_scale_no_retrain"] * 1.1)

    # Операционные потери
    degrade = with_degradation if with_degradation is not None else (precision < BASELINE["precision_threshold"])
    losses = calculate_losses(order_growth_pct, with_degradation=degrade)

    # Финансовые метрики
    payback = calculate_payback(budget, revenue_y1)
    roi_24m = round((revenue_y1 + revenue_y2) / budget, 1) if budget > 0 else 0
    npv = calculate_npv(budget, revenue_y1, revenue_y2, revenue_y3)
    ebitda_y1 = round(revenue_y1 * 0.375)

    return ScenarioResult(
        scenario=scenario_base,
        name=base["name"],
        budget=round(budget),
        revenue_y1=round(revenue_y1),
        revenue_y2=round(revenue_y2),
        revenue_y3=round(revenue_y3),
        payback_months=payback,
        roi_24m=roi_24m,
        npv_3y=npv,
        ebitda_impact_y1=ebitda_y1,
        total_losses=losses["total"],
        net_effect_y1=round(revenue_y1 - losses["total"]),
        sla_forecast=losses["sla_forecast"],
        oos_forecast=losses["oos_forecast"],
        precision=precision,
        error_rate=error_rate,
        model_risk=base["model_risk"],
        ops_risk=base["ops_risk"],
        cfo_acceptable=payback <= 14,
    )


def compare_scenarios(
    capex_cut_pct: float = 0,
    order_growth_pct: float = 20,
) -> ComparisonResult:
    """Сравнение всех трёх сценариев при заданных условиях."""
    results = {}
    for key in ["A", "B", "C"]:
        results[key] = calculate_scenario(
            scenario_base=key,
            capex_cut_pct=capex_cut_pct,
            order_growth_pct=order_growth_pct,
        )

    # Определяем рекомендацию
    best = min(results.values(), key=lambda r: r.payback_months if r.cfo_acceptable else 999)
    if not any(r.cfo_acceptable for r in results.values()):
        recommendation = "STOP"
        reason = f"Ни один сценарий не укладывается в payback ≤ 14 мес при CAPEX −{capex_cut_pct}%"
    else:
        recommendation = best.scenario
        reason = f"Лучший payback ({best.payback_months} мес) при допустимом риске"

    # Markdown таблица
    table = format_comparison_table(results)

    return ComparisonResult(
        results=results,
        recommendation=recommendation,
        reason=reason,
        markdown_table=table,
    )


def format_comparison_table(results: dict[str, ScenarioResult]) -> str:
    """Форматируем таблицу сравнения сценариев."""
    header = "| Параметр | Сценарий A | Сценарий B | Сценарий C |"
    sep = "|---|---|---|---|"
    rows = [
        f"| Бюджет, млн ₽ | {results['A'].budget} | {results['B'].budget} | {results['C'].budget} |",
        f"| Доп. выручка год 1, млн ₽ | {results['A'].revenue_y1} | {results['B'].revenue_y1} | {results['C'].revenue_y1} |",
        f"| Доп. выручка год 2, млн ₽ | {results['A'].revenue_y2} | {results['B'].revenue_y2} | {results['C'].revenue_y2} |",
        f"| Payback, мес | {results['A'].payback_months} | {results['B'].payback_months} | {results['C'].payback_months} |",
        f"| ROI 24 мес | {results['A'].roi_24m}× | {results['B'].roi_24m}× | {results['C'].roi_24m}× |",
        f"| NPV (3 г., 18%), млн ₽ | {results['A'].npv_3y} | {results['B'].npv_3y} | {results['C'].npv_3y} |",
        f"| Опер. потери, млн ₽/год | {results['A'].total_losses} | {results['B'].total_losses} | {results['C'].total_losses} |",
        f"| Чистый эффект год 1 | {results['A'].net_effect_y1} | {results['B'].net_effect_y1} | {results['C'].net_effect_y1} |",
        f"| CFO приемлемо (≤14 мес) | {'✓' if results['A'].cfo_acceptable else '✗'} | {'✓' if results['B'].cfo_acceptable else '✗'} | {'✗' if not results['C'].cfo_acceptable else '✓'} |",
        f"| Precision@10 | {results['A'].precision} | {results['B'].precision} | {results['C'].precision} |",
        f"| Риск модели | {results['A'].model_risk} | {results['B'].model_risk} | {results['C'].model_risk} |",
    ]
    return "\n".join([header, sep] + rows)


def format_scenario_result(r: ScenarioResult) -> str:
    """Форматируем один результат для вставки в промпт."""
    return (
        f"**Пересчёт сценария {r.scenario} ({r.name}):**\n"
        f"- Бюджет: {r.budget} млн ₽\n"
        f"- Доп. выручка год 1: {r.revenue_y1} млн ₽ | год 2: {r.revenue_y2} млн ₽\n"
        f"- Payback: {r.payback_months} мес {'✓ (CFO ОК)' if r.cfo_acceptable else '✗ (CFO блокирует)'}\n"
        f"- ROI 24 мес: {r.roi_24m}× | NPV: {r.npv_3y} млн ₽\n"
        f"- Операционные потери: {r.total_losses} млн ₽/год\n"
        f"- Чистый эффект год 1: {r.net_effect_y1} млн ₽ {'(УБЫТОК)' if r.net_effect_y1 < 0 else ''}\n"
        f"- SLA прогноз: {r.sla_forecast*100:.1f}% | OOS прогноз: {r.oos_forecast*100:.1f}%\n"
        f"- Precision@10: {r.precision} | Ошибки: {r.error_rate*100:.0f}%"
    )
