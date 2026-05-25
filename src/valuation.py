import numpy as np
import pandas as pd

from src.config import (
    BASE_REVENUE_M,
    BASE_FCF_M,
    NET_DEBT_M,
    DILUTED_SHARES_M,
    TAX_RATE,
    CAPEX_PCT_REVENUE,
    D_AND_A_PCT_REVENUE,
    NWC_PCT_REVENUE_CHANGE,
    SCENARIOS,
    TABLES_DIR,
)


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _safe_float(value, default=np.nan) -> float:
    """Safely convert a value to float."""

    try:
        if value is None:
            return default
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _summary_row(
    metric: str,
    value,
    value_type: str,
    signal: str,
    interpretation: str,
) -> dict:
    """Create a standardized Metric / Value row."""

    return {
        "Metric": metric,
        "Value": value,
        "Value Type": value_type,
        "Signal": signal,
        "Interpretation": interpretation,
    }


def _dcf_summary_type_map() -> dict[str, str]:
    """Return value-type metadata for DCF summary columns."""

    return {
        "Revenue CAGR": "Percent",
        "Year 5 Revenue": "Money Millions",
        "EBITDA Margin": "Percent",
        "Year 5 EBITDA": "Money Millions",
        "Year 5 FCF": "Money Millions",
        "Year 5 FCF Margin": "Percent",
        "WACC": "Percent",
        "Terminal Growth": "Percent",
        "PV of Explicit FCF": "Money Millions",
        "PV of Terminal Value": "Money Millions",
        "Terminal Value % of EV": "Percent",
        "Enterprise Value": "Money Millions",
        "Net Debt": "Money Millions",
        "Equity Value": "Money Millions",
        "Diluted Shares": "Plain Number",
        "Fair Value Per Share": "Money",
        "Implied EV/EBITDA": "Multiple",
        "Implied FCF Yield": "Percent",
        "FY2026 FCF Guidance Cross-Check": "Money Millions",
    }


def _dcf_detail_type_map() -> dict[str, str]:
    """Return value-type metadata for DCF scenario detail columns."""

    return {
        "Year": "Plain Number",
        "Revenue": "Money Millions",
        "Revenue Growth": "Percent",
        "EBITDA": "Money Millions",
        "EBITDA Margin": "Percent",
        "D&A": "Money Millions",
        "EBIT": "Money Millions",
        "Tax": "Money Millions",
        "NOPAT": "Money Millions",
        "Capex": "Money Millions",
        "NWC Investment": "Money Millions",
        "Free Cash Flow": "Money Millions",
        "FCF Margin": "Percent",
        "PV of FCF": "Money Millions",
    }


def _add_column_type_metadata(df: pd.DataFrame, type_map: dict[str, str]) -> pd.DataFrame:
    """Add '<Column> Type' metadata columns when the source column exists."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    for column, value_type in type_map.items():
        if column in result.columns:
            result[f"{column} Type"] = value_type

    return result


def _assumption_value_type(assumption: str) -> str:
    """Assign formatting metadata to DCF assumption rows."""

    assumption_lower = str(assumption).lower()

    if any(
        token in assumption_lower
        for token in [
            "revenue cagr",
            "ebitda margin",
            "wacc",
            "terminal growth",
            "tax rate",
            "capex % revenue",
            "d&a % revenue",
            "nwc % revenue change",
        ]
    ):
        return "Percent"

    if any(
        token in assumption_lower
        for token in [
            "base revenue",
            "net debt",
            "fy2026 fcf guidance cross-check",
        ]
    ):
        return "Money Millions"

    if "diluted shares" in assumption_lower:
        return "Plain Number"

    return "Plain Number"


def _bridge_value_type(bridge_item: str) -> str:
    """Assign formatting metadata to valuation bridge rows."""

    item_lower = str(bridge_item).lower()

    if "fair value" in item_lower:
        return "Money"

    if any(
        token in item_lower
        for token in [
            "pv of explicit fcf",
            "pv of terminal value",
            "enterprise value",
            "net debt",
            "equity value",
        ]
    ):
        return "Money Millions"

    if "implied ev/ebitda" in item_lower:
        return "Multiple"

    if "implied fcf yield" in item_lower:
        return "Percent"

    if "diluted shares" in item_lower:
        return "Plain Number"

    return "Plain Number"


def _validate_dcf_inputs(
    revenue_growth: list[float],
    ebitda_margin: float,
    wacc: float,
    terminal_growth: float,
) -> None:
    """Validate core DCF assumptions."""

    if not revenue_growth:
        raise ValueError("Revenue growth assumptions cannot be empty.")

    if DILUTED_SHARES_M <= 0:
        raise ValueError("Diluted shares must be greater than zero.")

    if BASE_REVENUE_M <= 0:
        raise ValueError("Base revenue must be greater than zero.")

    if wacc <= terminal_growth:
        raise ValueError(
            f"WACC must be greater than terminal growth. "
            f"Received WACC={wacc:.2%}, terminal growth={terminal_growth:.2%}."
        )

    if ebitda_margin <= 0 or ebitda_margin > 1:
        raise ValueError(f"EBITDA margin appears invalid: {ebitda_margin}")

    for growth in revenue_growth:
        if pd.isna(growth):
            raise ValueError("Revenue growth assumptions contain missing values.")

        if growth < -0.50 or growth > 0.50:
            raise ValueError(
                f"Revenue growth assumption appears extreme: {growth:.2%}"
            )


def _value_signal(fair_value: float) -> str:
    """Convert fair value level into a valuation signal."""

    if pd.isna(fair_value):
        return "Unavailable"
    if fair_value >= 750:
        return "High Upside Case"
    if fair_value >= 625:
        return "Constructive"
    if fair_value >= 500:
        return "Neutral / Monitor"
    if fair_value >= 400:
        return "Cautious"
    return "Downside Case"


def _scenario_signal(scenario: str, fair_value: float) -> str:
    """Create a scenario-specific signal."""

    scenario_text = str(scenario).lower()

    if pd.isna(fair_value):
        return "Unavailable"

    if scenario_text == "bear":
        return "Downside Case"
    if scenario_text == "base":
        return _value_signal(fair_value)
    if scenario_text == "bull":
        return "Upside Case"

    return _value_signal(fair_value)


def _assumption_signal(metric: str, value: float) -> str:
    """Assign simple report-ready signals to major DCF assumptions."""

    if pd.isna(value):
        return "Unavailable"

    metric_lower = str(metric).lower()

    if "revenue cagr" in metric_lower or "revenue growth" in metric_lower:
        if value >= 0.08:
            return "Strong Growth"
        if value >= 0.06:
            return "Constructive"
        if value >= 0.04:
            return "Stable / Supportive"
        return "Soft Growth"

    if "ebitda margin" in metric_lower:
        if value >= 0.13:
            return "Strong Margin"
        if value >= 0.12:
            return "Constructive"
        if value >= 0.11:
            return "Stable"
        return "Margin Pressure"

    if "wacc" in metric_lower:
        if value <= 0.075:
            return "Low Discount Rate"
        if value <= 0.090:
            return "Normal Range"
        if value <= 0.105:
            return "Elevated Discount Rate"
        return "High Discount Rate"

    if "terminal growth" in metric_lower:
        if value >= 0.030:
            return "High Terminal Growth"
        if value >= 0.020:
            return "Normal Range"
        return "Conservative"

    if "terminal value" in metric_lower:
        if value >= 0.80:
            return "High Terminal Dependency"
        if value >= 0.70:
            return "Moderate Terminal Dependency"
        return "Balanced"

    if "fcf margin" in metric_lower:
        if value >= 0.08:
            return "Strong FCF Margin"
        if value >= 0.06:
            return "Constructive"
        if value >= 0.04:
            return "Stable"
        return "Low FCF Margin"

    return "Reference"


def _scenario_interpretation(
    scenario: str,
    fair_value_per_share: float,
    revenue_cagr: float,
    ebitda_margin: float,
    wacc: float,
    terminal_growth: float,
) -> str:
    """Create concise scenario interpretation text."""

    return (
        f"{scenario} case fair value reflects a {revenue_cagr:.1%} revenue CAGR, "
        f"{ebitda_margin:.1%} EBITDA margin, {wacc:.1%} WACC, and "
        f"{terminal_growth:.1%} terminal growth, producing an estimated fair value "
        f"of ${fair_value_per_share:,.2f} per share."
    )


def run_dcf_scenario(
    scenario_name: str,
    revenue_growth: list[float],
    ebitda_margin: float,
    wacc: float,
    terminal_growth: float,
) -> dict:
    """Run a five-year DCF scenario for CACI.

    All financial values are in millions except per-share values.
    """

    revenue_growth = [_safe_float(value) for value in revenue_growth]
    ebitda_margin = _safe_float(ebitda_margin)
    wacc = _safe_float(wacc)
    terminal_growth = _safe_float(terminal_growth)

    _validate_dcf_inputs(
        revenue_growth=revenue_growth,
        ebitda_margin=ebitda_margin,
        wacc=wacc,
        terminal_growth=terminal_growth,
    )

    years = list(range(1, len(revenue_growth) + 1))

    revenue = []
    ebitda = []
    da = []
    ebit = []
    tax = []
    nopat = []
    capex = []
    nwc = []
    fcf = []
    fcf_margin = []
    pv_fcf = []

    previous_revenue = BASE_REVENUE_M

    for i, growth in enumerate(revenue_growth):
        current_revenue = previous_revenue * (1 + growth)
        current_ebitda = current_revenue * ebitda_margin
        current_da = current_revenue * D_AND_A_PCT_REVENUE
        current_ebit = current_ebitda - current_da
        current_tax = max(current_ebit * TAX_RATE, 0)
        current_nopat = current_ebit - current_tax
        current_capex = current_revenue * CAPEX_PCT_REVENUE
        current_nwc = (current_revenue - previous_revenue) * NWC_PCT_REVENUE_CHANGE
        current_fcf = current_nopat + current_da - current_capex - current_nwc
        current_fcf_margin = (
            current_fcf / current_revenue
            if current_revenue > 0
            else np.nan
        )
        current_pv_fcf = current_fcf / ((1 + wacc) ** (i + 1))

        revenue.append(current_revenue)
        ebitda.append(current_ebitda)
        da.append(current_da)
        ebit.append(current_ebit)
        tax.append(current_tax)
        nopat.append(current_nopat)
        capex.append(current_capex)
        nwc.append(current_nwc)
        fcf.append(current_fcf)
        fcf_margin.append(current_fcf_margin)
        pv_fcf.append(current_pv_fcf)

        previous_revenue = current_revenue

    terminal_fcf = fcf[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(years))

    enterprise_value = sum(pv_fcf) + pv_terminal_value
    equity_value = enterprise_value - NET_DEBT_M
    fair_value_per_share = equity_value / DILUTED_SHARES_M

    terminal_value_pct_ev = (
        pv_terminal_value / enterprise_value
        if enterprise_value > 0
        else np.nan
    )
    implied_ev_ebitda = (
        enterprise_value / ebitda[-1]
        if ebitda[-1] > 0
        else np.nan
    )
    implied_fcf_yield = (
        fcf[-1] / equity_value
        if equity_value > 0
        else np.nan
    )

    revenue_cagr = (revenue[-1] / BASE_REVENUE_M) ** (1 / len(years)) - 1

    detail = pd.DataFrame(
        {
            "Year": years,
            "Revenue": revenue,
            "Revenue Growth": revenue_growth,
            "EBITDA": ebitda,
            "EBITDA Margin": [ebitda_margin] * len(years),
            "D&A": da,
            "EBIT": ebit,
            "Tax": tax,
            "NOPAT": nopat,
            "Capex": capex,
            "NWC Investment": nwc,
            "Free Cash Flow": fcf,
            "FCF Margin": fcf_margin,
            "PV of FCF": pv_fcf,
        }
    )

    detail["Scenario"] = scenario_name
    detail["Signal"] = detail["Free Cash Flow"].apply(
        lambda value: "Positive FCF" if value > 0 else "Negative FCF"
    )

    detail_for_export = _add_column_type_metadata(detail, _dcf_detail_type_map())
    _safe_to_csv(detail_for_export, f"dcf_{scenario_name.lower()}_detail.csv")

    return {
        "Scenario": scenario_name,
        "Revenue CAGR": revenue_cagr,
        "Year 5 Revenue": revenue[-1],
        "EBITDA Margin": ebitda_margin,
        "Year 5 EBITDA": ebitda[-1],
        "Year 5 FCF": fcf[-1],
        "Year 5 FCF Margin": fcf_margin[-1],
        "WACC": wacc,
        "Terminal Growth": terminal_growth,
        "PV of Explicit FCF": sum(pv_fcf),
        "PV of Terminal Value": pv_terminal_value,
        "Terminal Value % of EV": terminal_value_pct_ev,
        "Enterprise Value": enterprise_value,
        "Net Debt": NET_DEBT_M,
        "Equity Value": equity_value,
        "Diluted Shares": DILUTED_SHARES_M,
        "Fair Value Per Share": fair_value_per_share,
        "Implied EV/EBITDA": implied_ev_ebitda,
        "Implied FCF Yield": implied_fcf_yield,
        "FY2026 FCF Guidance Cross-Check": BASE_FCF_M,
        "Signal": _scenario_signal(scenario_name, fair_value_per_share),
        "Interpretation": _scenario_interpretation(
            scenario=scenario_name,
            fair_value_per_share=fair_value_per_share,
            revenue_cagr=revenue_cagr,
            ebitda_margin=ebitda_margin,
            wacc=wacc,
            terminal_growth=terminal_growth,
        ),
    }


def run_all_dcf_scenarios() -> pd.DataFrame:
    """Run all configured DCF scenarios and save DCF output tables."""

    rows = []

    for name, assumptions in SCENARIOS.items():
        rows.append(
            run_dcf_scenario(
                scenario_name=name,
                revenue_growth=assumptions["revenue_growth"],
                ebitda_margin=assumptions["ebitda_margin"],
                wacc=assumptions["wacc"],
                terminal_growth=assumptions["terminal_growth"],
            )
        )

    df = pd.DataFrame(rows)
    df_for_export = _add_column_type_metadata(df, _dcf_summary_type_map())
    _safe_to_csv(df_for_export, "dcf_summary.csv")

    sensitivity = build_dcf_sensitivity_table()
    _safe_to_csv(sensitivity, "dcf_sensitivity.csv")

    assumptions_table = build_dcf_assumptions_table()
    _safe_to_csv(assumptions_table, "dcf_assumptions.csv")

    valuation_bridge = build_base_case_valuation_bridge(df)
    _safe_to_csv(valuation_bridge, "dcf_base_case_valuation_bridge.csv")

    return df


def calculate_fair_value_from_assumptions(
    revenue_growth: list[float],
    ebitda_margin: float,
    wacc: float,
    terminal_growth: float,
) -> float:
    """Calculate fair value per share from custom assumptions.

    This is used for the WACC / terminal growth sensitivity table.
    """

    revenue_growth = [_safe_float(value) for value in revenue_growth]
    ebitda_margin = _safe_float(ebitda_margin)
    wacc = _safe_float(wacc)
    terminal_growth = _safe_float(terminal_growth)

    if (
        pd.isna(ebitda_margin)
        or pd.isna(wacc)
        or pd.isna(terminal_growth)
        or wacc <= terminal_growth
        or DILUTED_SHARES_M <= 0
        or not revenue_growth
    ):
        return float("nan")

    previous_revenue = BASE_REVENUE_M
    fcf_values = []

    for growth in revenue_growth:
        if pd.isna(growth):
            return float("nan")

        revenue = previous_revenue * (1 + growth)
        ebitda = revenue * ebitda_margin
        da = revenue * D_AND_A_PCT_REVENUE
        ebit = ebitda - da
        tax = max(ebit * TAX_RATE, 0)
        nopat = ebit - tax
        capex = revenue * CAPEX_PCT_REVENUE
        nwc = (revenue - previous_revenue) * NWC_PCT_REVENUE_CHANGE
        fcf = nopat + da - capex - nwc

        fcf_values.append(fcf)
        previous_revenue = revenue

    if not fcf_values:
        return float("nan")

    pv_fcf = sum(
        fcf / ((1 + wacc) ** (i + 1))
        for i, fcf in enumerate(fcf_values)
    )

    terminal_fcf = fcf_values[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal_value = terminal_value / ((1 + wacc) ** len(fcf_values))

    enterprise_value = pv_fcf + pv_terminal_value
    equity_value = enterprise_value - NET_DEBT_M
    fair_value_per_share = equity_value / DILUTED_SHARES_M

    return float(fair_value_per_share)


def build_dcf_sensitivity_table() -> pd.DataFrame:
    """Build a WACC / terminal growth sensitivity table using base-case assumptions."""

    base_case = SCENARIOS["Base"]

    wacc_values = [0.075, 0.080, 0.085, 0.090, 0.095]
    terminal_growth_values = [0.015, 0.020, 0.025, 0.030, 0.035]

    rows = []

    for terminal_growth in terminal_growth_values:
        row = {
            "Terminal Growth": terminal_growth,
            "Terminal Growth Type": "Percent",
            "Signal": _assumption_signal("Terminal Growth", terminal_growth),
        }

        for wacc in wacc_values:
            column_name = f"WACC {wacc:.1%}"
            fair_value = calculate_fair_value_from_assumptions(
                revenue_growth=base_case["revenue_growth"],
                ebitda_margin=base_case["ebitda_margin"],
                wacc=wacc,
                terminal_growth=terminal_growth,
            )

            row[column_name] = fair_value
            row[f"{column_name} Type"] = "Money"

        rows.append(row)

    sensitivity = pd.DataFrame(rows)

    return sensitivity


def build_dcf_assumptions_table() -> pd.DataFrame:
    """Build an audit table of key DCF scenario assumptions."""

    rows = []

    for scenario_name, assumptions in SCENARIOS.items():
        revenue_growth = assumptions.get("revenue_growth", [])
        ebitda_margin = assumptions.get("ebitda_margin", np.nan)
        wacc = assumptions.get("wacc", np.nan)
        terminal_growth = assumptions.get("terminal_growth", np.nan)

        revenue_growth = [_safe_float(value) for value in revenue_growth]

        revenue_cagr = (
            np.prod([1 + value for value in revenue_growth]) ** (1 / len(revenue_growth)) - 1
            if revenue_growth
            else np.nan
        )

        assumption_rows = [
            {
                "Scenario": scenario_name,
                "Assumption": "Revenue CAGR",
                "Value": revenue_cagr,
                "Signal": _assumption_signal("Revenue CAGR", revenue_cagr),
                "Interpretation": "Implied compound annual revenue growth across explicit forecast years.",
            },
            {
                "Scenario": scenario_name,
                "Assumption": "EBITDA Margin",
                "Value": ebitda_margin,
                "Signal": _assumption_signal("EBITDA Margin", ebitda_margin),
                "Interpretation": "Scenario EBITDA margin used to convert revenue into operating cash flow proxy.",
            },
            {
                "Scenario": scenario_name,
                "Assumption": "WACC",
                "Value": wacc,
                "Signal": _assumption_signal("WACC", wacc),
                "Interpretation": "Scenario weighted average cost of capital used to discount free cash flow.",
            },
            {
                "Scenario": scenario_name,
                "Assumption": "Terminal Growth",
                "Value": terminal_growth,
                "Signal": _assumption_signal("Terminal Growth", terminal_growth),
                "Interpretation": "Long-term growth assumption used in terminal value calculation.",
            },
        ]

        rows.extend(assumption_rows)

    rows.extend(
        [
            {
                "Scenario": "Model-Wide",
                "Assumption": "Base Revenue",
                "Value": BASE_REVENUE_M,
                "Signal": "Reference",
                "Interpretation": "Starting revenue base for the explicit DCF forecast.",
            },
            {
                "Scenario": "Model-Wide",
                "Assumption": "Net Debt",
                "Value": NET_DEBT_M,
                "Signal": "Reference",
                "Interpretation": "Net debt deducted from enterprise value to estimate equity value.",
            },
            {
                "Scenario": "Model-Wide",
                "Assumption": "Diluted Shares",
                "Value": DILUTED_SHARES_M,
                "Signal": "Reference",
                "Interpretation": "Diluted share count used to convert equity value into fair value per share.",
            },
            {
                "Scenario": "Model-Wide",
                "Assumption": "Tax Rate",
                "Value": TAX_RATE,
                "Signal": "Reference",
                "Interpretation": "Tax rate applied to EBIT to estimate NOPAT.",
            },
            {
                "Scenario": "Model-Wide",
                "Assumption": "Capex % Revenue",
                "Value": CAPEX_PCT_REVENUE,
                "Signal": "Reference",
                "Interpretation": "Capital expenditure assumption as a percentage of revenue.",
            },
            {
                "Scenario": "Model-Wide",
                "Assumption": "D&A % Revenue",
                "Value": D_AND_A_PCT_REVENUE,
                "Signal": "Reference",
                "Interpretation": "Depreciation and amortization assumption as a percentage of revenue.",
            },
            {
                "Scenario": "Model-Wide",
                "Assumption": "NWC % Revenue Change",
                "Value": NWC_PCT_REVENUE_CHANGE,
                "Signal": "Reference",
                "Interpretation": "Net working capital investment as a percentage of revenue change.",
            },
            {
                "Scenario": "Model-Wide",
                "Assumption": "FY2026 FCF Guidance Cross-Check",
                "Value": BASE_FCF_M,
                "Signal": "Reference",
                "Interpretation": "Management free cash flow guidance used as a reasonableness cross-check.",
            },
        ]
    )

    assumptions_table = pd.DataFrame(rows)
    assumptions_table["Value Type"] = assumptions_table["Assumption"].apply(
        _assumption_value_type
    )

    return assumptions_table


def build_base_case_valuation_bridge(
    dcf_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a base-case enterprise value to fair value bridge."""

    if dcf_summary is None:
        dcf_summary = run_all_dcf_scenarios()

    if dcf_summary is None or dcf_summary.empty:
        return pd.DataFrame(
            [
                {
                    "Bridge Item": "Base Case Valuation Bridge",
                    "Value": np.nan,
                    "Value Type": "Plain Number",
                    "Signal": "Unavailable",
                    "Interpretation": "DCF summary was unavailable.",
                }
            ]
        )

    base_row = dcf_summary.loc[dcf_summary["Scenario"] == "Base"]

    if base_row.empty:
        return pd.DataFrame(
            [
                {
                    "Bridge Item": "Base Case Valuation Bridge",
                    "Value": np.nan,
                    "Value Type": "Plain Number",
                    "Signal": "Unavailable",
                    "Interpretation": "Base-case DCF scenario was unavailable.",
                }
            ]
        )

    base_row = base_row.iloc[0]

    rows = [
        {
            "Bridge Item": "PV of Explicit FCF",
            "Value": base_row.get("PV of Explicit FCF", np.nan),
            "Signal": "Positive FCF",
            "Interpretation": "Present value of the five-year explicit free cash flow forecast.",
        },
        {
            "Bridge Item": "PV of Terminal Value",
            "Value": base_row.get("PV of Terminal Value", np.nan),
            "Signal": _assumption_signal(
                "Terminal Value % of EV",
                base_row.get("Terminal Value % of EV", np.nan),
            ),
            "Interpretation": "Present value of terminal value after the explicit forecast period.",
        },
        {
            "Bridge Item": "Enterprise Value",
            "Value": base_row.get("Enterprise Value", np.nan),
            "Signal": "Valuation Output",
            "Interpretation": "Enterprise value implied by explicit free cash flow plus terminal value.",
        },
        {
            "Bridge Item": "Net Debt",
            "Value": base_row.get("Net Debt", np.nan),
            "Signal": "Deduction",
            "Interpretation": "Net debt deducted from enterprise value.",
        },
        {
            "Bridge Item": "Equity Value",
            "Value": base_row.get("Equity Value", np.nan),
            "Signal": "Valuation Output",
            "Interpretation": "Equity value after deducting net debt.",
        },
        {
            "Bridge Item": "Diluted Shares",
            "Value": base_row.get("Diluted Shares", np.nan),
            "Signal": "Share Count",
            "Interpretation": "Diluted shares used for fair value per share conversion.",
        },
        {
            "Bridge Item": "Base Fair Value Per Share",
            "Value": base_row.get("Fair Value Per Share", np.nan),
            "Signal": _value_signal(base_row.get("Fair Value Per Share", np.nan)),
            "Interpretation": "Base-case DCF fair value per share.",
        },
        {
            "Bridge Item": "Implied EV/EBITDA",
            "Value": base_row.get("Implied EV/EBITDA", np.nan),
            "Signal": "Valuation Multiple",
            "Interpretation": "Enterprise value divided by year-five EBITDA.",
        },
        {
            "Bridge Item": "Implied FCF Yield",
            "Value": base_row.get("Implied FCF Yield", np.nan),
            "Signal": "Cash Flow Yield",
            "Interpretation": "Year-five free cash flow divided by equity value.",
        },
    ]

    bridge = pd.DataFrame(rows)
    bridge["Value Type"] = bridge["Bridge Item"].apply(_bridge_value_type)

    return bridge