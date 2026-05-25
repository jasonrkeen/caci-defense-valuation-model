import numpy as np
import pandas as pd

from src.config import (
    TABLES_DIR,
    BASE_REVENUE_M,
    BASE_EBITDA_MARGIN,
    BASE_FCF_M,
    NET_DEBT_M,
    DILUTED_SHARES_M,
    TAX_RATE,
    CAPEX_PCT_REVENUE,
    D_AND_A_PCT_REVENUE,
    NWC_PCT_REVENUE_CHANGE,
    WACC,
    TERMINAL_GROWTH,
    SCENARIOS,
)


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
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


def _stress_signal(fair_value: float, base_fair_value: float) -> str:
    """Classify stress-test downside severity."""

    if pd.isna(fair_value) or pd.isna(base_fair_value) or base_fair_value == 0:
        return "Unavailable"

    downside = fair_value / base_fair_value - 1

    if downside >= -0.05:
        return "Limited Impact"
    if downside >= -0.15:
        return "Moderate Stress"
    if downside >= -0.25:
        return "Elevated Stress"
    return "Severe Stress"


def _valuation_bias(fair_value: float, base_fair_value: float) -> str:
    """Convert stress-test output into a valuation-bias label."""

    if pd.isna(fair_value) or pd.isna(base_fair_value) or base_fair_value == 0:
        return "Unavailable"

    downside = fair_value / base_fair_value - 1

    if downside >= 0.05:
        return "Positive"
    if downside >= -0.05:
        return "Neutral"
    if downside >= -0.15:
        return "Moderately Negative"
    if downside >= -0.25:
        return "Negative"
    return "Severely Negative"


def _interpret_stress_case(case_name: str, downside: float) -> str:
    """Create plain-English interpretation for a stress case."""

    if pd.isna(downside):
        return "Stress case output unavailable."

    downside_text = f"{downside:.1%}"

    descriptions = {
        "Base Case": (
            "Reference case using the current base DCF scenario assumptions. "
            "This is the anchor for measuring stress-test downside."
        ),
        "Federal Budget Delay": (
            "Models delayed award timing and slower near-term revenue conversion from federal budget or appropriations timing risk."
        ),
        "Continuing Resolution / Award Timing Delay": (
            "Models a sharper temporary procurement-delay environment where contract starts and backlog conversion are pushed out."
        ),
        "Higher WACC / Rate Shock": (
            "Models valuation pressure from a higher discount-rate environment and multiple compression."
        ),
        "Margin Compression": (
            "Models lower profitability from labor pressure, contract mix, integration costs, or execution issues."
        ),
        "Contract Recompete Loss": (
            "Models revenue and margin pressure from lost recompetes or lower-margin contract renewals."
        ),
        "Revenue Growth Slowdown": (
            "Models weaker growth from slower backlog conversion, lower federal modernization demand, or program delays."
        ),
        "Combined Downside Case": (
            "Models a combined stress case with slower growth, lower margin, higher WACC, and lower terminal growth."
        ),
    }

    base_description = descriptions.get(
        case_name,
        "Stress case measuring downside sensitivity versus the base case.",
    )

    return f"{base_description} Estimated downside versus base case: {downside_text}."


def _build_revenue_path(
    base_revenue_m: float,
    growth_rates: list[float],
) -> list[float]:
    """Build explicit revenue forecast path."""

    revenues = []
    current_revenue = base_revenue_m

    for growth_rate in growth_rates:
        current_revenue = current_revenue * (1 + growth_rate)
        revenues.append(current_revenue)

    return revenues


def _calculate_fcf_path(
    revenues: list[float],
    ebitda_margin: float,
    tax_rate: float,
    capex_pct_revenue: float,
    d_and_a_pct_revenue: float,
    nwc_pct_revenue_change: float,
) -> list[float]:
    """Estimate free cash flow path from revenue and operating assumptions."""

    fcf_values = []
    prior_revenue = BASE_REVENUE_M

    for revenue in revenues:
        ebitda = revenue * ebitda_margin
        depreciation = revenue * d_and_a_pct_revenue
        ebit = ebitda - depreciation
        nopat = ebit * (1 - tax_rate)
        capex = revenue * capex_pct_revenue
        change_in_nwc = (revenue - prior_revenue) * nwc_pct_revenue_change

        fcf = nopat + depreciation - capex - change_in_nwc
        fcf_values.append(fcf)

        prior_revenue = revenue

    return fcf_values


def _discount_cash_flows(
    fcf_values: list[float],
    discount_rate: float,
) -> float:
    """Discount explicit free cash flows."""

    pv = 0.0

    for year, fcf in enumerate(fcf_values, start=1):
        pv += fcf / ((1 + discount_rate) ** year)

    return pv


def _terminal_value(
    final_fcf: float,
    discount_rate: float,
    terminal_growth: float,
) -> float:
    """Calculate terminal value using Gordon growth model."""

    if discount_rate <= terminal_growth:
        return np.nan

    return final_fcf * (1 + terminal_growth) / (discount_rate - terminal_growth)


def _calculate_dcf_case(
    case_name: str,
    revenue_growth: list[float],
    ebitda_margin: float,
    discount_rate: float,
    terminal_growth: float,
    net_debt_m: float = NET_DEBT_M,
    diluted_shares_m: float = DILUTED_SHARES_M,
) -> dict:
    """Calculate a DCF stress case."""

    revenues = _build_revenue_path(BASE_REVENUE_M, revenue_growth)

    fcf_values = _calculate_fcf_path(
        revenues=revenues,
        ebitda_margin=ebitda_margin,
        tax_rate=TAX_RATE,
        capex_pct_revenue=CAPEX_PCT_REVENUE,
        d_and_a_pct_revenue=D_AND_A_PCT_REVENUE,
        nwc_pct_revenue_change=NWC_PCT_REVENUE_CHANGE,
    )

    pv_explicit_fcf = _discount_cash_flows(fcf_values, discount_rate)
    terminal_value = _terminal_value(fcf_values[-1], discount_rate, terminal_growth)

    if pd.isna(terminal_value):
        pv_terminal_value = np.nan
        enterprise_value = np.nan
        equity_value = np.nan
        fair_value_per_share = np.nan
        implied_ev_ebitda = np.nan
    else:
        pv_terminal_value = terminal_value / ((1 + discount_rate) ** len(fcf_values))
        enterprise_value = pv_explicit_fcf + pv_terminal_value
        equity_value = enterprise_value - net_debt_m
        fair_value_per_share = equity_value / diluted_shares_m

        year_5_ebitda = revenues[-1] * ebitda_margin
        implied_ev_ebitda = (
            enterprise_value / year_5_ebitda
            if year_5_ebitda != 0
            else np.nan
        )

    revenue_cagr = (revenues[-1] / BASE_REVENUE_M) ** (1 / len(revenues)) - 1
    year_5_revenue = revenues[-1]
    year_5_ebitda = revenues[-1] * ebitda_margin
    year_5_fcf = fcf_values[-1]

    return {
        "Stress Case": case_name,
        "Fair Value Per Share": fair_value_per_share,
        "Fair Value Per Share Type": "Money",
        "Revenue CAGR": revenue_cagr,
        "Revenue CAGR Type": "Percent",
        "EBITDA Margin": ebitda_margin,
        "EBITDA Margin Type": "Percent",
        "WACC": discount_rate,
        "WACC Type": "Percent",
        "Terminal Growth": terminal_growth,
        "Terminal Growth Type": "Percent",
        "Year 5 Revenue": year_5_revenue,
        "Year 5 Revenue Type": "Money Millions",
        "Year 5 EBITDA": year_5_ebitda,
        "Year 5 EBITDA Type": "Money Millions",
        "Year 5 FCF": year_5_fcf,
        "Year 5 FCF Type": "Money Millions",
        "PV of Explicit FCF": pv_explicit_fcf,
        "PV of Explicit FCF Type": "Money Millions",
        "PV of Terminal Value": pv_terminal_value,
        "PV of Terminal Value Type": "Money Millions",
        "Enterprise Value": enterprise_value,
        "Enterprise Value Type": "Money Millions",
        "Equity Value": equity_value,
        "Equity Value Type": "Money Millions",
        "Implied EV/EBITDA": implied_ev_ebitda,
        "Implied EV/EBITDA Type": "Plain Number",
    }


def _base_growth_rates() -> list[float]:
    """Return base-case revenue growth assumptions."""

    return list(SCENARIOS["Base"]["revenue_growth"])


def _adjust_growth_rates(
    growth_rates: list[float],
    adjustment: float,
    floor: float = -0.05,
) -> list[float]:
    """Apply a flat adjustment to revenue growth assumptions."""

    return [max(rate + adjustment, floor) for rate in growth_rates]


def build_stress_case_inputs() -> pd.DataFrame:
    """Build stress case assumption table."""

    base_growth = _base_growth_rates()
    base_margin = SCENARIOS["Base"]["ebitda_margin"]
    base_wacc = SCENARIOS["Base"]["wacc"]
    base_terminal_growth = SCENARIOS["Base"]["terminal_growth"]

    cases = [
        {
            "Stress Case": "Base Case",
            "Revenue Growth Adjustment": 0.000,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": 0.000,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.000,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": 0.000,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Current base-case DCF assumptions.",
        },
        {
            "Stress Case": "Federal Budget Delay",
            "Revenue Growth Adjustment": -0.010,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": -0.003,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.0025,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": -0.0010,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Moderate award timing delay and slower backlog conversion.",
        },
        {
            "Stress Case": "Continuing Resolution / Award Timing Delay",
            "Revenue Growth Adjustment": -0.0175,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": -0.005,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.0050,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": -0.0015,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Sharper federal procurement delay and contract-start timing pressure.",
        },
        {
            "Stress Case": "Higher WACC / Rate Shock",
            "Revenue Growth Adjustment": -0.0025,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": -0.001,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.0100,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": -0.0025,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Higher discount-rate environment and valuation multiple pressure.",
        },
        {
            "Stress Case": "Margin Compression",
            "Revenue Growth Adjustment": -0.0025,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": -0.010,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.0025,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": -0.0010,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Lower profitability from labor pressure, mix, or integration costs.",
        },
        {
            "Stress Case": "Contract Recompete Loss",
            "Revenue Growth Adjustment": -0.0200,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": -0.0075,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.0040,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": -0.0020,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Lost recompetes or lower-margin renewals pressure revenue and margins.",
        },
        {
            "Stress Case": "Revenue Growth Slowdown",
            "Revenue Growth Adjustment": -0.0250,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": -0.0025,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.0025,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": -0.0015,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Broad growth slowdown from lower award conversion or program delays.",
        },
        {
            "Stress Case": "Combined Downside Case",
            "Revenue Growth Adjustment": -0.0300,
            "Revenue Growth Adjustment Type": "Percent",
            "EBITDA Margin Adjustment": -0.0125,
            "EBITDA Margin Adjustment Type": "Percent",
            "WACC Adjustment": 0.0125,
            "WACC Adjustment Type": "Percent",
            "Terminal Growth Adjustment": -0.0035,
            "Terminal Growth Adjustment Type": "Percent",
            "Description": "Combined lower growth, lower margin, higher WACC, and lower terminal growth.",
        },
    ]

    df = pd.DataFrame(cases)

    df["Base Revenue CAGR"] = (np.prod([1 + rate for rate in base_growth]) ** (1 / len(base_growth)) - 1)
    df["Base Revenue CAGR Type"] = "Percent"
    df["Base EBITDA Margin"] = base_margin
    df["Base EBITDA Margin Type"] = "Percent"
    df["Base WACC"] = base_wacc
    df["Base WACC Type"] = "Percent"
    df["Base Terminal Growth"] = base_terminal_growth
    df["Base Terminal Growth Type"] = "Percent"

    _safe_to_csv(df, "stress_test_assumptions.csv")

    return df


def build_stress_test_summary() -> pd.DataFrame:
    """Run stress cases and build summary table."""

    assumptions = build_stress_case_inputs()

    base_growth = _base_growth_rates()
    base_margin = SCENARIOS["Base"]["ebitda_margin"]
    base_wacc = SCENARIOS["Base"]["wacc"]
    base_terminal_growth = SCENARIOS["Base"]["terminal_growth"]

    case_rows = []

    for _, row in assumptions.iterrows():
        case_name = row["Stress Case"]

        revenue_adjustment = _safe_float(row["Revenue Growth Adjustment"], 0.0)
        margin_adjustment = _safe_float(row["EBITDA Margin Adjustment"], 0.0)
        wacc_adjustment = _safe_float(row["WACC Adjustment"], 0.0)
        terminal_growth_adjustment = _safe_float(row["Terminal Growth Adjustment"], 0.0)

        adjusted_growth = _adjust_growth_rates(base_growth, revenue_adjustment)
        adjusted_margin = max(base_margin + margin_adjustment, 0.01)
        adjusted_wacc = max(base_wacc + wacc_adjustment, 0.001)
        adjusted_terminal_growth = max(base_terminal_growth + terminal_growth_adjustment, 0.0)

        case_output = _calculate_dcf_case(
            case_name=case_name,
            revenue_growth=adjusted_growth,
            ebitda_margin=adjusted_margin,
            discount_rate=adjusted_wacc,
            terminal_growth=adjusted_terminal_growth,
        )

        case_output["Description"] = row["Description"]
        case_rows.append(case_output)

    summary = pd.DataFrame(case_rows)

    base_value_series = summary.loc[
        summary["Stress Case"] == "Base Case",
        "Fair Value Per Share",
    ]

    if base_value_series.empty:
        base_fair_value = np.nan
    else:
        base_fair_value = _safe_float(base_value_series.iloc[0])

    summary["Downside vs Base Case"] = summary["Fair Value Per Share"].apply(
        lambda value: value / base_fair_value - 1
        if pd.notna(value) and pd.notna(base_fair_value) and base_fair_value != 0
        else np.nan
    )
    summary["Downside vs Base Case Type"] = "Percent"

    summary["Stress Signal"] = summary["Fair Value Per Share"].apply(
        lambda value: _stress_signal(value, base_fair_value)
    )

    summary["Valuation Bias"] = summary["Fair Value Per Share"].apply(
        lambda value: _valuation_bias(value, base_fair_value)
    )

    summary["Interpretation"] = summary.apply(
        lambda row: _interpret_stress_case(
            row["Stress Case"],
            row["Downside vs Base Case"],
        ),
        axis=1,
    )

    display_columns = [
        "Stress Case",
        "Fair Value Per Share",
        "Fair Value Per Share Type",
        "Downside vs Base Case",
        "Downside vs Base Case Type",
        "Revenue CAGR",
        "Revenue CAGR Type",
        "EBITDA Margin",
        "EBITDA Margin Type",
        "WACC",
        "WACC Type",
        "Terminal Growth",
        "Terminal Growth Type",
        "Enterprise Value",
        "Enterprise Value Type",
        "Equity Value",
        "Equity Value Type",
        "Stress Signal",
        "Valuation Bias",
        "Interpretation",
        "Description",
    ]

    summary = summary[display_columns].copy()

    _safe_to_csv(summary, "stress_test_summary.csv")

    return summary


def build_stress_test_bridge(summary: pd.DataFrame) -> pd.DataFrame:
    """Build stress-test bridge versus base case."""

    if summary is None or summary.empty:
        return pd.DataFrame()

    if "Stress Case" not in summary.columns or "Fair Value Per Share" not in summary.columns:
        return pd.DataFrame()

    base_row = summary.loc[summary["Stress Case"] == "Base Case"]

    if base_row.empty:
        return pd.DataFrame()

    base_fair_value = _safe_float(base_row["Fair Value Per Share"].iloc[0])

    rows = []

    for _, row in summary.iterrows():
        case_name = row["Stress Case"]
        fair_value = _safe_float(row["Fair Value Per Share"])

        dollar_change = (
            fair_value - base_fair_value
            if pd.notna(fair_value) and pd.notna(base_fair_value)
            else np.nan
        )

        percent_change = (
            dollar_change / base_fair_value
            if pd.notna(dollar_change) and pd.notna(base_fair_value) and base_fair_value != 0
            else np.nan
        )

        rows.append(
            {
                "Stress Case": case_name,
                "Base Fair Value": base_fair_value,
                "Base Fair Value Type": "Money",
                "Stress Fair Value": fair_value,
                "Stress Fair Value Type": "Money",
                "Dollar Impact": dollar_change,
                "Dollar Impact Type": "Money",
                "Percent Impact": percent_change,
                "Percent Impact Type": "Percent",
                "Stress Signal": row.get("Stress Signal", "Unavailable"),
                "Interpretation": (
                    f"{case_name} changes estimated fair value by "
                    f"{percent_change:.1%} versus the base case."
                    if pd.notna(percent_change)
                    else f"{case_name} stress impact unavailable."
                ),
            }
        )

    bridge = pd.DataFrame(rows)
    _safe_to_csv(bridge, "stress_test_bridge.csv")

    return bridge


def build_stress_test_score(summary: pd.DataFrame) -> pd.DataFrame:
    """Build compact stress-test score table."""

    if summary is None or summary.empty:
        return pd.DataFrame()

    non_base = summary[summary["Stress Case"] != "Base Case"].copy()

    if non_base.empty:
        return pd.DataFrame()

    non_base["Downside vs Base Case"] = pd.to_numeric(
        non_base["Downside vs Base Case"],
        errors="coerce",
    )

    average_downside = non_base["Downside vs Base Case"].mean()
    worst_downside = non_base["Downside vs Base Case"].min()

    worst_case_row = non_base.loc[
        non_base["Downside vs Base Case"].idxmin()
    ]

    # Convert downside into a 0-100 resilience score.
    # A 0% average downside maps near 100, while -40% or worse maps near 0.
    stress_resilience_score = max(
        0,
        min(
            100,
            100 + (average_downside / 0.40) * 100,
        ),
    )

    if stress_resilience_score >= 80:
        signal = "High Stress Resilience"
    elif stress_resilience_score >= 65:
        signal = "Moderate / High Stress Resilience"
    elif stress_resilience_score >= 50:
        signal = "Moderate Stress Resilience"
    else:
        signal = "Low Stress Resilience"

    rows = [
        {
            "Metric": "Stress Resilience Score",
            "Value": stress_resilience_score,
            "Value Type": "Score",
            "Signal": signal,
            "Interpretation": "Score derived from average downside across non-base stress cases.",
        },
        {
            "Metric": "Average Stress Downside",
            "Value": average_downside,
            "Value Type": "Percent",
            "Signal": "Reference",
            "Interpretation": "Average downside versus base case across all non-base stress scenarios.",
        },
        {
            "Metric": "Worst Stress Downside",
            "Value": worst_downside,
            "Value Type": "Percent",
            "Signal": "Reference",
            "Interpretation": "Largest downside versus base case across the stress scenarios.",
        },
        {
            "Metric": "Worst Stress Case",
            "Value": worst_case_row["Stress Case"],
            "Value Type": "Text",
            "Signal": worst_case_row.get("Stress Signal", "Unavailable"),
            "Interpretation": worst_case_row.get("Interpretation", ""),
        },
    ]

    score = pd.DataFrame(rows)
    _safe_to_csv(score, "stress_test_score.csv")

    return score


def run_stress_tests() -> dict[str, pd.DataFrame]:
    """Run the full defense contractor stress-testing module."""

    try:
        summary = build_stress_test_summary()
        bridge = build_stress_test_bridge(summary)
        score = build_stress_test_score(summary)
        assumptions = read_stress_test_assumptions()

        return {
            "stress_test_summary": summary,
            "stress_test_bridge": bridge,
            "stress_test_score": score,
            "stress_test_assumptions": assumptions,
        }

    except Exception as exc:
        print(f"Warning: stress testing failed: {exc}")

        fallback = pd.DataFrame(
            [
                {
                    "Metric": "Stress Testing Module",
                    "Value": "Unavailable",
                    "Value Type": "Text",
                    "Signal": "Unavailable",
                    "Interpretation": f"Stress testing failed or data was unavailable: {exc}",
                }
            ]
        )

        _safe_to_csv(fallback, "stress_test_score.csv")

        return {
            "stress_test_summary": pd.DataFrame(),
            "stress_test_bridge": pd.DataFrame(),
            "stress_test_score": fallback,
            "stress_test_assumptions": pd.DataFrame(),
        }


def read_stress_test_assumptions() -> pd.DataFrame:
    """Read the saved stress-test assumptions table."""

    path = TABLES_DIR / "stress_test_assumptions.csv"

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read stress_test_assumptions.csv: {exc}")
        return pd.DataFrame()