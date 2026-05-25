import numpy as np
import pandas as pd

from src.config import (
    BASE_REVENUE_M,
    BASE_EBITDA_MARGIN,
    NET_DEBT_M,
    DILUTED_SHARES_M,
    TAX_RATE,
    CAPEX_PCT_REVENUE,
    D_AND_A_PCT_REVENUE,
    NWC_PCT_REVENUE_CHANGE,
    WACC,
    TERMINAL_GROWTH,
    MONTE_CARLO_RUNS,
    RANDOM_SEED,
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


def _value_signal(value: float) -> str:
    """Convert fair value level into a plain-English signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 750:
        return "High Upside Case"
    if value >= 625:
        return "Constructive"
    if value >= 500:
        return "Neutral / Monitor"
    if value >= 400:
        return "Cautious"
    return "Downside Case"


def _return_signal(value: float) -> str:
    """Convert return value into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.25:
        return "Attractive Upside"
    if value >= 0.10:
        return "Constructive"
    if value >= -0.05:
        return "Neutral / Monitor"
    if value >= -0.15:
        return "Cautious"
    return "Unattractive"


def _assumption_signal(metric: str, value: float) -> str:
    """Assign a basic signal to Monte Carlo assumption averages."""

    if pd.isna(value):
        return "Unavailable"

    metric_lower = metric.lower()

    if "revenue cagr" in metric_lower:
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
        if value >= 0.03:
            return "High Terminal Growth"
        if value >= 0.02:
            return "Normal Range"
        return "Conservative"

    return "Reference"


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


def _add_simulation_type_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add type metadata to the full Monte Carlo simulation export."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    type_map = {
        "Fair Value Per Share": "Money",
        "Enterprise Value": "Money Millions",
        "Equity Value": "Money Millions",
        "Revenue CAGR": "Percent",
        "EBITDA Margin": "Percent",
        "WACC": "Percent",
        "Terminal Growth": "Percent",
        "Year 5 Revenue": "Money Millions",
        "Year 5 EBITDA": "Money Millions",
        "Year 5 Free Cash Flow": "Money Millions",
        "Implied EV/EBITDA": "Multiple",
        "Terminal Value % of EV": "Percent",
    }

    for column, value_type in type_map.items():
        if column in result.columns:
            result[f"{column} Type"] = value_type

    return result


def simulate_single_valuation(rng: np.random.Generator) -> dict:
    """Run one Monte Carlo valuation simulation.

    Returns both the fair value output and the assumptions used in the run.
    """

    annual_growth = rng.normal(0.06, 0.02, 5)
    annual_growth = np.clip(annual_growth, 0.00, 0.12)

    ebitda_margin = float(
        np.clip(
            rng.normal(BASE_EBITDA_MARGIN, 0.008),
            0.100,
            0.145,
        )
    )

    wacc = float(
        np.clip(
            rng.normal(WACC, 0.008),
            0.065,
            0.115,
        )
    )

    terminal_growth = float(
        np.clip(
            rng.normal(TERMINAL_GROWTH, 0.004),
            0.010,
            0.035,
        )
    )

    # Defensive guard against invalid terminal value math.
    if wacc <= terminal_growth + 0.005:
        wacc = terminal_growth + 0.005

    previous_revenue = BASE_REVENUE_M
    fcf_values = []
    revenue_values = []
    ebitda_values = []

    for growth in annual_growth:
        revenue = previous_revenue * (1 + growth)
        ebitda = revenue * ebitda_margin
        da = revenue * D_AND_A_PCT_REVENUE
        ebit = ebitda - da
        tax = max(ebit * TAX_RATE, 0)
        nopat = ebit - tax
        capex = revenue * CAPEX_PCT_REVENUE
        nwc = (revenue - previous_revenue) * NWC_PCT_REVENUE_CHANGE
        fcf = nopat + da - capex - nwc

        revenue_values.append(revenue)
        ebitda_values.append(ebitda)
        fcf_values.append(fcf)

        previous_revenue = revenue

    pv_fcf = sum(
        fcf / ((1 + wacc) ** (i + 1))
        for i, fcf in enumerate(fcf_values)
    )

    terminal_fcf = fcf_values[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcf / (wacc - terminal_growth)
    pv_terminal = terminal_value / ((1 + wacc) ** len(fcf_values))

    enterprise_value = pv_fcf + pv_terminal
    equity_value = enterprise_value - NET_DEBT_M
    fair_value = equity_value / DILUTED_SHARES_M

    revenue_cagr = (
        (revenue_values[-1] / BASE_REVENUE_M) ** (1 / len(revenue_values)) - 1
        if revenue_values
        else np.nan
    )

    implied_ev_ebitda = (
        enterprise_value / ebitda_values[-1]
        if ebitda_values and ebitda_values[-1] > 0
        else np.nan
    )

    terminal_value_pct_ev = (
        pv_terminal / enterprise_value
        if enterprise_value and enterprise_value > 0
        else np.nan
    )

    return {
        "Fair Value Per Share": fair_value,
        "Enterprise Value": enterprise_value,
        "Equity Value": equity_value,
        "Revenue CAGR": revenue_cagr,
        "EBITDA Margin": ebitda_margin,
        "WACC": wacc,
        "Terminal Growth": terminal_growth,
        "Year 5 Revenue": revenue_values[-1],
        "Year 5 EBITDA": ebitda_values[-1],
        "Year 5 Free Cash Flow": fcf_values[-1],
        "Implied EV/EBITDA": implied_ev_ebitda,
        "Terminal Value % of EV": terminal_value_pct_ev,
    }


def summarize_monte_carlo(
    df: pd.DataFrame,
    current_price: float | None = None,
) -> pd.DataFrame:
    """Create a PDF-ready summary of Monte Carlo valuation outputs."""

    if df is None or df.empty or "Fair Value Per Share" not in df.columns:
        summary = pd.DataFrame(
            [
                _summary_row(
                    metric="Monte Carlo Summary",
                    value="Monte Carlo output unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Monte Carlo simulation output was unavailable.",
                )
            ]
        )
        return summary

    fair_values = pd.to_numeric(df["Fair Value Per Share"], errors="coerce").dropna()

    if fair_values.empty:
        summary = pd.DataFrame(
            [
                _summary_row(
                    metric="Monte Carlo Summary",
                    value="Monte Carlo fair values unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Monte Carlo fair value outputs were unavailable.",
                )
            ]
        )
        return summary

    current_price = _safe_float(current_price)

    probability_above_current_price = (
        float((fair_values > current_price).mean())
        if not pd.isna(current_price) and current_price > 0
        else np.nan
    )

    median_expected_return = (
        fair_values.median() / current_price - 1
        if not pd.isna(current_price) and current_price > 0
        else np.nan
    )

    rows = [
        _summary_row(
            metric="Mean Fair Value",
            value=fair_values.mean(),
            value_type="Money",
            signal=_value_signal(fair_values.mean()),
            interpretation="Average simulated fair value per share.",
        ),
        _summary_row(
            metric="Median Fair Value",
            value=fair_values.median(),
            value_type="Money",
            signal=_value_signal(fair_values.median()),
            interpretation="Median simulated fair value per share.",
        ),
        _summary_row(
            metric="Fair Value Standard Deviation",
            value=fair_values.std(),
            value_type="Money",
            signal="Volatility Measure",
            interpretation="Dispersion of simulated fair value outcomes.",
        ),
        _summary_row(
            metric="P5 Downside Case",
            value=fair_values.quantile(0.05),
            value_type="Money",
            signal=_value_signal(fair_values.quantile(0.05)),
            interpretation="Fifth-percentile downside fair value case.",
        ),
        _summary_row(
            metric="P10 Downside Case",
            value=fair_values.quantile(0.10),
            value_type="Money",
            signal=_value_signal(fair_values.quantile(0.10)),
            interpretation="Tenth-percentile downside fair value case.",
        ),
        _summary_row(
            metric="P25 Lower Quartile",
            value=fair_values.quantile(0.25),
            value_type="Money",
            signal=_value_signal(fair_values.quantile(0.25)),
            interpretation="Twenty-fifth percentile fair value outcome.",
        ),
        _summary_row(
            metric="P75 Upper Quartile",
            value=fair_values.quantile(0.75),
            value_type="Money",
            signal=_value_signal(fair_values.quantile(0.75)),
            interpretation="Seventy-fifth percentile fair value outcome.",
        ),
        _summary_row(
            metric="P90 Upside Case",
            value=fair_values.quantile(0.90),
            value_type="Money",
            signal=_value_signal(fair_values.quantile(0.90)),
            interpretation="Ninetieth-percentile upside fair value case.",
        ),
        _summary_row(
            metric="P95 High Upside Case",
            value=fair_values.quantile(0.95),
            value_type="Money",
            signal=_value_signal(fair_values.quantile(0.95)),
            interpretation="Ninety-fifth-percentile high upside fair value case.",
        ),
        _summary_row(
            metric="Minimum Simulated Value",
            value=fair_values.min(),
            value_type="Money",
            signal=_value_signal(fair_values.min()),
            interpretation="Lowest simulated fair value outcome.",
        ),
        _summary_row(
            metric="Maximum Simulated Value",
            value=fair_values.max(),
            value_type="Money",
            signal=_value_signal(fair_values.max()),
            interpretation="Highest simulated fair value outcome.",
        ),
        _summary_row(
            metric="Probability Above Current Price",
            value=probability_above_current_price,
            value_type="Percent",
            signal=(
                _return_signal(probability_above_current_price - 0.50)
                if not pd.isna(probability_above_current_price)
                else "Unavailable"
            ),
            interpretation="Share of simulations where fair value exceeds current market price.",
        ),
        _summary_row(
            metric="Median Monte Carlo Expected Return",
            value=median_expected_return,
            value_type="Percent",
            signal=_return_signal(median_expected_return),
            interpretation="Median simulated fair value versus current market price.",
        ),
        _summary_row(
            metric="Mean Revenue CAGR",
            value=df["Revenue CAGR"].mean(),
            value_type="Percent",
            signal=_assumption_signal("Mean Revenue CAGR", df["Revenue CAGR"].mean()),
            interpretation="Average simulated five-year revenue CAGR assumption.",
        ),
        _summary_row(
            metric="Mean EBITDA Margin",
            value=df["EBITDA Margin"].mean(),
            value_type="Percent",
            signal=_assumption_signal("Mean EBITDA Margin", df["EBITDA Margin"].mean()),
            interpretation="Average simulated EBITDA margin assumption.",
        ),
        _summary_row(
            metric="Mean WACC",
            value=df["WACC"].mean(),
            value_type="Percent",
            signal=_assumption_signal("Mean WACC", df["WACC"].mean()),
            interpretation="Average simulated weighted average cost of capital.",
        ),
        _summary_row(
            metric="Mean Terminal Growth",
            value=df["Terminal Growth"].mean(),
            value_type="Percent",
            signal=_assumption_signal("Mean Terminal Growth", df["Terminal Growth"].mean()),
            interpretation="Average simulated terminal growth assumption.",
        ),
        _summary_row(
            metric="Mean Implied EV/EBITDA",
            value=df["Implied EV/EBITDA"].mean(),
            value_type="Multiple",
            signal="Valuation Multiple",
            interpretation="Average simulated implied EV/EBITDA multiple.",
        ),
        _summary_row(
            metric="Mean Terminal Value % of EV",
            value=df["Terminal Value % of EV"].mean(),
            value_type="Percent",
            signal="Terminal Value Dependency",
            interpretation="Average share of enterprise value represented by discounted terminal value.",
        ),
        _summary_row(
            metric="Simulation Count",
            value=len(df),
            value_type="Plain Number",
            signal="Reference",
            interpretation="Number of Monte Carlo valuation simulations.",
        ),
    ]

    summary = pd.DataFrame(rows)

    return summary


def run_monte_carlo(current_price: float | None = None) -> pd.DataFrame:
    """Run full Monte Carlo valuation workflow.

    Returns the full simulation table. Also saves:
    - monte_carlo_values.csv
    - monte_carlo_summary.csv
    """

    rng = np.random.default_rng(RANDOM_SEED)

    simulations = [
        simulate_single_valuation(rng)
        for _ in range(MONTE_CARLO_RUNS)
    ]

    df = pd.DataFrame(simulations)

    # Remove invalid extreme observations while keeping the model transparent.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Fair Value Per Share"])

    df = df[
        (df["Fair Value Per Share"] > 0)
        & (df["Enterprise Value"] > 0)
        & (df["Equity Value"] > 0)
    ].copy()

    df_for_export = _add_simulation_type_metadata(df)

    _safe_to_csv(df_for_export, "monte_carlo_values.csv")

    summary = summarize_monte_carlo(df, current_price=current_price)
    _safe_to_csv(summary, "monte_carlo_summary.csv")

    return df