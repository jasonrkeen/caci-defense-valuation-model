import pandas as pd

from src.config import (
    TABLES_DIR,
    CACI_CUSTOMER_MIX,
    CACI_TECHNOLOGY_MIX,
)


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _score_to_signal(score: float) -> str:
    """Convert a macro score into a plain-English signal."""

    if pd.isna(score):
        return "Unavailable"
    if score >= 85:
        return "Strong Tailwind"
    if score >= 75:
        return "Constructive Tailwind"
    if score >= 60:
        return "Stable / Supportive"
    if score >= 50:
        return "Mixed"
    return "Headwind"


def _weighted_score(df: pd.DataFrame) -> float:
    """Calculate weighted score from Score and Weight columns."""

    if df is None or df.empty:
        return 0.0

    required_columns = {"Score", "Weight"}

    if not required_columns.issubset(df.columns):
        return 0.0

    scores = pd.to_numeric(df["Score"], errors="coerce")
    weights = pd.to_numeric(df["Weight"], errors="coerce")

    valid = scores.notna() & weights.notna()

    if not valid.any():
        return 0.0

    total_weight = weights[valid].sum()

    if total_weight == 0:
        return 0.0

    return float((scores[valid] * weights[valid]).sum() / total_weight)


def _add_component_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add score-driven fields and formatting metadata to the defense macro component table."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    result["Score"] = pd.to_numeric(result["Score"], errors="coerce")
    result["Score Type"] = "Score"

    result["Weight"] = pd.to_numeric(result["Weight"], errors="coerce")
    result["Weight Type"] = "Percent"

    result["Signal"] = result["Score"].apply(_score_to_signal)

    return result


def build_defense_macro_components() -> pd.DataFrame:
    """Build broad defense macro component scores.

    This is a qualitative scoring layer designed to translate the defense demand
    environment into a structured signal for the valuation framework.
    """

    rows = [
        {
            "Component": "U.S. Defense Budget Environment",
            "Score": 78,
            "Weight": 0.18,
            "Interpretation": (
                "The defense budget environment remains supportive for contractors "
                "with exposure to modernization, cyber, intelligence, and mission technology."
            ),
        },
        {
            "Component": "Intelligence and Cyber Demand",
            "Score": 84,
            "Weight": 0.18,
            "Interpretation": (
                "CACI's Intelligence Community and cyber exposure are aligned with durable "
                "national-security priorities."
            ),
        },
        {
            "Component": "Geopolitical Risk Backdrop",
            "Score": 76,
            "Weight": 0.14,
            "Interpretation": (
                "Elevated geopolitical risk supports demand for defense readiness, "
                "secure communications, electronic warfare, and intelligence capabilities."
            ),
        },
        {
            "Component": "Technology Modernization Priority",
            "Score": 86,
            "Weight": 0.16,
            "Interpretation": (
                "Federal demand continues to favor contractors with technology, software, "
                "data, cyber, AI, and mission systems exposure."
            ),
        },
        {
            "Component": "Federal Procurement and Continuing Resolution Risk",
            "Score": 58,
            "Weight": 0.12,
            "Interpretation": (
                "Procurement timing, budget delays, and continuing resolutions can shift "
                "award timing even when long-term demand remains favorable."
            ),
        },
        {
            "Component": "Labor and Cleared Talent Constraint",
            "Score": 61,
            "Weight": 0.10,
            "Interpretation": (
                "Cleared labor availability remains a constraint, but established defense "
                "technology contractors can benefit from scale and incumbency."
            ),
        },
        {
            "Component": "Interest Rate and Discount Rate Backdrop",
            "Score": 55,
            "Weight": 0.07,
            "Interpretation": (
                "Higher rates can pressure valuation multiples and DCF discount rates, "
                "partly offsetting defense-demand resilience."
            ),
        },
        {
            "Component": "Defense Contractor Multiple Environment",
            "Score": 64,
            "Weight": 0.05,
            "Interpretation": (
                "Defense technology peers can sustain premium valuation support when backlog, "
                "growth, and cash conversion remain visible."
            ),
        },
    ]

    df = pd.DataFrame(rows)
    df = _add_component_fields(df)

    df = df[
        [
            "Component",
            "Score",
            "Score Type",
            "Weight",
            "Weight Type",
            "Signal",
            "Interpretation",
        ]
    ]

    _safe_to_csv(df, "defense_macro_components.csv")

    return df


def build_defense_macro_summary(
    components_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create the headline defense macro signal."""

    if components_df is None:
        components_df = build_defense_macro_components()

    score = round(_weighted_score(components_df), 1)
    signal = _score_to_signal(score)

    rows = [
        {
            "Metric": "Defense Macro Score",
            "Value": score,
            "Value Type": "Score",
            "Signal": signal,
            "Interpretation": (
                "Weighted macro score summarizing budget, geopolitical, cyber, technology, "
                "procurement, labor, rate, and valuation conditions."
            ),
        },
        {
            "Metric": "Primary Macro Tailwind",
            "Value": "Technology modernization and intelligence/cyber demand",
            "Value Type": "Text",
            "Signal": "Constructive",
            "Interpretation": (
                "CACI's defense technology and intelligence exposure align with durable "
                "federal modernization priorities."
            ),
        },
        {
            "Metric": "Primary Macro Risk",
            "Value": "Procurement timing and discount-rate pressure",
            "Value Type": "Text",
            "Signal": "Monitor",
            "Interpretation": (
                "Budget timing delays and higher discount rates can pressure near-term "
                "valuation even when fundamental demand is supportive."
            ),
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "defense_macro_summary.csv")

    return df


def build_caci_macro_exposure_table() -> pd.DataFrame:
    """Bridge CACI's revenue mix to the defense macro environment."""

    rows = [
        {
            "Exposure Category": "Department of Defense",
            "CACI Revenue Exposure": f"{CACI_CUSTOMER_MIX.get('Department of Defense', 0):.1%}",
            "CACI Revenue Exposure Value Type": "Text",
            "Raw CACI Revenue Exposure": CACI_CUSTOMER_MIX.get("Department of Defense", 0),
            "Macro Link": (
                "Defense modernization, cyber, electronic warfare, space, communications, "
                "mission technology"
            ),
            "Valuation Relevance": (
                "Supports backlog durability and defense technology valuation premium"
            ),
            "Signal": "High Relevance",
        },
        {
            "Exposure Category": "Intelligence Community",
            "CACI Revenue Exposure": f"{CACI_CUSTOMER_MIX.get('Intelligence Community', 0):.1%}",
            "CACI Revenue Exposure Value Type": "Text",
            "Raw CACI Revenue Exposure": CACI_CUSTOMER_MIX.get("Intelligence Community", 0),
            "Macro Link": (
                "National intelligence, classified missions, signals intelligence, "
                "cyber operations"
            ),
            "Valuation Relevance": (
                "Supports intelligence-contractor thesis and demand resilience"
            ),
            "Signal": "High Relevance",
        },
        {
            "Exposure Category": "Federal Civilian Agencies",
            "CACI Revenue Exposure": f"{CACI_CUSTOMER_MIX.get('Federal Civilian Agencies', 0):.1%}",
            "CACI Revenue Exposure Value Type": "Text",
            "Raw CACI Revenue Exposure": CACI_CUSTOMER_MIX.get("Federal Civilian Agencies", 0),
            "Macro Link": "Federal IT modernization, cybersecurity, digital transformation",
            "Valuation Relevance": (
                "Adds federal technology modernization exposure outside DoD"
            ),
            "Signal": "Moderate Relevance",
        },
        {
            "Exposure Category": "Technology Revenue Mix",
            "CACI Revenue Exposure": f"{CACI_TECHNOLOGY_MIX.get('Technology Revenue Mix', 0):.1%}",
            "CACI Revenue Exposure Value Type": "Text",
            "Raw CACI Revenue Exposure": CACI_TECHNOLOGY_MIX.get("Technology Revenue Mix", 0),
            "Macro Link": (
                "Software, cyber, data, AI, mission systems, and technology-enabled solutions"
            ),
            "Valuation Relevance": (
                "Supports higher-quality growth and potential margin resilience"
            ),
            "Signal": "High Relevance",
        },
        {
            "Exposure Category": "Technology Revenue Growth",
            "CACI Revenue Exposure": f"{CACI_TECHNOLOGY_MIX.get('Technology Revenue Growth', 0):.1%} YoY",
            "CACI Revenue Exposure Value Type": "Text",
            "Raw CACI Revenue Exposure": CACI_TECHNOLOGY_MIX.get("Technology Revenue Growth", 0),
            "Macro Link": "Demand shift toward higher-value technology solutions",
            "Valuation Relevance": "Supports bull-case revenue and margin assumptions",
            "Signal": "Constructive",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "caci_macro_exposure.csv")

    return df


def build_defense_macro_adjustments(
    summary_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Translate defense macro signal into valuation-model adjustments."""

    if summary_df is None:
        summary_df = build_defense_macro_summary()

    score_row = summary_df.loc[
        summary_df["Metric"] == "Defense Macro Score",
        "Value",
    ]

    score = float(score_row.iloc[0]) if not score_row.empty else 60.0

    if score >= 80:
        revenue_growth_adjustment = 0.0050
        wacc_adjustment = -0.0025
        terminal_growth_adjustment = 0.0010
        valuation_bias = "Positive"
    elif score >= 70:
        revenue_growth_adjustment = 0.0025
        wacc_adjustment = -0.0010
        terminal_growth_adjustment = 0.0005
        valuation_bias = "Moderately Positive"
    elif score >= 60:
        revenue_growth_adjustment = 0.0000
        wacc_adjustment = 0.0000
        terminal_growth_adjustment = 0.0000
        valuation_bias = "Neutral"
    elif score >= 50:
        revenue_growth_adjustment = -0.0025
        wacc_adjustment = 0.0015
        terminal_growth_adjustment = -0.0005
        valuation_bias = "Moderately Negative"
    else:
        revenue_growth_adjustment = -0.0050
        wacc_adjustment = 0.0030
        terminal_growth_adjustment = -0.0010
        valuation_bias = "Negative"

    rows = [
        {
            "Adjustment": "Revenue Growth Adjustment",
            "Value": revenue_growth_adjustment,
            "Value Type": "Percent",
            "Valuation Bias": valuation_bias,
            "Interpretation": (
                "Directional adjustment that could be applied to scenario revenue growth "
                "based on the defense macro environment."
            ),
        },
        {
            "Adjustment": "WACC Adjustment",
            "Value": wacc_adjustment,
            "Value Type": "Percent",
            "Valuation Bias": valuation_bias,
            "Interpretation": (
                "Directional discount-rate adjustment reflecting macro support or pressure."
            ),
        },
        {
            "Adjustment": "Terminal Growth Adjustment",
            "Value": terminal_growth_adjustment,
            "Value Type": "Percent",
            "Valuation Bias": valuation_bias,
            "Interpretation": (
                "Directional terminal-growth adjustment based on long-run defense demand visibility."
            ),
        },
        {
            "Adjustment": "Defense Macro Valuation Bias",
            "Value": valuation_bias,
            "Value Type": "Text",
            "Valuation Bias": valuation_bias,
            "Interpretation": (
                "Overall valuation bias derived from the defense macro score."
            ),
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "defense_macro_adjustments.csv")

    return df


def run_defense_macro_model() -> dict[str, pd.DataFrame]:
    """Run the full defense macro model."""

    components = build_defense_macro_components()
    summary = build_defense_macro_summary(components)
    exposure = build_caci_macro_exposure_table()
    adjustments = build_defense_macro_adjustments(summary)

    return {
        "defense_macro_components": components,
        "defense_macro_summary": summary,
        "caci_macro_exposure": exposure,
        "defense_macro_adjustments": adjustments,
    }