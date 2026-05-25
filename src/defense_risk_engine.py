import pandas as pd

from src.config import TABLES_DIR


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _score_to_signal(score: float) -> str:
    """Convert a risk score into a plain-English signal.

    Higher score = lower risk / stronger positioning.
    """

    if pd.isna(score):
        return "Unavailable"
    if score >= 80:
        return "Strong"
    if score >= 70:
        return "Constructive"
    if score >= 60:
        return "Neutral / Monitor"
    if score >= 50:
        return "Elevated Risk"
    return "High Risk"


def _risk_direction_from_score(score: float) -> str:
    """Convert a score into a risk-direction label."""

    if pd.isna(score):
        return "Unavailable"
    if score >= 80:
        return "Favorable"
    if score >= 70:
        return "Constructive"
    if score >= 60:
        return "Moderate"
    if score >= 50:
        return "Elevated"
    return "High"


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
    """Add score-driven fields and formatting metadata to the component table."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    result["Score"] = pd.to_numeric(result["Score"], errors="coerce")
    result["Score Type"] = "Score"

    result["Weight"] = pd.to_numeric(result["Weight"], errors="coerce")
    result["Weight Type"] = "Percent"

    result["Weighted Contribution"] = result["Score"] * result["Weight"]
    result["Weighted Contribution Type"] = "Score"

    result["Risk Direction"] = result["Score"].apply(_risk_direction_from_score)
    result["Signal"] = result["Score"].apply(_score_to_signal)

    return result


def build_defense_risk_components() -> pd.DataFrame:
    """Build CACI-specific defense and national security risk components.

    Scores are interpreted as quality / resilience scores:
    - 100 = very favorable / lower risk
    - 50 = neutral or mixed
    - 0 = unfavorable / high risk
    """

    rows = [
        {
            "Risk Component": "Federal Budget / Appropriations Risk",
            "Score": 63,
            "Weight": 0.15,
            "Rationale": (
                "CACI benefits from durable national-security demand, but federal budget timing, "
                "continuing resolutions, shutdown risk, and appropriations delays can affect "
                "award timing, contract starts, and working-capital cadence."
            ),
            "Model Impact": "Revenue growth, backlog conversion, working capital timing",
        },
        {
            "Risk Component": "Contract Recompete Risk",
            "Score": 66,
            "Weight": 0.12,
            "Rationale": (
                "CACI has long-duration contracts and strong backlog visibility, but recompetes "
                "remain a structural risk for government services and defense technology companies."
            ),
            "Model Impact": "Revenue retention, EBITDA margin, terminal growth",
        },
        {
            "Risk Component": "Backlog Visibility",
            "Score": 84,
            "Weight": 0.14,
            "Rationale": (
                "Large backlog, book-to-bill above 1.0x, and long award duration support forward "
                "revenue visibility and reduce near-term demand uncertainty."
            ),
            "Model Impact": "Revenue durability, scenario confidence, business quality score",
        },
        {
            "Risk Component": "Customer Concentration",
            "Score": 58,
            "Weight": 0.11,
            "Rationale": (
                "CACI has significant exposure to U.S. government customers, especially defense "
                "and intelligence customers. This creates durable mission demand, but also raises "
                "policy, budget, procurement, and contract-concentration sensitivity."
            ),
            "Model Impact": "WACC, terminal growth, revenue volatility",
        },
        {
            "Risk Component": "Defense Technology Positioning",
            "Score": 86,
            "Weight": 0.14,
            "Rationale": (
                "Technology revenue mix, cyber, intelligence, national security, and modernization exposure "
                "support a higher-quality business profile than commodity government IT services."
            ),
            "Model Impact": "Valuation multiple support, margin durability, qualitative premium",
        },
        {
            "Risk Component": "Acquisition Integration Risk",
            "Score": 62,
            "Weight": 0.09,
            "Rationale": (
                "Acquisitions can expand CACI's technology capabilities, but integration risk can affect "
                "margins, execution, leverage, and free cash flow conversion."
            ),
            "Model Impact": "EBITDA margin, free cash flow conversion, leverage",
        },
        {
            "Risk Component": "Cleared Labor / Talent Availability",
            "Score": 61,
            "Weight": 0.08,
            "Rationale": (
                "Defense and intelligence programs often require specialized cleared personnel. "
                "Hiring constraints can limit execution speed, increase compensation pressure, "
                "or delay program ramp-up."
            ),
            "Model Impact": "Revenue execution, gross margin, operating margin",
        },
        {
            "Risk Component": "Margin Execution",
            "Score": 70,
            "Weight": 0.10,
            "Rationale": (
                "Guidance and recent results support disciplined profitability, but mix, labor cost, "
                "contract type, recompete pricing, and integration can affect EBITDA margin."
            ),
            "Model Impact": "EBITDA margin, FCF margin, fair value per share",
        },
        {
            "Risk Component": "Interest Rate / WACC Risk",
            "Score": 57,
            "Weight": 0.07,
            "Rationale": (
                "Higher Treasury yields and credit costs can pressure DCF value through higher discount rates, "
                "lower terminal value, and tighter valuation multiples."
            ),
            "Model Impact": "WACC, terminal value, valuation sensitivity",
        },
    ]

    df = pd.DataFrame(rows)
    df = _add_component_fields(df)

    df = df[
        [
            "Risk Component",
            "Score",
            "Score Type",
            "Weight",
            "Weight Type",
            "Weighted Contribution",
            "Weighted Contribution Type",
            "Risk Direction",
            "Signal",
            "Rationale",
            "Model Impact",
        ]
    ]

    _safe_to_csv(df, "defense_risk_components.csv")

    return df


def build_defense_risk_summary(risk_components: pd.DataFrame | None = None) -> pd.DataFrame:
    """Create a summary table for the defense risk engine."""

    if risk_components is None:
        risk_components = build_defense_risk_components()

    if risk_components is None or risk_components.empty:
        summary = pd.DataFrame(
            [
                {
                    "Metric": "Overall Defense Risk Score",
                    "Value": 0.0,
                    "Value Type": "Score",
                    "Signal": "Unavailable",
                    "Interpretation": "No defense risk components were available.",
                }
            ]
        )
        _safe_to_csv(summary, "defense_risk_summary.csv")
        return summary

    overall_score = round(_weighted_score(risk_components), 1)
    overall_signal = _score_to_signal(overall_score)

    strongest_component = risk_components.sort_values("Score", ascending=False).iloc[0]
    weakest_component = risk_components.sort_values("Score", ascending=True).iloc[0]

    summary = pd.DataFrame(
        [
            {
                "Metric": "Overall Defense Risk Score",
                "Value": overall_score,
                "Value Type": "Score",
                "Signal": overall_signal,
                "Interpretation": (
                    "Weighted resilience score across CACI-specific defense, government, "
                    "contracting, execution, margin, and discount-rate risks."
                ),
            },
            {
                "Metric": "Strongest Risk Component",
                "Value": strongest_component["Risk Component"],
                "Value Type": "Text",
                "Signal": strongest_component["Signal"],
                "Interpretation": strongest_component["Rationale"],
            },
            {
                "Metric": "Weakest Risk Component",
                "Value": weakest_component["Risk Component"],
                "Value Type": "Text",
                "Signal": weakest_component["Signal"],
                "Interpretation": weakest_component["Rationale"],
            },
            {
                "Metric": "Primary Positive Driver",
                "Value": "Backlog visibility and defense technology positioning",
                "Value Type": "Text",
                "Signal": "Constructive",
                "Interpretation": (
                    "Backlog durability, book-to-bill momentum, and technology-oriented demand "
                    "support revenue visibility and qualitative valuation support."
                ),
            },
            {
                "Metric": "Primary Risk Driver",
                "Value": "Customer concentration, federal budget timing, and WACC sensitivity",
                "Value Type": "Text",
                "Signal": "Monitor",
                "Interpretation": (
                    "The most important risk factors are policy and budget timing, contract concentration, "
                    "and valuation sensitivity to discount-rate assumptions."
                ),
            },
        ]
    )

    _safe_to_csv(summary, "defense_risk_summary.csv")

    return summary


def build_defense_risk_adjustments(risk_components: pd.DataFrame | None = None) -> pd.DataFrame:
    """Translate the defense risk engine into valuation adjustment logic.

    This table does not directly change the DCF. Instead, it gives the report a
    structured way to explain when a valuation premium or haircut is appropriate.
    """

    if risk_components is None:
        risk_components = build_defense_risk_components()

    overall_score = round(_weighted_score(risk_components), 1)
    overall_signal = _score_to_signal(overall_score)

    if overall_score >= 80:
        valuation_adjustment = 0.050
        confidence_adjustment = 0.050
        interpretation = "Risk profile supports a modest valuation premium."
    elif overall_score >= 70:
        valuation_adjustment = 0.025
        confidence_adjustment = 0.025
        interpretation = "Risk profile supports a small valuation premium."
    elif overall_score >= 60:
        valuation_adjustment = 0.000
        confidence_adjustment = 0.000
        interpretation = "Risk profile is balanced; no valuation adjustment applied."
    elif overall_score >= 50:
        valuation_adjustment = -0.025
        confidence_adjustment = -0.025
        interpretation = "Risk profile warrants a modest valuation haircut."
    else:
        valuation_adjustment = -0.050
        confidence_adjustment = -0.050
        interpretation = "Risk profile warrants a larger valuation haircut."

    adjustments = pd.DataFrame(
        [
            {
                "Adjustment": "Defense Risk Valuation Adjustment",
                "Value": valuation_adjustment,
                "Value Type": "Percent",
                "Signal": overall_signal,
                "Interpretation": interpretation,
            },
            {
                "Adjustment": "Confidence Score Adjustment",
                "Value": confidence_adjustment,
                "Value Type": "Percent",
                "Signal": overall_signal,
                "Interpretation": "Used as optional input for expected return or final signal models.",
            },
            {
                "Adjustment": "Overall Defense Risk Score",
                "Value": overall_score,
                "Value Type": "Score",
                "Signal": overall_signal,
                "Interpretation": (
                    "Weighted score summarizing CACI's defense risk and resilience profile."
                ),
            },
        ]
    )

    _safe_to_csv(adjustments, "defense_risk_adjustments.csv")

    return adjustments


def run_defense_risk_engine() -> dict[str, pd.DataFrame]:
    """Run the full defense risk engine."""

    components = build_defense_risk_components()
    summary = build_defense_risk_summary(components)
    adjustments = build_defense_risk_adjustments(components)

    return {
        "defense_risk_components": components,
        "defense_risk_summary": summary,
        "defense_risk_adjustments": adjustments,
    }