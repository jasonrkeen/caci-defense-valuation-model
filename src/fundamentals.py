import pandas as pd

from src.config import (
    TABLES_DIR,
    BASE_REVENUE_M,
    BASE_EBITDA_MARGIN,
    BASE_FCF_M,
    DILUTED_SHARES_M,
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
    """Convert a business quality score into a plain-English signal."""

    if pd.isna(score):
        return "Unavailable"
    if score >= 85:
        return "Strong"
    if score >= 75:
        return "Constructive"
    if score >= 60:
        return "Neutral / Monitor"
    if score >= 50:
        return "Monitor"
    return "Weak"


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


def _add_business_quality_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add score-driven signal, weighted contribution, and formatting metadata."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    result["Score"] = pd.to_numeric(result["Score"], errors="coerce")
    result["Score Type"] = "Score"

    result["Weight"] = pd.to_numeric(result["Weight"], errors="coerce")
    result["Weight Type"] = "Percent"

    result["Weighted Contribution"] = result["Score"] * result["Weight"]
    result["Weighted Contribution Type"] = "Score"

    result["Signal"] = result["Score"].apply(_score_to_signal)

    return result


def build_fundamental_summary() -> pd.DataFrame:
    """Build the core fundamental inputs used across the CACI model."""

    data = [
        {
            "Category": "Revenue Base",
            "Metric": "FY2026 revenue guidance midpoint",
            "Value": f"${BASE_REVENUE_M / 1000:,.2f}B",
            "Value Type": "Text",
            "Model Use": "Starting revenue base for DCF",
            "Signal": "Positive",
        },
        {
            "Category": "Profitability",
            "Metric": "EBITDA margin assumption",
            "Value": BASE_EBITDA_MARGIN,
            "Value Type": "Percent",
            "Model Use": "Base-case operating margin proxy informed by Q3 FY2026 margin and guidance",
            "Signal": "Constructive",
        },
        {
            "Category": "Cash Flow",
            "Metric": "Free cash flow guidance",
            "Value": f"At least ${BASE_FCF_M:,.0f}M",
            "Value Type": "Text",
            "Model Use": "Cross-check for DCF free cash flow level",
            "Signal": "Positive",
        },
        {
            "Category": "Capital Structure",
            "Metric": "Diluted shares",
            "Value": f"{DILUTED_SHARES_M:,.1f}M",
            "Value Type": "Text",
            "Model Use": "Equity value per share conversion",
            "Signal": "Important",
        },
        {
            "Category": "Customer Mix",
            "Metric": "Department of Defense revenue exposure",
            "Value": CACI_CUSTOMER_MIX.get("Department of Defense", 0),
            "Value Type": "Percent",
            "Model Use": "Supports link to U.S. defense spending and budget cycle",
            "Signal": "High Relevance",
        },
        {
            "Category": "Customer Mix",
            "Metric": "Intelligence Community revenue exposure",
            "Value": CACI_CUSTOMER_MIX.get("Intelligence Community", 0),
            "Value Type": "Percent",
            "Model Use": "Supports intelligence-contractor thesis and NIP/MIP spending relevance",
            "Signal": "High Relevance",
        },
        {
            "Category": "Technology Mix",
            "Metric": "Technology revenue mix",
            "Value": CACI_TECHNOLOGY_MIX.get("Technology Revenue Mix", 0),
            "Value Type": "Percent",
            "Model Use": "Supports defense technology premium versus lower-growth services contractors",
            "Signal": "Positive",
        },
        {
            "Category": "Technology Mix",
            "Metric": "Technology revenue growth",
            "Value": CACI_TECHNOLOGY_MIX.get("Technology Revenue Growth", 0),
            "Value Type": "Percent",
            "Model Use": "Supports higher-quality growth thesis",
            "Signal": "Positive",
        },
        {
            "Category": "Backlog / Visibility",
            "Metric": "Backlog and contract award visibility",
            "Value": "Strong",
            "Value Type": "Text",
            "Model Use": "Supports forward revenue visibility and qualitative business quality score",
            "Signal": "Positive",
        },
        {
            "Category": "Business Quality",
            "Metric": "Defense technology / national security exposure",
            "Value": "High",
            "Value Type": "Text",
            "Model Use": "Supports qualitative intelligence score",
            "Signal": "Positive",
        },
    ]

    df = pd.DataFrame(data)
    _safe_to_csv(df, "fundamental_summary.csv")

    return df


def build_business_quality_summary() -> pd.DataFrame:
    """Build a qualitative business quality table for the report."""

    rows = [
        {
            "Quality Driver": "Mission-Critical Demand",
            "Score": 85,
            "Weight": 0.20,
            "Rationale": (
                "CACI serves national security, defense, intelligence, cyber, "
                "and federal technology missions."
            ),
            "Valuation Link": "Supports revenue durability and terminal demand confidence",
        },
        {
            "Quality Driver": "Defense Technology Positioning",
            "Score": 86,
            "Weight": 0.20,
            "Rationale": (
                "Technology revenue mix and cyber/intelligence exposure support a differentiated "
                "defense technology thesis."
            ),
            "Valuation Link": "Supports valuation premium versus lower-growth services contractors",
        },
        {
            "Quality Driver": "Revenue Visibility",
            "Score": 82,
            "Weight": 0.18,
            "Rationale": (
                "Backlog, long-duration awards, and contract award momentum support forward visibility."
            ),
            "Valuation Link": "Supports base-case revenue confidence",
        },
        {
            "Quality Driver": "Customer Concentration",
            "Score": 58,
            "Weight": 0.13,
            "Rationale": (
                "Heavy U.S. government exposure creates policy, budget, appropriations, "
                "and procurement timing risk."
            ),
            "Valuation Link": "Raises WACC and scenario sensitivity risk",
        },
        {
            "Quality Driver": "Cash Flow Conversion",
            "Score": 76,
            "Weight": 0.17,
            "Rationale": (
                "FY2026 free cash flow guidance supports DCF quality and shareholder-value framework."
            ),
            "Valuation Link": "Supports fair value per share and expected return model",
        },
        {
            "Quality Driver": "Acquisition Execution",
            "Score": 62,
            "Weight": 0.12,
            "Rationale": (
                "Acquisitions can expand technology capabilities but introduce integration "
                "and execution risk."
            ),
            "Valuation Link": "Impacts margin, free cash flow conversion, and leverage",
        },
    ]

    df = pd.DataFrame(rows)
    df = _add_business_quality_fields(df)

    df = df[
        [
            "Quality Driver",
            "Score",
            "Score Type",
            "Weight",
            "Weight Type",
            "Weighted Contribution",
            "Weighted Contribution Type",
            "Signal",
            "Rationale",
            "Valuation Link",
        ]
    ]

    _safe_to_csv(df, "business_quality_summary.csv")

    return df


def build_business_quality_score_summary(
    business_quality_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Create a compact business quality score summary."""

    if business_quality_df is None:
        business_quality_df = build_business_quality_summary()

    if business_quality_df is None or business_quality_df.empty:
        df = pd.DataFrame(
            [
                {
                    "Metric": "Business Quality Score",
                    "Value": 0.0,
                    "Value Type": "Score",
                    "Signal": "Unavailable",
                    "Interpretation": "No business quality drivers were available.",
                }
            ]
        )
        _safe_to_csv(df, "business_quality_score_summary.csv")
        return df

    score = round(_weighted_score(business_quality_df), 1)
    signal = _score_to_signal(score)

    strongest_driver = business_quality_df.sort_values(
        "Score",
        ascending=False,
    ).iloc[0]

    weakest_driver = business_quality_df.sort_values(
        "Score",
        ascending=True,
    ).iloc[0]

    rows = [
        {
            "Metric": "Business Quality Score",
            "Value": score,
            "Value Type": "Score",
            "Signal": signal,
            "Interpretation": (
                "Weighted score summarizing CACI's mission demand, technology positioning, "
                "revenue visibility, customer concentration, cash flow conversion, and acquisition execution."
            ),
        },
        {
            "Metric": "Strongest Business Quality Driver",
            "Value": strongest_driver["Quality Driver"],
            "Value Type": "Text",
            "Signal": strongest_driver["Signal"],
            "Interpretation": strongest_driver["Rationale"],
        },
        {
            "Metric": "Weakest Business Quality Driver",
            "Value": weakest_driver["Quality Driver"],
            "Value Type": "Text",
            "Signal": weakest_driver["Signal"],
            "Interpretation": weakest_driver["Rationale"],
        },
        {
            "Metric": "Primary Quality Tailwind",
            "Value": "Mission-critical demand and defense technology positioning",
            "Value Type": "Text",
            "Signal": "Constructive",
            "Interpretation": (
                "CACI's national-security mission exposure and technology revenue mix support "
                "a higher-quality business profile."
            ),
        },
        {
            "Metric": "Primary Quality Risk",
            "Value": "Customer concentration and acquisition execution",
            "Value Type": "Text",
            "Signal": "Monitor",
            "Interpretation": (
                "Government concentration, budget timing, and integration risk remain the main "
                "business-quality watch items."
            ),
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "business_quality_score_summary.csv")

    return df


def build_valuation_driver_summary() -> pd.DataFrame:
    """Build a table connecting fundamental drivers to valuation assumptions."""

    rows = [
        {
            "Valuation Driver": "Revenue Growth",
            "Company Evidence": "FY2026 guidance, backlog, contract awards, defense and intelligence demand",
            "Model Connection": "Bear/base/bull revenue growth assumptions",
            "Primary Risk": "Budget timing, recompete losses, procurement delays",
        },
        {
            "Valuation Driver": "EBITDA Margin",
            "Company Evidence": "Base-case margin assumption and technology mix",
            "Model Connection": "DCF EBITDA margin and FCF generation",
            "Primary Risk": "Labor costs, contract mix, integration costs, execution risk",
        },
        {
            "Valuation Driver": "Free Cash Flow",
            "Company Evidence": "FY2026 FCF guidance of at least $725M",
            "Model Connection": "DCF free cash flow cross-check and expected return confidence",
            "Primary Risk": "Working capital timing, capex, acquisition integration",
        },
        {
            "Valuation Driver": "Discount Rate / WACC",
            "Company Evidence": "Government customer concentration and interest rate environment",
            "Model Connection": "DCF present value and terminal value sensitivity",
            "Primary Risk": "Higher Treasury yields, credit spreads, valuation multiple compression",
        },
        {
            "Valuation Driver": "Terminal Growth",
            "Company Evidence": "Long-duration national security, cyber, intelligence, and modernization demand",
            "Model Connection": "Terminal value and sensitivity table",
            "Primary Risk": "Policy shifts, lower federal spending growth, competitive pressure",
        },
        {
            "Valuation Driver": "Defense Technology Premium",
            "Company Evidence": "Technology revenue mix, cyber/intelligence exposure, mission-critical programs",
            "Model Connection": "Qualitative score, peer valuation interpretation, expected-return confidence",
            "Primary Risk": "If growth/margins resemble lower-growth government services peers",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "valuation_driver_summary.csv")

    return df


def build_thesis_points() -> list[str]:
    """Return the core investment thesis bullets used in the PDF report."""

    return [
        "CACI benefits from long-duration national security, intelligence, cyber, and federal technology demand.",
        "The company has a differentiated defense technology profile relative to traditional government IT services firms.",
        "Backlog, contract awards, acquisitions, technology mix, and mission-critical programs are important valuation drivers.",
        "Management's FY2026 guidance and three-year framework support a valuation model centered on high-single-digit revenue growth, disciplined margins, and free cash flow expansion.",
        "U.S. defense spending, public intelligence budget toplines, cyber demand, and great-power competition support the broader demand backdrop.",
        "The main risks are federal budget timing, contract recompete pressure, margin execution, customer concentration, acquisition integration, and valuation multiple compression.",
    ]


def build_thesis_table() -> pd.DataFrame:
    """Save thesis points as a CSV table for documentation and report use."""

    rows = [
        {
            "Thesis Point": str(idx + 1),
            "Thesis Point Type": "Text",
            "Description": point,
        }
        for idx, point in enumerate(build_thesis_points())
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "investment_thesis_points.csv")

    return df


def build_complete_fundamental_package() -> dict[str, pd.DataFrame]:
    """Run the full fundamentals package."""

    fundamental_summary = build_fundamental_summary()
    business_quality = build_business_quality_summary()
    business_quality_score_summary = build_business_quality_score_summary(business_quality)
    valuation_drivers = build_valuation_driver_summary()
    thesis_table = build_thesis_table()

    return {
        "fundamental_summary": fundamental_summary,
        "business_quality_summary": business_quality,
        "business_quality_score_summary": business_quality_score_summary,
        "valuation_driver_summary": valuation_drivers,
        "investment_thesis_points": thesis_table,
    }