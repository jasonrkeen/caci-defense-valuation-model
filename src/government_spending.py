import pandas as pd

from src.config import (
    TABLES_DIR,
    GOVERNMENT_SPENDING_TABLE,
    CACI_CUSTOMER_MIX,
)


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _score_to_signal(score: float) -> str:
    """Convert a spending score into a plain-English signal.

    Higher score = stronger government spending tailwind.
    """

    if pd.isna(score):
        return "Unavailable"
    if score >= 85:
        return "Strong Spending Tailwind"
    if score >= 75:
        return "Constructive Spending Tailwind"
    if score >= 60:
        return "Stable / Supportive"
    if score >= 50:
        return "Mixed"
    return "Spending Headwind"


def _component_signal(score: float) -> str:
    """Convert component score into a shorter component-level signal."""

    if pd.isna(score):
        return "Unavailable"
    if score >= 85:
        return "Strong"
    if score >= 75:
        return "Constructive"
    if score >= 60:
        return "Stable / Supportive"
    if score >= 50:
        return "Mixed / Monitor"
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


def _format_billions(amount) -> str:
    """Format a numeric amount in billions of dollars."""

    if amount is None or pd.isna(amount):
        return "Unavailable"

    try:
        return f"${float(amount):,.2f}B"
    except Exception:
        return str(amount)


def _add_component_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add score-driven fields and formatting metadata to the government spending component table."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    result["Score"] = pd.to_numeric(result["Score"], errors="coerce")
    result["Score Type"] = "Score"

    result["Weight"] = pd.to_numeric(result["Weight"], errors="coerce")
    result["Weight Type"] = "Percent"

    result["Weighted Contribution"] = result["Score"] * result["Weight"]
    result["Weighted Contribution Type"] = "Score"

    result["Signal"] = result["Score"].apply(_component_signal)

    return result


def _spending_category_signal(category: str, amount: float | None) -> str:
    """Assign a simple signal to government spending table rows."""

    category_text = str(category).lower()

    if "defense appropriations" in category_text:
        return "Constructive"
    if "defense funding component" in category_text:
        return "Constructive"
    if "nondefense" in category_text:
        return "Limited"
    if "revenue guidance" in category_text:
        return "Positive"
    if "free cash flow" in category_text:
        return "Positive"

    if amount is not None and not pd.isna(amount):
        if amount >= 100:
            return "High Relevance"
        if amount >= 1:
            return "Moderate Relevance"

    return "Reference"


def _spending_category_relevance(category: str) -> str:
    """Create CACI relevance language for government spending table rows."""

    category_text = str(category).lower()

    if "defense appropriations" in category_text:
        return (
            "Supports Department of Defense, intelligence, cyber, modernization, "
            "mission technology, and national-security programs."
        )

    if "defense funding component" in category_text:
        return (
            "Directly relevant to CACI's Department of Defense and national-security "
            "customer exposure."
        )

    if "nondefense" in category_text:
        return (
            "Small funding component; less relevant than defense, intelligence, "
            "and national-security funding."
        )

    if "revenue guidance" in category_text:
        return (
            "Company guidance provides a direct bridge from federal spending demand "
            "to revenue expectations."
        )

    if "free cash flow" in category_text:
        return (
            "Free cash flow guidance supports valuation quality and the expected-return framework."
        )

    return "Relevant as a public spending or company-level reference point."


def _spending_category_source(category: str) -> str:
    """Assign source labels for government spending table rows."""

    category_text = str(category).lower()

    if (
        "caci" in category_text
        or "revenue guidance" in category_text
        or "free cash flow" in category_text
    ):
        return "CACI Q3 FY2026 earnings release"

    return "Senate Appropriations FY2026 Defense Appropriations release"


def build_government_spending_table() -> pd.DataFrame:
    """Build a public government spending table relevant to CACI.

    Dollar values are stored as display-ready text in billions.
    """

    rows = []

    for item in GOVERNMENT_SPENDING_TABLE:
        category = item.get("Spending Category", "")
        amount = item.get("Amount")

        rows.append(
            {
                "Fiscal Year": item.get("Fiscal Year"),
                "Fiscal Year Type": "Plain Number",
                "Spending Category": category,
                "Amount": _format_billions(amount),
                "Amount Value Type": "Text",
                "Amount Unit": "Billions USD",
                "Raw Amount": amount,
                "Status": item.get("Status"),
                "Signal": _spending_category_signal(category, amount),
                "CACI Relevance": _spending_category_relevance(category),
                "Source": _spending_category_source(category),
            }
        )

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "government_spending_table.csv")

    return df


def build_caci_government_exposure_table() -> pd.DataFrame:
    """Build CACI's customer exposure bridge to government spending."""

    exposure_links = {
        "Department of Defense": {
            "Spending Link": (
                "Defense appropriations, military modernization, cyber, space, "
                "command and control, and mission technology"
            ),
            "Valuation Relevance": "Primary federal spending driver for revenue durability",
            "Signal": "High Relevance",
        },
        "Intelligence Community": {
            "Spending Link": (
                "National Intelligence Program, Military Intelligence Program, "
                "classified mission demand, cyber, signals, and national-security technology"
            ),
            "Valuation Relevance": "Supports intelligence-contractor thesis and technology premium",
            "Signal": "High Relevance",
        },
        "Federal Civilian Agencies": {
            "Spending Link": "Federal IT modernization, cyber, cloud, data, and digital services",
            "Valuation Relevance": "Secondary diversification channel outside DoD and IC",
            "Signal": "Moderate Relevance",
        },
        "Commercial and Other": {
            "Spending Link": "Limited direct federal spending link",
            "Valuation Relevance": "Small diversification component",
            "Signal": "Low Relevance",
        },
    }

    rows = []

    for category, exposure in CACI_CUSTOMER_MIX.items():
        link_info = exposure_links.get(
            category,
            {
                "Spending Link": "General government and mission technology exposure",
                "Valuation Relevance": "Supplemental revenue exposure",
                "Signal": "Reference",
            },
        )

        rows.append(
            {
                "Customer / Exposure Category": category,
                "CACI Exposure": f"{exposure:.1%} of Q3 FY2026 revenue",
                "CACI Exposure Value Type": "Text",
                "Raw CACI Exposure": exposure,
                "Spending Link": link_info["Spending Link"],
                "Valuation Relevance": link_info["Valuation Relevance"],
                "Signal": link_info["Signal"],
            }
        )

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "caci_government_exposure.csv")

    return df


def build_government_spending_components() -> pd.DataFrame:
    """Score government-spending conditions relevant to CACI.

    Scores are interpreted as spending support scores:
    - 100 = strong funding tailwind
    - 50 = mixed / neutral
    - 0 = funding headwind
    """

    rows = [
        {
            "Spending Component": "Defense Appropriations Base",
            "Score": 80,
            "Weight": 0.20,
            "Evidence": "FY2026 defense appropriations provide a large defense funding base.",
            "CACI Relevance": (
                "Supports DoD programs, modernization, cyber, intelligence support, "
                "and mission technology demand."
            ),
            "Model Link": "Revenue growth confidence, backlog conversion, business quality score",
        },
        {
            "Spending Component": "Intelligence Funding Relevance",
            "Score": 84,
            "Weight": 0.18,
            "Evidence": (
                "CACI has meaningful Intelligence Community exposure and benefits from "
                "the public NIP/MIP funding ecosystem."
            ),
            "CACI Relevance": (
                "Supports intelligence, cyber, data, signals, space, and national-security "
                "technology work."
            ),
            "Model Link": "Technology premium, revenue durability, defense intelligence score",
        },
        {
            "Spending Component": "Federal Civilian Modernization",
            "Score": 66,
            "Weight": 0.10,
            "Evidence": (
                "Federal civilian agencies continue to require IT modernization, cybersecurity, "
                "cloud, and data services."
            ),
            "CACI Relevance": "Supports non-DoD federal revenue diversification.",
            "Model Link": "Revenue diversification and secondary growth support",
        },
        {
            "Spending Component": "Contract Award Environment",
            "Score": 78,
            "Weight": 0.14,
            "Evidence": (
                "CACI reported continued contract award momentum and strong backlog visibility."
            ),
            "CACI Relevance": (
                "Awards and backlog translate government funding into company-level revenue visibility."
            ),
            "Model Link": "Backlog conversion, expected return confidence, scenario support",
        },
        {
            "Spending Component": "Appropriations Timing Risk",
            "Score": 56,
            "Weight": 0.12,
            "Evidence": (
                "Continuing resolutions, shutdown risk, and procurement delays can affect award timing."
            ),
            "CACI Relevance": (
                "CACI depends on federal budget execution and contract timing."
            ),
            "Model Link": "Working capital timing, revenue recognition, WACC risk",
        },
        {
            "Spending Component": "Defense Technology Funding Priority",
            "Score": 82,
            "Weight": 0.14,
            "Evidence": (
                "Cyber, space, command and control, data, AI, and mission systems remain "
                "priority areas."
            ),
            "CACI Relevance": (
                "Supports CACI's technology revenue mix and defense technology positioning."
            ),
            "Model Link": "Margin quality, peer premium, terminal growth support",
        },
        {
            "Spending Component": "Budget Concentration Risk",
            "Score": 58,
            "Weight": 0.12,
            "Evidence": (
                "Heavy dependence on U.S. government spending creates exposure to policy, "
                "budget, and procurement changes."
            ),
            "CACI Relevance": (
                "CACI has high exposure to DoD, Intelligence Community, and federal civilian customers."
            ),
            "Model Link": "WACC, terminal value sensitivity, valuation haircut risk",
        },
    ]

    df = pd.DataFrame(rows)
    df = _add_component_fields(df)

    df = df[
        [
            "Spending Component",
            "Score",
            "Score Type",
            "Weight",
            "Weight Type",
            "Weighted Contribution",
            "Weighted Contribution Type",
            "Signal",
            "Evidence",
            "CACI Relevance",
            "Model Link",
        ]
    ]

    _safe_to_csv(df, "government_spending_components.csv")

    return df


def build_government_spending_summary(
    components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact spending summary for the PDF report."""

    if components is None:
        components = build_government_spending_components()

    if components is None or components.empty:
        summary = pd.DataFrame(
            [
                {
                    "Metric": "Overall Government Spending Score",
                    "Value": 0.0,
                    "Value Type": "Score",
                    "Signal": "Unavailable",
                    "Interpretation": "No government spending components were available.",
                }
            ]
        )
        _safe_to_csv(summary, "government_spending_summary.csv")
        return summary

    overall_score = round(_weighted_score(components), 1)
    overall_signal = _score_to_signal(overall_score)

    strongest_component = components.sort_values("Score", ascending=False).iloc[0]
    weakest_component = components.sort_values("Score", ascending=True).iloc[0]

    summary = pd.DataFrame(
        [
            {
                "Metric": "Overall Government Spending Score",
                "Value": overall_score,
                "Value Type": "Score",
                "Signal": overall_signal,
                "Interpretation": (
                    "Weighted score across defense appropriations, intelligence funding relevance, "
                    "federal civilian modernization, contract awards, appropriations timing, "
                    "defense technology priorities, and budget concentration risk."
                ),
            },
            {
                "Metric": "Strongest Spending Driver",
                "Value": strongest_component["Spending Component"],
                "Value Type": "Text",
                "Signal": strongest_component["Signal"],
                "Interpretation": strongest_component["CACI Relevance"],
            },
            {
                "Metric": "Weakest Spending Driver",
                "Value": weakest_component["Spending Component"],
                "Value Type": "Text",
                "Signal": weakest_component["Signal"],
                "Interpretation": weakest_component["CACI Relevance"],
            },
            {
                "Metric": "Primary Positive Driver",
                "Value": "Defense and intelligence spending relevance",
                "Value Type": "Text",
                "Signal": "Positive",
                "Interpretation": (
                    "Supports CACI's DoD, Intelligence Community, cyber, mission technology, "
                    "and modernization exposure."
                ),
            },
            {
                "Metric": "Primary Monitoring Risk",
                "Value": "Appropriations timing and budget concentration",
                "Value Type": "Text",
                "Signal": "Monitor",
                "Interpretation": (
                    "Federal budget timing and concentration create risks around award timing, "
                    "working-capital cadence, and valuation sensitivity."
                ),
            },
        ]
    )

    _safe_to_csv(summary, "government_spending_summary.csv")

    return summary


def build_government_spending_adjustments(
    components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Translate the government spending score into optional model adjustments.

    These are not directly applied to the DCF. They are reportable signals that can
    support expected-return confidence, valuation review, or scenario interpretation.
    """

    if components is None:
        components = build_government_spending_components()

    overall_score = round(_weighted_score(components), 1)
    overall_signal = _score_to_signal(overall_score)

    if overall_score >= 85:
        valuation_adjustment = 0.050
        revenue_confidence_adjustment = 0.050
        interpretation = "Very strong spending backdrop supports a valuation premium."
    elif overall_score >= 75:
        valuation_adjustment = 0.025
        revenue_confidence_adjustment = 0.030
        interpretation = "Constructive spending backdrop supports a modest valuation premium."
    elif overall_score >= 60:
        valuation_adjustment = 0.000
        revenue_confidence_adjustment = 0.000
        interpretation = "Supportive but balanced spending backdrop; no valuation adjustment applied."
    elif overall_score >= 50:
        valuation_adjustment = -0.025
        revenue_confidence_adjustment = -0.020
        interpretation = "Mixed spending backdrop warrants a modest valuation haircut."
    else:
        valuation_adjustment = -0.050
        revenue_confidence_adjustment = -0.050
        interpretation = "Spending headwinds warrant a larger valuation haircut."

    adjustments = pd.DataFrame(
        [
            {
                "Adjustment": "Government Spending Valuation Adjustment",
                "Value": valuation_adjustment,
                "Value Type": "Percent",
                "Signal": overall_signal,
                "Interpretation": interpretation,
            },
            {
                "Adjustment": "Revenue Confidence Adjustment",
                "Value": revenue_confidence_adjustment,
                "Value Type": "Percent",
                "Signal": overall_signal,
                "Interpretation": (
                    "Optional input for expected-return confidence or scenario interpretation."
                ),
            },
            {
                "Adjustment": "Overall Government Spending Score",
                "Value": overall_score,
                "Value Type": "Score",
                "Signal": overall_signal,
                "Interpretation": (
                    "Weighted score summarizing the government spending backdrop relevant to CACI."
                ),
            },
        ]
    )

    _safe_to_csv(adjustments, "government_spending_adjustments.csv")

    return adjustments


def run_government_spending_model() -> dict[str, pd.DataFrame]:
    """Run the full government spending intelligence model."""

    spending_table = build_government_spending_table()
    exposure_table = build_caci_government_exposure_table()
    components = build_government_spending_components()
    summary = build_government_spending_summary(components)
    adjustments = build_government_spending_adjustments(components)

    return {
        "government_spending_table": spending_table,
        "caci_government_exposure": exposure_table,
        "government_spending_components": components,
        "government_spending_summary": summary,
        "government_spending_adjustments": adjustments,
    }