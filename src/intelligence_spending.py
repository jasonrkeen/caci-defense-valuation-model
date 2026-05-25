import numpy as np
import pandas as pd

from src.config import (
    TABLES_DIR,
    INTELLIGENCE_BUDGET_TABLE,
    CACI_CUSTOMER_MIX,
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


def _format_billions(value) -> str:
    """Format a numeric value as billions of dollars for report display."""

    value = _safe_float(value)

    if pd.isna(value):
        return "Unavailable"

    return f"${value:,.2f}B"


def _format_year(value) -> str:
    """Format fiscal year values without decimals."""

    value = _safe_float(value)

    if pd.isna(value):
        return "Unavailable"

    return str(int(value))


def _score_to_signal(score: float) -> str:
    """Convert spending score into a signal."""

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


def _add_component_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add score-driven fields to a component table."""

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


def build_intelligence_budget_table() -> pd.DataFrame:
    """Build public U.S. intelligence budget table from centralized config.

    NIP = National Intelligence Program.
    MIP = Military Intelligence Program.

    Public topline amounts are stored as report-ready text in billions,
    with raw numeric values retained for auditability.
    """

    df = pd.DataFrame(INTELLIGENCE_BUDGET_TABLE)

    expected_columns = [
        "Fiscal Year",
        "NIP Appropriated",
        "NIP Requested",
        "MIP Appropriated",
        "MIP Requested",
        "Total Appropriated",
    ]

    for column in expected_columns:
        if column not in df.columns:
            df[column] = np.nan

    numeric_columns = [
        "Fiscal Year",
        "NIP Appropriated",
        "NIP Requested",
        "MIP Appropriated",
        "MIP Requested",
        "Total Appropriated",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values("Fiscal Year").reset_index(drop=True)

    df["NIP Appropriated YoY Growth"] = df["NIP Appropriated"].pct_change(
        fill_method=None
    )
    df["MIP Appropriated YoY Growth"] = df["MIP Appropriated"].pct_change(
        fill_method=None
    )
    df["Total Appropriated YoY Growth"] = df["Total Appropriated"].pct_change(
        fill_method=None
    )
    df["NIP Requested YoY Growth"] = df["NIP Requested"].pct_change(
        fill_method=None
    )
    df["MIP Requested YoY Growth"] = df["MIP Requested"].pct_change(
        fill_method=None
    )

    df["Raw Fiscal Year"] = df["Fiscal Year"]
    df["Fiscal Year"] = df["Fiscal Year"].apply(_format_year)
    df["Fiscal Year Type"] = "Text"

    amount_columns = [
        "NIP Appropriated",
        "NIP Requested",
        "MIP Appropriated",
        "MIP Requested",
        "Total Appropriated",
    ]

    for column in amount_columns:
        df[f"Raw {column}"] = df[column]
        df[column] = df[column].apply(_format_billions)
        df[f"{column} Type"] = "Text"

    growth_columns = [
        "NIP Appropriated YoY Growth",
        "MIP Appropriated YoY Growth",
        "Total Appropriated YoY Growth",
        "NIP Requested YoY Growth",
        "MIP Requested YoY Growth",
    ]

    for column in growth_columns:
        df[f"{column} Type"] = "Percent"

    df["Source"] = "ODNI IC Budget"

    df = df[
        [
            "Fiscal Year",
            "Fiscal Year Type",
            "NIP Appropriated",
            "NIP Appropriated Type",
            "NIP Requested",
            "NIP Requested Type",
            "MIP Appropriated",
            "MIP Appropriated Type",
            "MIP Requested",
            "MIP Requested Type",
            "Total Appropriated",
            "Total Appropriated Type",
            "NIP Appropriated YoY Growth",
            "NIP Appropriated YoY Growth Type",
            "MIP Appropriated YoY Growth",
            "MIP Appropriated YoY Growth Type",
            "Total Appropriated YoY Growth",
            "Total Appropriated YoY Growth Type",
            "NIP Requested YoY Growth",
            "NIP Requested YoY Growth Type",
            "MIP Requested YoY Growth",
            "MIP Requested YoY Growth Type",
            "Source",
            "Raw Fiscal Year",
            "Raw NIP Appropriated",
            "Raw NIP Requested",
            "Raw MIP Appropriated",
            "Raw MIP Requested",
            "Raw Total Appropriated",
        ]
    ]

    _safe_to_csv(df, "intelligence_budget_table.csv")

    return df


def _numeric_budget_table_for_scoring(budget_df: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric version of the budget table for scoring logic."""

    if budget_df is None or budget_df.empty:
        return pd.DataFrame()

    result = budget_df.copy()

    raw_to_clean = {
        "Raw Fiscal Year": "Fiscal Year",
        "Raw NIP Appropriated": "NIP Appropriated",
        "Raw NIP Requested": "NIP Requested",
        "Raw MIP Appropriated": "MIP Appropriated",
        "Raw MIP Requested": "MIP Requested",
        "Raw Total Appropriated": "Total Appropriated",
    }

    for raw_column, clean_column in raw_to_clean.items():
        if raw_column in result.columns:
            result[clean_column] = pd.to_numeric(
                result[raw_column],
                errors="coerce",
            )
        elif clean_column in result.columns:
            result[clean_column] = pd.to_numeric(
                result[clean_column],
                errors="coerce",
            )

    growth_columns = [
        "NIP Appropriated YoY Growth",
        "MIP Appropriated YoY Growth",
        "Total Appropriated YoY Growth",
        "NIP Requested YoY Growth",
        "MIP Requested YoY Growth",
    ]

    for column in growth_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def build_intelligence_spending_signal(
    budget_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a spending signal from public intelligence budget toplines."""

    if budget_df is None:
        budget_df = build_intelligence_budget_table()

    if budget_df is None or budget_df.empty:
        df = pd.DataFrame(
            [
                _summary_row(
                    metric="Intelligence Spending Signal Score",
                    value=0.0,
                    value_type="Score",
                    signal="Unavailable",
                    interpretation="No intelligence budget table was available.",
                )
            ]
        )
        _safe_to_csv(df, "intelligence_spending_signal.csv")
        return df

    scoring_df = _numeric_budget_table_for_scoring(budget_df)

    if scoring_df.empty:
        df = pd.DataFrame(
            [
                _summary_row(
                    metric="Intelligence Spending Signal Score",
                    value=0.0,
                    value_type="Score",
                    signal="Unavailable",
                    interpretation="No valid fiscal year data was available.",
                )
            ]
        )
        _safe_to_csv(df, "intelligence_spending_signal.csv")
        return df

    scoring_df["Fiscal Year"] = pd.to_numeric(
        scoring_df["Fiscal Year"],
        errors="coerce",
    )

    scoring_df = scoring_df.dropna(subset=["Fiscal Year"])

    if scoring_df.empty:
        df = pd.DataFrame(
            [
                _summary_row(
                    metric="Intelligence Spending Signal Score",
                    value=0.0,
                    value_type="Score",
                    signal="Unavailable",
                    interpretation="No valid fiscal year data was available.",
                )
            ]
        )
        _safe_to_csv(df, "intelligence_spending_signal.csv")
        return df

    latest_year = int(scoring_df["Fiscal Year"].max())
    latest_row = scoring_df.loc[scoring_df["Fiscal Year"] == latest_year].iloc[0]

    prior_available = scoring_df.dropna(subset=["Total Appropriated"])

    latest_total_appropriated = latest_row.get("Total Appropriated")
    latest_nip_requested_growth = latest_row.get("NIP Requested YoY Growth")

    if pd.notna(latest_total_appropriated):
        topline_metric = "Total Appropriated"
        topline_value = latest_total_appropriated
        growth_metric = "Total Appropriated YoY Growth"
        growth_value = latest_row.get("Total Appropriated YoY Growth")
    else:
        topline_metric = "NIP Requested"
        topline_value = latest_row.get("NIP Requested")
        growth_metric = "NIP Requested YoY Growth"
        growth_value = latest_nip_requested_growth

    growth_value = _safe_float(growth_value)

    if pd.notna(growth_value) and growth_value >= 0.08:
        score = 84.0
        interpretation = "Latest public intelligence budget request growth is supportive."
    elif pd.notna(growth_value) and growth_value >= 0.03:
        score = 76.0
        interpretation = "Latest public intelligence budget trend is modestly supportive."
    elif pd.notna(growth_value) and growth_value >= 0:
        score = 66.0
        interpretation = "Latest public intelligence budget trend is stable."
    elif pd.notna(growth_value):
        score = 52.0
        interpretation = "Latest public intelligence budget trend is softer and should be monitored."
    else:
        score = 62.0
        interpretation = "Latest public topline is available, but comparable growth is limited."

    signal = _score_to_signal(score)

    if not prior_available.empty:
        latest_appropriated_year = int(prior_available["Fiscal Year"].max())
        latest_appropriated_total = _safe_float(
            prior_available.loc[
                prior_available["Fiscal Year"] == latest_appropriated_year,
                "Total Appropriated",
            ].iloc[0]
        )
    else:
        latest_appropriated_year = np.nan
        latest_appropriated_total = np.nan

    rows = [
        _summary_row(
            metric="Latest Public Intelligence Budget Year",
            value=latest_year,
            value_type="Plain Number",
            signal="Reference",
            interpretation="Latest fiscal year included in the public budget table.",
        ),
        _summary_row(
            metric=f"Latest {topline_metric}",
            value=_format_billions(topline_value),
            value_type="Text",
            signal=signal,
            interpretation=(
                f"Latest available public intelligence budget measure used for scoring: "
                f"{topline_metric}."
            ),
        ),
        _summary_row(
            metric=growth_metric,
            value=growth_value,
            value_type="Percent",
            signal=signal,
            interpretation=interpretation,
        ),
        _summary_row(
            metric="Latest Combined Appropriated Topline Year",
            value=latest_appropriated_year,
            value_type="Plain Number",
            signal="Reference",
            interpretation="Latest year with combined public NIP and MIP appropriated topline.",
        ),
        _summary_row(
            metric="Latest Combined Appropriated Topline",
            value=_format_billions(latest_appropriated_total),
            value_type="Text",
            signal="Reference",
            interpretation="Latest available public combined NIP and MIP appropriated topline in billions.",
        ),
        _summary_row(
            metric="Intelligence Spending Signal Score",
            value=score,
            value_type="Score",
            signal=signal,
            interpretation=(
                "Composite signal based on public intelligence spending trend and relevance "
                "to CACI's Intelligence Community, cyber, signals, and national-security exposure."
            ),
        ),
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "intelligence_spending_signal.csv")

    return df


def build_intelligence_exposure_components() -> pd.DataFrame:
    """Score CACI's exposure to intelligence-related spending themes."""

    rows = [
        {
            "Exposure Component": "Intelligence Community Revenue Exposure",
            "Score": 86,
            "Weight": 0.28,
            "Exposure": f"{CACI_CUSTOMER_MIX.get('Intelligence Community', 0):.1%} of Q3 FY2026 revenue",
            "Spending Link": "National Intelligence Program, classified mission demand, cyber, signals, and data analytics",
            "Valuation Relevance": "Supports intelligence-contractor thesis, mission durability, and technology premium",
        },
        {
            "Exposure Component": "Department of Defense Intelligence Link",
            "Score": 82,
            "Weight": 0.24,
            "Exposure": f"{CACI_CUSTOMER_MIX.get('Department of Defense', 0):.1%} of Q3 FY2026 revenue",
            "Spending Link": "Military Intelligence Program, cyber, electronic warfare, space, C4ISR, and mission technology",
            "Valuation Relevance": "Supports defense technology demand, backlog visibility, and modernization exposure",
        },
        {
            "Exposure Component": "Cyber / National Security Adjacency",
            "Score": 80,
            "Weight": 0.18,
            "Exposure": "Cyber, intelligence, data, and mission technology exposure",
            "Spending Link": "Cybersecurity, secure networks, signals, data fusion, AI-enabled mission systems",
            "Valuation Relevance": "Supports higher-quality growth and defense technology positioning",
        },
        {
            "Exposure Component": "Federal Civilian Security Modernization",
            "Score": 66,
            "Weight": 0.12,
            "Exposure": f"{CACI_CUSTOMER_MIX.get('Federal Civilian Agencies', 0):.1%} of Q3 FY2026 revenue",
            "Spending Link": "Civilian cybersecurity, IT modernization, data, cloud, and national-security adjacent missions",
            "Valuation Relevance": "Provides secondary federal modernization exposure outside DoD and IC",
        },
        {
            "Exposure Component": "Budget Timing / Classified Procurement Risk",
            "Score": 58,
            "Weight": 0.10,
            "Exposure": "Federal budget and classified contract timing exposure",
            "Spending Link": "Appropriations timing, continuing resolutions, contract starts, and procurement cadence",
            "Valuation Relevance": "Creates monitoring risk for backlog conversion and working-capital timing",
        },
        {
            "Exposure Component": "Commercial / Other Offset",
            "Score": 52,
            "Weight": 0.08,
            "Exposure": f"{CACI_CUSTOMER_MIX.get('Commercial and Other', 0):.1%} of Q3 FY2026 revenue",
            "Spending Link": "Limited direct intelligence budget linkage",
            "Valuation Relevance": "Small diversification component with limited direct intelligence-spending sensitivity",
        },
    ]

    df = pd.DataFrame(rows)
    df = _add_component_fields(df)

    df = df[
        [
            "Exposure Component",
            "Score",
            "Score Type",
            "Weight",
            "Weight Type",
            "Weighted Contribution",
            "Weighted Contribution Type",
            "Signal",
            "Exposure",
            "Spending Link",
            "Valuation Relevance",
        ]
    ]

    _safe_to_csv(df, "intelligence_exposure_components.csv")

    return df


def build_caci_intelligence_exposure_table() -> pd.DataFrame:
    """Build a CACI-specific bridge to intelligence spending."""

    rows = [
        {
            "Exposure": "Intelligence Community",
            "CACI Exposure": f"{CACI_CUSTOMER_MIX.get('Intelligence Community', 0):.1%} of Q3 FY2026 revenue",
            "CACI Exposure Type": "Text",
            "Raw CACI Exposure": CACI_CUSTOMER_MIX.get("Intelligence Community", 0),
            "Spending Link": "National Intelligence Program, Military Intelligence Program, classified mission demand",
            "Valuation Relevance": "Supports intelligence-contractor thesis, revenue durability, and technology premium",
            "Signal": "High Relevance",
        },
        {
            "Exposure": "Department of Defense",
            "CACI Exposure": f"{CACI_CUSTOMER_MIX.get('Department of Defense', 0):.1%} of Q3 FY2026 revenue",
            "CACI Exposure Type": "Text",
            "Raw CACI Exposure": CACI_CUSTOMER_MIX.get("Department of Defense", 0),
            "Spending Link": "Military Intelligence Program, defense technology, cyber, space, electronic warfare",
            "Valuation Relevance": "Supports defense technology demand and backlog visibility",
            "Signal": "High Relevance",
        },
        {
            "Exposure": "Federal Civilian Agencies",
            "CACI Exposure": f"{CACI_CUSTOMER_MIX.get('Federal Civilian Agencies', 0):.1%} of Q3 FY2026 revenue",
            "CACI Exposure Type": "Text",
            "Raw CACI Exposure": CACI_CUSTOMER_MIX.get("Federal Civilian Agencies", 0),
            "Spending Link": "Cybersecurity, IT modernization, national-security adjacent civilian missions",
            "Valuation Relevance": "Provides secondary federal modernization exposure",
            "Signal": "Moderate Relevance",
        },
        {
            "Exposure": "Commercial and Other",
            "CACI Exposure": f"{CACI_CUSTOMER_MIX.get('Commercial and Other', 0):.1%} of Q3 FY2026 revenue",
            "CACI Exposure Type": "Text",
            "Raw CACI Exposure": CACI_CUSTOMER_MIX.get("Commercial and Other", 0),
            "Spending Link": "Limited direct intelligence budget link",
            "Valuation Relevance": "Small diversification component",
            "Signal": "Low Relevance",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "caci_intelligence_exposure.csv")

    return df


def build_intelligence_exposure_summary(
    exposure_components: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact intelligence exposure summary table."""

    if exposure_components is None:
        exposure_components = build_intelligence_exposure_components()

    if exposure_components is None or exposure_components.empty:
        df = pd.DataFrame(
            [
                _summary_row(
                    metric="Intelligence Exposure Score",
                    value=0.0,
                    value_type="Score",
                    signal="Unavailable",
                    interpretation="No intelligence exposure components were available.",
                )
            ]
        )
        _safe_to_csv(df, "intelligence_exposure_summary.csv")
        return df

    score = round(_weighted_score(exposure_components), 1)
    signal = _score_to_signal(score)

    strongest = exposure_components.sort_values("Score", ascending=False).iloc[0]
    weakest = exposure_components.sort_values("Score", ascending=True).iloc[0]

    rows = [
        _summary_row(
            metric="Intelligence Exposure Score",
            value=score,
            value_type="Score",
            signal=signal,
            interpretation=(
                "Weighted score summarizing CACI's exposure to public intelligence, "
                "classified mission, cyber, DoD intelligence, and national-security technology demand."
            ),
        ),
        _summary_row(
            metric="Strongest Intelligence Exposure Driver",
            value=strongest["Exposure Component"],
            value_type="Text",
            signal=strongest["Signal"],
            interpretation=strongest["Valuation Relevance"],
        ),
        _summary_row(
            metric="Weakest Intelligence Exposure Driver",
            value=weakest["Exposure Component"],
            value_type="Text",
            signal=weakest["Signal"],
            interpretation=weakest["Valuation Relevance"],
        ),
        _summary_row(
            metric="Primary Intelligence Tailwind",
            value="Intelligence Community and DoD intelligence-linked demand",
            value_type="Text",
            signal="Constructive",
            interpretation=(
                "CACI's revenue mix is directly aligned with intelligence, cyber, classified mission, "
                "and national-security technology demand."
            ),
        ),
        _summary_row(
            metric="Primary Intelligence Spending Risk",
            value="Budget timing and classified procurement cadence",
            value_type="Text",
            signal="Monitor",
            interpretation=(
                "Appropriations timing, continuing resolutions, and classified procurement timing "
                "can affect award conversion and revenue cadence."
            ),
        ),
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "intelligence_exposure_summary.csv")

    return df


def run_intelligence_spending_model() -> dict[str, pd.DataFrame]:
    """Run the full intelligence spending model."""

    budget_table = build_intelligence_budget_table()
    spending_signal = build_intelligence_spending_signal(budget_table)
    exposure_table = build_caci_intelligence_exposure_table()
    exposure_components = build_intelligence_exposure_components()
    exposure_summary = build_intelligence_exposure_summary(exposure_components)

    return {
        "intelligence_budget_table": budget_table,
        "intelligence_spending_signal": spending_signal,
        "caci_intelligence_exposure": exposure_table,
        "intelligence_exposure_components": exposure_components,
        "intelligence_exposure_summary": exposure_summary,
    }