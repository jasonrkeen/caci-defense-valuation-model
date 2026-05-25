import pandas as pd

from src.config import TABLES_DIR, SOURCE_URLS


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _source_url(source_name: str) -> str:
    """Return a source URL if available."""

    return SOURCE_URLS.get(source_name, "")


def _has_url(source_name: str) -> bool:
    """Check whether a source has a configured URL."""

    url = _source_url(source_name)
    return bool(str(url).strip())


def _priority_score(priority: str) -> int:
    """Convert source priority into a numeric score."""

    mapping = {
        "Very High": 95,
        "High": 85,
        "Medium / High": 78,
        "Medium": 65,
        "Low": 45,
    }

    return mapping.get(str(priority).strip(), 60)


def _valuation_relevance_score(relevance: str) -> int:
    """Convert valuation relevance into a numeric score."""

    mapping = {
        "Very High": 95,
        "High": 85,
        "Medium / High": 78,
        "Medium": 65,
        "Low": 45,
    }

    return mapping.get(str(relevance).strip(), 60)


def _quality_signal(score: float) -> str:
    """Convert a source or input score into a report-ready signal."""

    if pd.isna(score):
        return "Unavailable"
    if score >= 90:
        return "Very High Confidence"
    if score >= 80:
        return "High Confidence"
    if score >= 70:
        return "Medium / High Confidence"
    if score >= 60:
        return "Medium Confidence"
    return "Low Confidence"


def _valuation_impact_signal(value: str) -> str:
    """Normalize valuation impact into a signal."""

    value = str(value).strip()

    if value == "Very High":
        return "Very High Impact"
    if value == "High":
        return "High Impact"
    if value == "Medium / High":
        return "Medium / High Impact"
    if value == "Medium":
        return "Medium Impact"
    if value == "Low":
        return "Low Impact"

    return "Reference"


def _freshness_signal(update_cadence: str) -> str:
    """Create a freshness signal from update cadence."""

    cadence = str(update_cadence).lower()

    if "quarterly" in cadence:
        return "Refresh Each Earnings Cycle"
    if "annual" in cadence:
        return "Refresh Annually"
    if "runtime" in cadence or "live" in cadence:
        return "Runtime Refresh"
    if "periodic" in cadence:
        return "Periodic Review"
    if "as needed" in cadence:
        return "Review As Needed"

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


def build_source_catalog() -> pd.DataFrame:
    """Build a source catalog documenting the main project sources."""

    rows = [
        {
            "Source": "CACI Investor Presentation January 2026",
            "Type": "Investor presentation",
            "Priority": "High",
            "Valuation Relevance": "High",
            "Update Cadence": "Periodic investor presentation refresh",
            "Source Freshness": "Current planning framework",
            "Model Use": "Strategy, FY26 guidance, 3-year targets, backlog, book-to-bill, capital allocation",
            "Limitation": "Management presentation; should be cross-checked against SEC filings and earnings releases",
            "URL": _source_url("CACI Investor Presentation January 2026"),
        },
        {
            "Source": "CACI Q3 FY2026 Earnings Slides",
            "Type": "Earnings presentation",
            "Priority": "High",
            "Valuation Relevance": "High",
            "Update Cadence": "Quarterly",
            "Source Freshness": "Latest quarterly operating update",
            "Model Use": "Q3 growth, EBITDA margin, contract awards, ARKA acquisition and technology portfolio context",
            "Limitation": "Quarterly snapshot; some metrics may not represent full-year trend",
            "URL": _source_url("CACI Q3 FY2026 Earnings Slides"),
        },
        {
            "Source": "CACI Q3 FY2026 Earnings Release",
            "Type": "Earnings release",
            "Priority": "High",
            "Valuation Relevance": "High",
            "Update Cadence": "Quarterly",
            "Source Freshness": "Latest quarterly guidance update",
            "Model Use": "Updated FY2026 guidance, Q3 income statement metrics, contract awards, revenue mix, and cash flow guidance",
            "Limitation": "Company-reported non-GAAP and guidance metrics require interpretation",
            "URL": _source_url("CACI Q3 FY2026 Earnings Release"),
        },
        {
            "Source": "CACI FY2025 Form 10-K",
            "Type": "SEC filing",
            "Priority": "Very High",
            "Valuation Relevance": "Very High",
            "Update Cadence": "Annual",
            "Source Freshness": "Audited annual baseline",
            "Model Use": "Backlog definitions, customer revenue mix, risk factors, capital allocation, audited historical financials",
            "Limitation": "Historical annual filing; does not include later Q3 FY2026 updates",
            "URL": _source_url("CACI FY2025 Form 10-K"),
        },
        {
            "Source": "Yahoo Finance Analyst Analysis",
            "Type": "Market data / analyst estimates",
            "Priority": "Medium",
            "Valuation Relevance": "Medium",
            "Update Cadence": "Runtime / live data feed",
            "Source Freshness": "Runtime dependent",
            "Model Use": "Runtime analyst estimate tables collected through yfinance when available",
            "Limitation": "Unofficial data feed; can be unavailable, rate-limited, delayed, or incomplete",
            "URL": _source_url("Yahoo Finance Analyst Analysis"),
        },
        {
            "Source": "ODNI IC Budget",
            "Type": "Government source",
            "Priority": "High",
            "Valuation Relevance": "High",
            "Update Cadence": "Annual / as released",
            "Source Freshness": "Public intelligence budget topline source",
            "Model Use": "Public National Intelligence Program and Military Intelligence Program budget toplines",
            "Limitation": "Public toplines only; does not provide program-level classified spending detail",
            "URL": _source_url("ODNI IC Budget"),
        },
        {
            "Source": "Senate FY2026 Defense Appropriations",
            "Type": "Government source",
            "Priority": "High",
            "Valuation Relevance": "High",
            "Update Cadence": "Annual / legislative cycle",
            "Source Freshness": "FY2026 appropriations reference",
            "Model Use": "Defense appropriations backdrop for CACI government spending and defense macro models",
            "Limitation": "Appropriations headline data does not map directly to CACI contract-level revenue",
            "URL": _source_url("Senate FY2026 Defense Appropriations"),
        },
        {
            "Source": "SIPRI Military Expenditure Database",
            "Type": "Defense macro data",
            "Priority": "Medium",
            "Valuation Relevance": "Medium",
            "Update Cadence": "Annual",
            "Source Freshness": "Macro context source",
            "Model Use": "Global military spending context for defense macro demand analysis",
            "Limitation": "Global spending context is indirect; CACI is primarily U.S. government-exposed",
            "URL": _source_url("SIPRI Military Expenditure Database"),
        },
    ]

    df = pd.DataFrame(rows)

    df["Priority Score"] = df["Priority"].apply(_priority_score)
    df["Priority Score Type"] = "Score"

    df["Valuation Relevance Score"] = df["Valuation Relevance"].apply(
        _valuation_relevance_score
    )
    df["Valuation Relevance Score Type"] = "Score"

    df["URL Available"] = df["URL"].astype(str).str.strip() != ""
    df["URL Available Type"] = "Text"

    df["Source Confidence Score"] = (
        df["Priority Score"] * 0.50
        + df["Valuation Relevance Score"] * 0.40
        + df["URL Available"].astype(int) * 10
    )
    df["Source Confidence Score Type"] = "Score"

    df["Signal"] = df["Source Confidence Score"].apply(_quality_signal)
    df["Freshness Signal"] = df["Update Cadence"].apply(_freshness_signal)
    df["Interpretation"] = df.apply(
        lambda row: (
            f"{row['Source']} is a {row['Signal'].lower()} source for this model; "
            f"main limitation: {row['Limitation']}"
        ),
        axis=1,
    )

    _safe_to_csv(df, "source_catalog.csv")
    return df


def build_source_quality_table() -> pd.DataFrame:
    """Build a source quality and reliability scoring table."""

    rows = [
        {
            "Source": "CACI FY2025 Form 10-K",
            "Reliability Score": 95,
            "Source Quality": "Very High",
            "Reason": "SEC-filed annual report with audited historical financial disclosures and risk factors.",
            "Best Used For": "Historical financials, backlog definitions, customer mix, risk factors",
        },
        {
            "Source": "CACI Q3 FY2026 Earnings Release",
            "Reliability Score": 88,
            "Source Quality": "High",
            "Reason": "Company-reported quarterly results and guidance update.",
            "Best Used For": "FY2026 guidance, quarterly revenue, cash flow, awards, customer mix",
        },
        {
            "Source": "CACI Q3 FY2026 Earnings Slides",
            "Reliability Score": 84,
            "Source Quality": "High",
            "Reason": "Company presentation that explains quarterly results and strategic positioning.",
            "Best Used For": "Q3 operating context, technology mix, contract award discussion",
        },
        {
            "Source": "CACI Investor Presentation January 2026",
            "Reliability Score": 82,
            "Source Quality": "High",
            "Reason": "Management presentation useful for guidance, strategy, and medium-term framework.",
            "Best Used For": "Three-year targets, backlog, book-to-bill, capital allocation",
        },
        {
            "Source": "ODNI IC Budget",
            "Reliability Score": 86,
            "Source Quality": "High",
            "Reason": "Official public intelligence budget topline source.",
            "Best Used For": "NIP and MIP public budget context",
        },
        {
            "Source": "Senate FY2026 Defense Appropriations",
            "Reliability Score": 84,
            "Source Quality": "High",
            "Reason": "Official congressional appropriations communication.",
            "Best Used For": "Defense spending backdrop and government funding context",
        },
        {
            "Source": "SIPRI Military Expenditure Database",
            "Reliability Score": 78,
            "Source Quality": "Medium / High",
            "Reason": "Widely used defense spending database, but less directly tied to CACI revenue.",
            "Best Used For": "Global military spending context",
        },
        {
            "Source": "Yahoo Finance Analyst Analysis",
            "Reliability Score": 60,
            "Source Quality": "Medium",
            "Reason": "Useful market data feed, but estimates can be delayed, incomplete, or unavailable through yfinance.",
            "Best Used For": "Runtime analyst estimate tables and market reference data",
        },
    ]

    df = pd.DataFrame(rows)
    df["Reliability Score Type"] = "Score"
    df["URL Available"] = df["Source"].apply(_has_url)
    df["URL Available Type"] = "Text"
    df["Signal"] = df["Reliability Score"].apply(_quality_signal)
    df["Interpretation"] = df.apply(
        lambda row: (
            f"{row['Source']} carries a {row['Source Quality'].lower()} reliability rating "
            f"and is best used for: {row['Best Used For']}."
        ),
        axis=1,
    )

    _safe_to_csv(df, "source_quality.csv")
    return df


def build_management_guidance_table() -> pd.DataFrame:
    """Build source-driven management guidance table."""

    rows = [
        {
            "Metric": "FY2026 revenue guidance",
            "Value": "$9.5B - $9.6B",
            "Value Type": "Text",
            "Model Input": "Base revenue set to $9.55B midpoint",
            "Valuation Impact": "High",
            "Source": "Q3 FY2026 earnings release",
        },
        {
            "Metric": "FY2026 revenue growth",
            "Value": "Raised guidance after Q3; Q3 revenue growth was 8.5%",
            "Value Type": "Text",
            "Model Input": "Scenario revenue-growth range anchored around high-single-digit growth",
            "Valuation Impact": "High",
            "Source": "Q3 FY2026 earnings release and earnings slides",
        },
        {
            "Metric": "Q3 FY2026 organic growth",
            "Value": "6.8%",
            "Value Type": "Text",
            "Model Input": "Supports base/bear organic-growth assumptions",
            "Valuation Impact": "Medium",
            "Source": "Q3 FY2026 earnings release",
        },
        {
            "Metric": "FY2026 free cash flow guidance",
            "Value": "At least $725M",
            "Value Type": "Text",
            "Model Input": "FCF cross-check for DCF output",
            "Valuation Impact": "Very High",
            "Source": "Q3 FY2026 earnings release",
        },
        {
            "Metric": "Diluted weighted average shares",
            "Value": "22.2M",
            "Value Type": "Text",
            "Model Input": "Per-share valuation denominator",
            "Valuation Impact": "High",
            "Source": "Q3 FY2026 earnings release",
        },
        {
            "Metric": "Investor presentation FY26 capex",
            "Value": "~$85M",
            "Value Type": "Text",
            "Model Input": "Capex approximated at ~1% of revenue",
            "Valuation Impact": "Medium",
            "Source": "January 2026 investor presentation",
        },
        {
            "Metric": "Investor presentation tax rate",
            "Value": "23.0% - 24.0%",
            "Value Type": "Text",
            "Model Input": "Tax rate set at 24%",
            "Valuation Impact": "Medium",
            "Source": "January 2026 investor presentation",
        },
        {
            "Metric": "3-year revenue target",
            "Value": "High-single-digit CAGR",
            "Value Type": "Text",
            "Model Input": "Mid-term DCF growth glide path",
            "Valuation Impact": "High",
            "Source": "January 2026 investor presentation",
        },
        {
            "Metric": "3-year cumulative FCF target",
            "Value": "At least $1.6B",
            "Value Type": "Text",
            "Model Input": "Longer-term FCF quality check",
            "Valuation Impact": "High",
            "Source": "January 2026 investor presentation",
        },
        {
            "Metric": "3-year FCF CAGR target",
            "Value": "Greater than 15%",
            "Value Type": "Text",
            "Model Input": "Bull-case free-cash-flow expansion support",
            "Valuation Impact": "High",
            "Source": "January 2026 investor presentation",
        },
    ]

    df = pd.DataFrame(rows)
    df["Signal"] = df["Valuation Impact"].apply(_valuation_impact_signal)
    df["Interpretation"] = df.apply(
        lambda row: (
            f"{row['Metric']} is a {row['Signal'].lower()} input used for: "
            f"{row['Model Input']}."
        ),
        axis=1,
    )

    _safe_to_csv(df, "management_guidance.csv")
    return df


def build_backlog_and_pipeline_table() -> pd.DataFrame:
    """Build backlog and pipeline source table."""

    rows = [
        {
            "Metric": "TTM contract awards",
            "Value": "$11.6B",
            "Value Type": "Text",
            "Interpretation": "Strong demand and contract conversion backdrop",
            "Valuation Link": "Supports revenue visibility and medium-term growth assumptions",
            "Source": "January 2026 investor presentation",
            "Signal": "Positive",
        },
        {
            "Metric": "TTM book-to-bill",
            "Value": "1.3x",
            "Value Type": "Text",
            "Interpretation": "Awards above revenue support forward visibility",
            "Valuation Link": "Supports premium business-quality score",
            "Source": "January 2026 investor presentation",
            "Signal": "Positive",
        },
        {
            "Metric": "Weighted average award duration",
            "Value": ">6 years",
            "Value Type": "Text",
            "Interpretation": "Supports long-duration revenue visibility",
            "Valuation Link": "Supports lower revenue volatility assumption",
            "Source": "January 2026 investor presentation",
            "Signal": "Positive",
        },
        {
            "Metric": "Backlog",
            "Value": "$33B",
            "Value Type": "Text",
            "Interpretation": "Approximately 3.7 years of annualized revenue",
            "Valuation Link": "Supports base-case and bull-case revenue durability",
            "Source": "January 2026 investor presentation",
            "Signal": "Positive",
        },
        {
            "Metric": "Submitted bids",
            "Value": "$6B, >70% for new business",
            "Value Type": "Text",
            "Interpretation": "Potential new-business pipeline",
            "Valuation Link": "Supports future growth optionality",
            "Source": "January 2026 investor presentation",
            "Signal": "Constructive",
        },
        {
            "Metric": "Expected bids next two quarters",
            "Value": "$20B, >70% for new business",
            "Value Type": "Text",
            "Interpretation": "Large near-term bid pipeline",
            "Valuation Link": "Supports contract-award momentum score",
            "Source": "January 2026 investor presentation",
            "Signal": "Constructive",
        },
        {
            "Metric": "FY2025 10-K backlog",
            "Value": "$31.4B total backlog; $4.2B funded backlog",
            "Value Type": "Text",
            "Interpretation": "SEC-filed baseline for backlog bridge",
            "Valuation Link": "Provides audited reference point for backlog analysis",
            "Source": "FY2025 Form 10-K",
            "Signal": "High Confidence Reference",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "backlog_pipeline_summary.csv")
    return df


def build_revenue_mix_table() -> pd.DataFrame:
    """Build revenue mix source table."""

    rows = [
        {
            "Category": "Q3 FY2026 Customer Mix",
            "Metric": "Department of Defense",
            "Value": "53.0% of Q3 revenue",
            "Value Type": "Text",
            "Interpretation": "Major defense exposure",
            "Source": "Q3 FY2026 earnings release",
            "Signal": "High Government Exposure",
        },
        {
            "Category": "Q3 FY2026 Customer Mix",
            "Metric": "Intelligence Community",
            "Value": "25.0% of Q3 revenue",
            "Value Type": "Text",
            "Interpretation": "High national-security exposure",
            "Source": "Q3 FY2026 earnings release",
            "Signal": "High Intelligence Exposure",
        },
        {
            "Category": "Q3 FY2026 Customer Mix",
            "Metric": "Federal civilian agencies",
            "Value": "17.8% of Q3 revenue",
            "Value Type": "Text",
            "Interpretation": "Federal modernization exposure",
            "Source": "Q3 FY2026 earnings release",
            "Signal": "Federal Modernization Exposure",
        },
        {
            "Category": "Q3 FY2026 Customer Mix",
            "Metric": "Commercial and other",
            "Value": "4.2% of Q3 revenue",
            "Value Type": "Text",
            "Interpretation": "Limited non-government exposure",
            "Source": "Q3 FY2026 earnings release",
            "Signal": "Low Commercial Diversification",
        },
        {
            "Category": "Q3 FY2026 Expertise / Technology",
            "Metric": "Technology",
            "Value": "56.5% of Q3 revenue; +11.2% YoY",
            "Value Type": "Text",
            "Interpretation": "Technology segment is majority of revenue and growing faster",
            "Source": "Q3 FY2026 earnings release",
            "Signal": "Positive Technology Mix",
        },
        {
            "Category": "Q3 FY2026 Expertise / Technology",
            "Metric": "Expertise",
            "Value": "43.5% of Q3 revenue; +5.2% YoY",
            "Value Type": "Text",
            "Interpretation": "Expertise remains important but grows slower than technology",
            "Source": "Q3 FY2026 earnings release",
            "Signal": "Stable Services Base",
        },
        {
            "Category": "FY2025 10-K Customer Mix",
            "Metric": "DoD",
            "Value": "75.4% of FY2025 revenue",
            "Value Type": "Text",
            "Interpretation": "Audited annual baseline shows heavy defense customer concentration",
            "Source": "FY2025 Form 10-K",
            "Signal": "High Customer Concentration",
        },
        {
            "Category": "FY2025 10-K Customer Mix",
            "Metric": "Federal civilian agencies",
            "Value": "20.3% of FY2025 revenue",
            "Value Type": "Text",
            "Interpretation": "Secondary federal customer channel",
            "Source": "FY2025 Form 10-K",
            "Signal": "Federal Civilian Exposure",
        },
        {
            "Category": "FY2025 10-K Customer Mix",
            "Metric": "Commercial and other",
            "Value": "4.3% of FY2025 revenue",
            "Value Type": "Text",
            "Interpretation": "Small commercial and other exposure",
            "Source": "FY2025 Form 10-K",
            "Signal": "Low Commercial Diversification",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "revenue_mix_summary.csv")
    return df


def build_capital_allocation_table() -> pd.DataFrame:
    """Build capital allocation source table."""

    rows = [
        {
            "Theme": "Free Cash Flow",
            "Signal": "Positive",
            "Evidence": "FY2026 free cash flow guidance of at least $725M and 3-year cumulative FCF target of at least $1.6B",
            "Valuation Relevance": "Supports DCF quality and shareholder-return capacity",
            "Interpretation": "Free cash flow guidance provides a high-value cross-check against DCF output.",
        },
        {
            "Theme": "Acquisitions",
            "Signal": "Important",
            "Evidence": "Recent acquisition activity, including ARKA, supports technology portfolio expansion",
            "Valuation Relevance": "Can accelerate growth but increases integration and capital allocation risk",
            "Interpretation": "Acquisitions support technology positioning but require integration monitoring.",
        },
        {
            "Theme": "Share Count",
            "Signal": "Positive",
            "Evidence": "Diluted weighted average shares guidance of 22.2M",
            "Valuation Relevance": "Lower share count supports per-share valuation if FCF growth is sustained",
            "Interpretation": "Share count is a direct driver of fair value per share.",
        },
        {
            "Theme": "Capex Discipline",
            "Signal": "Positive",
            "Evidence": "FY2026 capex framework around ~$85M",
            "Valuation Relevance": "Low capital intensity supports FCF conversion",
            "Interpretation": "Capex discipline supports free cash flow conversion and valuation quality.",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "capital_allocation_summary.csv")
    return df


def build_risk_factor_table() -> pd.DataFrame:
    """Build source-driven risk factor table."""

    rows = [
        {
            "Risk": "Federal budget and appropriations timing",
            "Potential Impact": "Delayed awards, revenue timing pressure, or reduced program funding",
            "Model Sensitivity": "Revenue growth, backlog conversion, working capital",
            "Signal": "Monitor",
            "Interpretation": "Budget timing can affect near-term award flow and working-capital cadence.",
        },
        {
            "Risk": "Contract recompete pressure",
            "Potential Impact": "Lost recompetes or lower-margin renewals could pressure growth and margins",
            "Model Sensitivity": "Revenue growth, EBITDA margin",
            "Signal": "Monitor",
            "Interpretation": "Recompete losses could reduce revenue visibility and margin durability.",
        },
        {
            "Risk": "Customer concentration",
            "Potential Impact": "Heavy exposure to U.S. government customers increases policy and budget risk",
            "Model Sensitivity": "WACC, terminal growth, revenue durability",
            "Signal": "Elevated Structural Risk",
            "Interpretation": "Government concentration supports visibility but raises policy and funding sensitivity.",
        },
        {
            "Risk": "Acquisition integration",
            "Potential Impact": "Integration issues could reduce expected synergies or increase costs",
            "Model Sensitivity": "EBITDA margin, FCF conversion",
            "Signal": "Monitor",
            "Interpretation": "Integration execution matters for margin realization and portfolio expansion.",
        },
        {
            "Risk": "Valuation multiple compression",
            "Potential Impact": "Higher rates or weaker defense technology sentiment could reduce market multiples",
            "Model Sensitivity": "WACC, terminal value, peer comparison",
            "Signal": "Market Risk",
            "Interpretation": "Multiple compression would pressure equity value even if fundamentals remain intact.",
        },
        {
            "Risk": "Labor and cleared talent constraints",
            "Potential Impact": "Difficulty hiring specialized cleared employees could limit program execution",
            "Model Sensitivity": "Revenue growth, margin execution",
            "Signal": "Execution Risk",
            "Interpretation": "Cleared talent availability is a key execution variable for defense technology programs.",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "risk_factor_summary.csv")
    return df


def build_model_input_audit_table() -> pd.DataFrame:
    """Document the source behind key model inputs."""

    rows = [
        {
            "Model Input": "Base FY2026 revenue",
            "Current Value": "$9.55B",
            "Current Value Type": "Text",
            "Primary Source": "CACI Q3 FY2026 Earnings Release",
            "Used In": "DCF, forecast model, expected return model",
            "Audit Note": "Uses midpoint of FY2026 revenue guidance range.",
            "Input Confidence Score": 90,
        },
        {
            "Model Input": "Base EBITDA margin",
            "Current Value": "12.3%",
            "Current Value Type": "Text",
            "Primary Source": "CACI Q3 FY2026 Earnings Release / model assumption",
            "Used In": "DCF valuation",
            "Audit Note": "Margin assumption should be reviewed after each earnings release.",
            "Input Confidence Score": 78,
        },
        {
            "Model Input": "Base free cash flow",
            "Current Value": "$725M",
            "Current Value Type": "Text",
            "Primary Source": "CACI Q3 FY2026 Earnings Release",
            "Used In": "DCF cross-check, expected return model",
            "Audit Note": "Uses minimum FY2026 FCF guidance.",
            "Input Confidence Score": 88,
        },
        {
            "Model Input": "Diluted shares",
            "Current Value": "22.2M",
            "Current Value Type": "Text",
            "Primary Source": "CACI Q3 FY2026 Earnings Release",
            "Used In": "DCF per-share conversion",
            "Audit Note": "Should be updated if buybacks, issuance, or guidance changes materially.",
            "Input Confidence Score": 88,
        },
        {
            "Model Input": "Backlog",
            "Current Value": "$33B investor presentation / $31.4B FY2025 10-K",
            "Current Value Type": "Text",
            "Primary Source": "CACI Investor Presentation and FY2025 Form 10-K",
            "Used In": "Defense risk engine, business quality score, report narrative",
            "Audit Note": "Investor presentation reflects later context; 10-K provides SEC-filed historical baseline.",
            "Input Confidence Score": 86,
        },
        {
            "Model Input": "Customer mix",
            "Current Value": "DoD 53.0%, IC 25.0%, federal civilian 17.8%, commercial/other 4.2%",
            "Current Value Type": "Text",
            "Primary Source": "CACI Q3 FY2026 Earnings Release",
            "Used In": "Government spending model, defense macro model, risk engine",
            "Audit Note": "Quarterly mix may differ from full-year SEC-reported mix.",
            "Input Confidence Score": 82,
        },
        {
            "Model Input": "Public intelligence budget",
            "Current Value": "NIP / MIP public toplines",
            "Current Value Type": "Text",
            "Primary Source": "ODNI IC Budget",
            "Used In": "Intelligence spending model",
            "Audit Note": "Public toplines do not reveal classified program-level allocations.",
            "Input Confidence Score": 80,
        },
        {
            "Model Input": "Defense appropriations",
            "Current Value": "FY2026 defense appropriations reference",
            "Current Value Type": "Text",
            "Primary Source": "Senate FY2026 Defense Appropriations",
            "Used In": "Government spending model, defense macro model",
            "Audit Note": "Appropriations provide broad funding context but do not directly map to CACI revenue.",
            "Input Confidence Score": 78,
        },
    ]

    df = pd.DataFrame(rows)
    df["Input Confidence Score Type"] = "Score"
    df["Signal"] = df["Input Confidence Score"].apply(_quality_signal)
    df["Interpretation"] = df.apply(
        lambda row: (
            f"{row['Model Input']} has {row['Signal'].lower()} based on "
            f"{row['Primary Source']}."
        ),
        axis=1,
    )

    _safe_to_csv(df, "model_input_audit.csv")
    return df


def build_source_coverage_summary(
    source_catalog: pd.DataFrame | None = None,
    source_quality: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact source coverage summary for the report."""

    if source_catalog is None:
        source_catalog = build_source_catalog()

    if source_quality is None:
        source_quality = build_source_quality_table()

    total_sources = len(source_catalog)
    high_priority_sources = int(
        source_catalog["Priority"].isin(["Very High", "High"]).sum()
    )
    high_relevance_sources = int(
        source_catalog["Valuation Relevance"].isin(["Very High", "High"]).sum()
    )
    urls_available = int(source_catalog["URL Available"].sum())
    url_coverage = urls_available / total_sources if total_sources else 0

    average_confidence = (
        source_catalog["Source Confidence Score"].mean()
        if "Source Confidence Score" in source_catalog.columns
        else 0
    )

    average_reliability = (
        source_quality["Reliability Score"].mean()
        if source_quality is not None
        and not source_quality.empty
        and "Reliability Score" in source_quality.columns
        else 0
    )

    rows = [
        _summary_row(
            metric="Total Sources",
            value=total_sources,
            value_type="Plain Number",
            signal="Reference",
            interpretation="Number of source categories used in the valuation and intelligence pipeline.",
        ),
        _summary_row(
            metric="High Priority Sources",
            value=high_priority_sources,
            value_type="Plain Number",
            signal="Source Depth",
            interpretation="Number of sources marked as high or very high priority.",
        ),
        _summary_row(
            metric="High Valuation-Relevance Sources",
            value=high_relevance_sources,
            value_type="Plain Number",
            signal="Valuation Coverage",
            interpretation="Number of sources directly relevant to valuation assumptions.",
        ),
        _summary_row(
            metric="URL Coverage",
            value=url_coverage,
            value_type="Percent",
            signal="Complete" if url_coverage >= 0.90 else "Needs Review",
            interpretation="Share of source catalog entries with configured URLs.",
        ),
        _summary_row(
            metric="Average Source Confidence Score",
            value=average_confidence,
            value_type="Score",
            signal=_quality_signal(average_confidence),
            interpretation="Average confidence score across cataloged sources.",
        ),
        _summary_row(
            metric="Average Reliability Score",
            value=average_reliability,
            value_type="Score",
            signal=_quality_signal(average_reliability),
            interpretation="Average reliability score across quality-rated sources.",
        ),
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "source_coverage_summary.csv")
    return df


def build_input_confidence_summary(
    model_input_audit: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a summary of model input confidence."""

    if model_input_audit is None:
        model_input_audit = build_model_input_audit_table()

    if (
        model_input_audit is None
        or model_input_audit.empty
        or "Input Confidence Score" not in model_input_audit.columns
    ):
        df = pd.DataFrame(
            [
                _summary_row(
                    metric="Average Input Confidence Score",
                    value=0,
                    value_type="Score",
                    signal="Unavailable",
                    interpretation="Model input audit table was unavailable.",
                )
            ]
        )
        _safe_to_csv(df, "input_confidence_summary.csv")
        return df

    average_score = model_input_audit["Input Confidence Score"].mean()
    minimum_score = model_input_audit["Input Confidence Score"].min()
    maximum_score = model_input_audit["Input Confidence Score"].max()

    rows = [
        _summary_row(
            metric="Average Input Confidence Score",
            value=average_score,
            value_type="Score",
            signal=_quality_signal(average_score),
            interpretation="Average confidence score across key model inputs.",
        ),
        _summary_row(
            metric="Lowest Input Confidence Score",
            value=minimum_score,
            value_type="Score",
            signal=_quality_signal(minimum_score),
            interpretation="Lowest confidence score among key model inputs.",
        ),
        _summary_row(
            metric="Highest Input Confidence Score",
            value=maximum_score,
            value_type="Score",
            signal=_quality_signal(maximum_score),
            interpretation="Highest confidence score among key model inputs.",
        ),
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "input_confidence_summary.csv")
    return df


def build_missing_source_url_table(
    source_catalog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Flag source catalog entries that do not have configured URLs."""

    if source_catalog is None:
        source_catalog = build_source_catalog()

    if source_catalog is None or source_catalog.empty:
        df = pd.DataFrame(
            [
                {
                    "Source": "No source catalog available",
                    "URL Available": False,
                    "URL Available Type": "Text",
                    "Signal": "Unavailable",
                    "Interpretation": "Source catalog was unavailable.",
                }
            ]
        )
        _safe_to_csv(df, "missing_source_urls.csv")
        return df

    missing = source_catalog.loc[
        source_catalog["URL Available"] == False,  # noqa: E712
        ["Source", "Type", "Priority", "Valuation Relevance", "URL Available"],
    ].copy()

    if missing.empty:
        missing = pd.DataFrame(
            [
                {
                    "Source": "All cataloged sources have URLs",
                    "Type": "Reference",
                    "Priority": "Reference",
                    "Valuation Relevance": "Reference",
                    "URL Available": True,
                }
            ]
        )

    missing["URL Available Type"] = "Text"
    missing["Signal"] = missing["URL Available"].apply(
        lambda value: "Complete" if value else "Needs URL Review"
    )
    missing["Interpretation"] = missing["URL Available"].apply(
        lambda value: (
            "No missing URL action required."
            if value
            else "Add or verify URL in SOURCE_URLS within config.py."
        )
    )

    _safe_to_csv(missing, "missing_source_urls.csv")
    return missing


def build_source_driven_tables() -> dict[str, pd.DataFrame]:
    """Build all source-driven audit and support tables."""

    source_catalog = build_source_catalog()
    source_quality = build_source_quality_table()
    model_input_audit = build_model_input_audit_table()

    source_coverage_summary = build_source_coverage_summary(
        source_catalog=source_catalog,
        source_quality=source_quality,
    )
    input_confidence_summary = build_input_confidence_summary(
        model_input_audit=model_input_audit,
    )
    missing_source_urls = build_missing_source_url_table(
        source_catalog=source_catalog,
    )

    return {
        "source_catalog": source_catalog,
        "source_quality": source_quality,
        "management_guidance": build_management_guidance_table(),
        "backlog_pipeline": build_backlog_and_pipeline_table(),
        "revenue_mix": build_revenue_mix_table(),
        "capital_allocation": build_capital_allocation_table(),
        "risk_factors": build_risk_factor_table(),
        "model_input_audit": model_input_audit,
        "source_coverage_summary": source_coverage_summary,
        "input_confidence_summary": input_confidence_summary,
        "missing_source_urls": missing_source_urls,
    }