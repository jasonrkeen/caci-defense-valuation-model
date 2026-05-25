from typing import Any

import pandas as pd
import yfinance as yf

from src.config import (
    TICKER,
    TABLES_DIR,
    BASE_REVENUE_M,
    BASE_FCF_M,
)


def _safe_dataframe(value: Any) -> pd.DataFrame:
    """Convert yfinance output into a DataFrame when possible."""

    if value is None:
        return pd.DataFrame()

    if isinstance(value, pd.DataFrame):
        return value.copy()

    if isinstance(value, pd.Series):
        return value.to_frame().T

    if isinstance(value, dict):
        return pd.DataFrame([value])

    try:
        return pd.DataFrame(value)
    except Exception:
        return pd.DataFrame()


def _clean_column_name(value: Any) -> str:
    """Clean column names for CSV/report use."""

    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
    )


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _normalize_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Normalize a Yahoo/yfinance table for CSV/report output."""

    if df.empty:
        return pd.DataFrame(
            {
                "Table": [table_name],
                "Metric": ["No data returned"],
                "Value": ["Update manually or retry yfinance later"],
            }
        )

    normalized = df.copy()

    if not isinstance(normalized.index, pd.RangeIndex):
        index_name = normalized.index.name or "Metric"
        normalized = normalized.reset_index().rename(columns={index_name: "Metric"})

    normalized.columns = [_clean_column_name(c) for c in normalized.columns]

    if "Table" not in normalized.columns:
        normalized.insert(0, "Table", table_name)

    return normalized


def _try_get_yfinance_table(
    ticker: yf.Ticker,
    attribute: str,
    method_name: str,
) -> pd.DataFrame:
    """Try both yfinance method-style and property-style access."""

    try:
        method = getattr(ticker, method_name, None)

        if callable(method):
            raw = method()
            df = _safe_dataframe(raw)

            if not df.empty:
                return df

    except Exception:
        pass

    try:
        raw = getattr(ticker, attribute, None)
        df = _safe_dataframe(raw)

        if not df.empty:
            return df

    except Exception:
        pass

    return pd.DataFrame()


def collect_yahoo_analysis_tables() -> dict[str, pd.DataFrame]:
    """Collect Yahoo Finance analysis-style tables through yfinance."""

    ticker = yf.Ticker(TICKER)

    table_specs = {
        "Earnings Estimate": {
            "attribute": "earnings_estimate",
            "method": "get_earnings_estimate",
        },
        "Revenue Estimate": {
            "attribute": "revenue_estimate",
            "method": "get_revenue_estimate",
        },
        "Earnings History": {
            "attribute": "earnings_history",
            "method": "get_earnings_history",
        },
        "EPS Trend": {
            "attribute": "eps_trend",
            "method": "get_eps_trend",
        },
        "EPS Revisions": {
            "attribute": "eps_revisions",
            "method": "get_eps_revisions",
        },
        "Growth Estimates": {
            "attribute": "growth_estimates",
            "method": "get_growth_estimates",
        },
        "Analyst Price Targets": {
            "attribute": "analyst_price_targets",
            "method": "get_analyst_price_targets",
        },
    }

    results = {}

    for label, spec in table_specs.items():
        df = _try_get_yfinance_table(
            ticker=ticker,
            attribute=spec["attribute"],
            method_name=spec["method"],
        )

        normalized = _normalize_table(df, label)

        file_name = label.lower().replace(" ", "_").replace("/", "_")
        _safe_to_csv(normalized, f"yahoo_{file_name}.csv")

        results[label] = normalized

    return results


def build_analyst_estimate_summary(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize which Yahoo Finance analysis tables were available."""

    rows = []

    for table_name, df in tables.items():
        no_data = (
            df.empty
            or (
                "Metric" in df.columns
                and "No data returned" in df.astype(str).to_string()
            )
        )

        rows.append(
            {
                "Source Table": table_name,
                "Available": "No" if no_data else "Yes",
                "Rows": 0 if no_data else len(df),
                "Columns": 0 if no_data else len(df.columns),
                "Summary": (
                    "No data returned"
                    if no_data
                    else f"{len(df)} row(s), {len(df.columns)} column(s) collected"
                ),
            }
        )

    summary = pd.DataFrame(rows)
    _safe_to_csv(summary, "analyst_estimate_summary.csv")

    return summary


def build_analyst_source_quality_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Create a source-quality table for the report."""

    available_count = int((summary["Available"] == "Yes").sum()) if not summary.empty else 0
    total_count = len(summary)

    availability_rate = available_count / total_count if total_count > 0 else 0.0

    if availability_rate >= 0.75:
        signal = "Strong live estimate coverage"
        score = 85
    elif availability_rate >= 0.50:
        signal = "Moderate live estimate coverage"
        score = 70
    elif availability_rate > 0:
        signal = "Limited live estimate coverage"
        score = 55
    else:
        signal = "Fallback guidance inputs required"
        score = 45

    df = pd.DataFrame(
        [
            {
                "Metric": "Yahoo Estimate Table Availability",
                "Value": availability_rate,
                "Value Type": "Percent",
                "Signal": signal,
                "Score": score,
                "Score Type": "Score",
                "Interpretation": (
                    "Measures how many yfinance estimate tables were available during this run."
                ),
            },
            {
                "Metric": "Available Tables",
                "Value": available_count,
                "Value Type": "Plain Number",
                "Signal": signal,
                "Score": score,
                "Score Type": "Score",
                "Interpretation": "Number of Yahoo/yfinance tables that returned usable data.",
            },
            {
                "Metric": "Total Tables Requested",
                "Value": total_count,
                "Value Type": "Plain Number",
                "Signal": signal,
                "Score": score,
                "Score Type": "Score",
                "Interpretation": "Number of analyst estimate tables requested by the model.",
            },
        ]
    )

    _safe_to_csv(df, "analyst_source_quality.csv")

    return df


def _parse_money(value: Any) -> float | None:
    """Parse money values and return values in millions.

    Handles:
    - 9.55B
    - 9550M
    - $9.55B
    - 9,550,000,000
    - raw yfinance values in dollars
    - raw yfinance values in millions
    """

    if value is None or pd.isna(value):
        return None

    text = str(value).strip().upper()
    text = text.replace("$", "").replace(",", "").replace("%", "")

    if text in {"", "N/A", "NA", "NONE", "NAN", "--"}:
        return None

    multiplier = 1.0
    has_suffix = False

    if text.endswith("B"):
        multiplier = 1000.0
        text = text[:-1]
        has_suffix = True
    elif text.endswith("M"):
        multiplier = 1.0
        text = text[:-1]
        has_suffix = True
    elif text.endswith("K"):
        multiplier = 0.001
        text = text[:-1]
        has_suffix = True

    try:
        numeric_value = float(text)
    except Exception:
        return None

    parsed = numeric_value * multiplier

    # If yfinance gives raw dollars, convert to millions.
    # Example: 9,550,000,000 -> 9550.
    if not has_suffix and abs(parsed) > 1_000_000:
        parsed = parsed / 1_000_000

    return float(parsed)


def _find_avg_estimate_row(df: pd.DataFrame) -> pd.Series | None:
    """Find the row that likely contains average revenue estimates."""

    if df.empty:
        return None

    search_df = df.copy()

    preferred_patterns = [
        "avg",
        "average",
        "mean",
        "estimate",
        "revenue estimate",
    ]

    for col in search_df.columns:
        col_lower = str(col).lower()

        if col_lower in {"metric", "index", "level_0", "period"}:
            matches = search_df[
                search_df[col]
                .astype(str)
                .str.lower()
                .str.contains("|".join(preferred_patterns), regex=True, na=False)
            ]

            if not matches.empty:
                return matches.iloc[0]

    matches = search_df[
        search_df.astype(str).apply(
            lambda row: row.str.lower().str.contains(
                "|".join(preferred_patterns),
                regex=True,
                na=False,
            ).any(),
            axis=1,
        )
    ]

    if not matches.empty:
        return matches.iloc[0]

    numeric_columns = [
        col for col in search_df.columns
        if pd.to_numeric(search_df[col], errors="coerce").notna().any()
    ]

    if numeric_columns:
        return search_df.iloc[0]

    return None


def _clean_period_label(value: Any) -> str:
    """Normalize period labels for matching current-year and next-year columns."""

    return (
        str(value)
        .lower()
        .replace("_", "")
        .replace(" ", "")
        .replace("-", "")
        .replace(".", "")
    )


def _find_period_column(df: pd.DataFrame, period_patterns: list[str]) -> str | None:
    """Find a current-year / next-year style column."""

    cleaned_patterns = [_clean_period_label(pattern) for pattern in period_patterns]

    for col in df.columns:
        col_clean = _clean_period_label(col)

        for pattern in cleaned_patterns:
            if pattern in col_clean:
                return col

    return None


def _find_revenue_value_columns(df: pd.DataFrame) -> list[str]:
    """Find likely revenue estimate value columns in left-to-right order."""

    excluded_tokens = [
        "table",
        "metric",
        "index",
        "level",
        "period",
        "growth",
    ]

    candidate_cols = []

    for col in df.columns:
        col_clean = _clean_period_label(col)

        if any(token in col_clean for token in excluded_tokens):
            continue

        non_null = df[col].dropna()

        if non_null.empty:
            continue

        parsed_values = non_null.map(_parse_money)
        usable_count = parsed_values.notna().sum()

        if usable_count > 0:
            candidate_cols.append(col)

    return candidate_cols


def extract_revenue_growth_hint(tables: dict[str, pd.DataFrame]) -> float | None:
    """Infer next-year revenue growth from Yahoo revenue estimates.

    Returns a decimal growth rate, such as 0.072 for 7.2%.
    Returns None if revenue estimates are unavailable or cannot be parsed safely.
    """

    revenue_df = tables.get("Revenue Estimate", pd.DataFrame())

    if revenue_df.empty:
        return None

    if "No data returned" in revenue_df.astype(str).to_string():
        return None

    df = revenue_df.copy()

    current_col = _find_period_column(
        df,
        [
            "currentYear",
            "current_year",
            "current year",
            "currentyr",
            "current fiscal year",
            "0y",
            "year0",
            "2026",
            "2026E",
            "FY2026",
            "currentFY",
        ],
    )

    next_col = _find_period_column(
        df,
        [
            "nextYear",
            "next_year",
            "next year",
            "nextyr",
            "next fiscal year",
            "+1y",
            "1y",
            "year1",
            "2027",
            "2027E",
            "FY2027",
            "nextFY",
        ],
    )

    # If explicit year columns are unavailable, fall back to the first two
    # numeric/money-like columns after metadata columns.
    if current_col is None or next_col is None:
        candidate_cols = _find_revenue_value_columns(df)

        if len(candidate_cols) >= 2:
            current_col = candidate_cols[0]
            next_col = candidate_cols[1]
        else:
            return None

    row = _find_avg_estimate_row(df)

    if row is None:
        return None

    current_revenue = _parse_money(row[current_col])
    next_revenue = _parse_money(row[next_col])

    if current_revenue is None or next_revenue is None:
        return None

    if current_revenue <= 0:
        return None

    growth = (next_revenue / current_revenue) - 1

    # Guard against bad parsing or quarterly/yearly mismatches.
    if growth < -0.50 or growth > 0.75:
        return None

    return float(growth)


def build_manual_analyst_input_table() -> pd.DataFrame:
    """Fallback table when Yahoo estimate tables are unavailable."""

    rows = [
        {
            "Metric": "FY2026 revenue guidance midpoint",
            "Value": BASE_REVENUE_M,
            "Value Type": "Money Millions",
            "Model Use": "DCF starting revenue base",
            "Source": "CACI Q3 FY2026 guidance",
            "Confidence": "High",
        },
        {
            "Metric": "FY2026 free cash flow guidance",
            "Value": BASE_FCF_M,
            "Value Type": "Money Millions",
            "Model Use": "DCF free cash flow reasonableness check",
            "Source": "CACI Q3 FY2026 guidance",
            "Confidence": "High",
        },
        {
            "Metric": "Mid-term revenue growth framework",
            "Value": "High-single-digit CAGR",
            "Value Type": "Text",
            "Model Use": "DCF revenue growth scenario support",
            "Source": "CACI January 2026 investor presentation",
            "Confidence": "Medium / High",
        },
        {
            "Metric": "Mid-term free cash flow framework",
            "Value": "Greater than 15% CAGR",
            "Value Type": "Text",
            "Model Use": "Bull-case FCF expansion support",
            "Source": "CACI January 2026 investor presentation",
            "Confidence": "Medium / High",
        },
        {
            "Metric": "Analyst estimate availability",
            "Value": "Variable through yfinance",
            "Value Type": "Text",
            "Model Use": "Optional revenue growth hint",
            "Source": "Yahoo Finance / yfinance",
            "Confidence": "Medium",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "manual_analyst_inputs.csv")

    return df


def build_complete_analyst_package() -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
    float | None,
]:
    """One-call helper for main.py."""

    tables = collect_yahoo_analysis_tables()
    summary = build_analyst_estimate_summary(tables)
    build_analyst_source_quality_table(summary)

    revenue_growth_hint = extract_revenue_growth_hint(tables)

    build_manual_analyst_input_table()

    return tables, summary, revenue_growth_hint