import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROCESSED_DIR, TABLES_DIR, BENCHMARK_NAME


SNAPSHOT_PATH = PROCESSED_DIR / "last_run_snapshot.json"


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


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _read_optional_table(file_name: str) -> pd.DataFrame:
    """Read an optional CSV table from outputs/tables."""

    path = TABLES_DIR / file_name

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read optional table {file_name}: {exc}")
        return pd.DataFrame()


def _format_percentage_value(value: float) -> str:
    """Format decimal or already-percent values safely."""

    if pd.isna(value):
        return ""

    if -5 <= value <= 5:
        return f"{value:.2%}"

    if -100 <= value <= 100:
        return f"{value:.2f}%"

    return f"{value:,.2f}%"


def _format_money_value(value: float, metric_name: str = "") -> str:
    """Format money-like values."""

    if pd.isna(value):
        return ""

    metric = str(metric_name).lower()

    if "topline" in metric or "appropriated" in metric:
        return f"${value:,.2f}B"

    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.1f}B"

    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.1f}M"

    if abs(value) >= 1000:
        return f"${value:,.2f}"

    return f"${value:,.2f}"


def _format_money_millions_value(value: float) -> str:
    """Format values stored in millions of dollars."""

    if pd.isna(value):
        return ""

    if abs(value) >= 1000:
        return f"${value / 1000:,.2f}B"

    return f"${value:,.0f}M"


def _format_plain_number(value: float) -> str:
    """Format a plain numeric value."""

    if pd.isna(value):
        return ""

    if float(value).is_integer():
        return f"{value:,.0f}"

    return f"{value:,.2f}"


def _metric_value_type(metric_name: str) -> str:
    """Classify a snapshot metric for display formatting."""

    metric = str(metric_name).lower()

    if "final signal" in metric or "worst stress case" in metric:
        return "Text"

    if any(
        token in metric
        for token in [
            "price",
            "fair value",
            "monte carlo median",
            "monte carlo p10",
            "monte carlo p90",
            "topline",
            "appropriated",
        ]
    ):
        return "Money"

    if any(
        token in metric
        for token in [
            "return",
            "rate",
            "drawdown",
            "upside",
            "downside",
            "correlation",
            "risk-free",
            "growth",
            "yield",
            "adjustment",
        ]
    ):
        return "Percent"

    if "score" in metric:
        return "Score"

    if any(
        token in metric
        for token in [
            "headlines",
            "count",
            "observations",
            "exceptions",
        ]
    ):
        return "Plain Number"

    if any(
        token in metric
        for token in [
            "sharpe",
            "sortino",
            "beta",
            "ratio",
        ]
    ):
        return "Plain Number"

    return "Plain Number"


def _format_metric_value(value, metric_name: str, value_type: str | None = None) -> str:
    """Format previous/current metric values for the PDF-facing change table."""

    if value is None:
        return ""

    value_type_clean = str(value_type or _metric_value_type(metric_name)).strip().lower()

    if value_type_clean == "text":
        return str(value)

    numeric_value = _safe_float(value)

    if pd.isna(numeric_value):
        return "" if value is None else str(value)

    if value_type_clean in ["money", "currency", "dollar", "dollars"]:
        return _format_money_value(numeric_value, metric_name=metric_name)

    if value_type_clean in [
        "money millions",
        "money m",
        "millions",
        "millions usd",
        "usd millions",
    ]:
        return _format_money_millions_value(numeric_value)

    if value_type_clean in ["percent", "percentage", "rate"]:
        return _format_percentage_value(numeric_value)

    if value_type_clean in ["score"]:
        return _format_plain_number(numeric_value)

    return _format_plain_number(numeric_value)


def _format_change_value(change: float, metric_name: str, value_type: str) -> str:
    """Format absolute change using the metric's value type."""

    if pd.isna(change):
        return ""

    value_type_clean = str(value_type).strip().lower()

    if value_type_clean in ["money", "currency", "dollar", "dollars"]:
        return _format_money_value(change, metric_name=metric_name)

    if value_type_clean in [
        "money millions",
        "money m",
        "millions",
        "millions usd",
        "usd millions",
    ]:
        return _format_money_millions_value(change)

    if value_type_clean in ["percent", "percentage", "rate"]:
        return _format_percentage_value(change)

    if value_type_clean in ["score"]:
        return _format_plain_number(change)

    if value_type_clean == "text":
        return ""

    return _format_plain_number(change)


def _get_metric_value(metrics_df: pd.DataFrame, metric_name: str, default=np.nan) -> float:
    """Pull a numeric value from a Metric / Value table."""

    if metrics_df is None or metrics_df.empty:
        return default

    if "Metric" not in metrics_df.columns or "Value" not in metrics_df.columns:
        return default

    match = metrics_df.loc[metrics_df["Metric"] == metric_name, "Value"]

    if match.empty:
        return default

    return _safe_float(match.iloc[0], default=default)


def _get_metric_text(
    metrics_df: pd.DataFrame,
    metric_name: str,
    default: str = "Unavailable",
) -> str:
    """Pull a text value from a Metric / Value table."""

    if metrics_df is None or metrics_df.empty:
        return default

    if "Metric" not in metrics_df.columns or "Value" not in metrics_df.columns:
        return default

    match = metrics_df.loc[metrics_df["Metric"] == metric_name, "Value"]

    if match.empty:
        return default

    value = match.iloc[0]

    if pd.isna(value):
        return default

    return str(value)


def _get_score_value(score_df: pd.DataFrame, component_name: str, default=np.nan) -> float:
    """Pull a score from the intelligence score table."""

    if score_df is None or score_df.empty:
        return default

    if "Component" not in score_df.columns or "Score" not in score_df.columns:
        return default

    match = score_df.loc[score_df["Component"] == component_name, "Score"]

    if match.empty:
        return default

    return _safe_float(match.iloc[0], default=default)


def _get_signal_value(
    score_df: pd.DataFrame,
    component_name: str,
    default="Unavailable",
) -> str:
    """Pull a signal from the intelligence score table."""

    if score_df is None or score_df.empty:
        return default

    if "Component" not in score_df.columns or "Signal" not in score_df.columns:
        return default

    match = score_df.loc[score_df["Component"] == component_name, "Signal"]

    if match.empty:
        return default

    value = match.iloc[0]

    if pd.isna(value) or str(value).strip() == "":
        return default

    return str(value)


def _get_dcf_value(dcf_summary: pd.DataFrame, scenario: str, default=np.nan) -> float:
    """Pull fair value per share from DCF summary."""

    if dcf_summary is None or dcf_summary.empty:
        return default

    required_columns = {"Scenario", "Fair Value Per Share"}

    if not required_columns.issubset(set(dcf_summary.columns)):
        return default

    match = dcf_summary.loc[
        dcf_summary["Scenario"] == scenario,
        "Fair Value Per Share",
    ]

    if match.empty:
        return default

    return _safe_float(match.iloc[0], default=default)


def _get_table_metric_value(
    table: pd.DataFrame | None,
    metric_name: str,
    default=np.nan,
) -> float:
    """Pull a value from any Metric / Value table."""

    if table is None or table.empty:
        return default

    if "Metric" not in table.columns or "Value" not in table.columns:
        return default

    match = table.loc[table["Metric"] == metric_name, "Value"]

    if match.empty:
        return default

    return _safe_float(match.iloc[0], default=default)


def _get_table_metric_text(
    table: pd.DataFrame | None,
    metric_name: str,
    default: str = "Unavailable",
) -> str:
    """Pull a text value from any Metric / Value table."""

    if table is None or table.empty:
        return default

    if "Metric" not in table.columns or "Value" not in table.columns:
        return default

    match = table.loc[table["Metric"] == metric_name, "Value"]

    if match.empty:
        return default

    value = match.iloc[0]

    if pd.isna(value):
        return default

    return str(value)


def _get_nested_table_metric_value(
    table_dict: dict[str, pd.DataFrame] | None,
    table_key: str,
    metric_name: str,
    default=np.nan,
) -> float:
    """Pull a value from a nested dictionary table using Metric / Value format."""

    if not table_dict:
        return default

    table = table_dict.get(table_key)

    return _get_table_metric_value(table, metric_name, default=default)


def _get_nested_table_metric_text(
    table_dict: dict[str, pd.DataFrame] | None,
    table_key: str,
    metric_name: str,
    default: str = "Unavailable",
) -> str:
    """Pull a text value from a nested dictionary table using Metric / Value format."""

    if not table_dict:
        return default

    table = table_dict.get(table_key)

    return _get_table_metric_text(table, metric_name, default=default)


def _get_optional_metric_value(
    table_dict: dict[str, pd.DataFrame] | None,
    table_key: str,
    fallback_file_name: str,
    metric_name: str,
    default=np.nan,
) -> float:
    """Pull a metric from a table dictionary, falling back to CSV."""

    value = _get_nested_table_metric_value(
        table_dict=table_dict,
        table_key=table_key,
        metric_name=metric_name,
        default=np.nan,
    )

    if not pd.isna(value):
        return value

    fallback = _read_optional_table(fallback_file_name)

    return _get_table_metric_value(fallback, metric_name, default=default)


def _get_optional_metric_text(
    table_dict: dict[str, pd.DataFrame] | None,
    table_key: str,
    fallback_file_name: str,
    metric_name: str,
    default: str = "Unavailable",
) -> str:
    """Pull a text metric from a table dictionary, falling back to CSV."""

    value = _get_nested_table_metric_text(
        table_dict=table_dict,
        table_key=table_key,
        metric_name=metric_name,
        default="Unavailable",
    )

    if value != "Unavailable":
        return value

    fallback = _read_optional_table(fallback_file_name)

    return _get_table_metric_text(fallback, metric_name, default=default)


def _get_mc_summary_value(
    mc_summary: pd.DataFrame,
    contains_text: str,
    default=np.nan,
) -> float:
    """Pull a Monte Carlo summary metric by partial name match."""

    if mc_summary is None or mc_summary.empty:
        return default

    if "Metric" not in mc_summary.columns or "Value" not in mc_summary.columns:
        return default

    match = mc_summary.loc[
        mc_summary["Metric"].astype(str).str.contains(
            contains_text,
            case=False,
            na=False,
        ),
        "Value",
    ]

    if match.empty:
        return default

    return _safe_float(match.iloc[0], default=default)


def build_current_snapshot(
    market_metrics: pd.DataFrame,
    dcf_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
    score_df: pd.DataFrame,
    defense_risk_tables: dict[str, pd.DataFrame] | None = None,
    expected_return_tables: dict[str, pd.DataFrame] | None = None,
    intelligence_spending_tables: dict[str, pd.DataFrame] | None = None,
    forecast_tables: dict[str, pd.DataFrame] | None = None,
    news_intelligence_tables: dict[str, pd.DataFrame] | None = None,
    stress_test_tables: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """Build a compact snapshot of the most important run outputs."""

    current_price = _get_metric_value(market_metrics, "Latest Price")
    base_fair_value = _get_dcf_value(dcf_summary, "Base")
    bear_fair_value = _get_dcf_value(dcf_summary, "Bear")
    bull_fair_value = _get_dcf_value(dcf_summary, "Bull")

    risk_adjusted_expected_return = _get_nested_table_metric_value(
        expected_return_tables,
        "expected_return_summary",
        "Risk-Adjusted Expected Return",
    )

    raw_expected_return = _get_nested_table_metric_value(
        expected_return_tables,
        "expected_return_summary",
        "Raw Expected Return",
    )

    stress_resilience_adjustment = _get_optional_metric_value(
        table_dict=expected_return_tables,
        table_key="expected_return_summary",
        fallback_file_name="expected_return_summary.csv",
        metric_name="Stress Resilience Adjustment",
    )

    confidence_score = _get_nested_table_metric_value(
        expected_return_tables,
        "expected_return_summary",
        "Confidence Score",
    )

    stress_resilience_score = _get_optional_metric_value(
        table_dict=stress_test_tables,
        table_key="stress_test_score",
        fallback_file_name="stress_test_score.csv",
        metric_name="Stress Resilience Score",
    )

    average_stress_downside = _get_optional_metric_value(
        table_dict=stress_test_tables,
        table_key="stress_test_score",
        fallback_file_name="stress_test_score.csv",
        metric_name="Average Stress Downside",
    )

    worst_stress_downside = _get_optional_metric_value(
        table_dict=stress_test_tables,
        table_key="stress_test_score",
        fallback_file_name="stress_test_score.csv",
        metric_name="Worst Stress Downside",
    )

    worst_stress_case = _get_optional_metric_text(
        table_dict=stress_test_tables,
        table_key="stress_test_score",
        fallback_file_name="stress_test_score.csv",
        metric_name="Worst Stress Case",
    )

    defense_risk_score = _get_nested_table_metric_value(
        defense_risk_tables,
        "defense_risk_summary",
        "Overall Defense Risk Score",
    )

    intelligence_spending_score = _get_nested_table_metric_value(
        intelligence_spending_tables,
        "intelligence_spending_signal",
        "Intelligence Spending Signal Score",
    )

    latest_combined_intelligence_topline = _get_nested_table_metric_value(
        intelligence_spending_tables,
        "intelligence_spending_signal",
        "Latest Combined Appropriated Topline",
    )

    price_forecast_median = _get_nested_table_metric_value(
        forecast_tables,
        "price_forecast_summary",
        "Median Forecast Price",
    )

    price_forecast_return = _get_nested_table_metric_value(
        forecast_tables,
        "price_forecast_summary",
        "Median Forecast Return",
    )

    overall_news_score = _get_nested_table_metric_value(
        news_intelligence_tables,
        "news_intelligence_summary",
        "Overall News Intelligence Score",
    )

    live_rss_headlines = _get_nested_table_metric_value(
        news_intelligence_tables,
        "news_intelligence_summary",
        "Live RSS Headlines",
    )

    manual_fallback_headlines = _get_nested_table_metric_value(
        news_intelligence_tables,
        "news_intelligence_summary",
        "Manual / Fallback Headlines",
    )

    positive_headlines = _get_nested_table_metric_value(
        news_intelligence_tables,
        "news_intelligence_summary",
        "Positive Headlines",
    )

    neutral_headlines = _get_nested_table_metric_value(
        news_intelligence_tables,
        "news_intelligence_summary",
        "Neutral Headlines",
    )

    mixed_headlines = _get_nested_table_metric_value(
        news_intelligence_tables,
        "news_intelligence_summary",
        "Mixed Headlines",
    )

    negative_headlines = _get_nested_table_metric_value(
        news_intelligence_tables,
        "news_intelligence_summary",
        "Negative Headlines",
    )

    snapshot = {
        "Run Timestamp": datetime.now().isoformat(timespec="seconds"),
        "Current Price": current_price,
        "Bear Fair Value": bear_fair_value,
        "Base Fair Value": base_fair_value,
        "Bull Fair Value": bull_fair_value,
        "Base Upside / Downside": (
            base_fair_value / current_price - 1
            if current_price and current_price > 0 and not pd.isna(base_fair_value)
            else np.nan
        ),
        "Raw Expected Return": raw_expected_return,
        "Risk-Adjusted Expected Return": risk_adjusted_expected_return,
        "Stress Resilience Adjustment": stress_resilience_adjustment,
        "Confidence Score": confidence_score,
        "Median Forecast Price": price_forecast_median,
        "Median Forecast Return": price_forecast_return,
        "Monte Carlo Median Fair Value": _get_mc_summary_value(mc_summary, "Median"),
        "Monte Carlo P10": _get_mc_summary_value(mc_summary, "P10"),
        "Monte Carlo P90": _get_mc_summary_value(mc_summary, "P90"),
        "Defense Technology Intelligence Score": _get_score_value(score_df, "Overall Score"),
        "Final Signal": _get_signal_value(score_df, "Overall Score"),
        "Stress Resilience Score": stress_resilience_score,
        "Average Stress Downside": average_stress_downside,
        "Worst Stress Downside": worst_stress_downside,
        "Worst Stress Case": worst_stress_case,
        "Defense Risk Score": defense_risk_score,
        "Intelligence Spending Signal Score": intelligence_spending_score,
        "Latest Combined Intelligence Appropriated Topline": latest_combined_intelligence_topline,
        "Overall News Intelligence Score": overall_news_score,
        "Live RSS Headlines": live_rss_headlines,
        "Manual / Fallback Headlines": manual_fallback_headlines,
        "Positive Headlines": positive_headlines,
        "Neutral Headlines": neutral_headlines,
        "Mixed Headlines": mixed_headlines,
        "Negative Headlines": negative_headlines,
        "Risk-Free Rate": _get_metric_value(market_metrics, "Risk-Free Rate"),
        "Sharpe Ratio": _get_metric_value(market_metrics, "Sharpe Ratio"),
        "Sortino Ratio": _get_metric_value(market_metrics, "Sortino Ratio"),
        f"Beta vs {BENCHMARK_NAME}": _get_metric_value(
            market_metrics,
            f"Beta vs {BENCHMARK_NAME}",
        ),
        f"Correlation vs {BENCHMARK_NAME}": _get_metric_value(
            market_metrics,
            f"Correlation vs {BENCHMARK_NAME}",
        ),
        "Maximum Drawdown": _get_metric_value(market_metrics, "Maximum Drawdown"),
        "One-Year Return": _get_metric_value(market_metrics, "One-Year Return"),
    }

    return snapshot


def load_previous_snapshot(snapshot_path: Path = SNAPSHOT_PATH) -> dict | None:
    """Load the prior run snapshot, if available."""

    if not snapshot_path.exists():
        return None

    try:
        with open(snapshot_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def _make_json_serializable(value):
    """Convert numpy/pandas values into JSON-safe values."""

    if isinstance(value, (np.floating, np.integer)):
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, str):
        return value

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    return value


def save_snapshot(snapshot: dict, snapshot_path: Path = SNAPSHOT_PATH) -> None:
    """Save the current run snapshot as JSON and CSV."""

    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    serializable_snapshot = {
        key: _make_json_serializable(value)
        for key, value in snapshot.items()
    }

    with open(snapshot_path, "w", encoding="utf-8") as file:
        json.dump(serializable_snapshot, file, indent=4)

    snapshot_df = pd.DataFrame([serializable_snapshot])
    snapshot_df.to_csv(PROCESSED_DIR / "last_run_snapshot.csv", index=False)


def _classify_change(metric: str, change: float | None) -> str:
    """Classify whether a change is meaningful."""

    if change is None or pd.isna(change):
        return "Unavailable"

    absolute_change = abs(change)

    percentage_like_metrics = [
        "Return",
        "Rate",
        "Ratio",
        "Drawdown",
        "Upside",
        "Downside",
        "WACC",
        "Growth",
        "Yield",
        "Adjustment",
    ]

    count_like_metrics = [
        "Headlines",
    ]

    if any(token in metric for token in percentage_like_metrics):
        if absolute_change >= 0.05:
            return "Major Change"
        if absolute_change >= 0.02:
            return "Moderate Change"
        if absolute_change >= 0.005:
            return "Minor Change"
        return "Stable"

    if any(token in metric for token in count_like_metrics):
        if absolute_change >= 10:
            return "Major Change"
        if absolute_change >= 5:
            return "Moderate Change"
        if absolute_change >= 1:
            return "Minor Change"
        return "Stable"

    if "Fair Value" in metric or "Price" in metric:
        if absolute_change >= 25:
            return "Major Change"
        if absolute_change >= 10:
            return "Moderate Change"
        if absolute_change >= 3:
            return "Minor Change"
        return "Stable"

    if "Score" in metric:
        if absolute_change >= 10:
            return "Major Change"
        if absolute_change >= 5:
            return "Moderate Change"
        if absolute_change >= 2:
            return "Minor Change"
        return "Stable"

    if absolute_change != 0:
        return "Changed"

    return "Stable"


def _direction_from_change(change: float) -> str:
    """Classify direction of numeric change."""

    if change > 0:
        return "Increased"

    if change < 0:
        return "Decreased"

    return "Unchanged"


def compare_snapshots(
    previous_snapshot: dict | None,
    current_snapshot: dict,
) -> pd.DataFrame:
    """Compare previous run against current run."""

    if previous_snapshot is None:
        rows = [
            {
                "Metric": "Change Detection",
                "Previous": "No previous run",
                "Current": "Current snapshot saved",
                "Change": "",
                "Change %": "",
                "Direction": "",
                "Classification": "First Run",
                "Interpretation": "This is the first saved run. Future runs will show changes.",
                "Value Type": "Text",
                "Raw Previous": np.nan,
                "Raw Current": np.nan,
                "Raw Change": np.nan,
                "Raw Change %": np.nan,
            }
        ]

        df = pd.DataFrame(rows)
        _safe_to_csv(df, "change_detection.csv")
        return df

    rows = []
    ignored_keys = {"Run Timestamp"}

    for key, current_value in current_snapshot.items():
        if key in ignored_keys:
            continue

        previous_value = previous_snapshot.get(key)

        previous_numeric = _safe_float(previous_value)
        current_numeric = _safe_float(current_value)
        value_type = _metric_value_type(key)

        if pd.notna(previous_numeric) and pd.notna(current_numeric):
            change = current_numeric - previous_numeric
            change_pct = change / abs(previous_numeric) if previous_numeric != 0 else np.nan
            classification = _classify_change(key, change)
            direction = _direction_from_change(change)

            rows.append(
                {
                    "Metric": key,
                    "Previous": _format_metric_value(
                        previous_numeric,
                        metric_name=key,
                        value_type=value_type,
                    ),
                    "Current": _format_metric_value(
                        current_numeric,
                        metric_name=key,
                        value_type=value_type,
                    ),
                    "Change": _format_change_value(
                        change,
                        metric_name=key,
                        value_type=value_type,
                    ),
                    "Change %": _format_percentage_value(change_pct),
                    "Direction": direction,
                    "Classification": classification,
                    "Interpretation": f"{key} {direction.lower()} versus the prior run.",
                    "Value Type": value_type,
                    "Raw Previous": previous_numeric,
                    "Raw Current": current_numeric,
                    "Raw Change": change,
                    "Raw Change %": change_pct,
                }
            )

        else:
            previous_text = "" if previous_value is None else str(previous_value)
            current_text = "" if current_value is None else str(current_value)

            changed = previous_text != current_text

            rows.append(
                {
                    "Metric": key,
                    "Previous": previous_text,
                    "Current": current_text,
                    "Change": "",
                    "Change %": "",
                    "Direction": "Changed" if changed else "Unchanged",
                    "Classification": "Changed" if changed else "Stable",
                    "Interpretation": (
                        f"{key} changed from '{previous_text}' to '{current_text}'."
                        if changed
                        else f"{key} is unchanged."
                    ),
                    "Value Type": "Text",
                    "Raw Previous": np.nan,
                    "Raw Current": np.nan,
                    "Raw Change": np.nan,
                    "Raw Change %": np.nan,
                }
            )

    df = pd.DataFrame(rows)

    display_columns = [
        "Metric",
        "Previous",
        "Current",
        "Change",
        "Change %",
        "Direction",
        "Classification",
        "Interpretation",
        "Value Type",
        "Raw Previous",
        "Raw Current",
        "Raw Change",
        "Raw Change %",
    ]

    df = df[display_columns]

    _safe_to_csv(df, "change_detection.csv")

    return df


def build_change_summary(change_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact summary of the most important changes."""

    if change_df is None or change_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "Metric": "Change Summary",
                    "Value": "Unavailable",
                    "Value Type": "Text",
                    "Interpretation": "No change detection data was generated.",
                }
            ]
        )
        _safe_to_csv(summary, "change_summary.csv")
        return summary

    if "Classification" not in change_df.columns:
        summary = pd.DataFrame(
            [
                {
                    "Metric": "Change Summary",
                    "Value": "Unavailable",
                    "Value Type": "Text",
                    "Interpretation": "Change detection table has no classification column.",
                }
            ]
        )
        _safe_to_csv(summary, "change_summary.csv")
        return summary

    major_count = int((change_df["Classification"] == "Major Change").sum())
    moderate_count = int((change_df["Classification"] == "Moderate Change").sum())
    minor_count = int((change_df["Classification"] == "Minor Change").sum())
    stable_count = int((change_df["Classification"] == "Stable").sum())
    changed_count = int((change_df["Classification"] == "Changed").sum())

    meaningful_changes = change_df[
        change_df["Classification"].isin(["Major Change", "Moderate Change"])
    ].copy()

    if meaningful_changes.empty:
        primary_change = "No major or moderate changes"
        primary_interpretation = "The latest run is broadly consistent with the prior run."
    else:
        if "Raw Change" in meaningful_changes.columns:
            meaningful_changes["Abs Change"] = pd.to_numeric(
                meaningful_changes["Raw Change"],
                errors="coerce",
            ).abs()
        else:
            meaningful_changes["Abs Change"] = pd.to_numeric(
                meaningful_changes["Change"],
                errors="coerce",
            ).abs()

        meaningful_changes = meaningful_changes.sort_values(
            "Abs Change",
            ascending=False,
        )

        primary_row = meaningful_changes.iloc[0]
        primary_change = primary_row["Metric"]
        primary_interpretation = primary_row["Interpretation"]

    rows = [
        {
            "Metric": "Major Changes",
            "Value": major_count,
            "Value Type": "Plain Number",
            "Interpretation": "Number of metrics with major movement versus the prior run",
        },
        {
            "Metric": "Moderate Changes",
            "Value": moderate_count,
            "Value Type": "Plain Number",
            "Interpretation": "Number of metrics with moderate movement versus the prior run",
        },
        {
            "Metric": "Minor Changes",
            "Value": minor_count,
            "Value Type": "Plain Number",
            "Interpretation": "Number of metrics with minor movement versus the prior run",
        },
        {
            "Metric": "Changed Text Metrics",
            "Value": changed_count,
            "Value Type": "Plain Number",
            "Interpretation": "Number of non-numeric metrics that changed versus the prior run",
        },
        {
            "Metric": "Stable Metrics",
            "Value": stable_count,
            "Value Type": "Plain Number",
            "Interpretation": "Number of metrics with limited or no movement",
        },
        {
            "Metric": "Primary Change Driver",
            "Value": primary_change,
            "Value Type": "Text",
            "Interpretation": primary_interpretation,
        },
    ]

    summary = pd.DataFrame(rows)
    _safe_to_csv(summary, "change_summary.csv")

    return summary


def run_change_detection(
    market_metrics: pd.DataFrame,
    dcf_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
    score_df: pd.DataFrame,
    defense_risk_tables: dict[str, pd.DataFrame] | None = None,
    expected_return_tables: dict[str, pd.DataFrame] | None = None,
    intelligence_spending_tables: dict[str, pd.DataFrame] | None = None,
    forecast_tables: dict[str, pd.DataFrame] | None = None,
    news_intelligence_tables: dict[str, pd.DataFrame] | None = None,
    stress_test_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run full change detection workflow.

    This should be called near the end of main.py after the key models have run.
    """

    current_snapshot = build_current_snapshot(
        market_metrics=market_metrics,
        dcf_summary=dcf_summary,
        mc_summary=mc_summary,
        score_df=score_df,
        defense_risk_tables=defense_risk_tables,
        expected_return_tables=expected_return_tables,
        intelligence_spending_tables=intelligence_spending_tables,
        forecast_tables=forecast_tables,
        news_intelligence_tables=news_intelligence_tables,
        stress_test_tables=stress_test_tables,
    )

    previous_snapshot = load_previous_snapshot()

    change_detection = compare_snapshots(
        previous_snapshot=previous_snapshot,
        current_snapshot=current_snapshot,
    )

    change_summary = build_change_summary(change_detection)

    save_snapshot(current_snapshot)

    return {
        "change_detection": change_detection,
        "change_summary": change_summary,
    }