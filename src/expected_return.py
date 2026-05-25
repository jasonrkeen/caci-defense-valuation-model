import numpy as np
import pandas as pd

from src.config import (
    TABLES_DIR,
    PROCESSED_DIR,
    EXPECTED_RETURN_PROBABILITIES,
)


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


def _safe_to_csv(df: pd.DataFrame, file_name: str, output_dir=TABLES_DIR) -> None:
    """Save a DataFrame with basic safety."""

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_dir / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _read_optional_table(file_name: str) -> pd.DataFrame:
    """Read an optional CSV from outputs/tables."""

    path = TABLES_DIR / file_name

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read optional table {file_name}: {exc}")
        return pd.DataFrame()


def _get_metric_value(metrics_df: pd.DataFrame, metric_name: str, default=np.nan) -> float:
    """Pull a metric from a standard Metric / Value table."""

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
    """Pull a text metric from a standard Metric / Value table."""

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


def _get_table_metric_value(
    table: pd.DataFrame | None,
    metric_name: str,
    default=np.nan,
) -> float:
    """Pull a numeric metric from a Metric / Value table."""

    if table is None or table.empty:
        return default

    if "Metric" not in table.columns or "Value" not in table.columns:
        return default

    match = table.loc[table["Metric"] == metric_name, "Value"]

    if match.empty:
        return default

    return _safe_float(match.iloc[0], default=default)


def _get_nested_table_metric_value(
    table_dict: dict[str, pd.DataFrame] | None,
    table_key: str,
    metric_name: str,
    default=np.nan,
) -> float:
    """Pull a numeric metric from a nested dictionary of report tables."""

    if not table_dict:
        return default

    table = table_dict.get(table_key)

    return _get_table_metric_value(table, metric_name, default=default)


def _score_to_confidence(score: float) -> str:
    """Convert a confidence score into a label."""

    if pd.isna(score):
        return "Unavailable"

    if score >= 80:
        return "High"

    if score >= 65:
        return "Moderate / High"

    if score >= 50:
        return "Moderate"

    return "Low"


def _return_signal(expected_return: float) -> str:
    """Convert expected return into an investment signal."""

    if pd.isna(expected_return):
        return "Unavailable"

    if expected_return >= 0.25:
        return "Attractive Upside"

    if expected_return >= 0.10:
        return "Constructive"

    if expected_return >= -0.05:
        return "Neutral / Monitor"

    if expected_return >= -0.15:
        return "Cautious"

    return "Unattractive"


def _value_signal(value: float) -> str:
    """Signal helper for valuation / return rows."""

    if pd.isna(value):
        return "Unavailable"

    if value > 0:
        return "Positive"

    if value < 0:
        return "Negative"

    return "Neutral"


def _stress_adjustment_signal(value: float) -> str:
    """Signal helper for stress resilience adjustment."""

    if pd.isna(value):
        return "Unavailable"

    if value == 0:
        return "No Stress Haircut"

    if value >= -0.01:
        return "Limited Stress Haircut"

    if value >= -0.025:
        return "Moderate Stress Haircut"

    return "Elevated Stress Haircut"


def _get_dcf_fair_value(
    dcf_summary: pd.DataFrame,
    scenario: str,
    default=np.nan,
) -> float:
    """Pull fair value per share from the DCF summary."""

    if dcf_summary is None or dcf_summary.empty:
        return default

    required_columns = {"Scenario", "Fair Value Per Share"}

    if not required_columns.issubset(dcf_summary.columns):
        return default

    match = dcf_summary.loc[
        dcf_summary["Scenario"] == scenario,
        "Fair Value Per Share",
    ]

    if match.empty:
        return default

    return _safe_float(match.iloc[0], default=default)


def _get_monte_carlo_median(mc_summary: pd.DataFrame) -> float:
    """Pull Monte Carlo median fair value from summary table."""

    if mc_summary is None or mc_summary.empty:
        return np.nan

    if "Metric" not in mc_summary.columns or "Value" not in mc_summary.columns:
        return np.nan

    exact_match = mc_summary.loc[
        mc_summary["Metric"].astype(str).str.lower() == "median fair value",
        "Value",
    ]

    if not exact_match.empty:
        return _safe_float(exact_match.iloc[0])

    fallback_match = mc_summary.loc[
        mc_summary["Metric"].astype(str).str.contains(
            "Median",
            case=False,
            na=False,
        ),
        "Value",
    ]

    if fallback_match.empty:
        return np.nan

    return _safe_float(fallback_match.iloc[0])


def _get_intelligence_score(score_df: pd.DataFrame) -> float:
    """Pull the overall Defense Technology Intelligence Score."""

    if score_df is None or score_df.empty:
        return np.nan

    if "Component" not in score_df.columns or "Score" not in score_df.columns:
        return np.nan

    match = score_df.loc[
        score_df["Component"] == "Overall Score",
        "Score",
    ]

    if match.empty:
        return np.nan

    return _safe_float(match.iloc[0])


def _get_stress_resilience_score(default=np.nan) -> float:
    """Pull Stress Resilience Score from stress_test_score.csv."""

    stress_score = _read_optional_table("stress_test_score.csv")

    if stress_score.empty:
        return default

    return _get_metric_value(
        stress_score,
        metric_name="Stress Resilience Score",
        default=default,
    )


def _get_average_stress_downside(default=np.nan) -> float:
    """Pull Average Stress Downside from stress_test_score.csv."""

    stress_score = _read_optional_table("stress_test_score.csv")

    if stress_score.empty:
        return default

    return _get_metric_value(
        stress_score,
        metric_name="Average Stress Downside",
        default=default,
    )


def _get_worst_stress_case(default: str = "Unavailable") -> str:
    """Pull Worst Stress Case from stress_test_score.csv."""

    stress_score = _read_optional_table("stress_test_score.csv")

    if stress_score.empty:
        return default

    return _get_metric_text(
        stress_score,
        metric_name="Worst Stress Case",
        default=default,
    )


def _stress_adjustment_from_score(stress_score: float) -> float:
    """Translate Stress Resilience Score into an expected-return adjustment."""

    if pd.isna(stress_score):
        return 0.000

    if stress_score >= 80:
        return 0.000

    if stress_score >= 65:
        return -0.010

    if stress_score >= 50:
        return -0.025

    return -0.050


def _get_defense_adjustment(defense_risk_tables: dict[str, pd.DataFrame] | None) -> float:
    """Read the defense risk valuation adjustment from the risk engine output."""

    if not defense_risk_tables:
        return 0.0

    adjustments = defense_risk_tables.get("defense_risk_adjustments")

    if adjustments is None or adjustments.empty:
        return 0.0

    if "Adjustment" not in adjustments.columns or "Value" not in adjustments.columns:
        return 0.0

    match = adjustments.loc[
        adjustments["Adjustment"] == "Defense Risk Valuation Adjustment",
        "Value",
    ]

    if match.empty:
        return 0.0

    return _safe_float(match.iloc[0], default=0.0)


def _get_defense_confidence_adjustment(
    defense_risk_tables: dict[str, pd.DataFrame] | None,
) -> float:
    """Read the confidence adjustment from the defense risk engine output."""

    if not defense_risk_tables:
        return 0.0

    adjustments = defense_risk_tables.get("defense_risk_adjustments")

    if adjustments is None or adjustments.empty:
        return 0.0

    if "Adjustment" not in adjustments.columns or "Value" not in adjustments.columns:
        return 0.0

    match = adjustments.loc[
        adjustments["Adjustment"] == "Confidence Score Adjustment",
        "Value",
    ]

    if match.empty:
        return 0.0

    return _safe_float(match.iloc[0], default=0.0)


def _get_news_adjustment(news_intelligence_tables: dict[str, pd.DataFrame] | None) -> float:
    """Read the news valuation adjustment from the news intelligence model."""

    if not news_intelligence_tables:
        return 0.0

    adjustments = news_intelligence_tables.get("news_valuation_adjustments")

    if adjustments is None or adjustments.empty:
        return 0.0

    if "Adjustment" not in adjustments.columns or "Value" not in adjustments.columns:
        return 0.0

    match = adjustments.loc[
        adjustments["Adjustment"] == "News Valuation Adjustment",
        "Value",
    ]

    if match.empty:
        return 0.0

    return _safe_float(match.iloc[0], default=0.0)


def _get_news_confidence_adjustment(
    news_intelligence_tables: dict[str, pd.DataFrame] | None,
) -> float:
    """Read the news confidence adjustment from the news intelligence model."""

    if not news_intelligence_tables:
        return 0.0

    adjustments = news_intelligence_tables.get("news_valuation_adjustments")

    if adjustments is None or adjustments.empty:
        return 0.0

    if "Adjustment" not in adjustments.columns or "Value" not in adjustments.columns:
        return 0.0

    match = adjustments.loc[
        adjustments["Adjustment"] == "News Confidence Adjustment",
        "Value",
    ]

    if match.empty:
        return 0.0

    return _safe_float(match.iloc[0], default=0.0)


def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Normalize scenario probabilities so they sum to 1.0."""

    cleaned = {
        "Bear": _safe_float(probabilities.get("Bear"), default=0.25),
        "Base": _safe_float(probabilities.get("Base"), default=0.50),
        "Bull": _safe_float(probabilities.get("Bull"), default=0.25),
    }

    total = sum(cleaned.values())

    if total <= 0:
        return {
            "Bear": 0.25,
            "Base": 0.50,
            "Bull": 0.25,
        }

    return {
        scenario: value / total
        for scenario, value in cleaned.items()
    }


def _intelligence_haircut_from_score(intelligence_score: float) -> float:
    """Translate the overall intelligence score into a valuation haircut."""

    if pd.isna(intelligence_score):
        return 0.000

    if intelligence_score >= 80:
        return 0.000

    if intelligence_score >= 70:
        return -0.025

    if intelligence_score >= 60:
        return -0.050

    if intelligence_score >= 50:
        return -0.075

    return -0.100


def _row(
    metric: str,
    value,
    value_type: str,
    signal: str,
    interpretation: str,
) -> dict:
    """Create a standardized expected-return row."""

    return {
        "Metric": metric,
        "Value": value,
        "Value Type": value_type,
        "Signal": signal,
        "Interpretation": interpretation,
    }


def build_expected_return_table(
    market_metrics: pd.DataFrame,
    dcf_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
    score_df: pd.DataFrame,
    defense_risk_tables: dict[str, pd.DataFrame] | None = None,
    news_intelligence_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Build a probability-weighted expected return framework.

    This combines:
    - DCF bear/base/bull fair values
    - Current market price
    - Monte Carlo median fair value
    - Defense Technology Intelligence Score
    - Stress Resilience Score
    - Defense risk adjustment
    - Optional news intelligence adjustment
    - Risk-adjusted expected return
    """

    current_price = _get_metric_value(market_metrics, "Latest Price")

    bear_value = _get_dcf_fair_value(dcf_summary, "Bear")
    base_value = _get_dcf_fair_value(dcf_summary, "Base")
    bull_value = _get_dcf_fair_value(dcf_summary, "Bull")

    probabilities = _normalize_probabilities(EXPECTED_RETURN_PROBABILITIES)

    probability_weighted_value = (
        bear_value * probabilities["Bear"]
        + base_value * probabilities["Base"]
        + bull_value * probabilities["Bull"]
    )

    monte_carlo_median = _get_monte_carlo_median(mc_summary)

    raw_expected_return = (
        probability_weighted_value / current_price - 1
        if current_price and current_price > 0 and not pd.isna(probability_weighted_value)
        else np.nan
    )

    monte_carlo_expected_return = (
        monte_carlo_median / current_price - 1
        if current_price and current_price > 0 and not pd.isna(monte_carlo_median)
        else np.nan
    )

    intelligence_score = _get_intelligence_score(score_df)
    intelligence_haircut = _intelligence_haircut_from_score(intelligence_score)

    stress_resilience_score = _get_stress_resilience_score(default=np.nan)
    average_stress_downside = _get_average_stress_downside(default=np.nan)
    worst_stress_case = _get_worst_stress_case(default="Unavailable")
    stress_resilience_adjustment = _stress_adjustment_from_score(stress_resilience_score)

    defense_valuation_adjustment = _get_defense_adjustment(defense_risk_tables)
    defense_confidence_adjustment = _get_defense_confidence_adjustment(defense_risk_tables)

    news_valuation_adjustment = _get_news_adjustment(news_intelligence_tables)
    news_confidence_adjustment = _get_news_confidence_adjustment(news_intelligence_tables)

    risk_adjusted_expected_return = (
        raw_expected_return
        + intelligence_haircut
        + stress_resilience_adjustment
        + defense_valuation_adjustment
        + news_valuation_adjustment
        if not pd.isna(raw_expected_return)
        else np.nan
    )

    confidence_score = (
        intelligence_score
        + defense_confidence_adjustment * 100
        + news_confidence_adjustment * 100
        if not pd.isna(intelligence_score)
        else np.nan
    )

    confidence_score = (
        max(min(confidence_score, 100), 0)
        if not pd.isna(confidence_score)
        else np.nan
    )

    confidence_label = _score_to_confidence(confidence_score)
    final_return_signal = _return_signal(risk_adjusted_expected_return)

    stress_interpretation = (
        f"Stress adjustment based on Stress Resilience Score. "
        f"Average stress downside is {average_stress_downside:.1%}; "
        f"worst case is {worst_stress_case}."
        if not pd.isna(average_stress_downside)
        else (
            "Stress adjustment based on Stress Resilience Score. "
            f"Worst case is {worst_stress_case}."
        )
    )

    rows = [
        _row(
            metric="Current Share Price",
            value=current_price,
            value_type="Money",
            signal="Reference",
            interpretation="Latest market price from Yahoo Finance.",
        ),
        _row(
            metric="Bear Case Fair Value",
            value=bear_value,
            value_type="Money",
            signal=_value_signal(
                bear_value - current_price
                if not pd.isna(current_price)
                else np.nan
            ),
            interpretation=f"DCF downside scenario with {probabilities['Bear']:.0%} probability weight.",
        ),
        _row(
            metric="Base Case Fair Value",
            value=base_value,
            value_type="Money",
            signal=_value_signal(
                base_value - current_price
                if not pd.isna(current_price)
                else np.nan
            ),
            interpretation=f"DCF base scenario with {probabilities['Base']:.0%} probability weight.",
        ),
        _row(
            metric="Bull Case Fair Value",
            value=bull_value,
            value_type="Money",
            signal=_value_signal(
                bull_value - current_price
                if not pd.isna(current_price)
                else np.nan
            ),
            interpretation=f"DCF upside scenario with {probabilities['Bull']:.0%} probability weight.",
        ),
        _row(
            metric="Bear Scenario Probability",
            value=probabilities["Bear"],
            value_type="Percent",
            signal="Reference",
            interpretation="Probability weight assigned to the bear-case DCF scenario.",
        ),
        _row(
            metric="Base Scenario Probability",
            value=probabilities["Base"],
            value_type="Percent",
            signal="Reference",
            interpretation="Probability weight assigned to the base-case DCF scenario.",
        ),
        _row(
            metric="Bull Scenario Probability",
            value=probabilities["Bull"],
            value_type="Percent",
            signal="Reference",
            interpretation="Probability weight assigned to the bull-case DCF scenario.",
        ),
        _row(
            metric="Probability-Weighted Fair Value",
            value=probability_weighted_value,
            value_type="Money",
            signal=_value_signal(
                probability_weighted_value - current_price
                if not pd.isna(current_price)
                else np.nan
            ),
            interpretation=(
                "Bear, base, and bull fair values weighted by the scenario probabilities above."
            ),
        ),
        _row(
            metric="Monte Carlo Median Fair Value",
            value=monte_carlo_median,
            value_type="Money",
            signal=_value_signal(
                monte_carlo_median - current_price
                if not pd.isna(current_price)
                else np.nan
            ),
            interpretation="Median simulated fair value from Monte Carlo model.",
        ),
        _row(
            metric="Raw Expected Return",
            value=raw_expected_return,
            value_type="Percent",
            signal=_return_signal(raw_expected_return),
            interpretation="Probability-weighted fair value versus current share price.",
        ),
        _row(
            metric="Monte Carlo Expected Return",
            value=monte_carlo_expected_return,
            value_type="Percent",
            signal=_return_signal(monte_carlo_expected_return),
            interpretation="Monte Carlo median fair value versus current share price.",
        ),
        _row(
            metric="Intelligence Score",
            value=intelligence_score,
            value_type="Score",
            signal=_score_to_confidence(intelligence_score),
            interpretation="Defense Technology Intelligence Score used to determine the valuation haircut.",
        ),
        _row(
            metric="Intelligence Score Haircut",
            value=intelligence_haircut,
            value_type="Percent",
            signal="Risk Adjustment",
            interpretation="Risk haircut based on the Defense Technology Intelligence Score.",
        ),
        _row(
            metric="Stress Resilience Score",
            value=stress_resilience_score,
            value_type="Score",
            signal=_score_to_confidence(stress_resilience_score),
            interpretation=(
                "Stress-test resilience score used to determine the expected-return stress haircut."
            ),
        ),
        _row(
            metric="Stress Resilience Adjustment",
            value=stress_resilience_adjustment,
            value_type="Percent",
            signal=_stress_adjustment_signal(stress_resilience_adjustment),
            interpretation=stress_interpretation,
        ),
        _row(
            metric="Defense Risk Valuation Adjustment",
            value=defense_valuation_adjustment,
            value_type="Percent",
            signal="Risk Adjustment",
            interpretation="Valuation adjustment from the defense risk engine.",
        ),
        _row(
            metric="News Valuation Adjustment",
            value=news_valuation_adjustment,
            value_type="Percent",
            signal="News Adjustment",
            interpretation="Optional valuation adjustment from the news intelligence model.",
        ),
        _row(
            metric="Risk-Adjusted Expected Return",
            value=risk_adjusted_expected_return,
            value_type="Percent",
            signal=final_return_signal,
            interpretation=(
                "Final expected return after intelligence score, stress resilience, defense risk, "
                "and news-intelligence adjustments."
            ),
        ),
        _row(
            metric="Confidence Score",
            value=confidence_score,
            value_type="Score",
            signal=confidence_label,
            interpretation=(
                "Confidence score based on the intelligence score plus defense-risk "
                "and news-confidence adjustments."
            ),
        ),
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "expected_return_summary.csv", output_dir=TABLES_DIR)

    snapshot = pd.DataFrame(
        [
            {
                "Current Price": current_price,
                "Bear Fair Value": bear_value,
                "Base Fair Value": base_value,
                "Bull Fair Value": bull_value,
                "Bear Probability": probabilities["Bear"],
                "Base Probability": probabilities["Base"],
                "Bull Probability": probabilities["Bull"],
                "Probability Weighted Fair Value": probability_weighted_value,
                "Monte Carlo Median Fair Value": monte_carlo_median,
                "Raw Expected Return": raw_expected_return,
                "Monte Carlo Expected Return": monte_carlo_expected_return,
                "Intelligence Score": intelligence_score,
                "Intelligence Score Haircut": intelligence_haircut,
                "Stress Resilience Score": stress_resilience_score,
                "Stress Resilience Adjustment": stress_resilience_adjustment,
                "Average Stress Downside": average_stress_downside,
                "Worst Stress Case": worst_stress_case,
                "Defense Risk Valuation Adjustment": defense_valuation_adjustment,
                "News Valuation Adjustment": news_valuation_adjustment,
                "Risk Adjusted Expected Return": risk_adjusted_expected_return,
                "Confidence Score": confidence_score,
                "Return Signal": final_return_signal,
                "Confidence Label": confidence_label,
            }
        ]
    )

    _safe_to_csv(
        snapshot,
        "expected_return_snapshot.csv",
        output_dir=PROCESSED_DIR,
    )

    return df


def _bridge_row(
    step: str,
    value,
    value_type: str,
    signal: str,
    explanation: str,
) -> dict:
    """Create a standardized expected-return bridge row."""

    return {
        "Step": step,
        "Value": value,
        "Value Type": value_type,
        "Signal": signal,
        "Explanation": explanation,
    }


def build_expected_return_bridge(expected_return_df: pd.DataFrame) -> pd.DataFrame:
    """Create a compact expected-return bridge table for the report."""

    if expected_return_df is None or expected_return_df.empty:
        df = pd.DataFrame(
            [
                _bridge_row(
                    step="Expected Return Bridge",
                    value=np.nan,
                    value_type="Unavailable",
                    signal="Unavailable",
                    explanation="Expected return table was unavailable.",
                )
            ]
        )
        _safe_to_csv(df, "expected_return_bridge.csv")
        return df

    metric_map = {
        row["Metric"]: row["Value"]
        for _, row in expected_return_df.iterrows()
    }

    signal_map = {
        row["Metric"]: row.get("Signal", "")
        for _, row in expected_return_df.iterrows()
    }

    value_type_map = {
        row["Metric"]: row.get("Value Type", "")
        for _, row in expected_return_df.iterrows()
    }

    rows = [
        _bridge_row(
            step="1. Current Price",
            value=metric_map.get("Current Share Price"),
            value_type=value_type_map.get("Current Share Price", "Money"),
            signal=signal_map.get("Current Share Price", "Reference"),
            explanation="Current market reference point.",
        ),
        _bridge_row(
            step="2. Probability-Weighted Fair Value",
            value=metric_map.get("Probability-Weighted Fair Value"),
            value_type=value_type_map.get("Probability-Weighted Fair Value", "Money"),
            signal=signal_map.get("Probability-Weighted Fair Value", ""),
            explanation=(
                "Blends bear, base, and bull DCF fair values using config-defined scenario probabilities."
            ),
        ),
        _bridge_row(
            step="3. Raw Expected Return",
            value=metric_map.get("Raw Expected Return"),
            value_type=value_type_map.get("Raw Expected Return", "Percent"),
            signal=signal_map.get("Raw Expected Return", ""),
            explanation="Fair value upside/downside before risk adjustments.",
        ),
        _bridge_row(
            step="4. Intelligence Score Haircut",
            value=metric_map.get("Intelligence Score Haircut"),
            value_type=value_type_map.get("Intelligence Score Haircut", "Percent"),
            signal=signal_map.get("Intelligence Score Haircut", ""),
            explanation=(
                "Risk haircut from the market, valuation, business-quality, defense-technology, "
                "and stress-aware intelligence score."
            ),
        ),
        _bridge_row(
            step="5. Stress Resilience Adjustment",
            value=metric_map.get("Stress Resilience Adjustment"),
            value_type=value_type_map.get("Stress Resilience Adjustment", "Percent"),
            signal=signal_map.get("Stress Resilience Adjustment", ""),
            explanation=(
                "Expected-return haircut based on downside sensitivity across defense-contractor stress cases."
            ),
        ),
        _bridge_row(
            step="6. Defense Risk Adjustment",
            value=metric_map.get("Defense Risk Valuation Adjustment"),
            value_type=value_type_map.get("Defense Risk Valuation Adjustment", "Percent"),
            signal=signal_map.get("Defense Risk Valuation Adjustment", ""),
            explanation="Adjustment from the defense-specific risk engine.",
        ),
        _bridge_row(
            step="7. News Valuation Adjustment",
            value=metric_map.get("News Valuation Adjustment"),
            value_type=value_type_map.get("News Valuation Adjustment", "Percent"),
            signal=signal_map.get("News Valuation Adjustment", ""),
            explanation="Optional adjustment from live and manual news intelligence signals.",
        ),
        _bridge_row(
            step="8. Risk-Adjusted Expected Return",
            value=metric_map.get("Risk-Adjusted Expected Return"),
            value_type=value_type_map.get("Risk-Adjusted Expected Return", "Percent"),
            signal=signal_map.get("Risk-Adjusted Expected Return", ""),
            explanation="Final risk-adjusted expected return signal.",
        ),
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "expected_return_bridge.csv")

    return df


def run_expected_return_model(
    market_metrics: pd.DataFrame,
    dcf_summary: pd.DataFrame,
    mc_summary: pd.DataFrame,
    score_df: pd.DataFrame,
    defense_risk_tables: dict[str, pd.DataFrame] | None = None,
    news_intelligence_tables: dict[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full expected return model."""

    expected_return_summary = build_expected_return_table(
        market_metrics=market_metrics,
        dcf_summary=dcf_summary,
        mc_summary=mc_summary,
        score_df=score_df,
        defense_risk_tables=defense_risk_tables,
        news_intelligence_tables=news_intelligence_tables,
    )

    expected_return_bridge = build_expected_return_bridge(expected_return_summary)

    return {
        "expected_return_summary": expected_return_summary,
        "expected_return_bridge": expected_return_bridge,
    }