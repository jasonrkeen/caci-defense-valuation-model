import numpy as np
import pandas as pd

from src.config import (
    TABLES_DIR,
    BENCHMARK_NAME,
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


def _score_to_signal(score: float) -> str:
    """Convert a component score into a plain-English signal."""

    if pd.isna(score):
        return "Unavailable"

    if score >= 85:
        return "Strong"

    if score >= 75:
        return "Constructive"

    if score >= 60:
        return "Neutral / Monitor"

    if score >= 50:
        return "Cautious / Monitor"

    return "Weak"


def _overall_signal(score: float) -> str:
    """Convert the total intelligence score into a final signal."""

    if pd.isna(score):
        return "Unavailable"

    if score >= 85:
        return "Strong / Attractive"

    if score >= 75:
        return "Constructive / Attractive"

    if score >= 60:
        return "Neutral / Monitor"

    if score >= 50:
        return "Cautious / Monitor"

    return "Cautious"


def _get_metric_value(
    metrics_df: pd.DataFrame,
    metric_name: str,
    default=np.nan,
) -> float:
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


def _get_dcf_fair_value(
    dcf_summary: pd.DataFrame,
    scenario: str = "Base",
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


def _get_stress_test_score(default=np.nan) -> float:
    """Read the Stress Resilience Score from stress_test_score.csv."""

    stress_score = _read_optional_table("stress_test_score.csv")

    if stress_score.empty:
        return default

    return _get_metric_value(
        stress_score,
        metric_name="Stress Resilience Score",
        default=default,
    )


def _get_average_stress_downside(default=np.nan) -> float:
    """Read average stress downside from stress_test_score.csv."""

    stress_score = _read_optional_table("stress_test_score.csv")

    if stress_score.empty:
        return default

    return _get_metric_value(
        stress_score,
        metric_name="Average Stress Downside",
        default=default,
    )


def _get_worst_stress_case(default: str = "Unavailable") -> str:
    """Read worst stress case from stress_test_score.csv."""

    stress_score = _read_optional_table("stress_test_score.csv")

    if stress_score.empty:
        return default

    return _get_metric_text(
        stress_score,
        metric_name="Worst Stress Case",
        default=default,
    )


def _score_market_momentum(sharpe_ratio: float) -> float:
    """Score market momentum using Sharpe ratio."""

    if pd.isna(sharpe_ratio):
        return 50.0

    # Maps roughly:
    # -0.5 Sharpe -> 0
    #  1.0 Sharpe -> 100
    return max(min((sharpe_ratio + 0.5) / 1.5 * 100, 100), 0)


def _score_beta_stability(beta: float) -> float:
    """Score beta stability relative to a market beta of 1.0."""

    if pd.isna(beta):
        return 50.0

    return max(min((1 - abs(beta - 1.0)) * 100, 100), 0)


def _score_drawdown_resilience(drawdown: float) -> float:
    """Score drawdown resilience.

    Maximum drawdown should usually be negative.
    Example:
    -20% drawdown -> 80 score
    -40% drawdown -> 60 score
    """

    if pd.isna(drawdown):
        return 50.0

    return max(min((1 + drawdown) * 100, 100), 0)


def _score_valuation_support(
    current_price: float,
    base_fair_value: float,
) -> float:
    """Score valuation support based on base-case upside/downside."""

    if (
        pd.isna(current_price)
        or pd.isna(base_fair_value)
        or current_price <= 0
        or base_fair_value <= 0
    ):
        return 50.0

    upside = base_fair_value / current_price - 1

    if upside >= 0.40:
        return 90.0

    if upside >= 0.25:
        return 82.0

    if upside >= 0.10:
        return 72.0

    if upside >= -0.05:
        return 60.0

    if upside >= -0.15:
        return 50.0

    return 40.0


def _calculate_dcf_upside(
    current_price: float,
    base_fair_value: float,
) -> float:
    """Calculate base-case DCF upside/downside versus current price."""

    if (
        pd.isna(current_price)
        or pd.isna(base_fair_value)
        or current_price <= 0
        or base_fair_value <= 0
    ):
        return np.nan

    return base_fair_value / current_price - 1


def _score_row(
    component: str,
    score,
    weight,
    signal: str,
    driver_value,
    driver_value_type: str,
    interpretation: str,
) -> dict:
    """Create a standardized intelligence score row."""

    return {
        "Component": component,
        "Score": score,
        "Score Type": "Score",
        "Weight": weight,
        "Weight Type": "Percent",
        "Weighted Contribution": (
            score * weight
            if not pd.isna(score) and not pd.isna(weight)
            else np.nan
        ),
        "Weighted Contribution Type": "Score",
        "Signal": signal,
        "Driver Value": driver_value,
        "Driver Value Type": driver_value_type,
        "Interpretation": interpretation,
    }


def _build_score_rows(
    market_metrics: pd.DataFrame,
    dcf_summary: pd.DataFrame,
) -> list[dict]:
    """Build weighted intelligence score rows."""

    current_price = _get_metric_value(market_metrics, "Latest Price")
    sharpe = _get_metric_value(market_metrics, "Sharpe Ratio")
    beta = _get_metric_value(market_metrics, f"Beta vs {BENCHMARK_NAME}")
    drawdown = _get_metric_value(market_metrics, "Maximum Drawdown")
    one_year_return = _get_metric_value(market_metrics, "One-Year Return")

    base_fair_value = _get_dcf_fair_value(dcf_summary, "Base")
    dcf_upside = _calculate_dcf_upside(
        current_price=current_price,
        base_fair_value=base_fair_value,
    )

    stress_resilience_score = _get_stress_test_score(default=np.nan)
    average_stress_downside = _get_average_stress_downside(default=np.nan)
    worst_stress_case = _get_worst_stress_case(default="Unavailable")

    market_score = _score_market_momentum(sharpe)
    beta_score = _score_beta_stability(beta)
    drawdown_score = _score_drawdown_resilience(drawdown)
    valuation_score = _score_valuation_support(current_price, base_fair_value)

    if pd.isna(stress_resilience_score):
        stress_resilience_score = 50.0
        stress_driver_value = np.nan
        stress_interpretation = (
            "Stress-test score unavailable; neutral fallback used so the final score remains stable."
        )
    else:
        stress_driver_value = stress_resilience_score
        stress_interpretation = (
            f"Scores valuation resilience across downside stress scenarios. "
            f"Average stress downside is "
            f"{average_stress_downside:.1%} and the worst case is {worst_stress_case}."
            if not pd.isna(average_stress_downside)
            else (
                "Scores valuation resilience across downside stress scenarios. "
                f"The worst case is {worst_stress_case}."
            )
        )

    # Qualitative scores are currently model assumptions.
    # These remain explicit so the report can explain the thesis clearly.
    business_quality_score = 82.0
    defense_technology_score = 86.0

    rows = [
        _score_row(
            component="Market Momentum / Sharpe",
            score=market_score,
            weight=0.12,
            signal=_score_to_signal(market_score),
            driver_value=sharpe,
            driver_value_type="Plain Number",
            interpretation="Scores market risk-adjusted performance using Sharpe ratio.",
        ),
        _score_row(
            component="Beta Stability",
            score=beta_score,
            weight=0.10,
            signal=_score_to_signal(beta_score),
            driver_value=beta,
            driver_value_type="Plain Number",
            interpretation=f"Scores volatility sensitivity relative to {BENCHMARK_NAME} beta.",
        ),
        _score_row(
            component="Drawdown Resilience",
            score=drawdown_score,
            weight=0.08,
            signal=_score_to_signal(drawdown_score),
            driver_value=drawdown,
            driver_value_type="Percent",
            interpretation="Scores downside resilience using maximum drawdown.",
        ),
        _score_row(
            component="DCF Valuation Support",
            score=valuation_score,
            weight=0.23,
            signal=_score_to_signal(valuation_score),
            driver_value=dcf_upside,
            driver_value_type="Percent",
            interpretation="Scores base-case DCF upside/downside relative to current market price.",
        ),
        _score_row(
            component="Business Quality",
            score=business_quality_score,
            weight=0.17,
            signal=_score_to_signal(business_quality_score),
            driver_value=business_quality_score,
            driver_value_type="Score",
            interpretation=(
                "Reflects mission-critical demand, revenue visibility, cash flow quality, "
                "and customer concentration considerations."
            ),
        ),
        _score_row(
            component="Defense Technology Exposure",
            score=defense_technology_score,
            weight=0.15,
            signal=_score_to_signal(defense_technology_score),
            driver_value=defense_technology_score,
            driver_value_type="Score",
            interpretation=(
                "Reflects CACI's cyber, intelligence, mission technology, modernization, "
                "and national-security exposure."
            ),
        ),
        _score_row(
            component="Stress Resilience",
            score=stress_resilience_score,
            weight=0.15,
            signal=_score_to_signal(stress_resilience_score),
            driver_value=stress_driver_value,
            driver_value_type="Score",
            interpretation=stress_interpretation,
        ),
    ]

    # Add reference-only row if available, without affecting weighted score.
    if not pd.isna(one_year_return):
        rows.append(
            _score_row(
                component="One-Year Return Reference",
                score=np.nan,
                weight=0.00,
                signal="Reference",
                driver_value=one_year_return,
                driver_value_type="Percent",
                interpretation=(
                    "Reference market performance metric; not directly weighted in the score."
                ),
            )
        )

    return rows


def calculate_intelligence_score(
    market_metrics: pd.DataFrame,
    dcf_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the Defense Technology Intelligence Score.

    This score combines market momentum, beta stability, drawdown resilience,
    valuation support, business quality, defense technology exposure, and
    stress-test resilience.
    """

    rows = _build_score_rows(
        market_metrics=market_metrics,
        dcf_summary=dcf_summary,
    )

    df = pd.DataFrame(rows)

    if df.empty:
        output = pd.DataFrame(
            [
                {
                    "Component": "Overall Score",
                    "Score": 0.0,
                    "Score Type": "Score",
                    "Weight": 1.0,
                    "Weight Type": "Percent",
                    "Weighted Contribution": 0.0,
                    "Weighted Contribution Type": "Score",
                    "Signal": "Unavailable",
                    "Driver Value": np.nan,
                    "Driver Value Type": "Score",
                    "Interpretation": "No score components were available.",
                }
            ]
        )

        _safe_to_csv(output, "defense_intelligence_score.csv")
        return output

    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df["Weight"] = pd.to_numeric(df["Weight"], errors="coerce")

    weighted_components = df[
        df["Score"].notna()
        & df["Weight"].notna()
        & (df["Weight"] > 0)
    ].copy()

    total_weight = weighted_components["Weight"].sum()

    if total_weight > 0:
        weighted_components["Normalized Weight"] = (
            weighted_components["Weight"] / total_weight
        )
        total_score = float(
            (
                weighted_components["Score"]
                * weighted_components["Normalized Weight"]
            ).sum()
        )
    else:
        total_score = 0.0

    df["Weighted Contribution"] = df["Score"] * df["Weight"]

    total_score = round(total_score, 1)
    signal = _overall_signal(total_score)

    overall_row = pd.DataFrame(
        [
            {
                "Component": "Overall Score",
                "Score": total_score,
                "Score Type": "Score",
                "Weight": 1.00,
                "Weight Type": "Percent",
                "Weighted Contribution": total_score,
                "Weighted Contribution Type": "Score",
                "Signal": signal,
                "Driver Value": total_score,
                "Driver Value Type": "Score",
                "Interpretation": (
                    "Weighted Defense Technology Intelligence Score combining market, risk, "
                    "valuation, business quality, defense technology, and stress resilience signals."
                ),
            }
        ]
    )

    output = pd.concat([df, overall_row], ignore_index=True)

    output = output[
        [
            "Component",
            "Score",
            "Score Type",
            "Weight",
            "Weight Type",
            "Weighted Contribution",
            "Weighted Contribution Type",
            "Signal",
            "Driver Value",
            "Driver Value Type",
            "Interpretation",
        ]
    ]

    _safe_to_csv(output, "defense_intelligence_score.csv")

    return output