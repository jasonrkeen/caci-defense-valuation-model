import pandas as pd

from src.market_data import (
    download_prices,
    calculate_market_metrics,
    get_live_risk_free_rate,
)
from src.fundamentals import build_complete_fundamental_package
from src.valuation import run_all_dcf_scenarios
from src.peer_analysis import collect_peer_data, build_peer_ranking_table
from src.monte_carlo import run_monte_carlo
from src.intelligence_score import calculate_intelligence_score
from src.analyst_estimates import build_complete_analyst_package
from src.visualization import create_all_charts
from src.report_generator import build_report
from src.source_notes import build_source_driven_tables
from src.defense_risk_engine import run_defense_risk_engine
from src.intelligence_spending import run_intelligence_spending_model
from src.government_spending import run_government_spending_model
from src.defense_macro import run_defense_macro_model
from src.news_intelligence import run_news_intelligence_model
from src.forecasting import run_forecasting_model
from src.expected_return import run_expected_return_model
from src.change_detection import run_change_detection
from src.garch_var import run_garch_var_model
from src.factor_model import run_factor_model
from src.stress_testing import run_stress_tests
from src.config import TABLES_DIR


def _safe_read_csv(file_name: str) -> pd.DataFrame:
    """Safely read a CSV from outputs/tables."""

    path = TABLES_DIR / file_name

    if not path.exists():
        print(f"Warning: {file_name} was not found.")
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read {file_name}: {exc}")
        return pd.DataFrame()


def _safe_metric_value(
    df: pd.DataFrame,
    metric_name: str,
    metric_column: str = "Metric",
    value_column: str = "Value",
    default=None,
):
    """Safely extract a value from a Metric / Value table."""

    if df is None or df.empty:
        return default

    if metric_column not in df.columns or value_column not in df.columns:
        return default

    match = df.loc[df[metric_column] == metric_name, value_column]

    if match.empty:
        return default

    return match.iloc[0]


def _safe_component_value(
    df: pd.DataFrame,
    component_name: str,
    value_column: str,
    default=None,
):
    """Safely extract a value from the intelligence score table."""

    if df is None or df.empty:
        return default

    if "Component" not in df.columns or value_column not in df.columns:
        return default

    match = df.loc[df["Component"] == component_name, value_column]

    if match.empty:
        return default

    return match.iloc[0]


def _safe_base_case_value(dcf_summary: pd.DataFrame, default=None):
    """Safely extract the base-case DCF fair value per share."""

    if dcf_summary is None or dcf_summary.empty:
        return default

    required_columns = {"Scenario", "Fair Value Per Share"}

    if not required_columns.issubset(set(dcf_summary.columns)):
        return default

    match = dcf_summary.loc[
        dcf_summary["Scenario"] == "Base",
        "Fair Value Per Share",
    ]

    if match.empty:
        return default

    return match.iloc[0]


def main():
    print("Starting CACI Defense Technology Valuation & Intelligence Pipeline...")

    # ----------------------------
    # Market data and risk metrics
    # ----------------------------

    risk_free_rate = get_live_risk_free_rate()
    print(f"Risk-free rate input: {risk_free_rate:.2%}")

    prices = download_prices()

    market_metrics = calculate_market_metrics(
        prices,
        risk_free_rate=risk_free_rate,
    )

    print("Market data and risk metrics completed.")

    # ----------------------------
    # GARCH / EWMA volatility and VaR model
    # ----------------------------

    garch_var_tables = run_garch_var_model(prices)
    garch_summary = garch_var_tables.get("garch_var_summary", pd.DataFrame())

    model_used = _safe_metric_value(
        garch_summary,
        metric_name="Volatility Model Used",
        default=None,
    )

    if model_used:
        print(f"GARCH / VaR model completed. Volatility model used: {model_used}")
    else:
        print("GARCH / VaR model completed.")

    # ----------------------------
    # Defense equity factor exposure model
    # ----------------------------

    factor_tables = run_factor_model()
    factor_summary = factor_tables.get("factor_exposure_summary", pd.DataFrame())

    factor_r_squared = _safe_metric_value(
        factor_summary,
        metric_name="Factor Model R-Squared",
        default=None,
    )

    if factor_r_squared is not None:
        try:
            print(
                f"Defense equity factor model completed. "
                f"R-squared: {float(factor_r_squared):.2%}"
            )
        except Exception:
            print("Defense equity factor model completed.")
    else:
        print("Defense equity factor model completed.")

    # ----------------------------
    # Company fundamentals and source tables
    # ----------------------------

    fundamental_tables = build_complete_fundamental_package()
    fundamental_summary = fundamental_tables.get(
        "fundamental_summary",
        pd.DataFrame(),
    )
    print("Fundamental tables created.")

    source_tables = build_source_driven_tables()
    print("Source-driven tables created.")

    # ----------------------------
    # Defense, government, macro, and news intelligence layers
    # ----------------------------

    defense_risk_tables = run_defense_risk_engine()
    print("Defense risk engine completed.")

    intelligence_spending_tables = run_intelligence_spending_model()
    print("Intelligence spending model completed.")

    government_spending_tables = run_government_spending_model()
    print("Government spending model completed.")

    defense_macro_tables = run_defense_macro_model()
    print("Defense macro model completed.")

    news_intelligence_tables = run_news_intelligence_model()

    live_news_count = len(
        news_intelligence_tables.get("live_news_inputs", pd.DataFrame())
    )
    classified_news_count = len(
        news_intelligence_tables.get("classified_news", pd.DataFrame())
    )

    print(
        f"News intelligence model completed "
        f"({live_news_count} live RSS headlines, "
        f"{classified_news_count} classified headlines)."
    )

    # ----------------------------
    # Analyst estimate package
    # ----------------------------

    _, analyst_summary, revenue_growth_hint = build_complete_analyst_package()

    if revenue_growth_hint is not None:
        print(f"Yahoo Finance next-year revenue growth hint: {revenue_growth_hint:.2%}")
    else:
        print("Yahoo Finance revenue growth hint unavailable; using config.py scenario assumptions.")

    # ----------------------------
    # DCF valuation
    # ----------------------------

    dcf_summary = run_all_dcf_scenarios()
    print("DCF valuation scenarios completed.")

    # ----------------------------
    # Defense contractor stress testing
    # ----------------------------

    stress_test_tables = run_stress_tests()
    stress_test_score = stress_test_tables.get("stress_test_score", pd.DataFrame())

    stress_resilience_score = _safe_metric_value(
        stress_test_score,
        metric_name="Stress Resilience Score",
        default=None,
    )

    if stress_resilience_score is not None:
        try:
            print(
                f"Defense contractor stress testing completed. "
                f"Stress resilience score: {float(stress_resilience_score):.1f}/100"
            )
        except Exception:
            print("Defense contractor stress testing completed.")
    else:
        print("Defense contractor stress testing completed.")

    # ----------------------------
    # Forecasting model
    # ----------------------------

    forecast_tables = run_forecasting_model(
        prices=prices,
        dcf_summary=dcf_summary,
    )
    print("Forecasting model completed.")

    # ----------------------------
    # Peer benchmarking
    # ----------------------------

    peer_df = collect_peer_data()

    try:
        build_peer_ranking_table(peer_df)
    except Exception as exc:
        print(f"Warning: peer ranking table could not be rebuilt: {exc}")

    print("Peer benchmarking completed.")

    # ----------------------------
    # Monte Carlo valuation
    # ----------------------------

    current_price = _safe_metric_value(
        market_metrics,
        metric_name="Latest Price",
        default=None,
    )

    if current_price is None:
        print("Warning: current price unavailable; Monte Carlo expected-return rows may be unavailable.")

    mc_values = run_monte_carlo(current_price=current_price)
    mc_summary = _safe_read_csv("monte_carlo_summary.csv")

    print("Monte Carlo valuation completed.")

    # ----------------------------
    # Intelligence score
    # ----------------------------

    score_df = calculate_intelligence_score(
        market_metrics,
        dcf_summary,
    )
    print("Defense Technology Intelligence Score completed.")

    # ----------------------------
    # Expected return model
    # ----------------------------

    expected_return_tables = run_expected_return_model(
        market_metrics=market_metrics,
        dcf_summary=dcf_summary,
        mc_summary=mc_summary,
        score_df=score_df,
        defense_risk_tables=defense_risk_tables,
        news_intelligence_tables=news_intelligence_tables,
    )
    print("Expected return model completed.")

    # ----------------------------
    # Change detection
    # ----------------------------

    change_detection_tables = run_change_detection(
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
    print("Change detection completed.")

    # ----------------------------
    # Charts
    # ----------------------------

    chart_paths = create_all_charts(
        prices,
        dcf_summary,
        mc_values,
        score_df,
    )

    garch_chart_paths = garch_var_tables.get("garch_var_chart_paths", [])

    if garch_chart_paths:
        chart_paths.extend(garch_chart_paths)

    print("Charts created.")

    # ----------------------------
    # Final report
    # ----------------------------

    report_path = build_report(
        market_metrics=market_metrics,
        fundamental_summary=fundamental_summary,
        dcf_summary=dcf_summary,
        analyst_summary=analyst_summary,
        source_tables=source_tables,
        defense_risk_tables=defense_risk_tables,
        intelligence_spending_tables=intelligence_spending_tables,
        government_spending_tables=government_spending_tables,
        defense_macro_tables=defense_macro_tables,
        news_intelligence_tables=news_intelligence_tables,
        expected_return_tables=expected_return_tables,
        change_detection_tables=change_detection_tables,
        forecast_tables=forecast_tables,
        peer_df=peer_df,
        mc_summary=mc_summary,
        score_df=score_df,
        chart_paths=chart_paths,
    )

    # ----------------------------
    # Console summary
    # ----------------------------

    base_case = _safe_base_case_value(dcf_summary, default=None)

    overall_score = _safe_component_value(
        score_df,
        component_name="Overall Score",
        value_column="Score",
        default=None,
    )

    signal = _safe_component_value(
        score_df,
        component_name="Overall Score",
        value_column="Signal",
        default="Unavailable",
    )

    print("Pipeline complete.")

    if base_case is not None:
        print(f"Base-case DCF fair value per share: ${base_case:,.2f}")
    else:
        print("Base-case DCF fair value per share: unavailable")

    if stress_resilience_score is not None:
        try:
            print(f"Stress Resilience Score: {float(stress_resilience_score):.1f}/100")
        except Exception:
            print("Stress Resilience Score: unavailable")
    else:
        print("Stress Resilience Score: unavailable")

    if overall_score is not None:
        print(f"Defense Technology Intelligence Score: {overall_score:.1f}/100")
    else:
        print("Defense Technology Intelligence Score: unavailable")

    print(f"Final Signal: {signal}")
    print(f"PDF report created: {report_path}")


if __name__ == "__main__":
    main()