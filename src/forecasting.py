import numpy as np
import pandas as pd

from src.config import (
    TICKER,
    TABLES_DIR,
    BASE_REVENUE_M,
    SCENARIOS,
    RANDOM_SEED,
    TRADING_DAYS,
    PRICE_FORECAST_DAYS,
    PRICE_FORECAST_RUNS,
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


def _forecast_return_signal(value: float) -> str:
    """Convert forecast return into a plain-English signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.25:
        return "Strong Upside"
    if value >= 0.10:
        return "Constructive"
    if value >= -0.05:
        return "Neutral / Monitor"
    if value >= -0.15:
        return "Cautious"
    return "Negative"


def _price_signal(forecast_price: float, latest_price: float) -> str:
    """Convert forecast price versus current price into a signal."""

    if pd.isna(forecast_price) or pd.isna(latest_price) or latest_price <= 0:
        return "Unavailable"

    forecast_return = forecast_price / latest_price - 1
    return _forecast_return_signal(forecast_return)


def _summary_row(
    metric: str,
    value,
    value_type: str,
    signal: str,
    interpretation: str,
) -> dict:
    """Create a standardized Metric / Value summary row."""

    return {
        "Metric": metric,
        "Value": value,
        "Value Type": value_type,
        "Signal": signal,
        "Interpretation": interpretation,
    }


def build_dcf_forecast_table(dcf_summary: pd.DataFrame) -> pd.DataFrame:
    """Create a compact DCF forecast summary from the scenario model."""

    if dcf_summary is None or dcf_summary.empty:
        df = pd.DataFrame(
            [
                {
                    "Scenario": "Unavailable",
                    "Revenue CAGR": np.nan,
                    "Revenue CAGR Type": "Percent",
                    "Year 5 Revenue": np.nan,
                    "Year 5 Revenue Type": "Money Millions",
                    "Year 5 EBITDA": np.nan,
                    "Year 5 EBITDA Type": "Money Millions",
                    "Year 5 FCF": np.nan,
                    "Year 5 FCF Type": "Money Millions",
                    "Fair Value Per Share": np.nan,
                    "Fair Value Per Share Type": "Money",
                    "Implied EV/EBITDA": np.nan,
                    "Implied EV/EBITDA Type": "Plain Number",
                    "Terminal Value % of EV": np.nan,
                    "Terminal Value % of EV Type": "Percent",
                    "Signal": "Unavailable",
                    "Interpretation": "DCF summary was unavailable.",
                }
            ]
        )
        _safe_to_csv(df, "dcf_forecast_summary.csv")
        return df

    rows = []

    for _, row in dcf_summary.iterrows():
        fair_value = _safe_float(row.get("Fair Value Per Share"))

        if pd.isna(fair_value):
            signal = "Unavailable"
            interpretation = "Fair value estimate unavailable."
        else:
            signal = "Scenario Output"
            interpretation = "DCF-derived scenario forecast output."

        rows.append(
            {
                "Scenario": row.get("Scenario"),
                "Revenue CAGR": row.get("Revenue CAGR"),
                "Revenue CAGR Type": "Percent",
                "Year 5 Revenue": row.get("Year 5 Revenue"),
                "Year 5 Revenue Type": "Money Millions",
                "Year 5 EBITDA": row.get("Year 5 EBITDA"),
                "Year 5 EBITDA Type": "Money Millions",
                "Year 5 FCF": row.get("Year 5 FCF"),
                "Year 5 FCF Type": "Money Millions",
                "Fair Value Per Share": fair_value,
                "Fair Value Per Share Type": "Money",
                "Implied EV/EBITDA": row.get("Implied EV/EBITDA"),
                "Implied EV/EBITDA Type": "Plain Number",
                "Terminal Value % of EV": row.get("Terminal Value % of EV"),
                "Terminal Value % of EV Type": "Percent",
                "Signal": signal,
                "Interpretation": interpretation,
            }
        )

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "dcf_forecast_summary.csv")

    return df


def build_revenue_forecast_detail() -> pd.DataFrame:
    """Build a year-by-year revenue forecast for each DCF scenario."""

    rows = []

    for scenario, assumptions in SCENARIOS.items():
        revenue = BASE_REVENUE_M
        revenue_growth_rates = assumptions.get("revenue_growth", [])
        ebitda_margin = assumptions.get("ebitda_margin", np.nan)

        for year, growth in enumerate(revenue_growth_rates, start=1):
            revenue = revenue * (1 + growth)

            forecast_ebitda = (
                revenue * ebitda_margin
                if not pd.isna(ebitda_margin)
                else np.nan
            )

            rows.append(
                {
                    "Scenario": scenario,
                    "Forecast Year": year,
                    "Forecast Year Type": "Plain Number",
                    "Revenue Growth": growth,
                    "Revenue Growth Type": "Percent",
                    "Forecast Revenue": revenue,
                    "Forecast Revenue Type": "Money Millions",
                    "EBITDA Margin": ebitda_margin,
                    "EBITDA Margin Type": "Percent",
                    "Forecast EBITDA": forecast_ebitda,
                    "Forecast EBITDA Type": "Money Millions",
                    "Interpretation": (
                        f"Year {year} revenue forecast for the {scenario} scenario."
                    ),
                }
            )

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "revenue_forecast_detail.csv")

    return df


def _validate_price_data(prices: pd.DataFrame) -> None:
    """Validate that price data contains the required ticker column."""

    if prices is None or prices.empty:
        raise ValueError("Price data is empty. Cannot run price forecast.")

    if TICKER not in prices.columns:
        raise ValueError(f"Price data is missing required ticker column: {TICKER}")

    clean_prices = prices[TICKER].dropna()

    if clean_prices.empty:
        raise ValueError(f"No usable price data found for {TICKER}.")

    if len(clean_prices) < 30:
        raise ValueError(
            f"Not enough price observations for {TICKER}. "
            "At least 30 observations are recommended."
        )


def run_price_forecast(
    prices: pd.DataFrame,
    forecast_days: int = PRICE_FORECAST_DAYS,
    simulations: int = PRICE_FORECAST_RUNS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a bootstrapped stock price forecast using historical daily returns."""

    _validate_price_data(prices)

    rng = np.random.default_rng(RANDOM_SEED)

    clean_prices = prices[TICKER].dropna()
    historical_returns = clean_prices.pct_change(fill_method=None).dropna()

    if historical_returns.empty:
        raise ValueError("Historical returns are empty. Cannot run price forecast.")

    latest_price = float(clean_prices.iloc[-1])

    forecast_paths = []

    for sim in range(simulations):
        sampled_returns = rng.choice(
            historical_returns.values,
            size=forecast_days,
            replace=True,
        )

        price_path = latest_price * np.cumprod(1 + sampled_returns)
        final_price = float(price_path[-1])
        forecast_return = final_price / latest_price - 1

        forecast_paths.append(
            {
                "Simulation": sim + 1,
                "Simulation Type": "Plain Number",
                "Final Forecast Price": final_price,
                "Final Forecast Price Type": "Money",
                "Forecast Return": forecast_return,
                "Forecast Return Type": "Percent",
                "Minimum Forecast Price": float(price_path.min()),
                "Minimum Forecast Price Type": "Money",
                "Maximum Forecast Price": float(price_path.max()),
                "Maximum Forecast Price Type": "Money",
            }
        )

    simulations_df = pd.DataFrame(forecast_paths)
    _safe_to_csv(simulations_df, "price_forecast_simulations.csv")

    mean_forecast_price = simulations_df["Final Forecast Price"].mean()
    median_forecast_price = simulations_df["Final Forecast Price"].median()
    p10_forecast_price = simulations_df["Final Forecast Price"].quantile(0.10)
    p25_forecast_price = simulations_df["Final Forecast Price"].quantile(0.25)
    p75_forecast_price = simulations_df["Final Forecast Price"].quantile(0.75)
    p90_forecast_price = simulations_df["Final Forecast Price"].quantile(0.90)

    mean_forecast_return = simulations_df["Forecast Return"].mean()
    median_forecast_return = simulations_df["Forecast Return"].median()
    p10_forecast_return = simulations_df["Forecast Return"].quantile(0.10)
    p90_forecast_return = simulations_df["Forecast Return"].quantile(0.90)

    summary = pd.DataFrame(
        [
            _summary_row(
                metric="Latest Price",
                value=latest_price,
                value_type="Money",
                signal="Reference",
                interpretation="Latest market price used as the forecast starting point.",
            ),
            _summary_row(
                metric="Mean Forecast Price",
                value=mean_forecast_price,
                value_type="Money",
                signal=_price_signal(mean_forecast_price, latest_price),
                interpretation="Average final simulated price across all forecast paths.",
            ),
            _summary_row(
                metric="Median Forecast Price",
                value=median_forecast_price,
                value_type="Money",
                signal=_price_signal(median_forecast_price, latest_price),
                interpretation="Median final simulated price across all forecast paths.",
            ),
            _summary_row(
                metric="P10 Forecast Price",
                value=p10_forecast_price,
                value_type="Money",
                signal=_price_signal(p10_forecast_price, latest_price),
                interpretation="Tenth-percentile downside forecast price.",
            ),
            _summary_row(
                metric="P25 Forecast Price",
                value=p25_forecast_price,
                value_type="Money",
                signal=_price_signal(p25_forecast_price, latest_price),
                interpretation="Twenty-fifth-percentile forecast price.",
            ),
            _summary_row(
                metric="P75 Forecast Price",
                value=p75_forecast_price,
                value_type="Money",
                signal=_price_signal(p75_forecast_price, latest_price),
                interpretation="Seventy-fifth-percentile forecast price.",
            ),
            _summary_row(
                metric="P90 Forecast Price",
                value=p90_forecast_price,
                value_type="Money",
                signal=_price_signal(p90_forecast_price, latest_price),
                interpretation="Ninetieth-percentile upside forecast price.",
            ),
            _summary_row(
                metric="Mean Forecast Return",
                value=mean_forecast_return,
                value_type="Percent",
                signal=_forecast_return_signal(mean_forecast_return),
                interpretation="Average simulated return over the forecast horizon.",
            ),
            _summary_row(
                metric="Median Forecast Return",
                value=median_forecast_return,
                value_type="Percent",
                signal=_forecast_return_signal(median_forecast_return),
                interpretation="Median simulated return over the forecast horizon.",
            ),
            _summary_row(
                metric="P10 Forecast Return",
                value=p10_forecast_return,
                value_type="Percent",
                signal=_forecast_return_signal(p10_forecast_return),
                interpretation="Tenth-percentile downside forecast return.",
            ),
            _summary_row(
                metric="P90 Forecast Return",
                value=p90_forecast_return,
                value_type="Percent",
                signal=_forecast_return_signal(p90_forecast_return),
                interpretation="Ninetieth-percentile upside forecast return.",
            ),
            _summary_row(
                metric="Forecast Horizon Days",
                value=forecast_days,
                value_type="Plain Number",
                signal="Reference",
                interpretation="Number of trading days simulated in each price forecast path.",
            ),
            _summary_row(
                metric="Forecast Simulations",
                value=simulations,
                value_type="Plain Number",
                signal="Reference",
                interpretation="Number of bootstrapped simulation paths used in the forecast.",
            ),
        ]
    )

    _safe_to_csv(summary, "price_forecast_summary.csv")

    return summary, simulations_df


def run_forecasting_model(
    prices: pd.DataFrame,
    dcf_summary: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Run the full forecasting package."""

    dcf_forecast_summary = build_dcf_forecast_table(dcf_summary)
    revenue_forecast_detail = build_revenue_forecast_detail()
    price_forecast_summary, price_forecast_simulations = run_price_forecast(
        prices=prices,
        forecast_days=PRICE_FORECAST_DAYS,
        simulations=PRICE_FORECAST_RUNS,
    )

    return {
        "dcf_forecast_summary": dcf_forecast_summary,
        "revenue_forecast_detail": revenue_forecast_detail,
        "price_forecast_summary": price_forecast_summary,
        "price_forecast_simulations": price_forecast_simulations,
    }