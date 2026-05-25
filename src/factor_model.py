import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    TICKER,
    START_DATE,
    TABLES_DIR,
    TRADING_DAYS,
    FACTOR_MODEL_TICKERS,
    FACTOR_MODEL_DISPLAY_ORDER,
)


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
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


def _factor_signal(factor_name: str, beta_value: float) -> str:
    """Convert factor beta into a plain-English signal."""

    if pd.isna(beta_value):
        return "Unavailable"

    name = str(factor_name).lower()

    if "s&p" in name or "market" in name:
        if beta_value >= 1.15:
            return "High Market Sensitivity"
        if beta_value >= 0.80:
            return "Market-Like Sensitivity"
        if beta_value >= 0.40:
            return "Moderate Market Sensitivity"
        return "Low Market Sensitivity"

    if "aerospace" in name or "defense" in name:
        if beta_value >= 1.00:
            return "Strong Defense Factor Exposure"
        if beta_value >= 0.60:
            return "Constructive Defense Exposure"
        if beta_value >= 0.25:
            return "Moderate Defense Exposure"
        return "Low Defense Exposure"

    if "technology" in name:
        if beta_value >= 0.80:
            return "High Technology Sensitivity"
        if beta_value >= 0.40:
            return "Constructive Technology Link"
        if beta_value >= 0.10:
            return "Moderate Technology Link"
        return "Low Technology Link"

    if "interest" in name or "rate" in name:
        if beta_value <= -0.20:
            return "Rate-Sensitive / Inverse"
        if beta_value < 0:
            return "Mild Rate Sensitivity"
        if beta_value <= 0.20:
            return "Limited Rate Sensitivity"
        return "Positive Rate Link"

    if "volatility" in name or "risk" in name:
        if beta_value <= -0.20:
            return "Risk-Off Sensitive"
        if beta_value < 0:
            return "Mild Risk-Off Sensitivity"
        if beta_value <= 0.20:
            return "Limited Volatility Sensitivity"
        return "Positive Volatility Link"

    return "Reference"


def _correlation_signal(value: float) -> str:
    """Convert correlation into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.80:
        return "High Correlation"
    if value >= 0.60:
        return "Moderate / High Correlation"
    if value >= 0.30:
        return "Moderate Correlation"
    if value >= 0.00:
        return "Low Correlation"
    return "Negative Correlation"


def _r_squared_signal(value: float) -> str:
    """Convert regression R-squared into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.70:
        return "High Factor Explanation"
    if value >= 0.50:
        return "Moderate / High Explanation"
    if value >= 0.30:
        return "Moderate Explanation"
    return "Low Factor Explanation"


def _alpha_signal(value: float) -> str:
    """Convert annualized alpha into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.05:
        return "Positive Alpha"
    if value >= 0.00:
        return "Slight Positive Alpha"
    if value >= -0.05:
        return "Slight Negative Alpha"
    return "Negative Alpha"


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


def _extract_close_prices(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract adjusted close prices from a yfinance response."""

    if data is None or data.empty:
        raise ValueError("No factor price data returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError("Yahoo Finance response did not include close prices.")

        prices = data["Close"].copy()
    else:
        if "Close" not in data.columns:
            raise ValueError("Yahoo Finance response did not include a Close column.")

        prices = data[["Close"]].copy()

        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=tickers[0])

    available_columns = [ticker for ticker in tickers if ticker in prices.columns]

    if TICKER not in available_columns:
        raise ValueError(f"Factor model is missing required target ticker: {TICKER}")

    prices = prices[available_columns].copy()

    for column in prices.columns:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")

    prices = prices.dropna(how="all")

    if prices.empty:
        raise ValueError("Factor price table became empty after cleaning.")

    return prices


def download_factor_prices(start_date: str = START_DATE) -> pd.DataFrame:
    """Download target and factor prices."""

    tickers = [TICKER] + FACTOR_MODEL_DISPLAY_ORDER

    data = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    prices = _extract_close_prices(data, tickers)
    _safe_to_csv(prices.reset_index(), "factor_price_history.csv")

    return prices


def build_factor_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Build aligned daily factor returns.

    Equity and index factors use percentage returns.
    Treasury and VIX factors use daily changes because they are level/rate indicators.
    """

    if prices is None or prices.empty:
        raise ValueError("Price data is empty. Cannot calculate factor returns.")

    returns = pd.DataFrame(index=prices.index)

    if TICKER not in prices.columns:
        raise ValueError(f"Price data is missing target ticker: {TICKER}")

    returns["CACI Return"] = prices[TICKER].pct_change(fill_method=None)

    for ticker in FACTOR_MODEL_DISPLAY_ORDER:
        if ticker not in prices.columns:
            continue

        label = FACTOR_MODEL_TICKERS.get(ticker, ticker)

        if ticker in ["^TNX", "^VIX"]:
            returns[label] = prices[ticker].diff()
        else:
            returns[label] = prices[ticker].pct_change(fill_method=None)

    returns = returns.replace([np.inf, -np.inf], np.nan).dropna()

    if returns.empty:
        raise ValueError("Factor returns table became empty after alignment.")

    _safe_to_csv(returns.reset_index(), "factor_return_history.csv")

    return returns


def _ols_regression(y: pd.Series, x: pd.DataFrame) -> dict:
    """Run an OLS regression using numpy.

    Returns coefficients, fitted values, residuals, R-squared, adjusted R-squared,
    annualized alpha, and residual volatility.
    """

    aligned = pd.concat([y, x], axis=1).dropna()

    if aligned.empty:
        raise ValueError("No aligned observations available for factor regression.")

    y_aligned = aligned.iloc[:, 0].astype(float)
    x_aligned = aligned.iloc[:, 1:].astype(float)

    x_matrix = np.column_stack(
        [
            np.ones(len(x_aligned)),
            x_aligned.values,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(x_matrix, y_aligned.values, rcond=None)

    fitted = x_matrix @ coefficients
    residuals = y_aligned.values - fitted

    ss_resid = np.sum(residuals ** 2)
    ss_total = np.sum((y_aligned.values - y_aligned.values.mean()) ** 2)

    r_squared = 1 - ss_resid / ss_total if ss_total != 0 else np.nan

    n_obs = len(y_aligned)
    n_factors = x_aligned.shape[1]

    adjusted_r_squared = (
        1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_factors - 1)
        if n_obs > n_factors + 1 and not pd.isna(r_squared)
        else np.nan
    )

    alpha_daily = coefficients[0]
    annualized_alpha = alpha_daily * TRADING_DAYS
    residual_volatility = np.std(residuals, ddof=1) * np.sqrt(TRADING_DAYS)

    return {
        "coefficients": coefficients,
        "factor_names": list(x_aligned.columns),
        "fitted": fitted,
        "residuals": residuals,
        "r_squared": r_squared,
        "adjusted_r_squared": adjusted_r_squared,
        "annualized_alpha": annualized_alpha,
        "residual_volatility": residual_volatility,
        "observations": n_obs,
    }


def build_factor_regression_results(
    factor_returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build factor regression result tables."""

    if factor_returns is None or factor_returns.empty:
        raise ValueError("Factor returns are empty.")

    if "CACI Return" not in factor_returns.columns:
        raise ValueError("Factor returns are missing CACI Return.")

    y = factor_returns["CACI Return"]

    factor_columns = [
        column
        for column in factor_returns.columns
        if column != "CACI Return"
    ]

    if not factor_columns:
        raise ValueError("No factor columns available for regression.")

    x = factor_returns[factor_columns]

    regression = _ols_regression(y, x)

    coefficient_rows = []

    for factor_name, coefficient in zip(
        regression["factor_names"],
        regression["coefficients"][1:],
    ):
        coefficient_rows.append(
            {
                "Factor": factor_name,
                "Beta / Sensitivity": coefficient,
                "Beta / Sensitivity Type": "Plain Number",
                "Signal": _factor_signal(factor_name, coefficient),
                "Interpretation": (
                    f"CACI daily returns have an estimated sensitivity of {coefficient:.2f} "
                    f"to the {factor_name} factor in the multi-factor model."
                ),
            }
        )

    coefficients_df = pd.DataFrame(coefficient_rows)

    summary_rows = [
        _summary_row(
            metric="Annualized Alpha",
            value=regression["annualized_alpha"],
            value_type="Percent",
            signal=_alpha_signal(regression["annualized_alpha"]),
            interpretation="Annualized intercept from the multi-factor regression.",
        ),
        _summary_row(
            metric="Factor Model R-Squared",
            value=regression["r_squared"],
            value_type="Percent",
            signal=_r_squared_signal(regression["r_squared"]),
            interpretation="Share of CACI daily return variation explained by the selected factors.",
        ),
        _summary_row(
            metric="Adjusted R-Squared",
            value=regression["adjusted_r_squared"],
            value_type="Percent",
            signal=_r_squared_signal(regression["adjusted_r_squared"]),
            interpretation="R-squared adjusted for the number of factors in the model.",
        ),
        _summary_row(
            metric="Annualized Residual Volatility",
            value=regression["residual_volatility"],
            value_type="Percent",
            signal="Idiosyncratic Risk",
            interpretation="Annualized volatility not explained by the selected factor model.",
        ),
        _summary_row(
            metric="Regression Observations",
            value=regression["observations"],
            value_type="Plain Number",
            signal="Reference",
            interpretation="Number of aligned daily observations used in the factor regression.",
        ),
    ]

    summary_df = pd.DataFrame(summary_rows)

    _safe_to_csv(coefficients_df, "factor_regression_results.csv")
    _safe_to_csv(summary_df, "factor_exposure_summary.csv")

    return coefficients_df, summary_df


def build_factor_correlation_summary(factor_returns: pd.DataFrame) -> pd.DataFrame:
    """Build CACI correlation summary versus each factor."""

    if factor_returns is None or factor_returns.empty:
        return pd.DataFrame()

    if "CACI Return" not in factor_returns.columns:
        return pd.DataFrame()

    rows = []

    for column in factor_returns.columns:
        if column == "CACI Return":
            continue

        corr = factor_returns["CACI Return"].corr(factor_returns[column])

        rows.append(
            {
                "Factor": column,
                "Correlation": corr,
                "Correlation Type": "Plain Number",
                "Signal": _correlation_signal(corr),
                "Interpretation": f"Correlation between CACI daily returns and {column}.",
            }
        )

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "factor_correlation_summary.csv")

    return df


def build_factor_contribution_summary(
    coefficients_df: pd.DataFrame,
    factor_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate directional contribution for return-based factors only.

    Interest-rate and volatility factors are modeled as daily level changes,
    not percentage returns. They remain useful in the regression and correlation
    tables, but they are excluded from this contribution table because their
    average annualized moves are not directly comparable to return-based factor
    contributions.
    """

    if coefficients_df is None or coefficients_df.empty:
        return pd.DataFrame()

    if factor_returns is None or factor_returns.empty:
        return pd.DataFrame()

    return_based_factors = [
        "S&P 500",
        "Aerospace & Defense Factor",
        "Technology Factor",
    ]

    if "Factor" not in coefficients_df.columns:
        return pd.DataFrame()

    filtered_coefficients = coefficients_df[
        coefficients_df["Factor"].isin(return_based_factors)
    ].copy()

    if filtered_coefficients.empty:
        return pd.DataFrame()

    rows = []

    for _, row in filtered_coefficients.iterrows():
        factor = row.get("Factor")
        beta_value = _safe_float(row.get("Beta / Sensitivity"))

        if factor not in factor_returns.columns:
            continue

        factor_mean = factor_returns[factor].mean() * TRADING_DAYS

        estimated_contribution = (
            beta_value * factor_mean
            if not pd.isna(beta_value) and not pd.isna(factor_mean)
            else np.nan
        )

        rows.append(
            {
                "Factor": factor,
                "Average Annualized Factor Return": factor_mean,
                "Average Annualized Factor Return Type": "Percent",
                "Beta / Sensitivity": beta_value,
                "Beta / Sensitivity Type": "Plain Number",
                "Estimated Annualized Contribution": estimated_contribution,
                "Estimated Annualized Contribution Type": "Percent",
                "Signal": _factor_signal(factor, beta_value),
                "Interpretation": (
                    "Estimated directional contribution equals factor sensitivity multiplied "
                    "by the average annualized factor return. Rate and volatility factors are "
                    "excluded because they are modeled as daily level changes, not returns."
                ),
            }
        )

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "factor_contribution_summary.csv")

    return df


def run_factor_model() -> dict[str, pd.DataFrame]:
    """Run the full Defense Equity Factor Exposure Model."""

    try:
        prices = download_factor_prices(start_date=START_DATE)
        returns = build_factor_returns(prices)

        regression_results, exposure_summary = build_factor_regression_results(returns)
        correlation_summary = build_factor_correlation_summary(returns)
        contribution_summary = build_factor_contribution_summary(
            regression_results,
            returns,
        )

        return {
            "factor_regression_results": regression_results,
            "factor_exposure_summary": exposure_summary,
            "factor_correlation_summary": correlation_summary,
            "factor_contribution_summary": contribution_summary,
        }

    except Exception as exc:
        print(f"Warning: factor model failed: {exc}")

        fallback_summary = pd.DataFrame(
            [
                _summary_row(
                    metric="Defense Equity Factor Exposure Model",
                    value="Unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation=f"Factor model failed or data was unavailable: {exc}",
                )
            ]
        )

        _safe_to_csv(fallback_summary, "factor_exposure_summary.csv")

        return {
            "factor_regression_results": pd.DataFrame(),
            "factor_exposure_summary": fallback_summary,
            "factor_correlation_summary": pd.DataFrame(),
            "factor_contribution_summary": pd.DataFrame(),
        }