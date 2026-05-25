"""
GARCH / EWMA volatility and Value-at-Risk module for the CACI Defense Technology
Valuation & Intelligence Pipeline.

Primary model:
- GARCH(1,1) using the arch package when available, enabled, and successfully fitted.

Fallback model:
- EWMA volatility process if arch is unavailable, disabled, or model fitting fails.

Outputs:
- outputs/tables/garch_var_summary.csv
- outputs/tables/volatility_regime_summary.csv
- outputs/tables/var_backtest_summary.csv
- outputs/tables/garch_var_timeseries.csv
- outputs/charts/conditional_volatility_chart.png
- outputs/charts/var_backtest_chart.png
"""

from __future__ import annotations

import math
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    TICKER,
    COMPANY_NAME,
    TABLES_DIR,
    CHARTS_DIR,
    TRADING_DAYS,
    USE_ARCH_GARCH,
    GARCH_P,
    GARCH_Q,
    GARCH_DISTRIBUTION,
    EWMA_LAMBDA,
    VAR_CONFIDENCE_LEVELS,
)


try:
    from arch import arch_model

    ARCH_AVAILABLE = True
except Exception:
    arch_model = None
    ARCH_AVAILABLE = False


Z_SCORE_BY_CONFIDENCE = {
    0.95: 1.6448536269514722,
    0.99: 2.3263478740408408,
}


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Safely save a DataFrame to outputs/tables."""

    try:
        TABLES_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _safe_savefig(path) -> str | None:
    """Safely save the current matplotlib figure."""

    try:
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)
    except Exception as exc:
        print(f"Warning: could not save chart {path}: {exc}")
        plt.close()
        return None


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


def _positive_or_nan(value: float) -> float:
    """Return a positive numeric value, otherwise nan."""

    value = _safe_float(value)

    if pd.isna(value):
        return np.nan

    return max(value, 0.0)


def _get_z_score(confidence_level: float) -> float:
    """Return normal z-score for the requested confidence level."""

    confidence_level = round(float(confidence_level), 2)

    if confidence_level in Z_SCORE_BY_CONFIDENCE:
        return Z_SCORE_BY_CONFIDENCE[confidence_level]

    if confidence_level >= 0.99:
        return Z_SCORE_BY_CONFIDENCE[0.99]

    return Z_SCORE_BY_CONFIDENCE[0.95]


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


def _extract_price_series(prices: pd.DataFrame) -> pd.Series:
    """Extract the CACI price series from the broader price DataFrame."""

    if prices is None or prices.empty:
        return pd.Series(dtype=float, name=TICKER)

    if isinstance(prices.columns, pd.MultiIndex):
        if TICKER in prices.columns.get_level_values(-1):
            series = prices.xs(TICKER, axis=1, level=-1).iloc[:, 0]
            return pd.to_numeric(series, errors="coerce").dropna().rename(TICKER)

        if TICKER in prices.columns.get_level_values(0):
            series = prices.xs(TICKER, axis=1, level=0).iloc[:, 0]
            return pd.to_numeric(series, errors="coerce").dropna().rename(TICKER)

    if TICKER in prices.columns:
        return pd.to_numeric(prices[TICKER], errors="coerce").dropna().rename(TICKER)

    numeric_columns = prices.select_dtypes(include=["number"]).columns

    if len(numeric_columns) == 0:
        return pd.Series(dtype=float, name=TICKER)

    return pd.to_numeric(prices[numeric_columns[0]], errors="coerce").dropna().rename(TICKER)


def _kupiec_pof_test(exceptions: int, observations: int, alpha: float) -> dict:
    """Run Kupiec proportion-of-failures test for VaR exceptions.

    Uses the chi-square survival function for 1 degree of freedom:
    p-value = erfc(sqrt(LR / 2))
    """

    if observations <= 0:
        return {
            "Exception Rate": np.nan,
            "Expected Exception Rate": alpha,
            "Kupiec LR": np.nan,
            "Kupiec p-value": np.nan,
            "Backtest Signal": "Unavailable",
        }

    exception_rate = exceptions / observations

    if exceptions == 0 or exceptions == observations:
        lr_stat = np.nan
        p_value = np.nan
    else:
        likelihood_null = (
            ((1 - alpha) ** (observations - exceptions))
            * (alpha**exceptions)
        )

        likelihood_alt = (
            ((1 - exception_rate) ** (observations - exceptions))
            * (exception_rate**exceptions)
        )

        if likelihood_null <= 0 or likelihood_alt <= 0:
            lr_stat = np.nan
            p_value = np.nan
        else:
            lr_stat = -2 * math.log(likelihood_null / likelihood_alt)
            p_value = math.erfc(math.sqrt(lr_stat / 2))

    if pd.isna(p_value):
        signal = "Unavailable"
    elif p_value >= 0.05:
        signal = "Pass"
    else:
        signal = "Review"

    return {
        "Exception Rate": exception_rate,
        "Expected Exception Rate": alpha,
        "Kupiec LR": lr_stat,
        "Kupiec p-value": p_value,
        "Backtest Signal": signal,
    }


def _historical_var(losses: pd.Series, confidence_level: float) -> float:
    """Estimate historical VaR as a positive upper-tail loss value."""

    if losses.empty:
        return np.nan

    clean_losses = pd.to_numeric(losses, errors="coerce").dropna()

    if clean_losses.empty:
        return np.nan

    return _positive_or_nan(clean_losses.quantile(confidence_level))


def _historical_expected_shortfall(losses: pd.Series, confidence_level: float) -> float:
    """Estimate expected shortfall as average loss beyond historical VaR."""

    if losses.empty:
        return np.nan

    clean_losses = pd.to_numeric(losses, errors="coerce").dropna()

    if clean_losses.empty:
        return np.nan

    threshold = _historical_var(clean_losses, confidence_level)
    tail_losses = clean_losses[clean_losses >= threshold]

    if tail_losses.empty:
        return np.nan

    return _positive_or_nan(tail_losses.mean())


def _fit_garch_volatility(returns: pd.Series) -> tuple[pd.Series | None, str, str]:
    """Fit a GARCH model and return daily conditional volatility."""

    if not USE_ARCH_GARCH:
        return None, "EWMA fallback", "GARCH disabled in config.py; using EWMA fallback."

    if not ARCH_AVAILABLE:
        return None, "EWMA fallback", "arch package unavailable; using EWMA fallback."

    clean_returns = pd.to_numeric(returns, errors="coerce").dropna()

    if len(clean_returns) < TRADING_DAYS:
        return None, "EWMA fallback", "Insufficient return history for stable GARCH fit; using EWMA fallback."

    scaled_returns = clean_returns * 100

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            model = arch_model(
                scaled_returns,
                mean="Constant",
                vol="GARCH",
                p=GARCH_P,
                o=0,
                q=GARCH_Q,
                dist=GARCH_DISTRIBUTION,
                rescale=False,
            )

            result = model.fit(disp="off", show_warning=False)

        conditional_vol = result.conditional_volatility / 100
        conditional_vol = pd.Series(
            conditional_vol,
            index=clean_returns.index,
            name="GARCH Daily Volatility",
        )

        convergence_flag = getattr(result, "convergence_flag", 0)
        model_name = f"GARCH({GARCH_P},{GARCH_Q})"

        if convergence_flag != 0:
            return (
                conditional_vol,
                model_name,
                f"{model_name} model fit with non-zero convergence flag: {convergence_flag}.",
            )

        return conditional_vol, model_name, f"{model_name} model fit successfully."

    except Exception as exc:
        return None, "EWMA fallback", f"GARCH fit failed; using EWMA fallback. Error: {exc}"


def _build_timeseries(returns: pd.Series) -> tuple[pd.DataFrame, str, str]:
    """Build volatility, VaR, loss, and exception time series."""

    returns = pd.to_numeric(returns, errors="coerce").dropna()

    if returns.empty:
        return pd.DataFrame(), "Unavailable", "Return data unavailable."

    garch_vol_daily, model_name, model_status = _fit_garch_volatility(returns)

    ewma_variance = returns.pow(2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    ewma_vol_daily = np.sqrt(ewma_variance)

    if garch_vol_daily is not None and not garch_vol_daily.empty:
        primary_vol_daily = garch_vol_daily.reindex(returns.index)
        primary_model_name = model_name
    else:
        primary_vol_daily = ewma_vol_daily
        primary_model_name = "EWMA fallback"

    primary_vol_daily = pd.to_numeric(primary_vol_daily, errors="coerce")
    primary_vol_annualized = primary_vol_daily * np.sqrt(TRADING_DAYS)
    ewma_vol_annualized = ewma_vol_daily * np.sqrt(TRADING_DAYS)

    daily_loss = -returns

    confidence_95 = 0.95 if 0.95 in VAR_CONFIDENCE_LEVELS else VAR_CONFIDENCE_LEVELS[0]
    confidence_99 = 0.99 if 0.99 in VAR_CONFIDENCE_LEVELS else VAR_CONFIDENCE_LEVELS[-1]

    z_95 = _get_z_score(confidence_95)
    z_99 = _get_z_score(confidence_99)

    primary_var_95 = z_95 * primary_vol_daily
    primary_var_99 = z_99 * primary_vol_daily

    ewma_var_95 = z_95 * ewma_vol_daily
    ewma_var_99 = z_99 * ewma_vol_daily

    rolling_vol_21 = returns.rolling(21).std() * np.sqrt(TRADING_DAYS)
    rolling_vol_63 = returns.rolling(63).std() * np.sqrt(TRADING_DAYS)
    rolling_vol_252 = returns.rolling(252).std() * np.sqrt(TRADING_DAYS)

    timeseries = pd.DataFrame(
        {
            "Date": returns.index,
            "Daily Return": returns.values,
            "Daily Return Type": "Percent",
            "Daily Loss": daily_loss.values,
            "Daily Loss Type": "Percent",
            "Primary Daily Volatility": primary_vol_daily.values,
            "Primary Daily Volatility Type": "Percent",
            "Primary Annualized Volatility": primary_vol_annualized.values,
            "Primary Annualized Volatility Type": "Percent",
            "Primary VaR 95": primary_var_95.values,
            "Primary VaR 95 Type": "Percent",
            "Primary VaR 99": primary_var_99.values,
            "Primary VaR 99 Type": "Percent",
            "EWMA Daily Volatility": ewma_vol_daily.values,
            "EWMA Daily Volatility Type": "Percent",
            "EWMA Annualized Volatility": ewma_vol_annualized.values,
            "EWMA Annualized Volatility Type": "Percent",
            "EWMA VaR 95": ewma_var_95.values,
            "EWMA VaR 95 Type": "Percent",
            "EWMA VaR 99": ewma_var_99.values,
            "EWMA VaR 99 Type": "Percent",
            "Rolling 21D Annualized Volatility": rolling_vol_21.values,
            "Rolling 21D Annualized Volatility Type": "Percent",
            "Rolling 63D Annualized Volatility": rolling_vol_63.values,
            "Rolling 63D Annualized Volatility Type": "Percent",
            "Rolling 252D Annualized Volatility": rolling_vol_252.values,
            "Rolling 252D Annualized Volatility Type": "Percent",
            "Volatility Model": primary_model_name,
            "Volatility Model Type": "Text",
            "Model Status": model_status,
            "Model Status Type": "Text",
        }
    )

    timeseries["Primary VaR 95 Exception"] = (
        timeseries["Daily Loss"] > timeseries["Primary VaR 95"]
    ).astype(int)

    timeseries["Primary VaR 95 Exception Type"] = "Plain Number"

    timeseries["Primary VaR 99 Exception"] = (
        timeseries["Daily Loss"] > timeseries["Primary VaR 99"]
    ).astype(int)

    timeseries["Primary VaR 99 Exception Type"] = "Plain Number"

    timeseries["EWMA VaR 95 Exception"] = (
        timeseries["Daily Loss"] > timeseries["EWMA VaR 95"]
    ).astype(int)

    timeseries["EWMA VaR 95 Exception Type"] = "Plain Number"

    timeseries["EWMA VaR 99 Exception"] = (
        timeseries["Daily Loss"] > timeseries["EWMA VaR 99"]
    ).astype(int)

    timeseries["EWMA VaR 99 Exception Type"] = "Plain Number"

    return timeseries, primary_model_name, model_status


def _build_summary(
    returns: pd.Series,
    timeseries: pd.DataFrame,
    volatility_model: str,
    model_status: str,
) -> pd.DataFrame:
    """Build GARCH / EWMA VaR summary table."""

    if returns.empty or timeseries.empty:
        return pd.DataFrame(
            [
                _summary_row(
                    metric="GARCH / VaR Status",
                    value="Return data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Return data was unavailable.",
                )
            ]
        )

    losses = -returns
    latest = timeseries.dropna(subset=["Primary Annualized Volatility"]).tail(1)

    latest_primary_vol = (
        _safe_float(latest["Primary Annualized Volatility"].iloc[0])
        if not latest.empty
        else np.nan
    )

    latest_primary_var_95 = (
        _safe_float(latest["Primary VaR 95"].iloc[0])
        if not latest.empty
        else np.nan
    )

    latest_primary_var_99 = (
        _safe_float(latest["Primary VaR 99"].iloc[0])
        if not latest.empty
        else np.nan
    )

    latest_ewma_vol = (
        _safe_float(latest["EWMA Annualized Volatility"].iloc[0])
        if not latest.empty
        else np.nan
    )

    latest_ewma_var_95 = (
        _safe_float(latest["EWMA VaR 95"].iloc[0])
        if not latest.empty
        else np.nan
    )

    latest_ewma_var_99 = (
        _safe_float(latest["EWMA VaR 99"].iloc[0])
        if not latest.empty
        else np.nan
    )

    rolling_21 = timeseries["Rolling 21D Annualized Volatility"].dropna()
    rolling_63 = timeseries["Rolling 63D Annualized Volatility"].dropna()
    rolling_252 = timeseries["Rolling 252D Annualized Volatility"].dropna()

    hist_var_95 = _historical_var(losses, 0.95)
    hist_var_99 = _historical_var(losses, 0.99)
    hist_es_95 = _historical_expected_shortfall(losses, 0.95)
    hist_es_99 = _historical_expected_shortfall(losses, 0.99)

    mean_daily = returns.mean()
    std_daily = returns.std()

    parametric_var_95 = _positive_or_nan(-(mean_daily - _get_z_score(0.95) * std_daily))
    parametric_var_99 = _positive_or_nan(-(mean_daily - _get_z_score(0.99) * std_daily))

    max_daily_loss = _positive_or_nan(losses.max())

    if pd.isna(latest_primary_vol):
        vol_signal = "Unavailable"
    elif latest_primary_vol < 0.20:
        vol_signal = "Low / Stable"
    elif latest_primary_vol < 0.35:
        vol_signal = "Moderate"
    elif latest_primary_vol < 0.50:
        vol_signal = "Elevated"
    else:
        vol_signal = "High Risk"

    rows = [
        _summary_row(
            metric="Volatility Model Used",
            value=volatility_model,
            value_type="Text",
            signal="Model Selection",
            interpretation=model_status,
        ),
        _summary_row(
            metric="Latest Primary Annualized Volatility",
            value=latest_primary_vol,
            value_type="Percent",
            signal=vol_signal,
            interpretation="Latest annualized volatility estimate from the selected primary volatility model.",
        ),
        _summary_row(
            metric="Latest EWMA Annualized Volatility",
            value=latest_ewma_vol,
            value_type="Percent",
            signal="EWMA Reference",
            interpretation="Latest EWMA annualized volatility estimate retained as a benchmark.",
        ),
        _summary_row(
            metric="Rolling 21D Annualized Volatility",
            value=rolling_21.iloc[-1] if not rolling_21.empty else np.nan,
            value_type="Percent",
            signal="Short-Term Volatility",
            interpretation="Recent one-month realized volatility.",
        ),
        _summary_row(
            metric="Rolling 63D Annualized Volatility",
            value=rolling_63.iloc[-1] if not rolling_63.empty else np.nan,
            value_type="Percent",
            signal="Medium-Term Volatility",
            interpretation="Recent three-month realized volatility.",
        ),
        _summary_row(
            metric="Rolling 252D Annualized Volatility",
            value=rolling_252.iloc[-1] if not rolling_252.empty else np.nan,
            value_type="Percent",
            signal="Long-Term Volatility",
            interpretation="Recent one-year realized volatility.",
        ),
        _summary_row(
            metric="Historical VaR 95",
            value=hist_var_95,
            value_type="Percent",
            signal="One-Day Loss Estimate",
            interpretation="Historical one-day loss threshold at the 95% confidence level.",
        ),
        _summary_row(
            metric="Historical VaR 99",
            value=hist_var_99,
            value_type="Percent",
            signal="Tail Loss Estimate",
            interpretation="Historical one-day loss threshold at the 99% confidence level.",
        ),
        _summary_row(
            metric="Parametric VaR 95",
            value=parametric_var_95,
            value_type="Percent",
            signal="One-Day Loss Estimate",
            interpretation="Normal-distribution one-day loss estimate at the 95% confidence level.",
        ),
        _summary_row(
            metric="Parametric VaR 99",
            value=parametric_var_99,
            value_type="Percent",
            signal="Tail Loss Estimate",
            interpretation="Normal-distribution one-day loss estimate at the 99% confidence level.",
        ),
        _summary_row(
            metric="Primary VaR 95",
            value=latest_primary_var_95,
            value_type="Percent",
            signal=f"{volatility_model} VaR",
            interpretation="Selected volatility-model one-day loss estimate at the 95% confidence level.",
        ),
        _summary_row(
            metric="Primary VaR 99",
            value=latest_primary_var_99,
            value_type="Percent",
            signal=f"{volatility_model} Tail VaR",
            interpretation="Selected volatility-model one-day loss estimate at the 99% confidence level.",
        ),
        _summary_row(
            metric="EWMA VaR 95",
            value=latest_ewma_var_95,
            value_type="Percent",
            signal="EWMA Reference VaR",
            interpretation="EWMA volatility-based one-day loss estimate at the 95% confidence level.",
        ),
        _summary_row(
            metric="EWMA VaR 99",
            value=latest_ewma_var_99,
            value_type="Percent",
            signal="EWMA Reference Tail VaR",
            interpretation="EWMA volatility-based one-day loss estimate at the 99% confidence level.",
        ),
        _summary_row(
            metric="Expected Shortfall 95",
            value=hist_es_95,
            value_type="Percent",
            signal="Tail Loss Average",
            interpretation="Average historical loss beyond the 95% VaR threshold.",
        ),
        _summary_row(
            metric="Expected Shortfall 99",
            value=hist_es_99,
            value_type="Percent",
            signal="Extreme Tail Loss Average",
            interpretation="Average historical loss beyond the 99% VaR threshold.",
        ),
        _summary_row(
            metric="Maximum Daily Loss",
            value=max_daily_loss,
            value_type="Percent",
            signal="Historical Stress Point",
            interpretation="Worst one-day loss observed in the study window.",
        ),
    ]

    return pd.DataFrame(rows)


def _build_regime_summary(timeseries: pd.DataFrame, volatility_model: str) -> pd.DataFrame:
    """Build volatility regime summary."""

    if timeseries.empty or "Primary Annualized Volatility" not in timeseries.columns:
        return pd.DataFrame(
            [
                _summary_row(
                    metric="Volatility Regime",
                    value="Volatility data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Volatility data was unavailable.",
                )
            ]
        )

    vol = pd.to_numeric(
        timeseries["Primary Annualized Volatility"],
        errors="coerce",
    ).dropna()

    if vol.empty:
        return pd.DataFrame(
            [
                _summary_row(
                    metric="Volatility Regime",
                    value="Volatility data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Volatility data was unavailable.",
                )
            ]
        )

    latest_vol = vol.iloc[-1]
    percentile = (vol <= latest_vol).mean()

    if percentile < 0.25:
        regime = "Low Volatility Regime"
        interpretation = "Current volatility is below most historical observations."
    elif percentile < 0.60:
        regime = "Normal Volatility Regime"
        interpretation = "Current volatility is within a normal historical range."
    elif percentile < 0.85:
        regime = "Elevated Volatility Regime"
        interpretation = "Current volatility is above typical historical levels."
    else:
        regime = "High Volatility Regime"
        interpretation = "Current volatility is near the upper end of the historical range."

    rows = [
        _summary_row(
            metric="Volatility Model Used",
            value=volatility_model,
            value_type="Text",
            signal="Model Selection",
            interpretation="Primary volatility model used for regime classification.",
        ),
        _summary_row(
            metric="Latest Primary Volatility",
            value=latest_vol,
            value_type="Percent",
            signal=regime,
            interpretation="Latest annualized volatility estimate from the selected model.",
        ),
        _summary_row(
            metric="Volatility Regime Percentile",
            value=percentile,
            value_type="Percent",
            signal=regime,
            interpretation=interpretation,
        ),
        _summary_row(
            metric="Median Primary Volatility",
            value=vol.median(),
            value_type="Percent",
            signal="Historical Reference",
            interpretation="Median annualized primary volatility across the study period.",
        ),
        _summary_row(
            metric="90th Percentile Primary Volatility",
            value=vol.quantile(0.90),
            value_type="Percent",
            signal="Stress Reference",
            interpretation="Historical high-volatility reference threshold.",
        ),
        _summary_row(
            metric="Maximum Primary Volatility",
            value=vol.max(),
            value_type="Percent",
            signal="Maximum Stress Reference",
            interpretation="Highest annualized primary volatility observed in the study period.",
        ),
    ]

    return pd.DataFrame(rows)


def _build_backtest_summary(timeseries: pd.DataFrame, volatility_model: str) -> pd.DataFrame:
    """Build primary VaR backtest summary."""

    if timeseries.empty:
        return pd.DataFrame(
            [
                _summary_row(
                    metric="VaR Backtest Status",
                    value="VaR backtest data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="VaR backtest data was unavailable.",
                )
            ]
        )

    backtest_data = timeseries.dropna(
        subset=["Primary VaR 95", "Primary VaR 99", "Daily Loss"]
    ).copy()

    if backtest_data.empty:
        return pd.DataFrame(
            [
                _summary_row(
                    metric="VaR Backtest Status",
                    value="VaR backtest data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="VaR backtest data was unavailable.",
                )
            ]
        )

    observations = len(backtest_data)
    exceptions_95 = int(backtest_data["Primary VaR 95 Exception"].sum())
    exceptions_99 = int(backtest_data["Primary VaR 99 Exception"].sum())

    test_95 = _kupiec_pof_test(exceptions_95, observations, alpha=0.05)
    test_99 = _kupiec_pof_test(exceptions_99, observations, alpha=0.01)

    rows = [
        _summary_row(
            metric="VaR Model Used",
            value=volatility_model,
            value_type="Text",
            signal="Model Selection",
            interpretation="Primary volatility model used for VaR backtesting.",
        ),
        _summary_row(
            metric="Backtest Observations",
            value=observations,
            value_type="Plain Number",
            signal="Reference",
            interpretation="Number of daily observations used for the VaR backtest.",
        ),
        _summary_row(
            metric="Primary VaR 95 Exceptions",
            value=exceptions_95,
            value_type="Plain Number",
            signal=test_95["Backtest Signal"],
            interpretation="Number of days when realized loss exceeded the primary 95% VaR threshold.",
        ),
        _summary_row(
            metric="Primary VaR 95 Exception Rate",
            value=test_95["Exception Rate"],
            value_type="Percent",
            signal=test_95["Backtest Signal"],
            interpretation="Observed exception rate versus expected 5% exception rate.",
        ),
        _summary_row(
            metric="Primary VaR 95 Kupiec p-value",
            value=test_95["Kupiec p-value"],
            value_type="Plain Number",
            signal=test_95["Backtest Signal"],
            interpretation="Kupiec POF test p-value; values above 0.05 generally indicate acceptable calibration.",
        ),
        _summary_row(
            metric="Primary VaR 99 Exceptions",
            value=exceptions_99,
            value_type="Plain Number",
            signal=test_99["Backtest Signal"],
            interpretation="Number of days when realized loss exceeded the primary 99% VaR threshold.",
        ),
        _summary_row(
            metric="Primary VaR 99 Exception Rate",
            value=test_99["Exception Rate"],
            value_type="Percent",
            signal=test_99["Backtest Signal"],
            interpretation="Observed exception rate versus expected 1% exception rate.",
        ),
        _summary_row(
            metric="Primary VaR 99 Kupiec p-value",
            value=test_99["Kupiec p-value"],
            value_type="Plain Number",
            signal=test_99["Backtest Signal"],
            interpretation="Kupiec POF test p-value; values above 0.05 generally indicate acceptable calibration.",
        ),
    ]

    return pd.DataFrame(rows)


def _save_charts(timeseries: pd.DataFrame) -> list[str]:
    """Save volatility and VaR backtest charts."""

    chart_paths = []

    if timeseries.empty:
        return chart_paths

    chart_data = timeseries.copy()
    chart_data["Date"] = pd.to_datetime(chart_data["Date"], errors="coerce")
    chart_data = chart_data.dropna(subset=["Date"])

    if chart_data.empty:
        return chart_paths

    vol_path = CHARTS_DIR / "conditional_volatility_chart.png"
    vol_data = chart_data.dropna(subset=["Primary Annualized Volatility"])

    if not vol_data.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(
            vol_data["Date"],
            vol_data["Primary Annualized Volatility"],
            label="Primary Annualized Volatility",
        )

        if "EWMA Annualized Volatility" in vol_data.columns:
            ewma_vol = pd.to_numeric(
                vol_data["EWMA Annualized Volatility"],
                errors="coerce",
            )

            if ewma_vol.notna().any():
                plt.plot(
                    vol_data["Date"],
                    ewma_vol,
                    label="EWMA Annualized Volatility",
                    linewidth=1.0,
                )

        plt.title(f"{COMPANY_NAME} Conditional Volatility")
        plt.xlabel("Date")
        plt.ylabel("Annualized Volatility")
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )
        plt.legend()

        saved = _safe_savefig(vol_path)

        if saved:
            chart_paths.append(saved)

    var_path = CHARTS_DIR / "var_backtest_chart.png"

    var_data = chart_data.dropna(
        subset=["Daily Loss", "Primary VaR 95", "Primary VaR 99"]
    )

    if not var_data.empty:
        plt.figure(figsize=(10, 5))
        plt.plot(
            var_data["Date"],
            var_data["Daily Loss"],
            label="Daily Loss",
            linewidth=0.8,
        )
        plt.plot(
            var_data["Date"],
            var_data["Primary VaR 95"],
            label="Primary VaR 95",
            linewidth=1.1,
        )
        plt.plot(
            var_data["Date"],
            var_data["Primary VaR 99"],
            label="Primary VaR 99",
            linewidth=1.1,
        )

        exceptions = var_data[var_data["Primary VaR 95 Exception"] == 1]

        if not exceptions.empty:
            plt.scatter(
                exceptions["Date"],
                exceptions["Daily Loss"],
                s=12,
                label="95% VaR Exceptions",
            )

        plt.title(f"{COMPANY_NAME} Primary VaR Backtest")
        plt.xlabel("Date")
        plt.ylabel("Daily Loss / VaR")
        plt.gca().yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.1%}")
        )
        plt.legend()

        saved = _safe_savefig(var_path)

        if saved:
            chart_paths.append(saved)

    return chart_paths


def run_garch_var_model(prices: pd.DataFrame) -> dict[str, pd.DataFrame | list[str]]:
    """Run GARCH / EWMA volatility and VaR analysis."""

    price_series = _extract_price_series(prices)

    if price_series.empty:
        summary = pd.DataFrame(
            [
                _summary_row(
                    metric="GARCH / VaR Status",
                    value="Price data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Price data was unavailable.",
                )
            ]
        )

        regime_summary = pd.DataFrame(
            [
                _summary_row(
                    metric="Volatility Regime",
                    value="Price data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Price data was unavailable.",
                )
            ]
        )

        backtest_summary = pd.DataFrame(
            [
                _summary_row(
                    metric="VaR Backtest Status",
                    value="Price data unavailable",
                    value_type="Text",
                    signal="Unavailable",
                    interpretation="Price data was unavailable.",
                )
            ]
        )

        timeseries = pd.DataFrame()
        chart_paths = []

        _safe_to_csv(summary, "garch_var_summary.csv")
        _safe_to_csv(regime_summary, "volatility_regime_summary.csv")
        _safe_to_csv(backtest_summary, "var_backtest_summary.csv")
        _safe_to_csv(timeseries, "garch_var_timeseries.csv")

        return {
            "garch_var_summary": summary,
            "volatility_regime_summary": regime_summary,
            "var_backtest_summary": backtest_summary,
            "garch_var_timeseries": timeseries,
            "garch_var_chart_paths": chart_paths,
        }

    returns = price_series.pct_change(fill_method=None).dropna()

    timeseries, volatility_model, model_status = _build_timeseries(returns)

    summary = _build_summary(
        returns=returns,
        timeseries=timeseries,
        volatility_model=volatility_model,
        model_status=model_status,
    )

    regime_summary = _build_regime_summary(
        timeseries=timeseries,
        volatility_model=volatility_model,
    )

    backtest_summary = _build_backtest_summary(
        timeseries=timeseries,
        volatility_model=volatility_model,
    )

    chart_paths = _save_charts(timeseries)

    _safe_to_csv(summary, "garch_var_summary.csv")
    _safe_to_csv(regime_summary, "volatility_regime_summary.csv")
    _safe_to_csv(backtest_summary, "var_backtest_summary.csv")
    _safe_to_csv(timeseries, "garch_var_timeseries.csv")

    return {
        "garch_var_summary": summary,
        "volatility_regime_summary": regime_summary,
        "var_backtest_summary": backtest_summary,
        "garch_var_timeseries": timeseries,
        "garch_var_chart_paths": chart_paths,
    }