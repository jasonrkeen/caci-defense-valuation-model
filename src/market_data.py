import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    TICKER,
    BENCHMARK,
    BENCHMARK_NAME,
    TREASURY_TICKER,
    START_DATE,
    DEFAULT_RISK_FREE_RATE,
    PROCESSED_DIR,
    TRADING_DAYS,
)


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to data/processed with basic safety."""

    try:
        df.to_csv(PROCESSED_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _safe_time_series_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a time-series DataFrame while preserving the date index."""

    try:
        df.to_csv(PROCESSED_DIR / file_name)
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


def _return_signal(value: float) -> str:
    """Convert a return metric into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.25:
        return "Strong"
    if value >= 0.10:
        return "Constructive"
    if value >= -0.05:
        return "Neutral / Monitor"
    if value >= -0.15:
        return "Cautious"
    return "Weak"


def _risk_signal(value: float, lower_is_better: bool = False) -> str:
    """Convert a risk metric into a signal."""

    if pd.isna(value):
        return "Unavailable"

    if lower_is_better:
        if value <= 0.20:
            return "Low Risk"
        if value <= 0.30:
            return "Moderate Risk"
        if value <= 0.40:
            return "Elevated Risk"
        return "High Risk"

    if value >= 0.75:
        return "Strong"
    if value >= 0.50:
        return "Constructive"
    if value >= 0.25:
        return "Moderate"
    return "Weak"


def _drawdown_signal(value: float) -> str:
    """Convert maximum drawdown into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= -0.20:
        return "Resilient"
    if value >= -0.35:
        return "Moderate Drawdown"
    if value >= -0.50:
        return "Elevated Drawdown"
    return "Severe Drawdown"


def _sharpe_signal(value: float) -> str:
    """Convert Sharpe ratio into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 1.0:
        return "Strong"
    if value >= 0.5:
        return "Constructive"
    if value >= 0:
        return "Neutral / Monitor"
    return "Weak"


def _beta_signal(value: float) -> str:
    """Convert beta into a benchmark sensitivity signal."""

    if pd.isna(value):
        return "Unavailable"
    if 0.75 <= value <= 1.15:
        return "Market-Like"
    if value < 0.75:
        return "Lower Beta"
    if value <= 1.40:
        return "Elevated Beta"
    return "High Beta"


def _correlation_signal(value: float) -> str:
    """Convert correlation into a benchmark relationship signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.80:
        return "High Correlation"
    if value >= 0.60:
        return "Moderate / High Correlation"
    if value >= 0.30:
        return "Moderate Correlation"
    if value >= 0:
        return "Low Correlation"
    return "Negative Correlation"


def _information_ratio_signal(value: float) -> str:
    """Convert information ratio into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.5:
        return "Strong Active Return"
    if value >= 0.2:
        return "Constructive"
    if value >= -0.2:
        return "Neutral / Monitor"
    return "Weak Active Return"


def _risk_free_rate_signal(value: float) -> str:
    """Convert risk-free rate into a discount-rate backdrop signal."""

    if pd.isna(value):
        return "Unavailable"
    if value >= 0.055:
        return "High Rate Pressure"
    if value >= 0.040:
        return "Moderate Rate Pressure"
    if value >= 0.025:
        return "Normal Rate Backdrop"
    return "Low Rate Backdrop"


def _metric_value_type(metric_name: str) -> str:
    """Assign explicit formatting metadata for market metric values."""

    metric = str(metric_name).lower()

    if "index level" in metric:
        return "Plain Number"

    if "price" in metric:
        return "Money"

    if "correlation" in metric:
        return "Plain Number"

    if any(
        token in metric
        for token in [
            "return",
            "volatility",
            "drawdown",
            "tracking error",
            "risk-free rate",
        ]
    ):
        return "Percent"

    if any(
        token in metric
        for token in [
            "sharpe",
            "sortino",
            "beta",
            "information ratio",
        ]
    ):
        return "Plain Number"

    return "Plain Number"


def _extract_close_frame(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Extract close prices from a yfinance download response."""

    if data is None or data.empty:
        raise ValueError("No market data returned from Yahoo Finance.")

    if isinstance(data.columns, pd.MultiIndex):
        first_level = data.columns.get_level_values(0)

        if "Close" not in first_level:
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

    prices = prices.dropna(how="all")

    missing_tickers = [ticker for ticker in tickers if ticker not in prices.columns]

    if missing_tickers:
        raise ValueError(f"Missing price data for: {missing_tickers}")

    prices = prices[tickers].dropna()

    if prices.empty:
        raise ValueError("Price table became empty after cleaning missing values.")

    for ticker in tickers:
        prices[ticker] = pd.to_numeric(prices[ticker], errors="coerce")

    prices = prices.dropna()

    if prices.empty:
        raise ValueError("Price table became empty after numeric conversion.")

    if (prices <= 0).any().any():
        raise ValueError("Price table contains nonpositive prices.")

    return prices


def download_prices(tickers=None, start_date=START_DATE) -> pd.DataFrame:
    """Download adjusted close prices for CACI and benchmark."""

    if tickers is None:
        tickers = [TICKER, BENCHMARK]

    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers,
        start=start_date,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    prices = _extract_close_frame(data, tickers)

    _safe_time_series_to_csv(prices, "market_prices.csv")

    return prices


def _extract_latest_close(data: pd.DataFrame) -> float | None:
    """Extract the latest close value from a yfinance response."""

    if data is None or data.empty:
        return None

    try:
        if isinstance(data.columns, pd.MultiIndex):
            if "Close" not in data.columns.get_level_values(0):
                return None

            close_data = data["Close"]

            if isinstance(close_data, pd.DataFrame):
                close_series = close_data.iloc[:, 0].dropna()
            else:
                close_series = close_data.dropna()

        else:
            if "Close" not in data.columns:
                return None

            close_data = data["Close"]

            if isinstance(close_data, pd.DataFrame):
                close_series = close_data.iloc[:, 0].dropna()
            else:
                close_series = close_data.dropna()

        if close_series.empty:
            return None

        latest = _safe_float(close_series.iloc[-1])

        if pd.isna(latest) or latest <= 0:
            return None

        return latest

    except Exception:
        return None


def get_live_risk_free_rate() -> float:
    """Pull the latest 10-year Treasury yield proxy from Yahoo Finance.

    ^TNX is usually quoted as yield * 10.
    Example: 45.0 means 4.50%.
    """

    try:
        data = yf.download(
            TREASURY_TICKER,
            period="10d",
            auto_adjust=True,
            progress=False,
            threads=False,
        )

        latest = _extract_latest_close(data)

        if latest is None:
            return DEFAULT_RISK_FREE_RATE

        # ^TNX is commonly quoted around 45 for 4.50%.
        # Some feeds can quote around 4.5, so handle both scales.
        if latest > 20:
            risk_free_rate = latest / 1000.0
        elif latest > 1:
            risk_free_rate = latest / 100.0
        else:
            risk_free_rate = latest

        if risk_free_rate <= 0 or risk_free_rate > 0.20:
            return DEFAULT_RISK_FREE_RATE

        return float(risk_free_rate)

    except Exception:
        return DEFAULT_RISK_FREE_RATE


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Calculate daily returns."""

    returns = prices.pct_change(fill_method=None).dropna()
    _safe_time_series_to_csv(returns, "daily_returns.csv")

    return returns


def max_drawdown(return_series: pd.Series) -> float:
    """Calculate maximum drawdown from a return series."""

    if return_series is None or return_series.empty:
        return np.nan

    wealth = (1 + return_series).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1

    return float(drawdown.min())


def downside_deviation(return_series: pd.Series, risk_free_rate: float) -> float:
    """Calculate annualized downside deviation."""

    if return_series is None or return_series.empty:
        return np.nan

    daily_rf = risk_free_rate / TRADING_DAYS
    downside_returns = return_series[return_series < daily_rf] - daily_rf

    if downside_returns.empty:
        return 0.0

    return float(downside_returns.std() * np.sqrt(TRADING_DAYS))


def sortino_ratio(
    return_series: pd.Series,
    annualized_return: float,
    risk_free_rate: float,
) -> float:
    """Calculate Sortino ratio."""

    downside_vol = downside_deviation(return_series, risk_free_rate)

    if pd.isna(downside_vol) or downside_vol == 0:
        return np.nan

    return float((annualized_return - risk_free_rate) / downside_vol)


def calculate_beta(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate beta versus benchmark."""

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()

    if aligned.empty:
        return np.nan

    aligned.columns = ["stock", "benchmark"]

    benchmark_variance = aligned["benchmark"].var()

    if pd.isna(benchmark_variance) or benchmark_variance == 0:
        return np.nan

    beta = aligned["stock"].cov(aligned["benchmark"]) / benchmark_variance

    return float(beta)


def calculate_correlation(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate correlation versus benchmark."""

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()

    if aligned.empty:
        return np.nan

    aligned.columns = ["stock", "benchmark"]

    return float(aligned["stock"].corr(aligned["benchmark"]))


def calculate_one_year_return(prices: pd.DataFrame, ticker: str) -> float:
    """Calculate approximate one-year return using 252 trading days."""

    if ticker not in prices.columns:
        return np.nan

    clean_prices = prices[ticker].dropna()

    if len(clean_prices) < TRADING_DAYS:
        return np.nan

    prior_price = clean_prices.iloc[-TRADING_DAYS]
    latest_price = clean_prices.iloc[-1]

    if pd.isna(prior_price) or prior_price <= 0:
        return np.nan

    return float(latest_price / prior_price - 1)


def annualized_return_from_prices(price_series: pd.Series) -> float:
    """Calculate annualized return from a price series."""

    if price_series is None or price_series.empty or len(price_series) < 2:
        return np.nan

    clean_prices = price_series.dropna()

    if clean_prices.empty or len(clean_prices) < 2:
        return np.nan

    first_price = clean_prices.iloc[0]
    latest_price = clean_prices.iloc[-1]

    if pd.isna(first_price) or first_price <= 0:
        return np.nan

    total_return = latest_price / first_price - 1
    periods = len(clean_prices) - 1

    return float((1 + total_return) ** (TRADING_DAYS / periods) - 1)


def annualized_volatility(return_series: pd.Series) -> float:
    """Calculate annualized volatility."""

    if return_series is None or return_series.empty:
        return np.nan

    return float(return_series.std() * np.sqrt(TRADING_DAYS))


def tracking_error(stock_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate annualized tracking error versus benchmark."""

    aligned = pd.concat([stock_returns, benchmark_returns], axis=1).dropna()

    if aligned.empty:
        return np.nan

    aligned.columns = ["stock", "benchmark"]

    active_returns = aligned["stock"] - aligned["benchmark"]

    return float(active_returns.std() * np.sqrt(TRADING_DAYS))


def information_ratio(
    stock_annualized_return: float,
    benchmark_annualized_return: float,
    tracking_error_value: float,
) -> float:
    """Calculate information ratio."""

    if pd.isna(tracking_error_value) or tracking_error_value == 0:
        return np.nan

    return float((stock_annualized_return - benchmark_annualized_return) / tracking_error_value)


def _require_price_columns(prices: pd.DataFrame) -> None:
    """Validate that required ticker columns exist."""

    if prices is None or prices.empty:
        raise ValueError("Prices DataFrame is empty.")

    required = [TICKER, BENCHMARK]
    missing = [ticker for ticker in required if ticker not in prices.columns]

    if missing:
        raise ValueError(f"Missing required price columns: {missing}")

    for ticker in required:
        clean_prices = pd.to_numeric(prices[ticker], errors="coerce").dropna()

        if clean_prices.empty:
            raise ValueError(f"No usable price data found for {ticker}.")

        if (clean_prices <= 0).any():
            raise ValueError(f"Nonpositive price data found for {ticker}.")


def calculate_market_metrics(
    prices: pd.DataFrame,
    risk_free_rate=None,
) -> pd.DataFrame:
    """Calculate market and risk metrics for CACI."""

    _require_price_columns(prices)

    if risk_free_rate is None:
        risk_free_rate = get_live_risk_free_rate()

    risk_free_rate = _safe_float(risk_free_rate, default=DEFAULT_RISK_FREE_RATE)

    returns = calculate_returns(prices)

    if TICKER not in returns.columns or BENCHMARK not in returns.columns:
        raise ValueError("Daily returns are missing required ticker columns.")

    stock_returns = returns[TICKER].dropna()
    benchmark_returns = returns[BENCHMARK].dropna()

    clean_stock_prices = prices[TICKER].dropna()
    clean_benchmark_prices = prices[BENCHMARK].dropna()

    latest_price = float(clean_stock_prices.iloc[-1])
    latest_benchmark_index_level = float(clean_benchmark_prices.iloc[-1])

    total_return = float(clean_stock_prices.iloc[-1] / clean_stock_prices.iloc[0] - 1)
    benchmark_total_return = float(
        clean_benchmark_prices.iloc[-1] / clean_benchmark_prices.iloc[0] - 1
    )

    annualized_stock_return = annualized_return_from_prices(clean_stock_prices)
    annualized_benchmark_return = annualized_return_from_prices(clean_benchmark_prices)

    stock_volatility = annualized_volatility(stock_returns)
    benchmark_volatility = annualized_volatility(benchmark_returns)

    sharpe_ratio = (
        float((annualized_stock_return - risk_free_rate) / stock_volatility)
        if pd.notna(stock_volatility) and stock_volatility != 0
        else np.nan
    )

    benchmark_sharpe = (
        float((annualized_benchmark_return - risk_free_rate) / benchmark_volatility)
        if pd.notna(benchmark_volatility) and benchmark_volatility != 0
        else np.nan
    )

    beta = calculate_beta(stock_returns, benchmark_returns)
    correlation = calculate_correlation(stock_returns, benchmark_returns)
    sortino = sortino_ratio(stock_returns, annualized_stock_return, risk_free_rate)
    max_dd = max_drawdown(stock_returns)
    benchmark_max_dd = max_drawdown(benchmark_returns)

    one_year_return = calculate_one_year_return(prices, TICKER)
    benchmark_one_year_return = calculate_one_year_return(prices, BENCHMARK)

    active_return = annualized_stock_return - annualized_benchmark_return
    tracking_error_value = tracking_error(stock_returns, benchmark_returns)
    info_ratio = information_ratio(
        annualized_stock_return,
        annualized_benchmark_return,
        tracking_error_value,
    )

    metrics = pd.DataFrame(
        {
            "Metric": [
                "Latest Price",
                f"Latest {BENCHMARK_NAME} Index Level",
                "Total Return",
                f"{BENCHMARK_NAME} Total Return",
                "One-Year Return",
                f"{BENCHMARK_NAME} One-Year Return",
                "Annualized Return",
                f"{BENCHMARK_NAME} Annualized Return",
                "Annualized Volatility",
                f"{BENCHMARK_NAME} Annualized Volatility",
                "Sharpe Ratio",
                f"{BENCHMARK_NAME} Sharpe Ratio",
                "Sortino Ratio",
                f"Beta vs {BENCHMARK_NAME}",
                f"Correlation vs {BENCHMARK_NAME}",
                "Maximum Drawdown",
                f"{BENCHMARK_NAME} Maximum Drawdown",
                "Active Annualized Return",
                "Tracking Error",
                "Information Ratio",
                "Risk-Free Rate",
            ],
            "Value": [
                latest_price,
                latest_benchmark_index_level,
                total_return,
                benchmark_total_return,
                one_year_return,
                benchmark_one_year_return,
                annualized_stock_return,
                annualized_benchmark_return,
                stock_volatility,
                benchmark_volatility,
                sharpe_ratio,
                benchmark_sharpe,
                sortino,
                beta,
                correlation,
                max_dd,
                benchmark_max_dd,
                active_return,
                tracking_error_value,
                info_ratio,
                risk_free_rate,
            ],
        }
    )

    metrics["Value Type"] = metrics["Metric"].apply(_metric_value_type)

    _safe_to_csv(metrics, "market_metrics.csv")

    market_snapshot = build_market_snapshot(metrics)
    _safe_to_csv(market_snapshot, "market_snapshot.csv")

    return metrics


def build_market_snapshot(metrics: pd.DataFrame) -> pd.DataFrame:
    """Build a compact market snapshot for the PDF report."""

    metric_map = {
        row["Metric"]: row["Value"]
        for _, row in metrics.iterrows()
    }

    latest_price = _safe_float(metric_map.get("Latest Price"))
    one_year_return = _safe_float(metric_map.get("One-Year Return"))
    annualized_volatility_value = _safe_float(metric_map.get("Annualized Volatility"))
    max_drawdown_value = _safe_float(metric_map.get("Maximum Drawdown"))
    sharpe_ratio_value = _safe_float(metric_map.get("Sharpe Ratio"))
    beta_value = _safe_float(metric_map.get(f"Beta vs {BENCHMARK_NAME}"))
    correlation_value = _safe_float(metric_map.get(f"Correlation vs {BENCHMARK_NAME}"))
    active_return_value = _safe_float(metric_map.get("Active Annualized Return"))
    information_ratio_value = _safe_float(metric_map.get("Information Ratio"))
    risk_free_rate_value = _safe_float(metric_map.get("Risk-Free Rate"))

    rows = [
        {
            "Category": "Price",
            "Metric": "Latest Price",
            "Value": latest_price,
            "Value Type": "Money",
            "Signal": "Reference",
            "Interpretation": "Current market reference point for valuation comparison.",
        },
        {
            "Category": "Return",
            "Metric": "One-Year Return",
            "Value": one_year_return,
            "Value Type": "Percent",
            "Signal": _return_signal(one_year_return),
            "Interpretation": "Recent stock performance over approximately one trading year.",
        },
        {
            "Category": "Risk",
            "Metric": "Annualized Volatility",
            "Value": annualized_volatility_value,
            "Value Type": "Percent",
            "Signal": _risk_signal(annualized_volatility_value, lower_is_better=True),
            "Interpretation": "Annualized daily return volatility.",
        },
        {
            "Category": "Risk",
            "Metric": "Maximum Drawdown",
            "Value": max_drawdown_value,
            "Value Type": "Percent",
            "Signal": _drawdown_signal(max_drawdown_value),
            "Interpretation": "Largest peak-to-trough decline during the study period.",
        },
        {
            "Category": "Risk-Adjusted Return",
            "Metric": "Sharpe Ratio",
            "Value": sharpe_ratio_value,
            "Value Type": "Plain Number",
            "Signal": _sharpe_signal(sharpe_ratio_value),
            "Interpretation": "Excess return per unit of volatility.",
        },
        {
            "Category": "Benchmark Sensitivity",
            "Metric": f"Beta vs {BENCHMARK_NAME}",
            "Value": beta_value,
            "Value Type": "Plain Number",
            "Signal": _beta_signal(beta_value),
            "Interpretation": f"Sensitivity of CACI daily returns versus the {BENCHMARK_NAME}.",
        },
        {
            "Category": "Benchmark Sensitivity",
            "Metric": f"Correlation vs {BENCHMARK_NAME}",
            "Value": correlation_value,
            "Value Type": "Plain Number",
            "Signal": _correlation_signal(correlation_value),
            "Interpretation": f"Correlation of CACI daily returns versus the {BENCHMARK_NAME}.",
        },
        {
            "Category": "Benchmark Comparison",
            "Metric": "Active Annualized Return",
            "Value": active_return_value,
            "Value Type": "Percent",
            "Signal": _return_signal(active_return_value),
            "Interpretation": f"CACI annualized return minus {BENCHMARK_NAME} annualized return.",
        },
        {
            "Category": "Benchmark Comparison",
            "Metric": "Information Ratio",
            "Value": information_ratio_value,
            "Value Type": "Plain Number",
            "Signal": _information_ratio_signal(information_ratio_value),
            "Interpretation": "Active return relative to tracking error.",
        },
        {
            "Category": "Discount Rate Input",
            "Metric": "Risk-Free Rate",
            "Value": risk_free_rate_value,
            "Value Type": "Percent",
            "Signal": _risk_free_rate_signal(risk_free_rate_value),
            "Interpretation": "Live 10-year Treasury proxy or fallback default.",
        },
    ]

    return pd.DataFrame(rows)