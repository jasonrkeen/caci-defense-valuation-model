import numpy as np
import pandas as pd
import yfinance as yf

from src.config import (
    TICKER,
    PEERS,
    TABLES_DIR,
)


def _safe_to_csv(df: pd.DataFrame, file_name: str) -> None:
    """Save a DataFrame to outputs/tables with basic safety."""

    try:
        df.to_csv(TABLES_DIR / file_name, index=False)
    except Exception as exc:
        print(f"Warning: could not save {file_name}: {exc}")


def _safe_get(info: dict, key: str, default=np.nan):
    """Safely pull a value from a Yahoo Finance info dictionary."""

    if not isinstance(info, dict):
        return default

    value = info.get(key, default)

    if value in [None, "None", "nan", "NaN", ""]:
        return default

    try:
        return float(value)
    except Exception:
        return value


def _safe_divide(numerator, denominator, default=np.nan) -> float:
    """Safely divide two values."""

    try:
        numerator = float(numerator)
        denominator = float(denominator)

        if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
            return default

        return numerator / denominator
    except Exception:
        return default


def _safe_numeric(series: pd.Series) -> pd.Series:
    """Convert a Series to numeric safely."""

    return pd.to_numeric(series, errors="coerce")


def _metric_value_type(metric: str) -> str:
    """Assign formatting metadata by metric name."""

    metric_lower = str(metric).lower()

    money_metrics = [
        "market cap",
        "enterprise value",
        "free cash flow",
        "total debt",
        "cash",
        "net debt",
        "current price",
        "target mean price",
    ]

    percent_metrics = [
        "profit margin",
        "operating margin",
        "ebitda margin",
        "revenue growth",
        "fcf yield",
        "target upside",
        "premium / discount",
    ]

    multiple_metrics = [
        "trailing p/e",
        "forward p/e",
        "peg ratio",
        "price/sales",
        "ev/revenue",
        "ev/ebitda",
    ]

    plain_number_metrics = [
        "beta",
        "average rank",
        "overall peer rank",
    ]

    if metric_lower in money_metrics:
        return "Money"

    if metric_lower in percent_metrics:
        return "Percent"

    if metric_lower in multiple_metrics:
        return "Multiple"

    if metric_lower in plain_number_metrics:
        return "Plain Number"

    return "Plain Number"


def _valuation_signal(metric: str, premium_discount: float) -> str:
    """Convert premium / discount into a signal based on metric type."""

    if pd.isna(premium_discount):
        return "Unavailable"

    metric_lower = str(metric).lower()

    lower_is_better_metrics = [
        "trailing p/e",
        "forward p/e",
        "peg ratio",
        "price/sales",
        "ev/revenue",
        "ev/ebitda",
        "beta",
    ]

    higher_is_better_metrics = [
        "profit margin",
        "operating margin",
        "ebitda margin",
        "revenue growth",
        "fcf yield",
        "target upside",
    ]

    if metric_lower in lower_is_better_metrics:
        if premium_discount <= -0.20:
            return "Discount / Attractive"
        if premium_discount <= -0.05:
            return "Modest Discount"
        if premium_discount <= 0.10:
            return "Near Peer Median"
        if premium_discount <= 0.25:
            return "Premium / Monitor"
        return "High Premium"

    if metric_lower in higher_is_better_metrics:
        if premium_discount >= 0.25:
            return "Strong Premium"
        if premium_discount >= 0.10:
            return "Constructive Premium"
        if premium_discount >= -0.10:
            return "Near Peer Median"
        if premium_discount >= -0.25:
            return "Below Peer Median"
        return "Weak Relative Position"

    return "Reference"


def _ranking_signal(average_rank: float, peer_count: int) -> str:
    """Convert average peer rank into a plain-English signal."""

    if pd.isna(average_rank) or peer_count <= 0:
        return "Unavailable"

    percentile = average_rank / peer_count

    if percentile <= 0.25:
        return "Top Quartile"
    if percentile <= 0.50:
        return "Above Median"
    if percentile <= 0.75:
        return "Below Median"
    return "Bottom Quartile"


def _recommendation_label(value) -> str:
    """Normalize Yahoo Finance recommendation values."""

    if value is None or pd.isna(value):
        return ""

    value = str(value).strip()

    if not value:
        return ""

    return value.replace("_", " ").title()


def _empty_peer_row(ticker: str, company: str) -> dict:
    """Return an empty peer row if Yahoo Finance data is unavailable."""

    return {
        "Ticker": ticker,
        "Company": company,
        "Market Cap": np.nan,
        "Enterprise Value": np.nan,
        "Trailing P/E": np.nan,
        "Forward P/E": np.nan,
        "PEG Ratio": np.nan,
        "Price/Sales": np.nan,
        "EV/Revenue": np.nan,
        "EV/EBITDA": np.nan,
        "Profit Margin": np.nan,
        "Operating Margin": np.nan,
        "EBITDA Margin": np.nan,
        "Revenue Growth": np.nan,
        "Free Cash Flow": np.nan,
        "Total Debt": np.nan,
        "Cash": np.nan,
        "Beta": np.nan,
        "Current Price": np.nan,
        "Target Mean Price": np.nan,
        "Recommendation": "",
        "Data Status": "Unavailable",
    }


def _collect_single_ticker(ticker: str, company: str) -> dict:
    """Collect peer valuation metrics for one ticker."""

    try:
        obj = yf.Ticker(ticker)
        info = obj.info

        if not isinstance(info, dict) or len(info) == 0:
            return _empty_peer_row(ticker, company)

        current_price = _safe_get(info, "currentPrice")

        if pd.isna(current_price):
            current_price = _safe_get(info, "regularMarketPrice")

        target_mean_price = _safe_get(info, "targetMeanPrice")

        return {
            "Ticker": ticker,
            "Company": company,
            "Market Cap": _safe_get(info, "marketCap"),
            "Enterprise Value": _safe_get(info, "enterpriseValue"),
            "Trailing P/E": _safe_get(info, "trailingPE"),
            "Forward P/E": _safe_get(info, "forwardPE"),
            "PEG Ratio": _safe_get(info, "pegRatio"),
            "Price/Sales": _safe_get(info, "priceToSalesTrailing12Months"),
            "EV/Revenue": _safe_get(info, "enterpriseToRevenue"),
            "EV/EBITDA": _safe_get(info, "enterpriseToEbitda"),
            "Profit Margin": _safe_get(info, "profitMargins"),
            "Operating Margin": _safe_get(info, "operatingMargins"),
            "EBITDA Margin": _safe_get(info, "ebitdaMargins"),
            "Revenue Growth": _safe_get(info, "revenueGrowth"),
            "Free Cash Flow": _safe_get(info, "freeCashflow"),
            "Total Debt": _safe_get(info, "totalDebt"),
            "Cash": _safe_get(info, "totalCash"),
            "Beta": _safe_get(info, "beta"),
            "Current Price": current_price,
            "Target Mean Price": target_mean_price,
            "Recommendation": _recommendation_label(info.get("recommendationKey", "")),
            "Data Status": "Collected",
        }

    except Exception as exc:
        print(f"Warning: could not collect peer data for {ticker}: {exc}")
        return _empty_peer_row(ticker, company)


def _add_peer_comparison_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Add formatting metadata to the peer comparison export."""

    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    type_map = {
        "Market Cap": "Money",
        "Enterprise Value": "Money",
        "Trailing P/E": "Multiple",
        "Forward P/E": "Multiple",
        "PEG Ratio": "Multiple",
        "Price/Sales": "Multiple",
        "EV/Revenue": "Multiple",
        "EV/EBITDA": "Multiple",
        "Profit Margin": "Percent",
        "Operating Margin": "Percent",
        "EBITDA Margin": "Percent",
        "Revenue Growth": "Percent",
        "Free Cash Flow": "Money",
        "Total Debt": "Money",
        "Cash": "Money",
        "Net Debt": "Money",
        "FCF Yield": "Percent",
        "Beta": "Plain Number",
        "Current Price": "Money",
        "Target Mean Price": "Money",
        "Target Upside": "Percent",
    }

    for column, value_type in type_map.items():
        if column in result.columns:
            result[f"{column} Type"] = value_type

    return result


def collect_peer_data() -> pd.DataFrame:
    """Collect valuation and operating metrics for CACI and its peer group."""

    tickers = {TICKER: "CACI International"} | PEERS

    rows = [
        _collect_single_ticker(ticker, company)
        for ticker, company in tickers.items()
    ]

    df = pd.DataFrame(rows)

    numeric_columns = [
        "Market Cap",
        "Enterprise Value",
        "Trailing P/E",
        "Forward P/E",
        "PEG Ratio",
        "Price/Sales",
        "EV/Revenue",
        "EV/EBITDA",
        "Profit Margin",
        "Operating Margin",
        "EBITDA Margin",
        "Revenue Growth",
        "Free Cash Flow",
        "Total Debt",
        "Cash",
        "Beta",
        "Current Price",
        "Target Mean Price",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = _safe_numeric(df[column])

    df["Net Debt"] = df["Total Debt"] - df["Cash"]
    df["FCF Yield"] = df.apply(
        lambda row: _safe_divide(row.get("Free Cash Flow"), row.get("Market Cap")),
        axis=1,
    )
    df["Target Upside"] = df.apply(
        lambda row: (
            _safe_divide(row.get("Target Mean Price"), row.get("Current Price")) - 1
            if pd.notna(_safe_divide(row.get("Target Mean Price"), row.get("Current Price")))
            else np.nan
        ),
        axis=1,
    )

    df["Peer Group"] = np.where(df["Ticker"] == TICKER, "Target Company", "Peer")

    output_columns = [
        "Ticker",
        "Company",
        "Peer Group",
        "Market Cap",
        "Enterprise Value",
        "Trailing P/E",
        "Forward P/E",
        "PEG Ratio",
        "Price/Sales",
        "EV/Revenue",
        "EV/EBITDA",
        "Profit Margin",
        "Operating Margin",
        "EBITDA Margin",
        "Revenue Growth",
        "Free Cash Flow",
        "Total Debt",
        "Cash",
        "Net Debt",
        "FCF Yield",
        "Beta",
        "Current Price",
        "Target Mean Price",
        "Target Upside",
        "Recommendation",
        "Data Status",
    ]

    df = df[output_columns]

    df_for_export = _add_peer_comparison_metadata(df)
    _safe_to_csv(df_for_export, "peer_comparison.csv")

    peer_summary = build_peer_summary(df)
    _safe_to_csv(peer_summary, "peer_summary.csv")

    ranking = build_peer_ranking_table(df)
    _safe_to_csv(ranking, "peer_ranking.csv")

    positioning_summary = build_peer_positioning_summary(df, peer_summary, ranking)
    _safe_to_csv(positioning_summary, "peer_positioning_summary.csv")

    return df


def _peer_summary_interpretation(
    metric: str,
    caci_value: float,
    peer_median: float,
    premium_discount: float,
    signal: str,
) -> str:
    """Create concise interpretation language for peer summary rows."""

    if pd.isna(caci_value) or pd.isna(peer_median):
        return f"{metric} comparison is unavailable due to missing Yahoo Finance data."

    metric_lower = metric.lower()

    if metric_lower in [
        "ev/ebitda",
        "forward p/e",
        "trailing p/e",
        "price/sales",
        "ev/revenue",
    ]:
        return (
            f"CACI trades at a {premium_discount:.1%} premium/discount to the peer median "
            f"on {metric}; signal: {signal}."
        )

    if metric_lower in [
        "revenue growth",
        "ebitda margin",
        "operating margin",
        "profit margin",
        "fcf yield",
    ]:
        return (
            f"CACI is {premium_discount:.1%} above/below the peer median on {metric}; "
            f"signal: {signal}."
        )

    if metric_lower == "beta":
        return (
            f"CACI's beta is {premium_discount:.1%} above/below the peer median; "
            f"signal: {signal}."
        )

    if metric_lower == "target upside":
        return (
            f"CACI's implied analyst target upside is {premium_discount:.1%} above/below "
            f"the peer median; signal: {signal}."
        )

    return f"CACI's {metric} is compared against the peer median; signal: {signal}."


def build_peer_summary(peer_df: pd.DataFrame) -> pd.DataFrame:
    """Build a CACI versus peer median valuation summary."""

    numeric_metrics = [
        "Market Cap",
        "Enterprise Value",
        "Trailing P/E",
        "Forward P/E",
        "PEG Ratio",
        "Price/Sales",
        "EV/Revenue",
        "EV/EBITDA",
        "Profit Margin",
        "Operating Margin",
        "EBITDA Margin",
        "Revenue Growth",
        "FCF Yield",
        "Beta",
        "Target Upside",
    ]

    if peer_df is None or peer_df.empty:
        return pd.DataFrame(
            [
                {
                    "Metric": "No peer data",
                    "CACI": np.nan,
                    "CACI Type": "Plain Number",
                    "Peer Median": np.nan,
                    "Peer Median Type": "Plain Number",
                    "Premium / Discount": np.nan,
                    "Premium / Discount Type": "Percent",
                    "Signal": "Unavailable",
                    "Interpretation": "Peer comparison data was unavailable.",
                }
            ]
        )

    caci_row = peer_df.loc[peer_df["Ticker"] == TICKER]

    if caci_row.empty:
        return pd.DataFrame(
            [
                {
                    "Metric": "No CACI data",
                    "CACI": np.nan,
                    "CACI Type": "Plain Number",
                    "Peer Median": np.nan,
                    "Peer Median Type": "Plain Number",
                    "Premium / Discount": np.nan,
                    "Premium / Discount Type": "Percent",
                    "Signal": "Unavailable",
                    "Interpretation": "CACI row was not available in peer data.",
                }
            ]
        )

    caci_row = caci_row.iloc[0]
    peer_only = peer_df.loc[peer_df["Ticker"] != TICKER].copy()

    rows = []

    for metric in numeric_metrics:
        if metric not in peer_df.columns:
            continue

        caci_value = _safe_divide(caci_row.get(metric), 1)
        peer_median = _safe_numeric(peer_only[metric]).median(skipna=True)

        if pd.notna(caci_value) and pd.notna(peer_median) and peer_median != 0:
            premium_discount = caci_value / peer_median - 1
        else:
            premium_discount = np.nan

        signal = _valuation_signal(metric, premium_discount)
        value_type = _metric_value_type(metric)

        rows.append(
            {
                "Metric": metric,
                "CACI": caci_value,
                "CACI Type": value_type,
                "Peer Median": peer_median,
                "Peer Median Type": value_type,
                "Premium / Discount": premium_discount,
                "Premium / Discount Type": "Percent",
                "Signal": signal,
                "Interpretation": _peer_summary_interpretation(
                    metric=metric,
                    caci_value=caci_value,
                    peer_median=peer_median,
                    premium_discount=premium_discount,
                    signal=signal,
                ),
            }
        )

    summary = pd.DataFrame(rows)
    return summary


def build_peer_ranking_table(peer_df: pd.DataFrame) -> pd.DataFrame:
    """Rank companies on selected valuation and quality metrics.

    Lower valuation multiples are ranked better.
    Higher growth, margins, FCF yield, and target upside are ranked better.
    Beta is ranked by distance from 1.0, favoring market-like risk.
    """

    if peer_df is None or peer_df.empty:
        return pd.DataFrame(
            [
                {
                    "Overall Peer Rank": np.nan,
                    "Overall Peer Rank Type": "Plain Number",
                    "Ticker": "",
                    "Company": "",
                    "Peer Group": "",
                    "Average Rank": np.nan,
                    "Average Rank Type": "Plain Number",
                    "Peer Ranking Signal": "Unavailable",
                }
            ]
        )

    base_columns = ["Ticker", "Company"]

    if "Peer Group" in peer_df.columns:
        base_columns.append("Peer Group")

    ranking = peer_df[base_columns].copy()

    metric_specs = {
        "EV/EBITDA": "lower",
        "Forward P/E": "lower",
        "EV/Revenue": "lower",
        "Revenue Growth": "higher",
        "EBITDA Margin": "higher",
        "FCF Yield": "higher",
        "Target Upside": "higher",
    }

    for metric, direction in metric_specs.items():
        if metric not in peer_df.columns:
            continue

        values = _safe_numeric(peer_df[metric])

        ranking[metric] = values
        ranking[f"{metric} Type"] = _metric_value_type(metric)

        if direction == "lower":
            ranking[f"{metric} Rank"] = values.rank(
                ascending=True,
                method="min",
                na_option="bottom",
            )
        else:
            ranking[f"{metric} Rank"] = values.rank(
                ascending=False,
                method="min",
                na_option="bottom",
            )

        ranking[f"{metric} Rank Type"] = "Plain Number"

    if "Beta" in peer_df.columns:
        beta_values = _safe_numeric(peer_df["Beta"])
        ranking["Beta"] = beta_values
        ranking["Beta Type"] = "Plain Number"
        ranking["Beta Distance From 1.0"] = (beta_values - 1.0).abs()
        ranking["Beta Distance From 1.0 Type"] = "Plain Number"
        ranking["Beta Rank"] = ranking["Beta Distance From 1.0"].rank(
            ascending=True,
            method="min",
            na_option="bottom",
        )
        ranking["Beta Rank Type"] = "Plain Number"

    rank_columns = [
        col
        for col in ranking.columns
        if col.endswith("Rank") and not col.endswith("Type")
    ]

    if rank_columns:
        ranking["Average Rank"] = ranking[rank_columns].mean(axis=1)
    else:
        ranking["Average Rank"] = np.nan

    ranking["Average Rank Type"] = "Plain Number"

    peer_count = len(ranking)
    ranking["Peer Ranking Signal"] = ranking["Average Rank"].apply(
        lambda value: _ranking_signal(value, peer_count)
    )

    ranking = ranking.sort_values("Average Rank", na_position="last").reset_index(drop=True)
    ranking["Overall Peer Rank"] = ranking.index + 1
    ranking["Overall Peer Rank Type"] = "Plain Number"

    output_columns = [
        "Overall Peer Rank",
        "Overall Peer Rank Type",
        "Ticker",
        "Company",
    ]

    if "Peer Group" in ranking.columns:
        output_columns.append("Peer Group")

    remaining_columns = [
        col for col in ranking.columns
        if col not in output_columns
    ]

    ranking = ranking[output_columns + remaining_columns]

    _safe_to_csv(ranking, "peer_ranking.csv")

    return ranking


def build_peer_positioning_summary(
    peer_df: pd.DataFrame | None = None,
    peer_summary: pd.DataFrame | None = None,
    ranking: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact CACI peer-positioning summary table."""

    if peer_df is None or peer_df.empty:
        summary = pd.DataFrame(
            [
                {
                    "Metric": "Peer Positioning",
                    "Value": "Unavailable",
                    "Value Type": "Text",
                    "Signal": "Unavailable",
                    "Interpretation": "Peer data was unavailable.",
                }
            ]
        )
        return summary

    if peer_summary is None:
        peer_summary = build_peer_summary(peer_df)

    if ranking is None:
        ranking = build_peer_ranking_table(peer_df)

    caci_rank_row = (
        ranking.loc[ranking["Ticker"] == TICKER]
        if ranking is not None and not ranking.empty and "Ticker" in ranking.columns
        else pd.DataFrame()
    )

    if not caci_rank_row.empty:
        caci_rank = caci_rank_row["Overall Peer Rank"].iloc[0]
        average_rank = caci_rank_row["Average Rank"].iloc[0]
        ranking_signal = caci_rank_row["Peer Ranking Signal"].iloc[0]
    else:
        caci_rank = np.nan
        average_rank = np.nan
        ranking_signal = "Unavailable"

    peer_count = len(peer_df)

    def _summary_value(metric_name: str):
        if peer_summary is None or peer_summary.empty:
            return np.nan

        match = peer_summary.loc[peer_summary["Metric"] == metric_name]

        if match.empty:
            return np.nan

        return match.iloc[0].get("Premium / Discount", np.nan)

    ev_ebitda_premium = _summary_value("EV/EBITDA")
    forward_pe_premium = _summary_value("Forward P/E")
    revenue_growth_premium = _summary_value("Revenue Growth")
    ebitda_margin_premium = _summary_value("EBITDA Margin")
    fcf_yield_premium = _summary_value("FCF Yield")
    target_upside_premium = _summary_value("Target Upside")

    rows = [
        {
            "Metric": "CACI Overall Peer Rank",
            "Value": caci_rank,
            "Value Type": "Plain Number",
            "Signal": ranking_signal,
            "Interpretation": (
                f"CACI ranks {caci_rank} out of {peer_count} companies on the composite peer ranking."
                if pd.notna(caci_rank)
                else "CACI composite peer rank is unavailable."
            ),
        },
        {
            "Metric": "CACI Average Peer Rank",
            "Value": average_rank,
            "Value Type": "Plain Number",
            "Signal": ranking_signal,
            "Interpretation": (
                "Average rank across valuation, growth, margin, FCF yield, "
                "target upside, and beta metrics."
            ),
        },
        {
            "Metric": "EV/EBITDA Premium / Discount",
            "Value": ev_ebitda_premium,
            "Value Type": "Percent",
            "Signal": _valuation_signal("EV/EBITDA", ev_ebitda_premium),
            "Interpretation": "CACI's EV/EBITDA multiple versus the peer median.",
        },
        {
            "Metric": "Forward P/E Premium / Discount",
            "Value": forward_pe_premium,
            "Value Type": "Percent",
            "Signal": _valuation_signal("Forward P/E", forward_pe_premium),
            "Interpretation": "CACI's forward P/E multiple versus the peer median.",
        },
        {
            "Metric": "Revenue Growth Premium / Discount",
            "Value": revenue_growth_premium,
            "Value Type": "Percent",
            "Signal": _valuation_signal("Revenue Growth", revenue_growth_premium),
            "Interpretation": "CACI's revenue growth versus the peer median.",
        },
        {
            "Metric": "EBITDA Margin Premium / Discount",
            "Value": ebitda_margin_premium,
            "Value Type": "Percent",
            "Signal": _valuation_signal("EBITDA Margin", ebitda_margin_premium),
            "Interpretation": "CACI's EBITDA margin versus the peer median.",
        },
        {
            "Metric": "FCF Yield Premium / Discount",
            "Value": fcf_yield_premium,
            "Value Type": "Percent",
            "Signal": _valuation_signal("FCF Yield", fcf_yield_premium),
            "Interpretation": "CACI's free-cash-flow yield versus the peer median.",
        },
        {
            "Metric": "Target Upside Premium / Discount",
            "Value": target_upside_premium,
            "Value Type": "Percent",
            "Signal": _valuation_signal("Target Upside", target_upside_premium),
            "Interpretation": "CACI's analyst target upside versus the peer median.",
        },
    ]

    summary = pd.DataFrame(rows)
    _safe_to_csv(summary, "peer_positioning_summary.csv")

    return summary