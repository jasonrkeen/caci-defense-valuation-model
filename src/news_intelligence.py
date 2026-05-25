import re
import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import parsedate_to_datetime
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests

from src.config import (
    TABLES_DIR,
    NEWS_SENTIMENT_WEIGHTS,
    NEWS_QUERIES,
    GOOGLE_NEWS_RSS_BASE,
    YAHOO_FINANCE_RSS_URL,
    NEWS_MAX_ITEMS_PER_QUERY,
    NEWS_MAX_LIVE_ROWS,
)


POSITIVE_KEYWORDS = [
    "award",
    "awards",
    "wins",
    "selected",
    "contract",
    "guidance raised",
    "raises guidance",
    "record",
    "growth",
    "beats",
    "strong",
    "backlog",
    "book-to-bill",
    "free cash flow",
    "margin expansion",
    "acquisition completed",
    "modernization",
    "cyber",
    "intelligence",
    "space",
    "electronic warfare",
    "national security",
]

NEGATIVE_KEYWORDS = [
    "miss",
    "misses",
    "cut guidance",
    "guidance cut",
    "shutdown",
    "delay",
    "delayed",
    "protest",
    "recompete loss",
    "lost contract",
    "margin pressure",
    "cost overrun",
    "investigation",
    "downgrade",
    "budget cuts",
    "continuing resolution",
    "appropriations delay",
]

MIXED_KEYWORDS = [
    "mixed",
    "uncertain",
    "monitor",
    "risk",
    "pressure",
    "flat",
    "soft",
    "challenging",
    "volatile",
]

THEME_KEYWORDS = {
    "Contract Award": [
        "contract",
        "award",
        "task order",
        "blanket purchase agreement",
        "idiq",
        "selected",
        "wins",
    ],
    "Earnings / Guidance": [
        "earnings",
        "revenue",
        "ebitda",
        "eps",
        "guidance",
        "free cash flow",
        "margin",
        "quarter",
        "fiscal",
    ],
    "Backlog / Book-to-Bill": [
        "backlog",
        "book-to-bill",
        "awards",
        "pipeline",
        "submitted bids",
    ],
    "Acquisition / M&A": [
        "acquisition",
        "acquires",
        "merger",
        "m&a",
        "divestiture",
        "integration",
    ],
    "Defense Budget": [
        "defense budget",
        "appropriations",
        "defense spending",
        "pentagon",
        "dod budget",
        "continuing resolution",
        "shutdown",
    ],
    "Intelligence / Cyber": [
        "intelligence",
        "cyber",
        "signals",
        "classified",
        "national security",
        "odni",
        "mip",
        "nip",
    ],
    "Geopolitical Conflict": [
        "ukraine",
        "russia",
        "china",
        "taiwan",
        "iran",
        "middle east",
        "israel",
        "nato",
        "indo-pacific",
        "great-power",
        "conflict",
    ],
    "Technology / Modernization": [
        "cloud",
        "ai",
        "artificial intelligence",
        "data",
        "modernization",
        "software",
        "space",
        "electronic warfare",
        "c5isr",
        "mission technology",
    ],
    "Analyst / Valuation": [
        "price target",
        "upgrade",
        "downgrade",
        "rating",
        "valuation",
        "multiple",
        "analyst",
        "outperform",
        "neutral",
        "buy",
        "sell",
    ],
    "Execution / Risk": [
        "delay",
        "protest",
        "recompete",
        "loss",
        "risk",
        "margin pressure",
        "cost",
        "investigation",
        "labor",
        "talent",
    ],
}


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    )
}


NEWS_COLUMNS = [
    "Date",
    "Source",
    "Title",
    "Summary",
    "URL / Reference",
    "Collection Method",
]


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


def _empty_news_frame() -> pd.DataFrame:
    """Return an empty news input table with standard columns."""

    return pd.DataFrame(columns=NEWS_COLUMNS)


def _normalize_text(text: str) -> str:
    """Normalize text for classification and deduplication."""

    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _clean_rss_text(value: str) -> str:
    """Clean RSS title/description text."""

    if value is None:
        return ""

    text = str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&amp;", "&")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def _parse_rss_date(value: str) -> str:
    """Parse RSS pubDate into YYYY-MM-DD when possible."""

    if not value:
        return datetime.now().strftime("%Y-%m-%d")

    try:
        parsed = parsedate_to_datetime(value)
        return parsed.strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def _fetch_rss_items(url: str, source_label: str, max_items: int = 10) -> pd.DataFrame:
    """Fetch RSS items from a public RSS feed."""

    rows = []

    try:
        response = requests.get(
            url,
            headers=REQUEST_HEADERS,
            timeout=12,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        items = root.findall(".//item")

        for item in items[:max_items]:
            title = _clean_rss_text(item.findtext("title", default=""))
            summary = _clean_rss_text(item.findtext("description", default=""))
            link = _clean_rss_text(item.findtext("link", default=""))
            pub_date = _parse_rss_date(item.findtext("pubDate", default=""))

            if not title:
                continue

            rows.append(
                {
                    "Date": pub_date,
                    "Source": source_label,
                    "Title": title,
                    "Summary": summary,
                    "URL / Reference": link,
                    "Collection Method": "Live RSS",
                }
            )

    except Exception as exc:
        print(f"Warning: could not fetch RSS source {source_label}: {exc}")

    if not rows:
        return _empty_news_frame()

    return pd.DataFrame(rows)


def fetch_google_news_rss(
    query: str,
    max_items: int = NEWS_MAX_ITEMS_PER_QUERY,
) -> pd.DataFrame:
    """Fetch Google News RSS results for a search query."""

    encoded_query = quote_plus(query)
    url = (
        f"{GOOGLE_NEWS_RSS_BASE}"
        f"?q={encoded_query}"
        f"&hl=en-US&gl=US&ceid=US:en"
    )

    return _fetch_rss_items(
        url=url,
        source_label=f"Google News RSS: {query}",
        max_items=max_items,
    )


def fetch_yahoo_finance_rss(max_items: int = 10) -> pd.DataFrame:
    """Fetch Yahoo Finance RSS headlines for CACI."""

    return _fetch_rss_items(
        url=YAHOO_FINANCE_RSS_URL,
        source_label="Yahoo Finance RSS: CACI",
        max_items=max_items,
    )


def deduplicate_news(news_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate news rows by normalized title."""

    if news_df is None or news_df.empty:
        return _empty_news_frame()

    df = news_df.copy()

    for column in NEWS_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    df = df[NEWS_COLUMNS]

    df["Title Clean"] = df["Title"].astype(str).map(_normalize_text)
    df = df[df["Title Clean"] != ""]
    df = df.drop_duplicates(subset=["Title Clean"], keep="first")
    df = df.drop(columns=["Title Clean"])

    if "Date" in df.columns:
        try:
            df = df.sort_values("Date", ascending=False)
        except Exception:
            pass

    return df.reset_index(drop=True)


def collect_live_news_inputs(
    max_items_per_query: int = NEWS_MAX_ITEMS_PER_QUERY,
    max_live_rows: int = NEWS_MAX_LIVE_ROWS,
) -> pd.DataFrame:
    """Collect live API-free news inputs from public RSS sources."""

    frames = []

    yahoo_news = fetch_yahoo_finance_rss(max_items=max_items_per_query)

    if not yahoo_news.empty:
        frames.append(yahoo_news)

    for query in NEWS_QUERIES:
        google_news = fetch_google_news_rss(
            query=query,
            max_items=max_items_per_query,
        )

        if not google_news.empty:
            frames.append(google_news)

    if not frames:
        live_news = _empty_news_frame()
    else:
        live_news = pd.concat(frames, ignore_index=True)

    live_news = deduplicate_news(live_news)

    if not live_news.empty:
        live_news = live_news.head(max_live_rows)

    _safe_to_csv(live_news, "live_news_inputs.csv")

    return live_news


def build_manual_news_input_table() -> pd.DataFrame:
    """Build a manual fallback news table.

    This keeps the project usable without any live web source. These rows also
    preserve the core valuation thesis when RSS feeds fail or return noisy data.
    """

    rows = [
        {
            "Date": "2026-04-23",
            "Source": "CACI Investor Relations",
            "Title": "CACI reports fiscal 2026 third-quarter results and raises FY2026 guidance",
            "Summary": (
                "CACI reported revenue growth, adjusted EBITDA performance, contract awards, "
                "free cash flow guidance, and updated FY2026 revenue guidance."
            ),
            "URL / Reference": "CACI Q3 FY2026 earnings release",
            "Collection Method": "Manual Source Input",
        },
        {
            "Date": "2026-04-23",
            "Source": "CACI Investor Relations",
            "Title": "CACI reports Q3 FY2026 contract awards and book-to-bill momentum",
            "Summary": (
                "Contract awards and backlog indicators support revenue visibility "
                "and the defense technology thesis."
            ),
            "URL / Reference": "CACI Q3 FY2026 earnings release and earnings slides",
            "Collection Method": "Manual Source Input",
        },
        {
            "Date": "2026-01-01",
            "Source": "CACI Investor Presentation",
            "Title": "CACI outlines three-year financial framework with high-single-digit revenue CAGR target",
            "Summary": (
                "Management framework supports revenue growth, free cash flow expansion, "
                "and technology-focused positioning."
            ),
            "URL / Reference": "CACI January 2026 investor presentation",
            "Collection Method": "Manual Source Input",
        },
        {
            "Date": "2026-01-01",
            "Source": "CACI Investor Presentation",
            "Title": "CACI highlights defense technology, intelligence, cyber, and national-security demand",
            "Summary": (
                "Technology and expertise mix support CACI's positioning as a "
                "national-security technology contractor."
            ),
            "URL / Reference": "CACI January 2026 investor presentation",
            "Collection Method": "Manual Source Input",
        },
        {
            "Date": "2026-01-01",
            "Source": "Government / Public Budget Sources",
            "Title": "U.S. defense and intelligence funding remains a central demand driver for contractors",
            "Summary": (
                "Defense appropriations, NIP, MIP, cyber, space, and intelligence funding "
                "support the broader demand environment."
            ),
            "URL / Reference": "Government spending and intelligence spending modules",
            "Collection Method": "Manual Source Input",
        },
        {
            "Date": "2026-01-01",
            "Source": "Macro / Defense Sources",
            "Title": "Global conflict, great-power competition, cyber threats, and intelligence demand support defense technology spending",
            "Summary": (
                "Geopolitical and cyber threat conditions support continued investment "
                "in national-security technology."
            ),
            "URL / Reference": "Defense macro module",
            "Collection Method": "Manual Source Input",
        },
        {
            "Date": "2026-01-01",
            "Source": "Model Risk Monitor",
            "Title": "Federal budget timing, recompete risk, and cleared labor constraints remain monitoring items",
            "Summary": (
                "CACI's long-term demand backdrop remains supportive, but appropriations timing, "
                "contract recompete pressure, and cleared labor availability remain key execution risks."
            ),
            "URL / Reference": "Defense risk engine and FY2025 Form 10-K risk factors",
            "Collection Method": "Manual Risk Input",
        },
    ]

    df = pd.DataFrame(rows)
    _safe_to_csv(df, "manual_news_inputs.csv")

    return df


def build_combined_news_input_table(
    include_live_news: bool = True,
    max_live_rows: int = NEWS_MAX_LIVE_ROWS,
    manual_news: pd.DataFrame | None = None,
    live_news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine live RSS news with manual source-driven fallback rows.

    Accepting pre-built manual/live tables prevents duplicate RSS fetches during
    the full pipeline run.
    """

    if manual_news is None:
        manual_news = build_manual_news_input_table()

    frames = []

    if include_live_news:
        if live_news is None:
            live_news = collect_live_news_inputs(max_live_rows=max_live_rows)

        if live_news is not None and not live_news.empty:
            frames.append(live_news.head(max_live_rows))

    if manual_news is not None and not manual_news.empty:
        frames.append(manual_news)

    if not frames:
        combined = _empty_news_frame()
    else:
        combined = pd.concat(frames, ignore_index=True)

    combined = deduplicate_news(combined)
    _safe_to_csv(combined, "combined_news_inputs.csv")

    return combined


def classify_theme(title: str, summary: str = "") -> str:
    """Classify a headline into a defense / valuation theme."""

    combined_text = f"{title} {summary}"
    normalized_text = _normalize_text(combined_text)

    theme_scores = {}

    for theme, keywords in THEME_KEYWORDS.items():
        score = sum(
            1
            for keyword in keywords
            if keyword.lower() in normalized_text
        )

        if score > 0:
            theme_scores[theme] = score

    if not theme_scores:
        return "General Defense / Company News"

    return max(theme_scores, key=theme_scores.get)


def classify_sentiment(title: str, summary: str = "") -> str:
    """Classify headline sentiment using simple keyword logic."""

    combined_text = f"{title} {summary}"
    normalized_text = _normalize_text(combined_text)

    positive_hits = sum(
        1
        for keyword in POSITIVE_KEYWORDS
        if keyword.lower() in normalized_text
    )

    negative_hits = sum(
        1
        for keyword in NEGATIVE_KEYWORDS
        if keyword.lower() in normalized_text
    )

    mixed_hits = sum(
        1
        for keyword in MIXED_KEYWORDS
        if keyword.lower() in normalized_text
    )

    if positive_hits > negative_hits and positive_hits >= mixed_hits:
        return "Positive"

    if negative_hits > positive_hits and negative_hits >= mixed_hits:
        return "Negative"

    if mixed_hits > 0:
        return "Mixed"

    return "Neutral"


def sentiment_to_score(sentiment: str) -> int:
    """Map sentiment to a numeric score."""

    mapping = {
        "Positive": 80,
        "Neutral": 60,
        "Mixed": 50,
        "Negative": 35,
    }

    return mapping.get(sentiment, 55)


def theme_to_relevance_score(theme: str) -> int:
    """Score how relevant a theme is to CACI's valuation thesis."""

    mapping = {
        "Contract Award": 90,
        "Earnings / Guidance": 95,
        "Backlog / Book-to-Bill": 90,
        "Acquisition / M&A": 75,
        "Defense Budget": 85,
        "Intelligence / Cyber": 90,
        "Geopolitical Conflict": 70,
        "Technology / Modernization": 85,
        "Analyst / Valuation": 75,
        "Execution / Risk": 80,
        "General Defense / Company News": 55,
    }

    return mapping.get(theme, 55)


def _valuation_signal_from_score(composite_score: float) -> str:
    """Convert composite news score into a valuation signal."""

    if pd.isna(composite_score):
        return "Unavailable"
    if composite_score >= 80:
        return "Strong Positive"
    if composite_score >= 70:
        return "Positive"
    if composite_score >= 55:
        return "Neutral / Monitor"
    if composite_score >= 45:
        return "Mixed / Cautious"
    return "Negative"


def _news_adjustment_signal(value: float) -> str:
    """Convert news adjustment into a signal."""

    if pd.isna(value):
        return "Unavailable"
    if value > 0:
        return "Positive"
    if value < 0:
        return "Negative"
    return "Neutral / Monitor"


def classify_news_table(news_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Classify headlines by theme, sentiment, relevance, and valuation impact."""

    if news_df is None:
        news_df = build_combined_news_input_table(include_live_news=True)

    if news_df is None or news_df.empty:
        classified = pd.DataFrame(
            columns=[
                "Date",
                "Source",
                "Title",
                "Theme",
                "Sentiment",
                "Sentiment Score",
                "Relevance Score",
                "Composite News Score",
                "Valuation Signal",
                "Summary",
                "URL / Reference",
                "Collection Method",
            ]
        )
        _safe_to_csv(classified, "classified_news.csv")
        return classified

    sentiment_weight = NEWS_SENTIMENT_WEIGHTS.get("Sentiment Score", 0.55)
    relevance_weight = NEWS_SENTIMENT_WEIGHTS.get("Relevance Score", 0.45)

    total_weight = sentiment_weight + relevance_weight

    if total_weight == 0:
        sentiment_weight = 0.55
        relevance_weight = 0.45
        total_weight = 1.0

    sentiment_weight = sentiment_weight / total_weight
    relevance_weight = relevance_weight / total_weight

    classified_rows = []

    for _, row in news_df.iterrows():
        title = row.get("Title", "")
        summary = row.get("Summary", "")

        theme = classify_theme(title, summary)
        sentiment = classify_sentiment(title, summary)

        sentiment_score = sentiment_to_score(sentiment)
        relevance_score = theme_to_relevance_score(theme)

        composite_score = (
            sentiment_score * sentiment_weight
            + relevance_score * relevance_weight
        )

        valuation_signal = _valuation_signal_from_score(composite_score)

        classified_rows.append(
            {
                "Date": row.get("Date", ""),
                "Source": row.get("Source", ""),
                "Title": title,
                "Theme": theme,
                "Sentiment": sentiment,
                "Sentiment Score": sentiment_score,
                "Sentiment Score Type": "Score",
                "Relevance Score": relevance_score,
                "Relevance Score Type": "Score",
                "Composite News Score": composite_score,
                "Composite News Score Type": "Score",
                "Valuation Signal": valuation_signal,
                "Summary": summary,
                "URL / Reference": row.get("URL / Reference", ""),
                "Collection Method": row.get("Collection Method", "Unknown"),
            }
        )

    classified = pd.DataFrame(classified_rows)

    if not classified.empty and "Date" in classified.columns:
        try:
            classified = classified.sort_values("Date", ascending=False)
        except Exception:
            pass

    _safe_to_csv(classified, "classified_news.csv")

    return classified


def build_news_theme_summary(
    classified_news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize news classification by theme."""

    if classified_news is None:
        classified_news = classify_news_table()

    if classified_news is None or classified_news.empty:
        summary = pd.DataFrame(
            [
                {
                    "Theme": "No news available",
                    "Headline Count": 0,
                    "Headline Count Type": "Plain Number",
                    "Average News Score": 0,
                    "Average News Score Type": "Score",
                    "Primary Sentiment": "Unavailable",
                    "Signal": "Unavailable",
                    "Interpretation": "No headlines were available for classification.",
                }
            ]
        )
        _safe_to_csv(summary, "news_theme_summary.csv")
        return summary

    rows = []

    for theme, group in classified_news.groupby("Theme"):
        sentiment_counts = group["Sentiment"].value_counts()
        primary_sentiment = (
            sentiment_counts.index[0]
            if not sentiment_counts.empty
            else "Unavailable"
        )
        average_score = group["Composite News Score"].mean()
        signal = _valuation_signal_from_score(average_score)

        if average_score >= 80:
            interpretation = "Strongly positive valuation-relevant news theme."
        elif average_score >= 70:
            interpretation = "Positive valuation-relevant news theme."
        elif average_score >= 55:
            interpretation = "Neutral or supportive theme to monitor."
        elif average_score >= 45:
            interpretation = "Mixed theme requiring monitoring."
        else:
            interpretation = "Negative or risk-oriented theme."

        rows.append(
            {
                "Theme": theme,
                "Headline Count": len(group),
                "Headline Count Type": "Plain Number",
                "Average News Score": average_score,
                "Average News Score Type": "Score",
                "Primary Sentiment": primary_sentiment,
                "Signal": signal,
                "Interpretation": interpretation,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        "Average News Score",
        ascending=False,
    )

    _safe_to_csv(summary, "news_theme_summary.csv")

    return summary


def build_news_intelligence_summary(
    classified_news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a compact news intelligence summary for the PDF report."""

    if classified_news is None:
        classified_news = classify_news_table()

    if classified_news is None or classified_news.empty:
        summary = pd.DataFrame(
            [
                _summary_row(
                    metric="Overall News Intelligence Score",
                    value=0,
                    value_type="Score",
                    signal="Unavailable",
                    interpretation="No news headlines were available for classification.",
                )
            ]
        )
        _safe_to_csv(summary, "news_intelligence_summary.csv")
        return summary

    overall_score = classified_news["Composite News Score"].mean()

    positive_count = int((classified_news["Sentiment"] == "Positive").sum())
    neutral_count = int((classified_news["Sentiment"] == "Neutral").sum())
    mixed_count = int((classified_news["Sentiment"] == "Mixed").sum())
    negative_count = int((classified_news["Sentiment"] == "Negative").sum())

    live_count = (
        int((classified_news["Collection Method"] == "Live RSS").sum())
        if "Collection Method" in classified_news.columns
        else 0
    )
    manual_count = len(classified_news) - live_count

    top_headline = classified_news.sort_values(
        "Composite News Score",
        ascending=False,
    ).iloc[0]

    weakest_headline = classified_news.sort_values(
        "Composite News Score",
        ascending=True,
    ).iloc[0]

    signal = _valuation_signal_from_score(overall_score)

    if overall_score >= 80:
        interpretation = "Recent news and source inputs strongly support the CACI valuation thesis."
    elif overall_score >= 70:
        interpretation = "Recent news and source inputs are supportive of the CACI valuation thesis."
    elif overall_score >= 55:
        interpretation = "Recent news and source inputs are generally balanced but worth monitoring."
    elif overall_score >= 45:
        interpretation = "Recent news and source inputs are mixed and may require valuation caution."
    else:
        interpretation = "Recent news and source inputs create a negative valuation signal."

    rows = [
        _summary_row(
            metric="Overall News Intelligence Score",
            value=overall_score,
            value_type="Score",
            signal=signal,
            interpretation=interpretation,
        ),
        _summary_row(
            metric="Live RSS Headlines",
            value=live_count,
            value_type="Plain Number",
            signal="Reference",
            interpretation="Number of live RSS headlines collected without a paid news API.",
        ),
        _summary_row(
            metric="Manual / Fallback Headlines",
            value=manual_count,
            value_type="Plain Number",
            signal="Reference",
            interpretation="Number of source-driven manual headlines included for model stability.",
        ),
        _summary_row(
            metric="Positive Headlines",
            value=positive_count,
            value_type="Plain Number",
            signal=signal,
            interpretation="Number of headlines classified as positive.",
        ),
        _summary_row(
            metric="Neutral Headlines",
            value=neutral_count,
            value_type="Plain Number",
            signal=signal,
            interpretation="Number of headlines classified as neutral.",
        ),
        _summary_row(
            metric="Mixed Headlines",
            value=mixed_count,
            value_type="Plain Number",
            signal=signal,
            interpretation="Number of headlines classified as mixed.",
        ),
        _summary_row(
            metric="Negative Headlines",
            value=negative_count,
            value_type="Plain Number",
            signal=signal,
            interpretation="Number of headlines classified as negative.",
        ),
        _summary_row(
            metric="Strongest Headline",
            value=top_headline["Title"],
            value_type="Text",
            signal=top_headline["Valuation Signal"],
            interpretation=top_headline["Theme"],
        ),
        _summary_row(
            metric="Weakest Headline",
            value=weakest_headline["Title"],
            value_type="Text",
            signal=weakest_headline["Valuation Signal"],
            interpretation=weakest_headline["Theme"],
        ),
        _summary_row(
            metric="Last Updated",
            value=datetime.now().strftime("%Y-%m-%d"),
            value_type="Text",
            signal="Reference",
            interpretation="Date the news intelligence model was last run.",
        ),
    ]

    summary = pd.DataFrame(rows)
    _safe_to_csv(summary, "news_intelligence_summary.csv")

    return summary


def build_news_valuation_adjustments(
    classified_news: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Translate news intelligence into optional valuation adjustments.

    These adjustments are not directly applied to the DCF. They are reportable
    signals that can support expected-return confidence and scenario review.
    """

    if classified_news is None:
        classified_news = classify_news_table()

    if classified_news is None or classified_news.empty:
        overall_score = 0.0
    else:
        overall_score = _safe_float(classified_news["Composite News Score"].mean(), 0.0)

    if overall_score >= 80:
        valuation_adjustment = 0.030
        confidence_adjustment = 0.040
        interpretation = "News backdrop supports a modest valuation premium."
    elif overall_score >= 70:
        valuation_adjustment = 0.015
        confidence_adjustment = 0.025
        interpretation = "News backdrop is supportive."
    elif overall_score >= 55:
        valuation_adjustment = 0.000
        confidence_adjustment = 0.000
        interpretation = "News backdrop is balanced; no valuation adjustment applied."
    elif overall_score >= 45:
        valuation_adjustment = -0.015
        confidence_adjustment = -0.025
        interpretation = "News backdrop is mixed and warrants monitoring."
    else:
        valuation_adjustment = -0.030
        confidence_adjustment = -0.040
        interpretation = "News backdrop warrants a valuation haircut."

    adjustments = pd.DataFrame(
        [
            {
                "Adjustment": "News Valuation Adjustment",
                "Value": valuation_adjustment,
                "Value Type": "Percent",
                "Signal": _news_adjustment_signal(valuation_adjustment),
                "Interpretation": interpretation,
            },
            {
                "Adjustment": "News Confidence Adjustment",
                "Value": confidence_adjustment,
                "Value Type": "Percent",
                "Signal": _news_adjustment_signal(confidence_adjustment),
                "Interpretation": "Optional confidence adjustment for expected-return model.",
            },
            {
                "Adjustment": "Overall News Intelligence Score",
                "Value": overall_score,
                "Value Type": "Score",
                "Signal": _valuation_signal_from_score(overall_score),
                "Interpretation": (
                    "Composite score based on headline sentiment and valuation relevance."
                ),
            },
        ]
    )

    _safe_to_csv(adjustments, "news_valuation_adjustments.csv")

    return adjustments


def run_news_intelligence_model() -> dict[str, pd.DataFrame]:
    """Run the full API-free news intelligence model."""

    manual_news = build_manual_news_input_table()
    live_news = collect_live_news_inputs(
        max_items_per_query=NEWS_MAX_ITEMS_PER_QUERY,
        max_live_rows=NEWS_MAX_LIVE_ROWS,
    )

    combined_news = build_combined_news_input_table(
        include_live_news=True,
        max_live_rows=NEWS_MAX_LIVE_ROWS,
        manual_news=manual_news,
        live_news=live_news,
    )

    classified_news = classify_news_table(combined_news)
    theme_summary = build_news_theme_summary(classified_news)
    intelligence_summary = build_news_intelligence_summary(classified_news)
    valuation_adjustments = build_news_valuation_adjustments(classified_news)

    return {
        "manual_news_inputs": manual_news,
        "live_news_inputs": live_news,
        "combined_news_inputs": combined_news,
        "classified_news": classified_news,
        "news_theme_summary": theme_summary,
        "news_intelligence_summary": intelligence_summary,
        "news_valuation_adjustments": valuation_adjustments,
    }