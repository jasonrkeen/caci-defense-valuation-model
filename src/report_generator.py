from pathlib import Path
from html import escape

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
)

from src.config import (
    REPORTS_DIR,
    REPORT_FILENAME,
    REPORT_TITLE,
    REPORT_SUBTITLE,
    AUTHOR_NAME,
    AUTHOR_EMAIL,
    COMPANY_NAME,
    TABLES_DIR,
)
from src.fundamentals import build_thesis_points


TABLE_WIDTH = 10.05 * inch


# ------------------------------------------------------------
# Formatting helpers
# ------------------------------------------------------------

def _safe_float(value):
    """Safely convert a value to float."""

    try:
        if value is None:
            return None

        if pd.isna(value):
            return None

        return float(value)
    except Exception:
        return None


def _normalized_context(column_name: str = "", row_label: str = "") -> str:
    """Build normalized formatting context."""

    return f"{column_name} {row_label}".lower().strip()


def _is_metadata_column(column_name: str) -> bool:
    """Identify internal formatting / raw-value columns that should be hidden."""

    name = str(column_name).strip()
    lower = name.lower()

    visible_type_columns = {
        "output type",
    }

    if lower in visible_type_columns:
        return False

    return (
        lower.startswith("raw ")
        or lower in {"value type", "driver value type"}
        or lower.endswith(" type")
        or lower.endswith(" value type")
    )


def _metadata_candidates_for_column(column_name: str) -> list[str]:
    """Return possible metadata columns for a visible data column."""

    name = str(column_name).strip()

    return [
        f"{name} Type",
        f"{name} Value Type",
    ]


def _value_type_for_cell(row: pd.Series, column_name: str) -> str:
    """Find explicit formatting metadata for a table cell."""

    column_clean = str(column_name).strip().lower()

    if column_clean == "value":
        return str(row.get("Value Type", "")).strip()

    if column_clean == "driver value":
        return str(row.get("Driver Value Type", "")).strip()

    for candidate in _metadata_candidates_for_column(column_name):
        if candidate in row.index:
            value_type = row.get(candidate, "")

            if pd.notna(value_type) and str(value_type).strip():
                return str(value_type).strip()

    return ""


def _is_wacc_sensitivity_value(column_name: str = "", row_label: str = "") -> bool:
    """Detect DCF sensitivity fair-value cells under WACC columns."""

    column_text = str(column_name).lower()
    row_text = str(row_label).lower()

    return column_text.startswith("wacc ") or (
        "terminal growth" in row_text and "wacc" in column_text
    )


def _is_yoy_growth_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect YoY growth rows / columns that should be formatted as percentages."""

    context = _normalized_context(column_name, row_label)
    return "yoy growth" in context


def _is_monte_carlo_fair_value_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect Monte Carlo percentile fair-value rows that should be dollars."""

    context = _normalized_context(column_name, row_label)

    fair_value_tokens = [
        "mean fair value",
        "median fair value",
        "fair value standard deviation",
        "p5 downside case",
        "p10 downside case",
        "p25 lower quartile",
        "p75 upper quartile",
        "p90 upside case",
        "p95 high upside case",
        "minimum simulated value",
        "maximum simulated value",
    ]

    return any(token in context for token in fair_value_tokens)


def _is_return_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect investment return rows / columns that should be formatted as percentages."""

    context = _normalized_context(column_name, row_label)

    return any(
        token in context
        for token in [
            "total return",
            "one-year return",
            "annualized return",
            "active annualized return",
            "forecast return",
            "expected return",
            "raw expected return",
            "risk-adjusted expected return",
            "monte carlo expected return",
            "median forecast return",
            "mean forecast return",
            "p10 forecast return",
            "p90 forecast return",
            "base upside / downside",
            "annualized factor move",
            "estimated annualized contribution",
            "annualized alpha",
            "annualized residual volatility",
            "downside vs base case",
            "percent impact",
        ]
    )


def _is_adjustment_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect adjustment fields that should usually be shown as percentages."""

    context = _normalized_context(column_name, row_label)

    return any(
        token in context
        for token in [
            "valuation adjustment",
            "risk adjustment",
            "defense risk adjustment",
            "news valuation adjustment",
            "confidence adjustment",
            "wacc adjustment",
            "terminal growth adjustment",
            "revenue growth adjustment",
            "haircut",
        ]
    )


def _is_rate_or_percent_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect fields that are rates or percentages."""

    context = _normalized_context(column_name, row_label)

    if _is_yoy_growth_context(column_name, row_label):
        return True

    if _is_return_context(column_name, row_label):
        return True

    if _is_adjustment_context(column_name, row_label):
        return True

    percent_specific_tokens = [
        "revenue cagr",
        "revenue growth",
        "ebitda margin",
        "fcf margin",
        "tax rate",
        "capex % revenue",
        "d&a % revenue",
        "d and a % revenue",
        "nwc % revenue change",
        "working capital",
        "terminal growth",
        "wacc",
        "margin",
        "growth",
        "yield",
        "drawdown",
        "annualized volatility",
        "volatility",
        "var",
        "expected shortfall",
        "maximum daily loss",
        "exception rate",
        "risk-free rate",
        "upside",
        "downside",
        "tracking error",
        "fcf yield",
        "premium / discount",
        "probability",
        "r-squared",
    ]

    return any(token in context for token in percent_specific_tokens)


def _is_ratio_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect valuation / risk ratios that should not be formatted as percentages."""

    context = _normalized_context(column_name, row_label)

    ratio_tokens = [
        "sharpe ratio",
        "sortino ratio",
        "information ratio",
        "beta",
        "beta / sensitivity",
        "sensitivity",
        "correlation",
        "peg ratio",
        "ev/ebitda",
        "ev/revenue",
        "price/sales",
        "trailing p/e",
        "forward p/e",
        "book-to-bill",
        "implied ev/ebitda",
        "kupiec p-value",
        "kupiec lr",
    ]

    return any(token in context for token in ratio_tokens)


def _is_score_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect score fields."""

    context = _normalized_context(column_name, row_label)

    return any(
        token in context
        for token in [
            "score",
            "reliability score",
            "confidence score",
            "input confidence score",
            "source confidence score",
            "priority score",
            "valuation relevance score",
            "average news score",
            "composite news score",
            "sentiment score",
            "relevance score",
        ]
    )


def _is_count_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect count / row / column fields."""

    context = _normalized_context(column_name, row_label)

    count_tokens = [
        "headline count",
        "headlines",
        "count",
        "simulation count",
        "forecast simulations",
        "forecast horizon days",
        "rows",
        "columns",
        "available tables",
        "total tables requested",
        "tables requested",
        "table availability",
        "observations",
        "regression observations",
        "exceptions",
    ]

    return any(token in context for token in count_tokens)


def _is_share_or_rank_context(column_name: str = "", row_label: str = "") -> bool:
    """Detect share count and ranking fields."""

    context = _normalized_context(column_name, row_label)

    plain_tokens = [
        "rank",
        "fiscal year",
        "year",
        "diluted shares",
        "shares",
        "average rank",
        "overall peer rank",
        "thesis point",
    ]

    return any(token in context for token in plain_tokens)


def _is_plain_number_context(column_name: str = "", row_label: str = "") -> bool:
    """Return True for numeric values that should not be shown as money or percent."""

    column_text = str(column_name).strip().lower()

    if column_text in ["weighted contribution"]:
        return True

    if _is_count_context(column_name, row_label):
        return True

    if _is_share_or_rank_context(column_name, row_label):
        return True

    if _is_ratio_context(column_name, row_label):
        return True

    if _is_score_context(column_name, row_label):
        return True

    return False


def _is_money_context(column_name: str = "", row_label: str = "") -> bool:
    """Return True when the value should be treated as a dollar amount."""

    context = _normalized_context(column_name, row_label)
    column_text = str(column_name).strip().lower()

    if _is_wacc_sensitivity_value(column_name, row_label):
        return True

    if _is_monte_carlo_fair_value_context(column_name, row_label):
        return True

    if _is_yoy_growth_context(column_name, row_label):
        return False

    if _is_rate_or_percent_context(column_name, row_label):
        return False

    if _is_plain_number_context(column_name, row_label):
        return False

    if column_text == "amount":
        return True

    money_tokens = [
        "market cap",
        "enterprise value",
        "equity value",
        "revenue",
        "ebitda",
        "free cash flow",
        "cash",
        "debt",
        "fair value",
        "price",
        "topline",
        "appropriated",
        "requested",
        "spending",
        "guidance",
        "forecast price",
        "current share price",
        "current price",
        "target mean price",
        "pv of explicit fcf",
        "pv of terminal value",
        "net debt",
        "base revenue",
        "base free cash flow",
        "year 5 fcf",
        "year 5 revenue",
        "year 5 ebitda",
        "probability-weighted fair value",
        "monte carlo median fair value",
        "bear case fair value",
        "base case fair value",
        "bull case fair value",
        "bear fair value",
        "base fair value",
        "bull fair value",
        "base fair value",
        "stress fair value",
        "dollar impact",
    ]

    return any(token in context for token in money_tokens)


def _format_percentage_value(numeric_value: float) -> str:
    """Format percentage values while handling decimal and already-percent representations."""

    if -5 <= numeric_value <= 5:
        return f"{numeric_value:.2%}"

    if -100 <= numeric_value <= 100:
        return f"{numeric_value:.2f}%"

    return f"{numeric_value:,.2f}%"


def _format_money_value(
    numeric_value: float,
    column_name: str = "",
    row_label: str = "",
) -> str:
    """Format money-like values."""

    context = _normalized_context(column_name, row_label)
    column_text = str(column_name).strip().lower()

    if _is_wacc_sensitivity_value(column_name, row_label):
        return f"${numeric_value:,.2f}"

    if _is_monte_carlo_fair_value_context(column_name, row_label):
        return f"${numeric_value:,.2f}"

    if column_text == "amount":
        return f"${numeric_value:,.2f}B"

    if "topline" in context or "appropriated" in context or "requested" in context:
        if abs(numeric_value) < 1000:
            return f"${numeric_value:,.2f}B"

    if "base revenue" in context or "revenue guidance" in context:
        if abs(numeric_value) < 1000:
            return f"${numeric_value:,.2f}B"

    if abs(numeric_value) >= 1_000_000_000:
        return f"${numeric_value / 1_000_000_000:,.1f}B"

    if abs(numeric_value) >= 1_000_000:
        return f"${numeric_value / 1_000_000:,.1f}M"

    if abs(numeric_value) >= 1000:
        return f"${numeric_value:,.1f}"

    return f"${numeric_value:,.2f}"


def _format_money_millions_value(numeric_value: float) -> str:
    """Format values stored in millions of dollars."""

    if abs(numeric_value) >= 1000:
        return f"${numeric_value / 1000:,.2f}B"

    return f"${numeric_value:,.0f}M"


def _format_plain_number(numeric_value: float) -> str:
    """Format plain numeric values."""

    if numeric_value.is_integer():
        if 1900 <= numeric_value <= 2100:
            return f"{int(numeric_value)}"

        return f"{numeric_value:,.0f}"

    return f"{numeric_value:,.2f}"


def _format_value(
    value,
    column_name: str = "",
    row_label: str = "",
    value_type: str = "",
):
    """Format report table values using metadata first, then context."""

    if pd.isna(value):
        return ""

    numeric_value = _safe_float(value)

    if numeric_value is None:
        return str(value)

    value_type_clean = str(value_type).strip().lower()
    context = _normalized_context(column_name, row_label)

    if value_type_clean in ["money", "currency", "dollar", "dollars"]:
        return _format_money_value(
            numeric_value,
            column_name=column_name,
            row_label=row_label,
        )

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

    if value_type_clean in ["multiple", "ratio"]:
        return _format_plain_number(numeric_value)

    if value_type_clean in ["score"]:
        return _format_plain_number(numeric_value)

    if value_type_clean in ["plain", "plain number", "number", "count", "integer"]:
        return _format_plain_number(numeric_value)

    if value_type_clean in ["text", "string"]:
        if numeric_value.is_integer():
            return f"{int(numeric_value)}"

        return str(value)

    if "year" in context and 1900 <= numeric_value <= 2100:
        return f"{int(numeric_value)}"

    if _is_wacc_sensitivity_value(column_name, row_label):
        return _format_money_value(
            numeric_value,
            column_name=column_name,
            row_label=row_label,
        )

    if _is_monte_carlo_fair_value_context(column_name, row_label):
        return _format_money_value(
            numeric_value,
            column_name=column_name,
            row_label=row_label,
        )

    if _is_plain_number_context(column_name, row_label):
        return _format_plain_number(numeric_value)

    if _is_money_context(column_name, row_label):
        return _format_money_value(
            numeric_value,
            column_name=column_name,
            row_label=row_label,
        )

    if _is_rate_or_percent_context(column_name, row_label):
        return _format_percentage_value(numeric_value)

    if abs(numeric_value) >= 1_000_000_000:
        return f"{numeric_value / 1_000_000_000:,.1f}B"

    if abs(numeric_value) >= 1_000_000:
        return f"{numeric_value / 1_000_000:,.1f}M"

    if abs(numeric_value) >= 1000:
        return f"{numeric_value:,.1f}"

    return f"{numeric_value:,.2f}"


def _shorten_text(
    value,
    max_length=75,
    column_name: str = "",
    row_label: str = "",
    value_type: str = "",
):
    """Format and truncate long table text."""

    text = _format_value(
        value,
        column_name=column_name,
        row_label=row_label,
        value_type=value_type,
    )

    if len(text) > max_length:
        return text[: max_length - 3] + "..."

    return text


def _row_label_from_row(row: pd.Series, columns: list[str]) -> str:
    """Find the most useful row label for context-aware formatting."""

    label_columns = [
        "Metric",
        "Spending Category",
        "Component",
        "Risk Component",
        "Spending Component",
        "Exposure Component",
        "Adjustment",
        "Bridge Item",
        "Assumption",
        "Category",
        "Theme",
        "Step",
        "Source",
        "Model Input",
        "Quality Driver",
        "Valuation Driver",
        "Exposure",
        "Exposure Category",
        "Customer / Exposure Category",
        "Scenario",
        "Stress Case",
        "Factor",
    ]

    labels = []

    for col in label_columns:
        if col in columns:
            value = row.get(col, "")

            if pd.notna(value):
                labels.append(str(value))

    return " ".join(labels)


# ------------------------------------------------------------
# Table display helpers
# ------------------------------------------------------------

def _clean_table_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Make a copy of a DataFrame and remove noisy columns from PDF layout."""

    if df is None or df.empty:
        return pd.DataFrame({"Message": ["No data available"]})

    cleaned = df.copy()

    drop_columns = [
        "URL",
        "URL / Reference",
    ]

    for col in drop_columns:
        if col in cleaned.columns and len(cleaned.columns) > 6:
            cleaned = cleaned.drop(columns=[col])

    return cleaned


def _select_display_columns(df: pd.DataFrame, max_columns: int | None = None) -> pd.DataFrame:
    """Reduce wide tables while preserving hidden formatting metadata."""

    if df is None or df.empty:
        return pd.DataFrame({"Message": ["No data available"]})

    metadata_columns = [col for col in df.columns if _is_metadata_column(col)]
    visible_columns = [col for col in df.columns if not _is_metadata_column(col)]

    if max_columns is None or len(visible_columns) <= max_columns:
        selected_visible = visible_columns
    else:
        priority_columns = [
            "Output Type",
            "Overall Peer Rank",
            "Ticker",
            "Company",
            "Peer Group",
            "Metric",
            "Value",
            "Signal",
            "Interpretation",
            "Factor",
            "Beta / Sensitivity",
            "Correlation",
            "Average Annualized Factor Return",
            "Estimated Annualized Contribution",
            "Stress Case",
            "Stress Signal",
            "Valuation Bias",
            "Downside vs Base Case",
            "Base Fair Value",
            "Stress Fair Value",
            "Dollar Impact",
            "Percent Impact",
            "Stress Resilience Score",
            "Stress Resilience Adjustment",
            "Scenario",
            "Fair Value Per Share",
            "Revenue CAGR",
            "EBITDA Margin",
            "WACC",
            "Terminal Growth",
            "Enterprise Value",
            "Equity Value",
            "Component",
            "Risk Component",
            "Spending Component",
            "Exposure Component",
            "Score",
            "Weight",
            "Weighted Contribution",
            "Driver Value",
            "Source",
            "Model Input",
            "Model Use",
            "Valuation Relevance",
            "Category",
            "Theme",
            "Sentiment",
            "Composite News Score",
            "Valuation Signal",
            "Summary",
            "Bridge Item",
            "Assumption",
            "Current Value",
            "Primary Source",
            "Audit Note",
            "Step",
            "Explanation",
        ]

        selected_visible = []

        for col in priority_columns:
            if col in visible_columns and col not in selected_visible:
                selected_visible.append(col)

        for col in visible_columns:
            if len(selected_visible) >= max_columns:
                break

            if col not in selected_visible:
                selected_visible.append(col)

        selected_visible = selected_visible[:max_columns]

    selected_columns = selected_visible + [
        col for col in metadata_columns
        if col not in selected_visible
    ]

    return df[selected_columns].copy()


def _cell_style(font_size: float, bold: bool = False, text_color=colors.black):
    """Create paragraph style for wrapped table cells."""

    return ParagraphStyle(
        name=f"CellStyle_{font_size}_{bold}_{str(text_color)}",
        fontName="Helvetica-Bold" if bold else "Helvetica",
        fontSize=font_size,
        leading=font_size + 1.6,
        textColor=text_color,
        wordWrap="CJK",
        splitLongWords=True,
    )


def _paragraph_cell(text, font_size=6.3, bold=False, text_color=colors.black):
    """Create a wrapped ReportLab Paragraph cell."""

    safe_text = escape(str(text)).replace("\n", "<br/>")

    return Paragraph(
        safe_text,
        _cell_style(font_size=font_size, bold=bold, text_color=text_color),
    )


def _column_widths(columns: list[str], total_width: float = TABLE_WIDTH) -> list[float]:
    """Generate proportional column widths so tables fit within landscape letter."""

    if not columns:
        return []

    weights = []

    for col in columns:
        c = str(col).lower()

        if c in [
            "metric",
            "component",
            "risk component",
            "spending component",
            "exposure component",
            "bridge item",
            "assumption",
            "step",
            "factor",
            "stress case",
        ]:
            weights.append(1.45)
        elif c in [
            "interpretation",
            "rationale",
            "summary",
            "evidence",
            "model link",
            "explanation",
            "how to read it",
            "example use",
            "process",
            "outputs",
            "inputs",
        ]:
            weights.append(2.15)
        elif c in ["source", "primary source", "model use", "audit note"]:
            weights.append(1.65)
        elif c in ["ticker"]:
            weights.append(0.65)
        elif c in ["company"]:
            weights.append(1.35)
        elif c in ["peer group"]:
            weights.append(1.05)
        elif c in ["overall peer rank"]:
            weights.append(0.80)
        elif c in ["output type"]:
            weights.append(1.05)
        elif c in ["signal", "valuation signal", "risk direction"]:
            weights.append(1.15)
        elif c in [
            "value",
            "score",
            "weight",
            "weighted contribution",
            "driver value",
            "beta / sensitivity",
            "correlation",
            "average annualized factor return",
            "estimated annualized contribution",
            "downside vs base case",
            "base fair value",
            "stress fair value",
            "dollar impact",
            "percent impact",
        ]:
            weights.append(0.85)
        else:
            weights.append(1.0)

    total_weight = sum(weights)

    return [(weight / total_weight) * total_width for weight in weights]


def dataframe_to_table(
    df,
    max_rows=12,
    font_size=6.3,
    max_text_length=78,
    max_columns=12,
):
    """Convert a DataFrame into a ReportLab table with wrapped cells."""

    cleaned = _clean_table_for_display(df)
    cleaned = _select_display_columns(cleaned, max_columns=max_columns)
    trimmed = cleaned.head(max_rows).copy()

    formatting_columns = list(trimmed.columns)

    display_columns = [
        col for col in formatting_columns
        if not _is_metadata_column(col)
    ]

    if not display_columns:
        display_columns = ["Message"]
        trimmed = pd.DataFrame({"Message": ["No displayable columns available"]})
        formatting_columns = list(trimmed.columns)

    header_style = _cell_style(font_size=font_size, bold=True, text_color=colors.white)

    data = [
        [
            Paragraph(
                escape(str(col)),
                header_style,
            )
            for col in display_columns
        ]
    ]

    for _, row in trimmed.iterrows():
        row_label = _row_label_from_row(row, formatting_columns)
        formatted_row = []

        for col in display_columns:
            col_value_type = _value_type_for_cell(row, str(col))

            formatted_value = _shorten_text(
                row[col],
                max_length=max_text_length,
                column_name=str(col),
                row_label=row_label,
                value_type=col_value_type,
            )

            formatted_row.append(
                _paragraph_cell(
                    formatted_value,
                    font_size=font_size,
                    bold=False,
                    text_color=colors.black,
                )
            )

        data.append(formatted_row)

    table = Table(
        data,
        repeatRows=1,
        colWidths=_column_widths(display_columns),
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                (
                    "ROWBACKGROUNDS",
                    (0, 1),
                    (-1, -1),
                    [colors.white, colors.HexColor("#F3F4F6")],
                ),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )

    return table


def read_optional_table(file_name: str) -> pd.DataFrame:
    """Read an optional CSV table from outputs/tables."""

    path = TABLES_DIR / file_name

    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read optional table {file_name}: {exc}")
        return pd.DataFrame()


def add_chart(elements, path, width=6.8 * inch, height=3.5 * inch):
    """Add a chart to the PDF if the file exists."""

    if not path:
        return

    chart_path = Path(path)

    if chart_path.exists():
        elements.append(Image(str(chart_path), width=width, height=height))
        elements.append(Spacer(1, 0.15 * inch))


def add_section(
    elements,
    title,
    table_df=None,
    paragraph=None,
    max_rows=12,
    page_break_after=False,
    font_size=6.3,
    max_text_length=78,
    max_columns=12,
):
    """Add a titled section with optional paragraph and table."""

    styles = get_report_styles()

    elements.append(Paragraph(title, styles["Section"]))

    if paragraph:
        elements.append(Paragraph(paragraph, styles["BodyText"]))
        elements.append(Spacer(1, 0.08 * inch))

    if table_df is not None:
        elements.append(
            dataframe_to_table(
                table_df,
                max_rows=max_rows,
                font_size=font_size,
                max_text_length=max_text_length,
                max_columns=max_columns,
            )
        )
        elements.append(Spacer(1, 0.12 * inch))

    if page_break_after:
        elements.append(PageBreak())


def add_optional_section(
    elements,
    title,
    table_df,
    paragraph=None,
    max_rows=12,
    page_break_after=False,
    font_size=6.3,
    max_text_length=78,
    max_columns=12,
):
    """Add a section only if the DataFrame exists and is not empty."""

    if table_df is None or table_df.empty:
        return

    add_section(
        elements=elements,
        title=title,
        table_df=table_df,
        paragraph=paragraph,
        max_rows=max_rows,
        page_break_after=page_break_after,
        font_size=font_size,
        max_text_length=max_text_length,
        max_columns=max_columns,
    )


# ------------------------------------------------------------
# Explanatory tables
# ------------------------------------------------------------

def build_model_process_table() -> pd.DataFrame:
    """Explain the major inputs, process steps, and outputs used in the report."""

    rows = [
        {
            "Stage": "1. Market Data, Risk Metrics, and Factor Exposure",
            "Inputs": "CACI price history, S&P 500 benchmark data, Treasury yield proxy, defense ETF, technology ETF, volatility index",
            "Process": "Calculates returns, volatility, drawdown, beta, correlation, Sharpe ratio, Sortino ratio, tracking error, GARCH/EWMA VaR risk metrics, and multi-factor return exposure.",
            "Outputs": "Market Metrics, Market Snapshot, GARCH/VaR outputs, factor exposure summary, factor regression results, factor correlation summary, factor contribution summary, and price/risk charts",
        },
        {
            "Stage": "2. Source and Guidance Audit",
            "Inputs": "CACI investor presentation, earnings release, earnings slides, Form 10-K, government budget sources, Yahoo Finance",
            "Process": "Maps source material to model inputs and assigns confidence, relevance, and audit notes.",
            "Outputs": "Source Catalog, Source Quality Review, Model Input Audit, Source Coverage Summary",
        },
        {
            "Stage": "3. Fundamental and Business Quality Review",
            "Inputs": "Revenue guidance, FCF guidance, backlog, customer mix, technology mix, business quality assumptions",
            "Process": "Scores business quality, customer concentration, revenue visibility, cash flow conversion, and defense technology positioning.",
            "Outputs": "Fundamental Summary, Business Quality Summary, Valuation Driver Summary",
        },
        {
            "Stage": "4. Defense / Government / Macro Intelligence",
            "Inputs": "Defense appropriations, intelligence budget toplines, CACI customer exposure, macro risk assumptions",
            "Process": "Builds weighted scorecards for defense risk, government spending support, intelligence exposure, and macro backdrop.",
            "Outputs": "Defense Risk Engine, Government Spending Model, Intelligence Spending Signal, Defense Macro Summary",
        },
        {
            "Stage": "5. Valuation, Stress Testing, and Forecasting",
            "Inputs": "Revenue growth, EBITDA margin, tax rate, capex, depreciation and amortization, working capital, WACC, terminal growth, shares, net debt, stress-case adjustments",
            "Process": "Runs bear/base/bull DCF scenarios, DCF sensitivity, defense contractor stress tests, revenue forecasts, and price simulations.",
            "Outputs": "DCF Valuation Summary, DCF Sensitivity Table, Stress Test Summary, Stress Test Bridge, Stress Resilience Score, Forecast Summary, Price Forecast Summary",
        },
        {
            "Stage": "6. Monte Carlo and Expected Return",
            "Inputs": "Randomized DCF assumptions, current price, scenario probabilities, intelligence score, risk adjustments, news signal",
            "Process": "Simulates fair value distribution and converts valuation output into raw and risk-adjusted expected return.",
            "Outputs": "Monte Carlo Valuation Summary, Expected Return Summary, Expected Return Bridge",
        },
        {
            "Stage": "7. Peer and Final Signal",
            "Inputs": "Peer company valuation, growth, margin, beta, and target-price data from yfinance",
            "Process": "Benchmarks CACI against defense, government services, and technology-adjacent peers.",
            "Outputs": "Peer Comparison, Peer Ranking, Peer Positioning Summary, final Defense Technology Intelligence Score",
        },
    ]

    return pd.DataFrame(rows)


def build_how_to_read_table() -> pd.DataFrame:
    """Explain how to read the major report outputs."""

    rows = [
        {
            "Output Type": "Fair Value Per Share",
            "How to Read It": "Estimated equity value per diluted share from the DCF model. Compare this value with the current share price to estimate upside or downside.",
            "Example Use": "Base fair value above current price suggests potential undervaluation, subject to assumptions.",
        },
        {
            "Output Type": "Signal",
            "How to Read It": "Plain-English model interpretation such as Positive, Constructive, Neutral / Monitor, Cautious, or Negative.",
            "Example Use": "Use the signal to quickly understand whether the data point supports or weakens the investment case.",
        },
        {
            "Output Type": "Score",
            "How to Read It": "Most scorecards use a 0 to 100 scale. Higher scores generally indicate stronger support, lower risk, or better strategic positioning.",
            "Example Use": "A score above 70 usually indicates supportive conditions; below 60 usually requires monitoring.",
        },
        {
            "Output Type": "Weight",
            "How to Read It": "The relative importance assigned to each scorecard component. Weights are used to calculate weighted contribution.",
            "Example Use": "A high-weight component has more influence on the final score than a low-weight component.",
        },
        {
            "Output Type": "Weighted Contribution",
            "How to Read It": "Component score multiplied by its weight. These contributions roll up into the overall model score.",
            "Example Use": "Shows which components are driving the final intelligence score.",
        },
        {
            "Output Type": "Factor Beta / Sensitivity",
            "How to Read It": "Estimated sensitivity of CACI daily returns to market, defense, technology, rate, and volatility factors.",
            "Example Use": "A high defense factor beta suggests CACI behaves more like a defense-sector equity; negative rate sensitivity may indicate pressure from rising yields.",
        },
        {
            "Output Type": "Factor Model R-Squared",
            "How to Read It": "Share of CACI daily return variation explained by the selected factor model.",
            "Example Use": "A higher R-squared indicates that market, defense, technology, rate, and volatility factors explain more of the stock's movement.",
        },
        {
            "Output Type": "VaR / Expected Shortfall",
            "How to Read It": "One-day loss estimates from historical, parametric, and GARCH/EWMA volatility methods. Higher values indicate larger estimated downside risk.",
            "Example Use": "Use VaR and expected shortfall to understand downside risk separate from valuation upside.",
        },
        {
            "Output Type": "DCF Sensitivity",
            "How to Read It": "Shows how fair value changes as WACC and terminal growth assumptions change.",
            "Example Use": "Higher WACC lowers fair value; higher terminal growth usually raises fair value.",
        },
        {
            "Output Type": "Stress Test",
            "How to Read It": "Shows how fair value changes under downside scenarios such as budget delays, continuing resolutions, higher WACC, margin compression, recompete loss, and combined downside pressure.",
            "Example Use": "Use the stress-test downside table to identify which risk case creates the largest valuation impact versus the base DCF case.",
        },
        {
            "Output Type": "Monte Carlo Percentiles",
            "How to Read It": "Percentile outcomes from thousands of simulated valuation runs. P10 is downside, median is central case, P90 is upside.",
            "Example Use": "A median fair value above current price suggests favorable simulated valuation support.",
        },
        {
            "Output Type": "Expected Return",
            "How to Read It": "Estimated upside or downside after blending DCF scenarios and applying selected risk or intelligence adjustments.",
            "Example Use": "Risk-adjusted expected return is the final return estimate after model haircuts or boosts.",
        },
        {
            "Output Type": "Change Detection",
            "How to Read It": "Compares the latest run with the prior saved model snapshot.",
            "Example Use": "Highlights whether price, valuation, score, or forecast outputs moved materially since the last run.",
        },
    ]

    return pd.DataFrame(rows)


def build_signal_guide_table() -> pd.DataFrame:
    """Explain common signal labels used across the report."""

    rows = [
        {
            "Signal Family": "Positive / Strong / Constructive",
            "Meaning": "The metric supports the valuation thesis, revenue visibility, business quality, factor exposure, or expected return profile.",
            "Where It Appears": "Business quality, defense technology score, government spending, news intelligence, factor model, expected return",
        },
        {
            "Signal Family": "Neutral / Monitor",
            "Meaning": "The metric is not clearly positive or negative and should be monitored for future movement.",
            "Where It Appears": "Defense risk, valuation, market metrics, factor model, change detection",
        },
        {
            "Signal Family": "Cautious / Mixed / Elevated Risk",
            "Meaning": "The metric introduces risk, uncertainty, or downside sensitivity.",
            "Where It Appears": "Customer concentration, appropriations timing, WACC risk, drawdown, Monte Carlo downside cases, VaR outputs, factor model, stress testing",
        },
        {
            "Signal Family": "Reference",
            "Meaning": "The metric is informational and does not directly imply a positive or negative model conclusion.",
            "Where It Appears": "Fiscal years, share count, source counts, forecast horizon, regression observations, simulation count",
        },
        {
            "Signal Family": "Attractive Upside / High Upside Case",
            "Meaning": "The valuation or simulation output shows meaningful upside relative to the current share price.",
            "Where It Appears": "Monte Carlo summary, expected return summary, price forecast summary",
        },
        {
            "Signal Family": "Negative",
            "Meaning": "The metric points to downside, weak sentiment, weak market performance, or unfavorable risk conditions.",
            "Where It Appears": "Forecast downside cases, negative headlines, market risk outputs",
        },
    ]

    return pd.DataFrame(rows)


def get_report_styles():
    """Build and return ReportLab paragraph styles."""

    styles = getSampleStyleSheet()

    if "Small" not in styles:
        styles.add(
            ParagraphStyle(
                name="Small",
                parent=styles["BodyText"],
                fontSize=8,
                leading=10,
            )
        )

    if "Section" not in styles:
        styles.add(
            ParagraphStyle(
                name="Section",
                parent=styles["Heading2"],
                textColor=colors.HexColor("#111827"),
                spaceBefore=12,
                spaceAfter=6,
            )
        )

    if "Subtle" not in styles:
        styles.add(
            ParagraphStyle(
                name="Subtle",
                parent=styles["BodyText"],
                fontSize=8,
                leading=10,
                textColor=colors.HexColor("#4B5563"),
            )
        )

    return styles


# ------------------------------------------------------------
# Report builder
# ------------------------------------------------------------

def build_report(
    market_metrics,
    fundamental_summary,
    dcf_summary,
    analyst_summary,
    source_tables,
    defense_risk_tables,
    intelligence_spending_tables,
    government_spending_tables,
    defense_macro_tables,
    news_intelligence_tables,
    expected_return_tables,
    change_detection_tables,
    forecast_tables,
    peer_df,
    mc_summary,
    score_df,
    chart_paths,
) -> str:
    """Build the final PDF report."""

    output_path = REPORTS_DIR / REPORT_FILENAME

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=landscape(letter),
        rightMargin=0.45 * inch,
        leftMargin=0.45 * inch,
        topMargin=0.45 * inch,
        bottomMargin=0.45 * inch,
    )

    styles = get_report_styles()
    elements = []

    dcf_summary_report = read_optional_table("dcf_summary.csv")
    dcf_sensitivity = read_optional_table("dcf_sensitivity.csv")
    dcf_assumptions = read_optional_table("dcf_assumptions.csv")
    dcf_base_case_valuation_bridge = read_optional_table("dcf_base_case_valuation_bridge.csv")
    peer_summary = read_optional_table("peer_summary.csv")
    peer_ranking = read_optional_table("peer_ranking.csv")
    peer_positioning_summary = read_optional_table("peer_positioning_summary.csv")
    manual_analyst_inputs = read_optional_table("manual_analyst_inputs.csv")
    analyst_source_quality = read_optional_table("analyst_source_quality.csv")
    market_snapshot = read_optional_table("market_snapshot.csv")
    business_quality = read_optional_table("business_quality_summary.csv")
    business_quality_score = read_optional_table("business_quality_score_summary.csv")
    valuation_drivers = read_optional_table("valuation_driver_summary.csv")
    investment_thesis_points = read_optional_table("investment_thesis_points.csv")

    garch_var_summary = read_optional_table("garch_var_summary.csv")
    volatility_regime_summary = read_optional_table("volatility_regime_summary.csv")
    var_backtest_summary = read_optional_table("var_backtest_summary.csv")

    factor_exposure_summary = read_optional_table("factor_exposure_summary.csv")
    factor_regression_results = read_optional_table("factor_regression_results.csv")
    factor_correlation_summary = read_optional_table("factor_correlation_summary.csv")
    factor_contribution_summary = read_optional_table("factor_contribution_summary.csv")

    stress_test_summary = read_optional_table("stress_test_summary.csv")
    stress_test_bridge = read_optional_table("stress_test_bridge.csv")
    stress_test_score = read_optional_table("stress_test_score.csv")
    stress_test_assumptions = read_optional_table("stress_test_assumptions.csv")

    company_name_clean = str(COMPANY_NAME).rstrip(".")

    elements.append(Paragraph(REPORT_TITLE, styles["Title"]))
    elements.append(Paragraph(REPORT_SUBTITLE, styles["Heading3"]))
    elements.append(Paragraph(f"{AUTHOR_NAME} | {AUTHOR_EMAIL}", styles["Small"]))
    elements.append(Spacer(1, 0.20 * inch))

    elements.append(Paragraph("Executive Summary", styles["Section"]))
    elements.append(
        Paragraph(
            f"The model combines market data, risk metrics, peer benchmarking, DCF scenarios, defense contractor stress testing, "
            "Monte Carlo simulation, company guidance, backlog indicators, defense risk scoring, government spending signals, "
            "defense macro analysis, news intelligence, forecasting, expected return modeling, factor exposure modeling, "
            "GARCH/EWMA VaR risk modeling, and change detection.",
            styles["BodyText"],
        )
    )

    elements.append(Spacer(1, 0.10 * inch))

    for point in build_thesis_points():
        elements.append(Paragraph(f"• {point}", styles["BodyText"]))

    elements.append(Spacer(1, 0.15 * inch))

    add_section(
        elements,
        "How to Read This Report",
        build_how_to_read_table(),
        paragraph=(
            "This section explains how to interpret the report outputs. The model is designed to move from raw inputs "
            "to structured scorecards, valuation outputs, factor exposure analysis, risk adjustments, downside-risk estimates, "
            "and a final signal."
        ),
        max_rows=14,
        font_size=5.8,
        max_text_length=90,
        max_columns=3,
    )

    add_section(
        elements,
        "Model Inputs, Process, and Outputs",
        build_model_process_table(),
        paragraph=(
            "The report follows a repeatable workflow: collect source and market inputs, process them into model tables, "
            "run valuation, factor, and risk modules, then summarize the results in PDF-ready outputs."
        ),
        max_rows=10,
        font_size=5.8,
        max_text_length=95,
        max_columns=4,
    )

    add_section(
        elements,
        "Signal / Output Field Guide",
        build_signal_guide_table(),
        paragraph=(
            "Signals are qualitative labels generated from quantitative and source-driven model outputs. "
            "They are intended to make the report easier to scan while preserving the underlying numeric tables."
        ),
        max_rows=10,
        font_size=6.0,
        max_text_length=90,
        max_columns=3,
        page_break_after=True,
    )

    add_section(elements, "Market Metrics", market_metrics, max_rows=22, max_columns=4)
    add_optional_section(elements, "Market Snapshot", market_snapshot, max_rows=12, max_columns=5)

    add_optional_section(
        elements,
        "GARCH / EWMA Volatility & VaR Summary",
        garch_var_summary,
        paragraph=(
            "This section uses an actual GARCH model when the arch package is available and enabled. "
            "If GARCH fitting is unavailable or fails, the module falls back to EWMA volatility. "
            "The table summarizes model selection, realized volatility, historical VaR, parametric VaR, "
            "primary-model VaR, EWMA reference VaR, expected shortfall, and maximum observed daily loss."
        ),
        max_rows=18,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "Volatility Regime Summary",
        volatility_regime_summary,
        paragraph=(
            "The volatility regime summary compares the latest primary annualized volatility estimate "
            "with its historical distribution. The primary model is GARCH when available and EWMA fallback otherwise."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "VaR Backtest Summary",
        var_backtest_summary,
        paragraph=(
            "The VaR backtest compares realized daily losses with primary-model VaR thresholds and reports "
            "exception counts, exception rates, and Kupiec proportion-of-failures test outputs."
        ),
        max_rows=10,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "Defense Equity Factor Exposure Summary",
        factor_exposure_summary,
        paragraph=(
            "The factor exposure model estimates how much of CACI's daily return variation is explained by market, "
            "defense, technology, interest-rate, and volatility/risk-sentiment factors."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "Defense Equity Factor Regression Results",
        factor_regression_results,
        paragraph=(
            "This table reports estimated factor betas / sensitivities from the multi-factor return model."
        ),
        max_rows=8,
        font_size=5.9,
        max_text_length=85,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "Factor Correlation Summary",
        factor_correlation_summary,
        paragraph=(
            "This table shows CACI's simple daily return correlation with each selected factor."
        ),
        max_rows=8,
        font_size=5.9,
        max_text_length=85,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "Factor Contribution Summary",
        factor_contribution_summary,
        paragraph=(
            "This table estimates descriptive annualized factor contribution using average factor returns and estimated sensitivities. "
            "It is an attribution view, not a forward forecast. Rate and volatility factors are excluded from this contribution table "
            "because they are modeled as level-change factors rather than return-based factors."
        ),
        max_rows=8,
        font_size=5.7,
        max_text_length=80,
        max_columns=6,
        page_break_after=True,
    )

    add_section(elements, "Fundamental Summary", fundamental_summary, max_rows=14, max_columns=6)
    add_optional_section(elements, "Business Quality Summary", business_quality, max_rows=10, max_columns=7)
    add_optional_section(elements, "Business Quality Score Summary", business_quality_score, max_rows=8, max_columns=4)
    add_optional_section(elements, "Valuation Driver Summary", valuation_drivers, max_rows=10, max_columns=5)
    add_optional_section(elements, "Investment Thesis Points", investment_thesis_points, max_rows=10, max_columns=2)

    elements.append(PageBreak())

    add_section(
        elements,
        "Source Catalog",
        source_tables.get("source_catalog"),
        paragraph=(
            "The model uses company filings, earnings materials, investor presentations, government budget sources, "
            "defense macro sources, and market data sources. SEC filings and company earnings releases receive the "
            "highest weighting for historical and guidance inputs."
        ),
        max_rows=10,
        font_size=5.6,
        max_text_length=72,
        max_columns=9,
    )

    add_section(
        elements,
        "Source Quality Review",
        source_tables.get("source_quality"),
        paragraph=(
            "The source quality table ranks the reliability and valuation relevance of the main sources used across the CACI model."
        ),
        max_rows=10,
        font_size=6.0,
        max_text_length=85,
        max_columns=7,
    )

    add_optional_section(
        elements,
        "Source Coverage Summary",
        source_tables.get("source_coverage_summary"),
        paragraph=(
            "The source coverage summary evaluates source depth, URL coverage, and average reliability across model inputs."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_section(
        elements,
        "Model Input Audit",
        source_tables.get("model_input_audit"),
        paragraph=(
            "The model input audit links key valuation and intelligence assumptions to their primary sources."
        ),
        max_rows=10,
        font_size=5.8,
        max_text_length=85,
        max_columns=7,
    )

    add_optional_section(
        elements,
        "Input Confidence Summary",
        source_tables.get("input_confidence_summary"),
        paragraph="This table summarizes confidence levels across core model inputs.",
        max_rows=8,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "Missing Source URL Review",
        source_tables.get("missing_source_urls"),
        paragraph="This table flags whether any source catalog entries are missing configured URLs.",
        max_rows=8,
        max_columns=6,
        page_break_after=True,
    )

    add_section(
        elements,
        "Management Guidance Inputs",
        source_tables.get("management_guidance"),
        max_rows=12,
        font_size=5.9,
        max_text_length=80,
        max_columns=7,
    )

    add_section(
        elements,
        "Backlog and Pipeline Indicators",
        source_tables.get("backlog_pipeline"),
        paragraph=(
            "Backlog, book-to-bill, award duration, and submitted bids are used as qualitative support for forward revenue visibility."
        ),
        max_rows=12,
        font_size=6.0,
        max_text_length=80,
        max_columns=6,
    )

    add_section(
        elements,
        "Revenue Mix and Business Exposure",
        source_tables.get("revenue_mix"),
        max_rows=12,
        font_size=6.0,
        max_text_length=78,
        max_columns=6,
        page_break_after=True,
    )

    add_section(
        elements,
        "Capital Allocation Summary",
        source_tables.get("capital_allocation"),
        max_rows=10,
        max_columns=5,
    )

    add_section(
        elements,
        "Key Risk Factors",
        source_tables.get("risk_factors"),
        max_rows=10,
        font_size=6.0,
        max_text_length=85,
        max_columns=5,
        page_break_after=True,
    )

    add_section(
        elements,
        "Defense Risk Engine Summary",
        defense_risk_tables.get("defense_risk_summary"),
        paragraph=(
            "The defense risk engine scores CACI-specific risks including federal budget risk, contract recompete risk, "
            "backlog visibility, customer concentration, acquisition integration, labor constraints, margin execution, "
            "and interest-rate sensitivity."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_section(
        elements,
        "Defense Risk Components",
        defense_risk_tables.get("defense_risk_components"),
        max_rows=12,
        font_size=5.7,
        max_text_length=80,
        max_columns=8,
    )

    add_section(
        elements,
        "Defense Risk Valuation Adjustments",
        defense_risk_tables.get("defense_risk_adjustments"),
        max_rows=8,
        max_columns=4,
        page_break_after=False,
    )

    add_section(
        elements,
        "U.S. Intelligence Spending Signal",
        intelligence_spending_tables.get("intelligence_spending_signal"),
        paragraph=(
            "The intelligence spending model evaluates public NIP and MIP budget toplines and connects them to CACI's "
            "Intelligence Community, cyber, signals, space, and national-security exposure."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_section(
        elements,
        "Public Intelligence Budget Table",
        intelligence_spending_tables.get("intelligence_budget_table"),
        max_rows=8,
        font_size=5.7,
        max_text_length=70,
        max_columns=9,
    )

    add_section(
        elements,
        "CACI Intelligence Exposure",
        intelligence_spending_tables.get("caci_intelligence_exposure"),
        max_rows=8,
        max_columns=5,
    )

    add_optional_section(
        elements,
        "Intelligence Exposure Components",
        intelligence_spending_tables.get("intelligence_exposure_components"),
        max_rows=10,
        font_size=5.8,
        max_text_length=78,
        max_columns=8,
    )

    add_optional_section(
        elements,
        "Intelligence Exposure Summary",
        intelligence_spending_tables.get("intelligence_exposure_summary"),
        max_rows=8,
        max_columns=4,
        page_break_after=True,
    )

    add_section(
        elements,
        "Government Spending Summary",
        government_spending_tables.get("government_spending_summary"),
        paragraph=(
            "The government spending model evaluates the federal funding environment most relevant to CACI, including "
            "defense appropriations, intelligence funding relevance, federal civilian modernization, contract award conditions, "
            "appropriations timing risk, and budget concentration."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_section(
        elements,
        "Government Spending Table",
        government_spending_tables.get("government_spending_table"),
        max_rows=8,
        font_size=5.8,
        max_text_length=78,
        max_columns=7,
    )

    add_section(
        elements,
        "CACI Government Exposure",
        government_spending_tables.get("caci_government_exposure"),
        max_rows=8,
        max_columns=5,
    )

    add_section(
        elements,
        "Government Spending Components",
        government_spending_tables.get("government_spending_components"),
        max_rows=12,
        font_size=5.8,
        max_text_length=78,
        max_columns=8,
    )

    add_section(
        elements,
        "Government Spending Valuation Adjustments",
        government_spending_tables.get("government_spending_adjustments"),
        max_rows=8,
        max_columns=4,
        page_break_after=True,
    )

    add_section(
        elements,
        "Defense Macro Summary",
        defense_macro_tables.get("defense_macro_summary"),
        paragraph=(
            "The defense macro model evaluates the broader demand environment for CACI, including U.S. defense spending, "
            "intelligence funding, global military spending, cyber/intelligence demand, great-power competition, "
            "appropriations risk, and defense technology positioning."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_section(
        elements,
        "Defense Macro Components",
        defense_macro_tables.get("defense_macro_components"),
        max_rows=12,
        max_columns=5,
    )

    add_section(
        elements,
        "CACI Macro Exposure Bridge",
        defense_macro_tables.get("caci_macro_exposure"),
        max_rows=8,
        max_columns=5,
    )

    add_section(
        elements,
        "Defense Macro Valuation Adjustments",
        defense_macro_tables.get("defense_macro_adjustments"),
        max_rows=8,
        max_columns=4,
        page_break_after=True,
    )

    add_section(
        elements,
        "News Intelligence Summary",
        news_intelligence_tables.get("news_intelligence_summary"),
        paragraph=(
            "The news intelligence model classifies CACI, defense, intelligence, government spending, and geopolitical "
            "headlines by theme, sentiment, relevance, and valuation signal."
        ),
        max_rows=10,
        max_columns=4,
    )

    add_section(
        elements,
        "News Theme Summary",
        news_intelligence_tables.get("news_theme_summary"),
        max_rows=12,
        max_columns=6,
    )

    add_section(
        elements,
        "Classified News Headlines",
        news_intelligence_tables.get("classified_news"),
        max_rows=12,
        font_size=5.6,
        max_text_length=70,
        max_columns=9,
    )

    add_section(
        elements,
        "News Valuation Adjustments",
        news_intelligence_tables.get("news_valuation_adjustments"),
        max_rows=8,
        max_columns=4,
        page_break_after=True,
    )

    add_section(
        elements,
        "DCF Valuation Summary",
        dcf_summary_report if not dcf_summary_report.empty else dcf_summary,
        paragraph=(
            "The DCF valuation uses revenue growth, EBITDA margin, tax rate, depreciation and amortization, capex, working capital, WACC, "
            "and terminal growth assumptions to estimate enterprise value, equity value, and fair value per share."
        ),
        max_rows=8,
        font_size=5.6,
        max_text_length=70,
        max_columns=10,
    )

    add_optional_section(
        elements,
        "DCF Assumptions Audit",
        dcf_assumptions,
        paragraph="This table summarizes scenario-level and model-wide DCF assumptions.",
        max_rows=18,
        max_columns=5,
    )

    add_optional_section(
        elements,
        "Base Case Valuation Bridge",
        dcf_base_case_valuation_bridge,
        paragraph=(
            "This bridge connects explicit free cash flow, terminal value, enterprise value, net debt, equity value, "
            "and fair value per share."
        ),
        max_rows=12,
        max_columns=4,
    )

    if not dcf_sensitivity.empty:
        add_section(
            elements,
            "DCF Sensitivity Table",
            dcf_sensitivity,
            paragraph=(
                "This table shows base-case fair value per share sensitivity across WACC and terminal growth assumptions."
            ),
            max_rows=12,
            max_columns=7,
        )

    add_optional_section(
        elements,
        "Defense Contractor Stress Test Score",
        stress_test_score,
        paragraph=(
            "The stress-test score summarizes CACI's valuation resilience across downside scenarios. "
            "The score is derived from average downside versus the base-case DCF fair value."
        ),
        max_rows=8,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "Defense Contractor Stress Test Summary",
        stress_test_summary,
        paragraph=(
            "The stress-test module evaluates downside valuation sensitivity across federal budget delay, "
            "continuing resolution / award timing delay, higher WACC / rate shock, margin compression, "
            "contract recompete loss, revenue growth slowdown, and a combined downside case."
        ),
        max_rows=10,
        font_size=5.6,
        max_text_length=78,
        max_columns=9,
    )

    add_optional_section(
        elements,
        "Stress Test Valuation Bridge",
        stress_test_bridge,
        paragraph=(
            "This bridge compares each stress-case fair value with the base-case fair value and shows the "
            "dollar and percentage impact of each downside case."
        ),
        max_rows=10,
        font_size=5.8,
        max_text_length=80,
        max_columns=8,
    )

    add_optional_section(
        elements,
        "Stress Test Assumptions",
        stress_test_assumptions,
        paragraph=(
            "This table documents the revenue growth, EBITDA margin, WACC, and terminal growth adjustments "
            "used in each stress scenario."
        ),
        max_rows=10,
        font_size=5.6,
        max_text_length=80,
        max_columns=8,
    )

    add_section(
        elements,
        "Forecast Summary",
        forecast_tables.get("dcf_forecast_summary"),
        paragraph=(
            "The forecasting section links the DCF scenario framework to revenue, fair value, and simulated price outcomes."
        ),
        max_rows=10,
        max_columns=9,
    )

    add_section(
        elements,
        "Revenue Forecast Detail",
        forecast_tables.get("revenue_forecast_detail"),
        max_rows=15,
        max_columns=6,
    )

    add_section(
        elements,
        "Price Forecast Summary",
        forecast_tables.get("price_forecast_summary"),
        max_rows=14,
        max_columns=4,
        page_break_after=True,
    )

    add_section(
        elements,
        "Yahoo Finance Analyst Estimate Tables",
        analyst_summary,
        paragraph=(
            "The model attempts to collect Yahoo Finance analysis-style tables through yfinance, including earnings estimates, "
            "revenue estimates, EPS trends, EPS revisions, growth estimates, and analyst price targets when available."
        ),
        max_rows=12,
        max_columns=5,
    )

    add_optional_section(
        elements,
        "Analyst Source Quality",
        analyst_source_quality,
        paragraph=(
            "This table evaluates the reliability of live and manual analyst inputs used in the project."
        ),
        max_rows=10,
        max_columns=5,
    )

    add_optional_section(
        elements,
        "Manual Analyst / Guidance Inputs",
        manual_analyst_inputs,
        paragraph=(
            "This fallback table preserves important forward-looking assumptions when live Yahoo Finance estimate tables "
            "are unavailable or incomplete."
        ),
        max_rows=10,
        max_columns=5,
    )

    add_section(
        elements,
        "Monte Carlo Valuation Summary",
        mc_summary,
        paragraph=(
            "The Monte Carlo model simulates fair value per share using randomized growth, margin, WACC, and terminal growth assumptions."
        ),
        max_rows=18,
        max_columns=4,
    )

    add_section(
        elements,
        "Expected Return Summary",
        expected_return_tables.get("expected_return_summary"),
        paragraph=(
            "The expected return model compares current market price to DCF scenario values, Monte Carlo output, "
            "intelligence score, stress resilience adjustment, defense risk adjustments, and news intelligence adjustments."
        ),
        max_rows=18,
        max_columns=5,
    )

    add_section(
        elements,
        "Expected Return Bridge",
        expected_return_tables.get("expected_return_bridge"),
        max_rows=14,
        max_columns=5,
        page_break_after=True,
    )

    add_section(
        elements,
        "Change Detection Summary",
        change_detection_tables.get("change_summary"),
        paragraph=(
            "The change detection module compares this run against the prior saved snapshot to highlight model movement."
        ),
        max_rows=10,
        max_columns=3,
    )

    add_section(
        elements,
        "Change Detection Detail",
        change_detection_tables.get("change_detection"),
        max_rows=16,
        max_columns=7,
    )

    add_section(
        elements,
        "Defense Technology Intelligence Score",
        score_df,
        paragraph=(
            "The intelligence score combines market risk, valuation support, business quality, defense technology exposure, "
            "and stress-test resilience. Stress testing now directly affects the final score through the Stress Resilience component."
        ),
        max_rows=12,
        max_columns=8,
        page_break_after=True,
    )

    elements.append(Paragraph("Charts", styles["Section"]))

    for path in chart_paths:
        add_chart(elements, path)

    elements.append(PageBreak())

    add_section(
        elements,
        "Peer Comparison",
        peer_df,
        paragraph=(
            "Peer data is collected through yfinance and may vary by data availability. "
            "The peer group includes government services, defense technology, and aerospace/defense companies."
        ),
        max_rows=12,
        font_size=5.7,
        max_text_length=70,
        max_columns=10,
    )

    add_optional_section(
        elements,
        "Peer Positioning Summary",
        peer_positioning_summary,
        paragraph="This table summarizes CACI's composite rank and valuation positioning versus the peer group.",
        max_rows=10,
        max_columns=4,
    )

    add_optional_section(
        elements,
        "CACI vs. Peer Median Summary",
        peer_summary,
        max_rows=18,
        max_columns=5,
    )

    add_optional_section(
        elements,
        "Peer Ranking Table",
        peer_ranking,
        max_rows=12,
        max_columns=10,
        page_break_after=False,
    )

    elements.append(Spacer(1, 0.20 * inch))

    elements.append(
        Paragraph(
            "Important: This report is for educational and portfolio demonstration purposes only. "
            "It is not investment advice. Assumptions should be reviewed and updated using the latest filings, "
            "earnings releases, market data, government budget sources, and analyst estimates.",
            styles["Small"],
        )
    )

    doc.build(elements)

    return str(output_path)