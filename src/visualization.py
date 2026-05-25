import matplotlib.pyplot as plt
import pandas as pd

from src.config import (
    TICKER,
    BENCHMARK,
    BENCHMARK_NAME,
    COMPANY_NAME,
    CHARTS_DIR,
    TABLES_DIR,
)


def _safe_savefig(path) -> str | None:
    """Safely save the current matplotlib figure."""

    try:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        return str(path)
    except Exception as exc:
        print(f"Warning: could not save chart {path}: {exc}")
        plt.close()
        return None


def _has_columns(df: pd.DataFrame, columns: list[str]) -> bool:
    """Check whether a DataFrame has required columns."""

    if df is None or df.empty:
        return False

    return all(column in df.columns for column in columns)


def _is_metadata_column(column: str) -> bool:
    """Identify metadata columns that should be ignored by chart logic."""

    name = str(column).strip()
    lower = name.lower()

    return (
        lower.startswith("raw ")
        or lower in {"value type", "driver value type"}
        or lower.endswith(" type")
        or lower.endswith(" value type")
    )


def _currency_axis(axis: str = "y"):
    """Format current axis as dollars."""

    formatter = plt.FuncFormatter(lambda value, _: f"${value:,.0f}")

    if axis == "x":
        plt.gca().xaxis.set_major_formatter(formatter)
    else:
        plt.gca().yaxis.set_major_formatter(formatter)


def _percent_axis(axis: str = "y"):
    """Format current axis as percentages."""

    formatter = plt.FuncFormatter(lambda value, _: f"{value:.0%}")

    if axis == "x":
        plt.gca().xaxis.set_major_formatter(formatter)
    else:
        plt.gca().yaxis.set_major_formatter(formatter)


def save_price_chart(prices: pd.DataFrame) -> str | None:
    """Save CACI price chart."""

    path = CHARTS_DIR / "price_chart.png"

    if not _has_columns(prices, [TICKER]):
        print("Warning: price chart skipped because price data is unavailable.")
        return None

    chart_data = prices[[TICKER]].dropna()

    if chart_data.empty:
        print("Warning: price chart skipped because CACI price data is empty.")
        return None

    plt.figure(figsize=(10, 5))
    plt.plot(chart_data.index, chart_data[TICKER], label=TICKER)
    plt.title(f"{COMPANY_NAME} Share Price")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Price")
    _currency_axis("y")
    plt.legend()

    return _safe_savefig(path)


def save_cumulative_return_chart(prices: pd.DataFrame) -> str | None:
    """Save cumulative return comparison chart."""

    path = CHARTS_DIR / "cumulative_return_chart.png"

    if not _has_columns(prices, [TICKER, BENCHMARK]):
        print("Warning: cumulative return chart skipped because price data is incomplete.")
        return None

    returns = prices[[TICKER, BENCHMARK]].pct_change(fill_method=None).dropna()

    if returns.empty:
        print("Warning: cumulative return chart skipped because return data is empty.")
        return None

    cumulative = (1 + returns).cumprod() - 1

    plt.figure(figsize=(10, 5))
    plt.plot(cumulative.index, cumulative[TICKER], label=TICKER)
    plt.plot(cumulative.index, cumulative[BENCHMARK], label=BENCHMARK_NAME)
    plt.title("Cumulative Return Comparison")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    _percent_axis("y")
    plt.legend()

    return _safe_savefig(path)


def save_drawdown_chart(prices: pd.DataFrame) -> str | None:
    """Save drawdown chart."""

    path = CHARTS_DIR / "drawdown_chart.png"

    if not _has_columns(prices, [TICKER]):
        print("Warning: drawdown chart skipped because price data is unavailable.")
        return None

    returns = prices[TICKER].pct_change(fill_method=None).dropna()

    if returns.empty:
        print("Warning: drawdown chart skipped because return data is empty.")
        return None

    wealth = (1 + returns).cumprod()
    peak = wealth.cummax()
    drawdown = wealth / peak - 1

    plt.figure(figsize=(10, 5))
    plt.plot(drawdown.index, drawdown)
    plt.title(f"{COMPANY_NAME} Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    _percent_axis("y")

    return _safe_savefig(path)


def save_dcf_chart(dcf_summary: pd.DataFrame) -> str | None:
    """Save DCF scenario fair value chart."""

    path = CHARTS_DIR / "dcf_scenario_chart.png"

    required_columns = ["Scenario", "Fair Value Per Share"]

    if not _has_columns(dcf_summary, required_columns):
        print("Warning: DCF chart skipped because DCF summary is incomplete.")
        return None

    chart_data = dcf_summary[required_columns].copy()
    chart_data["Fair Value Per Share"] = pd.to_numeric(
        chart_data["Fair Value Per Share"],
        errors="coerce",
    )
    chart_data = chart_data.dropna(subset=["Fair Value Per Share"])

    if chart_data.empty:
        print("Warning: DCF chart skipped because DCF fair value data is empty.")
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(chart_data["Scenario"], chart_data["Fair Value Per Share"])

    base_row = chart_data.loc[
        chart_data["Scenario"].astype(str).str.lower() == "base"
    ]

    if not base_row.empty:
        base_value = base_row["Fair Value Per Share"].iloc[0]
        plt.axhline(
            base_value,
            linestyle="--",
            linewidth=1.2,
            label=f"Base Case: ${base_value:,.0f}",
        )
        plt.legend()

    plt.title("DCF Fair Value Per Share by Scenario")
    plt.xlabel("Scenario")
    plt.ylabel("Fair Value Per Share")
    _currency_axis("y")

    return _safe_savefig(path)


def save_monte_carlo_chart(mc_values: pd.DataFrame) -> str | None:
    """Save Monte Carlo fair value distribution chart."""

    path = CHARTS_DIR / "monte_carlo_distribution.png"

    if not _has_columns(mc_values, ["Fair Value Per Share"]):
        print("Warning: Monte Carlo chart skipped because simulation data is incomplete.")
        return None

    values = pd.to_numeric(
        mc_values["Fair Value Per Share"],
        errors="coerce",
    ).dropna()

    values = values[values > 0]

    if values.empty:
        print("Warning: Monte Carlo chart skipped because fair value data is empty.")
        return None

    median_value = values.median()
    p10 = values.quantile(0.10)
    p90 = values.quantile(0.90)

    plt.figure(figsize=(10, 5))
    plt.hist(values, bins=50)
    plt.axvline(
        median_value,
        linestyle="--",
        linewidth=1.5,
        label=f"Median: ${median_value:,.0f}",
    )
    plt.axvline(
        p10,
        linestyle=":",
        linewidth=1.2,
        label=f"P10: ${p10:,.0f}",
    )
    plt.axvline(
        p90,
        linestyle=":",
        linewidth=1.2,
        label=f"P90: ${p90:,.0f}",
    )
    plt.title("Monte Carlo Fair Value Distribution")
    plt.xlabel("Fair Value Per Share")
    plt.ylabel("Frequency")
    _currency_axis("x")
    plt.legend()

    return _safe_savefig(path)


def save_intelligence_score_chart(score_df: pd.DataFrame) -> str | None:
    """Save Defense Technology Intelligence Score chart."""

    path = CHARTS_DIR / "defense_intelligence_score.png"

    if not _has_columns(score_df, ["Component", "Score"]):
        print("Warning: intelligence score chart skipped because score data is incomplete.")
        return None

    components = score_df[score_df["Component"] != "Overall Score"].copy()
    components["Score"] = pd.to_numeric(components["Score"], errors="coerce")
    components = components.dropna(subset=["Score"])
    components = components.sort_values("Score")

    if components.empty:
        print("Warning: intelligence score chart skipped because component scores are empty.")
        return None

    plt.figure(figsize=(10, 5))
    plt.barh(components["Component"], components["Score"])
    plt.title("CACI Defense Technology Intelligence Score")
    plt.xlabel("Score")
    plt.xlim(0, 100)

    return _safe_savefig(path)


def save_dcf_sensitivity_chart() -> str | None:
    """Save DCF WACC / terminal growth sensitivity heatmap."""

    path = CHARTS_DIR / "dcf_sensitivity_heatmap.png"
    sensitivity_path = TABLES_DIR / "dcf_sensitivity.csv"

    if not sensitivity_path.exists():
        print("Warning: DCF sensitivity chart skipped because dcf_sensitivity.csv does not exist.")
        return None

    sensitivity = pd.read_csv(sensitivity_path)

    if sensitivity.empty or "Terminal Growth" not in sensitivity.columns:
        print("Warning: DCF sensitivity chart skipped because sensitivity table is incomplete.")
        return None

    sensitivity = sensitivity.copy()
    sensitivity["Terminal Growth"] = pd.to_numeric(
        sensitivity["Terminal Growth"],
        errors="coerce",
    )

    wacc_columns = [
        column
        for column in sensitivity.columns
        if str(column).startswith("WACC")
        and not _is_metadata_column(column)
    ]

    if not wacc_columns:
        print("Warning: DCF sensitivity chart skipped because no WACC columns were found.")
        return None

    heatmap_data = sensitivity[["Terminal Growth"] + wacc_columns].dropna(
        subset=["Terminal Growth"]
    )

    for column in wacc_columns:
        heatmap_data[column] = pd.to_numeric(
            heatmap_data[column],
            errors="coerce",
        )

    heatmap_data = heatmap_data.set_index("Terminal Growth")

    if heatmap_data.empty:
        print("Warning: DCF sensitivity chart skipped because heatmap data is empty.")
        return None

    heatmap_data = heatmap_data.dropna(how="all")

    if heatmap_data.empty:
        print("Warning: DCF sensitivity chart skipped because all sensitivity values are missing.")
        return None

    plt.figure(figsize=(10, 5))
    image = plt.imshow(heatmap_data.values, aspect="auto")
    plt.colorbar(image, label="Fair Value Per Share")

    plt.title("DCF Sensitivity: WACC vs. Terminal Growth")
    plt.xlabel("WACC")
    plt.ylabel("Terminal Growth")

    plt.xticks(
        ticks=range(len(heatmap_data.columns)),
        labels=heatmap_data.columns,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=range(len(heatmap_data.index)),
        labels=[f"{x:.1%}" for x in heatmap_data.index],
    )

    for i in range(heatmap_data.shape[0]):
        for j in range(heatmap_data.shape[1]):
            value = heatmap_data.iloc[i, j]

            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"${value:,.0f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    return _safe_savefig(path)


def save_peer_ranking_chart() -> str | None:
    """Save optional peer ranking chart if peer_ranking.csv exists."""

    path = CHARTS_DIR / "peer_ranking_chart.png"
    ranking_path = TABLES_DIR / "peer_ranking.csv"

    if not ranking_path.exists():
        return None

    ranking = pd.read_csv(ranking_path)

    required_columns = ["Ticker", "Average Rank"]

    if not _has_columns(ranking, required_columns):
        return None

    ranking["Average Rank"] = pd.to_numeric(
        ranking["Average Rank"],
        errors="coerce",
    )

    ranking = ranking.dropna(subset=["Average Rank"]).sort_values(
        "Average Rank",
        ascending=False,
    )

    if ranking.empty:
        return None

    plt.figure(figsize=(10, 5))
    plt.barh(ranking["Ticker"], ranking["Average Rank"])
    plt.title("Peer Ranking: Average Rank")
    plt.xlabel("Average Rank")
    plt.ylabel("Ticker")

    return _safe_savefig(path)


def _clean_bridge_step(step: str) -> str:
    """Remove leading numbering from an expected return bridge step label."""

    text = str(step).strip()

    if ". " in text and text.split(". ", 1)[0].isdigit():
        return text.split(". ", 1)[1]

    return text


def save_expected_return_bridge_chart() -> str | None:
    """Save expected return bridge as a waterfall chart.

    The expected return bridge contains dollar rows and percentage rows.
    This chart keeps only the percentage bridge rows and displays:
    - Raw Expected Return as the starting value
    - Risk / news adjustments as floating bars
    - Risk-Adjusted Expected Return as the ending value
    """

    path = CHARTS_DIR / "expected_return_bridge_chart.png"
    bridge_path = TABLES_DIR / "expected_return_bridge.csv"

    if not bridge_path.exists():
        return None

    bridge = pd.read_csv(bridge_path)

    required_columns = ["Step", "Value"]

    if not _has_columns(bridge, required_columns):
        return None

    bridge = bridge.copy()

    if "Value Type" in bridge.columns:
        bridge = bridge[
            bridge["Value Type"].astype(str).str.lower().isin(
                ["percent", "percentage", "rate"]
            )
        ].copy()
    else:
        keep_steps = [
            "3. Raw Expected Return",
            "4. Intelligence Score Haircut",
            "5. Defense Risk Adjustment",
            "6. News Valuation Adjustment",
            "7. Risk-Adjusted Expected Return",
        ]

        bridge = bridge[bridge["Step"].isin(keep_steps)].copy()

    bridge["Value"] = pd.to_numeric(bridge["Value"], errors="coerce")
    bridge = bridge.dropna(subset=["Step", "Value"])

    if bridge.empty:
        return None

    raw_row = bridge[
        bridge["Step"].astype(str).str.contains("Raw Expected Return", case=False, na=False)
    ]

    final_row = bridge[
        bridge["Step"].astype(str).str.contains("Risk-Adjusted Expected Return", case=False, na=False)
    ]

    adjustment_rows = bridge[
        bridge["Step"].astype(str).str.contains("Adjustment|Haircut", case=False, na=False)
    ].copy()

    if raw_row.empty or final_row.empty:
        print("Warning: expected return waterfall skipped because raw or final return row is missing.")
        return None

    raw_value = float(raw_row["Value"].iloc[0])
    final_value = float(final_row["Value"].iloc[0])

    steps = ["Raw Expected Return"]
    values = [raw_value]
    bottoms = [0.0]
    labels = [f"{raw_value:.1%}"]

    running_total = raw_value

    for _, row in adjustment_rows.iterrows():
        step_label = _clean_bridge_step(row["Step"])
        adjustment = float(row["Value"])

        if adjustment >= 0:
            bottom = running_total
        else:
            bottom = running_total + adjustment

        steps.append(step_label)
        values.append(abs(adjustment))
        bottoms.append(bottom)
        labels.append(f"{adjustment:+.1%}")

        running_total += adjustment

    steps.append("Risk-Adjusted Expected Return")
    values.append(final_value)
    bottoms.append(0.0)
    labels.append(f"{final_value:.1%}")

    plt.figure(figsize=(11, 5.5))
    bars = plt.bar(
        steps,
        values,
        bottom=bottoms,
    )

    for i in range(len(steps) - 1):
        current_top = bottoms[i] + values[i]

        if i == 0:
            connector_y = current_top
        elif i < len(steps) - 2:
            connector_y = bottoms[i] + values[i] if labels[i].startswith("+") else bottoms[i]
        else:
            connector_y = current_top

        next_left = i + 0.40
        next_right = i + 1 - 0.40

        plt.plot(
            [next_left, next_right],
            [connector_y, connector_y],
            linestyle="--",
            linewidth=0.8,
        )

    for bar, label, bottom, value in zip(bars, labels, bottoms, values):
        y = bottom + value

        if label.startswith("-"):
            y = bottom

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            label,
            ha="center",
            va="bottom" if not label.startswith("-") else "top",
            fontsize=9,
        )

    plt.axhline(0, linewidth=0.8)
    plt.title("Expected Return Waterfall Bridge")
    plt.ylabel("Expected Return")
    plt.xticks(rotation=25, ha="right")
    _percent_axis("y")

    return _safe_savefig(path)


def save_price_forecast_chart() -> str | None:
    """Save optional price forecast chart if price_forecast_summary.csv exists."""

    path = CHARTS_DIR / "price_forecast_summary_chart.png"
    forecast_path = TABLES_DIR / "price_forecast_summary.csv"

    if not forecast_path.exists():
        return None

    forecast = pd.read_csv(forecast_path)

    required_columns = ["Metric", "Value"]

    if not _has_columns(forecast, required_columns):
        return None

    keep_metrics = [
        "Median Forecast Price",
        "P10 Forecast Price",
        "P90 Forecast Price",
    ]

    chart_data = forecast[forecast["Metric"].isin(keep_metrics)].copy()
    chart_data["Value"] = pd.to_numeric(chart_data["Value"], errors="coerce")
    chart_data = chart_data.dropna(subset=["Value"])

    if chart_data.empty:
        return None

    plt.figure(figsize=(8, 5))
    plt.bar(chart_data["Metric"], chart_data["Value"])
    plt.title("Price Forecast Summary")
    plt.xlabel("Metric")
    plt.ylabel("Forecast Price")
    plt.xticks(rotation=20, ha="right")
    _currency_axis("y")

    return _safe_savefig(path)


def save_factor_exposure_chart() -> str | None:
    """Save Defense Equity Factor Exposure chart if factor_regression_results.csv exists."""

    path = CHARTS_DIR / "factor_exposure_chart.png"
    factor_path = TABLES_DIR / "factor_regression_results.csv"

    if not factor_path.exists():
        return None

    factors = pd.read_csv(factor_path)

    required_columns = ["Factor", "Beta / Sensitivity"]

    if not _has_columns(factors, required_columns):
        return None

    chart_data = factors[required_columns].copy()
    chart_data["Beta / Sensitivity"] = pd.to_numeric(
        chart_data["Beta / Sensitivity"],
        errors="coerce",
    )

    chart_data = chart_data.dropna(subset=["Factor", "Beta / Sensitivity"])

    if chart_data.empty:
        return None

    chart_data = chart_data.sort_values("Beta / Sensitivity")

    plt.figure(figsize=(10, 5))
    plt.barh(chart_data["Factor"], chart_data["Beta / Sensitivity"])
    plt.axvline(0, linewidth=0.8)
    plt.title("Defense Equity Factor Exposure Model")
    plt.xlabel("Beta / Sensitivity")
    plt.ylabel("Factor")

    return _safe_savefig(path)


def save_stress_test_fair_value_chart() -> str | None:
    """Save stress-test fair value per share chart if stress_test_summary.csv exists."""

    path = CHARTS_DIR / "stress_test_fair_value_chart.png"
    stress_path = TABLES_DIR / "stress_test_summary.csv"

    if not stress_path.exists():
        return None

    stress = pd.read_csv(stress_path)

    required_columns = ["Stress Case", "Fair Value Per Share"]

    if not _has_columns(stress, required_columns):
        return None

    chart_data = stress[required_columns].copy()
    chart_data["Fair Value Per Share"] = pd.to_numeric(
        chart_data["Fair Value Per Share"],
        errors="coerce",
    )
    chart_data = chart_data.dropna(subset=["Stress Case", "Fair Value Per Share"])

    if chart_data.empty:
        return None

    base_row = chart_data[
        chart_data["Stress Case"].astype(str).str.lower() == "base case"
    ]

    base_value = None

    if not base_row.empty:
        base_value = float(base_row["Fair Value Per Share"].iloc[0])

    chart_data = chart_data.sort_values("Fair Value Per Share")

    plt.figure(figsize=(11, 5.5))
    plt.barh(chart_data["Stress Case"], chart_data["Fair Value Per Share"])

    if base_value is not None:
        plt.axvline(
            base_value,
            linestyle="--",
            linewidth=1.2,
            label=f"Base Case: ${base_value:,.0f}",
        )
        plt.legend()

    plt.title("Defense Contractor Stress Test: Fair Value Per Share")
    plt.xlabel("Fair Value Per Share")
    plt.ylabel("Stress Case")
    _currency_axis("x")

    return _safe_savefig(path)


def save_stress_test_downside_chart() -> str | None:
    """Save stress-test downside versus base case chart if stress_test_summary.csv exists."""

    path = CHARTS_DIR / "stress_test_downside_chart.png"
    stress_path = TABLES_DIR / "stress_test_summary.csv"

    if not stress_path.exists():
        return None

    stress = pd.read_csv(stress_path)

    required_columns = ["Stress Case", "Downside vs Base Case"]

    if not _has_columns(stress, required_columns):
        return None

    chart_data = stress[required_columns].copy()
    chart_data["Downside vs Base Case"] = pd.to_numeric(
        chart_data["Downside vs Base Case"],
        errors="coerce",
    )
    chart_data = chart_data.dropna(subset=["Stress Case", "Downside vs Base Case"])

    if chart_data.empty:
        return None

    chart_data = chart_data[
        chart_data["Stress Case"].astype(str).str.lower() != "base case"
    ].copy()

    if chart_data.empty:
        return None

    chart_data = chart_data.sort_values("Downside vs Base Case")

    plt.figure(figsize=(11, 5.5))
    plt.barh(chart_data["Stress Case"], chart_data["Downside vs Base Case"])
    plt.axvline(0, linewidth=0.8)
    plt.title("Defense Contractor Stress Test: Downside vs Base Case")
    plt.xlabel("Downside vs Base Case")
    plt.ylabel("Stress Case")
    _percent_axis("x")

    return _safe_savefig(path)


def create_all_charts(prices, dcf_summary, mc_values, score_df) -> list[str]:
    """Create all report charts and return valid chart paths."""

    chart_functions = [
        lambda: save_price_chart(prices),
        lambda: save_cumulative_return_chart(prices),
        lambda: save_drawdown_chart(prices),
        lambda: save_dcf_chart(dcf_summary),
        lambda: save_monte_carlo_chart(mc_values),
        lambda: save_intelligence_score_chart(score_df),
        save_dcf_sensitivity_chart,
        save_peer_ranking_chart,
        save_expected_return_bridge_chart,
        save_price_forecast_chart,
        save_factor_exposure_chart,
        save_stress_test_fair_value_chart,
        save_stress_test_downside_chart,
    ]

    chart_paths = []

    for chart_function in chart_functions:
        try:
            chart_path = chart_function()

            if chart_path is not None:
                chart_paths.append(chart_path)

        except Exception as exc:
            print(f"Warning: chart generation failed: {exc}")

    return chart_paths