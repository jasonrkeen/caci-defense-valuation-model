from pathlib import Path

# ----------------------------
# Project settings
# ----------------------------

PROJECT_NAME = "CACI Defense Technology Valuation & Intelligence Pipeline"
REPORT_TITLE = "CACI International Valuation & Defense Technology Intelligence Report"
REPORT_SUBTITLE = (
    "DCF valuation, peer benchmarking, factor exposure modeling, defense contractor stress testing, "
    "Monte Carlo simulation, GARCH/EWMA VaR risk modeling, defense macro analysis, "
    "government spending signals, news intelligence, and expected return modeling"
)
REPORT_FILENAME = "caci_valuation_report.pdf"

AUTHOR_NAME = "Jason Keen"
AUTHOR_EMAIL = "keenjasonr@gmail.com"

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHARTS_DIR = OUTPUTS_DIR / "charts"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"

for directory in [RAW_DIR, PROCESSED_DIR, CHARTS_DIR, TABLES_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ----------------------------
# Market settings
# ----------------------------

TICKER = "CACI"
COMPANY_NAME = "CACI International Inc."
BENCHMARK = "^GSPC"
BENCHMARK_NAME = "S&P 500"
START_DATE = "2020-01-01"

# Treasury proxy:
# ^TNX = CBOE 10-Year Treasury Yield Index, quoted as yield * 10.
TREASURY_TICKER = "^TNX"
DEFAULT_RISK_FREE_RATE = 0.045

PEERS = {
    "LDOS": "Leidos",
    "SAIC": "Science Applications International",
    "BAH": "Booz Allen Hamilton",
    "GD": "General Dynamics",
    "NOC": "Northrop Grumman",
    "LHX": "L3Harris Technologies",
    "RTX": "RTX Corporation",
}


# ----------------------------
# Defense equity factor model settings
# ----------------------------

# Factor proxies used by factor_model.py.
# These are downloaded through yfinance and used to estimate CACI's exposure
# to market, defense, technology, rates, and volatility/risk sentiment factors.
FACTOR_MODEL_TICKERS = {
    BENCHMARK: BENCHMARK_NAME,
    "ITA": "Aerospace & Defense Factor",
    "XLK": "Technology Factor",
    "^TNX": "Interest Rate Factor",
    "^VIX": "Volatility / Risk Sentiment Factor",
}

FACTOR_MODEL_DISPLAY_ORDER = [
    BENCHMARK,
    "ITA",
    "XLK",
    "^TNX",
    "^VIX",
]


# ----------------------------
# Valuation assumptions
# ----------------------------

# Editable model base assumptions.
# These are deliberately centralized so the model can be updated quickly after new filings.
# Financial statement values are in millions unless noted otherwise.

BASE_REVENUE_M = 9550.0
BASE_EBITDA_MARGIN = 0.123
BASE_FCF_M = 725.0
NET_DEBT_M = 2900.0
DILUTED_SHARES_M = 22.2

TAX_RATE = 0.24
CAPEX_PCT_REVENUE = 0.010
D_AND_A_PCT_REVENUE = 0.0235
NWC_PCT_REVENUE_CHANGE = 0.015

WACC = 0.085
TERMINAL_GROWTH = 0.025

SCENARIOS = {
    "Bear": {
        "revenue_growth": [0.055, 0.045, 0.040, 0.035, 0.030],
        "ebitda_margin": 0.112,
        "wacc": 0.095,
        "terminal_growth": 0.020,
    },
    "Base": {
        "revenue_growth": [0.075, 0.065, 0.055, 0.045, 0.040],
        "ebitda_margin": 0.122,
        "wacc": 0.085,
        "terminal_growth": 0.025,
    },
    "Bull": {
        "revenue_growth": [0.095, 0.080, 0.070, 0.060, 0.050],
        "ebitda_margin": 0.132,
        "wacc": 0.078,
        "terminal_growth": 0.030,
    },
}


# ----------------------------
# Monte Carlo / forecasting settings
# ----------------------------

MONTE_CARLO_RUNS = 5000
PRICE_FORECAST_RUNS = 5000
PRICE_FORECAST_DAYS = 252
TRADING_DAYS = 252
RANDOM_SEED = 42


# ----------------------------
# GARCH / VaR risk settings
# ----------------------------

# If True, garch_var.py will attempt to use the arch package for a GARCH(1,1)
# volatility model. If arch is unavailable or the model fit fails, the module
# should fall back to EWMA volatility.
USE_ARCH_GARCH = True

# GARCH(p, q) settings.
GARCH_P = 1
GARCH_Q = 1

# Supported arch distributions include "normal", "t", "skewt", and "ged".
# Keep "normal" for a simple baseline model.
GARCH_DISTRIBUTION = "normal"

# EWMA fallback smoothing parameter.
EWMA_LAMBDA = 0.94

# VaR confidence levels used by the risk module.
VAR_CONFIDENCE_LEVELS = [0.95, 0.99]


# ----------------------------
# Expected return settings
# ----------------------------

EXPECTED_RETURN_PROBABILITIES = {
    "Bear": 0.25,
    "Base": 0.50,
    "Bull": 0.25,
}


# ----------------------------
# Intelligence / government spending assumptions
# ----------------------------

# Public U.S. intelligence budget toplines in billions.
# NIP = National Intelligence Program.
# MIP = Military Intelligence Program.
INTELLIGENCE_BUDGET_TABLE = [
    {
        "Fiscal Year": 2024,
        "NIP Appropriated": 72.4,
        "NIP Requested": 76.5,
        "MIP Appropriated": 29.3,
        "MIP Requested": 29.8,
        "Total Appropriated": 106.3,
    },
    {
        "Fiscal Year": 2025,
        "NIP Appropriated": 73.4,
        "NIP Requested": 73.3,
        "MIP Appropriated": 28.2,
        "MIP Requested": 27.8,
        "Total Appropriated": 101.1,
    },
    {
        "Fiscal Year": 2026,
        "NIP Appropriated": None,
        "NIP Requested": 81.9,
        "MIP Appropriated": None,
        "MIP Requested": None,
        "Total Appropriated": None,
    },
]

# Public government spending / company guidance figures in billions.
GOVERNMENT_SPENDING_TABLE = [
    {
        "Fiscal Year": 2026,
        "Spending Category": "Defense Appropriations",
        "Amount": 838.7,
        "Status": "Approved / enacted funding bill",
    },
    {
        "Fiscal Year": 2026,
        "Spending Category": "Defense Funding Component",
        "Amount": 838.5,
        "Status": "Defense funding within FY2026 defense appropriations bill",
    },
    {
        "Fiscal Year": 2026,
        "Spending Category": "Nondefense Component in Defense Bill",
        "Amount": 0.18,
        "Status": "Nondefense funding within FY2026 defense appropriations bill",
    },
    {
        "Fiscal Year": 2026,
        "Spending Category": "CACI FY2026 Revenue Guidance Midpoint",
        "Amount": 9.55,
        "Status": "Company guidance",
    },
    {
        "Fiscal Year": 2026,
        "Spending Category": "CACI FY2026 Free Cash Flow Guidance",
        "Amount": 0.725,
        "Status": "Company guidance; at least this amount",
    },
]

# CACI Q3 FY2026 revenue exposure.
CACI_CUSTOMER_MIX = {
    "Department of Defense": 0.530,
    "Intelligence Community": 0.250,
    "Federal Civilian Agencies": 0.178,
    "Commercial and Other": 0.042,
}

CACI_TECHNOLOGY_MIX = {
    "Technology Revenue Mix": 0.565,
    "Expertise Revenue Mix": 0.435,
    "Technology Revenue Growth": 0.112,
    "Expertise Revenue Growth": 0.052,
}


# ----------------------------
# News intelligence settings
# ----------------------------

NEWS_SENTIMENT_WEIGHTS = {
    "Sentiment Score": 0.55,
    "Relevance Score": 0.45,
}

# API-free public RSS news collection settings.
# These queries are used by news_intelligence.py to collect live headlines
# without requiring a paid news API or API key.
NEWS_QUERIES = [
    "CACI International",
    "CACI defense contract",
    "CACI cyber intelligence",
    "CACI government contract award",
    "defense appropriations cyber intelligence contractors",
    "U.S. intelligence budget defense contractors",
]

GOOGLE_NEWS_RSS_BASE = "https://news.google.com/rss/search"

YAHOO_FINANCE_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s=CACI&region=US&lang=en-US"
)

NEWS_MAX_ITEMS_PER_QUERY = 6
NEWS_MAX_LIVE_ROWS = 30


# ----------------------------
# Source URLs
# ----------------------------

SOURCE_URLS = {
    "CACI Investor Relations": "https://investor.caci.com/",
    "CACI Investor Presentation January 2026": "https://s21.q4cdn.com/708811725/files/doc_presentations/2025/CACI-Investor-Presentation-January-2026-FINAL-Previous-Template.pdf",
    "CACI Q3 FY2026 Earnings Slides": "https://s21.q4cdn.com/708811725/files/doc_financials/2026/q3/CACI-Q3-FY26-Earning-Slides-vFINAL.pdf",
    "CACI Q3 FY2026 Earnings Release": "https://investor.caci.com/news/news-details/2026/CACI-Reports-Results-for-Its-Fiscal-2026-Third-Quarter/default.aspx",
    "CACI FY2025 Form 10-K": "https://www.sec.gov/Archives/edgar/data/16058/000162828025038739/caci-20250630.htm",
    "Yahoo Finance Analyst Analysis": "https://finance.yahoo.com/quote/CACI/analysis/",
    "ODNI IC Budget": "https://www.dni.gov/index.php/what-we-do/ic-budget",
    "Senate FY2026 Defense Appropriations": "https://www.appropriations.senate.gov/news/majority/congress-approves-fy-2026-defense-appropriations-bill",
    "SIPRI Military Expenditure Database": "https://www.sipri.org/databases/milex",
}