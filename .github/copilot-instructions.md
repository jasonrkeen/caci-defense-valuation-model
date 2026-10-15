# Copilot instructions: CACI International Defense Technology Valuation & Intelligence Model

## Purpose
CACI defense technology valuation pipeline with DCF, stress testing, Monte Carlo, GARCH/VaR, factor exposure, and automated PDF reporting.

## Tech stack
- Python; dependencies in `requirements.txt`
- Key libraries: `yfinance`, `pandas`, `numpy`, `matplotlib`, `reportlab`, `requests`, `openpyxl`, `arch`

## Folder structure
- `reports/` (1 files)
- `src/` (24 files)
- `README.md`
- `requirements.txt`

## Setup, run, test
Commands documented in README.md (referenced files exist):
```bash
python -m venv .venv
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install arch
python -m src.main
```
- No automated tests exist yet; if adding them, use pytest under `tests/`.

## Conventions observed
- Functions use type hints; keep annotating new code.
- Config lives in: `src/config.py`; avoid hardcoding parameters elsewhere.
- Random seeds are set for reproducibility (Monte Carlo/simulation); keep seeded, deterministic runs.
- Raw data (`data/raw/`) is gitignored; download/fetch it via the pipeline, do not commit it.
- Public data sources referenced: SEC, yfinance, SIPRI.
- Write paths relative to the repo root (no absolute local paths).

## Output artifacts
- A PDF report is produced with `reportlab`; keep it in sync with model changes.
- Charts are generated with matplotlib.

## Guardrails
- Never commit secrets, API keys, tokens, or `.env` files.
- Do not commit large raw datasets or binaries; keep data fetched or documented with its source.
- Keep analysis source-backed: cite data sources and assumptions; label scenario/forecast outputs as model estimates, not facts.
- Preserve reproducibility: no unseeded randomness, no hidden manual steps.
