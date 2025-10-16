# AlphaAnalyst Trading System v2.1 — Phase 1

Minimal Phase 1 Streamlit app with:
- Instruments onboarding & data explorer (Polygon preferred, yfinance fallback, or deterministic demo)
- Research PDF upload, text extraction, and embeddings (OpenAI or placeholder)
- Supabase helpers with lightweight schema creation and DB upserts/inserts

## Quick Start

1) Python 3.10+

2) Install packages:
```bash
pip install streamlit pandas yfinance pdfplumber supabase openai sqlalchemy psycopg2-binary requests
```

3) Set environment variables (PowerShell example):
```powershell
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_KEY="your-supabase-service-key"
$env:POLYGON_API_KEY="your-polygon-key"
$env:OPENAI_API_KEY="your-openai-key"
$env:DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/dbname"
```
- All are optional; without them the app runs in demo mode.
- DATABASE_URL enables lightweight schema creation. Prefer migrations in production.

4) Run Streamlit:
```bash
streamlit run phase1_alpha_analyst.py
```

## Tabs
- Instruments: select symbols, fetch 5y OHLCV, preview last rows, save instrument to DB
- Uploads: upload PDFs, extract text, embed via OpenAI or placeholder, insert into DB
- Session Log: show recent `instrument_master_data` rows (if Supabase configured)

## Tests
```bash
pip install pytest
pytest -q
```

## Notes
- Secrets via environment variables only.
- Add retries/rate limiting & full migrations in later phases.
