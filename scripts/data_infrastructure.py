"""
AlphaAnalyst Trading System — Phase 1

This single-file Streamlit app delivers Phase 1 capabilities:
- Streamlit UI (single page with tabs): Instruments, Uploads, Session Log
- Supabase integration helpers (upsert instruments, insert research documents)
- Lightweight schema init (via DATABASE_URL if provided)
- Historical data fetching (Polygon if POLYGON_API_KEY else yfinance fallback; demo mode available)
- PDF research upload + text extraction (pdfplumber fallback), embeddings (OpenAI or deterministic placeholder)
- DB upserts for instrument_master_data & research_documents

Notes:
- For production, prefer migrations for schema. Avoid storing secrets in code; use env vars.
- Rate limits and error handling should be hardened in later phases.

Run:
  streamlit run phase1_alpha_analyst.py
"""

import io
import json
import logging
import os
import sys
import hashlib
import datetime as dt
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Optional dependencies are imported guardedly
try:
    import requests
except Exception:  # pragma: no cover - basic fallback
    requests = None  # type: ignore

try:
    import yfinance as yf
except Exception:  # pragma: no cover - tests use demo mode
    yf = None  # type: ignore

try:
    import pdfplumber
except Exception:  # pragma: no cover
    pdfplumber = None  # type: ignore

try:
    from supabase import create_client, Client as SupabaseClient
except Exception:  # pragma: no cover
    create_client = None  # type: ignore
    SupabaseClient = None  # type: ignore

try:
    # OpenAI >= 1.x client
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore

try:
    from sqlalchemy import create_engine, text
except Exception:  # pragma: no cover
    create_engine = None  # type: ignore
    text = None  # type: ignore

import streamlit as st


LOGGER = logging.getLogger("alpha_analyst.phase1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


CORE_WATCHLIST: List[str] = [
    'COIN','TSLA','NVDA','AVGO','PLTR','LRCX','CRWD','HOOD','APP',
    'MU','SNPS','CDNS','VST','DASH','DELL','NOW','PANW','AXON','URI'
]


# --------------------
# Environment helpers
# --------------------
def get_env() -> Dict[str, Optional[str]]:
    return {
        "SUPABASE_URL": os.getenv("SUPABASE_URL"),
        "SUPABASE_KEY": os.getenv("SUPABASE_KEY"),
        "POLYGON_API_KEY": os.getenv("POLYGON_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "DATABASE_URL": os.getenv("DATABASE_URL"),
    }


def supabase_client_or_none(env: Dict[str, Optional[str]]):
    if not env.get("SUPABASE_URL") or not env.get("SUPABASE_KEY"):
        return None
    if create_client is None:
        return None
    try:
        return create_client(env["SUPABASE_URL"], env["SUPABASE_KEY"])  # type: ignore[arg-type]
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Failed to init Supabase client: %s", e)
        return None


def sql_engine_or_none(env: Dict[str, Optional[str]]):
    db_url = env.get("DATABASE_URL")
    if not db_url or create_engine is None:
        return None
    try:
        return create_engine(db_url)
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Failed to init SQL engine: %s", e)
        return None


# --------------------
# DB schema (lightweight)
# --------------------
INSTRUMENT_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS instrument_master_data (
    symbol TEXT PRIMARY KEY,
    data_fetch_status JSONB,
    last_updated TIMESTAMPTZ,
    health_score NUMERIC
);
"""

RESEARCH_DOCS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS research_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol TEXT,
    file_name TEXT,
    file_content TEXT,
    embedding_vector JSONB,
    uploaded_at TIMESTAMPTZ DEFAULT now()
);
"""


def init_schema_if_possible(env: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    """Attempt to create minimal schema using DATABASE_URL if available.

    Returns (ok, message).
    """
    engine = sql_engine_or_none(env)
    if engine is None:
        return False, (
            "No DATABASE_URL or SQLAlchemy available; skipping schema init. "
            "For production, run migrations."
        )
    try:
        with engine.begin() as conn:  # autocommit mode
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto;"))  # for gen_random_uuid()
            conn.execute(text(INSTRUMENT_TABLE_SQL))
            conn.execute(text(RESEARCH_DOCS_TABLE_SQL))
        return True, "Schema ensured for Phase 1 tables."
    except Exception as e:  # pragma: no cover
        return False, f"Schema init error: {e}"


# --------------------
# Data fetching
# --------------------
def _date_range_5y() -> Tuple[str, str]:
    end = dt.date.today()
    start = end - dt.timedelta(days=365 * 5 + 10)
    return start.isoformat(), end.isoformat()


def fetch_historical_data(symbol: str, env: Dict[str, Optional[str]], use_demo: bool = False) -> pd.DataFrame:
    """Fetch 5y daily OHLCV for symbol.

    Priority: Polygon via REST → yfinance → deterministic demo dataframe.
    """
    if use_demo:
        return _demo_price_frame(symbol)

    polygon_key = env.get("POLYGON_API_KEY")
    if polygon_key and requests is not None:
        try:
            start, end = _date_range_5y()
            url = (
                f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start}/{end}"
                f"?adjusted=true&sort=asc&limit=50000&apiKey={polygon_key}"
            )
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    df = pd.DataFrame(results)
                    # Polygon columns: t (ms), o,h,l,c,v
                    df["date"] = pd.to_datetime(df["t"], unit="ms").dt.date
                    df = df.rename(columns={
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume",
                    })[["date","open","high","low","close","volume"]]
                    df["symbol"] = symbol
                    return df
        except Exception as e:
            LOGGER.warning("Polygon fetch failed, falling back to yfinance: %s", e)

    # yfinance fallback
    if yf is not None:
        try:
            start, end = _date_range_5y()
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=start, end=end, interval="1d")
            if not hist.empty:
                hist = hist.reset_index()
                # yfinance index column can be Datetime
                hist["date"] = pd.to_datetime(hist["Date"]).dt.date
                hist = hist.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                })[["date","open","high","low","close","volume"]]
                hist["symbol"] = symbol
                return hist
        except Exception as e:  # pragma: no cover
            LOGGER.warning("yfinance fetch failed, using demo data: %s", e)

    return _demo_price_frame(symbol)


def _demo_price_frame(symbol: str) -> pd.DataFrame:
    # Deterministic series using symbol hash as seed
    seed = int(hashlib.sha256(symbol.encode()).hexdigest(), 16) % 10_000
    dates = pd.date_range(end=dt.date.today(), periods=252 * 2, freq="B")  # ~2y business days
    base = 50 + (seed % 50)
    trend = (pd.Series(range(len(dates))) * 0.02)
    noise = pd.Series(((seed % 7) - 3) for _ in dates).astype(float) * 0.05
    close = base + trend + noise.cumsum()
    open_ = close * (1 + 0.001)
    high = pd.concat([open_, close], axis=1).max(axis=1) * 1.002
    low = pd.concat([open_, close], axis=1).min(axis=1) * 0.998
    volume = pd.Series(1_000_000 + (seed % 1000) * 100 for _ in dates)
    df = pd.DataFrame({
        "date": dates.date,
        "open": open_.round(2),
        "high": high.round(2),
        "low": low.round(2),
        "close": close.round(2),
        "volume": volume.astype(int),
    })
    df["symbol"] = symbol
    return df


# --------------------
# Document processing & embeddings
# --------------------
def extract_text_from_pdf(file_bytes: bytes) -> str:
    if pdfplumber is None:
        return ""
    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            texts = []
            for page in pdf.pages:
                try:
                    texts.append(page.extract_text() or "")
                except Exception:  # pragma: no cover
                    texts.append("")
            return "\n\n".join(texts).strip()
    except Exception as e:  # pragma: no cover
        LOGGER.warning("pdfplumber failed: %s", e)
        return ""


def generate_embedding(text_content: str, env: Dict[str, Optional[str]]) -> List[float]:
    # Prefer OpenAI if available
    api_key = env.get("OPENAI_API_KEY")
    if api_key and OpenAI is not None and text_content.strip():
        try:
            client = OpenAI(api_key=api_key)
            resp = client.embeddings.create(
                model="text-embedding-3-small",
                input=text_content[:7000],  # token safety; Phase 2+ can chunk
            )
            return list(resp.data[0].embedding)  # type: ignore[attr-defined]
        except Exception as e:  # pragma: no cover
            LOGGER.warning("OpenAI embedding failed, using placeholder: %s", e)

    # Deterministic placeholder vector of length 256
    digest = hashlib.sha256(text_content.encode("utf-8", errors="ignore")).digest()
    nums = list(digest) * (256 // len(digest) + 1)
    vec = [float(x) / 255.0 for x in nums[:256]]
    return vec


# --------------------
# Supabase helpers
# --------------------
def upsert_instrument_record(symbol: str, data_fetch_status: Dict, env: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    sb = supabase_client_or_none(env)
    if sb is None:
        return False, "Supabase not configured; running in demo mode."
    try:
        payload = {
            "symbol": symbol,
            "data_fetch_status": data_fetch_status,
            "last_updated": dt.datetime.utcnow().isoformat(),
            "health_score": None,
        }
        sb.table("instrument_master_data").upsert(payload).execute()
        return True, "Instrument upserted."
    except Exception as e:  # pragma: no cover
        return False, f"Upsert error: {e}"


def insert_research_document(symbol: Optional[str], file_name: str, text_content: str, embedding: List[float], env: Dict[str, Optional[str]]) -> Tuple[bool, str]:
    sb = supabase_client_or_none(env)
    if sb is None:
        return False, "Supabase not configured; document not stored (demo mode)."
    try:
        payload = {
            "symbol": symbol,
            "file_name": file_name,
            "file_content": text_content,
            "embedding_vector": embedding,
        }
        sb.table("research_documents").insert(payload).execute()
        return True, "Document inserted."
    except Exception as e:  # pragma: no cover
        return False, f"Insert error: {e}"


def fetch_recent_instruments(env: Dict[str, Optional[str]], limit: int = 20) -> pd.DataFrame:
    sb = supabase_client_or_none(env)
    if sb is None:
        return pd.DataFrame()
    try:
        res = sb.table("instrument_master_data").select("*").order("last_updated", desc=True).limit(limit).execute()
        data = getattr(res, "data", []) or []
        return pd.DataFrame(data)
    except Exception:  # pragma: no cover
        return pd.DataFrame()


# --------------------
# Streamlit UI
# --------------------
def render_sidebar(env: Dict[str, Optional[str]]):
    st.sidebar.title("AlphaAnalyst Phase 1")
    st.sidebar.markdown("Environment status:")
    st.sidebar.write({k: ("set" if bool(v) else "missing") for k, v in env.items()})

    if st.sidebar.button("Initialize/Ensure Schema"):
        ok, msg = init_schema_if_possible(env)
        if ok:
            st.sidebar.success(msg)
        else:
            st.sidebar.warning(msg)

    st.sidebar.markdown("""
    Notes:
    - Migrations are preferred for production.
    - Demo mode works without Supabase/Polygon/OpenAI.
    """)


def tab_instruments(env: Dict[str, Optional[str]]):
    st.header("Instruments")
    selected = st.multiselect("Onboard symbols", options=CORE_WATCHLIST, default=[])
    symbol = st.selectbox("Explore symbol", options=CORE_WATCHLIST)
    use_demo = st.toggle("Use demo data (deterministic)", value=False)
    if st.button("Fetch Data"):
        with st.spinner("Fetching historical data..."):
            df = fetch_historical_data(symbol, env, use_demo=use_demo)
        st.success(f"Fetched {len(df)} rows for {symbol}")
        st.dataframe(df.tail(10), use_container_width=True)

        if st.button("Save instrument to DB"):
            status = {
                "symbol": symbol,
                "fetched_rows": len(df),
                "source": ("Polygon" if env.get("POLYGON_API_KEY") else ("yfinance" if yf is not None else "demo")),
                "saved_at": dt.datetime.utcnow().isoformat(),
            }
            ok, msg = upsert_instrument_record(symbol, status, env)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)

    if selected:
        st.markdown("### Batch Onboarding Preview")
        st.write(f"Selected: {', '.join(selected)}")


def tab_uploads(env: Dict[str, Optional[str]]):
    st.header("Research Uploads")
    symbol_opt = st.selectbox("Associate with symbol (optional)", options=[None] + CORE_WATCHLIST, index=0)
    uploaded = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)
    if not uploaded:
        st.info("Upload one or more PDFs to extract text and store.")
        return

    for file in uploaded:
        st.subheader(file.name)
        content = file.read()
        text = extract_text_from_pdf(content)
        st.text_area("Text preview", value=text[:2000], height=200)
        with st.spinner("Generating embedding..."):
            emb = generate_embedding(text, env)
        st.write(f"Embedding length: {len(emb)}")
        if st.button(f"Store {file.name}"):
            ok, msg = insert_research_document(symbol_opt, file.name, text, emb, env)
            if ok:
                st.success(msg)
            else:
                st.warning(msg)


def tab_session_log(env: Dict[str, Optional[str]]):
    st.header("Session Log")
    df = fetch_recent_instruments(env, limit=50)
    if df.empty:
        st.info("No records found or Supabase not configured. In demo mode, actions are not persisted.")
        return
    st.dataframe(df[["symbol","last_updated","data_fetch_status"]], use_container_width=True)


def main():
    st.set_page_config(page_title="AlphaAnalyst Phase 1", layout="wide")
    env = get_env()
    render_sidebar(env)

    tab1, tab2, tab3 = st.tabs(["Instruments", "Uploads", "Session Log"])
    with tab1:
        tab_instruments(env)
    with tab2:
        tab_uploads(env)
    with tab3:
        tab_session_log(env)

    st.caption("Phase 1 complete — Further phases will add AI scoring, sessions, and trading engines.")


if __name__ == "__main__":
    # Allow running as a script for quick checks
    if "streamlit" in sys.argv[0].lower():
        main()
    else:
        # Simple CLI demo
        env = get_env()
        df_demo = fetch_historical_data("AAPL", env, use_demo=True)
        print(df_demo.head())


