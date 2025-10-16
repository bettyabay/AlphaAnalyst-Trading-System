import sys
from pathlib import Path

# Ensure project root is on sys.path for scripts import
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from scripts.data_infrastructure import (
    get_env,
    render_sidebar,
    tab_instruments,
    tab_uploads,
    tab_session_log,
)


def main():
    st.set_page_config(page_title="AlphaAnalyst", layout="wide")
    env = get_env()
    render_sidebar(env)

    tab1, tab2, tab3 = st.tabs(["Instruments", "Uploads", "Session Log"])
    with tab1:
        tab_instruments(env)
    with tab2:
        tab_uploads(env)
    with tab3:
        tab_session_log(env)

    st.caption("Phase 1 — data infrastructure and research ingestion. Further phases coming.")


if __name__ == "__main__":
    main()


