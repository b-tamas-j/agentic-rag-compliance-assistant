"""Streamlit UI entrypoint (skeleton).

Run with: `streamlit run ui/streamlit_app.py`
"""

import streamlit as st


def main() -> None:
    st.set_page_config(
        page_title="Compliance Assistant",
        page_icon="??",
        layout="wide",
    )
    st.title("Agentic RAG Compliance Assistant")
    st.info("UI skeleton - implementation comes in the `feat/streamlit-ui` branch.")


if __name__ == "__main__":
    main()
