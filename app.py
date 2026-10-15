
import io
import os
import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.pipeline import run_pipeline
from src.chat_rag import answer_question, build_documents

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Freight Cost Assistant",
    page_icon="🚚",
    layout="wide",
)

st.title("🚚 Freight Cost Assistant")
st.caption(
    "Weekly freight-cost monitoring with statistical detection, "
    "evidence-grounded explanations, and an LLM chat assistant using RAG."
)

# ============================================================
# Sidebar settings
# ============================================================

with st.sidebar:
    st.header("Analysis settings")

    mode = st.radio(
        "Detection mode",
        ["Combined", "Rule-based only", "Unsupervised only"],
        help=(
            "Rule-based compares with history and peers. "
            "Unsupervised uses robust statistical outlier scoring. "
            "Combined flags either."
        ),
    )

    threshold = st.slider(
        "Rule-based anomaly threshold (%)",
        min_value=5,
        max_value=100,
        value=15,
        step=5,
    )

    contamination = st.slider(
        "Unsupervised outlier fraction",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01,
    )

    # Existing explanation feature may still use OpenAI in notes_llm.py.
    # Do not enable it until that module is also migrated to Groq.
    llm_enabled = st.checkbox(
        "Use LLM for explanations",
        value=False,
        help=(
            "This existing feature may still be configured for OpenAI. "
            "The chat assistant below is configured separately for Groq."
        ),
    )

    model = st.text_input(
        "Explanation model",
        value=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        disabled=not llm_enabled,
    )

    st.divider()
    st.subheader("Chat assistant")

    chat_model = st.text_input(
        "Groq chat model",
        value=os.getenv(
            "GROQ_MODEL",
            "llama-3.3-70b-versatile",
        ),
        help="Uses Groq and retrieved analysis evidence.",
    )

    if os.getenv("GROQ_API_KEY", "").strip():
        st.success("Groq API key detected")
    else:
        st.warning(
            "Add GROQ_API_KEY to your .env file and restart Streamlit."
        )


# ============================================================
# Upload data
# ============================================================

st.subheader("1. Upload your data")

left, right = st.columns(2)

with left:
    shipment_file = st.file_uploader(
        "Shipment records CSV (required)",
        type=["csv"],
        key="shipments",
    )

with right:
    notes_file = st.file_uploader(
        "Context notes CSV (optional)",
        type=["csv"],
        key="notes",
        help=(
            "For a justified verdict, notes should have note ID/text, "
            "route, and effective date fields."
        ),
    )

st.info(
    "The app detects common shipment headers automatically. "
    "If your file uses different names, use origin, destination, "
    "route_type, quantity_tonnes, distance_km, freight_cost_inr, "
    "and shipment_date."
)

if shipment_file is None:
    st.markdown("### Try the included example")
    st.write(
        "Use the included data/shipment_records.csv "
        "or upload your own shipment CSV."
    )
    demo = st.button("Load included example")
else:
    demo = False


def read_uploaded(upload):
    if upload is None:
        return None
    return pd.read_csv(upload)


# ============================================================
# Run analysis
# ============================================================

run_clicked = st.button("Run analysis", type="primary")

if run_clicked or demo:
    try:
        if shipment_file is not None:
            shipment_df = read_uploaded(shipment_file)
        else:
            shipment_df = pd.read_csv(
                ROOT / "data" / "shipment_records.csv"
            )

        notes_df = (
            read_uploaded(notes_file)
            if notes_file is not None
            else None
        )

        if llm_enabled and not os.getenv("GROQ_API_KEY", "").strip():
            st.warning(
                "The existing explanation feature expects GROQ_API_KEY. "
                "Deterministic explanations will be used if the pipeline "
                "supports that fallback."
            )

        with st.spinner("Analyzing weekly route costs..."):
            output, scored, metrics = run_pipeline(
                shipment_df,
                notes_df,
                mode=mode,
                threshold_pct=float(threshold),
                contamination=float(contamination),
                llm_enabled=llm_enabled,
                model=model,
            )

        st.session_state["freight_output"] = output
        st.session_state["freight_scored"] = scored
        st.session_state["freight_notes_raw"] = notes_df
        st.session_state["freight_metrics"] = metrics
        st.session_state["freight_mode"] = mode
        st.session_state["freight_analysis_ready"] = True

        # Clear old chat because the evidence has changed.
        st.session_state["freight_chat_history"] = []
        st.session_state["freight_chat_last_sources"] = []

        st.success(
            "Analysis complete. The Groq chat assistant can now "
            "answer questions using this run's results."
        )

    except Exception as e:
        st.error(f"Could not process these files: {e}")
        st.markdown(
            "Check the required columns and date/numeric formats, "
            "then try again."
        )


# ============================================================
# Display analysis results
# ============================================================

if st.session_state.get("freight_analysis_ready"):
    output = st.session_state["freight_output"]
    scored = st.session_state["freight_scored"]
    metrics = st.session_state["freight_metrics"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Shipment rows", f"{metrics['shipment_rows']:,}")
    c2.metric("Route-weeks", f"{metrics['route_week_rows']:,}")
    c3.metric("Flagged", f"{metrics['flagged_rows']:,}")
    c4.metric("Explanation LLM calls", str(metrics["llm_calls"]))

    st.subheader("2. Results")
    st.dataframe(output, use_container_width=True, hide_index=True)

    st.download_button(
        "Download output CSV",
        output.to_csv(index=False).encode("utf-8"),
        file_name="freight_cost_results.csv",
        mime="text/csv",
    )

    st.subheader("3. Detection diagnostics")

    diag_cols = [
        "origin",
        "destination",
        "route_type",
        "week_of",
        "cost_per_tonne_km",
        "own_history_avg",
        "peer_avg",
        "own_change_pct",
        "peer_change_pct",
        "flag_rule",
        "unsupervised_anomaly",
        "anomaly_score",
        "history_weeks_used",
    ]

    available = [c for c in diag_cols if c in scored.columns]

    st.dataframe(
        scored[available].round(4),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download diagnostics CSV",
        scored[available].to_csv(index=False).encode("utf-8"),
        file_name="freight_cost_diagnostics.csv",
        mime="text/csv",
    )

    st.subheader("4. Run metrics / token log")
    st.json(metrics)

    st.download_button(
        "Download run metrics JSON",
        json.dumps(metrics, indent=2).encode("utf-8"),
        file_name="run_metrics.json",
        mime="application/json",
    )

    st.caption(
        "The app logs actual token counts when supported by the LLM response. "
        "Estimated dollar cost is not guessed."
    )

else:
    st.info(
        "Run the shipment analysis first. The results will then be "
        "available to the RAG chatbot below."
    )


# ============================================================
# Groq chatbot with RAG
# ============================================================

st.divider()
st.header("5. Ask Freight Cost Assistant")

st.write(
    "Ask questions about the latest analysis, cost changes, flagged routes, "
    "diagnostics, or uploaded context notes. The chatbot retrieves relevant "
    "project evidence before generating an answer."
)

if "freight_chat_history" not in st.session_state:
    st.session_state["freight_chat_history"] = []

# Prepare evidence from the current analysis.
documents = []

if st.session_state.get("freight_analysis_ready"):
    documents = build_documents(
        results=st.session_state["freight_output"],
        diagnostics=st.session_state["freight_scored"],
        notes=st.session_state.get("freight_notes_raw"),
    )


# Display previous messages.
for item in st.session_state["freight_chat_history"]:
    with st.chat_message(item["role"]):
        st.markdown(item["content"])

        if item.get("sources"):
            with st.expander("Retrieved evidence"):
                for source in item["sources"]:
                    st.markdown(
                        f"**[{source.get('id', 'SOURCE')}] — "
                        f"{source.get('type', 'document')}**"
                    )
                    st.caption(source.get("content", ""))

        if item.get("usage"):
            usage = item["usage"]
            st.caption(
                f"Model: {usage.get('model', 'Groq')} · "
                f"Input tokens: {usage.get('input_tokens', 0)} · "
                f"Output tokens: {usage.get('output_tokens', 0)}"
            )


prompt = st.chat_input("Ask about your freight-cost results...")

if prompt:
    st.session_state["freight_chat_history"].append(
        {"role": "user", "content": prompt}
    )

    if not st.session_state.get("freight_analysis_ready"):
        st.session_state["freight_chat_history"].append(
            {
                "role": "assistant",
                "content": (
                    "Please run the shipment analysis first so I have "
                    "project results to retrieve."
                ),
            }
        )

    elif not os.getenv("GROQ_API_KEY", "").strip():
        st.session_state["freight_chat_history"].append(
            {
                "role": "assistant",
                "content": (
                    "The analysis is ready, but chat needs a Groq API key. "
                    "Add GROQ_API_KEY to your .env file and restart Streamlit."
                ),
            }
        )

    else:
        try:
            with st.spinner(
                "Searching your analysis and preparing an answer..."
            ):
                # The latest question is passed separately, so send only
                # the earlier conversation turns as chat history.
                prior_history = st.session_state[
                    "freight_chat_history"
                ][:-1]

                result = answer_question(
                    question=prompt,
                    documents=documents,
                    chat_history=prior_history,
                    top_k=7,
                )

            st.session_state["freight_chat_history"].append(
                {
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "usage": {
                        "model": result.get("model", chat_model),
                        "input_tokens": (
                            result.get("usage") or {}
                        ).get("prompt_tokens", 0),
                        "output_tokens": (
                            result.get("usage") or {}
                        ).get("completion_tokens", 0),
                    },
                }
            )

        except Exception as e:
            st.session_state["freight_chat_history"].append(
                {
                    "role": "assistant",
                    "content": f"I couldn't answer that question: {e}",
                }
            )

    st.rerun()


# ============================================================
# Information
# ============================================================

with st.expander("Expected data formats and safety notes"):
    st.markdown(
        """
**Shipment CSV minimum fields:** origin, destination, route_type,
quantity (tonnes), distance (km), cost, shipment_date.

**Context notes recommended fields:** note_id, note_text, origin,
destination, route_type, start_date, end_date.

**Unsupervised mode:** Robust statistical outlier scoring identifies
statistically unusual route-weeks. An outlier is not automatically
an unjustified increase.

**LLM safety:** The LLM does not compute numeric metrics or set flags.
The chatbot retrieves analysis outputs, diagnostics, and uploaded notes.
It should cite retrieved evidence and state when evidence is insufficient.

**RAG implementation:** Retrieval uses lightweight lexical matching.
It does not require a vector database or scikit-learn.
"""
    )
