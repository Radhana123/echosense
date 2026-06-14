import streamlit as st
import requests
import pandas as pd

API = "http://localhost:8000"

st.set_page_config(
    page_title="EchoSense",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EchoSense — AI Causal Story Engine")
st.caption("Upload any dataset → Discover WHY patterns occur")

# SIDEBAR
with st.sidebar:
    st.header("Upload Dataset")
    file = st.file_uploader("Choose CSV", type="csv")
    target = st.text_input("Target variable (optional)", placeholder="e.g. cancellation")
    run_btn = st.button("Run EchoSense ▶", type="primary", use_container_width=True)

if file and run_btn:
    # Upload
    with st.spinner("Uploading + profiling..."):
        up = requests.post(
            f"{API}/upload",
            files={"file": ("data.csv", file.getvalue(), "text/csv")}
        )

    if up.status_code != 200:
        st.error(f"Upload failed: {up.text}")
        st.stop()

    ds_id  = up.json()["dataset_id"]
    domain = up.json()["domain"]
    cols   = up.json()["columns"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows",    up.json()["rows"])
    col2.metric("Domain",  domain.upper())
    col3.metric("Columns", len(cols))

    # Analyze
    with st.spinner("Running causal analysis..."):
        params = {"target_col": target} if target else {}
        an = requests.post(f"{API}/analyze/{ds_id}", params=params)

    if an.status_code != 200:
        st.error(f"Analysis failed: {an.text}")
        st.stop()

    res = an.json()

    # Causal Chains
    st.subheader("🔗 Causal Chains Discovered")
    chains = res.get("causal_chains", [])
    if chains:
        for ch in chains:
            st.success(f"**{ch['chain_str']}**  (confidence: {ch['min_confidence']*100:.0f}%)")
    else:
        st.info("No causal chains found.")

    # Anomalies
    st.subheader("⚠️ Anomalies Found")
    st.metric("Total anomalies", res.get("anomaly_count", 0))

    # Report
    st.subheader("📄 Investigative Report")
    st.code(res.get("report", ""), language=None)
    st.download_button(
        "⬇️ Download Report",
        res.get("report", ""),
        "echosense_report.txt"
    )

else:
    st.info("Upload a CSV file from the sidebar and click Run EchoSense.")
    st.markdown("""
    **Try these datasets:**
    - Zomato delivery (logistics)
    - IBM HR Analytics (HR)
    - Heart Disease UCI (healthcare)
    """)