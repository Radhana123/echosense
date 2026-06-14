import streamlit as st
import requests
import pandas as pd

API = "http://localhost:8000"

st.set_page_config(
    page_title="EchoSense",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .causal-chain {
        background: linear-gradient(90deg, #e8f5e9, #c8e6c9);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 0.5rem 0;
        font-weight: bold;
    }
    .report-box {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1.5rem;
        border-radius: 8px;
        font-family: monospace;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>📊 EchoSense</h1>
    <h3>AI Causal Story Engine</h3>
    <p>Upload any dataset → Discover WHY patterns occur</p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("EchoSense")
    st.markdown("---")
    st.header("📁 Upload Dataset")
    file = st.file_uploader("Choose CSV file", type="csv")
    target = st.text_input(
        "🎯 Target variable (optional)",
        placeholder="e.g. cancellation"
    )
    st.markdown("---")
    st.markdown("**Try these datasets:**")
    st.markdown("- 🚗 Zomato delivery")
    st.markdown("- 👥 IBM HR Analytics")
    st.markdown("- 🏥 Heart Disease UCI")
    st.markdown("---")
    run_btn = st.button(
        "🚀 Run EchoSense",
        type="primary",
        use_container_width=True
    )

if file and run_btn:
    # Upload
    with st.spinner("📤 Uploading + profiling dataset..."):
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

    # Metrics
    st.markdown("## 📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Total Rows",    up.json()["rows"])
    col2.metric("🌐 Domain",        domain.upper())
    col3.metric("📋 Columns",       len(cols))
    col4.metric("🎯 Target",        target if target else "Auto-detect")

    st.markdown("---")

    # Column profiles
    with st.expander("🔍 Column Profiles", expanded=False):
        profile_data = []
        for c in cols:
            profile_data.append({
                "Column": c["column_name"],
                "Type": c["inferred_type"],
                "Semantic": c["semantic_type"],
                "Null %": c["null_percent"],
                "Target": "✅" if c.get("is_target") else ""
            })
        st.dataframe(pd.DataFrame(profile_data), use_container_width=True)

    # Analyze
    with st.spinner("🧠 Running causal analysis..."):
        params = {"target_col": target} if target else {}
        an = requests.post(f"{API}/analyze/{ds_id}", params=params)

    if an.status_code != 200:
        st.error(f"Analysis failed: {an.text}")
        st.stop()

    res = an.json()

    # Causal Chains
    st.markdown("## 🔗 Causal Chains Discovered")
    chains = res.get("causal_chains", [])
    if chains:
        for i, ch in enumerate(chains):
            st.markdown(f"""
            <div class="causal-chain">
                🔗 Chain {i+1}: {ch['chain_str']}<br>
                <small>Confidence: {ch['min_confidence']*100:.0f}%</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No causal chains found.")

    st.markdown("---")

    # Anomalies
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("## ⚠️ Anomalies")
        st.metric("Total anomalies found", res.get("anomaly_count", 0))
    with col2:
        st.markdown("## ✅ Analysis Status")
        st.success(f"Run ID: {res.get('run_id')} — Completed!")

    st.markdown("---")

    # Report
    st.markdown("## 📄 Investigative Report")
    report = res.get("report", "")
    st.code(report, language=None)

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "⬇️ Download Report (.txt)",
            report,
            "echosense_report.txt",
            use_container_width=True
        )
    with col2:
        st.download_button(
            "⬇️ Download as CSV",
            pd.DataFrame(chains).to_csv(index=False) if chains else "",
            "causal_chains.csv",
            use_container_width=True
        )

else:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** 📁\n\nUpload any CSV file from the sidebar")
    with col2:
        st.info("**Step 2** 🎯\n\nOptionally enter target variable")
    with col3:
        st.info("**Step 3** 🚀\n\nClick Run EchoSense!")

    st.markdown("---")
    st.markdown("### 🧠 How EchoSense works")
    col1, col2, col3, col4 = st.columns(4)
    col1.success("**1. NLP Profiler**\nAuto-detects domain & column types")
    col2.success("**2. Anomaly Detection**\nIsolation Forest + Z-Score + IQR")
    col3.success("**3. Causal AI**\nDoWhy discovers WHY patterns occur")
    col4.success("**4. AI Report**\nGroq LLM generates insights")