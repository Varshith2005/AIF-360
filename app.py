import streamlit as st
import pandas as pd
from pipeline import run_bias_pipeline

st.set_page_config(layout="wide")
st.markdown("""
<style>

/* Main App */
.stApp {
    background-color: #FFFFFF;
    color: #000000;
}

/* Headings & text (light areas) */
h1, h2, h3, h4, h5, h6, p, label {
    color: #000000 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111 !important;
}

section[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

/* Buttons */
.stButton>button {
    background-color: peachpuff;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: peachpuff;
    color: white;
}

/* Selectbox, dropdown, inputs */
div[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

input, textarea {
    color: #000000 !important;
    background-color: #FFFFFF !important;
}

/* Metrics / JSON blocks */
.stJson {
    background-color: #F8F9FA !important;
    color: #000000 !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
    color: #000000 !important;
}

/* Success / warning messages */
.stAlert {
    border-radius: 8px;
}

/* Radio buttons (sidebar) */
.stRadio label {
    color: #FFFFFF !important;
}

</style>
""", unsafe_allow_html=True)


st.title("AIF360 Dashboard")

# Sidebar navigation
step = st.sidebar.radio("Workflow", [
    "1. Data Upload",
    "2. Diagnosis",
    "3. Run Pipeline",
    "4. Results",
    "5. Recommendation"
])

# ======================
# STEP 1: DATA UPLOAD
# ======================
if step == "1. Data Upload":

    st.header("Data Upload & Configuration")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.success(f"{df.shape[0]} rows, {df.shape[1]} columns loaded")
        st.dataframe(df.head())

        columns = df.columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            target = st.selectbox("Target Column", columns)
            protected = st.selectbox("Protected Attribute", columns)

            fav = st.selectbox("Favorable Label", df[target].unique())
            priv = st.selectbox("Privileged Group", df[protected].unique())

        with col2:
            st.subheader("Advanced Settings")
            st.write("Default thresholds used")

        if st.button("Save Configuration"):
            st.session_state["df"] = df
            st.session_state["target"] = target
            st.session_state["protected"] = protected
            st.session_state["fav"] = fav
            st.session_state["priv"] = priv

            st.success("Saved! Go to next step")

            

# ======================
# STEP 2: DIAGNOSIS
# ======================
if step == "2. Diagnosis":

    st.header("Dataset Diagnosis")

    if "df" not in st.session_state:
        st.warning("Upload dataset first")
    else:
        df = st.session_state["df"]
        protected = st.session_state["protected"]

        st.dataframe(df.head())

        st.subheader("Group Distribution")
        st.bar_chart(df[protected].value_counts())

# ======================
# STEP 3: RUN PIPELINE
# ======================
if step == "3. Run Pipeline":

    st.header("Run Bias Analysis")

    if "df" not in st.session_state:
        st.warning("Upload dataset first")
    else:
        if st.button("Run Pipeline"):

            with st.spinner("Running pipeline..."):

                results, fig, best = run_bias_pipeline(
                    st.session_state["df"],
                    st.session_state["target"],
                    st.session_state["protected"],
                    st.session_state["fav"],
                    st.session_state["priv"]
                )

                st.session_state["results"] = results
                st.session_state["fig"] = fig
                st.session_state["best"] = best

                st.success("Done!")

# ======================
# STEP 4: RESULTS
# ======================
if step == "4. Results":

    st.header("Results")

    if "results" not in st.session_state:
        st.warning("Run pipeline first")
    else:
        for stage, metrics in st.session_state["results"].items():
            st.subheader(stage)
            st.json(metrics)

        st.pyplot(st.session_state["fig"])

# ======================
# STEP 5: RECOMMENDATION
# ======================
if step == "5. Recommendation":

    st.header("Recommendation")

    if "best" not in st.session_state:
        st.warning("Run pipeline first")
    else:
        best = st.session_state["best"]
        metrics = st.session_state["results"][best]

        spd = abs(metrics["SPD"])
        di_dev = abs(1 - metrics["DI"])
        eod = abs(metrics["EOD"])
        aod = abs(metrics["AOD"])

        if spd <= 0.10 and di_dev <= 0.20 and eod <= 0.10 and aod <= 0.10:
            st.success("✅ Model is FAIR (within threshold)")
        else:
            st.error("❌ Model is BIASED (exceeds threshold)")

        st.write("### Selected Method:", best)
        st.json(metrics)
