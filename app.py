import streamlit as st
import pandas as pd
import numpy as np
from aifpipeline import run_bias_pipeline

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
    background-color: #FF8C42;
    color: white;
    border-radius: 8px;
    border: none;
    padding: 8px 16px;
    font-weight: 500;
}

.stButton>button:hover {
    background-color: #FF6B2C;
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

/* Expander */
.streamlit-expanderHeader {
    background-color: #F0F2F6 !important;
    color: #000000 !important;
}

/* Number input */
input[type="number"] {
    color: #000000 !important;
}

</style>
""", unsafe_allow_html=True)


st.title("⚖️ AIF360 Fairness Dashboard")

# Initialize session state for thresholds
if "use_default_thresholds" not in st.session_state:
    st.session_state["use_default_thresholds"] = True
if "custom_thresholds" not in st.session_state:
    st.session_state["custom_thresholds"] = {
        "SPD": 0.10,
        "DI_lower": 0.80,
        "DI_upper": 1.00,
        "EOD": 0.10,
        "AOD": 0.10
    }

# Sidebar navigation
step = st.sidebar.radio("📋 Workflow", [
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
    st.header("📁 Data Upload & Configuration")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], help="Upload a CSV file with your dataset")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.success(f"✅ {df.shape[0]} rows, {df.shape[1]} columns loaded successfully!")
        st.dataframe(df.head())

        columns = df.columns.tolist()

        col1, col2 = st.columns(2)

        with col1:
            target = st.selectbox("🎯 Target Column", columns, help="The column you want to predict")
            protected = st.selectbox("🛡️ Protected Attribute", columns, help="The sensitive attribute (e.g., race, gender)")
            
            fav = st.selectbox("✅ Positive Outcome of Target Label", df[target].unique(), 
                              help="The value that indicates a positive outcome (e.g., 'yes', 1, 'approved')")
            priv = st.selectbox("👑 Privileged Group", df[protected].unique(), 
                               help="The group that traditionally has advantage")

        with col2:
            st.subheader("⚙️ Advanced Settings")
            
            # Radio button for threshold selection
            threshold_option = st.radio(
                "Threshold Configuration",
                ["Use Default Thresholds", "Use Custom Thresholds"],
                index=0 if st.session_state["use_default_thresholds"] else 1
            )
            
            st.session_state["use_default_thresholds"] = (threshold_option == "Use Default Thresholds")
            
            if threshold_option == "Use Default Thresholds":
                st.info("📊 **Default Thresholds:**\n\n" +
                       "• SPD (Statistical Parity Difference): ≤ 0.10\n" +
                       "• DI (Disparate Impact): 0.8 - 1.0\n" +
                       "• EOD (Equal Opportunity Difference): ≤ 0.10\n" +
                       "• AOD (Average Odds Difference): ≤ 0.10")
                
                st.session_state["custom_thresholds"] = {
                    "SPD": 0.10,
                    "DI_lower": 0.80,
                    "DI_upper": 1.00,
                    "EOD": 0.10,
                    "AOD": 0.10
                }
            else:
                st.subheader("Set Custom Thresholds")
                spd_thresh = st.number_input("SPD Threshold (Statistical Parity Difference)", 
                                            min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                                            help="Closer to 0 is better")
                
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    di_lower = st.number_input("DI Lower Bound", 
                                               min_value=0.0, max_value=1.0, value=0.80, step=0.01,
                                               help="Disparate Impact should be ≥ this value")
                with col_d2:
                    di_upper = st.number_input("DI Upper Bound", 
                                               min_value=0.8, max_value=1.0, value=1.00, step=0.01,
                                               help="Disparate Impact should be ≤ this value (typically 1.0)")
                
                eod_thresh = st.number_input("EOD Threshold (Equal Opportunity Difference)", 
                                            min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                                            help="Closer to 0 is better")
                aod_thresh = st.number_input("AOD Threshold (Average Odds Difference)", 
                                            min_value=0.0, max_value=1.0, value=0.10, step=0.01,
                                            help="Closer to 0 is better")
                
                st.session_state["custom_thresholds"] = {
                    "SPD": spd_thresh,
                    "DI_lower": di_lower,
                    "DI_upper": di_upper,
                    "EOD": eod_thresh,
                    "AOD": aod_thresh
                }

        if st.button("💾 Save Configuration", type="primary"):
            st.session_state["df"] = df
            st.session_state["target"] = target
            st.session_state["protected"] = protected
            st.session_state["fav"] = fav
            st.session_state["priv"] = priv

            st.success("✅ Configuration saved! Go to next step")

# ======================
# STEP 2: DIAGNOSIS
# ======================
if step == "2. Diagnosis":
    st.header("🔍 Dataset Diagnosis")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first")
    else:
        df = st.session_state["df"]
        protected = st.session_state["protected"]

        st.subheader("📊 Data Preview")
        st.dataframe(df.head())

        st.subheader("📈 Group Distribution")
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(df[protected].value_counts())
        with col2:
            st.write("**Value Counts:**")
            st.write(df[protected].value_counts())

# ======================
# STEP 3: RUN PIPELINE
# ======================
if step == "3. Run Pipeline":
    st.header("🚀 Run Bias Analysis")

    if "df" not in st.session_state:
        st.warning("⚠️ Please upload a dataset first")
    else:
        if st.button("▶️ Run Pipeline", type="primary"):
            with st.spinner("🔄 Running fairness pipeline... This may take a few moments."):
                results, fig, best = run_bias_pipeline(
                    st.session_state["df"],
                    st.session_state["target"],
                    st.session_state["protected"],
                    st.session_state["fav"],
                    st.session_state["priv"],
                    st.session_state["custom_thresholds"]
                )

                st.session_state["results"] = results
                st.session_state["fig"] = fig
                st.session_state["best"] = best
                st.session_state["thresholds_used"] = st.session_state["custom_thresholds"]

                st.success("✅ Pipeline completed successfully!")

# ======================
# STEP 4: RESULTS
# ======================
if step == "4. Results":
    st.header("📊 Fairness Analysis Results")

    if "results" not in st.session_state:
        st.warning("⚠️ Please run the pipeline first")
    else:
        thresholds = st.session_state.get("thresholds_used", {
            "SPD": 0.10,
            "DI_lower": 0.80,
            "DI_upper": 1.00,
            "EOD": 0.10,
            "AOD": 0.10
        })
        
        # Extract threshold values
        spd_thresh = thresholds.get('SPD', 0.10)
        di_lower = thresholds.get('DI_lower', 0.80)
        di_upper = thresholds.get('DI_upper', 1.00)
        eod_thresh = thresholds.get('EOD', 0.10)
        aod_thresh = thresholds.get('AOD', 0.10)
        
        # Display results for each stage
        for stage, metrics in st.session_state["results"].items():
            with st.expander(f"📌 {stage}", expanded=(stage == st.session_state.get("best", ""))):
                # Create metrics table
                metrics_data = []
                
                # SPD
                spd_val = abs(metrics["SPD"])
                spd_fair = spd_val <= spd_thresh
                metrics_data.append({
                    "Sl No": 1,
                    "Metric": "SPD (Statistical Parity Difference)",
                    "Threshold Range": f"≤ {spd_thresh}",
                    "Actual Value": f"{metrics['SPD']:+.4f}",
                    "Fair or Unfair": "✅ Fair" if spd_fair else "❌ Unfair"
                })
                
                # DI - using range 0.8 to 1.0
                di_val = metrics["DI"]
                di_fair = di_lower <= di_val <= di_upper
                metrics_data.append({
                    "Sl No": 2,
                    "Metric": "DI (Disparate Impact)",
                    "Threshold Range": f"{di_lower} - {di_upper}",
                    "Actual Value": f"{di_val:.4f}",
                    "Fair or Unfair": "✅ Fair" if di_fair else "❌ Unfair"
                })
                
                # EOD
                eod_val = abs(metrics["EOD"])
                eod_fair = eod_val <= eod_thresh
                metrics_data.append({
                    "Sl No": 3,
                    "Metric": "EOD (Equal Opportunity Difference)",
                    "Threshold Range": f"≤ {eod_thresh}",
                    "Actual Value": f"{metrics['EOD']:+.4f}",
                    "Fair or Unfair": "✅ Fair" if eod_fair else "❌ Unfair"
                })
                
                # AOD
                aod_val = abs(metrics["AOD"])
                aod_fair = aod_val <= aod_thresh
                metrics_data.append({
                    "Sl No": 4,
                    "Metric": "AOD (Average Odds Difference)",
                    "Threshold Range": f"≤ {aod_thresh}",
                    "Actual Value": f"{metrics['AOD']:+.4f}",
                    "Fair or Unfair": "✅ Fair" if aod_fair else "❌ Unfair"
                })
                
                # Performance metrics
                metrics_data.append({
                    "Sl No": 5,
                    "Metric": "Accuracy",
                    "Threshold Range": "N/A",
                    "Actual Value": f"{metrics['Accuracy']:.4f}",
                    "Fair or Unfair": "N/A"
                })
                
                metrics_data.append({
                    "Sl No": 6,
                    "Metric": "F1 Score",
                    "Threshold Range": "N/A",
                    "Actual Value": f"{metrics['F1']:.4f}",
                    "Fair or Unfair": "N/A"
                })
                
                metrics_data.append({
                    "Sl No": 7,
                    "Metric": "Precision",
                    "Threshold Range": "N/A",
                    "Actual Value": f"{metrics['Precision']:.4f}",
                    "Fair or Unfair": "N/A"
                })
                
                metrics_data.append({
                    "Sl No": 8,
                    "Metric": "Recall",
                    "Threshold Range": "N/A",
                    "Actual Value": f"{metrics['Recall']:.4f}",
                    "Fair or Unfair": "N/A"
                })
                
                df_metrics = pd.DataFrame(metrics_data)
                st.table(df_metrics)
        
        # Display only fairness and performance graphs
        st.subheader("📈 Visualization")
        fig = st.session_state["fig"]
        
        # Extract only the first two subplots (fairness and performance)
        if hasattr(fig, 'axes'):
            # Create a new figure with only the first two subplots
            import matplotlib.pyplot as plt
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            fig2.patch.set_facecolor("#FFFFFF")
            
            # Copy fairness plot
            fairness_data = []
            stages = list(st.session_state["results"].keys())
            for i, stage in enumerate(stages):
                metrics = st.session_state["results"][stage]
                # Calculate DI deviation
                di_val = metrics['DI']
                if di_val < di_lower:
                    di_dev = di_lower - di_val
                elif di_val > di_upper:
                    di_dev = di_val - di_upper
                else:
                    di_dev = 0
                
                fairness_data.append({
                    'stage': stage,
                    'SPD': abs(metrics['SPD']),
                    'DI_dev': di_dev,
                    'EOD': abs(metrics['EOD']),
                    'AOD': abs(metrics['AOD'])
                })
            
            df_fairness = pd.DataFrame(fairness_data)
            x = np.arange(len(stages))
            width = 0.2
            
            # Plot fairness metrics
            bars1 = ax1.bar(x - 1.5*width, df_fairness['SPD'], width, label='SPD', color='#4C72B0', alpha=0.8)
            bars2 = ax1.bar(x - 0.5*width, df_fairness['DI_dev'], width, label='DI Deviation', color='#E24B4A', alpha=0.8)
            bars3 = ax1.bar(x + 0.5*width, df_fairness['EOD'], width, label='EOD', color='#55A868', alpha=0.8)
            bars4 = ax1.bar(x + 1.5*width, df_fairness['AOD'], width, label='AOD', color='#C44E52', alpha=0.8)
            
            # Add threshold lines
            ax1.axhline(y=spd_thresh, color='#4C72B0', linestyle='--', linewidth=1, alpha=0.7, label=f'SPD Threshold ({spd_thresh})')
            ax1.axhline(y=eod_thresh, color='#55A868', linestyle='--', linewidth=1, alpha=0.7, label=f'EOD Threshold ({eod_thresh})')
            ax1.axhline(y=aod_thresh, color='#C44E52', linestyle='--', linewidth=1, alpha=0.7, label=f'AOD Threshold ({aod_thresh})')
            
            # Highlight bars that exceed thresholds
            for i, (spd, di_dev, eod, aod) in enumerate(zip(df_fairness['SPD'], df_fairness['DI_dev'], 
                                                              df_fairness['EOD'], df_fairness['AOD'])):
                if spd > spd_thresh:
                    ax1.bar(i - 1.5*width, spd, width, color='#4C72B0', alpha=0.8, edgecolor='red', linewidth=2)
                if di_dev > 0:  # Any DI deviation is bad
                    ax1.bar(i - 0.5*width, di_dev, width, color='#E24B4A', alpha=0.8, edgecolor='red', linewidth=2)
                if eod > eod_thresh:
                    ax1.bar(i + 0.5*width, eod, width, color='#55A868', alpha=0.8, edgecolor='red', linewidth=2)
                if aod > aod_thresh:
                    ax1.bar(i + 1.5*width, aod, width, color='#C44E52', alpha=0.8, edgecolor='red', linewidth=2)
            
            ax1.set_xlabel('Stages')
            ax1.set_ylabel('Absolute Deviation')
            ax1.set_title('Fairness Metrics (lower = better)\n🔴 Red border indicates threshold violation', fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(stages, rotation=45, ha='right')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.grid(True, alpha=0.3)
            ax1.set_facecolor('#F8F9FA')
            
            # Plot performance metrics
            perf_data = []
            for stage in stages:
                metrics = st.session_state["results"][stage]
                perf_data.append({
                    'stage': stage,
                    'Accuracy': metrics['Accuracy'],
                    'F1': metrics['F1'],
                    'Precision': metrics['Precision'],
                    'Recall': metrics['Recall']
                })
            
            df_perf = pd.DataFrame(perf_data)
            
            ax2.bar(x - 1.5*width, df_perf['Accuracy'], width, label='Accuracy', color='#2196F3', alpha=0.8)
            ax2.bar(x - 0.5*width, df_perf['F1'], width, label='F1', color='#4CAF50', alpha=0.8)
            ax2.bar(x + 0.5*width, df_perf['Precision'], width, label='Precision', color='#FF9800', alpha=0.8)
            ax2.bar(x + 1.5*width, df_perf['Recall'], width, label='Recall', color='#9C27B0', alpha=0.8)
            
            ax2.set_xlabel('Stages')
            ax2.set_ylabel('Score')
            ax2.set_title('Performance Metrics (higher = better)', fontweight='bold')
            ax2.set_xticks(x)
            ax2.set_xticklabels(stages, rotation=45, ha='right')
            ax2.set_ylim(0, 1)
            ax2.legend(loc='lower right', fontsize=8)
            ax2.grid(True, alpha=0.3)
            ax2.set_facecolor('#F8F9FA')
            
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)

# ======================
# STEP 5: RECOMMENDATION
# ======================
if step == "5. Recommendation":
    st.header("💡 Recommendation")

    if "best" not in st.session_state:
        st.warning("⚠️ Please run the pipeline first")
    else:
        best = st.session_state["best"]
        metrics = st.session_state["results"][best]
        thresholds = st.session_state.get("thresholds_used", {
            "SPD": 0.10,
            "DI_lower": 0.80,
            "DI_upper": 1.00,
            "EOD": 0.10,
            "AOD": 0.10
        })
        
        spd_thresh = thresholds.get('SPD', 0.10)
        di_lower = thresholds.get('DI_lower', 0.80)
        di_upper = thresholds.get('DI_upper', 1.00)
        eod_thresh = thresholds.get('EOD', 0.10)
        aod_thresh = thresholds.get('AOD', 0.10)
        
        # Create detailed metrics table for recommendation
        st.subheader(f"🏆 Recommended Method: **{best}**")
        
        metrics_data = []
        
        spd_val = abs(metrics["SPD"])
        spd_fair = spd_val <= spd_thresh
        metrics_data.append({
            "Sl No": 1,
            "Metric": "SPD (Statistical Parity Difference)",
            "Threshold Range": f"≤ {spd_thresh}",
            "Actual Value": f"{metrics['SPD']:+.4f}",
            "Fair or Unfair": "✅ Fair" if spd_fair else "❌ Unfair"
        })
        
        di_val = metrics["DI"]
        di_fair = di_lower <= di_val <= di_upper
        metrics_data.append({
            "Sl No": 2,
            "Metric": "DI (Disparate Impact)",
            "Threshold Range": f"{di_lower} - {di_upper}",
            "Actual Value": f"{di_val:.4f}",
            "Fair or Unfair": "✅ Fair" if di_fair else "❌ Unfair"
        })
        
        eod_val = abs(metrics["EOD"])
        eod_fair = eod_val <= eod_thresh
        metrics_data.append({
            "Sl No": 3,
            "Metric": "EOD (Equal Opportunity Difference)",
            "Threshold Range": f"≤ {eod_thresh}",
            "Actual Value": f"{metrics['EOD']:+.4f}",
            "Fair or Unfair": "✅ Fair" if eod_fair else "❌ Unfair"
        })
        
        aod_val = abs(metrics["AOD"])
        aod_fair = aod_val <= aod_thresh
        metrics_data.append({
            "Sl No": 4,
            "Metric": "AOD (Average Odds Difference)",
            "Threshold Range": f"≤ {aod_thresh}",
            "Actual Value": f"{metrics['AOD']:+.4f}",
            "Fair or Unfair": "✅ Fair" if aod_fair else "❌ Unfair"
        })
        
        metrics_data.append({
            "Sl No": 5,
            "Metric": "Accuracy",
            "Threshold Range": "N/A",
            "Actual Value": f"{metrics['Accuracy']:.4f}",
            "Fair or Unfair": "N/A"
        })
        
        metrics_data.append({
            "Sl No": 6,
            "Metric": "F1 Score",
            "Threshold Range": "N/A",
            "Actual Value": f"{metrics['F1']:.4f}",
            "Fair or Unfair": "N/A"
        })
        
        metrics_data.append({
            "Sl No": 7,
            "Metric": "Precision",
            "Threshold Range": "N/A",
            "Actual Value": f"{metrics['Precision']:.4f}",
            "Fair or Unfair": "N/A"
        })
        
        metrics_data.append({
            "Sl No": 8,
            "Metric": "Recall",
            "Threshold Range": "N/A",
            "Actual Value": f"{metrics['Recall']:.4f}",
            "Fair or Unfair": "N/A"
        })
        
        df_recommendation = pd.DataFrame(metrics_data)
        st.table(df_recommendation)
        
        # Determine overall fairness
        is_fair = spd_fair and di_fair and eod_fair and aod_fair
        
        if is_fair:
            st.success("🎉 **Overall Assessment:** The model is FAIR (within threshold)")
            st.balloons()
        else:
            st.error("⚠️ **Overall Assessment:** The model is BIASED (exceeds threshold)")
            
            # Provide specific recommendations for violated metrics
            st.subheader("📋 Recommendations for Improvement:")
            if not spd_fair:
                st.write("• **SPD Violation:** Consider using Reweighing or DisparateImpactRemover with higher repair_level")
            if not di_fair:
                if di_val < di_lower:
                    st.write(f"• **DI Violation (too low - {di_val:.4f}):** Positive outcomes are favoring the unprivileged group. Try SMOTE oversampling or increase repair_level in DIR")
                else:
                    st.write(f"• **DI Violation (too high - {di_val:.4f}):** Positive outcomes are favoring the privileged group. Use Reweighing or CalibratedEqOddsPostprocessing")
            if not eod_fair:
                st.write("• **EOD Violation:** CalibratedEqOddsPostprocessing is specifically designed for equal opportunity")
            if not aod_fair:
                st.write("• **AOD Violation:** Adversarial Debiasing or ExponentiatedGradient with DemographicParity constraint")
