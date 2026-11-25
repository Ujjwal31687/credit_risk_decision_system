import os
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Credit Risk Decision System",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_validation.csv")
MAX_ROWS = 1000
LGD = 0.45  # 45% Basel conservative unsecured Loss Given Default


@st.cache_resource
def load_artifacts():
    model_path = os.path.join(BASE_DIR, "model", "lgb_model.pkl")
    preprocessor_path = os.path.join(BASE_DIR, "model", "preprocessor.pkl")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor


@st.cache_resource
def get_shap_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_data
def load_sample_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def score_portfolio(df: pd.DataFrame, _preprocessor, _model):
    required_cols = list(_preprocessor.feature_names_in_)
    feature_df = df[required_cols].replace([np.inf, -np.inf], np.nan)
    X_transformed = _preprocessor.transform(feature_df)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    pd_scores = _model.predict_proba(X_transformed)[:, 1]
    return X_transformed, pd_scores


model, preprocessor = load_artifacts()
required_cols = list(preprocessor.feature_names_in_)

st.title("📊 Credit Risk Decision Dashboard")
st.markdown(
    "Predict **Probability of Default (PD)**, simulate risk-based approval policies & **Expected Loss**, "
    "and inspect per-applicant **SHAP** explanations using a pre-aggregated customer feature schema "
    f"({len(required_cols)} features across application, bureau, previous loans, and installment history)."
)

# --- Data Source Controls ---
with st.sidebar:
    st.header("1. Portfolio Input")
    has_sample = os.path.exists(SAMPLE_PATH)

    mode_options = (
        ["Use Built-in Validation Sample (Demo)", "Upload Custom CSV"]
        if has_sample
        else ["Upload Custom CSV"]
    )
    input_mode = st.radio("Select Data Source", mode_options)

    if has_sample:
        with open(SAMPLE_PATH, "rb") as f:
            st.download_button(
                label="⬇️ Download Sample CSV (Template)",
                data=f.read(),
                file_name="sample_validation.csv",
                mime="text/csv",
                help="300 unseen validation applicants matching the exact feature-engineered schema.",
            )

    st.divider()
    st.header("2. Policy Controls")
    threshold = st.slider(
        "Risk Cutoff Threshold (PD)",
        min_value=0.05,
        max_value=0.90,
        value=0.30,
        step=0.01,
        help="Applicants with predicted risk below this cutoff are APPROVED.",
    )
    st.caption(
        "Expected Loss = PD × AMT_CREDIT × 45% LGD. "
        "Note: Scores reflect class-weighted LightGBM ranking (`scale_pos_weight`)."
    )

# --- Load Selected Dataset ---
if input_mode == "Use Built-in Validation Sample (Demo)":
    raw_data = load_sample_csv(SAMPLE_PATH)
    st.info(
        f"Showing **{len(raw_data)} unseen validation applicants** (`sample_validation.csv`). "
        "Switch to **Upload Custom CSV** in the sidebar to test your own batch."
    )
else:
    uploaded_file = st.file_uploader(
        "Upload pre-aggregated customer data (CSV)",
        type=["csv"],
        help="Must match the 168-column feature-engineered schema. Download the sample CSV in the sidebar as a template.",
    )
    if uploaded_file is None:
        st.warning(
            "Upload a feature-engineered CSV above, or switch to **Use Built-in Validation Sample** in the sidebar."
        )
        st.stop()
    try:
        raw_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read the uploaded CSV: {e}")
        st.stop()

# --- Schema & Row-Count Guards ---
if raw_data.empty:
    st.error("The provided CSV is empty. Please upload a file with at least 1 applicant row.")
    st.stop()

missing_cols = [c for c in required_cols if c not in raw_data.columns]
if missing_cols:
    preview = ", ".join(missing_cols[:10])
    extra = f" ... and {len(missing_cols) - 10} more" if len(missing_cols) > 10 else ""
    st.error(
        f"**Schema mismatch — missing {len(missing_cols)} of {len(required_cols)} required features.**\n\n"
        "This inference app consumes **pre-aggregated customer-level features** (matching a production "
        "feature-store table) rather than raw multi-table files.\n\n"
        f"**Missing columns:** `{preview}{extra}`\n\n"
        "Download `sample_validation.csv` from the sidebar to see the exact schema, or switch to the built-in demo."
    )
    st.stop()

if len(raw_data) > MAX_ROWS:
    st.warning(
        f"Uploaded file has {len(raw_data):,} rows. Scoring the first **{MAX_ROWS:,} rows** "
        "to keep interactive cloud latency fast."
    )
    raw_data = raw_data.head(MAX_ROWS).copy()

# --- Cached Model Inference ---
X, pd_scores = score_portfolio(raw_data, preprocessor, model)

data = raw_data.copy()
data["PD"] = pd_scores
data["EXPECTED_LOSS"] = data["PD"] * data["AMT_CREDIT"].fillna(0) * LGD
data["DECISION"] = np.where(data["PD"] < threshold, "APPROVE", "REJECT")

approved = data[data["DECISION"] == "APPROVE"]
baseline_loss = data["EXPECTED_LOSS"].sum()
model_loss = approved["EXPECTED_LOSS"].sum()
money_saved = baseline_loss - model_loss
loss_reduction_pct = (money_saved / baseline_loss) if baseline_loss > 0 else 0.0

# --- Portfolio KPIs ---
st.subheader("📌 Portfolio Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Applications", f"{len(data):,}")
col2.metric("Approval Rate", f"{len(approved) / len(data):.2%}")
col3.metric("Approved Expected Loss (₹)", f"{model_loss:,.0f}")
col4.metric("Loss Avoided vs. Approve-All (₹)", f"{money_saved:,.0f}", f"{loss_reduction_pct:.1%} cut")

# --- Individual Applicant SHAP Waterfall ---
st.subheader("🔍 Individual Applicant Explanation (TreeSHAP)")
explainer = get_shap_explainer(model)

if len(data) == 1:
    idx = 0
else:
    idx = st.number_input(
        "Select applicant row index",
        min_value=0,
        max_value=len(data) - 1,
        value=0,
        step=1,
    )

cust_pd = data.iloc[idx]["PD"]
cust_decision = data.iloc[idx]["DECISION"]
cust_credit = data.iloc[idx]["AMT_CREDIT"]
cust_el = data.iloc[idx]["EXPECTED_LOSS"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Applicant Index", int(idx))
c2.metric("Predicted Risk Score (PD)", f"{cust_pd:.2%}")
c3.metric("Loan Amount (₹)", f"{cust_credit:,.0f}")
c4.metric("Policy Decision", cust_decision)

shap_values = explainer.shap_values(X[idx : idx + 1])

if isinstance(shap_values, list):
    shap_vals = shap_values[1][0]
    base_value = explainer.expected_value[1]
elif np.ndim(shap_values) == 3:
    shap_vals = shap_values[0, :, 1]
    ev = np.atleast_1d(explainer.expected_value)
    base_value = float(ev[1] if len(ev) > 1 else ev[0])
else:
    shap_vals = shap_values[0]
    ev = np.atleast_1d(explainer.expected_value)
    base_value = float(ev[1] if len(ev) > 1 else ev[0])

fig, ax = plt.subplots(figsize=(9, 5))
shap.waterfall_plot(
    shap.Explanation(
        values=shap_vals,
        base_values=base_value,
        data=X[idx],
        feature_names=preprocessor.get_feature_names_out(),
    ),
    show=False,
)
st.pyplot(fig)
plt.close(fig)