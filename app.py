
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="PredictiveAI | Machine Health",
    page_icon="⚙️",
    layout="wide"
)

# ---------------- CUSTOM DESIGN ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #111c30);
    color: #f8fafc;
}
[data-testid="stSidebar"] {
    background: #101827;
}
.block-container {
    padding-top: 2rem;
}
.hero {
    padding: 28px;
    border-radius: 20px;
    background: linear-gradient(110deg, #172554, #164e63);
    border: 1px solid #31536d;
    margin-bottom: 24px;
}
.hero h1 {
    color: white;
    font-size: 2.5rem;
}
.hero p {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: #172033;
    border: 1px solid #293449;
    padding: 16px;
    border-radius: 14px;
}
div.stButton > button {
    border-radius: 10px;
    font-weight: bold;
    min-height: 44px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_artifacts():
    rf = joblib.load("rf_model.pkl")
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    fill_values = joblib.load("fill_values.pkl")

    return rf, knn, scaler, feature_columns, fill_values

try:
    rf, knn, scaler, feature_columns, fill_values = load_artifacts()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.info(
        "Check that all five .pkl files are uploaded "
        "to the same GitHub repository as app.py."
    )
    st.stop()

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.title("⚙️ Machine Controls")
st.sidebar.caption("Enter operating measurements")

machine_type = st.sidebar.selectbox(
    "Machine Type", ["L", "M", "H"]
)

air_temp = st.sidebar.number_input(
    "Air Temperature (K)", 250.0, 350.0, 300.0
)

process_temp = st.sidebar.number_input(
    "Process Temperature (K)", 250.0, 400.0, 310.0
)

rotational_speed = st.sidebar.number_input(
    "Rotational Speed (rpm)", 500, 5000, 1500
)

torque = st.sidebar.number_input(
    "Torque (Nm)", 0.0, 100.0, 40.0
)

tool_wear = st.sidebar.number_input(
    "Tool Wear (minutes)", 0, 300, 100
)

run = st.sidebar.button(
    "🔍 Run Diagnostics",
    type="primary",
    use_container_width=True
)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>⚙️ PredictiveAI</h1>
    <p>Smart Machine Health Monitoring Dashboard</p>
    <p>Predictive maintenance powered by Machine Learning.</p>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
c1.metric("Models Available", "02")
c2.metric("Prediction Type", "Binary")
c3.metric("Analysis", "Machine Health")

# ---------------- PREPARE INPUT ----------------
def prepare_input():
    data = pd.DataFrame([{
        "Type": machine_type,
        "Air temperature [K]": air_temp,
        "Process temperature [K]": process_temp,
        "Rotational speed [rpm]": rotational_speed,
        "Torque [Nm]": torque,
        "Tool wear [min]": tool_wear
    }])

    # Feature engineering: must match training code
    data["Temperature Difference"] = (
        data["Process temperature [K]"]
        - data["Air temperature [K]"]
    )

    data["Power Proxy"] = (
        data["Torque [Nm]"]
        * data["Rotational speed [rpm]"]
        / 9550
    )

    data["Tool Wear Category"] = pd.cut(
        data["Tool wear [min]"],
        bins=[-1, 100, 200, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    data = pd.get_dummies(
        data,
        columns=["Type", "Tool Wear Category"],
        drop_first=True,
        dtype=int
    )

    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    data = data.fillna(fill_values)
    return data

# ---------------- PREDICTIONS ----------------
if run:
    try:
        X_input = prepare_input()
        X_scaled = scaler.transform(X_input)

        rf_prediction = int(rf.predict(X_input)[0])
        knn_prediction = int(knn.predict(X_scaled)[0])

        predictions = {
            "Random Forest": rf_prediction,
            "KNN": knn_prediction
        }

        probabilities = {}

        if hasattr(rf, "predict_proba"):
            probabilities["Random Forest"] = (
                float(rf.predict_proba(X_input)[0][1]) * 100
            )

        if hasattr(knn, "predict_proba"):
            probabilities["KNN"] = (
                float(knn.predict_proba(X_scaled)[0][1]) * 100
            )

        st.session_state["results"] = predictions
        st.session_state["probabilities"] = probabilities
        st.session_state["inputs"] = {
            "Machine Type": machine_type,
            "Air Temperature (K)": air_temp,
            "Process Temperature (K)": process_temp,
            "Rotational Speed (rpm)": rotational_speed,
            "Torque (Nm)": torque,
            "Tool Wear (minutes)": tool_wear
        }

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------------- DISPLAY RESULTS ----------------
if "results" in st.session_state:
    results = st.session_state["results"]
    probabilities = st.session_state["probabilities"]

    st.divider()
    st.subheader("📊 Machine Health Report")

    rf_pred = results["Random Forest"]
    knn_pred = results["KNN"]

    if rf_pred == knn_pred == 1:
        st.error("🚨 Both models predict machine failure.")
    elif rf_pred == knn_pred == 0:
        st.success("✅ Both models predict no machine failure.")
    else:
        st.warning(
            "⚠️ The models disagree. Consider inspecting the machine."
        )

    a, b, c = st.columns(3)
    a.metric("Random Forest", "Failure" if rf_pred else "No Failure")
    b.metric("KNN", "Failure" if knn_pred else "No Failure")
    c.metric("Models Agree", "Yes" if rf_pred == knn_pred else "No")

    st.subheader("🤖 Individual Model Results")
    left, right = st.columns(2)

    for column, name in [
        (left, "Random Forest"),
        (right, "KNN")
    ]:
        with column:
            st.markdown(f"### {name}")

            if results[name] == 1:
                st.error("🚨 Failure Predicted")
            else:
                st.success("✅ No Failure Predicted")

            if name in probabilities:
                p = probabilities[name]
                st.write(f"Estimated failure probability: **{p:.2f}%**")
                st.progress(int(np.clip(p, 0, 100)))

    if probabilities:
        st.subheader("📈 Failure Probability Comparison")

        chart_data = pd.DataFrame({
            "Model": list(probabilities.keys()),
            "Failure Probability (%)": list(probabilities.values())
        }).set_index("Model")

        st.bar_chart(chart_data)

    st.subheader("🧾 Input Summary")
    summary = pd.DataFrame(
        [st.session_state["inputs"]]
    )
    st.dataframe(summary, use_container_width=True, hide_index=True)

    report = pd.DataFrame({
        "Model": list(results.keys()),
        "Prediction": [
            "Failure" if p == 1 else "No Failure"
            for p in results.values()
        ],
        "Failure Probability (%)": [
            probabilities.get(name, np.nan)
            for name in results
        ]
    })

    st.download_button(
        "⬇️ Download Prediction Report",
        report.to_csv(index=False).encode("utf-8"),
        file_name="machine_health_report.csv",
        mime="text/csv"
    )

else:
    st.info(
        "Enter the machine measurements in the sidebar, "
        "then click Run Diagnostics."
    )

    st.subheader("✨ Dashboard Features")
    a, b, c = st.columns(3)

    a.markdown("### 🔎 Predict")
    a.write("Predict possible machine failure.")

    b.markdown("### ⚖️ Compare")
    b.write("Compare Random Forest and KNN predictions.")

    c.markdown("### 📥 Export")
    c.write("Download your prediction report as CSV.")

