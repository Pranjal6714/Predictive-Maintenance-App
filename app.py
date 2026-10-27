
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------
st.set_page_config(
    page_title="Predictive Maintenance",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance Dashboard")
st.write(
    "Compare Random Forest and KNN models to predict "
    "whether a machine may experience failure."
)

# -------------------------------------------------
# 2. LOAD SAVED MODELS AND SUPPORTING FILES
# -------------------------------------------------
@st.cache_resource
def load_files():
    rf = joblib.load("rf_model.pkl")
    knn = joblib.load("knn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    fill_values = joblib.load("fill_values.pkl")
    metrics = joblib.load("model_metrics.pkl")

    return rf, knn, scaler, feature_columns, fill_values, metrics


try:
    rf, knn, scaler, feature_columns, fill_values, metrics = load_files()
except Exception as e:
    st.error(
        "Unable to load model files. Check that all required "
        "files are in the GitHub repository and that the "
        "scikit-learn version is compatible with training."
    )
    st.exception(e)
    st.stop()

st.success("Both trained models loaded successfully!")

# -------------------------------------------------
# 3. MACHINE INPUTS
# -------------------------------------------------
st.header("Enter Machine Measurements")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        ["H", "L", "M"]
    )

    air_temp = st.number_input(
        "Air Temperature (K)",
        min_value=250.0,
        max_value=350.0,
        value=300.0
    )

    process_temp = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)",
        min_value=1,
        max_value=10000,
        value=1500
    )

    torque = st.number_input(
        "Torque (Nm)",
        min_value=0.0,
        max_value=200.0,
        value=40.0
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)",
        min_value=0,
        max_value=500,
        value=100
    )

# -------------------------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------------------------
def prepare_input():
    row = pd.DataFrame({
        "Type": pd.Categorical(
            [machine_type],
            categories=["H", "L", "M"]
        ),
        "Air temperature [K]": [air_temp],
        "Process temperature [K]": [process_temp],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    # Create the same features used during training
    row["Temperature Difference"] = (
        row["Process temperature [K]"]
        - row["Air temperature [K]"]
    )

    row["Power Proxy"] = (
        row["Torque [Nm]"]
        * row["Rotational speed [rpm]"] / 9550
    )

    row["Tool Wear Category"] = pd.cut(
        row["Tool wear [min]"],
        bins=[-1, 100, 200, float("inf")],
        labels=["Low", "Medium", "High"]
    )

    # Preserve the training categories for one-row input
    row["Tool Wear Category"] = pd.Categorical(
        row["Tool Wear Category"],
        categories=["Low", "Medium", "High"]
    )

    # Use the same encoding as the training notebook
    row = pd.get_dummies(
        row,
        columns=["Type", "Tool Wear Category"],
        drop_first=True,
        dtype=int
    )

    # Match the exact training columns and their order
    row = row.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # Use training medians if any values are missing
    row = row.replace(
        [np.inf, -np.inf], np.nan
    )
    row = row.fillna(fill_values)

    return row

# -------------------------------------------------
# 5. PREDICT USING BOTH MODELS
# -------------------------------------------------
if st.button("Compare Models", type="primary"):

    input_data = prepare_input()

    # Random Forest uses unscaled input
    rf_prediction = int(rf.predict(input_data)[0])
    rf_probability = rf.predict_proba(input_data)[0][1]

    # KNN uses scaled input
    scaled_input = scaler.transform(input_data)

    knn_prediction = int(knn.predict(scaled_input)[0])
    knn_probability = knn.predict_proba(scaled_input)[0][1]

    st.header("Prediction Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest")

        if rf_prediction == 1:
            st.error("⚠️ Possible Machine Failure")
        else:
            st.success("✅ No Failure Predicted")

        st.metric(
            "Estimated Failure Probability",
            f"{rf_probability * 100:.2f}%"
        )

    with col2:
        st.subheader("KNN")

        if knn_prediction == 1:
            st.error("⚠️ Possible Machine Failure")
        else:
            st.success("✅ No Failure Predicted")

        st.metric(
            "Estimated Failure Probability",
            f"{knn_probability * 100:.2f}%"
        )

    st.subheader("Model Agreement")

    if rf_prediction == knn_prediction:
        st.info(
            "Both models predict the same class: "
            + ("Failure" if rf_prediction == 1 else "No Failure")
        )
    else:
        st.warning(
            "The models disagree. Consider further inspection "
            "rather than relying on either prediction alone."
        )

    st.caption(
        "Probabilities are model estimates, not guarantees "
        "of real-world failure risk."
    )

# -------------------------------------------------
# 6. MODEL PERFORMANCE COMPARISON
# -------------------------------------------------
st.divider()
st.header("Model Performance Comparison")

metrics_df = pd.DataFrame(metrics).set_index("Model")

st.dataframe(
    metrics_df.style.format("{:.4f}"),
    use_container_width=True
)

st.bar_chart(
    metrics_df[
        ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    ]
)

st.caption(
    "Performance metrics are calculated on the held-out "
    "test dataset used during model evaluation."
)


