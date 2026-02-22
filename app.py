import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Stroke Prediction", layout="centered")

st.title("💙 Stroke Prediction - Logistic Regression")
st.markdown(
    "Aplikasi sederhana untuk memprediksi kemungkinan stroke menggunakan model Logistic Regression."
)

# =========================
# LOAD MODEL ARTIFACTS
# =========================
@st.cache_data
def load_artifacts():
    model = joblib.load("model_artifacts/model.joblib")
    scaler = joblib.load("model_artifacts/scaler.joblib")
    feature_columns = joblib.load("model_artifacts/feature_columns.joblib")
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# =========================
# INPUT FORM
# =========================
st.sidebar.header("Input Data Pasien")

gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
age = st.sidebar.slider("Age", 0, 120, 65)
hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
ever_married = st.sidebar.selectbox("Ever Married", ["Yes", "No"])
work_type = st.sidebar.selectbox(
    "Work Type",
    ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
)
Residence_type = st.sidebar.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.sidebar.number_input(
    "Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0
)
bmi = st.sidebar.number_input("BMI", min_value=5.0, max_value=70.0, value=25.0)
smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    ["never smoked", "formerly smoked", "smokes", "Unknown"],
)

input_dict = {
    "gender": gender,
    "age": age,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "ever_married": ever_married,
    "work_type": work_type,
    "Residence_type": Residence_type,
    "avg_glucose_level": avg_glucose_level,
    "bmi": bmi,
    "smoking_status": smoking_status,
}

input_df = pd.DataFrame([input_dict])

# =========================
# PREPROCESS INPUT (ONE HOT)
# =========================
input_encoded = pd.get_dummies(input_df)

# Samakan kolom dengan training
input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

# Scaling
input_scaled = scaler.transform(input_encoded)

# =========================
# PREDICTION
# =========================
st.subheader("Hasil Prediksi")

if st.button("Predict"):
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    if pred == 1:
        st.error(f"🚨 Terindikasi kemungkinan stroke (Probabilitas = {prob:.4f})")
    else:
        st.success(f"✅ Tidak terindikasi stroke (Probabilitas = {prob:.4f})")

# =========================
# DEBUG / INFORMASI MODEL
# =========================
with st.expander("Input Data (Encoded)"):
    st.write(input_encoded)

with st.expander("Model Insight (Top Coefficients)"):
    coef = model.coef_[0]
    imp_df = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": coef
    })
    imp_df = imp_df.reindex(
        imp_df.coefficient.abs().sort_values(ascending=False).index
    )
    st.table(imp_df.head(10))

st.markdown("---")
st.caption(
    "Catatan: Aplikasi ini hanya demonstrasi akademik. "
    "Tidak digunakan untuk diagnosis medis."
)
