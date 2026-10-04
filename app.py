import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Wine Quality Classifier | IT2011",
    page_icon="🍷",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .metric-card {
        padding: 20px;
        border-radius: 8px;
        background-color: #1e293b;
        color: white;
        text-align: center;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🍷 Physicochemical Wine Quality Classifier")
st.caption("IT2011 AI & ML Final Project // Group Evaluation Dashboard")

# 1. Load Preprocessors & Models
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("saved_models/scaler.joblib")
    pca = joblib.load("saved_models/pca.joblib")
    models = {
        "Random Forest (Chamithu - IT25100185)": joblib.load("saved_models/random.joblib"),
        "K-Nearest Neighbors (Madubasha - IT25101499)": joblib.load("saved_models/k-nearest.joblib"),
        "Support Vector Machine (Yasith - IT25103271)": joblib.load("saved_models/support.joblib"),
        "Multi-Layer Perceptron (Menuja - IT25102298)": joblib.load("saved_models/multi-layer.joblib"),
        "Decision Tree (Nudara - IT25102855)": joblib.load("saved_models/decision.joblib"),
        "Logistic Regression (Chamodi - IT25100788)": joblib.load("saved_models/logistic.joblib"),
    }
    return scaler, pca, models

try:
    scaler, pca, models = load_artifacts()
except Exception as e:
    st.error("Error loading model artifacts. Make sure you ran 'export_models.py' first!")
    st.stop()

# 2. Sidebar Configuration
st.sidebar.header("🕹️ Model Selection")
selected_model_name = st.sidebar.selectbox(
    "Choose Active Model:",
    list(models.keys())
)
active_model = models[selected_model_name]

st.sidebar.markdown("---")
st.sidebar.subheader("Wine Category")
wine_type = st.sidebar.radio("Type", ["White Wine", "Red Wine"])
wine_type_val = 1 if wine_type == "White Wine" else 0

# 3. Main Input Interface
tab_predict, tab_benchmark = st.tabs(["🧪 Live Inference & Diagnostics", "📊 Group Performance Benchmark"])

with tab_predict:
    st.subheader("Enter Physicochemical Attributes")
    st.markdown("Adjust chemical parameters to evaluate predicted wine classification:")

    col1, col2, col3 = st.columns(3)
    
    with col1:
        fixed_acidity = st.slider("Fixed Acidity (g/dm³)", 3.0, 16.0, 7.2, 0.1)
        volatile_acidity = st.slider("Volatile Acidity (g/dm³)", 0.08, 1.60, 0.34, 0.01)
        citric_acid = st.slider("Citric Acid (g/dm³)", 0.0, 1.7, 0.32, 0.01)
        residual_sugar = st.slider("Residual Sugar (g/dm³)", 0.5, 30.0, 5.4, 0.1)

    with col2:
        chlorides = st.slider("Chlorides (g/dm³)", 0.01, 0.65, 0.056, 0.001)
        free_so2 = st.slider("Free Sulfur Dioxide (mg/dm³)", 1.0, 290.0, 30.5, 1.0)
        total_so2 = st.slider("Total Sulfur Dioxide (mg/dm³)", 6.0, 440.0, 115.0, 1.0)
        density = st.slider("Density (g/cm³)", 0.985, 1.040, 0.9947, 0.0005)

    with col3:
        pH = st.slider("pH", 2.7, 4.0, 3.22, 0.01)
        sulphates = st.slider("Sulphates (g/dm³)", 0.2, 2.0, 0.53, 0.01)
        alcohol = st.slider("Alcohol (% vol)", 8.0, 15.0, 10.5, 0.1)

    # 4. Prediction Execution
    input_data = np.array([[
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_so2, total_so2, density, pH, sulphates, alcohol, wine_type_val
    ]])

    st.markdown("---")
    if st.button("🚀 Run Live Classification", use_container_width=True):
        # Scale features
        input_scaled = scaler.transform(input_data)

        # Apply PCA if Menuja's MLP model is selected
        if "Multi-Layer Perceptron" in selected_model_name:
            input_processed = pca.transform(input_scaled)
        else:
            input_processed = input_scaled

        # Predict
        pred = active_model.predict(input_processed)[0]
        
        # Calculate Probability
        if hasattr(active_model, "predict_proba"):
            probs = active_model.predict_proba(input_processed)[0]
            prob_high = probs[1]
        else:
            prob_high = 0.5  # fallback

        # Visual Output
        res_col1, res_col2 = st.columns([1, 1])

        with res_col1:
            if pred == 1:
                st.success("### ✅ PREDICTION: HIGH QUALITY (Score ≥ 7)")
                st.write("The model classifies this sample as **Premium / High Quality**.")
            else:
                st.warning("### ⚠️ PREDICTION: STANDARD / POOR (Score < 7)")
                st.write("The model classifies this sample as **Ordinary / Standard Quality**.")

        with res_col2:
            st.metric("High Quality Confidence", f"{prob_high * 100:.1f}%")
            st.progress(prob_high)

with tab_benchmark:
    st.subheader("Official Group Benchmark (Test Set Evaluation)")
    benchmark_data = {
        "Student ID": ["IT25100185", "IT25101499", "IT25103271", "IT25102855", "IT25102298", "IT25100788"],
        "Student": ["Chamithu G.", "Madubasha W.", "Yasith K.", "Nudara F.", "Mahith M.", "Chamodi C."],
        "Model": ["Random Forest", "K-Nearest Neighbors", "Support Vector Machine", "Decision Tree", "MLP (Neural Net)", "Logistic Regression"],
        "Feature Representation": ["Standardized (12)", "Standardized (12)", "Standardized (12)", "Standardized (12)", "PCA (6 components)", "Standardized (12)"],
        "Accuracy": ["88.46%", "87.15%", "84.15%", "84.85%", "82.38%", "82.23%"],
        "Weighted F1": ["87.99%", "86.79%", "82.75%", "84.92%", "81.62%", "79.24%"],
        "ROC-AUC": [0.9104, 0.8725, 0.8497, 0.7656, 0.8430, 0.8050]
    }
    df_bench = pd.DataFrame(benchmark_data)
    st.dataframe(df_bench, use_container_width=True)
    
    st.info("💡 **Viva Demonstration Tip:** Random Forest achieved top rank across ROC-AUC (0.9104) and Weighted F1 (87.99%). Menuja's MLP utilized the 6-component PCA subspace to demonstrate dimensionality reduction trade-offs.")