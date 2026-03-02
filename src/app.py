import sys
import os

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

from src.data_prep import load_and_clean
from lime.lime_tabular import LimeTabularExplainer

# ---------- Load model + feature names ----------
saved = joblib.load("models/model.joblib")
if isinstance(saved, tuple) and len(saved) == 2:
    model, feature_names = saved
else:
    model = saved
    feature_names = list(model.feature_names_in_)

# ---------- SHAP explainer ----------
@st.cache_resource
def get_shap_explainer():
    return shap.TreeExplainer(model)

explainer_shap = get_shap_explainer()


# ---------- LIME explainer (use background training data) ----------
@st.cache_resource
def get_lime_explainer():
    df = load_and_clean("data/raw/carbon_data.csv")
    X = df[feature_names]  # ensure same columns
    explainer = LimeTabularExplainer(
        X.values,
        feature_names=feature_names,
        mode="regression"
    )
    return explainer

explainer_lime = get_lime_explainer()


# ---------- Streamlit Page Setup ----------
st.set_page_config(page_title="🌍 Carbon Emission Predictor", page_icon="🌿", layout="wide")
st.title("🌍 Carbon Emission Prediction with Explainable AI")
st.markdown(
    "This app predicts **CO₂ emissions** from socio-economic and energy features "
    "and explains the prediction using **SHAP & LIME (Explainable AI)**."
)

tab_pred, tab_global, tab_local = st.tabs(["🔮 Predict", "📊 Global Insights", "🔍 Local Explanation"])

# ========================= PREDICT TAB =========================
with tab_pred:
    st.header("🔧 Input Parameters")

    col1, col2 = st.columns(2)
    with col1:
        year = st.number_input("Year", min_value=2000, max_value=2035, value=2013)
        region = st.selectbox("Region", ["Asia", "Europe", "North America"])
        gdp_per_capita = st.number_input("GDP per Capita (USD)", value=3800.0, step=100.0)
        energy_consumption = st.number_input("Energy Consumption (MWh)", value=2750.0, step=50.0)
    with col2:
        industrial_output = st.number_input("Industrial Output (Index)", value=2200.0, step=50.0)
        vehicle_count = st.number_input("Vehicle Count", value=560000.0, step=10000.0)
        population = st.number_input("Population", value=84500000.0, step=100000.0)
        renewable_usage = st.number_input("Renewable Usage (%)", value=18.0, step=1.0)

    # Derived features (same as training)
    energy_intensity = energy_consumption / population if population > 0 else 0.0
    gdp_energy_interaction = gdp_per_capita * energy_intensity

    if st.button("🔍 Predict CO₂ Emission", key="predict_btn"):
        # Row dict that matches feature_names
        row = {}
        for feat in feature_names:
            if feat == "year":
                row[feat] = year
            elif feat == "gdp_per_capita":
                row[feat] = gdp_per_capita
            elif feat == "energy_consumption":
                row[feat] = energy_consumption
            elif feat == "industrial_output":
                row[feat] = industrial_output
            elif feat == "vehicle_count":
                row[feat] = vehicle_count
            elif feat == "population":
                row[feat] = population
            elif feat == "renewable_usage":
                row[feat] = renewable_usage
            elif feat == "energy_intensity":
                row[feat] = energy_intensity
            elif feat == "gdp_energy_interaction":
                row[feat] = gdp_energy_interaction
            elif feat.startswith("region_"):
                expected_region = feat.split("region_")[1]
                row[feat] = 1 if expected_region == region else 0
            else:
                row[feat] = 0  # unknown feature default

        input_df = pd.DataFrame([row], columns=feature_names)

        st.subheader("🧮 Input Row Passed to Model")
        st.dataframe(input_df)

        # Prediction
        pred = float(model.predict(input_df)[0])
        st.success(f"🌿 Predicted CO₂ Emission: **{pred:.2f} tons**")

        # Save for Local Explanation tab
        st.session_state["last_input_df"] = input_df
        st.session_state["last_pred"] = pred


# ========================= GLOBAL TAB =========================
with tab_global:
    st.header("📊 Global Feature Importance (SHAP)")
    img_path = Path("reports/shap_summary.png")
    if img_path.exists():
        st.image(str(img_path), caption="Global SHAP summary (feature importance)", use_container_width=True)
        st.markdown(
            "- **Red dots** → higher feature values\n"
            "- **Blue dots** → lower feature values\n"
            "- X-axis position shows whether the feature tends to increase or decrease emissions."
        )
    else:
        st.info(
            "Global SHAP summary image not found.\n\n"
            "Run this command once in terminal to generate it:\n"
            "```bash\npython -m src.explain\n```"
        )


# ========================= LOCAL TAB =========================
with tab_local:
    st.header("🔍 Local Explanation for Last Prediction")

    if "last_input_df" not in st.session_state:
        st.info("Firstly **Predict** after that explaination can be seen.")
    else:
        input_df = st.session_state["last_input_df"]
        pred = st.session_state["last_pred"]

        st.markdown(f"**Last predicted emission:** `{pred:.2f}` tons")

        # ---------- SHAP local explanation ----------
        st.subheader("🧠 SHAP Explanation (feature-wise contribution)")

        shap_values = explainer_shap.shap_values(input_df)[0]

        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Value": input_df.iloc[0].values,
            "SHAP Value": shap_values
        })
        shap_df["|SHAP|"] = shap_df["SHAP Value"].abs()
        shap_df = shap_df.sort_values("|SHAP|", ascending=False)

        st.dataframe(shap_df[["Feature", "Value", "SHAP Value"]])

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(shap_df["Feature"], shap_df["SHAP Value"])
        ax.set_xlabel("SHAP value (impact on prediction)")
        ax.invert_yaxis()
        st.pyplot(fig)

        st.caption("Positive SHAP value → Emission increase, Negative SHAP value → Emission decrease.")

        # ---------- LIME local explanation ----------
        st.subheader("🔍 LIME Explanation (Top factors for this prediction)")

        # Explain this single instance
        instance = input_df.iloc[0].values
        exp = explainer_lime.explain_instance(
            instance,
            model.predict,
            num_features=5
        )

        lime_list = exp.as_list()   # list of (condition, effect)
        lime_df = pd.DataFrame(lime_list, columns=["Condition", "Effect on prediction"])
        st.dataframe(lime_df)

        st.markdown(
            "LIME conditions indicate which feature ranges contributed to increasing (+)"
               "or decreasing (-) the prediction and the magnitude of their impact."

        )
