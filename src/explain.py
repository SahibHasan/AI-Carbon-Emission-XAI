import joblib
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from src.data_prep import load_and_clean

MODEL_PATH = Path("models/model.joblib")
REPORTS_DIR = Path("reports")


def generate_global_shap():
    """Generate global SHAP summary plot for the trained model."""
    
    # Load model + feature names
    saved = joblib.load(MODEL_PATH)
    if isinstance(saved, tuple):
        model, feature_names = saved
    else:
        model = saved
        feature_names = list(model.feature_names_in_)

    # Load data
    df = load_and_clean("data/raw/carbon_data.csv")
    X = df[feature_names]  # ensure perfect alignment

    # SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Create output folder
    REPORTS_DIR.mkdir(exist_ok=True)

    # Plot
    plt.figure(figsize=(8, 6))
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()

    out_path = REPORTS_DIR / "shap_summary.png"
    plt.savefig(out_path, dpi=200)

    print(f"✅ Global SHAP summary saved → {out_path}")


if __name__ == "__main__":
    generate_global_shap()
