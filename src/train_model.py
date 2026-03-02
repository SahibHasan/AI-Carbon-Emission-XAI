import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from src.data_prep import load_and_clean

MODEL_PATH = Path("models/model.joblib")


def train_and_save(path: str = "data/raw/carbon_data.csv"):
    # Load cleaned + engineered data
    df = load_and_clean(path)
    X = df.drop(columns=["emissions"])
    y = df["emissions"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"✅ RMSE: {rmse:.3f}")
    print(f"✅ MAE : {mae:.3f}")
    print(f"✅ R²  : {r2:.3f}")

    # 5-fold Cross Validation RMSE (global performance)
    scores = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    cv_rmse = np.mean(np.sqrt(-scores))
    print(f"✅ 5-fold CV RMSE: {cv_rmse:.3f}")

    # Save model + feature names for app.py & explain.py
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((model, X.columns.tolist()), MODEL_PATH)
    print(f"✅ Model + feature names saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
