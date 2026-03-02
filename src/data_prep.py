import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

DATA_RAW = Path("data/raw/carbon_data.csv")
TRAIN_OUT = Path("data/processed/train.csv")
TEST_OUT = Path("data/processed/test.csv")


def load_and_clean(path: str | Path = DATA_RAW):
    """Load raw CSV, clean it, do basic feature engineering, and return processed DataFrame."""
    df = pd.read_csv(path)

    # 1. Target ke bina rows hata do
    df = df.dropna(subset=["emissions"])

    # 2. Numeric columns ka missing data median se fill
    numeric = df.select_dtypes(include="number").columns.tolist()
    df[numeric] = df[numeric].fillna(df[numeric].median())

    # 3. Basic feature engineering (paper se inspired)
    # energy_intensity = energy_consumption / population
    if "energy_consumption" in df.columns and "population" in df.columns:
        denom = df["population"].replace(0, 1)  # zero se divide na ho
        df["energy_intensity"] = df["energy_consumption"] / denom

    # interaction: gdp_per_capita * energy_intensity
    if "gdp_per_capita" in df.columns and "energy_intensity" in df.columns:
        df["gdp_energy_interaction"] = df["gdp_per_capita"] * df["energy_intensity"]

    # 4. Categorical (e.g. region) ko one-hot encode karo
    df = pd.get_dummies(df, drop_first=True)

    return df


def split_df(df, target: str = "emissions", test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


if __name__ == "__main__":
    df = load_and_clean(DATA_RAW)
    X_train, X_test, y_train, y_test = split_df(df)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(TRAIN_OUT, index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(TEST_OUT, index=False)

    print("✅ Data processed and saved to data/processed/")
