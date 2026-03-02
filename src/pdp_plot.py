import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

def pdp_plot(model, X, feature):
    fig, ax = plt.subplots(figsize=(8, 6))
    PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
    plt.tight_layout()
    plt.savefig(f"reports/pdp_{feature}.png")
    print(f"✅ PDP saved for {feature}")
