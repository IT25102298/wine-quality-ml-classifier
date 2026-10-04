"""Generates the remaining report figures (1, 2, 3, 5, 7) into results/.
Run from the notebooks/ folder (or project root):  python make_report_figures.py
Figures 4 and 6 already come from group_model_comparison."""
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings("ignore")
RS = 42
RES = "results"
os.makedirs(RES, exist_ok=True)
UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
sns.set_theme(style="whitegrid", font="DejaVu Sans")
NAVY, BLUE, GREY = "#0f172a", "#1e3a8a", "#94a3b8"

def load(kind):
    f = f"winequality-{kind}.csv"
    for d in ["../data/raw", "../data", "data/raw", "data", "."]:
        if os.path.exists(os.path.join(d, f)):
            return pd.read_csv(os.path.join(d, f), sep=";")
    return pd.read_csv(UCI + f, sep=";")

red, white = load("red"), load("white")
red["wine_type"], white["wine_type"] = 0, 1
df = pd.concat([red, white], ignore_index=True)
df["target"] = (df["quality"] >= 7).astype(int)
X, y = df.drop(columns=["quality", "target"]), df["target"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)

# Figure 1 - class distribution
vc = y.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6, 3.6))
bars = ax.bar(["Standard / Poor (0)\nquality < 7", "High Quality (1)\nquality \u2265 7"], vc.values, color=[GREY, BLUE], width=0.55)
for b, v in zip(bars, vc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v:,} ({v / vc.sum():.1%})", ha="center", fontsize=10, color=NAVY)
ax.set_ylabel("Number of wines"); ax.set_ylim(0, vc.max() * 1.15); ax.set_title("Target class distribution", color=NAVY, weight="bold")
plt.tight_layout(); plt.savefig(f"{RES}/class_distribution.png", dpi=300); plt.close()

# Figure 2 - correlation heatmap of the 11 physicochemical attributes
corr = df.drop(columns=["quality", "target", "wine_type"]).corr()
fig, ax = plt.subplots(figsize=(7, 5.8))
sns.heatmap(corr, mask=np.triu(np.ones_like(corr, bool), 1), cmap="RdBu_r", center=0, annot=True, fmt=".2f",
            annot_kws={"size": 7}, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Correlation matrix of physicochemical attributes", color=NAVY, weight="bold")
plt.tight_layout(); plt.savefig(f"{RES}/correlation_matrix.png", dpi=300); plt.close()

# Figure 3 - PCA cumulative variance (fit on standardized TRAIN data, as in the notebooks)
sc = StandardScaler().fit(Xtr)
Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
ev = PCA(random_state=RS).fit(Xtr_s).explained_variance_ratio_
cum = np.cumsum(ev); k = np.arange(1, len(ev) + 1)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.bar(k, ev, color=GREY, label="Individual"); ax.plot(k, cum, "o-", color=BLUE, label="Cumulative")
ax.axvline(6, ls="--", color="#b91c1c", lw=1); ax.text(6.1, 0.05, f"6 components\n{cum[5]:.1%}", color="#b91c1c", fontsize=9)
ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio"); ax.set_xticks(k)
ax.set_title("PCA explained variance", color=NAVY, weight="bold"); ax.legend(loc="center right")
plt.tight_layout(); plt.savefig(f"{RES}/pca_variance_scree.png", dpi=300); plt.close()
print(f"6 PCs retain {cum[5]:.1%} of variance")

# Figure 5 - 2x3 normalized confusion matrices from the exported JSONs
ids = [("IT25100788", "Logistic Regression"), ("IT25102855", "Decision Tree"), ("IT25101499", "KNN"),
       ("IT25100185", "Random Forest"), ("IT25103271", "SVC"), ("IT25102298", "MLP")]
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax, (sid, name) in zip(axes.ravel(), ids):
    p = f"{RES}/{sid}_metrics.json"
    if not os.path.exists(p):
        ax.axis("off"); ax.set_title(f"{name}\n(metrics file missing)"); continue
    cm = np.array(json.load(open(p))["confusion_matrix"], float)
    sns.heatmap(cm / cm.sum(1, keepdims=True), annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1, cbar=False, ax=ax,
                xticklabels=["Std", "High"], yticklabels=["Std", "High"])
    ax.set_title(name, color=NAVY, weight="bold"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.suptitle("Row-normalized confusion matrices (test set)", color=NAVY, weight="bold")
plt.tight_layout(); plt.savefig(f"{RES}/confusion_matrices_grid.png", dpi=300); plt.close()

# Figure 7 - Random Forest feature importance (refit with the selected hyperparameters)
params = dict(n_estimators=300, max_depth=20, max_features="sqrt", min_samples_leaf=2, class_weight="balanced_subsample")
p = f"{RES}/IT25100185_metrics.json"
if os.path.exists(p):
    params = {k: v for k, v in json.load(open(p))["best_params"].items()}
rf = RandomForestClassifier(random_state=RS, n_jobs=-1, **params).fit(Xtr_s, ytr)   # full standardized features
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.barh(imp.index, imp.values, color=BLUE)
ax.set_xlabel("Mean decrease in Gini impurity"); ax.set_title("Random Forest feature importance", color=NAVY, weight="bold")
plt.tight_layout(); plt.savefig(f"{RES}/rf_feature_importance.png", dpi=300); plt.close()
print("RF params used:", params); print(imp.sort_values(ascending=False).round(3).head(5).to_string())
print("Saved figures to", RES)
