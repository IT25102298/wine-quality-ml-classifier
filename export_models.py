import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

os.makedirs("saved_models", exist_ok=True)

# 1. Ingest & Preprocess
red_path = os.path.join("data", "winequality-red.csv")
white_path = os.path.join("data", "winequality-white.csv")

if os.path.exists(red_path) and os.path.exists(white_path):
    df_red = pd.read_csv(red_path, sep=';')
    df_white = pd.read_csv(white_path, sep=';')
else:
    df_red = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')
    df_white = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

df_red['wine_type'] = 0
df_white['wine_type'] = 1
df = pd.concat([df_red, df_white], ignore_index=True)

df['target'] = (df['quality'] >= 7).astype(int)
X = df.drop(columns=['quality', 'target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

# 2. Fit Scaler & PCA on Training Data Only
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=6, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)

joblib.dump(scaler, "saved_models/scaler.joblib")
joblib.dump(pca, "saved_models/pca.joblib")

# 3. Train & Save Models using Group Optimal Parameters
models = {
    "Logistic Regression (Chamodi - IT25100788)": (
        LogisticRegression(C=10.0, penalty="l1", solver="saga", random_state=42, max_iter=2000),
        X_train_scaled
    ),
    "Decision Tree (Nudara - IT25102855)": (
        DecisionTreeClassifier(criterion="entropy", class_weight="balanced", random_state=42),
        X_train_scaled
    ),
    "K-Nearest Neighbors (Madubasha - IT25101499)": (
        KNeighborsClassifier(n_neighbors=5, weights="distance", p=2),
        X_train_scaled
    ),
    "Random Forest (Chamithu - IT25100185)": (
        RandomForestClassifier(n_estimators=300, max_depth=20, max_features="sqrt", 
                               min_samples_leaf=2, class_weight="balanced_subsample", random_state=42),
        X_train_scaled
    ),
    "Support Vector Machine (Yasith - IT25103271)": (
        SVC(C=10.0, kernel="rbf", gamma=0.1, probability=True, random_state=42),
        X_train_scaled
    ),
    "Multi-Layer Perceptron (Menuja - IT25102298)": (
        MLPClassifier(hidden_layer_sizes=(64, 32, 16), activation="tanh", alpha=0.001, 
                      learning_rate_init=0.01, random_state=42, max_iter=500),
        X_train_pca  # Menuja's model uses 6 PCA components
    )
}

print("Fitting and exporting models...")
for name, (model, data_train) in models.items():
    model.fit(data_train, y_train)
    clean_name = name.split(" ")[0].lower()
    joblib.dump(model, f"saved_models/{clean_name}.joblib")
    print(f" -> Saved: saved_models/{clean_name}.joblib")

print("All models successfully saved!")