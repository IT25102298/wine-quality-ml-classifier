# 🍷 Physicochemical Wine Quality Classifier

### IT2011 – Artificial Intelligence & Machine Learning | Final Evaluation (2026)

**Faculty of Computing – Sri Lanka Institute of Information Technology (SLIIT)**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python\&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg?logo=scikit-learn\&logoColor=white)](https://scikit-learn.org/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://wine-quality-it2011.streamlit.app)
[![License: Academic](https://img.shields.io/badge/License-Academic%20Coursework-green.svg)](#)

---

## 📌 Executive Overview

End-to-end supervised machine learning benchmark and interactive deployment for classifying wine quality using physicochemical properties.

The combined **UCI Wine Quality Dataset** contains **6,497 observations and 12 features**. Wine quality is converted into a binary classification task:

$$
y =
\begin{cases}
1 & \text{if quality} \geq 7 \quad (\text{High Quality}) \\
0 & \text{if quality} < 7 \quad (\text{Standard})
\end{cases}
$$

Due to the natural class imbalance (~80% negative / ~20% positive), models were optimized using **5-Fold Stratified Cross-Validation**, **Weighted F1-score**, and **ROC-AUC**.

🔗 **Live App:** https://wine-quality-it2011.streamlit.app

---

## 👥 Team & Contributions

| IT Number      | Student              | Algorithm               | Primary Role                                |
| :------------- | :------------------- | :---------------------- | :------------------------------------------ |
| **IT25100185** | Chamithu Gunathilaka | **Random Forest**       | Class weighting & feature importance        |
| **IT25101499** | Madubasha W M N      | **KNN**                 | Distance metrics & PCA                      |
| **IT25103271** | Yasith Kobewaththa   | **SVM (SVC)**           | RBF kernel & margin tuning                  |
| **IT25102855** | Nudara Fernando      | **Decision Tree**       | Entropy splitting & pruning                 |
| **IT25102298** | Mahith Menuja        | **MLP**                 | PCA neural architecture & Adam optimization |
| **IT25100788** | Chamodi Chathurangi  | **Logistic Regression** | L1 regularization & baseline                |

---

## 📊 Model Performance

Evaluation was performed on an untouched **20% stratified test set (N = 1,300)**.

| Model                | Feature Space     |  Accuracy  |  Precision |   Recall   | Weighted F1 |  Macro F1  |   ROC-AUC  |
| :------------------- | :---------------- | :--------: | :--------: | :--------: | :---------: | :--------: | :--------: |
| **Random Forest 🏆** | Standardized (12) | **88.46%** | **75.48%** | **61.33%** |  **87.99%** | **80.32%** | **0.9104** |
| KNN                  | Standardized (12) |   87.15%   |   70.14%   |   60.55%   |    86.79%   |   78.56%   |   0.8725   |
| Decision Tree        | Standardized (12) |   84.85%   |   61.22%   |   62.89%   |    84.92%   |   76.29%   |   0.7656   |
| SVM                  | Standardized (12) |   84.15%   |   65.24%   |   41.80%   |    82.75%   |   70.75%   |   0.8497   |
| MLP                  | PCA (6)           |   82.38%   |   56.59%   |   45.31%   |    81.62%   |   69.81%   |   0.8430   |
| Logistic Regression  | Standardized (12) |   82.23%   |   61.47%   |   26.17%   |    79.24%   |   63.19%   |   0.8050   |

### Key Findings

* **Random Forest** achieved the best overall performance with **ROC-AUC = 0.9104**.
* **PCA improved KNN** ROC-AUC from **0.8725 → 0.8906** by reducing noisy dimensions.
* PCA reduced performance for some models by removing subtle class-discriminative information.
* **Logistic Regression** had low minority-class recall (**26.17%**), showing limitations of a linear decision boundary.

---

## 🏗️ Repository Structure

```text
wine-quality-ml-classifier/
├── data/
│   ├── winequality-red.csv
│   └── winequality-white.csv
├── notebooks/
│   ├── IT25100788_logistic_regression.ipynb
│   ├── IT25102855_decision_tree_classifier.ipynb
│   ├── IT25101499_k_nearest_neighbors_knn.ipynb
│   ├── IT25100185_random_forest_classifier.ipynb
│   ├── IT25103271_support_vector_machine_svc.ipynb
│   ├── IT25102298_multi_layer_perceptron_mlp.ipynb
│   └── group_model_comparison.ipynb
├── results/
│   ├── IT2510*_metrics.json
│   ├── IT2510*_cm.png
│   ├── IT2510*_roc.png
│   ├── group_metrics_barchart.png
│   └── group_roc_comparison.png
├── saved_models/
│   ├── scaler.joblib
│   ├── pca.joblib
│   └── *.joblib
├── app.py
├── export_models.py
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚡ Quickstart

### 1. Clone & Setup

```bash
git clone https://github.com/Mahith-dev/wine-quality-ml-classifier.git
cd wine-quality-ml-classifier

python -m venv venv
```

**Windows:**

```powershell
.\venv\Scripts\Activate.ps1
```

**Linux/macOS:**

```bash
source venv/bin/activate
```

### 2. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Train & Export Models

```bash
python export_models.py
```

### 4. Run Streamlit App

```bash
python -m streamlit run app.py
```

Open **http://localhost:8501**

---

## 🔬 Dashboard Features

* Dynamic switching between all six models
* Premium and standard sample presets
* Real-time physicochemical feature inputs
* Feature contribution analysis
* Model prediction comparison
* Confusion matrices
* Group performance benchmark

---

## 📜 Limitations

1. **Subjective Labels:** Quality scores originate from Portuguese sensory evaluations and represent human preferences.
2. **Geographical Scope:** Data comes from the **Vinho Verde region of Portugal**, limiting generalization to other regions.
3. **Class Imbalance:** High-quality wines represent a minority class.
4. **Decision Support:** Predictions should support, not replace, human expert judgment.

---

## 📚 References

1. Cortez, P. et al. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547–553.
2. UCI Machine Learning Repository. *Wine Quality Dataset*.
   https://archive.ics.uci.edu/dataset/186/wine+quality
3. Hastie, T., Tibshirani, R., & Friedman, J. *The Elements of Statistical Learning*. Springer, 2009.
4. Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.

---

## 🍷 Live Demo

**https://wine-quality-it2011.streamlit.app**

**Academic Project — IT2011 Artificial Intelligence & Machine Learning | SLIIT — 2026**
