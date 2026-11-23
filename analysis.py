import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

df = pd.read_csv("classification_escalation.csv")
target = "escalated"
X = df.drop(columns=target)
y = df[target]

numeric = X.select_dtypes(include="number").columns
categorical = X.select_dtypes(exclude="number").columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric),
    ("cat", categorical_pipeline, categorical),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

models = {
    "Majority baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "LDA": LinearDiscriminantAnalysis(),
    "k-NN (k=15)": KNeighborsClassifier(n_neighbors=15),
}

for name, classifier in models.items():
    model = Pipeline([("preprocess", preprocess), ("model", classifier)])
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    print(f"\\n{name}")
    print("Accuracy:", round(accuracy_score(y_test, prediction), 3))
    print("Precision:", round(precision_score(y_test, prediction, zero_division=0), 3))
    print("Recall:", round(recall_score(y_test, prediction, zero_division=0), 3))
    print("F1:", round(f1_score(y_test, prediction, zero_division=0), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, probability), 3))

lr = Pipeline([
    ("preprocess", preprocess),
    ("model", LogisticRegression(max_iter=1000)),
])
lr.fit(X_train, y_train)
prob = lr.predict_proba(X_test)[:, 1]
print("\\nConfusion matrix at threshold 0.50:")
print(confusion_matrix(y_test, (prob >= 0.50).astype(int)))

print("\\nThreshold results")
for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    cost = 5 * fn + fp
    print(
        threshold,
        "precision", round(precision_score(y_test, pred, zero_division=0), 3),
        "recall", round(recall_score(y_test, pred, zero_division=0), 3),
        "F1", round(f1_score(y_test, pred, zero_division=0), 3),
        "FP", fp, "FN", fn, "cost", cost,
    )

for k in [5, 9, 15, 21, 31]:
    knn = Pipeline([
        ("preprocess", preprocess),
        ("model", KNeighborsClassifier(n_neighbors=k)),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring="f1")
    print(f"k={k}, mean F1={scores.mean():.3f}")
