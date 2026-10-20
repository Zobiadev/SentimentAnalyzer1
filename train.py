"""Train and compare sentiment models on the IMDB 50K reviews dataset.

Run:  python train.py
Needs: data/IMDB Dataset.csv  (columns: review, sentiment)
Outputs: models/sentiment_model.joblib, assets/*.png, assets/results.csv
"""
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

BASE = Path(__file__).parent
DATA = BASE / "data" / "IMDB Dataset.csv"
ASSETS = BASE / "assets"
MODELS = BASE / "models"
ASSETS.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)


def clean(text: str) -> str:
    text = re.sub(r"<.*?>", " ", text)          # remove HTML tags
    text = re.sub(r"[^a-zA-Z\s]", " ", text)    # keep letters only
    return re.sub(r"\s+", " ", text).lower().strip()


def make_pipeline(clf):
    return Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=50000,
                                  ngram_range=(1, 2), min_df=2)),
        ("clf", clf),
    ])


def main():
    # ---- 1. Load ----
    df = pd.read_csv(DATA)
    print("Shape:", df.shape)
    print("Missing:\n", df.isnull().sum())
    print("Duplicates:", df.duplicated().sum())
    df = df.drop_duplicates().dropna().reset_index(drop=True)

    # ---- 2. EDA charts ----
    plt.figure(figsize=(5, 4))
    sns.countplot(x="sentiment", data=df, hue="sentiment", palette="Set2", legend=False)
    plt.title("Class distribution")
    plt.tight_layout(); plt.savefig(ASSETS / "class_balance.png", dpi=150); plt.close()

    df["words"] = df["review"].str.split().str.len()
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="words", hue="sentiment", bins=60)
    plt.xlim(0, 1000); plt.title("Review length (words)")
    plt.tight_layout(); plt.savefig(ASSETS / "length_dist.png", dpi=150); plt.close()

    # ---- 3. Clean ----
    df["clean"] = df["review"].apply(clean)
    df["label"] = (df["sentiment"] == "positive").astype(int)

    # ---- 4. Split ----
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    # ---- 5. Compare models ----
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=2.0),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(),
    }
    rows, fitted = [], {}
    for name, clf in candidates.items():
        pipe = make_pipeline(clf).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        acc, f1 = accuracy_score(y_test, pred), f1_score(y_test, pred)
        rows.append({"Model": name, "Accuracy": round(acc, 4), "F1": round(f1, 4)})
        fitted[name] = pipe
        print(f"\n=== {name} ===  acc={acc:.4f}  f1={f1:.4f}")
        print(classification_report(y_test, pred, target_names=["negative", "positive"]))

    results = pd.DataFrame(rows).sort_values("Accuracy", ascending=False)
    results.to_csv(ASSETS / "results.csv", index=False)
    print("\n", results.to_string(index=False))

    plt.figure(figsize=(6, 4))
    sns.barplot(x="Model", y="Accuracy", data=results, hue="Model", palette="viridis", legend=False)
    plt.ylim(0.8, 0.95); plt.title("Model comparison")
    plt.tight_layout(); plt.savefig(ASSETS / "model_comparison.png", dpi=150); plt.close()

    # ---- 6. Final model: Logistic Regression (gives probabilities for the app) ----
    final = fitted["Logistic Regression"]
    pred = final.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["negative", "positive"], cmap="Blues")
    plt.title("Confusion matrix - Logistic Regression")
    plt.tight_layout(); plt.savefig(ASSETS / "confusion_matrix.png", dpi=150); plt.close()

    # Top words per class (for the report)
    vec, clf = final.named_steps["tfidf"], final.named_steps["clf"]
    names = vec.get_feature_names_out()
    order = clf.coef_[0].argsort()
    top_neg, top_pos = names[order[:15]], names[order[-15:]]
    print("\nTop negative words:", list(top_neg))
    print("Top positive words:", list(top_pos))

    joblib.dump(final, MODELS / "sentiment_model.joblib", compress=3)
    print("\nSaved model ->", MODELS / "sentiment_model.joblib")


if __name__ == "__main__":
    main()
