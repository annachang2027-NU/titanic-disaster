"""Titanic survival prediction with logistic regression (Python).

Reads train.csv / test.csv from DATA_DIR (default: ../data relative to this
file, or /app/data inside Docker), cleans and engineers features, fits a
logistic regression, reports accuracy, and writes test-set predictions.
"""

import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))
RANDOM_STATE = 42

FEATURES = [
    "Pclass", "Sex", "Age", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone",
]
CATEGORICAL = ["Pclass", "Embarked", "Title"]


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_csv(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] Could not find {path}.\n"
            "Download train.csv and test.csv from "
            "https://www.kaggle.com/competitions/titanic/data and put them in "
            "src/data/ (see README.md)."
        )
    return pd.read_csv(path)


def extract_title(name_col):
    """Pull the honorific ('Mr', 'Mrs', ...) out of 'Last, Title. First'."""
    title = name_col.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    common = {"Mr", "Mrs", "Miss", "Master"}
    return title.where(title.isin(common), "Rare")


def clean(df, fill_values, name):
    """Apply the same cleaning / feature steps to train or test, printing each."""
    df = df.copy()
    section(f"Cleaning & feature engineering: {name}")

    df["Title"] = extract_title(df["Name"])
    print("[add] Title extracted from Name. Counts:")
    print(df["Title"].value_counts().to_string())

    n_age = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(fill_values["age_by_title"]))
    print(f"[fill] Age: {n_age} missing values filled with median age of each Title "
          f"{ {k: round(v, 1) for k, v in fill_values['age_by_title'].items()} }")

    n_emb = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fill_values["embarked_mode"])
    print(f"[fill] Embarked: {n_emb} missing values filled with mode "
          f"'{fill_values['embarked_mode']}'")

    n_fare = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fill_values["fare_median"])
    print(f"[fill] Fare: {n_fare} missing values filled with median "
          f"{fill_values['fare_median']:.2f}")

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print("[add] FamilySize = SibSp + Parch + 1; IsAlone = 1 if FamilySize == 1. "
          f"Passengers travelling alone: {df['IsAlone'].sum()} of {len(df)}")

    df["Sex"] = (df["Sex"] == "female").astype(int)
    print("[encode] Sex -> 1 = female, 0 = male")

    dropped = ["Cabin", "Ticket", "Name", "SibSp", "Parch"]
    df = df.drop(columns=dropped)
    print(f"[drop] {dropped} (Cabin is ~77% missing; Ticket/Name are free text; "
          "SibSp/Parch are replaced by FamilySize)")

    print(f"Remaining missing values: {int(df[FEATURES].isna().sum().sum())}")
    return df


def to_matrix(df, columns=None):
    """One-hot encode categoricals; align to training columns if given."""
    X = pd.get_dummies(df[FEATURES], columns=CATEGORICAL, drop_first=True, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def main():
    section("Environment")
    print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | "
          f"numpy {np.__version__} | scikit-learn {sklearn.__version__}")
    print(f"Reading data from: {os.path.abspath(DATA_DIR)}")

    # ---- Load training data -------------------------------------------------
    section("Step 1: Load train.csv")
    train_raw = load_csv("train.csv")
    print(f"train.csv loaded: {train_raw.shape[0]} rows x {train_raw.shape[1]} columns")
    print(train_raw.head().to_string())

    # ---- Explore ------------------------------------------------------------
    section("Step 2: Explore the training data")
    print("Missing values per column:")
    print(train_raw.isna().sum()[lambda s: s > 0].to_string())
    print(f"\nOverall survival rate: {train_raw['Survived'].mean():.1%}")
    print("\nSurvival rate by Sex:")
    print(train_raw.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:")
    print(train_raw.groupby("Pclass")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Embarked:")
    print(train_raw.groupby("Embarked")["Survived"].mean().round(3).to_string())

    # Fill values are learned from TRAIN only, then reused on test (no leakage)
    titles = extract_title(train_raw["Name"])
    fill_values = {
        "age_by_title": train_raw.groupby(titles)["Age"].median().to_dict(),
        "embarked_mode": train_raw["Embarked"].mode()[0],
        "fare_median": train_raw["Fare"].median(),
    }

    # ---- Clean / engineer ---------------------------------------------------
    train = clean(train_raw, fill_values, "train.csv")
    print("\nSurvival rate by engineered Title:")
    print(train.groupby("Title")["Survived"].mean().round(3).to_string())

    X = to_matrix(train)
    y = train["Survived"]
    print(f"\nModel matrix after one-hot encoding: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Columns: {list(X.columns)}")

    # ---- Model --------------------------------------------------------------
    section("Step 3: Logistic regression")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    holdout = LogisticRegression(max_iter=1000)
    holdout.fit(X_tr, y_tr)
    print(f"80/20 split check -> train acc: {accuracy_score(y_tr, holdout.predict(X_tr)):.4f}, "
          f"held-out validation acc: {accuracy_score(y_val, holdout.predict(X_val)):.4f}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
    print("\nFinal model fit on all 891 training rows. Coefficients (log-odds):")
    print(coefs.round(3).to_string())
    print(f"Intercept: {model.intercept_[0]:.3f}")

    # ---- Training accuracy --------------------------------------------------
    section("Step 4: Accuracy on the training set")
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Training accuracy: {train_acc:.4f} ({train_acc:.1%})")

    # ---- Test set -----------------------------------------------------------
    section("Step 5: Load test.csv and predict")
    test_raw = load_csv("test.csv")
    print(f"test.csv loaded: {test_raw.shape[0]} rows x {test_raw.shape[1]} columns")
    test = clean(test_raw, fill_values, "test.csv")
    X_test = to_matrix(test, columns=X.columns)
    test_pred = model.predict(X_test)
    print(f"\nPredicted survivors in test set: {test_pred.sum()} of {len(test_pred)} "
          f"({test_pred.mean():.1%})")

    out_path = os.path.join(DATA_DIR, "predictions_python.csv")
    pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": test_pred}).to_csv(
        out_path, index=False
    )
    print(f"Predictions saved to {out_path} (Kaggle submission format)")

    # ---- Test accuracy ------------------------------------------------------
    section("Step 6: Accuracy on the test set")
    if "Survived" in test_raw.columns:
        print(f"Test accuracy: {accuracy_score(test_raw['Survived'], test_pred):.4f}")
    else:
        print("Kaggle's test.csv does not include the true 'Survived' labels, so true "
              "test accuracy\ncan only be computed by uploading predictions_python.csv "
              "to Kaggle.")
        print(f"Best local estimate (held-out 20% of train): "
              f"{accuracy_score(y_val, holdout.predict(X_val)):.4f}")
        gs_path = os.path.join(DATA_DIR, "gender_submission.csv")
        if os.path.exists(gs_path):
            gs = pd.read_csv(gs_path).set_index("PassengerId")["Survived"]
            agree = (gs.loc[test_raw["PassengerId"]].values == test_pred).mean()
            print(f"Agreement with Kaggle's gender_submission.csv baseline "
                  f"(women survive, men don't): {agree:.4f}")

    section("Done")


if __name__ == "__main__":
    main()
