"""Titanic survival prediction with logistic regression (Python).

What this script does, in order:
  1. Load train.csv
  2. Explore it (missing values, survival rates)
  3. Clean it and build new features
  4. Fit a logistic regression
  5. Report accuracy on the training data
  6. Predict test.csv and save the predictions

Where the data comes from:
  - In Docker, the data folder is mounted at /app/data (DATA_DIR is set in the Dockerfile).
  - Run locally, it falls back to ../data, which is src/data/.
"""

import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Make printed tables wide enough that pandas doesn't wrap them.
pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# Folder with the CSVs. The DATA_DIR environment variable wins if it's set.
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(HERE, "..", "data"))

# A fixed seed means the random 80/20 split is the same every run,
# so anyone who runs this gets the same numbers.
RANDOM_STATE = 42

# Columns the model uses. Sex isn't here on purpose: Title already contains it
# (see the "drop" step in clean()).
FEATURES = [
    "Pclass", "Age", "Fare", "Embarked",
    "Title", "FamilySize", "IsAlone",
]

# Columns that are categories, not numbers. Pclass looks like a number (1, 2, 3),
# but 3rd class isn't "3 times" 1st class, so we treat it as a category.
CATEGORICAL = ["Pclass", "Embarked", "Title"]


def section(title):
    """Print a banner so each step is easy to find in the terminal output."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_csv(name):
    """Read a CSV from DATA_DIR. If it's missing, stop with a helpful message."""
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        sys.exit(
            f"[ERROR] Could not find {path}.\n"
            "Download train.csv and test.csv from "
            "https://www.kaggle.com/competitions/titanic/data and put them in "
            "src/data/ (see README.md)."
        )
    return pd.read_csv(path)


def extract_title(name_col, sex_col):
    """Get the title (Mr, Mrs, Miss, Master) from each passenger's name.

    Names look like "Braund, Mr. Owen Harris". The title sits between the
    comma and the period. Titles are useful because they tell us sex, rough
    age (Master = boy) and marital status (Miss vs Mrs) all at once.
    """
    # Regex: after ", " grab everything up to the next "."
    title = name_col.str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()

    # French titles mean the same as English ones.
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    # Uncommon titles (Dr, Rev, Col, Countess, ...) are too few to learn from
    # one by one, so we group them. Rare women go to "Mrs" and rare men go
    # to "Rare". This way Title always tells us the passenger's sex.
    common = {"Mr", "Mrs", "Miss", "Master"}
    rare = ~title.isin(common)
    title = title.mask(rare & (sex_col == "female"), "Mrs")
    return title.mask(rare & (sex_col == "male"), "Rare")


def clean(df, fill_values, name):
    """Clean one dataset (train or test) and print every change.

    fill_values holds numbers learned from the TRAINING data only. We reuse
    them on the test data. If we computed them from the test data, the model
    would be peeking at data it shouldn't see (this is called "data leakage").
    """
    df = df.copy()  # work on a copy so the raw data stays untouched
    section(f"Cleaning & feature engineering: {name}")

    # --- New feature: Title ---
    df["Title"] = extract_title(df["Name"], df["Sex"])
    print("[add] Title extracted from Name (rare titles -> 'Rare' for men, 'Mrs' for women). Counts:")
    print(df["Title"].value_counts().to_string())

    # --- Fill missing Age ---
    # About 20% of ages are missing. Filling with one overall median would make
    # every little boy 28 years old. Using the median for each title is
    # smarter: a missing "Master" gets a child's age, a missing "Mrs" an adult's.
    n_age = df["Age"].isna().sum()
    df["Age"] = df["Age"].fillna(df["Title"].map(fill_values["age_by_title"]))
    print(f"[fill] Age: {n_age} missing values filled with median age of each Title "
          f"{ {k: round(v, 1) for k, v in fill_values['age_by_title'].items()} }")

    # --- Fill missing Embarked (port) ---
    # Only 2 are missing, so the most common port is a safe guess.
    n_emb = df["Embarked"].isna().sum()
    df["Embarked"] = df["Embarked"].fillna(fill_values["embarked_mode"])
    print(f"[fill] Embarked: {n_emb} missing values filled with mode "
          f"'{fill_values['embarked_mode']}'")

    # --- Fill missing Fare ---
    # Only 1 is missing (in test.csv). We use the median, not the mean,
    # because a few very expensive tickets would pull the mean up.
    n_fare = df["Fare"].isna().sum()
    df["Fare"] = df["Fare"].fillna(fill_values["fare_median"])
    print(f"[fill] Fare: {n_fare} missing values filled with median "
          f"{fill_values['fare_median']:.2f}")

    # --- New features: FamilySize and IsAlone ---
    # SibSp = siblings + spouses on board, Parch = parents + children.
    # Adding them (plus the passenger) gives the whole family size.
    # People travelling alone and people in very big families both did worse.
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    print("[add] FamilySize = SibSp + Parch + 1; IsAlone = 1 if FamilySize == 1. "
          f"Passengers travelling alone: {df['IsAlone'].sum()} of {len(df)}")

    # --- Drop columns we won't use ---
    # Cabin: about 77% empty. Ticket and Name: free text (Name is already used
    # for Title). SibSp and Parch: now combined in FamilySize.
    # Sex: Title already says it. With both, the model can't tell which one
    # deserves the credit, and the coefficients blow up.
    dropped = ["Cabin", "Ticket", "Name", "SibSp", "Parch", "Sex"]
    df = df.drop(columns=dropped)
    print(f"[drop] {dropped} (Cabin is mostly missing; Ticket/Name are free text; "
          "SibSp/Parch are replaced by FamilySize;\n       Sex is fully captured by "
          "Title, and keeping both makes the coefficients unstable)")

    # Sanity check: the model can't handle missing values, so this should be 0.
    print(f"Remaining missing values: {int(df[FEATURES].isna().sum().sum())}")
    return df


def to_matrix(df, columns=None):
    """Turn the cleaned data into the all-numbers table the model needs.

    One-hot encoding: a category column like Embarked (C, Q, S) becomes
    0/1 columns (Embarked_Q, Embarked_S). drop_first=True drops one level
    (here C), which becomes the baseline the others are compared against.

    For the test data, we pass in the training columns. That makes sure the
    test table has exactly the same columns in the same order, even if some
    category never shows up in the test data.
    """
    X = pd.get_dummies(df[FEATURES], columns=CATEGORICAL, drop_first=True, dtype=int)
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


def main():
    section("Environment")
    # Print versions so anyone can check they're running the same setup.
    print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | "
          f"numpy {np.__version__} | scikit-learn {sklearn.__version__}")
    print(f"Reading data from: {os.path.abspath(DATA_DIR)}")

    # ---- Step 1: Load the training data ------------------------------------
    section("Step 1: Load train.csv")
    train_raw = load_csv("train.csv")
    print(f"train.csv loaded: {train_raw.shape[0]} rows x {train_raw.shape[1]} columns")
    print(train_raw.head().to_string())

    # ---- Step 2: Explore -----------------------------------------------------
    # Before modelling, look at what's missing and which groups survived more.
    # This tells us which features are likely to matter.
    section("Step 2: Explore the training data")
    print("Missing values per column:")
    print(train_raw.isna().sum()[lambda s: s > 0].to_string())
    print(f"\nOverall survival rate: {train_raw['Survived'].mean():.1%}")
    # Survived is 0/1, so its mean within a group is that group's survival rate.
    print("\nSurvival rate by Sex:")
    print(train_raw.groupby("Sex")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Pclass:")
    print(train_raw.groupby("Pclass")["Survived"].mean().round(3).to_string())
    print("\nSurvival rate by Embarked:")
    print(train_raw.groupby("Embarked")["Survived"].mean().round(3).to_string())

    # Learn the fill values from the training data only (see clean() for why).
    titles = extract_title(train_raw["Name"], train_raw["Sex"])
    fill_values = {
        "age_by_title": train_raw.groupby(titles)["Age"].median().to_dict(),
        "embarked_mode": train_raw["Embarked"].mode()[0],
        "fare_median": train_raw["Fare"].median(),
    }

    # ---- Clean and build features --------------------------------------------
    train = clean(train_raw, fill_values, "train.csv")
    print("\nSurvival rate by engineered Title:")
    print(train.groupby("Title")["Survived"].mean().round(3).to_string())

    # X = the inputs (features), y = what we want to predict (survived or not).
    X = to_matrix(train)
    y = train["Survived"]
    print(f"\nModel matrix after one-hot encoding: {X.shape[0]} rows x {X.shape[1]} columns")
    print(f"Columns: {list(X.columns)}")

    # ---- Step 3: Fit the model -----------------------------------------------
    # Logistic regression predicts the probability that someone survived.
    # If the probability is above 0.5, we predict "survived".
    section("Step 3: Logistic regression")

    # Practice run: train on 80% and test on the 20% the model never saw.
    # This is our honest estimate of how well it works on new passengers.
    # stratify=y keeps the same survival rate (about 38%) in both parts.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    # max_iter=1000 gives the solver enough steps to finish (the default 100
    # sometimes isn't enough and prints a warning).
    holdout = LogisticRegression(max_iter=1000)
    holdout.fit(X_tr, y_tr)
    print(f"80/20 split check -> train acc: {accuracy_score(y_tr, holdout.predict(X_tr)):.4f}, "
          f"held-out validation acc: {accuracy_score(y_val, holdout.predict(X_val)):.4f}")

    # Final model: train on all 891 passengers, since more data = a better model.
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    # How to read the coefficients: positive means higher survival odds,
    # negative means lower. Each is compared to the baseline level
    # (1st class, port C, title Master).
    coefs = pd.Series(model.coef_[0], index=X.columns).sort_values()
    print("\nFinal model fit on all 891 training rows. Coefficients (log-odds):")
    print(coefs.round(3).to_string())
    print(f"Intercept: {model.intercept_[0]:.3f}")

    # ---- Step 4: Training accuracy -------------------------------------------
    # Accuracy = share of passengers predicted correctly. This is a bit
    # optimistic, because the model has already seen these passengers.
    section("Step 4: Accuracy on the training set")
    train_acc = accuracy_score(y, model.predict(X))
    print(f"Training accuracy: {train_acc:.4f} ({train_acc:.1%})")

    # ---- Step 5: Predict the test set ----------------------------------------
    section("Step 5: Load test.csv and predict")
    test_raw = load_csv("test.csv")
    print(f"test.csv loaded: {test_raw.shape[0]} rows x {test_raw.shape[1]} columns")

    # Same cleaning, same fill values from training, same columns.
    test = clean(test_raw, fill_values, "test.csv")
    X_test = to_matrix(test, columns=X.columns)
    test_pred = model.predict(X_test)
    print(f"\nPredicted survivors in test set: {test_pred.sum()} of {len(test_pred)} "
          f"({test_pred.mean():.1%})")

    # Save in the format Kaggle expects: PassengerId, Survived.
    out_path = os.path.join(DATA_DIR, "predictions_python.csv")
    pd.DataFrame({"PassengerId": test_raw["PassengerId"], "Survived": test_pred}).to_csv(
        out_path, index=False
    )
    print(f"Predictions saved to {out_path} (Kaggle submission format)")

    # ---- Step 6: Test accuracy -----------------------------------------------
    section("Step 6: Accuracy on the test set")
    if "Survived" in test_raw.columns:
        # Only possible if someone gives us a test file with the answers.
        print(f"Test accuracy: {accuracy_score(test_raw['Survived'], test_pred):.4f}")
    else:
        # Kaggle's real test.csv has no answers, so we explain what we can do.
        print("Kaggle's test.csv does not include the true 'Survived' labels, so true "
              "test accuracy\ncan only be computed by uploading predictions_python.csv "
              "to Kaggle.")
        print(f"Best local estimate (held-out 20% of train): "
              f"{accuracy_score(y_val, holdout.predict(X_val)):.4f}")

        # gender_submission.csv is Kaggle's simple guess: every woman survives,
        # every man dies. Comparing against it is a sanity check, not accuracy.
        gs_path = os.path.join(DATA_DIR, "gender_submission.csv")
        if os.path.exists(gs_path):
            gs = pd.read_csv(gs_path).set_index("PassengerId")["Survived"]
            agree = (gs.loc[test_raw["PassengerId"]].values == test_pred).mean()
            print(f"Agreement with Kaggle's gender_submission.csv baseline "
                  f"(women survive, men don't): {agree:.4f}")

    section("Done")


if __name__ == "__main__":
    main()
