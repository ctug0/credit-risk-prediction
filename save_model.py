"""Recreate the notebook's tuned model once and save it for later inference."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

from credit_risk_model import MODEL_PATH, predict_applicants, predict_bad_credit_probability


TARGET = "Good/Bad Credit"
THRESHOLD = 0.35
FEATURE_NAMES = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence_duration",
    "property",
    "age",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job",
    "dependents",
    "telephone",
    "foreign_worker",
    TARGET,
]
TEST_FIXTURE_PATH = (
    Path(__file__).parent / "tests" / "fixtures" / "model_persistence.json"
)


def verification_applicants() -> pd.DataFrame:
    """Create stable, fictional applicants for the save/load regression test."""
    return pd.DataFrame(
        [
            {
                "checking_account": "A11",
                "duration": 12,
                "credit_history": "A34",
                "purpose": "A43",
                "credit_amount": 5000,
                "savings": "A65",
                "employment": "A75",
                "installment_rate": 4,
                "personal_status": "A93",
                "other_debtors": "A101",
                "residence_duration": 4,
                "property": "A121",
                "age": 35,
                "other_installment_plans": "A143",
                "housing": "A152",
                "existing_credits": 1,
                "job": "A173",
                "dependents": 1,
                "telephone": "A191",
                "foreign_worker": "A201",
            },
            {
                "checking_account": "A14",
                "duration": 48,
                "credit_history": "A32",
                "purpose": "A40",
                "credit_amount": 15000,
                "savings": "A61",
                "employment": "A71",
                "installment_rate": 4,
                "personal_status": "A93",
                "other_debtors": "A101",
                "residence_duration": 1,
                "property": "A124",
                "age": 22,
                "other_installment_plans": "A143",
                "housing": "A151",
                "existing_credits": 2,
                "job": "A171",
                "dependents": 1,
                "telephone": "A192",
                "foreign_worker": "A201",
            },
            {
                "checking_account": "A12",
                "duration": 24,
                "credit_history": "A33",
                "purpose": "A42",
                "credit_amount": 8000,
                "savings": "A62",
                "employment": "A73",
                "installment_rate": 2,
                "personal_status": "A92",
                "other_debtors": "A102",
                "residence_duration": 3,
                "property": "A122",
                "age": 45,
                "other_installment_plans": "A142",
                "housing": "A152",
                "existing_credits": 1,
                "job": "A172",
                "dependents": 2,
                "telephone": "A191",
                "foreign_worker": "A201",
            },
        ]
    )


def main() -> None:
    credit = fetch_ucirepo(id=144)
    data = credit.data.original.copy()
    data.columns = FEATURE_NAMES

    x = data.drop(columns=[TARGET])
    y = data[TARGET]
    x_train, _, y_train, _ = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=1234,
    )

    categorical_cols = x_train.select_dtypes(
        include=["object", "category"]
    ).columns
    numerical_cols = x_train.select_dtypes(include=["number"]).columns
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("numerical", StandardScaler(), numerical_cols),
        ]
    )
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    grid = GridSearchCV(
        pipeline,
        {"classifier__C": [0.01, 0.1, 1, 10, 100]},
        cv=5,
        scoring="accuracy",
    )
    grid.fit(x_train, y_train.squeeze())
    if grid.best_params_ != {"classifier__C": 0.1}:
        raise RuntimeError(
            "The tuned model differs from the notebook's recorded C=0.1 model: "
            f"{grid.best_params_}"
        )

    bundle = {
        "model": grid.best_estimator_,
        "threshold": THRESHOLD,
        "bad_credit_label": 2,
        "good_credit_label": 1,
    }
    verification_data = verification_applicants()
    original_probabilities = predict_bad_credit_probability(
        verification_data, bundle
    )
    original_predictions = predict_applicants(verification_data, bundle)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    loaded_bundle = joblib.load(MODEL_PATH)
    loaded_probabilities = predict_bad_credit_probability(
        verification_data, loaded_bundle
    )
    loaded_predictions = predict_applicants(verification_data, loaded_bundle)
    np.testing.assert_allclose(loaded_probabilities, original_probabilities)
    np.testing.assert_array_equal(loaded_predictions, original_predictions)

    TEST_FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fixture = {
        "applicants": verification_data.to_dict(orient="records"),
        "bad_credit_probabilities": original_probabilities.tolist(),
        "predictions": original_predictions.tolist(),
    }
    TEST_FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n")

    print(f"Saved model: {MODEL_PATH}")
    print(f"Best parameters: {grid.best_params_}")
    print("Save/load prediction equivalence: passed")


if __name__ == "__main__":
    main()
