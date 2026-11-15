"""Load and use the persisted German Credit risk model."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).parent / "artifacts" / "credit_risk_model.joblib"


def load_model(path: str | Path = MODEL_PATH) -> dict[str, Any]:
    """Load the fitted pipeline and its decision threshold."""
    bundle = joblib.load(path)
    required_keys = {"model", "threshold", "bad_credit_label", "good_credit_label"}
    if not isinstance(bundle, dict) or not required_keys.issubset(bundle):
        raise ValueError(f"Model artifact must contain {sorted(required_keys)}")
    return bundle


def predict_bad_credit_probability(
    applicants: pd.DataFrame, bundle: dict[str, Any] | None = None
) -> np.ndarray:
    """Return each applicant's probability of class 2 (bad credit)."""
    if bundle is None:
        bundle = load_model()
    model = bundle["model"]
    classes = list(model.classes_)
    bad_credit_label = bundle["bad_credit_label"]
    if bad_credit_label not in classes:
        raise ValueError(f"Model does not contain class {bad_credit_label}")
    class_index = classes.index(bad_credit_label)
    return model.predict_proba(applicants)[:, class_index]


def predict_applicants(
    applicants: pd.DataFrame, bundle: dict[str, Any] | None = None
) -> np.ndarray:
    """Classify applicants using the threshold saved with the model."""
    if bundle is None:
        bundle = load_model()
    probabilities = predict_bad_credit_probability(applicants, bundle)
    return np.where(
        probabilities >= bundle["threshold"],
        bundle["bad_credit_label"],
        bundle["good_credit_label"],
    )
