# Model Registry

This file records the dataset source, training setup, and evaluation results for
the credit-risk model. Add a new entry rather than overwriting this one when
you train a materially different model.

## Credit Risk Model — v1 (notebook run)

| Field | Recorded value |
|---|---|
| Notebook | `credot-risk_modelTraining.ipynb` |
| Dataset source | UCI Machine Learning Repository, fetched with `ucimlrepo.fetch_ucirepo(id=144)` |
| Dataset name / identifier | Statlog (German Credit Data), UCI repository ID 144 |
| Local raw CSV | None is loaded by this notebook; it fetches the dataset through `ucimlrepo` |
| Dataset checksum | Not recorded; no local source CSV is used |
| Target | `Good/Bad Credit` (`1` = good credit, `2` = bad credit) |
| Train/test split | 80/20, stratified, `random_state=1234` |
| Selected model | Logistic Regression in a scikit-learn pipeline |
| Preprocessing | One-hot encode categorical columns (`handle_unknown="ignore"`); standardize numeric columns |
| Selected regularization | `C=0.1`, selected by 5-fold `GridSearchCV` using accuracy |
| Decision threshold | `0.35` probability for class `2` (bad credit) |
| Threshold selection | Compared thresholds using bad-credit F1 on cross-validated training predictions; 0.35 scored 0.592 |
| Test accuracy | 0.755 |
| Test bad-credit precision | 0.57 |
| Test bad-credit recall | 0.73 |
| Test bad-credit F1 | 0.64 |
| Training date | Not recorded in the notebook |
| Git commit for this exact run | Not recorded. The notebook has uncommitted changes, so the current `HEAD` alone does not identify this exact notebook state. |
| Saved model artifact | `artifacts/credit_risk_model.joblib` (fitted pipeline and threshold saved together) |

### Lineage and reproducibility notes

- The notebook fetches data from UCI at runtime rather than loading a checked-in
  raw CSV. The source identifier is recorded above, but the exact downloaded
  data bytes and retrieval date are not.
- Git can version this registry and the notebook. To associate a future run
  with exact code, commit the notebook and this file, then record that commit
  hash in the entry.
- These metrics are the outputs saved in the notebook; they have not been
  independently rerun as part of creating this registry.
- The persisted artifact was created by repeating the notebook's deterministic
  split and grid search because the original fitted object existed only in the
  notebook's kernel memory. Prediction equivalence was checked before and after
  serialization. Use `python save_model.py` to recreate it and
  `python -m unittest discover -s tests` to verify it.
- The model and metrics are a project experiment, not evidence that the model
  is validated or approved for real credit decisions.
