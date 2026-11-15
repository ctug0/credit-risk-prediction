import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk_model import (
    load_model,
    predict_applicants,
    predict_bad_credit_probability,
)


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "model_persistence.json"
)


class ModelPersistenceTests(unittest.TestCase):
    def test_loaded_model_matches_saved_original_predictions(self) -> None:
        fixture = json.loads(FIXTURE_PATH.read_text())
        applicants = pd.DataFrame(fixture["applicants"])
        bundle = load_model()

        probabilities = predict_bad_credit_probability(applicants, bundle)
        predictions = predict_applicants(applicants, bundle)

        np.testing.assert_allclose(
            probabilities, fixture["bad_credit_probabilities"]
        )
        np.testing.assert_array_equal(predictions, fixture["predictions"])
        self.assertEqual(bundle["threshold"], 0.35)


if __name__ == "__main__":
    unittest.main()
