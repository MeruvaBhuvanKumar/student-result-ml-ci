
import json
import joblib
import unittest
import pandas as pd

from pathlib import Path


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(Path("student_results.csv").exists())

    def test_model_exists(self):
        self.assertTrue(Path("student_result_model.pkl").exists())

    def test_metrics_exist(self):
        self.assertTrue(Path("metrics.json").exists())

    def test_dataset_columns(self):
        data = pd.read_csv("student_results.csv")
        self.assertEqual(
            set(data.columns),
            {
                "internal_marks",
                "attendance",
                "assignment_marks",
                "passed"
            }
        )

    def test_dataset_row_count(self):
        data = pd.read_csv("student_results.csv")
        self.assertEqual(len(data), 300)

    def test_accuracy_range(self):
        with open("metrics.json") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_model_prediction(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "internal_marks": 80,
            "attendance": 90,
            "assignment_marks": 85
        }])

        prediction = model.predict(sample)
        self.assertIn(int(prediction[0]), [0, 1])


if __name__ == "__main__":
    unittest.main()
