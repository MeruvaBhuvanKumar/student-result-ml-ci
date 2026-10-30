
import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_high_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "internal_marks": 85,
                "attendance": 90,
                "assignment_marks": 88
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "PASS"
        )

    def test_low_performance_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "internal_marks": 30,
                "attendance": 55,
                "assignment_marks": 40
            }
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json()["prediction"],
            "FAIL"
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "internal_marks": 85,
                "attendance": 90
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn(
            "missing_fields",
            response.get_json()
        )


if __name__ == "__main__":
    unittest.main()
