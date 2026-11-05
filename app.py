
import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

app = Flask(__name__)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "internal_marks",
    "attendance",
    "assignment_marks"
]

model = None


# Request tracking
@app.before_request
def before_request():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )
    g.start_time = time.perf_counter()


@app.after_request
def after_request(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())

    duration_ms = (time.perf_counter() - start_time) * 1000

    log_method = logger.warning if response.status_code >= 400 else logger.info

    log_method(
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    response.headers["X-Request-ID"] = request_id

    return response


# Load and cache the machine learning model
def load_model():
    global model

    if model is None:
        try:
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    "student_result_model.pkl was not found."
                )

            model = joblib.load(MODEL_PATH)

            logger.info(
                "event=model_loaded model_file=%s",
                MODEL_PATH
            )

        except Exception:
            logger.exception("event=model_load_failed")
            raise

    return model


# Root endpoint
@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
        "deployment": "Practical-10",
        "message": "Latest Docker image deployed successfully"
    })


# Health endpoint
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


# Prediction endpoint
@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not data or not isinstance(data, dict):
        logger.warning(
            "event=validation_error request_id=%s reason=invalid_json",
            g.request_id
        )

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=validation_error request_id=%s missing_fields=%s",
            g.request_id,
            missing_fields
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: data[feature]
            for feature in FEATURES
        }])

        loaded_model = load_model()

        prediction_code = int(
            loaded_model.predict(sample)[0]
        )

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id,
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


if __name__ == "__main__":
    logger.info("event=application_starting port=5000")

    app.run(
        host="0.0.0.0",
        port=5000
    )
