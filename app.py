
import logging
import sys
import time
import uuid
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

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

# Prometheus Metrics

HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded"
)

MODEL_LOADED.set(0)


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

    duration_seconds = time.perf_counter() - start_time
    duration_ms = duration_seconds * 1000

    # Exclude Prometheus scraping from user traffic metrics
    if request.path != "/metrics":
        HTTP_REQUESTS.labels(
            method=request.method,
            path=request.path,
            status=str(response.status_code)
        ).inc()

        HTTP_LATENCY.labels(
            method=request.method,
            path=request.path
        ).observe(duration_seconds)

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

            MODEL_LOADED.set(1)

            logger.info(
                "event=model_loaded model_file=%s",
                MODEL_PATH
            )

        except Exception:
            MODEL_LOADED.set(0)
            logger.exception(
                "event=model_load_failed model_file=%s",
                MODEL_PATH
            )
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

        try:
            prediction_code = int(
                loaded_model.predict(sample)[0]
            )
            PREDICTIONS.labels(result="success").inc()

        except Exception:
            PREDICTIONS.labels(result="error").inc()
            raise

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


# Prometheus metrics endpoint
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        content_type=CONTENT_TYPE_LATEST
    )


if __name__ == "__main__":
    logger.info("event=application_starting port=5000")

    app.run(
        host="0.0.0.0",
        port=5000
    )
