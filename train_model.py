
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def main():
    np.random.seed(42)
    n = 300

    marks = np.random.randint(10, 101, n)
    attendance = np.random.randint(40, 101, n)
    assignments = np.random.randint(10, 101, n)

    score = 0.5 * marks + 0.3 * assignments + 0.2 * attendance
    passed = ((score >= 50) & (attendance >= 65)).astype(int)

    data = pd.DataFrame({
        "internal_marks": marks,
        "attendance": attendance,
        "assignment_marks": assignments,
        "passed": passed
    })

    data.to_csv("student_results.csv", index=False)

    X = data[[
        "internal_marks",
        "attendance",
        "assignment_marks"
    ]]
    y = data["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])

    joblib.dump(model, "student_result_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "confusion_matrix": matrix.tolist(),
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model training completed.")
    print(f"Accuracy: {accuracy:.4f}")
    print("Confusion matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
