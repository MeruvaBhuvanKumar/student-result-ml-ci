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

    students = 300

    internal_marks = np.random.randint(10, 101, students)
    attendance = np.random.randint(40, 101, students)
    assignment_marks = np.random.randint(10, 101, students)

    total_score = (
        0.5 * internal_marks
        + 0.3 * assignment_marks
        + 0.2 * attendance
    )

    passed = (
        (total_score >= 50)
        & (attendance >= 65)
    ).astype(int)

    data = pd.DataFrame({
        "internal_marks": internal_marks,
        "attendance": attendance,
        "assignment_marks": assignment_marks,
        "passed": passed
    })

    data.to_csv("student_results.csv", index=False)

    X = data[
        ["internal_marks", "attendance", "assignment_marks"]
    ]
    y = data["passed"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(
        y_test, predictions, labels=[0, 1]
    )

    joblib.dump(model, "student_result_model.pkl")

    metrics = {
        "accuracy": round(float(accuracy), 4),
        "confusion_matrix": matrix.tolist(),
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model training completed.")
    print("Accuracy:", accuracy)
    print("Confusion Matrix:")
    print(matrix)


if __name__ == "__main__":
    main()
