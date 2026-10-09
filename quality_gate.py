import json
import sys


MINIMUM_ACCURACY = 0.85


def main():
    with open("metrics.json") as file:
        metrics = json.load(file)

    accuracy = metrics["accuracy"]

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Required accuracy: {MINIMUM_ACCURACY:.2f}")

    if accuracy < MINIMUM_ACCURACY:
        print("QUALITY GATE FAILED")
        sys.exit(1)

    print("QUALITY GATE PASSED")


if __name__ == "__main__":
    main()
