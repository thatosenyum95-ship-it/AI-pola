"""Train a simple linear regression model using only Python's standard library."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "angka.csv"
OUT = ROOT / "models" / "number_model.json"


def train(rows):
    rows = list(rows)
    if len(rows) < 2:
        raise ValueError("At least two training examples are required.")
    try:
        xs = [float(row["x"]) for row in rows]
        ys = [float(row["y"]) for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Each row must contain numeric x and y values.") from exc
    if not all(math.isfinite(value) for value in xs + ys):
        raise ValueError("Training values must be finite numbers.")
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Training x values must not all be identical.")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
    intercept = my - slope * mx
    return {"slope": slope, "intercept": intercept, "training_examples": len(xs)}


def predict(model, x):
    value = float(x)
    if not math.isfinite(value):
        raise ValueError("Prediction input must be a finite number.")
    return model["slope"] * value + model["intercept"]


def main():
    with DATA.open(encoding="utf-8", newline="") as f:
        model = train(list(csv.DictReader(f)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Trained linear regression on {model['training_examples']} examples.")
    print(f"Model: y = {model['slope']:.4f}x + {model['intercept']:.4f}")
    print(f"Prediction for x=11: {predict(model, 11):.2f}")
    print(f"Saved model: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
