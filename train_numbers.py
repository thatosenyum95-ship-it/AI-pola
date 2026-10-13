"""Train a simple linear regression model using only Python's standard library."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "angka.csv"
OUT = ROOT / "models" / "number_model.json"

def train(rows):
    xs = [float(row["x"]) for row in rows]
    ys = [float(row["y"]) for row in rows]
    if len(xs) < 2:
        raise ValueError("At least two training examples are required.")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Training x values must not all be identical.")
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
    intercept = my - slope * mx
    return {"slope": slope, "intercept": intercept, "training_examples": len(xs)}

def predict(model, x):
    return model["slope"] * float(x) + model["intercept"]

def main():
    with DATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    model = train(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Trained linear regression on {len(rows)} examples.")
    print(f"Model: y = {model['slope']:.4f}x + {model['intercept']:.4f}")
    print(f"Saved model: {OUT.relative_to(ROOT)}")
    print(f"Prediction for x=11: {predict(model, 11):.2f}")

if __name__ == "__main__":
    main()
