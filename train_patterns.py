"""Train a tiny K-Means clustering model using only Python's standard library."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "pola.csv"
OUT = ROOT / "models" / "pattern_model.json"

def distance(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def mean(points):
    return [sum(p[i] for p in points) / len(points) for i in range(len(points))]

def train(points, k=3, iterations=30):
    if len(points) < k:
        raise ValueError("Number of points must be at least k.")
    if k > 1:
        centers = [list(points[round(i * (len(points) - 1) / (k - 1))]) for i in range(k)]
    else:
        centers = [mean(points)]
    assignments = [-1] * len(points)
    for _ in range(iterations):
        new_assignments = [min(range(k), key=lambda c: distance(p, centers[c])) for p in points]
        new_centers = []
        for c in range(k):
            cluster = [p for p, a in zip(points, new_assignments) if a == c]
            new_centers.append(mean(cluster) if cluster else centers[c])
        if new_assignments == assignments and new_centers == centers:
            break
        assignments, centers = new_assignments, new_centers
    return {"centers": centers, "clusters": assignments, "training_examples": len(points)}

def predict(model, point):
    return min(range(len(model["centers"])), key=lambda c: distance(point, model["centers"][c]))

def main():
    with DATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    points = [[float(row["x"]), float(row["y"])] for row in rows]
    model = train(points)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, indent=2), encoding="utf-8")
    print(f"Trained K-Means on {len(points)} points.")
    for i, center in enumerate(model["centers"]):
        print(f"  cluster {i}: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"New point (1.0, 1.0) belongs to cluster {predict(model, [1.0, 1.0])}.")
    print(f"Saved model: {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
