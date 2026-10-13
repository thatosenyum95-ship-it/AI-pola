"""Train a deterministic K-Means clustering model using the Python standard library."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "pola.csv"
OUT = ROOT / "models" / "pattern_model.json"


def _validate_points(points):
    try:
        points = [list(map(float, point)) for point in points]
    except (TypeError, ValueError) as exc:
        raise ValueError("Points must contain numeric coordinates.") from exc
    if not points:
        raise ValueError("At least one point is required.")
    dimension = len(points[0])
    if dimension == 0 or any(len(point) != dimension for point in points):
        raise ValueError("All points must have the same non-zero dimension.")
    if any(not math.isfinite(value) for point in points for value in point):
        raise ValueError("Point coordinates must be finite numbers.")
    return points


def distance(a, b):
    if len(a) != len(b):
        raise ValueError("Points must have the same dimension.")
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def mean(points):
    points = _validate_points(points)
    return [sum(point[i] for point in points) / len(points) for i in range(len(points[0]))]


def train(points, k=3, iterations=30):
    points = _validate_points(points)
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= len(points):
        raise ValueError("k must be an integer between 1 and the number of points.")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer.")
    if k == 1:
        centers = [mean(points)]
    else:
        centers = [list(points[round(i * (len(points) - 1) / (k - 1))]) for i in range(k)]
    assignments = [-1] * len(points)
    for _ in range(iterations):
        new_assignments = [min(range(k), key=lambda c: distance(point, centers[c])) for point in points]
        new_centers = []
        for cluster_id in range(k):
            cluster = [point for point, a in zip(points, new_assignments) if a == cluster_id]
            new_centers.append(mean(cluster) if cluster else centers[cluster_id])
        if new_assignments == assignments and new_centers == centers:
            assignments, centers = new_assignments, new_centers
            break
        assignments, centers = new_assignments, new_centers
    return {"centers": centers, "clusters": assignments, "training_examples": len(points)}


def predict(model, point):
    point = _validate_points([point])[0]
    centers = model.get("centers", [])
    if not centers:
        raise ValueError("Model has no cluster centers.")
    if len(point) != len(centers[0]):
        raise ValueError("Point dimension does not match the model.")
    return min(range(len(centers)), key=lambda c: distance(point, centers[c]))


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
