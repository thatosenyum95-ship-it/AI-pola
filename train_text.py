"""Train a small Multinomial Naive Bayes text classifier using only Python's standard library."""
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "teks.csv"
OUT = ROOT / "models" / "text_model.json"

def tokenize(text):
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", text.lower())

def train(rows):
    labels = Counter(row["label"] for row in rows)
    word_counts = {label: Counter() for label in labels}
    totals = Counter()
    vocabulary = set()
    for row in rows:
        words = tokenize(row["text"])
        word_counts[row["label"]].update(words)
        totals[row["label"]] += len(words)
        vocabulary.update(words)
    return {
        "labels": dict(labels),
        "word_counts": {k: dict(v) for k, v in word_counts.items()},
        "totals": dict(totals),
        "vocabulary": sorted(vocabulary),
    }

def predict(model, text):
    words = tokenize(text)
    vocab = model["vocabulary"]
    if not vocab:
        return {"label": "unknown", "scores": {}}
    total_docs = sum(model["labels"].values())
    scores = {}
    for label, count in model["labels"].items():
        score = math.log(count / total_docs)
        denom = model["totals"].get(label, 0) + len(vocab)
        for word in words:
            score += math.log((model["word_counts"][label].get(word, 0) + 1) / denom)
        scores[label] = score
    best = max(scores, key=scores.get)
    return {"label": best, "scores": {k: round(v, 4) for k, v in scores.items()}}

def main():
    with DATA.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    model = train(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Trained text classifier on {len(rows)} examples.")
    print(f"Saved model: {OUT.relative_to(ROOT)}")
    for sample in ["halo selamat pagi", "bagaimana cara belajar", "saya merasa senang"]:
        print(f"{sample!r} -> {predict(model, sample)['label']}")

if __name__ == "__main__":
    main()
