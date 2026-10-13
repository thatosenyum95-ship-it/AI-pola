"""Train a small Multinomial Naive Bayes text classifier using stdlib only."""
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "datasets" / "teks.csv"
OUT = ROOT / "models" / "text_model.json"


def tokenize(text):
    return re.findall(r"[a-zA-ZÀ-ÿ0-9]+", str(text).lower())


def train(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("At least one labeled training example is required.")
    if any(not str(row.get("label", "")).strip() for row in rows):
        raise ValueError("Every training example must have a non-empty label.")
    labels = Counter(row["label"] for row in rows)
    word_counts = {label: Counter() for label in labels}
    totals = Counter()
    vocabulary = set()
    for row in rows:
        words = tokenize(row.get("text", ""))
        word_counts[row["label"]].update(words)
        totals[row["label"]] += len(words)
        vocabulary.update(words)
    return {
        "labels": dict(labels),
        "word_counts": {key: dict(value) for key, value in word_counts.items()},
        "totals": dict(totals),
        "vocabulary": sorted(vocabulary),
    }


def predict(model, text):
    vocab, labels = model.get("vocabulary", []), model.get("labels", {})
    if not labels or not vocab:
        return {"label": "unknown", "scores": {}}
    words = tokenize(text)
    total_docs, vocab_size = sum(labels.values()), len(vocab)
    scores = {}
    for label, count in labels.items():
        denom = model["totals"].get(label, 0) + vocab_size
        if denom <= 0:
            continue
        score = math.log(count / total_docs)
        word_counts = model["word_counts"].get(label, {})
        for word in words:
            score += math.log((word_counts.get(word, 0) + 1) / denom)
        scores[label] = score
    if not scores:
        return {"label": "unknown", "scores": {}}
    best = max(scores, key=scores.get)
    return {"label": best, "scores": {key: round(value, 4) for key, value in scores.items()}}


def main():
    with DATA.open(encoding="utf-8", newline="") as f:
        model = train(list(csv.DictReader(f)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Trained text classifier on {sum(model['labels'].values())} examples.")
    print(f"Saved model: {OUT.relative_to(ROOT)}")
    for sample in ("halo selamat pagi", "bagaimana cara belajar", "saya merasa senang"):
        print(f"{sample!r} -> {predict(model, sample)['label']}")


if __name__ == "__main__":
    main()
