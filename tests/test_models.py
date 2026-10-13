import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class TestAiPolaModels(unittest.TestCase):
    def test_text_classifier(self):
        mod = load_module("train_text.py", "train_text")
        rows = [
            {"text": "halo selamat pagi", "label": "sapaan"},
            {"text": "hai selamat malam", "label": "sapaan"},
            {"text": "bagaimana cara belajar", "label": "pertanyaan"},
            {"text": "apa itu python", "label": "pertanyaan"},
        ]
        model = mod.train(rows)
        self.assertEqual(mod.predict(model, "halo pagi")["label"], "sapaan")
        self.assertEqual(mod.predict(model, "bagaimana belajar")["label"], "pertanyaan")

    def test_linear_regression(self):
        mod = load_module("train_numbers.py", "train_numbers")
        rows = [{"x": str(x), "y": str(2*x)} for x in range(1, 6)]
        model = mod.train(rows)
        self.assertAlmostEqual(mod.predict(model, 6), 12.0, places=6)

    def test_kmeans(self):
        mod = load_module("train_patterns.py", "train_patterns")
        points = [[0.0, 0.0], [0.1, 0.1], [10.0, 10.0], [10.1, 10.1]]
        model = mod.train(points, k=2)
        a = mod.predict(model, [0.0, 0.1])
        b = mod.predict(model, [10.0, 10.1])
        self.assertNotEqual(a, b)

if __name__ == "__main__":
    unittest.main()
