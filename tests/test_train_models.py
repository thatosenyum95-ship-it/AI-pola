import unittest

from train_numbers import predict as predict_linear, train as train_linear
from train_patterns import predict as predict_cluster, train as train_clusters
from train_text import predict as predict_text, train as train_text_model


class TrainingModelTests(unittest.TestCase):
    def test_linear_regression_fits_line(self):
        model = train_linear([{"x": "1", "y": "2"}, {"x": "2", "y": "4"}, {"x": "3", "y": "6"}])
        self.assertAlmostEqual(model["slope"], 2.0)
        self.assertAlmostEqual(predict_linear(model, 4), 8.0)

    def test_linear_regression_rejects_identical_x(self):
        with self.assertRaises(ValueError):
            train_linear([{"x": 1, "y": 2}, {"x": 1, "y": 3}])

    def test_text_classifier(self):
        model = train_text_model([
            {"text": "halo selamat pagi", "label": "sapaan"},
            {"text": "bagaimana cara belajar", "label": "pertanyaan"},
        ])
        self.assertEqual(predict_text(model, "halo pagi")["label"], "sapaan")

    def test_text_model_rejects_empty_dataset(self):
        with self.assertRaises(ValueError):
            train_text_model([])

    def test_kmeans_assigns_nearby_points(self):
        model = train_clusters([[0, 0], [0, 1], [10, 10], [10, 11]], k=2)
        self.assertEqual(predict_cluster(model, [0, 0.2]), predict_cluster(model, [0, 0.8]))
        self.assertNotEqual(predict_cluster(model, [0, 0]), predict_cluster(model, [10, 10]))

    def test_kmeans_rejects_invalid_k_and_dimension(self):
        with self.assertRaises(ValueError):
            train_clusters([[0, 0]], k=2)
        with self.assertRaises(ValueError):
            train_clusters([[0, 0], [1]], k=1)


if __name__ == "__main__":
    unittest.main()
