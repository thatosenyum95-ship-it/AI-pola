import unittest

from sequence_model import predict_next, validate_numbers, walk_forward_backtest


class SequenceModelTests(unittest.TestCase):
    def test_preserves_leading_zero(self):
        self.assertEqual(validate_numbers(["0940"]), ["0940"])

    def test_rejects_invalid_values(self):
        for value in ("940", "12345", "12a4", ""):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate_numbers([value])

    def test_requires_minimum_history(self):
        self.assertIsNone(predict_next(["1234", "1235", "1236", "1237"]))

    def test_frequency_candidate(self):
        self.assertEqual(predict_next(["1111", "2222", "1111", "2222", "1111"]), "1111")

    def test_tie_break_uses_most_recent_tied_digit(self):
        self.assertEqual(predict_next(["1111", "2222", "1111", "2222", "1111", "2222"]), "2222")

    def test_minimum_history_must_be_positive_integer(self):
        for value in (0, -1, 1.5, True):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    predict_next(["1234"] * 5, minimum_history=value)

    def test_walk_forward_does_not_test_initial_training_window(self):
        result = walk_forward_backtest(["1111"] * 6)
        self.assertEqual(result["tested"], 1)
        self.assertEqual(result["correct"], 1)
        self.assertEqual(result["accuracy"], 1.0)

    def test_backtest_uses_only_prior_values(self):
        result = walk_forward_backtest(["1111", "2222", "1111", "2222", "1111", "9999"])
        self.assertEqual(result["records"][0]["history_size"], 5)
        self.assertEqual(result["records"][0]["predicted"], "1111")
        self.assertEqual(result["records"][0]["actual"], "9999")

    def test_empty_backtest_has_no_accuracy(self):
        result = walk_forward_backtest(["1234", "2345"])
        self.assertIsNone(result["accuracy"])
        self.assertEqual(result["tested"], 0)
        self.assertEqual(result["correct"], 0)


if __name__ == "__main__":
    unittest.main()
