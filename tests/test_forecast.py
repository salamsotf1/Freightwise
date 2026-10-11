import unittest

from freightwise.forecast import make_recommendation


class RecommendationTests(unittest.TestCase):
    def test_rising_prices_suggest_booking(self):
        message = make_recommendation("prophet", 10.0, 10.5)
        self.assertIn("rise by 5.0%", message)
        self.assertIn("consider booking now", message)

    def test_falling_prices_suggest_waiting(self):
        message = make_recommendation("xgboost", 10.0, 9.5)
        self.assertIn("fall by 5.0%", message)
        self.assertIn("consider waiting", message)

    def test_small_change_is_described_as_stable(self):
        message = make_recommendation("prophet", 10.0, 10.1)
        self.assertIn("stay broadly stable", message)


if __name__ == "__main__":
    unittest.main()