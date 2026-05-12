import csv
import tempfile
import unittest

from stock_prediction_agents import (
    ManagerAgent,
    NewsSample,
    load_financial_news_csv,
)


class StockPredictionAgentTests(unittest.TestCase):
    def test_end_to_end_training_and_explanation(self):
        samples = [
            NewsSample("Company reports strong earnings growth and profit beat", 1),
            NewsSample("Stock rises after major contract win", 1),
            NewsSample("Revenue jumps and outlook upgraded by analysts", 1),
            NewsSample("Shares climb on positive guidance", 1),
            NewsSample("Bullish momentum with rising demand and margins", 1),
            NewsSample("Company misses earnings and cuts guidance", 0),
            NewsSample("Stock drops after weak sales and layoffs", 0),
            NewsSample("Profit warning and falling revenue hit sentiment", 0),
            NewsSample("Bearish outlook after legal investigation", 0),
            NewsSample("Shares plunge on downgrade and losses", 0),
            NewsSample("Strong quarter with record cash flow", 1),
            NewsSample("Market rallies as firm announces expansion", 1),
            NewsSample("Demand slowdown triggers concerns and decline", 0),
            NewsSample("Debt issues and weak forecast pressure shares", 0),
        ]
        for seed in (7, 11):
            with self.subTest(seed=seed):
                manager = ManagerAgent(random_seed=seed)
                metrics = manager.run_iterative_training(samples)
                self.assertGreaterEqual(metrics["validation_accuracy"], 0.0)
                self.assertGreaterEqual(metrics["test_accuracy"], 0.0)

                output = manager.predict_with_justification(
                    "Analysts upgrade stock after strong demand"
                )
                self.assertIn(output["prediction"], {"UP", "DOWN"})
                self.assertIn("Top evidence tokens:", output["explanation"])

                manual_rows = manager.generate_manual_evaluation_rows(samples, limit=5)
                self.assertEqual(len(manual_rows), 5)
                self.assertIn("manual_score_1_to_5", manual_rows[0])

    def test_csv_loader_detects_supported_columns(self):
        with tempfile.NamedTemporaryFile("w+", newline="", suffix=".csv") as handle:
            writer = csv.DictWriter(handle, fieldnames=["headline", "movement"])
            writer.writeheader()
            writer.writerow({"headline": "Stock rises after guidance raise", "movement": "up"})
            writer.writerow({"headline": "Shares fall after poor quarter", "movement": "down"})
            handle.flush()

            data = load_financial_news_csv(handle.name)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0].label, 1)
        self.assertEqual(data[1].label, 0)


if __name__ == "__main__":
    unittest.main()
