"""Tests for training metrics."""
import pytest

from src.training.metrics import compute_metrics


class TestMetrics:
    def test_compute_metrics_rejects_empty_predictions(self):
        with pytest.raises(ValueError, match="empty"):
            compute_metrics([], [])

    def test_bleu_exact_match(self):
        preds = ["the cat sat on the mat"]
        refs = [["the cat sat on the mat"]]
        metrics = compute_metrics(preds, refs)
        assert metrics["BLEU"] > 90

    def test_bleu_partial(self):
        preds = ["the cat sat on mat"]
        refs = [["the cat sat on the mat"]]
        metrics = compute_metrics(preds, refs)
        assert 0 < metrics["BLEU"] < 100

    def test_rouge_l(self):
        preds = ["hello world"]
        refs = [["hello world"]]
        metrics = compute_metrics(preds, refs)
        assert "ROUGE-L" in metrics
        assert metrics["ROUGE-L"] > 50

    def test_returns_only_bleu_and_rouge(self):
        preds = ["test"]
        refs = [["test"]]
        metrics = compute_metrics(preds, refs)
        assert set(metrics.keys()) == {"BLEU", "ROUGE-L"}

    def test_multiple_samples(self):
        preds = ["a", "b", "c"]
        refs = [["a"], ["b"], ["c"]]
        metrics = compute_metrics(preds, refs)
        assert isinstance(metrics["BLEU"], float)
        assert isinstance(metrics["ROUGE-L"], float)
