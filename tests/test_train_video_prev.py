"""Tests for previous-text mixing helpers in train_video_prev."""

import pytest

from scripts import train_video_prev


def test_mix_prev_texts_uses_predictions_when_ratio_zero():
    mixed = train_video_prev.mix_prev_texts(
        prev_texts=["gt-1", "gt-2"],
        sentence_names=["sent-1", "sent-2"],
        prev_predictions={"sent-1": "pred-1", "sent-2": "pred-2"},
        prev_gt_ratio=0.0,
        random_values=iter([0.1, 0.9]),
    )

    assert mixed == ["pred-1", "pred-2"]


def test_mix_prev_texts_uses_prediction_when_ratio_zero_and_draw_is_zero():
    mixed = train_video_prev.mix_prev_texts(
        prev_texts=["gt-1"],
        sentence_names=["sent-1"],
        prev_predictions={"sent-1": "pred-1"},
        prev_gt_ratio=0.0,
        random_values=iter([0.0]),
    )

    assert mixed == ["pred-1"]


def test_mix_prev_texts_falls_back_to_empty_string_when_prediction_missing():
    mixed = train_video_prev.mix_prev_texts(
        prev_texts=["gt-1"],
        sentence_names=["sent-1"],
        prev_predictions={},
        prev_gt_ratio=0.0,
        random_values=iter([0.5]),
    )

    assert mixed == [""]


def test_validate_prev_gt_ratio_rejects_invalid_values():
    with pytest.raises(ValueError, match="--prev-gt-ratio must be between 0.0 and 1.0"):
        train_video_prev.validate_prev_gt_ratio(-0.01)

    with pytest.raises(ValueError, match="--prev-gt-ratio must be between 0.0 and 1.0"):
        train_video_prev.validate_prev_gt_ratio(1.01)

    assert train_video_prev.validate_prev_gt_ratio(0.0) == 0.0
    assert train_video_prev.validate_prev_gt_ratio(1.0) == 1.0


def test_parse_args_accepts_prev_prediction_options():
    args = train_video_prev.parse_args([
        "--train-lmdb", "train.lmdb",
        "--train-metadata", "train.csv",
        "--val-lmdb", "val.lmdb",
        "--val-metadata", "val.csv",
        "--pretrained-llm", "llm",
        "--output-dir", "outputs/run",
        "--prev-predictions", "preds.json",
        "--prev-gt-ratio", "0.5",
    ])

    assert args.prev_predictions == "preds.json"
    assert args.prev_gt_ratio == 0.5
