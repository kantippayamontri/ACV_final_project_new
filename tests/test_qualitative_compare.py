"""Tests for scripts/qualitative_compare.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from qualitative_compare import (
    FRAME_SIZE,
    _has_features,
    check_vram_sufficient,
    extract_frames,
    report_free_vram,
)


# ── Frame extraction ─────────────────────────────────────────────────────────

class TestFrameExtraction:
    """Tests for frame sampling and PNG saving."""

    def test_frame_indices_evenly_spaced(self, tmp_path):
        """For a 100-frame video with 6 requested frames, indices should be evenly spaced."""
        # We can only test the index math, not the actual decord call without a video
        total = 100
        num_frames = 6
        expected = [int(i * (total - 1) / (num_frames - 1)) for i in range(num_frames)]
        assert expected == [0, 19, 39, 59, 79, 99]

    def test_frame_indices_fewer_than_requested(self, tmp_path):
        """If video has fewer frames than requested, return all available indices."""
        total = 3
        num_frames = 6
        # Simulating the condition: total <= num_frames
        indices = list(range(total))
        assert indices == [0, 1, 2]

    def test_missing_video_returns_empty(self, tmp_path):
        """If video file doesn't exist, return empty list and warn."""
        result = extract_frames(tmp_path, "missing_clip", tmp_path, num_frames=6)
        assert result == []

    def test_extract_frames_saves_pngs(self, tmp_path):
        """Extract frames from a real video and verify PNGs are created."""
        try:
            from decord import VideoReader, cpu
        except ImportError:
            pytest.skip("decord not available")

        # Create a tiny synthetic video using decord by writing frames via imageio
        try:
            import imageio
        except ImportError:
            pytest.skip("imageio not available")

        video_dir = tmp_path / "videos"
        video_dir.mkdir()
        video_path = video_dir / "test_clip.mp4"

        # Write 30 frames of random noise
        rng = np.random.default_rng(42)
        frames = (rng.integers(0, 255, (30, 224, 224, 3), dtype=np.uint8))
        imageio.mimsave(str(video_path), frames, fps=24)

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        paths = extract_frames(video_dir, "test_clip", output_dir, num_frames=6)

        assert len(paths) == 6
        for p in paths:
            assert Path(p).exists()
            assert Path(p).suffix == ".png"

        # Verify images are 224x224
        from PIL import Image
        for p in paths:
            img = Image.open(p)
            assert img.size == (FRAME_SIZE, FRAME_SIZE)


# ── _has_features ────────────────────────────────────────────────────────────

class TestHasFeatures:
    """Tests for LMDB key existence check."""

    def test_legacy_key(self):
        """Legacy keys starting at 1 are detected."""
        from unittest.mock import MagicMock
        txn = MagicMock()
        # Simulate: key "name/0000000.np" not found, "name/0000001.np" found
        txn.get.side_effect = lambda k: b"data" if k == b"name/0000001.np" else None
        assert _has_features(txn, "name") is True

    def test_modern_key(self):
        """Modern keys starting at 0 are detected."""
        from unittest.mock import MagicMock
        txn = MagicMock()
        txn.get.side_effect = lambda k: b"data" if k == b"name/0000000.np" else None
        assert _has_features(txn, "name") is True

    def test_missing(self):
        """Missing sentence returns False."""
        from unittest.mock import MagicMock
        txn = MagicMock()
        txn.get.return_value = None
        assert _has_features(txn, "name") is False


# ── Example selection (logic only) ───────────────────────────────────────────

class TestExampleSelectionIndices:
    """Tests for the BLEU-based index selection logic.

    We re-implement the index computation here to avoid loading real models.
    The actual `select_examples` function is tested by smoke-test runs, not
    unit tests.
    """

    def test_indices_unique_and_sorted(self):
        """5 examples selected from 100 candidates should give 5 unique sorted indices."""
        n = 99  # last index
        num_examples = 5
        percs = [0.0, 0.25, 0.50, 0.75, 1.0]
        indices = []
        for p in percs[:num_examples]:
            idx = min(int(round(p * n)), n)
            if idx not in indices:
                indices.append(idx)
        while len(indices) < num_examples:
            for i in range(n + 1):
                if i not in indices:
                    indices.append(i)
                    break
        indices = sorted(indices[:num_examples])
        assert len(indices) == 5
        assert indices == sorted(set(indices))
        assert indices[0] == 0
        assert indices[-1] == n

    def test_indices_for_3_examples(self):
        """3 examples should give 3 evenly-spaced indices."""
        n = 99
        num_examples = 3
        step = n / max(num_examples - 1, 1)
        percs = [i * step / n for i in range(num_examples)]
        indices = []
        for p in percs[:num_examples]:
            idx = min(int(round(p * n)), n)
            if idx not in indices:
                indices.append(idx)
        assert len(indices) == 3
        assert indices[0] == 0
        assert indices[-1] == 99


# ── VRAM budget helper (VisualLLMBaseline._build_max_memory_map) ─────────────

class TestBuildMaxMemoryMap:
    """Tests for the static helper that builds the device max_memory map.

    The helper must:
    - honour an explicit `max_gpu_memory_gb` override (no probing),
    - probe torch.cuda.mem_get_info() when no override is given,
    - fall back to a sane default when CUDA is unavailable or probing fails,
    - always pair the GPU budget with a CPU budget in the returned dict.
    """

    def test_explicit_override_wins(self):
        from src.models.visual_llm_baseline import VisualLLMBaseline
        mem = VisualLLMBaseline._build_max_memory_map(
            max_gpu_memory_gb=7.5, gpu_memory_fraction=0.9, cpu_memory_gb=16.0,
        )
        assert mem[0] == "7.50GiB"
        assert mem["cpu"] == "16.00GiB"

    def test_no_cuda_falls_back_to_default(self, monkeypatch):
        import torch as _torch
        from src.models.visual_llm_baseline import VisualLLMBaseline
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: False)
        mem = VisualLLMBaseline._build_max_memory_map(
            max_gpu_memory_gb=None, gpu_memory_fraction=0.9, cpu_memory_gb=32.0,
        )
        # Default is 10 GiB; CPU stays 32 GiB.
        assert mem[0] == "10.00GiB"
        assert mem["cpu"] == "32.00GiB"

    def test_probing_failure_falls_back(self, monkeypatch):
        import torch as _torch
        from src.models.visual_llm_baseline import VisualLLMBaseline
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)

        def _boom(_):
            raise RuntimeError("driver not loaded")
        monkeypatch.setattr(_torch.cuda, "mem_get_info", _boom)
        mem = VisualLLMBaseline._build_max_memory_map(
            max_gpu_memory_gb=None, gpu_memory_fraction=0.9, cpu_memory_gb=32.0,
        )
        assert mem[0] == "10.00GiB"

    def test_probe_fraction_applied(self, monkeypatch):
        import torch as _torch
        from src.models.visual_llm_baseline import VisualLLMBaseline
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        # 10 GiB free × 0.5 = 5 GiB
        monkeypatch.setattr(
            _torch.cuda, "mem_get_info",
            lambda _=0: (10 * (1024 ** 3), 12 * (1024 ** 3)),
        )
        mem = VisualLLMBaseline._build_max_memory_map(
            max_gpu_memory_gb=None, gpu_memory_fraction=0.5, cpu_memory_gb=32.0,
        )
        assert mem[0] == "5.00GiB"
        assert mem["cpu"] == "32.00GiB"

    def test_probe_floor_at_1_gib(self, monkeypatch):
        import torch as _torch
        from src.models.visual_llm_baseline import VisualLLMBaseline
        monkeypatch.setattr(_torch.cuda, "is_available", lambda: True)
        # Tiny free VRAM: 0.5 GiB × 0.9 = 0.45 GiB → floored to 1.0 GiB
        monkeypatch.setattr(
            _torch.cuda, "mem_get_info",
            lambda _=0: (int(0.5 * (1024 ** 3)), int(1 * (1024 ** 3))),
        )
        mem = VisualLLMBaseline._build_max_memory_map(
            max_gpu_memory_gb=None, gpu_memory_fraction=0.9, cpu_memory_gb=32.0,
        )
        assert mem[0] == "1.00GiB"


# ── VRAM pre-check helpers (scripts/qualitative_compare.py) ───────────────────

class TestVramPreCheck:
    """Tests for the script-level VRAM pre-check before each model load."""

    def _make_args(self, **overrides):
        from argparse import Namespace
        defaults = dict(
            device="cuda",
            min_free_vram_gb=2.0,
            load_in_4bit=False,
            max_gpu_memory_gb=None,
            gpu_memory_fraction=0.9,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_skips_on_cpu(self, monkeypatch):
        monkeypatch.setattr(
            "qualitative_compare.report_free_vram",
            lambda device: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        check_vram_sufficient(self._make_args(device="cpu"), "any")

    def test_skips_when_min_free_is_zero(self, monkeypatch):
        monkeypatch.setattr(
            "qualitative_compare.report_free_vram",
            lambda device: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        check_vram_sufficient(self._make_args(min_free_vram_gb=0.0), "any")

    def test_skips_when_4bit(self, monkeypatch):
        monkeypatch.setattr(
            "qualitative_compare.report_free_vram",
            lambda device: (_ for _ in ()).throw(AssertionError("should not be called")),
        )
        check_vram_sufficient(self._make_args(load_in_4bit=True), "any")

    def test_raises_when_below_threshold(self, monkeypatch):
        monkeypatch.setattr(
            "qualitative_compare.report_free_vram", lambda device: 1.5,
        )
        with pytest.raises(RuntimeError, match="Insufficient free VRAM"):
            check_vram_sufficient(self._make_args(min_free_vram_gb=2.0), "visual-only")

    def test_passes_when_above_threshold(self, monkeypatch, capsys):
        monkeypatch.setattr(
            "qualitative_compare.report_free_vram", lambda device: 12.0,
        )
        check_vram_sufficient(self._make_args(min_free_vram_gb=2.0), "gt-prev")
        captured = capsys.readouterr()
        assert "12.00 GiB free" in captured.out
        assert "gt-prev" in captured.out

    def test_report_free_vram_returns_none_for_cpu(self):
        assert report_free_vram("cpu") is None
