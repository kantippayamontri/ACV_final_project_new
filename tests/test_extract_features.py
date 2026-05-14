"""Regression tests for extract_features.py bug fixes."""
import lmdb
import torch
from pathlib import Path
from unittest.mock import MagicMock


def _open_lmdb(path, map_size=10 * 1024 ** 2):
    return lmdb.open(str(path), subdir=False, map_size=map_size, readonly=False, meminit=False)


def _read_lmdb_keys(env):
    with env.begin() as txn:
        cursor = txn.cursor()
        return sorted(k.decode() for k, _ in cursor)


class TestEmptyVideoIsMarkedDone:
    """Bug: clips <16 frames return empty tensor, no /done marker,
    so resume mode retries them forever."""

    def test_empty_video_gets_done_marker(self, tmp_path):
        from scripts.extract_features import process_single_video

        env = _open_lmdb(tmp_path / "test.lmdb")
        sentence_name = "short_clip"

        def fake_extract(video_path, model, device, batch_size):
            return torch.empty(0, 768)

        result = process_single_video(
            env=env,
            sentence_name=sentence_name,
            video_path=Path("/fake/video.mp4"),
            model=MagicMock(),
            device="cpu",
            batch_size=8,
            extract_fn=fake_extract,
        )

        assert result in ("skipped_empty", "ok")
        keys = _read_lmdb_keys(env)
        assert f"{sentence_name}/done" in keys, f"Empty video not marked done, keys: {keys}"

        env.close()


class TestDecodeErrorDoesNotAbortJob:
    """Bug: no try/except around VideoReader/decode/inference.
    Single corrupt video aborts entire job."""

    def test_transient_exception_does_not_mark_done(self, tmp_path):
        from scripts.extract_features import process_single_video

        env = _open_lmdb(tmp_path / "test.lmdb")
        sentence_name = "corrupt_clip"

        def bad_extract(video_path, model, device, batch_size):
            raise RuntimeError("decord failed: corrupt video")

        result = process_single_video(
            env=env,
            sentence_name=sentence_name,
            video_path=Path("/fake/corrupt.mp4"),
            model=MagicMock(),
            device="cpu",
            batch_size=8,
            extract_fn=bad_extract,
        )

        assert result == "skipped_error"

        keys = _read_lmdb_keys(env)
        assert f"{sentence_name}/done" not in keys

        env.close()


class TestFeatureKeysStartAtZero:
    """Bug: main script uses feat_idx+1 (1-based), test script uses feat_idx (0-based)."""

    def test_keys_are_zero_based(self, tmp_path):
        from scripts.extract_features import process_single_video

        env = _open_lmdb(tmp_path / "test.lmdb")
        sentence_name = "test_clip"

        fake_features = torch.randn(3, 768)

        def fake_extract(video_path, model, device, batch_size):
            return fake_features

        result = process_single_video(
            env=env,
            sentence_name=sentence_name,
            video_path=Path("/fake/video.mp4"),
            model=MagicMock(),
            device="cpu",
            batch_size=8,
            extract_fn=fake_extract,
        )

        assert result == "ok"

        keys = _read_lmdb_keys(env)
        feat_keys = [k for k in keys if not k.endswith("/done")]
        assert f"{sentence_name}/0000000.np" in feat_keys, f"First key not 0-based: {feat_keys}"
        assert f"{sentence_name}/0000001.np" in feat_keys
        assert f"{sentence_name}/0000002.np" in feat_keys
        assert len(feat_keys) == 3

        env.close()


class TestLegacyKeysAreMigrated:
    """Bug: clips extracted with the old 1-based scheme are skipped forever on resume."""

    def test_migrate_legacy_done_clip_to_zero_based_keys(self, tmp_path):
        from scripts.extract_features import migrate_legacy_feature_keys

        env = _open_lmdb(tmp_path / "test.lmdb")
        sentence_name = "legacy_clip"

        with env.begin(write=True) as txn:
            txn.put(f"{sentence_name}/0000001.np".encode("ascii"), b"a")
            txn.put(f"{sentence_name}/0000002.np".encode("ascii"), b"b")
            txn.put(f"{sentence_name}/done".encode("ascii"), b"1")

        migrated = migrate_legacy_feature_keys(env, sentence_name)

        assert migrated is True

        keys = _read_lmdb_keys(env)
        assert f"{sentence_name}/0000000.np" in keys
        assert f"{sentence_name}/0000001.np" in keys
        assert f"{sentence_name}/0000002.np" not in keys
        assert f"{sentence_name}/done" in keys

        env.close()
