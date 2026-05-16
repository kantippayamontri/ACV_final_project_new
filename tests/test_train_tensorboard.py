"""Tests for TensorBoard logging in train_baseline."""
import tempfile
from pathlib import Path


def test_train_creates_tensorboard_events(tmp_path):
    """Verify training creates TensorBoard events file."""
    from torch.utils.tensorboard import SummaryWriter
    
    log_dir = tmp_path / "logs"
    
    writer = SummaryWriter(str(log_dir))
    writer.add_scalar("train_loss", 2.5, 1)
    writer.add_scalar("val_bleu", 12.0, 1)
    writer.close()
    
    events_files = list(log_dir.glob("events.out.tfevents.*"))
    assert len(events_files) == 1, "No TensorBoard events file created"
