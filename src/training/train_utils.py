"""Training utilities: seeding, optimizer, checkpointing, epoch loops."""
from contextlib import nullcontext
import json
import random
import torch
from pathlib import Path


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model, lr: float, weight_decay: float = 0.01):
    """AdamW optimizer that skips frozen params."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def mixed_precision_context(precision: str, device: str):
    """Context manager for autocast + GradScaler (fp16) or a no-op context."""
    if precision == "fp16":
        if device.startswith("cuda"):
            scaler = torch.amp.GradScaler("cuda")
            return torch.amp.autocast("cuda", dtype=torch.float16), scaler
        return nullcontext(), None
    return nullcontext(), None


def save_checkpoint(model, optimizer, epoch: int, metrics: dict, args: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "epoch": epoch,
        "model_state": {k: v.clone() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "args": args,
    }
    torch.save(data, str(path))


def load_checkpoint(model, optimizer, path: Path | None, device: str, checkpoint_data: dict | None = None):
    data = checkpoint_data
    if data is None:
        data = torch.load(str(path), map_location=device, weights_only=False)
    model.load_state_dict(data["model_state"])
    optimizer.load_state_dict(data["optimizer_state"])
    return data["epoch"], data["metrics"], data.get("args", {})


def save_metrics_json(metrics: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)


def save_predictions_jsonl(predictions: list[str], references: list[list[str]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for pred, ref_list in zip(predictions, references):
            json.dump({"prediction": pred, "reference": ref_list[0]}, f)
            f.write("\n")
