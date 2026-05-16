"""Minimal smoke test for training pipeline — trains 1 step and exits."""
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.visual_llm_baseline import VisualLLMBaseline
from src.data.how2sign_lmdb_dataset import How2SignLMDBDataset, collate_variable_features
from src.training.train_utils import build_optimizer
from torch.utils.data import DataLoader, Subset


def main():
    val_lmdb = Path("features/val_features.lmdb")
    val_meta = Path("features/val_metadata.csv")

    if not val_lmdb.exists() or not val_meta.exists():
        print("SKIP: val features not found. Run extract_features.py first.")
        return

    ds = How2SignLMDBDataset(str(val_lmdb), str(val_meta))

    test_llm_path = Path("models/Llama-3.2-1B")
    if not test_llm_path.exists():
        test_llm_path = Path("models/Meta-Llama-3-8B")
    if not test_llm_path.exists():
        print("SKIP: no LLM model found in models/")
        return

    print(f"Using LLM: {test_llm_path}")
    print(f"Dataset size: {len(ds)} clips")

    subset = Subset(ds, range(min(2, len(ds))))
    loader = DataLoader(subset, batch_size=1, shuffle=True, collate_fn=collate_variable_features)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model...")
    model = VisualLLMBaseline(
        pretrained_llm=str(test_llm_path),
        use_lora=True,
        lora_r=8,
        lora_alpha=16,
    )
    model = model.to(device)
    model.train()

    optimizer = build_optimizer(model, lr=1e-4)
    print(f"Model loaded. Hidden size: {model.llm_hidden_size}")

    batch = next(iter(loader))
    features = batch["features"].to(device)
    mask = batch["attention_mask"].to(device)
    targets = batch["target_texts"]

    print(f"\nInput: {features.shape} features, targets: {targets}")
    print("Running 1 training step...")

    optimizer.zero_grad()
    loss = model.forward(features=features, mask=mask, targets=targets)
    print(f"Loss: {loss.item():.4f}")
    loss.backward()
    optimizer.step()

    print("Verifying generate...")
    model.eval()
    with torch.no_grad():
        output = model.generate(features[:1], mask[:1], max_new_tokens=16)
    print(f"Generate output: {output}")

    print("\n✓ Smoke test passed — training pipeline works.")


if __name__ == "__main__":
    main()
