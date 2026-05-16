"""PyTorch Lightning wrapper for VisualLLMBaseline."""
import torch
from pytorch_lightning import LightningModule
from src.models.visual_llm_baseline import VisualLLMBaseline
from src.training.metrics import compute_metrics


class VisualLLMLitModule(LightningModule):
    """LightningModule wrapper for VisualLLMBaseline with BLEU/ROUGE metrics."""

    def __init__(
        self,
        pretrained_llm: str,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        lr: float = 2e-4,
        weight_decay: float = 0.01,
        max_new_tokens: int = 128,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = VisualLLMBaseline(
            pretrained_llm=pretrained_llm,
            use_lora=True,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.model.train()

        self.val_predictions = []
        self.val_references = []

    def forward(self, features, mask, targets):
        return self.model(features, mask, targets)

    def training_step(self, batch, batch_idx):
        features = batch["features"]
        mask = batch["attention_mask"]
        targets = batch["target_texts"]

        loss = self.model(features, mask, targets)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        features = batch["features"]
        mask = batch["attention_mask"]
        targets = batch["target_texts"]

        loss = self.model(features, mask, targets)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=len(targets),
        )

        with torch.no_grad():
            predictions = self.model.generate(
                features, mask, max_new_tokens=self.hparams.max_new_tokens
            )
            self.val_predictions.extend(predictions)
            self.val_references.extend([[t] for t in targets])

    def on_validation_epoch_end(self):
        if not self.val_predictions:
            return

        metrics = compute_metrics(self.val_predictions, self.val_references)

        self.log("val_bleu", metrics["BLEU"], prog_bar=True, sync_dist=False)
        self.log("val_rouge", metrics["ROUGE-L"], prog_bar=True, sync_dist=False)

        self.val_predictions.clear()
        self.val_references.clear()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
