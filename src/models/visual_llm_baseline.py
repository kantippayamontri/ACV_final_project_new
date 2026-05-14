"""Visual-only LLM baseline: projector + LoRA decoder + training/generation."""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from src.models.projector import VisualProjector


FEATURE_DIM = 768
PROMPT_TEMPLATE = "Translate this sign language video into English. The following are the video tokens:"


class VisualLLMBaseline(nn.Module):
    """Maps VideoMAE features through a projector into a frozen LLM with LoRA.

    Hidden size is inferred from the pretrained LLM config, not user-provided.
    """

    def __init__(self,
                 pretrained_llm: str,
                 use_lora: bool = True,
                 lora_r: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 lora_target_modules: tuple = ("q_proj", "v_proj"),
                 projector_layers: int = 3):
        super().__init__()

        self.pretrained_llm = pretrained_llm
        self.use_lora = use_lora

        config = AutoConfig.from_pretrained(pretrained_llm)
        self.llm_hidden_size = config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_llm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.llm = AutoModelForCausalLM.from_pretrained(pretrained_llm, torch_dtype=torch.float16)
        self.llm.requires_grad_(False)
        self._llm_dtype = next(self.llm.parameters()).dtype

        if use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=list(lora_target_modules),
            )
            self.llm = get_peft_model(self.llm, lora_config)

        self.projector = VisualProjector(
            input_dim=FEATURE_DIM,
            output_dim=self.llm_hidden_size,
            num_layers=projector_layers,
        )

    def _tokenize_targets(self, targets: list[str], max_length: int = 512):
        tokenized = self.tokenizer(
            targets, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def _build_inputs_embeds(self, projected: torch.Tensor, target_ids: torch.Tensor):
        B = projected.shape[0]
        prompt_ids = self.tokenizer(
            [PROMPT_TEMPLATE] * B, return_tensors="pt",
            padding=True, truncation=True,
        ).input_ids.to(projected.device)

        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        target_embeds = self.llm.get_input_embeddings()(target_ids)

        inputs_embeds = torch.cat([prompt_embeds, projected, target_embeds], dim=1)
        return inputs_embeds, prompt_ids, target_ids

    def _build_labels_mask(self, prompt_ids: torch.Tensor, projected_len: int,
                           target_ids: torch.Tensor, device: torch.device):
        B = prompt_ids.shape[0]
        target_len = target_ids.shape[1]
        total_len = prompt_ids.shape[1] + projected_len + target_len

        labels = torch.full((B, total_len), -100, dtype=torch.long, device=device)
        labels[:, prompt_ids.shape[1] + projected_len:] = target_ids
        labels[:, prompt_ids.shape[1]:prompt_ids.shape[1] + projected_len] = -100

        return labels

    def forward(self, features: torch.Tensor, mask: torch.Tensor, targets: list[str]):
        projected, _ = self.projector(features.float(), mask)
        projected = projected.to(dtype=self._llm_dtype)
        target_ids, target_mask = self._tokenize_targets(targets)
        target_ids = target_ids.to(features.device)

        inputs_embeds, prompt_ids, target_ids_tensor = self._build_inputs_embeds(projected, target_ids)
        labels = self._build_labels_mask(prompt_ids, projected.shape[1], target_ids, features.device)

        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=features.device)
        attn_mask[:, prompt_ids.shape[1]:prompt_ids.shape[1] + mask.shape[1]] = mask.long()

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                          labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, features: torch.Tensor, mask: torch.Tensor,
                 max_new_tokens: int = 128, **gen_kwargs):
        projected, _ = self.projector(features.float(), mask)
        projected = projected.to(dtype=self._llm_dtype)

        B = projected.shape[0]
        prompt_ids = self.tokenizer(
            [PROMPT_TEMPLATE] * B, return_tensors="pt",
            padding=True, truncation=True,
        ).input_ids.to(features.device)

        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
        inputs_embeds = torch.cat([prompt_embeds, projected], dim=1)

        attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=features.device)
        attn_mask[:, prompt_ids.shape[1]:prompt_ids.shape[1] + mask.shape[1]] = mask.long()

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return predictions
