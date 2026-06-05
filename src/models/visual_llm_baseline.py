"""Visual-only LLM baseline: projector + LoRA decoder + training/generation."""
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
                 projector_layers: int = 3,
                 load_in_4bit: bool = False):
        super().__init__()

        self.pretrained_llm = pretrained_llm
        self.use_lora = use_lora

        config = AutoConfig.from_pretrained(pretrained_llm)
        self.llm_hidden_size = config.hidden_size

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_llm)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        llm_kwargs = {"dtype": torch.float16, "device_map": "auto"}
        if load_in_4bit:
            llm_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.llm = AutoModelForCausalLM.from_pretrained(pretrained_llm, **llm_kwargs)
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
        self.projector = self.projector.to(self.llm.device)

    def _tokenize_targets(self, targets: list[str], max_length: int = 512):
        tokenized = self.tokenizer(
            targets, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length,
        )
        return tokenized["input_ids"], tokenized["attention_mask"]

    def _build_sample_inputs(self, projected: torch.Tensor, mask: torch.Tensor,
                             target_ids: torch.Tensor | None = None,
                             target_mask: torch.Tensor | None = None):
        prompt_ids = self.tokenizer(
            [PROMPT_TEMPLATE], return_tensors="pt",
            padding=True, truncation=True,
        ).input_ids.to(projected.device)
        embedding_layer = self.llm.get_input_embeddings()
        prompt_embeds = embedding_layer(prompt_ids)[0]

        sample_embeds = []
        sample_attn = []
        sample_labels = []

        for idx in range(projected.shape[0]):
            valid_len = int(mask[idx].sum().item())
            visual_embeds = projected[idx, :valid_len]
            parts = [prompt_embeds, visual_embeds]
            sample_len = prompt_embeds.shape[0] + valid_len

            if target_ids is not None and target_mask is not None:
                valid_target_ids = target_ids[idx, target_mask[idx].bool()]
                if valid_target_ids.numel() > 0:
                    target_embeds = embedding_layer(valid_target_ids.unsqueeze(0))[0]
                    parts.append(target_embeds)
                else:
                    valid_target_ids = target_ids.new_empty((0,))

                labels = torch.full(
                    (sample_len + valid_target_ids.shape[0],),
                    -100,
                    dtype=torch.long,
                    device=projected.device,
                )
                labels[sample_len:] = valid_target_ids
                sample_labels.append(labels)

            sample_input = torch.cat(parts, dim=0)
            sample_embeds.append(sample_input)
            sample_attn.append(torch.ones(sample_input.shape[0], dtype=torch.long, device=projected.device))

        max_len = max(sample.shape[0] for sample in sample_embeds)
        hidden_size = prompt_embeds.shape[-1]
        inputs_embeds = projected.new_zeros((projected.shape[0], max_len, hidden_size))
        attention_mask = torch.zeros((projected.shape[0], max_len), dtype=torch.long, device=projected.device)

        labels = None
        if sample_labels:
            labels = torch.full((projected.shape[0], max_len), -100, dtype=torch.long, device=projected.device)

        for idx, sample_input in enumerate(sample_embeds):
            seq_len = sample_input.shape[0]
            inputs_embeds[idx, :seq_len] = sample_input
            attention_mask[idx, :seq_len] = sample_attn[idx]
            if labels is not None:
                labels[idx, :sample_labels[idx].shape[0]] = sample_labels[idx]

        return inputs_embeds, attention_mask, labels

    def forward(self, features: torch.Tensor, mask: torch.Tensor, targets: list[str]):
        projected, _ = self.projector(features.float(), mask)
        projected = projected.to(dtype=self._llm_dtype)
        target_ids, target_mask = self._tokenize_targets(targets)
        target_ids = target_ids.to(features.device)
        target_mask = target_mask.to(features.device)

        inputs_embeds, attn_mask, labels = self._build_sample_inputs(projected, mask, target_ids, target_mask)

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                          labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, features: torch.Tensor, mask: torch.Tensor,
                 max_new_tokens: int = 128, **gen_kwargs):
        projected, _ = self.projector(features.float(), mask)
        projected = projected.to(dtype=self._llm_dtype)
        inputs_embeds, attn_mask, _ = self._build_sample_inputs(projected, mask)

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        cleaned_predictions = []
        for prediction in predictions:
            if prediction.startswith(PROMPT_TEMPLATE):
                prediction = prediction[len(PROMPT_TEMPLATE):]
            cleaned_predictions.append(prediction.strip())
        return cleaned_predictions
