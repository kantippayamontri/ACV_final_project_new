"""Video + previous-sentence LLM: projector + LoRA decoder with context."""
import torch
import torch.nn as nn
from src.models.visual_llm_baseline import VisualLLMBaseline


PREV_PROMPT_TEMPLATE = "Previous sentence: {prev}\n\nTranslate this sign language video into English. The following are the video tokens:"


class VideoPrevLLM(VisualLLMBaseline):
    """Extends VisualLLMBaseline with previous-sentence context.

    Prompt format (per-sample):
        "Previous sentence: {prev}\n\nTranslate this sign language video into
         English. The following are the video tokens:"

    Input structure: [prompt_embeds, visual_embeds, target_embeds]
    """

    def _build_sample_inputs(self, projected, mask, prev_texts,
                             target_ids=None, target_mask=None):
        embedding_layer = self.llm.get_input_embeddings()

        prompts = [PREV_PROMPT_TEMPLATE.format(prev=t if t else "(none)") for t in prev_texts]
        prompt_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True,
        ).input_ids.to(projected.device)

        sample_embeds = []
        sample_attn = []
        sample_labels = []

        for idx in range(projected.shape[0]):
            prompt_embeds = embedding_layer(prompt_ids[idx:idx + 1])[0]
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
                    -100, dtype=torch.long, device=projected.device,
                )
                labels[sample_len:] = valid_target_ids
                sample_labels.append(labels)

            sample_input = torch.cat(parts, dim=0)
            sample_embeds.append(sample_input)
            sample_attn.append(torch.ones(sample_input.shape[0], dtype=torch.long,
                                          device=projected.device))

        max_len = max(s.shape[0] for s in sample_embeds)
        hidden_size = prompt_embeds.shape[-1]
        inputs_embeds = projected.new_zeros((projected.shape[0], max_len, hidden_size))
        attention_mask = torch.zeros((projected.shape[0], max_len), dtype=torch.long,
                                     device=projected.device)

        labels = None
        if sample_labels:
            labels = torch.full((projected.shape[0], max_len), -100, dtype=torch.long,
                                device=projected.device)

        for idx, sample_input in enumerate(sample_embeds):
            seq_len = sample_input.shape[0]
            inputs_embeds[idx, :seq_len] = sample_input
            attention_mask[idx, :seq_len] = sample_attn[idx]
            if labels is not None:
                labels[idx, :sample_labels[idx].shape[0]] = sample_labels[idx]

        return inputs_embeds, attention_mask, labels

    def forward(self, features, mask, prev_texts, targets):
        projected, _ = self.projector(features.float(), mask)
        projected = projected.to(dtype=self._llm_dtype)
        target_ids, target_mask = self._tokenize_targets(targets)
        target_ids = target_ids.to(features.device)
        target_mask = target_mask.to(features.device)

        inputs_embeds, attn_mask, labels = self._build_sample_inputs(
            projected, mask, prev_texts, target_ids, target_mask,
        )

        outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attn_mask,
                          labels=labels)
        return outputs.loss

    @torch.no_grad()
    def generate(self, features, mask, prev_texts, max_new_tokens=128, **gen_kwargs):
        projected, _ = self.projector(features.float(), mask)
        projected = projected.to(dtype=self._llm_dtype)
        inputs_embeds, attn_mask, _ = self._build_sample_inputs(
            projected, mask, prev_texts,
        )

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_kwargs,
        )

        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return [p.strip() for p in predictions]
