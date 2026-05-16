"""Tests for visual-only LLM baseline module."""
from types import SimpleNamespace

import torch
from transformers import AutoModelForCausalLM, AutoConfig


def _make_fake_llm(tmp_path, hidden_size=64, vocab_size=1000):
    """Create a tiny fake LLM on disk for testing."""
    from transformers import LlamaTokenizerFast
    from tokenizers import Tokenizer, models

    model_dir = tmp_path / "tiny-llm"
    model_dir.mkdir()

    tokenizer_obj = Tokenizer(models.BPE())
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    tokenizer_obj.add_special_tokens(special_tokens)
    tokenizer_obj.enable_padding(pad_token="<pad>")

    tokenizer = LlamaTokenizerFast(tokenizer_object=tokenizer_obj)
    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.unk_token = "<unk>"
    tokenizer.pad_token = "<pad>"
    tokenizer.save_pretrained(str(model_dir))

    config = AutoConfig.for_model("llama",
        hidden_size=hidden_size,
        intermediate_size=hidden_size * 4,
        num_attention_heads=4,
        num_hidden_layers=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        vocab_size=vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    config.save_pretrained(str(model_dir))

    model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32)
    model.save_pretrained(str(model_dir))

    return str(model_dir)


class TestVisualLLMBaseline:
    def test_hidden_size_inferred_from_config(self, tmp_path):
        from src.models.visual_llm_baseline import VisualLLMBaseline

        llm_path = _make_fake_llm(tmp_path, hidden_size=64)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)
        assert model.llm_hidden_size == 64

    def test_hidden_size_is_not_user_provided(self, tmp_path):
        from src.models.visual_llm_baseline import VisualLLMBaseline

        llm_path = _make_fake_llm(tmp_path, hidden_size=128)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)
        assert model.llm_hidden_size == 128
        assert not hasattr(model, "embedding_size")

    def test_lora_wrapping_only_when_enabled(self, tmp_path):
        from src.models.visual_llm_baseline import VisualLLMBaseline

        llm_path = _make_fake_llm(tmp_path, hidden_size=64)

        model_no_lora = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)
        model_lora = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=True)

        assert model_no_lora.llm_hidden_size == 64
        assert model_lora.llm_hidden_size == 64

    def test_forward_produces_loss(self, tmp_path):
        from src.models.visual_llm_baseline import VisualLLMBaseline

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)

        features = torch.randn(2, 3, 768)
        mask = torch.ones(2, 3, dtype=torch.bool)
        targets = ["hello world", "test target"]

        loss = model.forward(features, mask, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_generate_returns_strings(self, tmp_path):
        from src.models.visual_llm_baseline import VisualLLMBaseline

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)
        model.eval()

        features = torch.randn(1, 5, 768)
        mask = torch.ones(1, 5, dtype=torch.bool)

        predictions = model.generate(features, mask, max_new_tokens=10)
        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], str)

    def test_masked_padded_positions_do_not_change_loss(self, tmp_path):
        from src.models.visual_llm_baseline import VisualLLMBaseline

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)

        torch.manual_seed(0)
        features = torch.randn(2, 3, 768)
        targets = ["hello world", "test target"]

        mask = torch.ones(2, 3, dtype=torch.bool)
        mask[1, 1:] = False
        loss_full = model.forward(features, mask, targets)

        features_with_changed_padding = features.clone()
        features_with_changed_padding[1, 1:] = torch.randn(2, 768)
        loss_masked = model.forward(features_with_changed_padding, mask, targets)

        assert not torch.isnan(loss_full)
        assert not torch.isnan(loss_masked)
        assert torch.allclose(loss_full, loss_masked, atol=1e-2), \
            f"Loss changed despite masking padded positions: {loss_full.item()} vs {loss_masked.item()}"

    def test_short_sample_loss_matches_single_vs_batched(self, tmp_path, monkeypatch):
        from src.models.visual_llm_baseline import VisualLLMBaseline

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)

        def fake_forward(*args, **kwargs):
            labels = kwargs["labels"]
            first_sample_target_positions = (labels[0] != -100).nonzero(as_tuple=False).flatten()
            return SimpleNamespace(loss=first_sample_target_positions.float().sum())

        monkeypatch.setattr(model.llm, "forward", fake_forward)

        torch.manual_seed(0)
        short_features = torch.randn(1, 2, 768)
        short_mask = torch.tensor([[True, True]])
        short_targets = ["hello world"]

        single_loss = model.forward(short_features, short_mask, short_targets)

        longer_features = torch.randn(1, 5, 768)
        batched_features = torch.nn.utils.rnn.pad_sequence(
            [short_features[0], longer_features[0]],
            batch_first=True,
        )
        batched_mask = torch.tensor([
            [True, True, False, False, False],
            [True, True, True, True, True],
        ])
        batched_targets = ["hello world", "longer target example"]

        batched_loss = model.forward(batched_features, batched_mask, batched_targets)

        assert torch.equal(single_loss, batched_loss)

    def test_target_padding_positions_are_ignored_in_labels(self, tmp_path, monkeypatch):
        from src.models.visual_llm_baseline import VisualLLMBaseline, PROMPT_TEMPLATE

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)

        captured = {}

        original_forward = model.llm.forward

        def fake_forward(*args, **kwargs):
            captured["labels"] = kwargs["labels"].detach().cpu()
            return original_forward(*args, **kwargs)

        monkeypatch.setattr(model.llm, "forward", fake_forward)

        features = torch.randn(2, 3, 768)
        mask = torch.ones(2, 3, dtype=torch.bool)
        targets = ["short", "this is a much longer target"]

        target_ids, target_mask = model._tokenize_targets(targets)
        prompt_ids = model.tokenizer(
            [PROMPT_TEMPLATE] * 2,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids

        model.forward(features, mask, targets)

        labels = captured["labels"]
        prompt_len = prompt_ids.shape[1]
        projected_len = features.shape[1]
        target_start = prompt_len + projected_len

        for row_idx in range(target_mask.shape[0]):
            for tok_idx in range(target_mask.shape[1]):
                label_value = labels[row_idx, target_start + tok_idx].item()
                if target_mask[row_idx, tok_idx].item() == 0:
                    assert label_value == -100

    def test_generate_strips_prompt_text(self, tmp_path, monkeypatch):
        from src.models.visual_llm_baseline import VisualLLMBaseline, PROMPT_TEMPLATE

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VisualLLMBaseline(pretrained_llm=llm_path, use_lora=False)
        model.eval()

        def fake_generate(*args, **kwargs):
            return torch.tensor([[1, 2, 3]])

        def fake_batch_decode(outputs, skip_special_tokens=True):
            return [f"{PROMPT_TEMPLATE} hello world"]

        monkeypatch.setattr(model.llm, "generate", fake_generate)
        monkeypatch.setattr(model.tokenizer, "batch_decode", fake_batch_decode)

        features = torch.randn(1, 2, 768)
        mask = torch.ones(1, 2, dtype=torch.bool)

        predictions = model.generate(features, mask, max_new_tokens=4)
        assert predictions == ["hello world"]
