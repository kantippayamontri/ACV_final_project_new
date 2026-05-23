"""Tests for video + previous-sentence LLM module."""
import torch


def _make_fake_llm(tmp_path, hidden_size=64, vocab_size=1000):
    from transformers import LlamaTokenizerFast, AutoModelForCausalLM, AutoConfig
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


class TestVideoPrevLLM:
    def test_hidden_size_inferred_from_config(self, tmp_path):
        from src.models.visual_prev_llm import VideoPrevLLM

        llm_path = _make_fake_llm(tmp_path, hidden_size=64)
        model = VideoPrevLLM(pretrained_llm=llm_path, use_lora=False)
        assert model.llm_hidden_size == 64

    def test_forward_produces_loss(self, tmp_path):
        from src.models.visual_prev_llm import VideoPrevLLM

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VideoPrevLLM(pretrained_llm=llm_path, use_lora=False)

        features = torch.randn(2, 3, 768)
        mask = torch.ones(2, 3, dtype=torch.bool)
        prev_texts = ["previous one", "previous two"]
        targets = ["hello world", "test target"]

        loss = model.forward(features, mask, prev_texts, targets)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_forward_handles_none_prev_text(self, tmp_path):
        from src.models.visual_prev_llm import VideoPrevLLM

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VideoPrevLLM(pretrained_llm=llm_path, use_lora=False)

        features = torch.randn(1, 2, 768)
        mask = torch.ones(1, 2, dtype=torch.bool)
        prev_texts = [""]
        targets = ["test"]

        loss = model.forward(features, mask, prev_texts, targets)
        assert not torch.isnan(loss)

    def test_generate_returns_strings(self, tmp_path):
        from src.models.visual_prev_llm import VideoPrevLLM

        llm_path = _make_fake_llm(tmp_path, hidden_size=64, vocab_size=200)
        model = VideoPrevLLM(pretrained_llm=llm_path, use_lora=False)
        model.eval()

        features = torch.randn(1, 5, 768)
        mask = torch.ones(1, 5, dtype=torch.bool)
        prev_texts = ["previous context"]

        predictions = model.generate(features, mask, prev_texts, max_new_tokens=10)
        assert isinstance(predictions, list)
        assert len(predictions) == 1
        assert isinstance(predictions[0], str)
