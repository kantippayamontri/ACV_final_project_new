"""Tests for visual projector."""
import torch


class TestVisualProjector:
    def test_output_shape_matches_input_frames(self):
        from src.models.projector import VisualProjector

        proj = VisualProjector(input_dim=768, output_dim=4096, num_layers=3)
        x = torch.randn(4, 10, 768)
        out = proj(x)
        assert out.shape == (4, 10, 4096)

    def test_output_dim_adapts_to_llm_hidden_size(self):
        from src.models.projector import VisualProjector

        proj_llama = VisualProjector(input_dim=768, output_dim=4096, num_layers=3)
        proj_qwen = VisualProjector(input_dim=768, output_dim=3584, num_layers=3)

        x = torch.randn(2, 5, 768)
        assert proj_llama(x).shape == (2, 5, 4096)
        assert proj_qwen(x).shape == (2, 5, 3584)

    def test_does_not_modify_attention_mask(self):
        from src.models.projector import VisualProjector

        proj = VisualProjector(input_dim=768, output_dim=4096, num_layers=3)
        x = torch.randn(3, 7, 768)
        mask = torch.tensor([[True, True, True, True, False, False, False],
                             [True, True, True, True, True, True, True],
                             [True, False, False, False, False, False, False]])

        out, out_mask = proj(x, mask)
        assert torch.equal(out_mask, mask)
        assert out.shape == (3, 7, 4096)

    def test_single_frame(self):
        from src.models.projector import VisualProjector

        proj = VisualProjector(input_dim=768, output_dim=4096, num_layers=3)
        x = torch.randn(1, 1, 768)
        out = proj(x)
        assert out.shape == (1, 1, 4096)

    def test_empty_sequence(self):
        from src.models.projector import VisualProjector

        proj = VisualProjector(input_dim=768, output_dim=4096, num_layers=3)
        x = torch.randn(1, 0, 768)
        out = proj(x)
        assert out.shape == (1, 0, 4096)
