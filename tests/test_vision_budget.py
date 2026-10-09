"""CPU contract tests; synthetic tensors are NOT model-quality benchmarks."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from model.model_vlm import (
    MMVisionProjector, MiniMindVLM, VLMConfig, fuse_visual_tokens, pool_visual_tokens,
)


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 16)
        self.requires_grad_(False)
        self.seen_images = 0

    def forward(self, pixel_values):
        self.seen_images += pixel_values.shape[0]
        features = self.proj(pixel_values.mean((-2, -1)))
        positions = torch.arange(64, device=features.device, dtype=features.dtype)[None, :, None] / 64
        return SimpleNamespace(last_hidden_state=features[:, None, :] + positions)


def tiny_model(monkeypatch, budget=4, vocab_size=128):
    torch.manual_seed(7)
    vision = TinyVision()
    monkeypatch.setattr(MiniMindVLM, 'get_vision_model', staticmethod(lambda _: (vision.eval(), None)))
    config = VLMConfig(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                       num_key_value_heads=2, vocab_size=vocab_size, image_hidden_size=16,
                       image_token_len=budget, max_position_embeddings=256)
    return MiniMindVLM(config).eval()


def test_default_projector_is_checkpoint_compatible():
    projector = MMVisionProjector(16, 32)
    features = torch.randn(2, 64, 16)
    assert pool_visual_tokens(features, 64) is features
    torch.testing.assert_close(projector(features), projector.mlp(features), rtol=0, atol=0)
    assert list(projector.state_dict()) == [
        'mlp.0.weight', 'mlp.0.bias', 'mlp.1.weight', 'mlp.1.bias', 'mlp.3.weight', 'mlp.3.bias',
    ]


@pytest.mark.parametrize('budget', [4, 16, 64])
def test_budget_shape_and_gradient(budget):
    features = torch.randn(2, 64, 16, requires_grad=True)
    output = MMVisionProjector(16, 32, target_tokens=budget)(features)
    assert output.shape == (2, budget, 32)
    output.square().mean().backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_spatial_pooling_uses_grid_not_flat_chunks():
    features = torch.arange(16.0).reshape(1, 16, 1)
    torch.testing.assert_close(pool_visual_tokens(features, 4).flatten(), torch.tensor([2.5, 4.5, 10.5, 12.5]))
    with pytest.raises(ValueError):
        VLMConfig(image_token_len=32)


def test_adjacent_images_preserve_length_order_and_text():
    tokens = torch.tensor([[5, 12, 12, 12, 12, 12, 12, 12, 12, 6]])
    text = torch.randn(1, 10, 3, requires_grad=True)
    vision = torch.arange(24.0).reshape(1, 2, 4, 3).requires_grad_()
    result = fuse_visual_tokens(tokens, text, vision, 12)
    assert result.shape == text.shape
    torch.testing.assert_close(result[0, 1:9], vision[0].flatten(0, 1))
    torch.testing.assert_close(result[:, [0, 9]], text[:, [0, 9]])
    result.sum().backward()
    assert torch.all(vision.grad == 1)
    assert torch.all(text.grad[0, 1:9] == 0)


def test_misaligned_or_truncated_markers_fail_loudly():
    with pytest.raises(ValueError, match='image markers'):
        fuse_visual_tokens(torch.tensor([[12, 12, 12]]), torch.zeros(1, 3, 2), torch.zeros(1, 1, 4, 2), 12)


def test_variable_image_count_masks_skip_encoder_work(monkeypatch):
    model = tiny_model(monkeypatch)
    tokens = torch.tensor([[12] * 8 + [3], [12] * 4 + [3] * 5, [3] * 9])
    pixels = torch.randn(3, 2, 3, 8, 8)
    mask = torch.tensor([[1, 1], [1, 0], [0, 0]], dtype=torch.bool)
    output = model(tokens, pixel_values={'pixel_values': pixels, 'image_mask': mask})
    assert output.logits.shape == (3, 9, 128)
    assert model.vision_encoder.seen_images == 3
    assert torch.isfinite(output.logits).all()


def test_raw_dict_legacy_pixels_and_cached_features_agree(monkeypatch):
    model = tiny_model(monkeypatch)
    tokens = torch.tensor([[12] * 4 + [3, 4]])
    pixels = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        expected = model(tokens, pixel_values=pixels).logits
        for value in ({'pixel_values': pixels}, pixels[:, None], pixels[:, None, None]):
            torch.testing.assert_close(model(tokens, pixel_values=value).logits, expected)
        features = model.encode_images(pixels)
        calls = model.vision_encoder.seen_images
        torch.testing.assert_close(model(tokens, image_embeddings=features).logits, expected)
        assert model.vision_encoder.seen_images == calls


def test_kv_cache_matches_full_forward_and_encodes_once(monkeypatch):
    model = tiny_model(monkeypatch)
    tokens = torch.tensor([[12] * 4 + [3]])
    pixels = torch.randn(1, 3, 8, 8)
    with torch.no_grad():
        first = model(tokens, pixel_values=pixels, use_cache=True)
        second = model(torch.tensor([[4]]), pixel_values=pixels, past_key_values=first.past_key_values)
        assert model.vision_encoder.seen_images == 1
        full = model(torch.cat((tokens, torch.tensor([[4]])), -1), pixel_values=pixels)
    torch.testing.assert_close(second.logits[:, -1], full.logits[:, -1], rtol=1e-4, atol=1e-5)


def test_text_only_batch_skips_dummy_vision_and_trains_projector(monkeypatch):
    model = tiny_model(monkeypatch).train()
    assert not model.vision_encoder.training
    tokens = torch.tensor([[3, 4, 5]])
    output = model(tokens, labels=tokens, pixel_values=torch.zeros(1, 1, 3, 8, 8))
    (output.loss + output.aux_loss).backward()
    assert model.vision_encoder.seen_images == 0
    assert all(p.grad is None for p in model.vision_encoder.parameters())
    assert all(p.grad is not None for p in model.vision_proj.parameters())


def test_supervision_reaches_projector_not_frozen_encoder(monkeypatch):
    model = tiny_model(monkeypatch).train()
    tokens = torch.tensor([[12] * 4 + [3, 4]])
    labels = torch.tensor([[-100] * 4 + [3, 4]])
    output = model(tokens, labels=labels, pixel_values=torch.randn(1, 3, 8, 8))
    (output.loss + output.aux_loss).backward()
    assert any(p.grad.abs().sum() > 0 for p in model.vision_proj.parameters())
    assert all(p.grad is None for p in model.vision_encoder.parameters())


def test_cached_generation_return_sequence_order(monkeypatch):
    model = tiny_model(monkeypatch)
    tokens = torch.tensor([[12] * 4 + [3], [12] * 4 + [4]])
    with torch.no_grad():
        features = model.encode_images(torch.randn(2, 3, 8, 8))
        single = model.generate(tokens, image_embeddings=features, max_new_tokens=2,
                                do_sample=False, top_k=0, top_p=1.0, eos_token_id=None)
        repeated = model.generate(tokens, image_embeddings=features, max_new_tokens=2,
                                  num_return_sequences=2, do_sample=False, top_k=0,
                                  top_p=1.0, eos_token_id=None)
    torch.testing.assert_close(repeated, single.repeat(2, 1))


@pytest.mark.parametrize('condition', ['actual', 'blank', 'shuffle'])
def test_evaluation_cli_pipeline_with_tiny_random_model(monkeypatch, tmp_path, condition):
    """Exercise reporting, images, the real tokenizer and generation; NOT quality."""
    import json
    from pathlib import Path
    from PIL import Image
    from transformers import AutoTokenizer, SiglipImageProcessor
    from scripts import eval_budget_vlm

    model = tiny_model(monkeypatch, vocab_size=6400)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parents[1] / 'model'))
    processor = SiglipImageProcessor(size={'height': 8, 'width': 8})
    monkeypatch.setattr(eval_budget_vlm, 'init_model', lambda _: (model, tokenizer, processor))
    for name, color in [('a.png', 'red'), ('b.png', 'blue')]:
        Image.new('RGB', (8, 8), color).save(tmp_path / name)
    rows = [{'id': name, 'image': name, 'question': 'What color?', 'answers': [answer]}
            for name, answer in [('a.png', 'red'), ('b.png', 'blue')]]
    manifest = tmp_path / 'eval.jsonl'
    manifest.write_text('\n'.join(json.dumps(row) for row in rows), encoding='utf-8')
    output = tmp_path / 'predictions.jsonl'
    monkeypatch.setattr('sys.argv', ['eval_budget_vlm', '--manifest', str(manifest),
                                    '--output', str(output), '--device', 'cpu', '--max_new_tokens', '2',
                                    '--image_token_len', '4', '--image_condition', condition])
    eval_budget_vlm.main()
    predictions = [json.loads(line) for line in output.read_text().splitlines()]
    summary = json.loads(output.with_suffix('.summary.json').read_text())
    assert summary['overall']['samples'] == 2
    assert summary['metadata']['cuda_peak_allocated_bytes'] is None
    assert all(row['generation_ms'] >= row['ttft_ms'] > 0 for row in predictions)
    assert all(row['image_tokens'] == 4 for row in predictions)
    if condition == 'shuffle':
        assert Path(predictions[0]['image_paths_used'][0]).name == 'b.png'
    assert all(row['request_ms'] >= row['generation_ms'] for row in predictions)
