"""Training correctness tests with random tiny models, NOT downstream scores."""
import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from test_vision_budget import tiny_model
from trainer.budget_training import (BudgetCollator, accumulation_windows, answer_kl_sum,
                                     backward_window, compress_image_markers)


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def examples():
    torch.manual_seed(99)
    result = []
    for n in (1, 3, 2):
        ids = torch.tensor([12] * 64 + [3] + list(range(4, 4 + n)))
        labels = torch.tensor([-100] * 65 + list(range(4, 4 + n)))
        pixels = {'pixel_values': torch.randn(1, 3, 8, 8), 'image_mask': torch.ones(1, dtype=torch.bool)}
        result.append((ids, labels, pixels))
    return result


def test_compress_adjacent_images_preserves_all_text_and_labels():
    ids = torch.tensor([3] + [12] * 128 + [4, 5])
    labels = torch.tensor([-100] * 129 + [4, 5])
    actual, target = compress_image_markers(ids, labels, 64, 16)
    assert actual.tolist() == [3] + [12] * 32 + [4, 5]
    assert target.tolist() == [-100] * 33 + [4, 5]
    with pytest.raises(ValueError, match='Incomplete'):
        compress_image_markers(ids[:-5], labels[:-5], 64, 16)


def test_dynamic_padding_uses_lengths_not_pad_token_identity():
    collator = BudgetCollator(4, student_budget=16)
    batch = collator(examples())
    assert batch['teacher']['input_ids'].shape == (3, 68)
    assert batch['student']['input_ids'].shape == (3, 20)
    # Token 4 is a real supervised token even though it also serves as pad ID here.
    assert batch['student']['attention_mask'][0, 17] == 1
    assert batch['student']['labels'][0, 17] == 4
    assert batch['student']['attention_mask'][0, 18] == 0
    assert batch['student']['labels'][0, 18] == -100
    assert batch['student']['pixel_values'] is batch['teacher']['pixel_values']
    fixed = BudgetCollator(0, student_budget=16, fixed_length=128)(examples())
    assert fixed['student']['input_ids'].shape == fixed['teacher']['input_ids'].shape == (3, 128)


def test_kl_aligns_next_token_answer_positions_not_absolute_offsets():
    sl = torch.tensor([[-100] * 5 + [3, 4]])
    tl = torch.tensor([[-100] * 8 + [3, 4]])
    s = torch.randn(1, 7, 8, requires_grad=True)
    t = torch.randn(1, 10, 8, requires_grad=True)
    with torch.no_grad():
        t[0, 7:9] = s[0, 4:6]
    loss, _, count = answer_kl_sum(s, t, sl, tl)
    assert count == 2
    assert abs(loss.item()) < 1e-5
    loss.backward()
    assert t.grad is None
    assert torch.all(s.grad[0, :4] == 0)
    tl[0, -1] = 5
    with pytest.raises(ValueError, match='do not align'):
        answer_kl_sum(s, t, sl, tl)


@pytest.mark.parametrize('temperature', [0, -1, float('nan')])
def test_invalid_distillation_temperature_rejected(temperature):
    with pytest.raises(ValueError, match='temperature'):
        answer_kl_sum(torch.zeros(1, 2, 8), torch.zeros(1, 2, 8),
                      torch.tensor([[-100, 3]]), torch.tensor([[-100, 3]]), temperature)


@pytest.mark.parametrize('distill', [False, True])
def test_unequal_microbatches_match_full_token_mean(monkeypatch, distill):
    model = tiny_model(monkeypatch).train()
    reference = copy.deepcopy(model)
    teacher = tiny_model(monkeypatch, budget=64).requires_grad_(False) if distill else None
    collator = BudgetCollator(0, student_budget=4)
    micro = [collator([sample]) for sample in examples()]
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    # Configured 4 microbatches, but only 3 remain: no under-scaling of the tail.
    windows = list(accumulation_windows(micro, 4))
    assert [len(window) for window in windows] == [3]
    metrics = backward_window(model, windows[0], scaler, 'cpu', teacher=teacher)
    full_metrics = backward_window(reference, [collator(examples())], scaler, 'cpu', teacher=teacher)
    assert metrics['supervised_tokens'] == 6
    assert metrics['loss'] == pytest.approx(full_metrics['loss'], rel=1e-5)
    for (name, parameter), other in zip(model.named_parameters(), reference.parameters()):
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=2e-4, atol=2e-5, msg=name)
    assert model.vision_encoder.seen_images == 3
    if teacher:
        assert teacher.vision_encoder.seen_images == 0
        assert all(p.grad is None for p in teacher.parameters())


def test_zero_kd_weight_does_not_call_teacher(monkeypatch):
    model = tiny_model(monkeypatch).train()
    teacher = tiny_model(monkeypatch, budget=64)
    monkeypatch.setattr(teacher, 'forward', lambda **_: pytest.fail('unused teacher called'))
    result = backward_window(model, [BudgetCollator(0, 4)(examples())],
                             torch.amp.GradScaler('cuda', enabled=False), 'cpu', teacher=teacher, kd_weight=0)
    assert result['kd'] == 0
    assert result['teacher_next_token_accuracy'] is None


def test_shared_raw_features_match_direct_pixels_and_backprop(monkeypatch):
    model = tiny_model(monkeypatch).train()
    batch = BudgetCollator(0, 4)(examples())['student']
    raw, mask = model.encode_image_features(batch['pixel_values'])
    assert not raw.requires_grad
    direct = model(**batch).logits
    features = {k: v for k, v in batch.items() if k != 'pixel_values'}
    output = model(**features, vision_features=raw, image_mask=mask)
    torch.testing.assert_close(output.logits, direct)
    output.loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vision_proj.parameters())


def test_tiny_training_reduces_training_objective_without_updating_teacher(monkeypatch):
    model = tiny_model(monkeypatch).train()
    teacher = tiny_model(monkeypatch, budget=64).requires_grad_(False)
    snapshot = {k: v.clone() for k, v in teacher.state_dict().items()}
    batch = BudgetCollator(0, 4)(examples())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler('cuda', enabled=False)
    history = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        metrics = backward_window(model, [batch], scaler, 'cpu', teacher=teacher)
        history.append(metrics['loss'])
        optimizer.step()
    assert history[-1] < history[0]
    for k, value in teacher.state_dict().items():
        torch.testing.assert_close(value, snapshot[k], rtol=0, atol=0)


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(20, 8)
        self.head = torch.nn.Linear(8, 20)

    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)), aux_loss=None)


def ddp_worker(rank, rendezvous, output):
    import torch.distributed as dist
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    torch.manual_seed(77)
    model = torch.nn.parallel.DistributedDataParallel(ToyLM())
    lengths = [1, 2] if rank == 0 else [4, 3]
    window = [{'student': {'input_ids': torch.tensor([[3] + [4] * n]),
                           'labels': torch.tensor([[-100] + [4] * n])}} for n in lengths]
    metrics = backward_window(model, window, torch.amp.GradScaler('cuda', enabled=False), 'cpu')
    if rank == 0:
        torch.save({'gradients': [p.grad for p in model.parameters()], 'tokens': metrics['supervised_tokens']}, output)
    dist.destroy_process_group()


def test_two_rank_ddp_matches_global_token_mean(tmp_path):
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip('Gloo is not available')
    output = tmp_path / 'gradients.pt'
    torch.multiprocessing.spawn(ddp_worker, args=((tmp_path / 'rendezvous').as_uri(), str(output)),
                                nprocs=2, join=True)
    result = torch.load(output, weights_only=True)
    torch.manual_seed(77)
    model = ToyLM()
    loss = torch.zeros(())
    for n in [1, 2, 4, 3]:
        logits = model(torch.tensor([[3] + [4] * n])).logits[:, :-1]
        loss = loss + F.cross_entropy(logits.reshape(-1, 20), torch.tensor([4] * n), reduction='sum')
    (loss / 10).backward()
    assert result['tokens'] == 10
    for actual, parameter in zip(result['gradients'], model.parameters()):
        torch.testing.assert_close(actual, parameter.grad, rtol=1e-5, atol=1e-6)
