"""Budget-controlled, answer-aligned OFFLINE distillation and token-mean training.

This is a small CE + forward-KL baseline, not LT-OPD/SCOPD or full TBD.
Teacher and student see the same text; only image-marker spans are shortened.
"""
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from trainer.trainer_utils import vlm_collate_fn


def compress_image_markers(input_ids, labels, source_budget, target_budget, marker=12):
    """Keep identical text/labels, replacing each full image span by a shorter span."""
    if source_budget not in (4, 16, 64) or target_budget not in (4, 16, 64) or target_budget > source_budget:
        raise ValueError('Expected budgets in {4,16,64} with target <= source')
    if input_ids.ndim != 1 or labels.shape != input_ids.shape:
        raise ValueError('Expected equally sized one-dimensional ids and labels')
    keep = torch.ones_like(input_ids, dtype=torch.bool)
    i = 0
    while i < input_ids.numel():
        if input_ids[i].item() != marker:
            i += 1
            continue
        start = i
        while i < input_ids.numel() and input_ids[i].item() == marker:
            i += 1
        if (i - start) % source_budget:
            raise ValueError('Incomplete image-marker span')
        if torch.any(labels[start:i] != -100):
            raise ValueError('Image markers must not be supervised answer tokens')
        for offset in range(start, i, source_budget):
            keep[offset + target_budget:offset + source_budget] = False
    return input_ids[keep], labels[keep]


@dataclass
class BudgetCollator:
    pad_token_id: int
    student_budget: int = 16
    source_budget: int = 64
    marker: int = 12
    fixed_length: int | None = None

    def _pad(self, samples):
        lengths = [sample[0].numel() for sample in samples]
        length = self.fixed_length or max(lengths)
        if length < max(lengths):
            raise ValueError('fixed_length is shorter than an input; collator never truncates')
        ids = torch.full((len(samples), length), self.pad_token_id, dtype=torch.long)
        labels = torch.full_like(ids, -100)
        mask = torch.zeros_like(ids)
        for i, (tokens, target, _) in enumerate(samples):
            ids[i, :lengths[i]], labels[i, :lengths[i]] = tokens, target
            mask[i, :lengths[i]] = 1  # Length-based, safe even when pad_token_id == EOS.
        return {'input_ids': ids, 'labels': labels, 'attention_mask': mask}

    def __call__(self, samples):
        if not samples:
            raise ValueError('Cannot collate an empty batch')
        compressed = [(*compress_image_markers(ids, labels, self.source_budget,
                                               self.student_budget, self.marker), pixels)
                      for ids, labels, pixels in samples]
        teacher, student = self._pad(samples), self._pad(compressed)
        # Stack/pad images once. Both branches share the same source image tensors.
        padded = [(teacher['input_ids'][i], teacher['labels'][i], sample[2])
                  for i, sample in enumerate(samples)]
        pixels = vlm_collate_fn(padded)[2]
        teacher['pixel_values'] = student['pixel_values'] = pixels
        return {'teacher': teacher, 'student': student}


def accumulation_windows(loader, steps):
    """Stage at most one accumulation window on CPU; no retained GPU graphs."""
    if steps < 1:
        raise ValueError('accumulation steps must be positive')
    iterator = iter(loader)
    while window := list(islice(iterator, steps)):
        yield window


def answer_kl_sum(student_logits, teacher_logits, student_labels, teacher_labels, temperature=2.0):
    """Forward KL(teacher || student) on aligned next-answer-token positions.

    Budgets shift absolute sequence positions. Align by sample and supervised
    token order, and reject mismatched labels rather than cropping to min length.
    The caller normalizes by ALL supervised tokens across the accumulation window.
    """
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    if student_logits.shape[0] != teacher_logits.shape[0] or student_logits.shape[-1] != teacher_logits.shape[-1]:
        raise ValueError('Teacher and student must share batch size and vocabulary')
    total = student_logits.sum() * 0
    correct, count = 0, 0
    for b in range(student_logits.shape[0]):
        sm, tm = student_labels[b, 1:] != -100, teacher_labels[b, 1:] != -100
        sy, ty = student_labels[b, 1:][sm], teacher_labels[b, 1:][tm]
        if not torch.equal(sy, ty):
            raise ValueError(f'Sample {b}: teacher/student answer tokens do not align')
        student = student_logits[b, :-1][sm]
        teacher = teacher_logits[b, :-1][tm].detach()
        correct += int((teacher.argmax(-1) == sy).sum())
        count += sy.numel()
        # Bound the float32 softmax workspace for longer answers.
        for start in range(0, sy.numel(), 256):
            s = F.log_softmax(student[start:start + 256].float() / temperature, dim=-1)
            t = F.log_softmax(teacher[start:start + 256].float() / temperature, dim=-1)
            total = total + F.kl_div(s, t, log_target=True, reduction='sum') * temperature ** 2
    return total, correct, count


def move_to_device(batch, device):
    return {key: move_to_device(value, device) if isinstance(value, dict)
            else value.to(device) for key, value in batch.items()}


def backward_window(model, window, scaler, device, teacher=None, kd_weight=1.0,
                    temperature=2.0, autocast_factory=nullcontext):
    """Accumulate a global valid-token mean, including short final windows.

    DDP averages gradients across ranks, so local token sums are multiplied by
    world_size/global_token_count. Only the final microbatch synchronizes gradients.
    MoE auxiliary terms are token-weighted; routing is NOT microbatch-invariant.
    This function does not clip, step, or clear gradients.
    """
    if not window or not math.isfinite(kd_weight) or kd_weight < 0:
        raise ValueError('Need a nonempty window and finite nonnegative kd_weight')
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('temperature must be finite and positive')
    counts = [int((batch['student']['labels'][:, 1:] != -100).sum()) for batch in window]
    denominator = torch.tensor(sum(counts), device=device, dtype=torch.long)
    distributed = isinstance(model, DistributedDataParallel)
    world = dist.get_world_size() if distributed else 1
    if distributed:
        dist.all_reduce(denominator)
    if denominator.item() == 0:
        raise ValueError('No supervised next tokens in the accumulation window')
    scale = world / denominator.item()
    raw = model.module if distributed else model
    use_teacher = teacher is not None and kd_weight > 0
    if use_teacher:
        teacher.eval()
    stats_dtype = torch.float32 if str(device).startswith('mps') else torch.float64
    totals = torch.zeros(6, device=device, dtype=stats_dtype)
    for index, (cpu_batch, tokens) in enumerate(zip(window, counts)):
        student = move_to_device(cpu_batch['student'], device)
        labels = student.pop('labels')
        context = model.no_sync() if distributed and index < len(window) - 1 else nullcontext()
        with context, autocast_factory():
            teacher_output = None
            if use_teacher:
                teacher_batch = {key: value.to(device) for key, value in cpu_batch['teacher'].items()
                                 if key != 'pixel_values'}
                teacher_labels = teacher_batch.pop('labels')
                # One frozen ViT call, two separate projectors. No stale projected cache.
                features, image_mask = raw.encode_image_features(student.pop('pixel_values'))
                student.update(vision_features=features, image_mask=image_mask)
                with torch.no_grad():
                    teacher_output = teacher(**teacher_batch, vision_features=features, image_mask=image_mask)
            output = model(**student)
            ce = F.cross_entropy(output.logits[:, :-1].reshape(-1, output.logits.shape[-1]).float(),
                                 labels[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
            kd, correct, teacher_tokens = output.logits.new_zeros(()), 0, 0
            if use_teacher:
                kd, correct, teacher_tokens = answer_kl_sum(output.logits, teacher_output.logits,
                                                           labels, teacher_labels, temperature)
            aux = output.aux_loss * tokens if output.aux_loss is not None else ce * 0
            loss = (ce + kd_weight * kd + aux) * scale
        scaler.scale(loss).backward()
        totals += torch.tensor([ce.detach().item(), kd.detach().item(), aux.detach().item(),
                                tokens, correct, teacher_tokens], device=device, dtype=stats_dtype)
        # Do not keep previous microbatch GPU activations while computing the next.
        del output, teacher_output, ce, kd, aux, loss, student, labels
    if distributed:
        dist.all_reduce(totals)
    ce, kd, aux, tokens, correct, teacher_tokens = totals.tolist()
    return {'ce': ce / tokens, 'kd': kd / tokens, 'aux': aux / tokens,
            'loss': (ce + kd_weight * kd + aux) / tokens, 'supervised_tokens': int(tokens),
            'teacher_next_token_accuracy': correct / teacher_tokens if teacher_tokens else None}
