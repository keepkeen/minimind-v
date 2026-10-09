"""Controlled dense MiniMind-V CE / offline budget-distillation training.

Run from any directory. Uses full-budget examples in BOTH objectives so shorter
student inputs do not silently receive additional answer text after truncation.
Checkpoint initialization is supported; exact interrupted-run resume is not.
"""
import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import transformers
from transformers import AutoTokenizer

from dataset.lm_dataset import VLMDataset
from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.budget_training import BudgetCollator, accumulation_windows, backward_window
from trainer.trainer_utils import init_distributed_mode, is_main_process, setup_seed


def load_nonvision_checkpoint(model, path, allow_missing_projector=False):
    """Reject incomplete teachers and architecture mismatches; never hide bad keys."""
    state = torch.load(path, map_location='cpu', weights_only=True)
    state = {key: value for key, value in state.items() if not key.startswith('vision_encoder.')}
    status = model.load_state_dict(state, strict=False)
    missing = [key for key in status.missing_keys if not key.startswith('vision_encoder.')
               and not (allow_missing_projector and key.startswith('vision_proj.'))]
    if missing or status.unexpected_keys:
        raise ValueError(f'Checkpoint mismatch: missing={missing}, unexpected={status.unexpected_keys}')
    return status


def set_trainable(model, freeze_llm):
    model.requires_grad_(False)
    model.vision_proj.requires_grad_(True)
    if freeze_llm == 0:
        model.model.requires_grad_(True)
        model.lm_head.requires_grad_(True)
    elif freeze_llm == 1:
        model.model.layers[0].requires_grad_(True)
        model.model.layers[-1].requires_grad_(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_path', type=Path, required=True)
    parser.add_argument('--init_checkpoint', type=Path, required=True)
    parser.add_argument('--init_kind', choices=['vlm', 'llm'], default='vlm')
    parser.add_argument('--teacher_checkpoint', type=Path)
    parser.add_argument('--run_dir', type=Path, required=True)
    parser.add_argument('--vision_model_path', type=Path, default=ROOT / 'model/siglip2-base-p32-256-ve')
    parser.add_argument('--tokenizer_path', type=Path, default=ROOT / 'model')
    parser.add_argument('--image_token_len', type=int, choices=[4, 16, 64], default=16)
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--image_hidden_size', type=int, default=768)
    parser.add_argument('--max_seq_len', type=int, default=768, help='Truncate full 64-token source BEFORE compression')
    parser.add_argument('--padding', choices=['dynamic', 'fixed'], default='dynamic')
    parser.add_argument('--freeze_llm', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--max_updates', type=int, default=0, help='0 means all planned updates')
    parser.add_argument('--learning_rate', type=float, default=5e-6)
    parser.add_argument('--warmup_updates', type=int, default=0)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--kd_weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', choices=['float32', 'bfloat16', 'float16'], default='float32')
    args = parser.parse_args()
    for name in ('batch_size', 'accumulation_steps', 'epochs', 'max_seq_len'):
        if getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if args.max_updates < 0 or args.num_workers < 0 or args.warmup_updates < 0:
        parser.error('max_updates, num_workers and warmup_updates must be nonnegative')
    if not math.isfinite(args.kd_weight) or args.kd_weight < 0:
        parser.error('kd_weight must be finite and nonnegative')
    for name in ('temperature', 'learning_rate', 'grad_clip'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    if args.dtype != 'float32' and not args.device.startswith('cuda'):
        parser.error('This entry supports mixed precision only on CUDA; use float32 on CPU/MPS')
    # Only rank zero checks/creates the directory, avoiding a torchrun mkdir race.
    rank = int(os.environ.get('RANK', '0'))
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        args.device = f'cuda:{local_rank}'
    exists = torch.tensor(int(args.run_dir.exists()) if rank == 0 else 0, device=args.device)
    if dist.is_initialized():
        dist.broadcast(exists, src=0)
    if exists.item():
        parser.error('run_dir exists; use a new directory to preserve previous runs')
    setup_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                       image_hidden_size=args.image_hidden_size, image_token_len=args.image_token_len)
    if args.max_seq_len > config.max_position_embeddings:
        parser.error('max_seq_len exceeds positional capacity')
    if tokenizer(config.image_special_token, add_special_tokens=False).input_ids != config.image_ids:
        parser.error('Tokenizer image marker does not match model config')
    model = MiniMindVLM(config, vision_model_path=str(args.vision_model_path))
    if model.vision_encoder is None or model.processor is None:
        raise RuntimeError('Could not load the configured vision encoder and processor')
    load_nonvision_checkpoint(model, args.init_checkpoint, args.init_kind == 'llm')
    set_trainable(model, args.freeze_llm)
    model.to(args.device).train()
    teacher = None
    if args.teacher_checkpoint is not None and args.kd_weight > 0:
        teacher_config = VLMConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                                   image_hidden_size=args.image_hidden_size, image_token_len=64)
        sidecar = args.teacher_checkpoint.with_suffix('.config.json')
        if sidecar.exists() and json.loads(sidecar.read_text())['image_token_len'] != 64:
            raise ValueError('Teacher checkpoint must be trained at the full 64-token budget')
        # No second ViT: the frozen raw features are computed once by the student.
        teacher = MiniMindVLM(teacher_config, vision_model_path='')
        load_nonvision_checkpoint(teacher, args.teacher_checkpoint)
        teacher.requires_grad_(False).to(args.device).eval()
    dataset = VLMDataset(str(args.data_path), tokenizer, preprocess=model.processor,
                         max_length=args.max_seq_len, image_token_len=64, pad_to_max_length=False)
    if len(dataset) == 0:
        raise ValueError('Training dataset is empty')
    sampler = (DistributedSampler(dataset, seed=args.seed) if dist.is_initialized() else
               RandomSampler(dataset, generator=torch.Generator().manual_seed(args.seed)))
    collate = BudgetCollator(tokenizer.pad_token_id, args.image_token_len,
                            fixed_length=args.max_seq_len if args.padding == 'fixed' else None)
    loader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=args.device.startswith('cuda'), collate_fn=collate,
                        generator=torch.Generator().manual_seed(args.seed + rank))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.learning_rate)
    scaler = torch.amp.GradScaler('cuda', enabled=args.dtype == 'float16')
    autocast = partial(torch.autocast, device_type='cuda', dtype=getattr(torch, args.dtype)) if args.dtype != 'float32' else nullcontext
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False)
    planned = math.ceil(len(loader) / args.accumulation_steps) * args.epochs
    total_updates = min(planned, args.max_updates) if args.max_updates else planned
    raw = model.module if isinstance(model, DistributedDataParallel) else model
    if is_main_process():
        args.run_dir.mkdir(parents=True, exist_ok=False)
        try:
            commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            dirty = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True).strip())
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = None, None
        metadata = {'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    'objective': 'offline_teacher_forced_ce_plus_forward_kl' if teacher else 'ce',
                    'source_image_budget': 64, 'teacher_active': teacher is not None,
                    'python': platform.python_version(), 'torch': torch.__version__,
                    'transformers': transformers.__version__, 'git_commit': commit, 'dirty_worktree': dirty,
                    'trainable_parameters': sum(p.numel() for p in raw.parameters() if p.requires_grad),
                    'world_size': dist.get_world_size() if dist.is_initialized() else 1,
                    'planned_updates': total_updates, 'resume_supported': False}
        (args.run_dir / 'run.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    # Constructing an optional teacher must not change data order or dropout RNG.
    setup_seed(args.seed + rank)
    optimizer.zero_grad(set_to_none=True)
    update = 0
    for epoch in range(args.epochs):
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)
        for window in accumulation_windows(loader, args.accumulation_steps):
            start = time.perf_counter()
            update += 1
            progress = max(0, update - args.warmup_updates - 1) / max(1, total_updates - args.warmup_updates - 1)
            lr = args.learning_rate * (0.1 + 0.45 * (1 + math.cos(math.pi * progress)))
            if update <= args.warmup_updates:
                lr = args.learning_rate * update / args.warmup_updates
            for group in optimizer.param_groups:
                group['lr'] = lr
            metrics = backward_window(model, window, scaler, args.device, teacher=teacher,
                                      kd_weight=args.kd_weight, temperature=args.temperature,
                                      autocast_factory=autocast)
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            if not torch.isfinite(norm):
                raise FloatingPointError('Non-finite gradients; aborting instead of reporting a successful update')
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if args.device.startswith('cuda'):
                torch.cuda.synchronize(args.device)
            elif args.device.startswith('mps'):
                torch.mps.synchronize()
            metrics.update(update=update, epoch=epoch, lr=lr, elapsed_ms=(time.perf_counter() - start) * 1000,
                           student_padded_positions=sum(b['student']['input_ids'].numel() for b in window),
                           source_padded_positions=sum(b['teacher']['input_ids'].numel() for b in window),
                           microbatches=len(window))
            if is_main_process():
                with (args.run_dir / 'train.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(metrics) + '\n')
                print(json.dumps(metrics), flush=True)
            if update >= total_updates:
                break
        if update >= total_updates:
            break
    if is_main_process():
        output = args.run_dir / f'model_{config.hidden_size}.pth'
        state = {k: v.detach().cpu() for k, v in raw.state_dict().items() if not k.startswith('vision_encoder.')}
        torch.save(state, output.with_suffix('.tmp'))
        os.replace(output.with_suffix('.tmp'), output)
        config.to_json_file(str(output.with_suffix('.config.json')), use_diff=False)
        print(f'Saved {output}; teacher is not part of the inference checkpoint', flush=True)
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
