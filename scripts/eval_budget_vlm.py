"""Deterministic, batch-size-one visual QA evaluation with explicit timing boundaries.

Run from the repository root. Requires real checkpoints and a local JSONL manifest.
This script does NOT fabricate a benchmark dataset, scores or a training result.
"""
import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import transformers
from PIL import Image

from eval_vlm import init_model
from model.model_vlm import MiniMindVLM
from scripts.vlm_metrics import score_answer
from trainer.trainer_utils import setup_seed


def synchronize(device):
    if device.startswith('cuda'):
        torch.cuda.synchronize(device)
    elif device.startswith('mps'):
        torch.mps.synchronize()


class FirstTokenTimer:
    def __init__(self, device):
        self.device, self.prompt_seen, self.first = device, False, None

    def put(self, value):
        # The upstream custom generator first emits the entire input prompt.
        if not self.prompt_seen:
            self.prompt_seen = True
        elif self.first is None:
            synchronize(self.device)
            self.first = time.perf_counter()

    def end(self):
        pass


def read_manifest(path):
    rows, ids = [], set()
    with path.open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get('id'), str) or row['id'] in ids:
                raise ValueError(f'Line {line_number}: id must be a unique string')
            ids.add(row['id'])
            if not isinstance(row.get('question'), str):
                raise ValueError(f'Line {line_number}: question must be a string')
            if 'group_id' in row and (not isinstance(row['group_id'], str) or not row['group_id']):
                raise ValueError(f'Line {line_number}: group_id must be a nonempty string')
            score_answer('', row.get('answers', []))
            paths = row.get('images', [row['image']] if 'image' in row else [])
            if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
                raise ValueError(f'Line {line_number}: provide image or a nonempty images list')
            row['images'] = [str((path.parent / p).resolve()) for p in paths]
            if not all(Path(p).is_file() for p in row['images']):
                raise FileNotFoundError(f'Line {line_number}: image path does not exist')
            rows.append(row)
    if not rows:
        raise ValueError('Manifest is empty')
    return rows


def summarize(rows):
    def aggregate(values):
        latencies = sorted(value['generation_ms'] for value in values)
        return {
            'samples': len(values),
            'normalized_em': statistics.mean(value['normalized_em'] for value in values),
            'anls_style': statistics.mean(value['anls_style'] for value in values),
            'generation_ms_p50': statistics.median(latencies),
            'generation_ms_p95_nearest_rank': latencies[max(0, (95 * len(latencies) + 99) // 100 - 1)],
            'ttft_ms_mean': statistics.mean(value['ttft_ms'] for value in values),
            'request_ms_mean': statistics.mean(value['request_ms'] for value in values),
        }
    return {'overall': aggregate(rows), 'by_category': {
        category: aggregate([row for row in rows if row['category'] == category])
        for category in sorted({row['category'] for row in rows})
    }}


@torch.inference_mode()
def evaluate_one(row, image_paths, model, tokenizer, processor, args):
    synchronize(args.device)
    request_start = time.perf_counter()
    processed = []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert('RGB')
            if args.image_condition == 'blank':
                image = Image.new('RGB', image.size, (0, 0, 0))
            processed.append(MiniMindVLM.image2tensor(image, processor)['pixel_values'])
    pixels = torch.cat(processed, dim=0).unsqueeze(0).to(args.device)
    question = row['question']
    if '<image>' not in question:
        question = '<image>\n' * len(image_paths) + question
    if question.count('<image>') != len(image_paths):
        raise ValueError(f"{row['id']}: question/image count mismatch")
    question = question.replace('<image>', model.config.image_special_token * model.config.image_token_len)
    prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                          tokenize=False, add_generation_prompt=True, open_thinking=False)
    inputs = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).to(args.device)
    if inputs.input_ids.shape[1] + args.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError(f"{row['id']}: request exceeds model positional capacity")
    synchronize(args.device)
    generation_start = time.perf_counter()
    timer = FirstTokenTimer(args.device)
    output = model.generate(inputs=inputs.input_ids, attention_mask=inputs.attention_mask,
                            pixel_values={'pixel_values': pixels}, max_new_tokens=args.max_new_tokens,
                            do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
                            eos_token_id=tokenizer.eos_token_id, use_cache=True, streamer=timer)
    synchronize(args.device)
    generation_end = time.perf_counter()
    generated = output[0, inputs.input_ids.shape[1]:]
    prediction = tokenizer.decode(generated, skip_special_tokens=True)
    if timer.first is None:
        raise RuntimeError('No generated token was observed by the timer')
    tokens = generated.numel()  # Includes EOS if generated.
    result = {
        'id': row['id'], 'category': row.get('category', 'unspecified'),
        'question': row['question'], 'source_images': row['images'],
        'group_id': row.get('group_id', json.dumps(row['images'], ensure_ascii=False)),
        'prediction': prediction, 'answers': row['answers'], 'image_condition': args.image_condition,
        'image_paths_used': image_paths, 'images': len(image_paths),
        'image_tokens': len(image_paths) * model.config.image_token_len,
        'prompt_tokens': inputs.input_ids.shape[1], 'generated_tokens_including_eos': tokens,
        'preprocess_ms': (generation_start - request_start) * 1000,
        'ttft_ms': (timer.first - generation_start) * 1000,
        'generation_ms': (generation_end - generation_start) * 1000,
        'request_ms': (generation_end - request_start) * 1000,
        'decode_tps_after_first_token': (tokens - 1) / (generation_end - timer.first) if tokens > 1 else None,
    }
    result.update(score_answer(prediction, row['answers']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--load_from', default='model')
    parser.add_argument('--save_dir', default='out')
    parser.add_argument('--vision_model_path', default='./model/siglip2-base-p32-256-ve')
    parser.add_argument('--weight', default='sft_vlm')
    parser.add_argument('--hidden_size', type=int, default=768)
    parser.add_argument('--num_hidden_layers', type=int, default=8)
    parser.add_argument('--use_moe', type=int, choices=[0, 1], default=0)
    parser.add_argument('--image_token_len', type=int, choices=[4, 16, 64], default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--image_condition', choices=['actual', 'blank', 'shuffle'], default='actual')
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.warmup < 0:
        parser.error('max_new_tokens must be positive and warmup nonnegative')
    if args.output.exists() or args.output.with_suffix('.summary.json').exists():
        parser.error('Output exists; choose a new filename to preserve previous results')
    rows = read_manifest(args.manifest)
    image_groups = [row['images'] for row in rows]
    if args.image_condition == 'shuffle':
        if any(len(group) != 1 for group in image_groups) or len(set(group[0] for group in image_groups)) < 2:
            parser.error('shuffle requires single-image samples and at least two distinct images')
        # Rotate UNIQUE image identities: repeated questions cannot accidentally retain their image.
        unique = list(dict.fromkeys(group[0] for group in image_groups))
        replacement = {path: unique[(i + 1) % len(unique)] for i, path in enumerate(unique)}
        image_groups = [[replacement[group[0]]] for group in image_groups]
    setup_seed(args.seed)
    model, tokenizer, processor = init_model(args)
    for _ in range(args.warmup):
        evaluate_one(rows[0], image_groups[0], model, tokenizer, processor, args)
    if args.device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with args.output.open('x', encoding='utf-8') as handle:
        for row, paths in zip(rows, image_groups):
            result = evaluate_one(row, paths, model, tokenizer, processor, args)
            results.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    summary = summarize(results)
    summary['metadata'] = {
        'git_commit': commit, 'dirty_worktree': dirty, 'python': platform.python_version(),
        'platform': platform.platform(), 'torch': torch.__version__, 'transformers': transformers.__version__,
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'all_model_parameters': sum(p.numel() for p in model.parameters()),
        'vision_parameters': sum(p.numel() for p in model.vision_encoder.parameters()),
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if args.device.startswith('cuda') else None,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(args.device) if args.device.startswith('cuda') else None,
        'timing_scope': 'batch=1; warmup excluded; TTFT includes vision+prefill; request includes image IO/preprocess, excludes model loading and answer detokenization; streaming instrumentation included',
        'metric_scope': 'Local normalized EM and ANLS-style; not official VQAv2/DocVQA/POPE scores',
    }
    with args.output.with_suffix('.summary.json').open('x', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
