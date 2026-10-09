"""Opt-in real-checkpoint integration probe, NOT an accuracy/speed benchmark.

Uses existing repository demo images. Saves generation prefixes at three budgets
and a two-row teacher-pseudolabel Parquet for a one-update training smoke test.
The pseudolabels are not human-verified and must not be used as a test set.
"""
import argparse
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import transformers
from PIL import Image
from transformers import AutoTokenizer

from model.model_vlm import MiniMindVLM, VLMConfig
from trainer.train_budget_vlm import load_nonvision_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'out/sft_vlm_768.pth')
    parser.add_argument('--vision_model_path', type=Path, default=ROOT / 'model/siglip2-base-p32-256-ve')
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=24)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error('output_dir exists; choose a new directory')
    if args.max_new_tokens < 1 or args.threads < 1:
        parser.error('max_new_tokens and threads must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(ROOT / 'model')
    model = MiniMindVLM(VLMConfig(), vision_model_path=str(args.vision_model_path))
    load_nonvision_checkpoint(model, args.checkpoint)
    model.to(args.device).eval()
    images = sorted((ROOT / 'dataset/eval_images').glob('*.jpg'))[:2]
    if len(images) != 2:
        raise FileNotFoundError('Expected at least two upstream demo images')
    predictions, teacher_answers = [], {}
    prompt_text = '请描述这张图中的主要物体和场景。'
    with torch.inference_mode():
        for path in images:
            with Image.open(path) as image:
                pixels = MiniMindVLM.image2tensor(image, model.processor)['pixel_values'].to(args.device)
            raw, mask = model.encode_image_features(pixels)
            for budget in (64, 16, 4):
                model.config.image_token_len = model.vision_proj.target_tokens = budget
                question = model.config.image_special_token * budget + '\n' + prompt_text
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}],
                                                        tokenize=False, add_generation_prompt=True,
                                                        open_thinking=False)
                ids = tokenizer(prompt, add_special_tokens=False, return_tensors='pt').input_ids.to(args.device)
                logits = model(ids, vision_features=raw, image_mask=mask).logits
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Non-finite real-checkpoint logits')
                generated = model.generate(ids, vision_features=raw, image_mask=mask,
                                            do_sample=False, top_k=0, top_p=1.0, temperature=1.0,
                                            max_new_tokens=args.max_new_tokens, eos_token_id=tokenizer.eos_token_id)
                suffix = generated[0, ids.shape[1]:]
                response = tokenizer.decode(suffix, skip_special_tokens=True)
                predictions.append({'image': str(path.relative_to(ROOT)), 'budget': budget,
                                    'response_prefix': response, 'tokens_including_eos': suffix.numel(),
                                    'stopped_at_eos': bool(suffix[-1] == tokenizer.eos_token_id),
                                    'finite_logits': True})
                print(json.dumps(predictions[-1], ensure_ascii=False), flush=True)
                if budget == 64:
                    if not response.strip():
                        raise ValueError('Teacher generated no usable text for the training fixture')
                    teacher_answers[path] = response
    report = {'scope': 'Real upstream weights, two public demo images, generation prefixes. '
                       'No independent labels, no benchmark score, no measured deployment speedup.',
              'python': platform.python_version(), 'torch': torch.__version__,
              'transformers': transformers.__version__, 'device': args.device,
              'max_new_tokens': args.max_new_tokens,
              'full_parameters': sum(p.numel() for p in model.parameters()),
              'vision_parameters': sum(p.numel() for p in model.vision_encoder.parameters()),
              'predictions': predictions}
    records = [{'image_bytes': path.read_bytes(), 'conversations': json.dumps([
        {'role': 'system', 'content': '你是一个视觉助手。'},
        {'role': 'user', 'content': '<image>\n' + prompt_text},
        {'role': 'assistant', 'content': teacher_answers[path]}], ensure_ascii=False)} for path in images]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'probe.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    pq.write_table(pa.Table.from_pylist(records), args.output_dir / 'smoke_pseudolabels.parquet')


if __name__ == '__main__':
    main()
