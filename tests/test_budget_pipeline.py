import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionConfig, SiglipVisionModel

from scripts.compare_budget_runs import paired_comparison, read_predictions


def predictions(scores):
    return {str(i): {'id': str(i), 'question': 'color?', 'answers': ['red'],
                     'source_images': [f'{i // 2}.png'], 'group_id': f'image-{i // 2}',
                     'category': 'color', 'image_condition': 'actual', 'normalized_em': score}
            for i, score in enumerate(scores)}


def test_cluster_bootstrap_is_paired_deterministic_and_reports_retention():
    left, right = predictions([1, 1, 0, 0]), predictions([1, 0, 1, 1])
    result = paired_comparison(left, right, bootstrap=200, seed=7)
    assert result == paired_comparison(left, right, bootstrap=200, seed=7)
    assert result['independent_groups'] == 2
    assert result['candidate_minus_baseline'] == 0.25
    assert result['retention_on_baseline_correct'] == 0.5
    assert result['improved'] == 2 and result['regressed'] == 1
    assert result['cluster_bootstrap_ci95'] == [-0.5, 1.0]
    assert len(result['warnings']) == 2


@pytest.mark.parametrize('field', ['question', 'answers', 'source_images', 'group_id', 'category'])
def test_unmatched_pair_metadata_is_rejected(field):
    left, right = predictions([1, 0]), predictions([1, 1])
    right['0'][field] = 'different'
    with pytest.raises(ValueError, match=field):
        paired_comparison(left, right)


def test_pairing_rejects_different_ids_and_nonfinite_scores():
    with pytest.raises(ValueError, match='same nonempty'):
        paired_comparison(predictions([1]), predictions([1, 0]))
    with pytest.raises(ValueError, match='finite score'):
        paired_comparison(predictions([1]), predictions([float('nan')]))


def test_comparison_cli_preserves_run_metadata(monkeypatch, tmp_path):
    from scripts import compare_budget_runs
    paths = []
    for name, scores in [('base', [1, 0]), ('candidate', [1, 1])]:
        path = tmp_path / f'{name}.jsonl'
        path.write_text('\n'.join(json.dumps(v) for v in predictions(scores).values()))
        path.with_suffix('.summary.json').write_text(json.dumps({'metadata': {'seed': 42, 'name': name}}))
        paths.append(path)
    output = tmp_path / 'comparison.json'
    monkeypatch.setattr('sys.argv', ['compare', '--baseline', str(paths[0]), '--candidate', str(paths[1]),
                                    '--output', str(output), '--bootstrap', '20'])
    compare_budget_runs.main()
    result = json.loads(output.read_text())
    assert result['run_metadata']['baseline']['seed'] == 42
    paths[0].write_text(paths[0].read_text() + '\n' + json.dumps(predictions([1])['0']))
    with pytest.raises(ValueError, match='unique'):
        read_predictions(paths[0])


@pytest.mark.parametrize('with_teacher', [False, True])
def test_full_training_cli_checkpoint_and_native_inference(monkeypatch, tmp_path, with_teacher):
    """Actual SigLIP module, real tokenizer/Parquet and optimizer; RANDOM weights."""
    from model.model_vlm import MiniMindVLM, VLMConfig
    from trainer import train_budget_vlm
    from eval_vlm import init_model
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        vision = tmp_path / 'vision'
        config = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                                    num_attention_heads=4, image_size=32, patch_size=4)
        SiglipVisionModel(config).save_pretrained(vision)
        SiglipImageProcessor(size={'height': 32, 'width': 32}).save_pretrained(vision)
        model_config = VLMConfig(hidden_size=32, num_hidden_layers=1, image_hidden_size=16)
        initial = MiniMindVLM(model_config, vision_model_path=str(vision))
        checkpoint = tmp_path / 'init.pth'
        torch.save({k: v for k, v in initial.state_dict().items() if not k.startswith('vision_encoder.')}, checkpoint)
        records = []
        for color in ['red', 'blue', 'green']:
            buffer = io.BytesIO()
            Image.new('RGB', (32, 32), color).save(buffer, format='PNG')
            records.append({'image_bytes': buffer.getvalue(), 'conversations': json.dumps([
                {'role': 'system', 'content': 'test'},
                {'role': 'user', 'content': '<image> What color?'},
                {'role': 'assistant', 'content': color}])})
        parquet = tmp_path / 'train.parquet'
        pq.write_table(pa.Table.from_pylist(records), parquet)
        output = tmp_path / 'run'
        argv = ['train_budget', '--data_path', str(parquet), '--init_checkpoint', str(checkpoint),
                '--run_dir', str(output), '--vision_model_path', str(vision), '--image_token_len', '4',
                '--hidden_size', '32', '--num_hidden_layers', '1', '--image_hidden_size', '16',
                '--max_seq_len', '256', '--batch_size', '2', '--accumulation_steps', '4',
                '--max_updates', '1', '--freeze_llm', '2', '--device', 'cpu']
        if with_teacher:
            argv.extend(['--teacher_checkpoint', str(checkpoint)])
        monkeypatch.setattr('sys.argv', argv)
        train_budget_vlm.main()
        state = torch.load(output / 'model_32.pth', weights_only=True)
        assert not any(k.startswith('vision_encoder.') for k in state)
        assert json.loads((output / 'model_32.config.json').read_text())['image_token_len'] == 4
        metadata = json.loads((output / 'run.json').read_text())
        metrics = json.loads((output / 'train.jsonl').read_text())
        assert metadata['teacher_active'] == with_teacher
        assert metrics['microbatches'] == 2 and metrics['supervised_tokens'] > 0
        assert metrics['student_padded_positions'] < metrics['source_padded_positions']
        assert all(torch.equal(state[k], v) for k, v in initial.state_dict().items()
                   if not k.startswith(('vision_encoder.', 'vision_proj.')))
        # Native loader accepts absolute directories and uses all saved dimensions.
        monkeypatch.chdir(Path(__file__).resolve().parents[1])
        args = SimpleNamespace(load_from='model', save_dir=str(output), weight='model', hidden_size=32,
                               num_hidden_layers=1, use_moe=0, image_token_len=4,
                               vision_model_path=str(vision), device='cpu')
        reloaded, _, _ = init_model(args)
        assert reloaded.config.image_hidden_size == 16
        ids = torch.tensor([[12] * 4 + [3, 4]])
        result = reloaded(ids, pixel_values=torch.zeros(1, 3, 32, 32))
        assert result.logits.shape == (1, 6, 6400)
        args.image_token_len = 16
        with pytest.raises(ValueError, match='config mismatch'):
            init_model(args)
    finally:
        torch.set_num_threads(old_threads)
