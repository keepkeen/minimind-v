import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image
from transformers import AutoTokenizer, SiglipImageProcessor

from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import vlm_collate_fn


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained(str(Path(__file__).resolve().parents[1] / 'model'))


def make_dataset(tmp_path, tokenizer, conversations, max_length=256, image_count=1):
    image = io.BytesIO()
    Image.new('RGB', (8, 8)).save(image, format='PNG')
    table = pa.Table.from_pylist([{'conversations': json.dumps(conversations),
                                 'image_bytes': [image.getvalue()] * image_count}])
    path = tmp_path / 'tiny.parquet'
    pq.write_table(table, path)
    return VLMDataset(str(path), tokenizer, preprocess=SiglipImageProcessor(size={'height': 8, 'width': 8}),
                      max_length=max_length, image_token_len=4)


def test_real_tokenizer_labels_and_image_contract(tmp_path, tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, [
        {'role': 'system', 'content': 'test'},
        {'role': 'user', 'content': '<image> What color?'},
        {'role': 'assistant', 'content': 'red'},
    ])
    inputs, labels, images = dataset[0]
    assert (inputs == dataset.image_id).sum() == 4
    assert (labels[inputs == tokenizer.pad_token_id] == -100).all()
    assert (labels[inputs == dataset.image_id] == -100).all()
    assert (labels != -100).any()
    assert images['pixel_values'].shape == (1, 3, 8, 8)


def test_truncation_without_answer_fails(tmp_path, tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, [
        {'role': 'system', 'content': 'test'},
        {'role': 'user', 'content': '<image> ' + 'question ' * 100},
        {'role': 'assistant', 'content': 'answer'},
    ], max_length=40)
    with pytest.raises(ValueError, match='no supervised answer'):
        dataset[0]


def test_text_only_ignores_upstream_dummy_image(tmp_path, tokenizer):
    dataset = make_dataset(tmp_path, tokenizer, [
        {'role': 'system', 'content': 'test'},
        {'role': 'user', 'content': 'hello'},
        {'role': 'assistant', 'content': 'hi'},
    ])
    _, _, images = dataset[0]
    assert images['pixel_values'].shape[0] == 0
    assert images['image_mask'].numel() == 0


def test_variable_image_collation_including_text_only():
    batch = []
    for count in (2, 0, 1):
        batch.append((torch.ones(8, dtype=torch.long), torch.ones(8, dtype=torch.long),
                      {'pixel_values': torch.ones(count, 3, 8, 8),
                       'image_mask': torch.ones(count, dtype=torch.bool)}))
    _, _, images = vlm_collate_fn(batch)
    assert images['pixel_values'].shape == (3, 2, 3, 8, 8)
    assert images['image_mask'].tolist() == [[True, True], [False, False], [True, False]]


def test_missing_eos_never_labels_padding(tokenizer):
    dataset = object.__new__(VLMDataset)
    dataset.bos_id = [1, 3]
    dataset.eos_id = [2, 4]
    dataset.max_length = 20
    labels = dataset.generate_labels([1, 3, 8, 9])
    assert labels == [-100, -100, 8, 9]
