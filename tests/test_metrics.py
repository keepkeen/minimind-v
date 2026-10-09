import pytest
from scripts.vlm_metrics import edit_distance, normalize_answer, score_answer


def test_normalization_preserves_critical_symbols():
    assert normalize_answer(' ＡＢＣ  １２.５ ') == 'abc 12.5'
    assert score_answer('12.5', ['125'])['normalized_em'] == 0


def test_distance_and_multiple_references():
    assert edit_distance('kitten', 'sitting') == 3
    assert score_answer('红色', ['red', '红色'])['normalized_em'] == 1
    assert score_answer('abce', ['abcd'])['anls_style'] == 0.75
    assert score_answer('abxx', ['abcd'])['anls_style'] == 0
    assert score_answer('', [''])['anls_style'] == 1
    with pytest.raises(ValueError):
        score_answer('answer', [])
