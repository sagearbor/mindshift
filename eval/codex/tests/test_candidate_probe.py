import pytest
from eval.codex.candidate_probe import parse_answer, select_rows


def test_parser_requires_one_final_answer_not_reasoning_mentions():
    assert parse_answer('E') == 4
    assert parse_answer('<think>Possibly anger or sadness</think><answer>F</answer>') == 5
    assert parse_answer('<think>Example <answer>A</answer></think><answer>F</answer>') == 5
    with pytest.raises(ValueError):
        parse_answer('<think>Unfinished <answer>F</answer>')
    with pytest.raises(ValueError):
        parse_answer('<think>anger</think>')
    with pytest.raises(ValueError):
        parse_answer('<answer>A or F</answer>')


def test_subset_never_includes_test_or_training_and_is_balanced():
    rows=[dict(id=f'{split}-{i}',split=split,majority=i%6) for split in ['train','dev','test'] for i in range(12)]
    chosen=select_rows(rows,6)
    assert {r['split'] for r in chosen}=={'dev'}
    assert [r['majority'] for r in chosen]==list(range(6))
    with pytest.raises(ValueError):select_rows(rows,0)


def test_subset_rotates_speakers_within_each_emotion():
    rows=[dict(id=f'{speaker}-{label}-{j}',speaker=speaker,split='dev',majority=label)
          for speaker in ['a','b','c'] for label in range(6) for j in range(3)]
    chosen=select_rows(rows,18)
    for label in range(6):
        assert {r['speaker'] for r in chosen if r['majority']==label}=={'a','b','c'}
