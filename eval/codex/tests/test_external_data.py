from eval.codex.external_data import perceived_label


def test_external_label_retains_seventh_class_before_selecting_shared_labels():
    assert perceived_label(['surprise']*3+['anger']*2) is None
    assert perceived_label(['anger']*2+['joy']*2+['fear']) is None
    assert perceived_label(['joy']*3+['surprise']*2) == 3
    assert perceived_label(['neutrality']*5) == 4
