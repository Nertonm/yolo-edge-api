from preprocessing.experiments import run_baseline


def test_parse_classes_arg_accepts_csv_and_spaces():
    assert run_baseline.parse_classes_arg("0,2") == [0, 2]
    assert run_baseline.parse_classes_arg("0 2") == [0, 2]
    assert run_baseline.parse_classes_arg(None) is None
