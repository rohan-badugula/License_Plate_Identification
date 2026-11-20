from alpr.cli import main, parser


def test_command_inventory():
    help_text = parser().format_help()
    for command in (
        "prepare-detection",
        "prepare-characters",
        "train-recognizer",
        "train-detector",
        "evaluate-detector",
        "evaluate-recognizer",
        "infer",
        "video",
        "evaluate",
    ):
        assert command in help_text


def test_missing_dataset_gives_actionable_nonzero_exit(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "alpr",
            "prepare-detection",
            "--images",
            str(tmp_path / "no"),
            "--labels",
            str(tmp_path / "labels"),
            "--output",
            str(tmp_path / "out"),
        ],
    )
    assert main() == 2
    assert "directories must exist" in capsys.readouterr().err
