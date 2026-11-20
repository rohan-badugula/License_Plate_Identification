import argparse
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "finish_public_run", Path(__file__).parents[1] / "scripts/finish_public_run.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def setup_run(tmp_path, monkeypatch, push=False):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text(
        "before<!-- PUBLIC_RESULTS_START -->pending<!-- PUBLIC_RESULTS_END -->after"
    )
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "VERIFICATION.md").write_text("## Full real-data experiments\npending\n## Hosted CI\n")
    run = tmp_path / "runs/b7"
    run.mkdir(parents=True)
    (run / "experiment.json").write_text('{"epochs":1}')
    (run / "history.json").write_text('[{"epoch":1}]')
    output = tmp_path / "runs/full"
    for relative, report in [
        ("detector-test/metrics.json", {"mAP50": 0.5}),
        ("character-test.json", {"accuracy": 0.6}),
        (
            "openalpr-test/metrics.json",
            {"plate_exact_match_rate": 0.2, "images": 10, "samples": []},
        ),
    ]:
        p = output / relative
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(report))
    calls = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    monkeypatch.setattr(module, "git", lambda *args: "revision")
    return argparse.Namespace(
        output=output,
        recognizer_run=run,
        prepared=tmp_path / "prepared",
        wait_pid=123,
        device="cpu",
        push_results=push,
    ), calls


def test_completed_run_records_truthful_results_without_pushing_by_default(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch)
    module.finish(args)
    result = json.loads((tmp_path / "docs/results/public-experiment.json").read_text())
    assert result["recognizer_training_epochs"] == 1
    assert "not the original" in result["experiment"]
    assert "samples" not in result["public_benchmark"]
    assert "0.2000" in (tmp_path / "README.md").read_text()
    assert all(command[0] != "git" for command in calls)
    assert (
        json.loads((tmp_path / "runs/public-followthrough-state.json").read_text())["status"]
        == "complete"
    )


def test_changed_branch_stops_automatic_git_publication(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch, push=True)
    monkeypatch.setattr(
        module, "git", lambda *args: "other-branch" if args[0] == "branch" else "revision"
    )
    with pytest.raises(RuntimeError, match="branch changed"):
        module.finish(args)
    assert all(command[0] != "git" for command in calls)


def test_incomplete_training_process_is_not_treated_as_success(tmp_path, monkeypatch):
    args, calls = setup_run(tmp_path, monkeypatch)
    (args.recognizer_run / "history.json").write_text("[]")

    def missing_process(*args):
        raise ProcessLookupError

    monkeypatch.setattr(module.os, "kill", missing_process)
    with pytest.raises(RuntimeError, match="before all requested epochs"):
        module.finish(args)
    assert calls == []
