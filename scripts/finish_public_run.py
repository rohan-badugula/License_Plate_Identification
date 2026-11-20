"""Continue training, verify outputs, and optionally push a results commit."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from alpr.io import write_json


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def finish(args) -> None:
    if args.wait_pid < 1:
        raise ValueError("Expected a positive training process ID")
    state = args.output.parent / "public-followthrough-state.json"
    starting_revision = git("rev-parse", "HEAD")
    write_json(state, {"status": "waiting_for_recognizer", "source_revision": starting_revision})
    history_file = args.recognizer_run / "history.json"
    experiment = json.loads((args.recognizer_run / "experiment.json").read_text())
    while True:
        history = json.loads(history_file.read_text()) if history_file.is_file() else []
        if history and history[-1]["epoch"] == experiment["epochs"]:
            break
        try:
            os.kill(args.wait_pid, 0)
        except ProcessLookupError as error:
            raise RuntimeError("Recognizer exited before all requested epochs completed") from error
        time.sleep(15)
    # Let the training process release its GPU allocations before the next workload.
    time.sleep(5)
    write_json(
        state, {"status": "training_detector_and_evaluating", "source_revision": starting_revision}
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/run_public_experiment.py",
            "--prepared",
            str(args.prepared),
            "--output",
            str(args.output),
            "--device",
            args.device,
            "--recognizer-checkpoint",
            str(args.recognizer_run / "best.pt"),
        ],
        check=True,
    )
    for command in [
        [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"],
        [sys.executable, "-m", "ruff", "format", "--check", "src", "tests", "scripts"],
        [sys.executable, "-m", "pytest", "-q"],
        [sys.executable, "-m", "build", "--no-isolation"],
    ]:
        subprocess.run(command, check=True)
    detector = json.loads((args.output / "detector-test/metrics.json").read_text())
    characters = json.loads((args.output / "character-test.json").read_text())
    benchmark = json.loads((args.output / "openalpr-test/metrics.json").read_text())
    result = {
        "experiment": "New public-data reconstruction; not the original paper benchmark",
        "source_revision": starting_revision,
        "recognizer_training_epochs": len(history),
        "recognizer_history": history,
        "character_test": characters,
        "detector_test": detector,
        "public_benchmark": {k: v for k, v in benchmark.items() if k != "samples"},
        "limitations": [
            "Original private 500-image benchmark unavailable",
            "Byte-group splits do not prove vehicle independence",
            "Public character data identity/provenance is not fully established",
        ],
    }
    result_file = Path("docs/results/public-experiment.json")
    write_json(result_file, result)
    start, end = "<!-- PUBLIC_RESULTS_START -->", "<!-- PUBLIC_RESULTS_END -->"
    readme = Path("README.md")
    text = readme.read_text()
    table = (
        "\nNew measured public-data run (separate from the paper):\n\n"
        "| Metric | New measured result |\n| --- | ---: |\n"
        f"| Fresh detector test mAP50 | {detector['mAP50']:.4f} |\n"
        f"| Character test accuracy | {characters['accuracy']:.4f} |\n"
        f"| OpenALPR exact plate match | {benchmark['plate_exact_match_rate']:.4f} |\n"
        f"| OpenALPR images evaluated | {benchmark['images']} |\n\n"
        "Metrics are fractions. See "
        "[the complete new results](docs/results/public-experiment.json) "
        "and [data provenance limits](docs/DATASETS.md).\n"
    )
    if start not in text or end not in text:
        raise ValueError("README result markers are missing; results remain available locally")
    readme.write_text(text.split(start)[0] + start + table + end + text.split(end, 1)[1])
    verification = Path("docs/VERIFICATION.md")
    completed_section = (
        "## Full real-data experiments\n\n"
        "The full run and held-out evaluations completed. See "
        "[results/public-experiment.json](results/public-experiment.json) for "
        "measured metrics, hashes, epoch history and benchmark limitations. "
        "Local lint, tests and packaging were rerun successfully before recording "
        "these results. This does not recover the original private benchmark.\n\n"
    )
    previous = verification.read_text()
    if "## Full real-data experiments" in previous and "## Hosted CI" in previous:
        verification.write_text(
            previous.split("## Full real-data experiments")[0]
            + completed_section
            + "## Hosted CI"
            + previous.split("## Hosted CI", 1)[1]
        )
    else:
        verification.write_text(previous + "\n" + completed_section)
    if args.push_results:
        if git("rev-parse", "HEAD") != starting_revision:
            raise RuntimeError("Repository revision changed; results saved locally for review")
        if git("branch", "--show-current") != "implementation/end-to-end":
            raise RuntimeError("Implementation branch changed; automatic publication stopped")
        if git("diff", "--cached", "--name-only"):
            raise RuntimeError("Staged changes exist; automatic publication stopped")
        expected_remote = "https://github.com/rohan-badugula/License_Plate_Identification.git"
        if git("remote", "get-url", "origin") != expected_remote:
            raise RuntimeError("Unexpected Git remote; automatic publication stopped")
        paths = [str(result_file), "README.md", "docs/VERIFICATION.md"]
        subprocess.run(["git", "add", "--", *paths], check=True)
        subprocess.run(
            [
                "git",
                "commit",
                "-m",
                "results: record verified full public-data ALPR experiment",
                "--only",
                "--",
                *paths,
            ],
            check=True,
        )
        subprocess.run(["git", "push", "origin", "HEAD:implementation/end-to-end"], check=True)
    write_json(
        state,
        {
            "status": "complete",
            "results": str(result_file),
            "pushed": args.push_results,
            "source_revision": starting_revision,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wait-pid", type=int, required=True)
    p.add_argument("--recognizer-run", type=Path, default=Path("runs/public-b7"))
    p.add_argument("--prepared", type=Path, default=Path("data/prepared"))
    p.add_argument("--output", type=Path, default=Path("runs/public-experiment"))
    p.add_argument("--device", default="mps")
    p.add_argument("--push-results", action="store_true")
    arguments = p.parse_args()
    try:
        finish(arguments)
    except Exception as error:
        write_json(
            arguments.output.parent / "public-followthrough-state.json",
            {"status": "failed", "error": str(error)},
        )
        raise
