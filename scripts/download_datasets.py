"""Download versioned public datasets with hashes and bounded, traversal-safe ZIP extraction."""

import argparse
import json
import shutil
import stat
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from alpr.io import sha256, write_json


def extract_zip(archive: Path, destination: Path, max_bytes: int = 8_000_000_000) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        if sum(m.file_size for m in members) > max_bytes:
            raise ValueError("ZIP exceeds the permitted uncompressed size")
        seen = set()
        for member in members:
            relative = PurePosixPath(member.filename)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
            ):
                raise ValueError(f"Unsafe ZIP entry: {member.filename}")
            target = root.joinpath(*relative.parts).resolve()
            if not target.is_relative_to(root) or target in seen:
                raise ValueError(f"Invalid or duplicate ZIP entry: {member.filename}")
            seen.add(target)
        destination.mkdir(parents=True, exist_ok=True)
        for member in members:
            target = root.joinpath(*PurePosixPath(member.filename).parts)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(member) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)


def download(name: str, specification: dict, output: Path) -> dict:
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / specification["archive"]
    temporary = archive.with_suffix(".part")
    if not archive.is_file():
        request = urllib.request.Request(
            specification["url"], headers={"User-Agent": "paper-alpr/0.1"}
        )
        started = time.monotonic()
        received = 0
        next_progress = 0
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            temporary.open("wb") as stream,
        ):
            expected = int(response.headers.get("Content-Length", 0))
            if expected > 3_000_000_000:
                raise ValueError("Dataset archive exceeds 3 GB download limit")
            while chunk := response.read(1024 * 1024):
                received += len(chunk)
                if received > 3_000_000_000:
                    raise ValueError("Dataset download exceeded 3 GB limit")
                stream.write(chunk)
                if time.monotonic() >= next_progress:
                    print(f"{name}: {received / 1e6:.1f} MB downloaded", flush=True)
                    next_progress = time.monotonic() + 15
        if expected and received != expected:
            raise ValueError("Incomplete dataset download")
        print(f"{name}: download completed in {time.monotonic() - started:.1f}s", flush=True)
        if not zipfile.is_zipfile(temporary):
            raise ValueError("Dataset server did not return a ZIP archive")
        temporary.replace(archive)
    digest = sha256(archive)
    if specification.get("sha256") and digest != specification["sha256"]:
        raise ValueError(f"Dataset archive hash mismatch: {name}")
    marker = directory / "source.json"
    files = directory / "files"
    if marker.is_file():
        if not files.is_dir():
            raise ValueError("Dataset extraction is missing despite an existing source marker")
        existing = json.loads(marker.read_text())
        if existing.get("archive_sha256") != digest:
            raise ValueError("Existing extraction belongs to a different dataset archive")
    else:
        if files.exists():
            raise ValueError(
                f"Incomplete extraction exists; review and remove {files} before retrying"
            )
        extract_zip(archive, files)
    metadata = {
        **specification,
        "archive_sha256": digest,
        "archive_bytes": archive.stat().st_size,
        "file_count": sum(p.is_file() for p in files.rglob("*")),
        "extracted_directory": str(files.resolve()),
    }
    write_json(marker, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset", choices=["all", "mercosur", "characters", "openalpr"], default="all"
    )
    p.add_argument("--output", type=Path, default=Path("data/downloads"))
    args = p.parse_args()
    registry = json.loads((Path(__file__).parents[1] / "configs/datasets.json").read_text())
    for key in registry if args.dataset == "all" else [args.dataset]:
        download(key, registry[key], args.output)
