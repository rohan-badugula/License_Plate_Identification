import importlib.util
import stat
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "download_datasets", Path(__file__).parents[1] / "scripts/download_datasets.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("filename", ["../escape.txt", "/absolute.txt", "dir\\escape.txt"])
def test_zip_path_traversal_is_rejected(tmp_path, filename):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr(filename, "untrusted content")
    with pytest.raises(ValueError, match="Unsafe"):
        module.extract_zip(archive, tmp_path / "extract")
    assert not (tmp_path / "extract").exists()


def test_zip_symlinks_and_expansion_limit(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as f:
        entry = zipfile.ZipInfo("link")
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        f.writestr(entry, "../../elsewhere")
    with pytest.raises(ValueError, match="Unsafe"):
        module.extract_zip(archive, tmp_path / "extract")
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("file.txt", "12345")
    with pytest.raises(ValueError, match="uncompressed"):
        module.extract_zip(archive, tmp_path / "extract", max_bytes=4)


def test_zip_extraction_preserves_bytes(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as f:
        f.writestr("nested/image.txt", b"sample")
    module.extract_zip(archive, tmp_path / "extract")
    assert (tmp_path / "extract/nested/image.txt").read_bytes() == b"sample"
