import io
import tarfile

import pytest

from bankrec.download import extract_selected


def _archive(path, names):
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            content = b"example"
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def test_extracts_only_transactions(tmp_path):
    archive = tmp_path / "detail.tar.gz"
    _archive(archive, ["detail/trx/part.parquet", "detail/geo/part.parquet"])
    assert extract_selected(archive, tmp_path / "out", "detail/trx/") == 1
    assert (tmp_path / "out/detail/trx/part.parquet").read_bytes() == b"example"
    assert not (tmp_path / "out/detail/geo/part.parquet").exists()


def test_rejects_path_escape(tmp_path):
    archive = tmp_path / "detail.tar.gz"
    _archive(archive, ["detail/trx/../escape.parquet"])
    with pytest.raises(RuntimeError, match="unsafe archive path"):
        extract_selected(archive, tmp_path / "out", "detail/trx/")
