"""Resumable byte-range download of the public MBD-mini archives."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import truststore

DATASET = "https://huggingface.co/datasets/ai-lab/MBD-mini/resolve/main"
ARCHIVES = {
    "client_split.tar.gz": "client_split/",
    "targets.tar.gz": "targets/",
    "detail.tar.gz": "detail/trx/",
}


def _metadata(name: str) -> tuple[int, str]:
    response = requests.head(f"{DATASET}/{name}", allow_redirects=False, timeout=30)
    response.raise_for_status()
    size = response.headers.get("X-Linked-Size")
    digest = response.headers.get("X-Linked-ETag", "").strip('"')
    if not size or not re.fullmatch(r"[a-fA-F0-9]{64}", digest):
        raise RuntimeError(f"missing trustworthy size/SHA-256 metadata for {name}")
    return int(size), digest.lower()


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def download_archive(root: Path, name: str, *, workers: int = 8) -> Path:
    """Resume nonoverlapping ranges and verify the complete archive before use."""
    if name not in ARCHIVES or workers < 1:
        raise ValueError("unknown archive or invalid worker count")
    root.mkdir(parents=True, exist_ok=True)
    size, digest = _metadata(name)
    target = root / name
    if target.is_file() and target.stat().st_size == size and _sha256(target) == digest:
        return target
    if target.is_file():
        raise RuntimeError(f"existing {target} failed size or SHA-256 verification")
    workers = min(workers, size)
    ranges = [(i * size // workers, (i + 1) * size // workers - 1) for i in range(workers)]

    def fetch(index: int, start: int, end: int) -> Path:
        part = root / f"{name}.part{index:02d}"
        expected = end - start + 1
        present = part.stat().st_size if part.exists() else 0
        if present > expected:
            raise RuntimeError(f"oversized partial download: {part}")
        for attempt in range(5):
            present = part.stat().st_size if part.exists() else 0
            if present == expected:
                return part
            first = start + present
            try:
                with requests.get(
                    f"{DATASET}/{name}",
                    headers={"Range": f"bytes={first}-{end}"},
                    stream=True,
                    timeout=(30, 120),
                ) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise RuntimeError("server ignored byte-range request")
                    content_range = response.headers.get("Content-Range", "")
                    if content_range != f"bytes {first}-{end}/{size}":
                        raise RuntimeError(f"unexpected Content-Range: {content_range}")
                    with part.open("ab") as output:
                        for block in response.iter_content(chunk_size=1024 * 1024):
                            if block:
                                output.write(block)
            except requests.RequestException:
                if attempt == 4:
                    raise
        if part.stat().st_size != expected:
            raise RuntimeError(f"incomplete range {index} for {name}")
        return part

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(
            pool.map(
                lambda item: fetch(*item),
                [(i, start, end) for i, (start, end) in enumerate(ranges)],
            )
        )
    assembled = root / f"{name}.assembling"
    try:
        with assembled.open("wb") as output:
            for part in parts:
                with part.open("rb") as source:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
        if assembled.stat().st_size != size or _sha256(assembled) != digest:
            raise RuntimeError(f"SHA-256 verification failed for {name}")
        os.replace(assembled, target)
        for part in parts:
            part.unlink()
    finally:
        if assembled.exists():
            assembled.unlink()
    return target


def extract_selected(archive_path: Path, root: Path, prefix: str) -> int:
    """Extract only regular files under an expected archive prefix."""
    root = root.resolve()
    allowed_root = (root / prefix).resolve()
    if not allowed_root.is_relative_to(root):
        raise RuntimeError("extraction destination escapes data directory")
    count = 0
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            name = member.name.removeprefix("./")
            if not name.startswith(prefix) or not member.isfile():
                continue
            target = (root / name).resolve()
            if not target.is_relative_to(allowed_root):
                raise RuntimeError(f"unsafe archive path: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and target.stat().st_size == member.size:
                count += 1
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot extract {name}")
            partial = target.with_name(target.name + ".partial")
            with source, partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if partial.stat().st_size != member.size:
                raise RuntimeError(f"incomplete extracted file: {name}")
            os.replace(partial, target)
            count += 1
    if count == 0:
        raise RuntimeError(f"archive has no files under {prefix}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Download MBD-mini transaction inputs")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    truststore.inject_into_ssl()
    for name, prefix in ARCHIVES.items():
        print(f"Downloading/verifying {name}", flush=True)
        archive = download_archive(args.data_dir, name, workers=args.workers)
        print(
            f"Extracting {prefix}: {extract_selected(archive, args.data_dir, prefix)} files",
            flush=True,
        )


if __name__ == "__main__":
    main()
