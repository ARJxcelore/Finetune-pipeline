"""Reproducibly obtain the LogHub HDFS_v1 dataset.

Primary source of truth per the loghub repository (https://github.com/logpai/loghub):
the Zenodo-hosted archive for HDFS_v1. In practice Zenodo has proven unreliable
for this ~186MB file -- repeated 502/503/504 gateway errors and mid-transfer
disconnects, observed independently from two different networks -- so a
second source is supported: the `shawhin/HDFS_v1_blocks` Hugging Face dataset,
which is the same underlying LogHub HDFS_v1 data (same 575,061 blocks, same
`blk_<id>` scheme, same label semantics -- verified by direct inspection),
pre-grouped into one row per block. Either source is converted/extracted into
the same `extracted/{HDFS.log,anomaly_label.csv}` layout so nothing downstream
of this module needs to know or care which source was used. The archive/rows
are never modified in place; parsing/reconstruction reads from the result
read-only.

One real difference: the Hugging Face version's `text` field strips the
date/time/pid columns that prefix each line in the true raw HDFS.log (kept:
`LEVEL COMPONENT: CONTENT`). This doesn't affect block-id grouping or
evidence selection (neither depends on timestamps), but it means traces
sourced from Hugging Face have slightly less raw fidelity than the genuine
Zenodo archive -- documented here rather than silently glossed over.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_URL = "https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1"
# Known-good MD5 of HDFS_v1.zip at the above Zenodo record (from the record's
# reported checksum). Verified when present; a mismatch fails loudly instead
# of silently training on a corrupted/partial download.
KNOWN_MD5 = "76a24b4d9a6164d543fb275f89773260"

HF_DATASET_REPO = "shawhin/HDFS_v1_blocks"
HF_SPLITS = ("train", "dev", "test")

EXPECTED_MEMBERS = [
    "HDFS.log",
    "anomaly_label.csv",
    "Event_traces.csv",
    "HDFS_templates.csv",
]
# What we actually read downstream (src.data.hdfs.build_dataset); the other two
# are part of the official Zenodo archive but aren't consumed by this pipeline,
# and don't exist at all in the Hugging Face conversion.
REQUIRED_MEMBERS = ["HDFS.log", "anomaly_label.csv"]


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _attempt_download(url: str, tmp: Path, *, timeout: int) -> None:
    """One resumable download attempt. Appends to tmp if it already has bytes.

    A server that closes the connection mid-transfer (what Zenodo does to us
    here) surfaces as a clean EOF to shutil.copyfileobj, not an exception --
    so this must independently check the transferred byte count against
    Content-Length/Content-Range and raise if the server hung up early,
    otherwise a truncated file is silently accepted as "downloaded".
    """
    resume_from = tmp.stat().st_size if tmp.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}
    request = Request(url, headers=headers)

    with urlopen(request, timeout=timeout) as response:
        # A server that ignores our Range header (200, not 206) is sending the
        # whole file again from byte 0 -- truncate instead of corrupting-append.
        resuming = bool(resume_from and response.status == 206)
        mode = "ab" if resuming else "wb"
        expected_body_length = response.length  # bytes left in *this* response, per Content-Length
        written = 0
        with tmp.open(mode) as out:
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)

    if expected_body_length is not None and written < expected_body_length:
        raise ConnectionError(
            f"server closed the connection early: got {written:,} of {expected_body_length:,} "
            "expected bytes for this attempt"
        )


def download(
    url: str,
    dest: Path,
    *,
    timeout: int = 120,
    max_retries: int = 10,
    backoff_seconds: float = 10.0,
) -> None:
    """Download with resume-on-reconnect and retries -- Zenodo's large-file
    serving is prone to transient 502/503/504s and mid-transfer disconnects,
    independent of anything this script controls."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            _attempt_download(url, tmp, timeout=timeout)
            tmp.rename(dest)
            return
        except (HTTPError, URLError, TimeoutError, ConnectionError, OSError) as exc:
            last_error = exc
            resumed = tmp.stat().st_size if tmp.exists() else 0
            wait = backoff_seconds * attempt
            print(
                f"  attempt {attempt}/{max_retries} failed ({exc!r}); "
                f"{resumed:,} bytes so far; retrying in {wait:.0f}s"
            )
            if attempt < max_retries:
                time.sleep(wait)

    raise SystemExit(
        f"Download failed after {max_retries} attempts: {last_error!r}. "
        "This is Zenodo intermittently returning 502/503/504 on this large file, not "
        "a bug in this script -- the partial download was kept at "
        f"{tmp} so re-running the same command resumes instead of restarting from 0."
    )


def extract(archive_path: Path, extract_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path} -> {extract_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(extract_dir)


def download_from_huggingface(extract_dir: Path, *, repo_id: str = HF_DATASET_REPO) -> None:
    """Fetch shawhin/HDFS_v1_blocks and materialize it as extracted/{HDFS.log,anomaly_label.csv}.

    All three of its splits (train/dev/test) are merged into one unlabeled-split
    pair of files -- src.data.hdfs.build_dataset does its own trace-level
    80/10/10 split (seed 42), so this dataset's own pre-made split is not used.
    """
    try:
        from datasets import concatenate_datasets, load_dataset
    except ImportError as exc:
        raise SystemExit(
            "The huggingface source requires the `datasets` package (pip install datasets, "
            "or `pip install -r requirements.txt`)."
        ) from exc

    print(f"Downloading {repo_id} from Hugging Face (splits: {', '.join(HF_SPLITS)})")
    splits = [load_dataset(repo_id, split=split) for split in HF_SPLITS]
    merged = concatenate_datasets(splits)
    print(f"  {len(merged)} blocks across all splits")

    extract_dir.mkdir(parents=True, exist_ok=True)
    log_path = extract_dir / "HDFS.log"
    label_path = extract_dir / "anomaly_label.csv"

    with log_path.open("w", encoding="utf-8") as log_file, label_path.open(
        "w", encoding="utf-8", newline=""
    ) as label_file:
        writer = csv.writer(label_file)
        writer.writerow(["BlockId", "Label"])
        for row in merged:
            block_id, text, label = row["block_id"], row["text"], row["label"]
            log_file.write(text)
            if not text.endswith("\n"):
                log_file.write("\n")
            writer.writerow([block_id, "Anomaly" if label else "Normal"])

    print(f"Wrote {log_path} and {label_path}")


def verify_members(extract_dir: Path, required: list[str] = REQUIRED_MEMBERS) -> list[str]:
    missing = []
    for name in required:
        if not any(extract_dir.rglob(name)):
            missing.append(name)
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and extract LogHub HDFS_v1")
    parser.add_argument(
        "--source", choices=("huggingface", "zenodo"), default="huggingface",
        help=(
            "'huggingface' (default): shawhin/HDFS_v1_blocks, same underlying data, "
            "reliable in practice. 'zenodo': the official archive "
            f"({DEFAULT_URL}) -- prefer this for provenance-sensitive work, but it has been "
            "unreliable (502/503/504, mid-transfer disconnects) even with retries."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/hdfs"))
    parser.add_argument("--url", default=DEFAULT_URL, help="Zenodo source only: override the archive URL if it moves")
    parser.add_argument("--hf-repo", default=HF_DATASET_REPO, help="Hugging Face source only: override the dataset repo")
    parser.add_argument("--skip-checksum", action="store_true", help="Zenodo source only: skip MD5 verification")
    parser.add_argument("--force", action="store_true", help="Discard any partial/existing download and start over")
    parser.add_argument("--max-retries", type=int, default=10, help="Zenodo source only: retries on transient network errors")
    parser.add_argument("--timeout", type=int, default=120, help="Zenodo source only: per-attempt socket timeout in seconds")
    args = parser.parse_args()

    extract_dir = args.data_dir / "extracted"

    if args.source == "huggingface":
        if args.force:
            shutil.rmtree(extract_dir, ignore_errors=True)
        if (extract_dir / "HDFS.log").exists() and (extract_dir / "anomaly_label.csv").exists() and not args.force:
            print(f"{extract_dir} already has HDFS.log + anomaly_label.csv (use --force to re-fetch)")
        else:
            download_from_huggingface(extract_dir, repo_id=args.hf_repo)
    else:
        archive_path = args.data_dir / "HDFS_v1.zip"
        partial_path = archive_path.with_suffix(archive_path.suffix + ".part")

        if args.force:
            archive_path.unlink(missing_ok=True)
            partial_path.unlink(missing_ok=True)

        if archive_path.exists():
            print(f"Archive already present at {archive_path} (use --force to discard it and start over)")
        else:
            if partial_path.exists():
                print(f"Resuming partial download at {partial_path} ({partial_path.stat().st_size:,} bytes so far)")
            download(args.url, archive_path, timeout=args.timeout, max_retries=args.max_retries)

        if not args.skip_checksum and args.url == DEFAULT_URL:
            digest = md5sum(archive_path)
            if digest != KNOWN_MD5:
                raise SystemExit(
                    f"MD5 mismatch for {archive_path}: got {digest}, expected {KNOWN_MD5}. "
                    "The download may be truncated or the upstream file changed; re-run with --force, "
                    "or pass --skip-checksum if you have confirmed the new checksum out of band."
                )
            print(f"MD5 verified: {digest}")
        else:
            print("Skipping checksum verification")

        extract(archive_path, extract_dir)

    missing = verify_members(extract_dir)
    if missing:
        raise SystemExit(f"{extract_dir} is missing expected files: {missing}")

    print(f"HDFS_v1 ready under {extract_dir} (source: {args.source})")
    for name in REQUIRED_MEMBERS:
        matches = list(extract_dir.rglob(name))
        print(f"  {name}: {matches[0]} ({matches[0].stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
