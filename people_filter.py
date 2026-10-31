#!/usr/bin/env python3
"""Keep person-containing clips from camera recordings and remove originals safely.

The destructive path is deliberately staged per source file:
1. Decode the complete source and detect people.
2. Extract buffered clips for every merged event.
3. Validate every generated clip with ffprobe.
4. Only then unlink the source file.

A SQLite journal makes the job resumable and records every deletion.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import logging
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO


LOG = logging.getLogger("people-filter")
STOP_REQUESTED = False


def request_stop(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    LOG.warning("stop requested by signal %s; current source will not be deleted", signum)


@dataclass(frozen=True)
class MediaInfo:
    duration: float | None
    width: int
    height: int
    codec: str


@dataclass(frozen=True)
class ScanResult:
    frames: int
    hit_times: list[float]
    elapsed: float
    stderr: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_checked(cmd: list[str], timeout: float | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=True,
    )


def probe(path: Path) -> MediaInfo:
    completed = run_checked(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height:format=duration",
            "-of",
            "json",
            str(path),
        ],
        timeout=60,
    )
    data = json.loads(completed.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError("ffprobe found no video stream")
    stream = streams[0]
    raw_duration = (data.get("format") or {}).get("duration")
    duration = float(raw_duration) if raw_duration not in (None, "N/A") else None
    if duration is not None and duration <= 0:
        duration = None
    return MediaInfo(
        duration=duration,
        width=int(stream["width"]),
        height=int(stream["height"]),
        codec=str(stream.get("codec_name") or "unknown"),
    )


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS files (
            path TEXT PRIMARY KEY,
            size INTEGER NOT NULL,
            mtime_ns INTEGER NOT NULL,
            status TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            frames INTEGER DEFAULT 0,
            hits INTEGER DEFAULT 0,
            clips INTEGER DEFAULT 0,
            output_bytes INTEGER DEFAULT 0,
            elapsed_seconds REAL DEFAULT 0,
            sample_fps REAL DEFAULT 2.0,
            error TEXT
        )
        """
    )
    columns = {row[1] for row in db.execute("PRAGMA table_info(files)")}
    if "sample_fps" not in columns:
        db.execute("ALTER TABLE files ADD COLUMN sample_fps REAL DEFAULT 2.0")
    db.commit()
    return db


def journal_start(db: sqlite3.Connection, src: Path, stat: os.stat_result) -> None:
    db.execute(
        """
        INSERT INTO files(path, size, mtime_ns, status, started_at, error)
        VALUES(?, ?, ?, 'analyzing', ?, NULL)
        ON CONFLICT(path) DO UPDATE SET
            size=excluded.size,
            mtime_ns=excluded.mtime_ns,
            status='analyzing',
            started_at=excluded.started_at,
            finished_at=NULL,
            frames=0,
            hits=0,
            clips=0,
            output_bytes=0,
            elapsed_seconds=0,
            error=NULL
        """,
        (str(src), stat.st_size, stat.st_mtime_ns, utc_now()),
    )
    db.commit()


def journal_finish(
    db: sqlite3.Connection,
    src: Path,
    status: str,
    *,
    frames: int = 0,
    hits: int = 0,
    clips: int = 0,
    output_bytes: int = 0,
    elapsed: float = 0,
    sample_fps: float = 2.0,
    error: str | None = None,
) -> None:
    db.execute(
        """
        UPDATE files SET status=?, finished_at=?, frames=?, hits=?, clips=?,
            output_bytes=?, elapsed_seconds=?, sample_fps=?, error=?
        WHERE path=?
        """,
        (
            status,
            utc_now(),
            frames,
            hits,
            clips,
            output_bytes,
            elapsed,
            sample_fps,
            error,
            str(src),
        ),
    )
    db.commit()


def unchanged_terminal_entry(db: sqlite3.Connection, src: Path, stat: os.stat_result) -> bool:
    row = db.execute(
        "SELECT size, mtime_ns, status FROM files WHERE path=?", (str(src),)
    ).fetchone()
    if not row:
        return False
    old_size, old_mtime_ns, status = row
    return (
        old_size == stat.st_size
        and old_mtime_ns == stat.st_mtime_ns
        and status in {"error", "kept_dry_run"}
    )


def load_model(model_path: str, device: int) -> YOLO:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    LOG.info("loading %s on CUDA device %s (%s)", model_path, device, torch.cuda.get_device_name(device))
    model = YOLO(model_path)
    warmup = np.zeros((360, 640, 3), dtype=np.uint8)
    model.predict(
        [warmup] * 8,
        device=device,
        imgsz=(384, 640),
        conf=0.15,
        classes=[0],
        quantize="fp16",
        verbose=False,
    )
    return model


def detect_people(
    src: Path,
    model: YOLO,
    *,
    device: int,
    sample_fps: float,
    confidence: float,
    batch_size: int,
    keyframes_only: bool,
) -> ScanResult:
    width, height = 640, 360
    frame_bytes = width * height * 3
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "error",
    ]
    if keyframes_only:
        cmd += ["-skip_frame", "nokey"]
    cmd += [
        "-hwaccel",
        "cuda",
        "-hwaccel_output_format",
        "cuda",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        f"fps={sample_fps},scale_cuda={width}:{height},hwdownload,format=nv12,format=bgr24",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]

    start = time.monotonic()
    frames_seen = 0
    hit_times: list[float] = []
    with tempfile.TemporaryFile() as err_file:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file)
        assert proc.stdout is not None
        try:
            while not STOP_REQUESTED:
                wanted = frame_bytes * batch_size
                data = proc.stdout.read(wanted)
                if not data:
                    break
                if len(data) % frame_bytes:
                    raise RuntimeError(
                        f"partial raw frame from ffmpeg: {len(data) % frame_bytes} trailing bytes"
                    )
                count = len(data) // frame_bytes
                array = np.frombuffer(data, dtype=np.uint8).reshape(count, height, width, 3)
                results = model.predict(
                    [array[i] for i in range(count)],
                    device=device,
                    imgsz=(384, 640),
                    conf=confidence,
                    classes=[0],
                    quantize="fp16",
                    verbose=False,
                )
                for offset, result in enumerate(results):
                    if result.boxes is not None and len(result.boxes):
                        hit_times.append((frames_seen + offset) / sample_fps)
                frames_seen += count
                if frames_seen and frames_seen % max(int(sample_fps * 1800), 1) < count:
                    LOG.info(
                        "scan progress source=%s sampled_minutes=%.1f hits=%d",
                        src.name,
                        frames_seen / sample_fps / 60,
                        len(hit_times),
                    )
        finally:
            if STOP_REQUESTED and proc.poll() is None:
                proc.terminate()
            try:
                return_code = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                return_code = proc.wait()
            err_file.seek(0)
            stderr = err_file.read().decode("utf-8", errors="replace")[-8000:]

    if STOP_REQUESTED:
        raise InterruptedError("scan interrupted")
    if return_code != 0:
        raise RuntimeError(f"ffmpeg decode failed with code {return_code}: {stderr.strip()}")
    if frames_seen == 0:
        raise RuntimeError("ffmpeg decoded zero sampled frames")
    return ScanResult(
        frames=frames_seen,
        hit_times=hit_times,
        elapsed=time.monotonic() - start,
        stderr=stderr,
    )


def merged_intervals(
    hit_times: list[float],
    scanned_duration: float,
    *,
    pre_seconds: float,
    post_seconds: float,
) -> list[tuple[float, float]]:
    if not hit_times:
        return []
    intervals: list[list[float]] = []
    for hit in hit_times:
        start = max(0.0, hit - pre_seconds)
        end = min(scanned_duration, hit + post_seconds)
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    return [(start, end) for start, end in intervals if end - start >= 1]


def timestamp_token(seconds: float) -> str:
    value = max(0, int(seconds))
    hours, remainder = divmod(value, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}{minutes:02d}{secs:02d}"


def valid_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 4096:
        return False
    try:
        info = probe(path)
    except Exception:
        return False
    return info.width > 0 and info.height > 0


def extract_clips(
    src: Path,
    output_root: Path,
    intervals: list[tuple[float, float]],
) -> list[Path]:
    camera = src.parent.name
    destination = output_root / camera
    destination.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    required_free = src.stat().st_size + 2 * 1024**3
    available = shutil.disk_usage(output_root).free
    if available < required_free:
        raise RuntimeError(
            f"insufficient staging space: need {required_free}, available {available}"
        )

    for index, (start, end) in enumerate(intervals, start=1):
        stem = (
            f"{src.stem}__person_{index:03d}_"
            f"{timestamp_token(start)}-{timestamp_token(end)}"
        )
        final = destination / f"{stem}.mkv"
        partial = destination / f".{stem}.partial.mkv"
        if final.exists() and valid_video(final):
            outputs.append(final)
            continue
        partial.unlink(missing_ok=True)
        duration = end - start
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "error",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            "-y",
            str(partial),
        ]
        run_checked(cmd, timeout=max(300, duration * 2))
        if not valid_video(partial):
            partial.unlink(missing_ok=True)
            raise RuntimeError(f"generated clip failed validation: {partial}")
        os.replace(partial, final)
        outputs.append(final)
    return outputs


def safe_unlink(src: Path, input_root: Path) -> None:
    resolved_src = src.resolve()
    resolved_root = input_root.resolve()
    if not resolved_src.is_relative_to(resolved_root):
        raise RuntimeError(f"refusing to delete path outside input root: {resolved_src}")
    if resolved_src.suffix.lower() != ".mkv" or not resolved_src.is_file():
        raise RuntimeError(f"refusing to delete unexpected path: {resolved_src}")
    resolved_src.unlink()


def candidate_files(
    root: Path,
    db: sqlite3.Connection,
    *,
    min_age_seconds: float,
) -> list[Path]:
    cutoff = time.time() - min_age_seconds
    candidates: list[tuple[int, Path]] = []
    for path in root.glob("*/*.mkv"):
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        if stat.st_mtime > cutoff:
            continue
        if unchanged_terminal_entry(db, path, stat):
            continue
        candidates.append((stat.st_mtime_ns, path))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def process_one(
    src: Path,
    input_root: Path,
    output_root: Path,
    db: sqlite3.Connection,
    model: YOLO,
    args: argparse.Namespace,
) -> str:
    stat = src.stat()
    journal_start(db, src, stat)
    LOG.info("start source=%s size_gib=%.3f", src, stat.st_size / 1024**3)
    started = time.monotonic()
    try:
        info = probe(src)
        keyframes_only = src.parent.name in set(args.keyframe_camera)
        sample_fps = args.keyframe_fps if keyframes_only else args.sample_fps
        LOG.info(
            "media source=%s codec=%s resolution=%dx%d duration=%s sample_fps=%.3f keyframes_only=%s",
            src.name,
            info.codec,
            info.width,
            info.height,
            f"{info.duration:.1f}" if info.duration is not None else "unknown",
            sample_fps,
            keyframes_only,
        )
        scan = detect_people(
            src,
            model,
            device=args.device,
            sample_fps=sample_fps,
            confidence=args.confidence,
            batch_size=args.batch_size,
            keyframes_only=keyframes_only,
        )
        scanned_duration = scan.frames / sample_fps
        intervals = merged_intervals(
            scan.hit_times,
            scanned_duration,
            pre_seconds=args.pre_seconds,
            post_seconds=args.post_seconds,
        )
        LOG.info(
            "detected source=%s sampled_frames=%d person_frames=%d events=%d scan_seconds=%.1f",
            src.name,
            scan.frames,
            len(scan.hit_times),
            len(intervals),
            scan.elapsed,
        )
        if intervals:
            LOG.info(
                "event_intervals source=%s intervals=%s",
                src.name,
                ",".join(f"{start:.1f}-{end:.1f}" for start, end in intervals),
            )

        if args.dry_run:
            status = "kept_dry_run"
            journal_finish(
                db,
                src,
                status,
                frames=scan.frames,
                hits=len(scan.hit_times),
                clips=len(intervals),
                elapsed=time.monotonic() - started,
                sample_fps=sample_fps,
            )
            return status

        outputs = extract_clips(src, output_root, intervals) if intervals else []
        output_bytes = sum(path.stat().st_size for path in outputs)
        safe_unlink(src, input_root)
        status = "deleted_with_clips" if outputs else "deleted_no_person"
        journal_finish(
            db,
            src,
            status,
            frames=scan.frames,
            hits=len(scan.hit_times),
            clips=len(outputs),
            output_bytes=output_bytes,
            elapsed=time.monotonic() - started,
            sample_fps=sample_fps,
        )
        LOG.info(
            "complete source=%s status=%s original_gib=%.3f output_gib=%.3f reclaimed_gib=%.3f",
            src,
            status,
            stat.st_size / 1024**3,
            output_bytes / 1024**3,
            (stat.st_size - output_bytes) / 1024**3,
        )
        return status
    except InterruptedError as exc:
        journal_finish(
            db,
            src,
            "interrupted",
            elapsed=time.monotonic() - started,
            sample_fps=locals().get("sample_fps", args.sample_fps),
            error=str(exc),
        )
        raise
    except Exception as exc:
        LOG.exception("error source=%s", src)
        journal_finish(
            db,
            src,
            "error",
            elapsed=time.monotonic() - started,
            sample_fps=locals().get("sample_fps", args.sample_fps),
            error=str(exc)[-8000:],
        )
        return "error"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("/home/markov/camera-recordings"))
    parser.add_argument(
        "--output", type=Path, default=Path("/home/markov/camera-recordings-people")
    )
    parser.add_argument(
        "--state-db",
        type=Path,
        default=Path("/home/markov/Documents/space-clean/people-filter.sqlite3"),
    )
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--keyframe-fps", type=float, default=1.0)
    parser.add_argument("--keyframe-camera", action="append", default=[])
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pre-seconds", type=float, default=30.0)
    parser.add_argument("--post-seconds", type=float, default=30.0)
    parser.add_argument("--min-age-seconds", type=float, default=5400.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch-interval", type=float, default=600.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if not input_root.is_dir():
        raise SystemExit(f"input directory does not exist: {input_root}")
    if output_root == input_root or output_root.is_relative_to(input_root):
        raise SystemExit("output must be outside input directory")
    output_root.mkdir(parents=True, exist_ok=True)

    lock_path = args.state_db.with_suffix(args.state_db.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("another people-filter process already holds the lock")

    db = init_db(args.state_db)
    model = load_model(args.model, args.device)
    processed = 0
    errors = 0

    while not STOP_REQUESTED:
        candidates = candidate_files(
            input_root,
            db,
            min_age_seconds=args.min_age_seconds,
        )
        if args.limit is not None:
            remaining = max(0, args.limit - processed)
            candidates = candidates[:remaining]
        LOG.info("queue candidates=%d processed_this_run=%d errors=%d", len(candidates), processed, errors)

        for src in candidates:
            if STOP_REQUESTED:
                break
            status = process_one(src, input_root, output_root, db, model, args)
            processed += 1
            if status == "error":
                errors += 1
            if args.limit is not None and processed >= args.limit:
                break

        if args.limit is not None and processed >= args.limit:
            break
        if not args.watch or STOP_REQUESTED:
            break
        LOG.info("queue drained; sleeping %.0f seconds", args.watch_interval)
        deadline = time.monotonic() + args.watch_interval
        while not STOP_REQUESTED and time.monotonic() < deadline:
            time.sleep(min(1.0, deadline - time.monotonic()))

    LOG.info("run finished processed=%d errors=%d stop_requested=%s", processed, errors, STOP_REQUESTED)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
