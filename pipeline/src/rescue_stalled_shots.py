"""Rescue Bridge videos whose local polling incorrectly declared a stall."""

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import requests

from local_video_client import DEFAULT_API_URL, _request_session


SUBMISSION_RE = re.compile(r"→\s*(S\d+):\s*提交")
TASK_ID_RE = re.compile(r"task_id=([^,\s]+)")
FAILURE_RE = re.compile(r"\b(S\d+)\b.*(?:stalled|失败)", re.IGNORECASE)
MIN_BYTES = 10 * 1024


def parse_log(log_path: Path) -> dict[str, str]:
    """Return the latest submitted Bridge task id for each shot in a Phase 5 log."""
    tasks: dict[str, str] = {}
    failed_shots: set[str] = set()
    pending_shot = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        submission = SUBMISSION_RE.search(line)
        if submission:
            pending_shot = submission.group(1)
            inline_task = TASK_ID_RE.search(line)
            if inline_task:
                tasks[pending_shot] = inline_task.group(1)
                pending_shot = None
            continue
        task = TASK_ID_RE.search(line)
        if pending_shot and task:
            tasks[pending_shot] = task.group(1)
            pending_shot = None
        failure = FAILURE_RE.search(line)
        if failure:
            failed_shots.add(failure.group(1))
    # Failed submissions are checked first. All submitted tasks remain included
    # because a live log may not have reached its failure line yet.
    return {
        shot: tasks[shot]
        for shot in (*sorted(failed_shots), *tasks)
        if shot in tasks
    }


def verify_video(path: Path) -> tuple[bool, str]:
    """Require a 1280-wide video with at least 140 frames."""
    if not path.is_file() or path.stat().st_size < MIN_BYTES:
        return False, "missing or smaller than 10KB"
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,nb_frames", "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            return False, f"ffprobe failed: {result.stderr.strip()}"
        streams = json.loads(result.stdout).get("streams", [])
        if not streams:
            return False, "ffprobe found no video stream"
        width = int(streams[0].get("width", 0))
        frames = int(streams[0].get("nb_frames", 0))
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired) as exc:
        return False, f"ffprobe failed: {exc}"
    if width != 1280 or frames < 140:
        return False, f"mismatch: width={width}, nb_frames={frames}"
    return True, f"width={width}, nb_frames={frames}"


def try_download(session, api_url: str, task_id: str, destination: Path) -> tuple[bool, str]:
    """Download and verify atomically, leaving destination untouched on failure."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        response = session.get(f"{api_url}/download/{task_id}", stream=True, timeout=120)
        if response.status_code != 200:
            return False, f"download HTTP {response.status_code}"
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temp_path = Path(handle.name)
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)
        valid, detail = verify_video(temp_path)
        if not valid:
            return False, detail
        os.replace(temp_path, destination)
        temp_path = None
        return True, detail
    except requests.RequestException as exc:
        return False, f"download error: {exc}"
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def rescue(output_dir: Path, log_path: Path, session=None, api_url: str | None = None) -> dict[str, list[str]]:
    """Check logged tasks and rescue any ready, valid Bridge outputs."""
    session = session or _request_session()
    api_url = (api_url or os.environ.get("LOCAL_VIDEO_API_URL", DEFAULT_API_URL)).rstrip("/")
    summary = {"rescued": [], "pending": [], "failed": [], "skipped": []}

    for shot, task_id in parse_log(log_path).items():
        destination = output_dir / "shots" / shot / "output.mp4"
        valid, detail = verify_video(destination)
        if valid:
            print(f"SKIP {shot}: existing output is valid ({detail})")
            summary["skipped"].append(shot)
            continue

        try:
            response = session.get(f"{api_url}/status/{task_id}", timeout=15)
            response.raise_for_status()
            status = response.json().get("status", "unknown")
        except (requests.RequestException, ValueError) as exc:
            print(f"FAILED {shot}: status check error: {exc}")
            summary["failed"].append(shot)
            continue

        if status not in ("completed", "running"):
            bucket = "pending" if status in ("queued", "unknown") else "failed"
            print(f"{bucket.upper()} {shot}: Bridge status={status}")
            summary[bucket].append(shot)
            continue

        downloaded, download_detail = try_download(session, api_url, task_id, destination)
        if downloaded:
            print(f"RESCUED {shot}: task_id={task_id} ({download_detail})")
            summary["rescued"].append(shot)
        elif status == "running":
            print(f"PENDING {shot}: status=running; {download_detail}")
            summary["pending"].append(shot)
        else:
            print(f"FAILED {shot}: completed task download rejected: {download_detail}")
            summary["failed"].append(shot)

    print(
        "Summary: "
        f"rescued={len(summary['rescued'])} "
        f"still pending={len(summary['pending'])} "
        f"failed={len(summary['failed'])} "
        f"skipped={len(summary['skipped'])}"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log", required=True, type=Path)
    args = parser.parse_args()
    rescue(args.output_dir, args.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
