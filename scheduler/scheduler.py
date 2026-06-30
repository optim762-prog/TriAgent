"""
Async experiment scheduler for TriAgent.

Reads job matrix from experiments/matrix.yaml, seeds the SQLite DB, then runs all
jobs respecting per-host concurrency limits:
  cloud    → up to 4 parallel (rate-limit safe for DeepSeek)
  local_b  → 1 at a time (single GPU, llama.cpp occupies all VRAM)

On failure: retries up to MAX_RETRIES times, then marks the job as failed.
Resume-safe: restarting the scheduler continues from pending jobs.
"""

import asyncio
import os
import sys
import time
import yaml
import requests
from pathlib import Path

import jobdb

ROOT = Path(__file__).parent.parent
MATRIX_FILE = ROOT / "experiments" / "matrix.yaml"
DATASETS_DIR = ROOT / "datasets"
RESULTS_DIR = ROOT / "results"

MAX_RETRIES = 2

SEM_CLOUD = asyncio.Semaphore(4)
SEM_LOCAL_B = asyncio.Semaphore(1)


LCPP_B_BASE = os.environ.get("LLAMACPP_B_BASE_URL", "http://triagent-llama-cpp-server:8080/v1")
_LCPP_B_ROOT = LCPP_B_BASE.rstrip("/").removesuffix("/v1")
_notified_swap: set[str] = set()  # track which aliases we've already printed the swap prompt for


async def _wait_for_model_b(alias: str, timeout_s: int = 1800):
    """
    Block until llama-cpp-server on Machine B reports the requested model alias.
    Prints a one-time swap prompt so the user knows what to do.
    Raises RuntimeError if timeout expires.
    """
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            health = requests.get(f"{_LCPP_B_ROOT}/health", timeout=5)
            if health.status_code == 200:
                models = requests.get(f"{_LCPP_B_ROOT}/v1/models", timeout=5).json()
                if any(m["id"] == alias for m in models.get("data", [])):
                    return
        except Exception:
            pass

        if alias not in _notified_swap:
            _notified_swap.add(alias)
            print(f"\n[scheduler] *** Model swap needed: '{alias}' not loaded on Machine B. ***")
            print(f"[scheduler] Run on Machine B:")
            print(f"[scheduler]   docker compose -f infra/docker-compose.yml stop triagent-llama-cpp-server")
            print(f"[scheduler]   LCPP_ALIAS={alias} docker compose -f infra/docker-compose.yml up -d triagent-llama-cpp-server")
            print(f"[scheduler] Scheduler will resume automatically once the model is ready.\n")

        await asyncio.sleep(30)

    raise RuntimeError(f"Timeout ({timeout_s}s) waiting for model '{alias}' on Machine B")


def _sem_for(host: str) -> asyncio.Semaphore:
    if host == "local_b":
        return SEM_LOCAL_B
    return SEM_CLOUD


def _build_cmd(job: dict, run_id: str) -> list[str]:
    """Build the CLI batch command for a given job row."""
    # result folder: results/<run_id>/<backbone>__<ablation>__n100/
    out_dir = RESULTS_DIR / run_id / f"{job['backbone']}__{job['ablation']}__n100"
    out_dir.mkdir(parents=True, exist_ok=True)
    output = str(out_dir / "results.jsonl")

    base_cmd = [
        sys.executable, "-m", "cli", "batch",
        "--input", str(DATASETS_DIR / "instances.jsonl"),
        "--output", output,
        "--backbone", job["backbone"],
        "--provider", job["provider"],
        "--ablation", job["ablation"],
        "--run-id", run_id,
        "--resume",  # safe to re-run; skips already-processed instances
    ]
    return base_cmd


async def run_job(job: dict, run_id: str):
    job_id = job["id"]
    sem = _sem_for(job["host"])

    async with sem:
        # For local_b jobs: ensure the right model is loaded before proceeding.
        # If the wrong model is loaded, prints a swap prompt and waits up to 30 min.
        if job["host"] == "local_b":
            await _wait_for_model_b(job["backbone"])

        jobdb.mark_running(job_id)

        cmd = _build_cmd(dict(job), run_id)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(ROOT / "backend"),
        )
        stdout, stderr = await proc.communicate()
        ok = (proc.returncode == 0)

        if ok:
            jobdb.mark_done(job_id)
        else:
            error_tail = stderr.decode(errors="replace")[-400:]
            if job["attempts"] < MAX_RETRIES:
                jobdb.mark_retry(job_id, error_tail)
                print(f"job#{job_id} retrying ({job['attempts']+1}/{MAX_RETRIES})")
            else:
                jobdb.mark_failed(job_id, error_tail)
                print(f"job#{job_id} FAILED after {MAX_RETRIES} retries — {error_tail[:200]}")


async def main():
    # load matrix
    with open(MATRIX_FILE) as f:
        matrix_cfg = yaml.safe_load(f)

    run_id = matrix_cfg.get("run_id") or time.strftime("%Y-%m-%d")

    jobdb.init()
    jobdb.seed_from_matrix(matrix_cfg["jobs"], run_id)

    t_start = time.time()

    while True:
        pending = jobdb.get_pending()
        if not pending:
            break
        await asyncio.gather(*[run_job(dict(j), run_id) for j in pending])
        # small pause to let DB settle before re-querying
        await asyncio.sleep(1)

    elapsed_min = (time.time() - t_start) / 60
    s = jobdb.summary()
    print(f"\nall done: {s['done']}/{s['total']} jobs in {elapsed_min:.1f} min")
    if s["failed"]:
        print(f"failed cells: {s['failed_cells']}")


if __name__ == "__main__":
    asyncio.run(main())
