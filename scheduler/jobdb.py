"""
SQLite-backed job state store.

Schema:
  jobs(id INTEGER PK, run_id TEXT, backbone TEXT, provider TEXT, ablation TEXT,
       host TEXT, status TEXT, attempts INTEGER,
       started_at TEXT, ended_at TEXT, error TEXT)

Status lifecycle: pending → running → done | failed
On scheduler restart, all 'running' jobs are reset to 'pending' (orphan recovery).
"""

import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "results" / "jobs.sqlite"


@contextmanager
def _conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init():
    with _conn() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     TEXT NOT NULL,
                backbone   TEXT NOT NULL,
                provider   TEXT NOT NULL,
                ablation   TEXT NOT NULL,
                host       TEXT NOT NULL,   -- 'local_b', 'local_a', 'cloud'
                status     TEXT NOT NULL DEFAULT 'pending',
                attempts   INTEGER NOT NULL DEFAULT 0,
                started_at TEXT,
                ended_at   TEXT,
                error      TEXT
            )
        """)
        # reset orphaned running jobs from a previous crashed run
        con.execute("UPDATE jobs SET status='pending' WHERE status='running'")


def seed_from_matrix(matrix: list[dict], run_id: str):
    """Insert all jobs from a matrix definition list. Idempotent: skips if run_id already seeded."""
    with _conn() as con:
        existing = con.execute(
            "SELECT COUNT(*) FROM jobs WHERE run_id=?", (run_id,)
        ).fetchone()[0]
        if existing > 0:
            print(f"run_id {run_id} already seeded ({existing} jobs), skipping")
            return
        for job in matrix:
            con.execute(
                "INSERT INTO jobs (run_id, backbone, provider, ablation, host) VALUES (?,?,?,?,?)",
                (run_id, job["backbone"], job["provider"], job["ablation"], job["host"]),
            )
        print(f"seeded {len(matrix)} jobs for run_id={run_id}")


def get_pending() -> list[sqlite3.Row]:
    with _conn() as con:
        return con.execute(
            "SELECT * FROM jobs WHERE status='pending' ORDER BY id"
        ).fetchall()


def mark_running(job_id: int):
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='running', started_at=?, attempts=attempts+1 WHERE id=?",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ"), job_id),
        )


def mark_done(job_id: int):
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='done', ended_at=? WHERE id=?",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ"), job_id),
        )


def mark_retry(job_id: int, error: str):
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='pending', error=? WHERE id=?",
            (error[:2000], job_id),
        )


def mark_failed(job_id: int, error: str):
    with _conn() as con:
        con.execute(
            "UPDATE jobs SET status='failed', ended_at=?, error=? WHERE id=?",
            (time.strftime("%Y-%m-%dT%H:%M:%SZ"), error[:2000], job_id),
        )


def summary() -> dict:
    with _conn() as con:
        rows = con.execute(
            "SELECT status, COUNT(*) as n FROM jobs GROUP BY status"
        ).fetchall()
        failed_cells = con.execute(
            "SELECT backbone || '__' || ablation FROM jobs WHERE status='failed'"
        ).fetchall()
    counts = {r["status"]: r["n"] for r in rows}
    return {
        "total": sum(counts.values()),
        "done": counts.get("done", 0),
        "pending": counts.get("pending", 0),
        "running": counts.get("running", 0),
        "failed": counts.get("failed", 0),
        "failed_cells": [r[0] for r in failed_cells],
    }
