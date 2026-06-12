"""
Command-line interface for running TriAgent batch experiments.

Usage:
    python -m cli batch --input datasets/instances.jsonl \
        --output results/2026-06-12_smoke/gpt-4o__full__n100/results.jsonl \
        --backbone gpt-4o --provider openai --ablation full

The output directory is created automatically.
Each line in the output JSONL has the schema:
    {run_id, instance_id, backbone, provider, ablation, latency_s,
     result: ProcessResult | null, error: str | null}
A manifest.json is written alongside results.jsonl on completion.
"""

import json
import time
import uuid
import hashlib
import os
from pathlib import Path
from datetime import datetime, timezone

import click
import requests

from orchestrator import TriAgentOrchestrator
from models.schemas import ProcessRequest
from config.settings import settings


@click.group()
def cli():
    pass


@cli.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True),
              help="JSONL file with test instances (datasets/instances.jsonl)")
@click.option("--output", "output_path", required=True, type=click.Path(),
              help="Output JSONL path, e.g. results/2026-06-12_full/gpt-4o__full__n100/results.jsonl")
@click.option("--backbone", default=None,
              help="LLM model name, e.g. gpt-4o, gemma4-local. Defaults to TRIAGENT_BACKBONE env.")
@click.option("--provider", default=None,
              type=click.Choice(["openai", "groq", "deepseek", "llamacpp_a", "llamacpp_b"]),
              help="Provider for the backbone. Defaults to TRIAGENT_BACKBONE_PROVIDER env.")
@click.option("--ablation", default="full",
              type=click.Choice(["full", "no_knowledge", "no_execution"]),
              help="Ablation mode.")
@click.option("--run-id", default=None,
              help="Optional run ID for tracing. Auto-generated if not set.")
@click.option("--limit", type=int, default=None,
              help="Stop after N instances (useful for smoke tests).")
@click.option("--resume", is_flag=True, default=False,
              help="Skip instances whose instance_id already appears in the output file.")
def batch(input_path, output_path, backbone, provider, ablation, run_id, limit, resume):
    backbone = backbone or settings.backbone
    provider = provider or settings.backbone_provider
    run_id = run_id or str(uuid.uuid4())[:8]

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # collect already-processed IDs for --resume
    done_ids: set[str] = set()
    if resume and out.exists():
        with open(out) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["instance_id"])
                except Exception:
                    pass
        click.echo(f"resuming: {len(done_ids)} instances already done")

    # dataset hash for manifest
    dataset_sha = _file_sha256(input_path)

    orch = TriAgentOrchestrator(
        model=backbone,
        provider=provider,
        rules_path=settings.rules_path,
        ablation_mode=ablation,
    )

    started_at = datetime.now(timezone.utc).isoformat()
    processed = 0
    failed = 0

    with open(input_path) as fin, open(out, "a") as fout:
        for line in fin:
            if limit and processed >= limit:
                break

            inst = json.loads(line.strip())
            iid = inst["instance_id"]

            if iid in done_ids:
                continue

            req = ProcessRequest(**inst["request"])
            t0 = time.time()

            try:
                res = orch.process(req)
                row = {
                    "run_id": run_id,
                    "instance_id": iid,
                    "process_id": inst.get("process_id"),
                    "backbone": backbone,
                    "provider": provider,
                    "ablation": ablation,
                    "latency_s": round(time.time() - t0, 2),
                    "result": res.model_dump(),
                    "error": None,
                }
            except Exception as e:
                failed += 1
                row = {
                    "run_id": run_id,
                    "instance_id": iid,
                    "process_id": inst.get("process_id"),
                    "backbone": backbone,
                    "provider": provider,
                    "ablation": ablation,
                    "latency_s": round(time.time() - t0, 2),
                    "result": None,
                    "error": str(e),
                }

            fout.write(json.dumps(row) + "\n")
            fout.flush()
            processed += 1

    ended_at = datetime.now(timezone.utc).isoformat()

    manifest = {
        "run_id": run_id,
        "backbone": backbone,
        "provider": provider,
        "ablation": ablation,
        "started_at": started_at,
        "ended_at": ended_at,
        "total_processed": processed,
        "total_failed": failed,
        "dataset_sha256": dataset_sha,
        "input_path": str(input_path),
    }
    manifest_path = out.parent / "manifest.json"
    with open(manifest_path, "w") as mf:
        json.dump(manifest, mf, indent=2)

    click.echo(f"done: {processed} instances, {failed} errors → {out}")
    click.echo(f"manifest: {manifest_path}")

    _telegram_notify(
        settings.telegram_bot_token,
        settings.telegram_chat_id,
        f"✅ batch done\n"
        f"backbone: {backbone} | ablation: {ablation}\n"
        f"processed: {processed} | failed: {failed}\n"
        f"started: {started_at[:19]} → ended: {ended_at[:19]}\n"
        f"output: {out}"
        if failed == 0 else
        f"⚠️ batch done WITH ERRORS\n"
        f"backbone: {backbone} | ablation: {ablation}\n"
        f"processed: {processed} | failed: {failed}\n"
        f"output: {out}"
    )


@cli.command()
@click.option("--rules", "rules_path", required=True, type=click.Path(exists=True),
              help="JSONL file with seed rules (datasets/seed_rules.jsonl)")
@click.option("--rules-path", "store_path", default=None,
              help="Path to the FAISS rule store directory. Defaults to settings.rules_path.")
def seed(rules_path, store_path):
    """Pre-seed the FAISS rule store with canonical process rules.

    Each line in the rules JSONL must have: rule_id, trigger, content, source.
    Run this before any 'full' ablation batch to avoid cold-start retrieval failures.
    """
    from database.faiss_store import FAISSRuleStore

    store_dir = store_path or settings.rules_path
    store = FAISSRuleStore()
    store.load(store_dir)

    with open(rules_path) as f:
        rules = [json.loads(line) for line in f if line.strip()]

    for rule in rules:
        meta = rule.get("metadata", {})
        meta["source"] = rule.get("source", "manual_seed")
        meta["process_id"] = rule.get("process_id", "")
        store.add_rule(
            trigger=rule["trigger"],
            content=rule["content"],
            metadata=meta,
        )

    store.save(store_dir)
    click.echo(f"seeded {len(rules)} rules into {store_dir}")


def _telegram_notify(token: str, chat_id: str, text: str) -> None:
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception:
        pass  # never crash the batch job for a notification failure


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


if __name__ == "__main__":
    cli()
