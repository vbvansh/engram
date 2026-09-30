"""End-to-end smoke test for the PostgreSQL-canonical stack.

Sends a few real LoCoMo turns through the same path the benchmark uses
(API -> PostgreSQL outbox -> dispatcher -> Temporal -> worker -> Neo4j), shows
what was stored, then asks a few questions. It checks that the pipeline works;
it does not measure accuracy (no judge; roughly 10 model calls).

Run from the repo root (loads .env.local):  .\\scripts\\local.ps1 smoke
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psycopg
from neo4j import GraphDatabase
from psycopg.rows import dict_row

sys.path.insert(0, str(Path(__file__).resolve().parent))

from engram_client import EngramClient  # noqa: E402
from loader import load_locomo  # noqa: E402
from run_locomo import _dated, _session_pairs  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "locomo10.json"

# Two real conv-26 questions whose evidence (D1:3, D1:5) lies in the first
# three pairs, plus one hand-made adversarial probe: it was Caroline, not
# Melanie, so the correct behaviour is to refuse.
QUESTIONS = [
    ("temporal", "When did Caroline go to the LGBTQ support group?", "7 May 2023"),
    ("attribution", "What is Caroline's identity?", "Transgender woman"),
    (
        "adversarial (hand-made)",
        "When did Melanie go to the LGBTQ support group?",
        "refuse - it was Caroline",
    ),
]

DONE_STATUSES = {"COMPLETE", "GATED_SKIP", "FAILED"}


def wait_for_events(client: EngramClient, event_ids: list[str], timeout_s: float) -> dict:
    """Poll each event until its canonical write and Neo4j copy are finished."""
    last: dict[str, tuple[str, str]] = {}
    start = time.monotonic()
    while time.monotonic() - start < timeout_s:
        pending = 0
        for event_id in event_ids:
            info = client.event_status(event_id)
            state = (info["status"], info["projection_status"])
            if last.get(event_id) != state:
                print(f"  [{time.monotonic() - start:5.0f}s] {event_id[:8]}  "
                      f"write={state[0]:<11} neo4j={state[1]}")
                last[event_id] = state
            status, projection = state
            finished = status in {"GATED_SKIP", "FAILED"} or (
                status == "COMPLETE" and projection in {"INDEXED", "FAILED"}
            )
            pending += not finished
        if pending == 0:
            break
        time.sleep(3)
    return last


def show_postgres(tenant_id: str) -> tuple[int, int]:
    """Print the canonical rows written for this tenant."""
    with psycopg.connect(os.environ["ENGRAM_DATABASE_URL"], row_factory=dict_row) as conn:
        events = conn.execute(
            "SELECT event_id, status, error_message FROM events WHERE tenant_id = %s "
            "ORDER BY created_at",
            (tenant_id,),
        ).fetchall()
        nodes = conn.execute(
            "SELECT memory_type, count(*) AS n FROM memory_nodes WHERE tenant_id = %s "
            "GROUP BY memory_type ORDER BY memory_type",
            (tenant_id,),
        ).fetchall()
        claims = conn.execute(
            "SELECT s.canonical_name AS subject, c.predicate, "
            "COALESCE(o.canonical_name, c.object_value #>> '{}') AS object, "
            "c.object_type, c.status, c.confidence, c.valid_from, c.asserted_at "
            "FROM memory_claims c "
            "JOIN memory_nodes s ON s.tenant_id = c.tenant_id AND s.id = c.subject_id "
            "LEFT JOIN memory_nodes o ON o.tenant_id = c.tenant_id AND o.id = c.object_entity_id "
            "WHERE c.tenant_id = %s ORDER BY c.asserted_at, c.predicate",
            (tenant_id,),
        ).fetchall()
    print("\nPostgreSQL - events:")
    for row in events:
        error = f"  error={row['error_message'][:120]}" if row["error_message"] else ""
        print(f"  {row['event_id'][:8]}  {row['status']}{error}")
    print("PostgreSQL - memory_nodes by type:", {r["memory_type"]: r["n"] for r in nodes})
    print(f"PostgreSQL - memory_claims ({len(claims)}):")
    for c in claims:
        valid = c["valid_from"].date().isoformat() if c["valid_from"] else "-"
        print(f"  ({c['subject']}) -[{c['predicate']}]-> ({c['object']})  "
              f"type={c['object_type']} conf={c['confidence']} valid_from={valid} "
              f"asserted_at={c['asserted_at'].date().isoformat()}")
    return sum(r["n"] for r in nodes), len(claims)


def show_neo4j(tenant_id: str) -> int:
    """Count the projected copy of this tenant's memories in Neo4j."""
    password = os.environ["NEO4J_ADMIN_PASSWORD"]
    with GraphDatabase.driver("bolt://127.0.0.1:7688", auth=("neo4j", password)) as driver:
        with driver.session() as session:
            nodes = session.run(
                "MATCH (n {tenant_id: $t}) RETURN count(n) AS n", t=tenant_id
            ).single()["n"]
            rels = session.run(
                "MATCH ()-[r {tenant_id: $t}]->() RETURN count(r) AS n", t=tenant_id
            ).single()["n"]
    print(f"\nNeo4j - nodes: {nodes}, relationships: {rels}")
    return nodes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default=os.environ.get("ENGRAM_BASE_URL", "http://127.0.0.1:8001"))
    parser.add_argument("--pairs", type=int, default=3, help="turn pairs from conv-26 session 1")
    parser.add_argument("--timeout", type=float, default=900, help="seconds to wait for ingest")
    args = parser.parse_args()

    admin = EngramClient(base_url=args.base_url, api_key=os.environ["ENGRAM_API_KEY"])
    print("API health:", admin.health().get("status"))
    admin.close()

    tenant_id = f"smoke-{datetime.now():%m%d%H%M%S}"
    client = EngramClient.create_tenant(
        base_url=args.base_url, admin_key=os.environ["ENGRAM_ADMIN_KEY"], tenant_id=tenant_id
    )
    print(f"tenant: {tenant_id}")

    conv = load_locomo(DATA)[0]  # conv-26: Caroline & Melanie
    event_ids: list[str] = []
    print(f"\nIngesting {args.pairs} pairs from {conv.sample_id}, session 1:")
    for idx, (session_id, user, asst) in enumerate(_session_pairs(conv)):
        if idx >= args.pairs:
            break
        user_text = _dated(user)
        asst_text = _dated(asst) if asst is not None else ""
        print(f"  USER      {user_text[:110]}")
        print(f"  ASSISTANT {asst_text[:110]}")
        resp = client.ingest_pair(
            session_id=session_id,
            user_content=user_text,
            assistant_content=asst_text,
            user_timestamp=user.timestamp,
            assistant_timestamp=asst.timestamp if asst is not None else user.timestamp,
            user_turn_idx=2 * idx,
            assistant_turn_idx=2 * idx + 1,
        )
        event_ids.append(resp["event_id"])

    print("\nWaiting for write + Neo4j copy (event_id  write-status  neo4j-status):")
    final = wait_for_events(client, event_ids, args.timeout)
    n_nodes, n_claims = show_postgres(tenant_id)
    n_neo = show_neo4j(tenant_id)

    print("\nQuestions:")
    answered = 0
    for label, question, gold in QUESTIONS:
        started = time.monotonic()
        try:
            result = client.query(question)
        except Exception as err:  # noqa: BLE001 - report and continue
            print(f"  [{label}] {question}\n    ERROR: {err}")
            continue
        answered += 1
        md = result.get("retrieval_metadata") or {}
        print(f"  [{label}] {question}")
        print(f"    gold:    {gold}")
        print(f"    engram:  {result['answer']}")
        print(f"    answerability={result.get('answerability')} "
              f"routes={md.get('routes_attempted')} answer_mode={md.get('answer_mode')} "
              f"stop={md.get('stop_reason')} evidence={md.get('verified_evidence')} "
              f"({time.monotonic() - started:.0f}s)")
    client.close()

    statuses = [s for s, _ in final.values()]
    checks = {
        "every event finished": len(final) == len(event_ids)
        and all(s in DONE_STATUSES for s in statuses),
        "no FAILED events": "FAILED" not in statuses,
        "canonical rows in PostgreSQL": n_nodes > 0,
        "claims extracted": n_claims > 0,
        "copy present in Neo4j": n_neo > 0,
        "all questions returned": answered == len(QUESTIONS),
    }
    print("\nSmoke checks:")
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print("Temporal UI: http://127.0.0.1:8080/namespaces/engram-local/workflows")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
