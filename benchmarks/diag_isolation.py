"""Diagnostic: are conversations in different tenants really kept apart?

The benchmark gives each LoCoMo conversation its own tenant, so this must hold
before a full run. PostgreSQL row-level security is not enforced for the table
owner Engram connects as, so separation relies on the app's own tenant filters.
Two fresh tenants each store one made-up fact; then three layers are checked:

1. Tagging -- each ingest is recorded under the tenant whose key sent it
   (the old architecture once filed everything under `_default`).
2. Search (decisive) -- the app's own Neo4j vector search, run as each tenant,
   may only return memories that PostgreSQL says belong to that tenant. The
   owner's search must return its own fact (control).
3. Answers -- a question asked in the other tenant must not mention the fact.
   Own-tenant answers are shown for information only: a refusal there is a
   retrieval-quality issue (e.g. the exact-label evidence filter), not a leak.

Costs ~10 model calls; needs the API, worker and dispatcher running.

    .\\scripts\\local.ps1 isolation
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import psycopg

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "benchmarks"))
# The search check uses Engram's own Neo4j store, so load the local config.
os.environ.setdefault("ENGRAM_CONFIG_PATH", str(REPO / "config.local.yaml"))

from engram_client import EngramClient  # noqa: E402
from envfile import load_env_file  # noqa: E402

FACTS = {
    "a": (
        "[2023-06-01T10:00:00] Priya: My neighbour Zorblax told me his favourite colour is "
        "chartreuse.",
        "[2023-06-01T10:00:00] Omar: Chartreuse? That's a bold favourite colour, Zorblax!",
    ),
    "b": (
        "[2023-06-01T10:00:00] Lena: I just adopted a cat and named her Biscuit.",
        "[2023-06-01T10:00:00] Tom: Biscuit is a lovely name for a cat, Lena!",
    ),
}
# (question, word that proves the fact was found, tenant that owns the fact)
PROBES = [
    ("What is Zorblax's favourite colour?", "chartreuse", "a"),
    ("What is the name of Lena's cat?", "biscuit", "b"),
]


def owners_of(dsn: str, memory_ids: list[str]) -> dict[str, str]:
    """memory_id -> tenant_id, according to PostgreSQL (the source of truth)."""
    with psycopg.connect(dsn) as conn:
        rows = conn.execute(
            "SELECT id::text, tenant_id FROM memory_nodes WHERE id::text = ANY(%s)",
            (memory_ids,),
        ).fetchall()
    return {str(i): str(t) for i, t in rows}


def main() -> int:
    load_env_file()
    base = os.environ.get("ENGRAM_BASE_URL", "http://127.0.0.1:8001")
    admin = os.environ.get("ENGRAM_ADMIN_KEY")
    dsn = os.environ.get("ENGRAM_DATABASE_URL")
    if not admin or not dsn:
        print("ERROR: ENGRAM_ADMIN_KEY and ENGRAM_DATABASE_URL are required")
        return 2

    stamp = time.strftime("%m%d%H%M%S")
    clients = {
        name: EngramClient.create_tenant(
            base_url=base, admin_key=admin, tenant_id=f"iso-{name}-{stamp}"
        )
        for name in FACTS
    }
    for name, client in clients.items():
        user, assistant = FACTS[name]
        client.ingest_pair(
            session_id="iso", user_content=user, assistant_content=assistant,
            user_turn_idx=0, assistant_turn_idx=1, source=f"iso-{client.tenant_id}",
        )
        print(f"tenant {client.tenant_id}: stored one fact, waiting for write + Neo4j copy...")
    for client in clients.values():
        print(f"  {client.tenant_id}: {client.wait_for_drain().summary()}")

    leaks = 0
    controls_missed = 0

    print("\n1) Tagging (which tenant each ingest was filed under)")
    with psycopg.connect(dsn) as conn:
        for client in clients.values():
            row = conn.execute(
                "SELECT tenant_id FROM events WHERE source = %s", (f"iso-{client.tenant_id}",)
            ).fetchone()
            got = row[0] if row else None
            leaks += got != client.tenant_id
            print(f"  {'PASS' if got == client.tenant_id else 'FAIL'}  "
                  f"expected {client.tenant_id}, got {got}")

    print("\n2) Search layer (who owns every memory the Neo4j search returns)")
    from engram.config import get_config
    from engram.models.embeddings import EmbeddingService
    from engram.storage.neo4j_store import Neo4jStore

    cfg = get_config()
    store = Neo4jStore(cfg.knowledge_graph)
    embed = EmbeddingService.get(cfg.gating)
    try:
        for question, _word, owner in PROBES:
            vector = embed.embed(question)
            for name, client in clients.items():
                rows = store.vector_search(vector, k=12, tenant_id=client.tenant_id)
                ids = [str(r["memory_id"]) for r in rows if r.get("memory_id")]
                owner_map = owners_of(dsn, ids)
                foreign = [i for i in ids if owner_map.get(i) != client.tenant_id]
                leaks += len(foreign)
                note = ""
                if name == owner and not ids:
                    controls_missed += 1
                    note = "  CONTROL MISSED"
                print(f"  [{client.tenant_id}] {question}: {len(ids)} result(s), "
                      f"{len(foreign)} from another tenant{note}")
    finally:
        store.close()

    print("\n3) Answers (the other tenant must not mention the fact)")
    for question, word, owner in PROBES:
        for name, client in clients.items():
            answer = client.query(question).get("answer", "")
            found = word in answer.lower()
            if name == owner:
                label = "own tenant (info)"
            else:
                label = "LEAK" if found else "no leak"
                leaks += found
            print(f"  [{client.tenant_id}] {question}\n      -> {label}: {answer[:140]}")

    for client in clients.values():
        client.close()
    if leaks:
        print(f"\nVERDICT: FAIL - {leaks} leak(s) or mis-tagged ingest(s), see above")
        return 1
    if controls_missed:
        print("\nVERDICT: INCONCLUSIVE - a tenant's own search found nothing, so the "
              "absence of leaks proves little")
        return 1
    print("\nVERDICT: PASS - tenants are kept apart (tagging, search and answers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
