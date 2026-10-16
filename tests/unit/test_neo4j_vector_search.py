"""Neo4jStore.vector_search must rank only the asking tenant's memories."""

from __future__ import annotations

from typing import Any

from engram.storage.neo4j_store import Neo4jStore


class _SharedIndexGraph:
    """Answers vector_search's query the way Neo4j would.

    ``db.index.vector.queryNodes`` ranks every tenant's nodes in the shared
    l0_idx index and only then applies the WHERE filters; any other query is
    taken to filter by tenant first and rank afterwards.
    """

    def __init__(self, nodes: list[dict[str, Any]]) -> None:
        self.nodes = nodes
        self.queries: list[str] = []

    def session(self) -> _SharedIndexGraph:
        return self

    def __enter__(self) -> _SharedIndexGraph:
        return self

    def __exit__(self, *_exc: Any) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.queries.append(cypher)

        def score(node: dict[str, Any]) -> float:
            cosine = sum(a * b for a, b in zip(node["l0_embedding"], params["vec"]))
            return (1 + cosine) / 2

        def kept(node: dict[str, Any]) -> bool:
            return (
                node["tenant_id"] == params["tenant_id"]
                and node.get("status", "ACTIVE") == "ACTIVE"
                and node.get("retrieval_weight", 1.0) >= params["floor"]
                and node["source_uri"].startswith(params.get("prefix", ""))
            )

        ranked = sorted(self.nodes, key=score, reverse=True)
        if "db.index.vector.queryNodes" in cypher:
            hits = [node for node in ranked[: params["k"]] if kept(node)]
        else:
            hits = [node for node in ranked if kept(node)][: params["k"]]
        return [
            {
                "source_uri": node["source_uri"],
                "l0_abstract": node["memory_id"],
                "score": score(node),
                "memory_id": node["memory_id"],
                "projected_revision": 1,
                "node_type": "EPISODE",
            }
            for node in hits
        ]


def _node(memory_id: str, tenant_id: str, embedding: list[float], **extra: Any) -> dict[str, Any]:
    return {
        "memory_id": memory_id,
        "tenant_id": tenant_id,
        "l0_embedding": embedding,
        "source_uri": f"canonical://{tenant_id}/{memory_id}",
        **extra,
    }


def _store(nodes: list[dict[str, Any]]) -> tuple[Neo4jStore, _SharedIndexGraph]:
    graph = _SharedIndexGraph(nodes)
    store = Neo4jStore.__new__(Neo4jStore)
    store._reader = graph  # type: ignore[assignment]
    return store, graph


def test_other_tenants_closer_memories_do_not_crowd_out_this_tenants() -> None:
    # Another tenant holds 20 memories closer to the query than any of acme's.
    crowd = [_node(f"other-{i}", "other", [1.0, 0.0]) for i in range(20)]
    acme = [
        _node("acme-best", "acme", [0.8, 0.6]),
        _node("acme-second", "acme", [0.6, 0.8]),
        _node("acme-far", "acme", [0.0, 1.0]),
    ]
    store, graph = _store(crowd + acme)

    rows = store.vector_search([1.0, 0.0], k=2, tenant_id="acme")

    assert [row["memory_id"] for row in rows] == ["acme-best", "acme-second"]
    assert all("db.index.vector.queryNodes" not in cypher for cypher in graph.queries)


def test_status_weight_and_uri_prefix_filters_still_apply() -> None:
    nodes = [
        _node("retired", "acme", [1.0, 0.0], status="SUPERSEDED"),
        _node("dormant", "acme", [1.0, 0.0], retrieval_weight=0.01),
        _node("episode", "acme", [0.9, 0.1]),
        {**_node("entity", "acme", [0.5, 0.5]), "source_uri": "mem://user/entities/entity"},
    ]
    store, _graph = _store(nodes)

    rows = store.vector_search([1.0, 0.0], k=5, tenant_id="acme")
    entities = store.vector_search(
        [1.0, 0.0], k=5, tenant_id="acme", uri_prefix="mem://user/entities/"
    )

    assert [row["memory_id"] for row in rows] == ["episode", "entity"]
    assert [row["memory_id"] for row in entities] == ["entity"]
