"""Neo4j knowledge graph store (§6).

Separate writer and reader drivers per §12.3 (on Enterprise; Community
collapses to one admin user). Every node carries a `tenant_id` property
and every read path filters by the ambient tenant context — tenants
cannot accidentally observe each other's data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from neo4j import Driver, GraphDatabase

from engram.config import KnowledgeGraphConfig
from engram.tenancy import current_tenant_id

log = logging.getLogger(__name__)


INDEX_STATEMENTS = [
    # L0 embedding vector index (§6.6). 384 dims for BGE-Small.
    """
    CREATE VECTOR INDEX l0_idx IF NOT EXISTS
    FOR (n:Node) ON (n.l0_embedding)
    OPTIONS {
      indexConfig: {
        `vector.dimensions`: 384,
        `vector.similarity_function`: 'cosine'
      }
    }
    """,
    # BM25 fulltext index over l0_abstract (§6.6)
    """
    CREATE FULLTEXT INDEX l0_text_idx IF NOT EXISTS
    FOR (n:Node) ON EACH [n.l0_abstract]
    """,
    # URI prefix index for filesystem-style navigation
    """
    CREATE INDEX uri_prefix_idx IF NOT EXISTS
    FOR (n:Node) ON (n.source_uri)
    """,
    # Status index for ACTIVE-only default filters
    """
    CREATE INDEX status_idx IF NOT EXISTS
    FOR (n:Node) ON (n.status)
    """,
    # Retrieval-weight index for decay-filtered queries
    """
    CREATE INDEX weight_idx IF NOT EXISTS
    FOR (n:Node) ON (n.retrieval_weight)
    """,
    # Tenant index — most read-path queries filter by tenant_id
    """
    CREATE INDEX tenant_idx IF NOT EXISTS
    FOR (n:Node) ON (n.tenant_id)
    """,
    # Composite: tenant + status (fast ACTIVE-within-tenant)
    """
    CREATE INDEX tenant_status_idx IF NOT EXISTS
    FOR (n:Node) ON (n.tenant_id, n.status)
    """,
    # Uniqueness: (tenant_id, source_uri) — same URI can reappear across tenants
    """
    CREATE CONSTRAINT node_tenant_source_uri_unique IF NOT EXISTS
    FOR (n:Node) REQUIRE (n.tenant_id, n.source_uri) IS UNIQUE
    """,
    # Canonical projection identity.  The legacy source URI remains available
    # for compatibility, but projection writes must key on stable UUIDs.
    """
    CREATE CONSTRAINT node_tenant_memory_id_unique IF NOT EXISTS
    FOR (n:Node) REQUIRE (n.tenant_id, n.memory_id) IS UNIQUE
    """,
]


class Neo4jStore:
    def __init__(self, cfg: KnowledgeGraphConfig) -> None:
        self.cfg = cfg
        self._writer: Driver | None = None
        self._reader: Driver | None = None

    # ------------------------------------------------------------------
    # Drivers
    # ------------------------------------------------------------------

    def writer(self) -> Driver:
        if self._writer is None:
            self._writer = GraphDatabase.driver(
                self.cfg.uri,
                auth=(self.cfg.writer_username, self.cfg.writer_password or ""),
            )
        return self._writer

    def reader(self) -> Driver:
        if self._reader is None:
            self._reader = GraphDatabase.driver(
                self.cfg.uri,
                auth=(self.cfg.reader_username, self.cfg.reader_password or ""),
            )
        return self._reader

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        if self._reader is not None:
            self._reader.close()

    # ------------------------------------------------------------------
    # Index / schema bootstrap
    # ------------------------------------------------------------------

    def ensure_indexes(self) -> None:
        with self.writer().session() as session:
            for stmt in INDEX_STATEMENTS:
                try:
                    session.run(stmt)
                except Exception as err:
                    # Old Neo4j versions may not support particular index syntaxes.
                    # We surface the error as a warning; `engram init` reports it.
                    log.warning("index statement failed: %s", err)

    def ping(self) -> bool:
        try:
            with self.reader().session() as session:
                session.run("RETURN 1").consume()
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Node upsert — writes after filesystem commit (§5.4.6)
    # ------------------------------------------------------------------

    def merge_node(
        self,
        *,
        source_uri: str,
        properties: dict[str, Any],
        parent_uri: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """MERGE on (tenant_id, source_uri); SET all properties; create CONTAINS edge from parent."""
        tid = tenant_id or current_tenant_id()
        props = {**properties, "tenant_id": tid}
        query = (
            "MERGE (n:Node {tenant_id: $tenant_id, source_uri: $source_uri}) "
            "SET n += $props "
            "RETURN n"
        )
        with self.writer().session() as session:
            result = session.run(
                query,
                tenant_id=tid,
                source_uri=source_uri,
                props=props,
            ).single()
            node = dict(result["n"]) if result else {}
            if parent_uri:
                session.run(
                    "MERGE (p:Node {tenant_id: $tenant_id, source_uri: $parent_uri}) "
                    "ON CREATE SET p.tenant_id = $tenant_id, p.status = 'ACTIVE' "
                    "MERGE (c:Node {tenant_id: $tenant_id, source_uri: $child_uri}) "
                    "MERGE (p)-[:CONTAINS]->(c)",
                    tenant_id=tid,
                    parent_uri=parent_uri,
                    child_uri=source_uri,
                )
        return node

    def merge_edge(
        self,
        *,
        subject_uri: str,
        object_uri: str,
        relation_label: str,
        edge_type: str = "RELATES_TO",
        properties: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> None:
        tid = tenant_id or current_tenant_id()
        props = dict(properties or {})
        props.setdefault("relation_label", relation_label)
        props.setdefault("status", "ACTIVE")
        props["tenant_id"] = tid
        query = (
            "MATCH (s:Node {tenant_id: $tenant_id, source_uri: $s_uri}) "
            "MATCH (o:Node {tenant_id: $tenant_id, source_uri: $o_uri}) "
            f"MERGE (s)-[r:{edge_type} {{relation_label: $relation_label}}]->(o) "
            "SET r += $props "
        )
        with self.writer().session() as session:
            session.run(
                query,
                tenant_id=tid,
                s_uri=subject_uri,
                o_uri=object_uri,
                relation_label=relation_label,
                props=props,
            )

    # ------------------------------------------------------------------
    # Canonical PostgreSQL projection ----------------------------------
    # ------------------------------------------------------------------

    @staticmethod
    def _projection_revision(revision: int) -> int:
        try:
            value = int(revision)
        except (TypeError, ValueError) as err:
            raise ValueError("projection revision must be an integer") from err
        if value < 0:
            raise ValueError("projection revision must be non-negative")
        return value

    def _run_projection_write(self, query: str, **parameters: Any) -> dict[str, Any] | None:
        """Execute a projection in a managed transaction.

        Neo4j's managed transaction API retries transient failures such as
        deadlocks. Bulk projection creates many relationships on the same
        entity nodes, so relying on auto-commit writes can otherwise turn a
        recoverable lock conflict into a terminal Temporal failure.
        """

        def write_one(tx: Any) -> dict[str, Any] | None:
            record = tx.run(query, **parameters).single()
            return dict(record) if record is not None else None

        with self.writer().session() as session:
            return session.execute_write(write_one)

    def upsert_memory_projection(
        self,
        *,
        memory_id: str,
        revision: int,
        properties: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Upsert a canonical memory projection with an optimistic revision guard.

        ``memory_id`` is the only identity used by this path.  A projection
        retry therefore updates the same node, while a late older outbox event
        becomes a no-op.  The caller must provide canonical data loaded from
        PostgreSQL; this method never treats a source URI as identity.
        """
        if not memory_id:
            raise ValueError("memory_id is required for projection")
        tid = tenant_id or current_tenant_id()
        projected_revision = self._projection_revision(revision)
        props = dict(properties or {})
        # These fields are controlled by the projection key/guard and must not
        # be smuggled in by a stale serialized payload.
        for key in ("tenant_id", "memory_id", "projected_revision"):
            props.pop(key, None)
        props.setdefault("status", "ACTIVE")
        query = (
            "MERGE (n:Node {tenant_id: $tenant_id, memory_id: $memory_id}) "
            "ON CREATE SET n.projected_revision = -1 "
            "WITH n, coalesce(n.projected_revision, -1) AS current_revision "
            "FOREACH (_ IN CASE WHEN current_revision <= $revision THEN [1] ELSE [] END | "
            "SET n += $props, n.tenant_id = $tenant_id, n.memory_id = $memory_id, "
            "n.projected_revision = $revision) "
            "RETURN properties(n) AS node, current_revision <= $revision AS applied, "
            "coalesce(n.projected_revision, current_revision) AS projected_revision"
        )
        row = self._run_projection_write(
            query,
            tenant_id=tid,
            memory_id=memory_id,
            revision=projected_revision,
            props=props,
        )
        if not row:
            return {"node": {}, "applied": False, "projected_revision": None}
        return {
            "node": dict(row.get("node") or {}),
            "applied": bool(row.get("applied")),
            "projected_revision": row.get("projected_revision"),
        }

    def delete_memory_projection(
        self,
        *,
        memory_id: str,
        revision: int,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Apply a revision-guarded memory delete while retaining a tombstone.

        Keeping the keyed node with ``status=DELETED`` prevents a delayed older
        upsert from resurrecting a canonical memory.  A rebuild may omit these
        tombstones; a newer upsert can safely reactivate the same identity.
        """
        if not memory_id:
            raise ValueError("memory_id is required for projection")
        tid = tenant_id or current_tenant_id()
        projected_revision = self._projection_revision(revision)
        query = (
            "MERGE (n:Node {tenant_id: $tenant_id, memory_id: $memory_id}) "
            "ON CREATE SET n.projected_revision = -1 "
            "WITH n, coalesce(n.projected_revision, -1) AS current_revision "
            "FOREACH (_ IN CASE WHEN current_revision <= $revision THEN [1] ELSE [] END | "
            "SET n.status = 'DELETED', n.tenant_id = $tenant_id, n.memory_id = $memory_id, "
            "n.projected_revision = $revision) "
            "RETURN properties(n) AS node, current_revision <= $revision AS applied, "
            "coalesce(n.projected_revision, current_revision) AS projected_revision"
        )
        row = self._run_projection_write(
            query,
            tenant_id=tid,
            memory_id=memory_id,
            revision=projected_revision,
        )
        if not row:
            return {"node": {}, "applied": False, "projected_revision": None}
        return {
            "node": dict(row.get("node") or {}),
            "applied": bool(row.get("applied")),
            "projected_revision": row.get("projected_revision"),
        }

    def upsert_claim_projection(
        self,
        *,
        claim_id: str,
        subject_memory_id: str,
        object_memory_id: str,
        predicate: str,
        revision: int,
        properties: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Upsert one stable claim relationship with a revision guard.

        Relationships are keyed by ``(tenant_id, claim_id)`` rather than by
        endpoint/predicate.  If a newer revision moves a claim, the old edge is
        removed inside the same Neo4j transaction before the new edge is
        merged.  An older revision is returned as ``applied=False``.
        """
        if not claim_id:
            raise ValueError("claim_id is required for projection")
        if not subject_memory_id or not object_memory_id:
            raise ValueError("subject_memory_id and object_memory_id are required")
        if not predicate:
            raise ValueError("predicate is required for projection")
        tid = tenant_id or current_tenant_id()
        projected_revision = self._projection_revision(revision)
        props = dict(properties or {})
        for key in (
            "tenant_id",
            "claim_id",
            "projected_revision",
            "subject_memory_id",
            "object_memory_id",
        ):
            props.pop(key, None)
        props.setdefault("predicate", predicate)
        props.setdefault("relation_label", predicate)
        props.setdefault("status", "ACTIVE")
        query = (
            "MATCH (s:Node {tenant_id: $tenant_id, memory_id: $subject_memory_id}) "
            "MATCH (o:Node {tenant_id: $tenant_id, memory_id: $object_memory_id}) "
            "CALL { "
            "WITH s, o "
            "OPTIONAL MATCH ()-[candidate]->() "
            "WHERE candidate.tenant_id = $tenant_id AND candidate.claim_id = $claim_id "
            "RETURN max(coalesce(candidate.projected_revision, -1)) AS current_revision "
            "} "
            "CALL { "
            "WITH s, o, current_revision "
            "WITH s, o, current_revision "
            "WHERE current_revision IS NULL OR current_revision <= $revision "
            "OPTIONAL MATCH ()-[old]->() "
            "WHERE old.tenant_id = $tenant_id AND old.claim_id = $claim_id "
            "DELETE old "
            "MERGE (s)-[r:RELATES_TO {tenant_id: $tenant_id, claim_id: $claim_id}]->(o) "
            "ON CREATE SET r.projected_revision = -1 "
            "WITH r, coalesce(r.projected_revision, -1) AS edge_revision "
            "FOREACH (_ IN CASE WHEN edge_revision <= $revision THEN [1] ELSE [] END | "
            "SET r += $props, r.tenant_id = $tenant_id, r.claim_id = $claim_id, "
            "r.predicate = $predicate, r.projected_revision = $revision) "
            "RETURN properties(r) AS edge, edge_revision <= $revision AS applied, "
            "coalesce(r.projected_revision, edge_revision) AS projected_revision "
            "} "
            "RETURN edge, applied, projected_revision"
        )
        row = self._run_projection_write(
            query,
            tenant_id=tid,
            claim_id=claim_id,
            subject_memory_id=subject_memory_id,
            object_memory_id=object_memory_id,
            predicate=predicate,
            revision=projected_revision,
            props=props,
        )
        if not row:
            # The guarded write subquery emits no row when a newer edge
            # already exists. Distinguish that harmless stale delivery from
            # genuinely missing endpoint nodes so Temporal does not retry an
            # already-converged projection until exhaustion.
            with self.reader().session() as session:
                existing = session.run(
                    "MATCH ()-[r]->() WHERE r.tenant_id = $tenant_id "
                    "AND r.claim_id = $claim_id "
                    "RETURN properties(r) AS edge, "
                    "r.projected_revision AS projected_revision "
                    "ORDER BY r.projected_revision DESC LIMIT 1",
                    tenant_id=tid,
                    claim_id=claim_id,
                ).single()
            if existing:
                return {
                    "edge": dict(existing.get("edge") or {}),
                    "applied": False,
                    "projected_revision": existing.get("projected_revision"),
                }
            return {"edge": {}, "applied": False, "projected_revision": None}
        return {
            "edge": dict(row.get("edge") or {}),
            "applied": bool(row.get("applied")),
            "projected_revision": row.get("projected_revision"),
        }

    def delete_claim_projection(
        self,
        *,
        claim_id: str,
        revision: int,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Delete a claim edge only when its incoming revision is current."""
        if not claim_id:
            raise ValueError("claim_id is required for projection")
        tid = tenant_id or current_tenant_id()
        projected_revision = self._projection_revision(revision)
        query = (
            "MATCH ()-[r]->() "
            "WHERE r.tenant_id = $tenant_id AND r.claim_id = $claim_id "
            "AND coalesce(r.projected_revision, -1) <= $revision "
            "DELETE r "
            "RETURN count(r) > 0 AS applied"
        )
        row = self._run_projection_write(
            query,
            tenant_id=tid,
            claim_id=claim_id,
            revision=projected_revision,
        )
        return {"applied": bool(row.get("applied"))} if row else {"applied": False}

    def upsert_hierarchy_projection(
        self,
        *,
        hierarchy_id: str,
        parent_memory_id: str,
        child_memory_id: str,
        revision: int,
        properties: dict[str, Any] | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, Any]:
        """Project a canonical hierarchy edge with a stable relation identity."""

        if not hierarchy_id or not parent_memory_id or not child_memory_id:
            raise ValueError("hierarchy_id, parent_memory_id, and child_memory_id are required")
        tid = tenant_id or current_tenant_id()
        incoming = self._projection_revision(revision)
        props = dict(properties or {})
        for key in ("tenant_id", "hierarchy_id", "projected_revision"):
            props.pop(key, None)
        query = (
            "MATCH (p:Node {tenant_id: $tenant_id, memory_id: $parent_id}) "
            "MATCH (c:Node {tenant_id: $tenant_id, memory_id: $child_id}) "
            "OPTIONAL MATCH ()-[existing:CONTAINS]->() "
            "WHERE existing.tenant_id = $tenant_id "
            "AND existing.hierarchy_id = $hierarchy_id "
            "WITH p, c, existing, coalesce(existing.projected_revision, -1) AS current "
            "FOREACH (_ IN CASE WHEN existing IS NOT NULL AND current <= $revision "
            "THEN [1] ELSE [] END | DELETE existing) "
            "WITH p, c, current WHERE current <= $revision "
            "MERGE (p)-[r:CONTAINS {tenant_id: $tenant_id, hierarchy_id: $hierarchy_id}]->(c) "
            "SET r += $props, r.projected_revision = $revision "
            "RETURN properties(r) AS edge, true AS applied, $revision AS projected_revision"
        )
        row = self._run_projection_write(
            query,
            tenant_id=tid,
            hierarchy_id=hierarchy_id,
            parent_id=parent_memory_id,
            child_id=child_memory_id,
            revision=incoming,
            props=props,
        )
        if not row:
            with self.reader().session() as session:
                existing = session.run(
                    "MATCH ()-[r:CONTAINS]->() WHERE r.tenant_id = $tenant_id "
                    "AND r.hierarchy_id = $hierarchy_id "
                    "RETURN properties(r) AS edge, "
                    "r.projected_revision AS projected_revision "
                    "ORDER BY r.projected_revision DESC LIMIT 1",
                    tenant_id=tid,
                    hierarchy_id=hierarchy_id,
                ).single()
            if existing:
                return {
                    "edge": dict(existing.get("edge") or {}),
                    "applied": False,
                    "projected_revision": existing.get("projected_revision"),
                }
            return {"edge": {}, "applied": False, "projected_revision": None}
        return {
            "edge": dict(row.get("edge") or {}),
            "applied": bool(row.get("applied")),
            "projected_revision": row.get("projected_revision"),
        }

    # Explicit aliases make the projection contract discoverable to the
    # canonical repository without exposing the legacy source-URI methods.
    project_memory = upsert_memory_projection
    project_claim = upsert_claim_projection

    # ------------------------------------------------------------------
    # Code-map replacement and read models
    # ------------------------------------------------------------------

    def replace_code_project(
        self,
        *,
        project_uri: str,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        tenant_id: str | None = None,
    ) -> None:
        """Atomically replace the structural graph for one named project."""
        tid = tenant_id or current_tenant_id()
        allowed = {"CONTAINS", "DEFINES", "IMPORTS", "CALLS", "EXTENDS"}
        if any(edge.get("edge_type") not in allowed for edge in edges):
            raise ValueError("unsupported code graph relationship type")
        with self.writer().session() as session:
            tx = session.begin_transaction()
            try:
                # Only nodes carrying this project marker are eligible.  A
                # project refresh cannot touch semantic memories or another project.
                tx.run(
                    "MATCH (n:Node {tenant_id: $tenant_id, project_uri: $project_uri}) "
                    "DETACH DELETE n",
                    tenant_id=tid,
                    project_uri=project_uri,
                )
                tx.run(
                    "UNWIND $nodes AS item "
                    "MERGE (n:Node {tenant_id: $tenant_id, source_uri: item.source_uri}) "
                    "SET n += item.properties, n.tenant_id = $tenant_id",
                    tenant_id=tid,
                    nodes=nodes,
                )
                for item in nodes:
                    parent_uri = item.get("parent_uri")
                    if not parent_uri:
                        continue
                    tx.run(
                        "MATCH (p:Node {tenant_id: $tenant_id, source_uri: $parent_uri}) "
                        "MATCH (c:Node {tenant_id: $tenant_id, source_uri: $child_uri}) "
                        "MERGE (p)-[r:CONTAINS {relation_label: 'contains'}]->(c) "
                        "SET r.status = 'ACTIVE', r.tenant_id = $tenant_id, "
                        "r.parser = 'archive-hierarchy', r.confidence = 1.0, "
                        "r.resolution = 'RESOLVED'",
                        tenant_id=tid,
                        parent_uri=parent_uri,
                        child_uri=item["source_uri"],
                    )
                for edge in edges:
                    edge_type = edge["edge_type"]
                    tx.run(
                        "MATCH (s:Node {tenant_id: $tenant_id, source_uri: $source_uri}) "
                        "MATCH (o:Node {tenant_id: $tenant_id, source_uri: $target_uri}) "
                        f"MERGE (s)-[r:{edge_type} {{relation_label: $relation_label}}]->(o) "
                        "SET r += $properties, r.tenant_id = $tenant_id",
                        tenant_id=tid,
                        source_uri=edge["subject_uri"],
                        target_uri=edge["object_uri"],
                        relation_label=edge["relation_label"],
                        properties=edge.get("properties") or {},
                    )
                tx.commit()
            except Exception:
                tx.rollback()
                raise

    def list_code_projects(self, *, tenant_id: str | None = None) -> list[dict[str, Any]]:
        tid = tenant_id or current_tenant_id()
        cypher = (
            "MATCH (n:Node {tenant_id: $tenant_id, node_type: 'PROJECT'}) "
            "OPTIONAL MATCH (n)-[:CONTAINS]->(child:Node) "
            "WITH n, n.project_uri AS own_project_uri, collect(child.project_uri) AS child_uris "
            "WITH n, coalesce(own_project_uri, child_uris[0]) AS project_uri "
            "WHERE project_uri IS NOT NULL "
            "RETURN project_uri AS source_uri, coalesce(n.display_name, project_uri) AS name, "
            "n.l0_abstract AS l0_abstract, n.created_at AS created_at "
            "ORDER BY name"
        )
        with self.reader().session() as session:
            return [dict(row) for row in session.run(cypher, tenant_id=tid)]

    def code_map(
        self,
        *,
        project_uri: str,
        depth: int = 2,
        limit: int = 200,
        tenant_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        tid = tenant_id or current_tenant_id()
        depth = max(0, min(int(depth), 6))
        limit = max(1, min(int(limit), 500))
        cypher = (
            "MATCH (root:Node {tenant_id: $tenant_id, node_type: 'PROJECT'}) "
            "OPTIONAL MATCH (root)-[:CONTAINS]->(project_child:Node) "
            "WITH root, project_child "
            "WHERE root.project_uri = $project_uri "
            "OR project_child.project_uri = $project_uri "
            f"MATCH p=(root)-[:CONTAINS|DEFINES*0..{depth}]->(n:Node) "
            "WHERE n.tenant_id = $tenant_id "
            "AND (n.node_type = 'PROJECT' OR n.project_uri = $project_uri) "
            "WITH collect(DISTINCT n)[0..$limit] AS nodes "
            "UNWIND nodes AS n WITH collect(DISTINCT n) AS nodes "
            "OPTIONAL MATCH (a)-[r:CONTAINS|DEFINES]->(b) "
            "WHERE a IN nodes AND b IN nodes AND r.tenant_id = $tenant_id "
            "RETURN [node IN nodes | {id: node.source_uri, source_uri: node.source_uri, "
            "label: coalesce(node.display_name, node.relative_path, node.source_uri), "
            "node_type: node.node_type, status: node.status, l0_abstract: node.l0_abstract, "
            "relative_path: node.relative_path, language: node.language, signature: node.signature, "
            "line_start: node.line_start, line_end: node.line_end}] AS nodes, "
            "[rel IN collect(DISTINCT r) WHERE rel IS NOT NULL | {source: startNode(rel).source_uri, "
            "target: endNode(rel).source_uri, type: type(rel), "
            "label: coalesce(rel.relation_label, type(rel)), status: rel.status}] AS edges"
        )
        with self.reader().session() as session:
            row = session.run(cypher, tenant_id=tid, project_uri=project_uri, limit=limit).single()
        if not row:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [dict(node) for node in row["nodes"]],
            "edges": [dict(edge) for edge in row["edges"]],
        }

    def code_node_details(
        self,
        *,
        source_uri: str,
        tenant_id: str | None = None,
    ) -> dict[str, Any] | None:
        tid = tenant_id or current_tenant_id()
        cypher = (
            "MATCH (n:Node {tenant_id: $tenant_id, source_uri: $source_uri}) "
            "CALL { WITH n "
            "OPTIONAL MATCH (n)-[r]->(other:Node) "
            "WHERE other.tenant_id = $tenant_id AND r.tenant_id = $tenant_id "
            "RETURN collect(DISTINCT {source_uri: other.source_uri, display_name: other.display_name, "
            "node_type: other.node_type, relation: type(r), label: r.relation_label, "
            "source_path: r.source_path, line_start: r.line_start, line_end: r.line_end, "
            "confidence: r.confidence, resolution: r.resolution}) AS outgoing } "
            "CALL { WITH n "
            "OPTIONAL MATCH (other:Node)-[r]->(n) "
            "WHERE other.tenant_id = $tenant_id AND r.tenant_id = $tenant_id "
            "RETURN collect(DISTINCT {source_uri: other.source_uri, display_name: other.display_name, "
            "node_type: other.node_type, relation: type(r), label: r.relation_label, "
            "source_path: r.source_path, line_start: r.line_start, line_end: r.line_end, "
            "confidence: r.confidence, resolution: r.resolution}) AS incoming } "
            "RETURN properties(n) AS node, outgoing, incoming"
        )
        with self.reader().session() as session:
            row = session.run(cypher, tenant_id=tid, source_uri=source_uri).single()
        if not row:
            return None
        node = dict(row["node"])
        node.pop("l0_embedding", None)
        return {
            "node": node,
            "outgoing": list(row["outgoing"] or []),
            "incoming": list(row["incoming"] or []),
        }

    # ------------------------------------------------------------------
    # Vector search (§3.2 L1) — tenant-scoped
    # ------------------------------------------------------------------

    def vector_search(
        self,
        query_embedding: list[float],
        k: int = 10,
        *,
        uri_prefix: str | None = None,
        dormant_floor: float = 0.05,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Top-K tenant-scoped nodes by cosine similarity on l0_embedding.

        Ranks only this tenant's nodes, exactly. The l0_idx vector index is
        shared by every tenant and cannot filter inside its search, so taking
        its global top-K and filtering afterwards let other tenants' similar
        memories crowd this tenant's out. ``vector.similarity.cosine`` gives
        the same normalized score as the index; ``tenant_idx`` limits the scan
        to this tenant's nodes.
        """
        tid = tenant_id or current_tenant_id()
        cypher = (
            "MATCH (node:Node) "
            "WHERE node.tenant_id = $tenant_id "
            "AND node.status = 'ACTIVE' "
            "AND node.l0_embedding IS NOT NULL "
            "AND coalesce(node.retrieval_weight, 1.0) >= $floor "
        )
        if uri_prefix:
            cypher += "AND node.source_uri STARTS WITH $prefix "
        cypher += (
            "WITH node, vector.similarity.cosine(node.l0_embedding, $vec) AS score "
            "RETURN node.source_uri AS source_uri, node.l0_abstract AS l0_abstract, "
            "score, node.memory_id AS memory_id, "
            "node.projected_revision AS projected_revision, "
            "node.node_type AS node_type ORDER BY score DESC LIMIT $k"
        )
        params: dict[str, Any] = {
            "k": k,
            "vec": query_embedding,
            "floor": dormant_floor,
            "tenant_id": tid,
        }
        if uri_prefix:
            params["prefix"] = uri_prefix
        with self.reader().session() as session:
            result = session.run(cypher, **params)
            rows = [dict(r) for r in result]
        return rows[:k]

    # ------------------------------------------------------------------
    # Read-only template runner (§8.5)
    # ------------------------------------------------------------------

    @contextmanager
    def read_session(self) -> Iterator[Any]:
        with self.reader().session() as session:
            yield session

    def run_template(
        self,
        cypher: str,
        params: dict[str, Any],
        timeout_s: int | None = None,
        *,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a parameterised Cypher template under the tenant's scope.

        Template authors reference `$tenant_id` in the WHERE clauses of node
        MATCH patterns. The runner injects the ambient tenant automatically
        so templates stay tenant-aware without per-callsite plumbing.
        """
        tid = tenant_id or current_tenant_id()
        params = {**params, "tenant_id": tid}
        with self.reader().session() as session:
            tx = session.begin_transaction(timeout=timeout_s)
            try:
                result = tx.run(cypher, **params)
                rows = [dict(r) for r in result]
                tx.commit()
                return rows
            except Exception:
                tx.rollback()
                raise

    def graph(
        self,
        *,
        root_uri: str | None = None,
        depth: int = 1,
        limit: int = 100,
        node_type: str | None = None,
        tenant_id: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Return a bounded tenant-scoped graph view for admin visualization."""
        tid = tenant_id or current_tenant_id()
        depth = max(0, min(int(depth), 4))
        limit = max(1, min(int(limit), 500))
        params: dict[str, Any]
        if root_uri:
            cypher = (
                f"MATCH p=(root:Node {{tenant_id: $tenant_id, source_uri: $root_uri}})"
                f"-[*0..{depth}]-(n:Node) "
                "WHERE n.tenant_id = $tenant_id "
                "AND ($node_type IS NULL OR n.node_type = $node_type) "
                "AND all(rel IN relationships(p) WHERE coalesce(rel.tenant_id, $tenant_id) = $tenant_id) "
                "WITH collect(DISTINCT n)[0..$limit] AS nodes "
                "UNWIND nodes AS n WITH collect(DISTINCT n) AS nodes "
                "OPTIONAL MATCH (a)-[r]-(b) "
                "WHERE a IN nodes AND b IN nodes "
                "AND r IS NOT NULL "
                "AND coalesce(r.tenant_id, $tenant_id) = $tenant_id "
                "RETURN "
                "[node IN nodes | {id: node.source_uri, "
                "label: coalesce(node.canonical_name, node.display_name, node.source_uri), "
                "source_uri: node.source_uri, node_type: node.node_type, "
                "status: node.status, l0_abstract: node.l0_abstract}] AS nodes, "
                "[rel IN collect(DISTINCT r) WHERE rel IS NOT NULL | "
                "{source: startNode(rel).source_uri, target: endNode(rel).source_uri, "
                "type: type(rel), label: coalesce(rel.relation_label, type(rel)), "
                "status: rel.status}] AS edges"
            )
            params = {
                "tenant_id": tid,
                "root_uri": root_uri,
                "node_type": node_type,
                "limit": limit,
            }
        else:
            cypher = (
                "MATCH (n:Node) WHERE n.tenant_id = $tenant_id "
                "AND ($node_type IS NULL OR n.node_type = $node_type) "
                "WITH n ORDER BY coalesce(n.created_at, '') DESC LIMIT $limit "
                "WITH collect(n) AS nodes "
                "UNWIND nodes AS n WITH collect(DISTINCT n) AS nodes "
                "OPTIONAL MATCH (a)-[r]-(b) "
                "WHERE a IN nodes AND b IN nodes "
                "AND r IS NOT NULL "
                "AND coalesce(r.tenant_id, $tenant_id) = $tenant_id "
                "RETURN "
                "[node IN nodes | {id: node.source_uri, "
                "label: coalesce(node.canonical_name, node.display_name, node.source_uri), "
                "source_uri: node.source_uri, node_type: node.node_type, "
                "status: node.status, l0_abstract: node.l0_abstract}] AS nodes, "
                "[rel IN collect(DISTINCT r) WHERE rel IS NOT NULL | "
                "{source: startNode(rel).source_uri, target: endNode(rel).source_uri, "
                "type: type(rel), label: coalesce(rel.relation_label, type(rel)), "
                "status: rel.status}] AS edges"
            )
            params = {"tenant_id": tid, "node_type": node_type, "limit": limit}
        with self.reader().session() as session:
            row = session.run(cypher, **params).single()
        if not row:
            return {"nodes": [], "edges": []}
        return {
            "nodes": [dict(n) for n in (row.get("nodes") or [])],
            "edges": [dict(e) for e in (row.get("edges") or [])],
        }
