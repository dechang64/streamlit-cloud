"""
Reading-FL Knowledge Graph Builder

Builds a knowledge graph from reading reflections using FedCtx KGBuilder.
When FedCtx is unavailable, falls back to a simple local graph.

Graph schema:
  Nodes: Book, Reader, Emotion, Campus, Genre, Excerpt
  Edges: READS, FEELS, BELONGS_TO, RESONATES_WITH, SIMILAR_TO
"""
from __future__ import annotations

import json
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class KGNode:
    """A node in the reading knowledge graph."""
    id: str
    label: str  # Book, Reader, Emotion, Campus, Genre, Excerpt
    properties: Dict[str, str] = field(default_factory=dict)


@dataclass
class KGEdge:
    """An edge in the reading knowledge graph."""
    source: str
    target: str
    label: str  # READS, FEELS, BELONGS_TO, RESONATES_WITH, SIMILAR_TO
    weight: float = 1.0


class ReadingKGBuilder:
    """
    Knowledge graph builder for reading reflections.

    FedCtx mode:
    - Delegates to unified-fl-backend KGBuilder + GraphRAG
    - Supports PageRank for importance scoring
    - Supports graph traversal for recommendation

    Local mode:
    - Simple in-memory graph for basic queries
    """

    def __init__(self, use_fedctx: bool = True):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[KGEdge] = []
        self._fedctx = None

        if use_fedctx:
            try:
                from core.grpc_client import get_fedctx_client
                self._fedctx = get_fedctx_client()
            except ImportError:
                pass

    @property
    def fedctx_available(self) -> bool:
        return self._fedctx is not None and self._fedctx.available

    def add_reading_event(
        self,
        reader_id: str,
        book_id: str,
        book_title: str,
        genre: str,
        campus: str,
        emotion: str,
        excerpt_id: str = "",
        resonance_score: float = 0.0,
    ):
        """
        Add a reading event to the knowledge graph.

        Creates nodes and edges for:
        - Reader READS Book
        - Reader FEELS Emotion
        - Book BELONGS_TO Genre
        - Reader BELONGS_TO Campus
        - If resonance_score > 0.5: Excerpt RESONATES_WITH Emotion
        """
        # Add nodes
        self._ensure_node(f"reader::{reader_id}", "Reader", {"id": reader_id, "campus": campus})
        self._ensure_node(f"book::{book_id}", "Book", {"id": book_id, "title": book_title})
        self._ensure_node(f"emotion::{emotion}", "Emotion", {"name": emotion})
        self._ensure_node(f"genre::{genre}", "Genre", {"name": genre})
        self._ensure_node(f"campus::{campus}", "Campus", {"name": campus})

        # Add edges
        self._add_edge(f"reader::{reader_id}", f"book::{book_id}", "READS")
        self._add_edge(f"reader::{reader_id}", f"emotion::{emotion}", "FEELS")
        self._add_edge(f"book::{book_id}", f"genre::{genre}", "BELONGS_TO")
        self._add_edge(f"reader::{reader_id}", f"campus::{campus}", "BELONGS_TO")

        if excerpt_id and resonance_score > 0.5:
            self._ensure_node(f"excerpt::{excerpt_id}", "Excerpt", {"id": excerpt_id})
            self._add_edge(f"excerpt::{excerpt_id}", f"emotion::{emotion}", "RESONATES_WITH", resonance_score)
            self._add_edge(f"book::{book_id}", f"excerpt::{excerpt_id}", "CONTAINS")

        # Sync to FedCtx graph store
        if self.fedctx_available:
            self._sync_to_fedctx(reader_id, book_id, book_title, genre, campus, emotion, excerpt_id, resonance_score)

    def _ensure_node(self, id: str, label: str, properties: Dict[str, str]):
        if id not in self.nodes:
            self.nodes[id] = KGNode(id=id, label=label, properties=properties)

    def _add_edge(self, source: str, target: str, label: str, weight: float = 1.0):
        self.edges.append(KGEdge(source=source, target=target, label=label, weight=weight))

    def _sync_to_fedctx(self, reader_id, book_id, book_title, genre, campus, emotion, excerpt_id, resonance_score):
        """Sync reading event to FedCtx graph store."""
        try:
            # Add nodes via vector insert with graph metadata
            self._fedctx.vector_insert(
                f"reader::{reader_id}",
                [0.0] * 128,  # placeholder vector
                metadata={"type": "Reader", "reader_id": reader_id, "campus": campus},
            )
            self._fedctx.vector_insert(
                f"book::{book_id}",
                [0.0] * 128,
                metadata={"type": "Book", "book_id": book_id, "title": book_title, "genre": genre},
            )
            self._fedctx.audit_append(
                event_type="kg_reading_event",
                node_id=campus,
                metadata={
                    "reader_id": reader_id,
                    "book_id": book_id,
                    "emotion": emotion,
                    "genre": genre,
                    "resonance_score": str(resonance_score),
                },
            )
        except Exception as e:
            logger.debug(f"FedCtx KG sync failed: {e}")

    def get_reader_journey(self, reader_id: str, max_hops: int = 3) -> Dict:
        """
        Get a reader's knowledge graph journey — what they read, felt, and who's similar.

        Returns a subgraph centered on the reader.
        """
        center = f"reader::{reader_id}"
        if center not in self.nodes:
            return {"nodes": [], "edges": []}

        # BFS from center
        visited = {center}
        frontier = [center]
        result_nodes = [self.nodes[center]]
        result_edges = []

        for _ in range(max_hops):
            next_frontier = []
            for node_id in frontier:
                for edge in self.edges:
                    if edge.source == node_id and edge.target not in visited:
                        visited.add(edge.target)
                        next_frontier.append(edge.target)
                        if edge.target in self.nodes:
                            result_nodes.append(self.nodes[edge.target])
                        result_edges.append(edge)
                    elif edge.target == node_id and edge.source not in visited:
                        visited.add(edge.source)
                        next_frontier.append(edge.source)
                        if edge.source in self.nodes:
                            result_nodes.append(self.nodes[edge.source])
                        result_edges.append(edge)
            frontier = next_frontier
            if not frontier:
                break

        return {
            "nodes": [{"id": n.id, "label": n.label, **n.properties} for n in result_nodes],
            "edges": [{"source": e.source, "target": e.target, "label": e.label, "weight": e.weight} for e in result_edges],
        }

    def get_cross_campus_resonance(self, emotion: str, min_campuses: int = 2) -> List[Dict]:
        """
        Find books/excerpts that resonate with the same emotion across campuses.

        This is the "坐忘·咖" coffee sleeve candidate generator.
        """
        emotion_node = f"emotion::{emotion}"
        # Find all readers who felt this emotion
        readers = [e.source for e in self.edges if e.target == emotion_node and e.label == "FEELS"]
        # Group by campus
        campus_readers: Dict[str, List[str]] = {}
        for r in readers:
            for e in self.edges:
                if e.source == r and e.label == "BELONGS_TO" and e.target.startswith("campus::"):
                    campus = e.target.replace("campus::", "")
                    campus_readers.setdefault(campus, []).append(r)

        if len(campus_readers) < min_campuses:
            return []

        # Find books read by these readers
        results = []
        for r in readers:
            for e in self.edges:
                if e.source == r and e.label == "READS":
                    book_id = e.target.replace("book::", "")
                    book_node = self.nodes.get(e.target)
                    if book_node:
                        results.append({
                            "book_id": book_id,
                            "title": book_node.properties.get("title", ""),
                            "emotion": emotion,
                            "n_campuses": len(campus_readers),
                            "n_readers": len(readers),
                        })

        # Deduplicate by book_id
        seen = set()
        unique = []
        for r in results:
            if r["book_id"] not in seen:
                seen.add(r["book_id"])
                unique.append(r)
        return unique

    def get_stats(self) -> Dict:
        return {
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "node_types": list(set(n.label for n in self.nodes.values())),
            "fedctx_available": self.fedctx_available,
        }
