

from dataclasses import asdict, dataclass
from enum import Enum
import heapq
from typing import Protocol, Sequence, TypeVar, Any
import json


class NodeCategory(Enum):
    DEFINITION = "DEFINITION"
    THEOREM = "THEOREM"
    AXIOM = "AXIOM"
    INSTANCE = "INSTANCE"
    INDUCTIVE = "INDUCTIVE"
    CONSTRUCTOR = "CONSTRUCTOR"
    RECURSOR = "RECURSOR"


class EdgeType(str, Enum):
    STATEMENT_USES = "STATEMENT_USES"
    PROOF_USES = "PROOF_USES"
    DEFINITION_USES = "DEFINITION_USES"
    REQUIRES = "REQUIRES"
    DEFINES_AS = "DEFINES_AS"


@dataclass(frozen=True)
class NLNode:
    id: int
    category: NodeCategory
    name : str
    statement: str
    definition: str
    depth : int
    proof: str | None = None
    evidence: str | None = None
   


@dataclass(frozen=True)
class NLEdge:
    source: NLNode
    target: NLNode
    type: EdgeType
    evidence: str | None = None
    explicit: bool = True
    confidence: float = 1.0


@dataclass(frozen=True)
class FLNode:
    id: int
    category: NodeCategory

    

    # Type / theorem proposition
    statement: str

    # Only for theorem / lemma
    proof: str | None

    # Only for definitions
    definition: str | None

    # Actual source .lean file
    directory: str

    # Lean module
    module_name: str

    # Entire declaration from source
    source_declaration: str | None = None
    name: str | None = None
    


@dataclass(frozen=True)
class FLEdge:
    source: FLNode
    target: FLNode
    type: EdgeType
    


@dataclass(frozen=True)
class FLGraph:
    root_id: int
    nodes: list[FLNode]
    edges: list[FLEdge]
    



@dataclass
class NLGraph:
    root_id: int
    nodes: list[NLNode]
    edges: list[NLEdge]

    def node_by_id(self, node_id: int) -> NLNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def dependencies_of(self, node: NLNode) -> list[NLNode]:
        """Return immediate outgoing dependency targets."""
        target_ids = {
            edge.target
            for edge in self.edges
            if edge.source == node.id
        }
        return [
            candidate
            for candidate in self.nodes
            if candidate.id in target_ids
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "nodes": [
                {
                    **asdict(node),
                    "category": node.category.value,
                }
                for node in self.nodes
            ],
            "edges": [
                {
                    **asdict(edge),
                    "type": edge.type.value,
                }
                for edge in self.edges
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=indent,
        )

class NodeLike(Protocol):
    id: int


class EdgeLike(Protocol):
    source: NodeLike
    target: NodeLike


NodeT = TypeVar("NodeT", bound=NodeLike)
EdgeT = TypeVar("EdgeT", bound=EdgeLike)


def topological_sort(
    nodes: Sequence[NodeT],
    edges: Sequence[EdgeT],
) -> list[NodeT]:
    """
    Return nodes in dependency-first topological order.

    Edge convention:
        edge.source depends on edge.target

    Therefore, edge.target appears before edge.source.

    Raises:
        ValueError:
            - if node IDs are not unique;
            - if an edge refers to an unknown node;
            - if the graph contains a cycle.
    """

    node_by_id: dict[int, NodeT] = {}

    for node in nodes:
        if node.id in node_by_id:
            raise ValueError(
                f"Duplicate node ID: {node.id}"
            )

        node_by_id[node.id] = node

    # dependencies[x] contains nodes that must appear before x.
    dependencies: dict[int, set[int]] = {
        node.id: set()
        for node in nodes
    }

    # dependents[x] contains nodes that depend on x.
    dependents: dict[int, set[int]] = {
        node.id: set()
        for node in nodes
    }

    for edge in edges:
        source_id = edge.source.id
        target_id = edge.target.id

        if source_id not in node_by_id:
            raise ValueError(
                f"Edge source {source_id} is not in the graph"
            )

        if target_id not in node_by_id:
            raise ValueError(
                f"Edge target {target_id} is not in the graph"
            )

        # source depends on target.
        dependencies[source_id].add(target_id)
        dependents[target_id].add(source_id)

    # Nodes with no dependencies can be processed first.
    ready = [
        node_id
        for node_id, node_dependencies in dependencies.items()
        if not node_dependencies
    ]

    # Heap makes the result deterministic by choosing smaller IDs first.
    heapq.heapify(ready)

    sorted_ids: list[int] = []

    while ready:
        completed_id = heapq.heappop(ready)
        sorted_ids.append(completed_id)

        # Completing this node may make its dependents ready.
        for dependent_id in sorted(dependents[completed_id]):
            dependencies[dependent_id].remove(completed_id)

            if not dependencies[dependent_id]:
                heapq.heappush(ready, dependent_id)

    if len(sorted_ids) != len(nodes):
        cyclic_ids = sorted(
            node_id
            for node_id, node_dependencies in dependencies.items()
            if node_dependencies
        )

        raise ValueError(
            "The graph contains a dependency cycle involving "
            f"node IDs {cyclic_ids}"
        )

    return [
        node_by_id[node_id]
        for node_id in sorted_ids
    ]


def find_dependencies_for_NL(node : NLNode, graph : tuple[list[NLNode], list[NLEdge]]) -> list[NLNode]:
    # Find all nodes that the given node depends on
    dependencies = []
    for edge in graph[1]:
        if edge.source == node:
            dependencies.append(edge.target)
    return dependencies

def find_dependencies_for_FL(node : FLNode, graph : tuple[list[FLNode], list[FLEdge]]) -> list[FLNode]:
    # Find all nodes that the given node depends on
    dependencies = []
    for edge in graph[1]:
        if edge.source == node:
            dependencies.append(edge.target)
    return dependencies





