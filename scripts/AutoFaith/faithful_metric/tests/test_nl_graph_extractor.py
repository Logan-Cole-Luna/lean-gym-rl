"""
test_nl_graph_extractor.py

Tests for the NL dependency-graph extractor.

Assumptions:
1. Your extractor file is named `nl_graph_extractor.py`.
2. `build_graph.py` contains the exact NLNode/NLEdge definitions:

       NLNode(id, category, statement, proof=None)
       NLEdge(source: NLNode, target: NLNode, type: EdgeType)

These tests MOCK the LLM calls, so they do not use API credits.

Run:
    pytest -q test_nl_graph_extractor.py
"""

import pytest

import graph.build_NL_graph as extractor

from graph.build_graph import (
    NLNode,
    NLEdge,
    NodeCategory,
    EdgeType,
    find_dependencies_for_NL,
    topological_sort,
)


PROOF = """
Theorem: If n is an even integer, then n^2 is even.

Proof:
Suppose n is even. Then there exists an integer k such that n = 2k.
Therefore n^2 = 4k^2 = 2(2k^2).
Since 2k^2 is an integer, n^2 is even.
"""


MOCK_ROOT_EXTRACTION = {
    "root_name": "square_of_even_is_even",
    "root_statement": "If n is an even integer, then n^2 is even.",
    "direct_dependencies": [
        {
            "name": "even integer",
            "canonical_name": "even integer",
            "category": "DEFINITION",
            "description": "An integer divisible by 2.",
            "edge_type": "STATEMENT_USES",
            "evidence": "If n is an even integer",
            "explicit": True,
            "confidence": 0.99,
        },
        {
            "name": "integer",
            "canonical_name": "integer",
            "category": "DEFINITION",
            "description": "An element of the integers.",
            "edge_type": "PROOF_USES",
            "evidence": "there exists an integer k",
            "explicit": True,
            "confidence": 0.99,
        },
    ],
}


def mock_expand_concept(client, *, model, name, description):
    """
    Background concept expansion:

        even integer
            |
            | DEFINES_AS
            v
        divisibility

    Other concepts have no additional prerequisites.
    """

    if name == "even integer":
        return {
            "concept_name": "even integer",
            "prerequisites": [
                {
                    "name": "divisibility",
                    "canonical_name": "divisibility",
                    "category": "DEFINITION",
                    "description": (
                        "For integers a and b, a divides b if there "
                        "exists an integer k such that b = ak."
                    ),
                    "edge_type": "DEFINES_AS",
                    "confidence": 0.98,
                }
            ],
        }

    return {
        "concept_name": name,
        "prerequisites": [],
    }


def test_extract_even_number_dependency_graph(monkeypatch):
    """
    Expected graph:

                       root theorem
                       /          \
                      /            \
        STATEMENT_USES              PROOF_USES
               |                         |
               v                         v
          even integer                integer
               |
               | DEFINES_AS
               v
           divisibility

    Edge convention:
        source depends on target.
    """

    # ---------------------------------------------------------
    # Mock the LLM. No real API request is made.
    # ---------------------------------------------------------

    monkeypatch.setattr(
        extractor,
        "_extract_root",
        lambda client, proof_text, model: MOCK_ROOT_EXTRACTION,
    )

    monkeypatch.setattr(
        extractor,
        "_expand_concept",
        mock_expand_concept,
    )

    # The OpenAI client itself is irrelevant because both model calls
    # are mocked, but the extractor still constructs it.
    monkeypatch.setattr(
        extractor,
        "OpenAI",
        lambda api_key: object(),
    )

    # ---------------------------------------------------------
    # Extract graph.
    # ---------------------------------------------------------

    graph = extractor.extract_nl_dependency_graph(
        proof_text=PROOF,
        api_key="fake-test-key",
        model="fake-model",
        max_depth=2,
        max_nodes=20,
        expand_background=True,
    )

    # If your function returns NLGraph:
    nodes = graph.nodes
    edges = graph.edges

    # ---------------------------------------------------------
    # Basic node checks.
    # ---------------------------------------------------------

    assert len(nodes) == 4

    root = graph.node_by_id(graph.root_id)

    assert root.category == NodeCategory.THEOREM
    assert root.proof == PROOF

    # Your exact NLNode definition does not contain `name`, so identify
    # concept nodes by their mathematical statements.
    even_node = next(
        node
        for node in nodes
        if "divisible by 2" in node.statement
    )

    integer_node = next(
        node
        for node in nodes
        if node.statement == "An element of the integers."
    )

    divisibility_node = next(
        node
        for node in nodes
        if "a divides b" in node.statement
    )

    assert even_node.category == NodeCategory.DEFINITION
    assert integer_node.category == NodeCategory.DEFINITION
    assert divisibility_node.category == NodeCategory.DEFINITION

    # Only the root should store the whole supplied proof.
    assert even_node.proof is None
    assert integer_node.proof is None
    assert divisibility_node.proof is None

    # ---------------------------------------------------------
    # IMPORTANT: enforce your NLEdge representation.
    #
    # source and target must be NLNode objects, NOT integer IDs.
    # ---------------------------------------------------------

    for edge in edges:
        assert isinstance(edge.source, NLNode)
        assert isinstance(edge.target, NLNode)

    # ---------------------------------------------------------
    # Check exact dependency edges.
    # ---------------------------------------------------------

    assert NLEdge(
        source=root,
        target=even_node,
        type=EdgeType.STATEMENT_USES,
    ) in edges

    assert NLEdge(
        source=root,
        target=integer_node,
        type=EdgeType.PROOF_USES,
    ) in edges

    assert NLEdge(
        source=even_node,
        target=divisibility_node,
        type=EdgeType.DEFINES_AS,
    ) in edges

    # ---------------------------------------------------------
    # Check your helper function.
    # ---------------------------------------------------------

    tuple_graph = (nodes, edges)

    root_dependencies = find_dependencies_for_NL(
        root,
        tuple_graph,
    )

    assert set(root_dependencies) == {
        even_node,
        integer_node,
    }

    even_dependencies = find_dependencies_for_NL(
        even_node,
        tuple_graph,
    )

    assert even_dependencies == [
        divisibility_node
    ]

    # ---------------------------------------------------------
    # Check dependency-first topological ordering.
    # ---------------------------------------------------------

    sorted_nodes = topological_sort(
        nodes,
        edges,
    )

    positions = {
        node.id: i
        for i, node in enumerate(sorted_nodes)
    }

    # Dependency must occur BEFORE the node depending on it.
    assert positions[divisibility_node.id] < positions[even_node.id]
    assert positions[even_node.id] < positions[root.id]
    assert positions[integer_node.id] < positions[root.id]


def test_explicit_only_does_not_expand_background(monkeypatch):
    """
    With expand_background=False, divisibility should not be inferred.
    """

    monkeypatch.setattr(
        extractor,
        "_extract_root",
        lambda client, proof_text, model: MOCK_ROOT_EXTRACTION,
    )

    monkeypatch.setattr(
        extractor,
        "OpenAI",
        lambda api_key: object(),
    )

    # Fail loudly if background expansion is accidentally called.
    def should_not_be_called(*args, **kwargs):
        raise AssertionError(
            "_expand_concept() should not be called "
            "when expand_background=False"
        )

    monkeypatch.setattr(
        extractor,
        "_expand_concept",
        should_not_be_called,
    )

    graph = extractor.extract_nl_dependency_graph(
        proof_text=PROOF,
        api_key="fake-test-key",
        model="fake-model",
        max_depth=2,
        expand_background=False,
    )

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2

    assert {
        edge.type
        for edge in graph.edges
    } == {
        EdgeType.STATEMENT_USES,
        EdgeType.PROOF_USES,
    }


def test_empty_proof_is_rejected():
    with pytest.raises(
        ValueError,
        match="proof_text cannot be empty",
    ):
        extractor.extract_nl_dependency_graph(
            proof_text="",
            api_key="fake-test-key",
        )
