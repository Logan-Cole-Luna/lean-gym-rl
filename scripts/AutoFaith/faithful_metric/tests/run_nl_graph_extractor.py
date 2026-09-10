"""
Run the NL graph extractor on a LaTeX proof and save the result as JSON.

Expected project structure:

faithful_metric/
├── graph/
│   ├── __init__.py
│   ├── build_graph.py
│   └── build_NL_graph.py
├── examples/
│   ├── even_sum_proof.tex
│   └── run_nl_graph_example.py
└── ...

Run from the faithful_metric project root:

    export OPENAI_API_KEY="..."
    python examples/run_nl_graph_example.py

Output:
    examples/nl_graph_example.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from enum import Enum

import graph.build_NL_graph as extractor


HERE = Path(__file__).resolve().parent
PROOF_FILE = HERE / "NL_proof_example.tex"
OUTPUT_FILE = HERE / "nl_graph_example.json"


def enum_value(value):
    if isinstance(value, Enum):
        return value.name
    return value


def node_to_dict(node):
    result = {
        "id": node.id,
        "category": enum_value(node.category),
        "statement": node.statement,
        "proof": node.proof,
    }

    for field in ("name", "depth", "evidence"):
        if hasattr(node, field):
            result[field] = getattr(node, field)

    return result


def edge_to_dict(edge):
    result = {
        "source": edge.source.id,
        "target": edge.target.id,
        "type": enum_value(edge.type),
    }

    for field in ("evidence", "explicit", "confidence"):
        if hasattr(edge, field):
            result[field] = getattr(edge, field)

    return result


def graph_to_dict(graph):
    return {
        "root_id": graph.root_id,
        "nodes": [
            node_to_dict(node)
            for node in graph.nodes
        ],
        "edges": [
            edge_to_dict(edge)
            for edge in graph.edges
        ],
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "Run:\n"
            '    export OPENAI_API_KEY="your-api-key"'
        )

    proof_text = PROOF_FILE.read_text(
        encoding="utf-8"
    )

    graph = extractor.extract_nl_dependency_graph(
        proof_text=proof_text,
        api_key=api_key,
        model=getattr(
            extractor,
            "model_name",
            "gpt-5.2",
        ),
        max_depth=2,
        max_nodes=30,
        expand_background=True,
    )

    output = graph_to_dict(graph)

    OUTPUT_FILE.write_text(
        json.dumps(
            output,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Root node: {graph.root_id}")
    print(f"Nodes: {len(graph.nodes)}")
    print(f"Edges: {len(graph.edges)}")
    print(f"JSON written to: {OUTPUT_FILE}")

    print("\nNodes:")
    for node in graph.nodes:
        name = getattr(
            node,
            "name",
            node.statement,
        )
        print(
            f"  [{node.id}] "
            f"{node.category.name}: {name}"
        )

    print("\nEdges:")
    for edge in graph.edges:
        print(
            f"  [{edge.source.id}] "
            f"--{edge.type.name}--> "
            f"[{edge.target.id}]"
        )


if __name__ == "__main__":
    main()
