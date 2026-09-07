from __future__ import annotations
from dataclasses import dataclass


import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from openai import OpenAI

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from faithful_metric.graph.build_graph import NLNode, NLEdge, FLNode, FLEdge, NodeCategory, EdgeType
else:
    from .build_graph import NLGraph, NLNode, NLEdge, NodeCategory, EdgeType

model_name = "gpt-5.2"



ROOT_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "root_name": {"type": "string"},
        "root_statement": {"type": "string"},
        "direct_dependencies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "canonical_name": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["DEFINITION", "THEOREM", "AXIOM"],
                    },
                    "description": {"type": "string"},
                    "edge_type": {
                        "type": "string",
                        "enum": [
                            "STATEMENT_USES",
                            "PROOF_USES",
                        ],
                    },
                    "evidence": {"type": ["string", "null"]},
                    "explicit": {"type": "boolean"},
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "name",
                    "canonical_name",
                    "category",
                    "description",
                    "edge_type",
                    "evidence",
                    "explicit",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "root_name",
        "root_statement",
        "direct_dependencies",
    ],
    "additionalProperties": False,
}


CONCEPT_EXPANSION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "concept_name": {"type": "string"},
        "prerequisites": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "canonical_name": {"type": "string"},
                    "category": {
                        "type": "string",
                        "enum": ["DEFINITION", "THEOREM"],
                    },
                    "description": {"type": "string"},
                    "edge_type": {
                        "type": "string",
                        "enum": [
                            "REQUIRES",
                            "DEFINES_AS",
                        ],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                    },
                },
                "required": [
                    "name",
                    "canonical_name",
                    "category",
                    "description",
                    "edge_type",
                    "confidence",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "concept_name",
        "prerequisites",
    ],
    "additionalProperties": False,
}


ROOT_INSTRUCTIONS = r"""
You extract a mathematical dependency graph from a natural-language
mathematical theorem/proof.

The graph should play the same conceptual role that LeanArchitect/Lean
Blueprint dependencies play for a formal Lean declaration, but for natural
language.

Extract ONLY mathematically meaningful high-level concepts/results.
Do not create nodes for ordinary English words, discourse markers, variables,
punctuation, arithmetic syntax, or generic proof verbs such as "therefore",
"let", "suppose", "show", or "apply".

Node categories:
- DEFINITION: a mathematical notion, structure, predicate, construction,
  object type, property, or operation whose mathematical meaning is defined.
- THEOREM: a named or clearly invoked mathematical result/lemma/proposition.

Edge types from the root theorem/proof:
- STATEMENT_USES: needed to understand the mathematical statement being proved.
- PROOF_USES: invoked/used by the proof argument, but not merely because it
  occurs in the theorem statement.

Important grounding rule:
- explicit=true only if the dependency is actually expressed, named, or
  unambiguously invoked in the supplied text.
- For explicit=true, give a short evidence span copied or nearly copied from
  the supplied text.
- Do not invent evidence.
- If a mathematical dependency is plausible background knowledge but is not
  actually present in the supplied proof, do NOT add it here. Background
  prerequisite expansion is handled separately.

Prefer canonical mathematical names. Merge synonyms. Keep the graph compact.
"""


EXPANSION_INSTRUCTIONS = r"""
You are expanding ONE mathematical concept into a high-level mathematical
concept dependency graph.

Return only DIRECT prerequisites necessary to understand the supplied concept
at the requested level. This is analogous to recursively traversing a formal
definition until reaching other meaningful mathematical concepts.

Allowed edge types:
- REQUIRES: understanding the source concept presupposes the target concept.
- DEFINES_AS: the source concept is directly characterized/constructed using
  the target concept.

Node categories:
- DEFINITION: mathematical notion/structure/predicate/construction.
- THEOREM: mathematical result needed as a conceptual prerequisite.

Rules:
- Keep dependencies high level and mathematical.
- Do not include implementation details, notation-only nodes, variables,
  generic logic words, or extremely foundational notions unless genuinely
  necessary.
- Return direct prerequisites only. Do not recursively include prerequisites
  of prerequisites in the same response.
- Do not repeat the source concept itself.
- This expansion is background mathematical knowledge, not evidence that the
  user's supplied proof explicitly mentioned the prerequisite.
- Prefer a small graph: usually 0-5 direct prerequisites.
"""



# ---------------------------------------------------------------------------
# LLM calls
# ---------------------------------------------------------------------------

def _structured_response(
    client: OpenAI,
    *,
    model: str,
    instructions: str,
    input_text: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    """
    Call OpenAI Responses API with strict JSON-schema structured output.
    """
    response = client.responses.create(
        model=model,
        instructions=instructions,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "strict": True,
                "schema": schema,
            }
        },
    )

    if not response.output_text:
        raise RuntimeError(
            "The model returned no structured text output."
        )

    return json.loads(response.output_text)


def _extract_root(
    client: OpenAI,
    proof_text: str,
    model: str,
) -> dict[str, Any]:
    return _structured_response(
        client,
        model=model,
        instructions=ROOT_INSTRUCTIONS,
        input_text=(
            "Extract the direct mathematical concept/result dependencies "
            "from this natural-language theorem/proof:\n\n"
            + proof_text
        ),
        schema_name="nl_root_dependency_extraction",
        schema=ROOT_EXTRACTION_SCHEMA,
    )


def _expand_concept(
    client: OpenAI,
    *,
    model: str,
    name: str,
    description: str,
) -> dict[str, Any]:
    return _structured_response(
        client,
        model=model,
        instructions=EXPANSION_INSTRUCTIONS,
        input_text=(
            f"Concept: {name}\n"
            f"Description/context: {description}\n\n"
            "Extract its direct mathematical prerequisites."
        ),
        schema_name="nl_concept_prerequisites",
        schema=CONCEPT_EXPANSION_SCHEMA,
    )


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _normalize_name(name: str) -> str:
    """
    Normalization is intentionally conservative. The LLM's canonical_name
    does most synonym resolution; this only handles superficial formatting.
    """
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name.casefold()


class _GraphBuilder:
    def __init__(self) -> None:
        self._next_id = 0
        self.nodes: list[NLNode] = []
        self.edges: list[NLEdge] = []
        self._name_to_id: dict[str, int] = {}
        self._edge_keys: set[tuple[int, int, str]] = set()

    def add_node(
        self,
        *,
        name: str,
        category: NodeCategory,
        statement: str,
        proof: str | None,
        depth: int,
    ) -> NLNode:
        key = _normalize_name(name)

        if key in self._name_to_id:
            existing_id = self._name_to_id[key]
            return self.nodes[existing_id]

        node = NLNode(
            id=self._next_id,
            name=name.strip(),
            category=category,
            statement=statement.strip(),
            proof=proof,
            depth=depth,
        )

        self._name_to_id[key] = node.id
        self.nodes.append(node)
        self._next_id += 1
        return node

    def add_edge(
        self,
        *,
        source: NLNode,
        target: NLNode,
        edge_type: EdgeType,
        evidence: str | None,
        explicit: bool,
        confidence: float,
    ) -> None:
        key = (
            source.id,
            target.id,
            edge_type.value,
        )

        if key in self._edge_keys:
            return

        self._edge_keys.add(key)

        self.edges.append(
            NLEdge(
                source=source,
                target=target,
                type=edge_type,
                evidence=evidence,
                explicit=explicit,
                confidence=max(
                    0.0,
                    min(1.0, float(confidence)),
                ),
            )
        )


def extract_nl_dependency_graph(
    proof_text: str,
    api_key: str,
    *,
    model: str = "gpt-5.2",
    max_depth: int = 2,
    max_nodes: int = 40,
    expand_background: bool = True,
) -> NLGraph:
    """
    Build a mathematical concept/dependency graph from a natural-language proof.

    Parameters
    ----------
    proof_text:
        Natural-language theorem/proof text.

    api_key:
        OpenAI API key.

    model:
        OpenAI model name. Kept configurable so you can swap models.

    max_depth:
        Root is depth 0. Concepts explicitly used by the proof are depth 1.
        Background prerequisite expansion proceeds until max_depth.

    max_nodes:
        Safety/cost cap on the total number of graph nodes.

    expand_background:
        If False, return only dependencies grounded in the supplied proof.
        If True, recursively infer conceptual prerequisites.

    Returns
    -------
    NLGraph
    """
    if not proof_text.strip():
        raise ValueError("proof_text cannot be empty.")

    if not api_key.strip():
        raise ValueError("api_key cannot be empty.")

    if max_depth < 1:
        raise ValueError("max_depth must be >= 1.")

    if max_nodes < 1:
        raise ValueError("max_nodes must be >= 1.")

    client = OpenAI(api_key=api_key)

    root_data = _extract_root(
        client,
        proof_text,
        model,
    )

    builder = _GraphBuilder()

    root = builder.add_node(
        name=root_data["root_name"] or "NL theorem/proof",
        category=NodeCategory.THEOREM,
        statement=root_data["root_statement"],
        proof=proof_text,
        depth=0,
    )

    # Frontier items are (node, description, depth).
    frontier: list[tuple[NLNode, str, int]] = []

    for dep in root_data["direct_dependencies"]:
        if len(builder.nodes) >= max_nodes:
            break

        child = builder.add_node(
            name=dep["canonical_name"] or dep["name"],
            category=NodeCategory(dep["category"]),
            statement=dep["description"],
            proof=None,
            depth=1,
        )

        builder.add_edge(
            source=root,
            target=child,
            edge_type=EdgeType(dep["edge_type"]),
            evidence=dep["evidence"],
            explicit=bool(dep["explicit"]),
            confidence=float(dep["confidence"]),
        )

        frontier.append(
            (
                child,
                dep["description"],
                1,
            )
        )

    # Do not use inferred background edges if the user wants only
    # text-grounded dependencies.
    if not expand_background or max_depth <= 1:
        return NLGraph(
            root_id=root.id,
            nodes=builder.nodes,
            edges=builder.edges,
        )

    # Cache expansions so equivalent concepts are not queried repeatedly.
    expanded: set[str] = set()

    while frontier and len(builder.nodes) < max_nodes:
        node, description, depth = frontier.pop(0)

        if depth >= max_depth:
            continue

        key = _normalize_name(node.name)

        if key in expanded:
            continue

        expanded.add(key)

        expansion = _expand_concept(
            client,
            model=model,
            name=node.name,
            description=description,
        )

        for prereq in expansion["prerequisites"]:
            if len(builder.nodes) >= max_nodes:
                break

            canonical_name = (
                prereq["canonical_name"]
                or prereq["name"]
            )

            # Avoid obvious self-loop after normalization.
            if (
                _normalize_name(canonical_name)
                == _normalize_name(node.name)
            ):
                continue

            child = builder.add_node(
                name=canonical_name,
                category=NodeCategory(
                    prereq["category"]
                ),
                statement=prereq["description"],
                proof=None,
                depth=depth + 1,
            )

            builder.add_edge(
                source=node,
                target=child,
                edge_type=EdgeType(
                    prereq["edge_type"]
                ),
                evidence=None,
                explicit=False,
                confidence=float(
                    prereq["confidence"]
                ),
            )

            frontier.append(
                (
                    child,
                    prereq["description"],
                    depth + 1,
                )
            )

    return NLGraph(
        root_id=root.id,
        nodes=builder.nodes,
        edges=builder.edges,
    )


# ---------------------------------------------------------------------------
# Compatibility helper for the user's existing graph code
# ---------------------------------------------------------------------------

def find_dependencies_for_NL(
    node: NLNode,
    graph: NLGraph,
) -> list[NLNode]:
    return graph.dependencies_of(node)


# ---------------------------------------------------------------------------
# CLI interface
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a natural-language mathematical dependency graph.",
    )
    
    parser.add_argument(
        "input",
        nargs="?",
        type=str,
        help="Input proof text file (or - for stdin)",
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output JSON file (default: print to stdout)",
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model name (default: gpt-4o)",
    )
    
    parser.add_argument(
        "--max-depth",
        type=int,
        default=2,
        help="Maximum depth for background expansion (default: 2)",
    )
    
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=40,
        help="Maximum number of nodes (default: 40)",
    )
    
    parser.add_argument(
        "--no-background",
        action="store_true",
        help="Do not expand background prerequisites",
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (default: OPENAI_API_KEY environment variable)",
    )
    
    args = parser.parse_args()
    
    # Read proof text
    if args.input is None or args.input == "-":
        proof_text = sys.stdin.read()
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            proof_text = f.read()
    
    # Get API key
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set and --api-key not provided", file=sys.stderr)
        sys.exit(1)
    
    # Extract graph
    print("Extracting NL dependency graph...", file=sys.stderr)
    graph = extract_nl_dependency_graph(
        proof_text,
        api_key,
        model=args.model,
        max_depth=args.max_depth,
        max_nodes=args.max_nodes,
        expand_background=not args.no_background,
    )
    
    # Output result
    output_json = graph.to_json(indent=2)
    
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output_json)
        print(f"Graph written to {args.output}", file=sys.stderr)
    else:
        print(output_json)
