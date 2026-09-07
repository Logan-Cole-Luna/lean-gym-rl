from __future__ import annotations
import json
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import deque
from dataclasses import MISSING, asdict, fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any
if __package__ in (None, ''):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from faithful_metric.graph.build_graph import FLGraph, FLNode, FLEdge, NodeCategory, EdgeType
else:
    from .build_graph import FLGraph, FLNode, FLEdge, NodeCategory, EdgeType
DECL_JSON_MARKER = 'AUTOFAITH_DECL_JSON:'

class FLGraphExtractionError(RuntimeError):
    pass

def _lean_identifier_is_safe(identifier: str) -> bool:
    """
    We only put source/global identifiers into generated Lean commands.

    Normal Lean declaration names such as

        Real.sqrt_sq_eq_abs
        Nat.add_comm
        sq

    pass this check. Compiler-generated names with unusual characters are
    intentionally ignored by this source-level graph.
    """
    segment = "(?:[A-Za-z_][A-Za-z0-9_']*|«[^»\\n]+»)"
    return re.fullmatch(f'{segment}(?:\\.{segment})*', identifier) is not None

def _run_decl_query(project_path: Path, environment_module: str, identifier: str, *, blueprint_module: str, namespace: str | None=None, opens: tuple[str, ...]=(), timeout: int=180, required: bool=False) -> dict[str, Any] | None:
    """
    Ask Lean to resolve ONE source-written identifier.

    `namespace` and `opens` reconstruct the most important source name-
    resolution context.

    This command never asks Lean for the proof term / definition body.
    """
    if not _lean_identifier_is_safe(identifier):
        return None
    lines = [f'import {environment_module}', f'import {blueprint_module}', '']
    clean_opens = [item for item in opens if _lean_identifier_is_safe(item)]
    if clean_opens:
        lines.append('open ' + ' '.join(clean_opens))
    if namespace:
        if not _lean_identifier_is_safe(namespace):
            namespace = None
    if namespace:
        lines.append(f'namespace {namespace}')
    lines.append(f'#autofaith_decl_info {identifier}')
    if namespace:
        lines.append(f'end {namespace}')
    source = '\n'.join(lines) + '\n'
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.lean', prefix='_autofaith_decl_', dir=project_path, encoding='utf-8', delete=False) as handle:
            handle.write(source)
            temp_path = Path(handle.name)
        process = subprocess.run(['lake', 'env', 'lean', str(temp_path)], cwd=project_path, capture_output=True, text=True, timeout=timeout)
        output = process.stdout + '\n' + process.stderr
        if process.returncode != 0:
            if required:
                raise FLGraphExtractionError(f'Lean could not resolve the requested declaration.\n\nidentifier: {identifier}\nnamespace: {namespace}\nopens: {opens}\n\n' + output)
            return None
        for line in output.splitlines():
            if DECL_JSON_MARKER in line:
                payload = line.split(DECL_JSON_MARKER, 1)[1].strip()
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise FLGraphExtractionError('Lean emitted malformed AutoFaith declaration JSON:\n' + payload) from exc
        if required:
            raise FLGraphExtractionError('Lean succeeded but emitted no AutoFaith declaration JSON.\n\n' + output)
        return None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

def _module_relative_path(module_name: str) -> Path:
    return Path(*module_name.split('.')).with_suffix('.lean')

def _lean_prefix(project_path: Path) -> Path | None:
    process = subprocess.run(['lake', 'env', 'lean', '--print-prefix'], cwd=project_path, capture_output=True, text=True)
    if process.returncode != 0:
        return None
    value = process.stdout.strip()
    return Path(value) if value else None

def resolve_lean_source_file(project_path: str | Path, module_name: str) -> Path | None:
    """
    Resolve project, Mathlib, other Lake-package, and Lean-core modules.
    """
    project_path = Path(project_path).resolve()
    relative = _module_relative_path(module_name)
    packages = project_path / '.lake' / 'packages'
    candidates: list[Path] = [project_path / relative, packages / 'mathlib' / relative, packages / 'lean4' / 'src' / 'lean' / relative]
    if packages.exists():
        for package in packages.iterdir():
            if package.is_dir():
                candidates.append(package / relative)
    prefix = _lean_prefix(project_path)
    if prefix is not None:
        candidates.extend([prefix / 'src' / 'lean' / relative, prefix / 'src' / relative])
        if module_name == 'Lake':
            candidates.append(prefix / 'src' / 'lean' / 'lake' / 'Lake.lean')
        elif module_name.startswith('Lake.'):
            tail = Path(*module_name.split('.')[1:]).with_suffix('.lean')
            candidates.append(prefix / 'src' / 'lean' / 'lake' / tail)
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None
_DECLARATION_KEYWORDS = ('theorem', 'lemma', 'def', 'abbrev', 'opaque', 'axiom', 'inductive', 'structure', 'class', 'instance')

def _short_name(full_name: str) -> str:
    return full_name.rsplit('.', 1)[-1]

def _looks_like_declaration(text: str, full_name: str) -> bool:
    short = re.escape(_short_name(full_name))
    keywords = '|'.join(_DECLARATION_KEYWORDS)
    return re.search(f"\n            (?m)^[ \\t]*\n            (?:\n                @\\[[^\\n]*\\][ \\t]*\\n[ \\t]*\n            )*\n            (?:\n                (?:protected|private|noncomputable|unsafe|partial)\n                [ \\t]+\n            )*\n            (?:{keywords})\n            [ \\t]+\n            (?:\n                [A-Za-z0-9_.'«»]+[.]\n            )*\n            {short}\n            \\b\n            ", text, re.VERBOSE) is not None

def _slice_from_lean_line_range(source: str, node: dict[str, Any]) -> str | None:
    """
    Prefer Lean's own recorded declaration range.

    We try both common line-base interpretations and validate the result
    against the declaration name.
    """
    start_line = node.get('sourceStartLine')
    end_line = node.get('sourceEndLine')
    if start_line is None or end_line is None:
        return None
    lines = source.splitlines(keepends=True)
    for base in (1, 0):
        start = start_line - base
        end = end_line - base
        if start < 0 or start >= len(lines):
            continue
        end = min(len(lines) - 1, max(start, end))
        candidate = ''.join(lines[start:end + 1]).strip()
        if _looks_like_declaration(candidate, node['name']):
            return candidate
    return None
_DECL_START_RE = re.compile("\n    ^(?P<indent>[ \\t]*)\n    (?:\n        @\\[[^\\n]*\\][ \\t]*\\n\n        (?P=indent)\n    )*\n    (?:\n        (?:protected|private|noncomputable|unsafe|partial)\n        [ \\t]+\n    )*\n    (?P<keyword>\n        theorem\n        |lemma\n        |def\n        |abbrev\n        |opaque\n        |axiom\n        |inductive\n        |structure\n        |class\n        |instance\n    )\n    [ \\t]+\n    (?P<name>[A-Za-z0-9_.'«»]+)\n    ", re.MULTILINE | re.VERBOSE)
_SCOPE_RE = re.compile("\n    ^[ \\t]*\n    (?:\n        namespace[ \\t]+\n        (?P<namespace>[A-Za-z0-9_.'«»]+)\n        |\n        section(?:[ \\t]+[A-Za-z0-9_.'«»]+)?\n        |\n        end(?:[ \\t]+[A-Za-z0-9_.'«»]+)?\n    )\n    [ \\t]*\n    (?:--.*)?\n    $\n    ", re.MULTILINE | re.VERBOSE)

def _namespace_at_offset(source: str, offset: int) -> str:
    stack: list[tuple[str, str | None]] = []
    for match in _SCOPE_RE.finditer(source, 0, offset):
        text = match.group(0).strip()
        if text.startswith('namespace '):
            stack.append(('namespace', match.group('namespace')))
        elif text.startswith('section'):
            stack.append(('section', None))
        elif text.startswith('end'):
            if stack:
                stack.pop()
    parts: list[str] = []
    for kind, value in stack:
        if kind == 'namespace' and value:
            parts.extend(value.split('.'))
    return '.'.join(parts)

def _qualified_decl_name(source: str, match: re.Match[str]) -> str:
    source_name = match.group('name')
    if '.' in source_name:
        return source_name
    namespace = _namespace_at_offset(source, match.start())
    if namespace:
        return f'{namespace}.{source_name}'
    return source_name

def _extract_declaration_fallback(source: str, full_name: str) -> str | None:
    """
    Fallback only.  Lean's source range should normally be used.

    This fallback is namespace-aware, avoiding the old `Real.sqrt` vs
    `NNReal.sqrt` confusion.
    """
    matches = [match for match in _DECL_START_RE.finditer(source) if _short_name(match.group('name')) == _short_name(full_name)]
    if not matches:
        return None
    exact = [match for match in matches if _qualified_decl_name(source, match) == full_name]
    match = exact[0] if exact else matches[0]
    start = match.start()
    line_start = source.rfind('\n', 0, start) + 1
    start_indent = len(source[line_start:start].expandtabs(4))
    end = len(source)
    for next_match in _DECL_START_RE.finditer(source, match.end()):
        next_line_start = source.rfind('\n', 0, next_match.start()) + 1
        next_indent = len(source[next_line_start:next_match.start()].expandtabs(4))
        if next_indent <= start_indent:
            end = next_match.start()
            break
    return source[start:end].strip()

def extract_source_declaration(source: str, node: dict[str, Any]) -> str | None:
    declaration = _slice_from_lean_line_range(source, node)
    if declaration is not None:
        return declaration
    return _extract_declaration_fallback(source, node['name'])

def normalize_assignment_layout(declaration: str | None) -> str | None:
    """
    Store declarations with the body beginning on the next line.

        theorem foo : P := by
          exact h

    becomes

        theorem foo : P :=
          by
            exact h

    This is a presentation/source-mining normalization only; the original
    Mathlib/project file is never modified.
    """
    if declaration is None:
        return None
    assignment = declaration.find(':=')
    if assignment == -1:
        return declaration
    before = declaration[:assignment + 2].rstrip()
    body = declaration[assignment + 2:].strip()
    if not body:
        return before
    return before + '\n' + textwrap.indent(body, '  ')

def extract_source_body(declaration: str | None, category: str) -> str | None:
    """
    Recover source-level theorem/definition content.

    THEOREM:
        `by ...` or source term after `:=`

    DEFINITION / INSTANCE:
        source RHS after `:=`
        or a `where ...` body when no `:=` occurs.
    """
    if declaration is None:
        return None
    assignment = declaration.find(':=')
    if assignment != -1:
        body = declaration[assignment + 2:].strip()
        return body or None
    if category in {'DEFINITION', 'INSTANCE'}:
        where_match = re.search('(?m)^[ \\t]*where\\b', declaration)
        if where_match is not None:
            return declaration[where_match.start():].strip()
    return None
_IDENTIFIER_RE = re.compile("\n    (?<![A-Za-z0-9_'])\n    (?P<identifier>\n        (?:[A-Za-z_][A-Za-z0-9_']*|«[^»\\n]+»)\n        (?:\n            \\.\n            (?:[A-Za-z_][A-Za-z0-9_']*|«[^»\\n]+»)\n        )*\n    )\n    ", re.VERBOSE)
_SOURCE_WORDS_TO_IGNORE = {'by', 'fun', 'let', 'in', 'if', 'then', 'else', 'match', 'with', 'where', 'do', 'return', 'show', 'from', 'have', 'suffices', 'case', 'nomatch', 'rw', 'rfl', 'simp', 'simpa', 'exact', 'apply', 'refine', 'intro', 'intros', 'constructor', 'left', 'right', 'assumption', 'contradiction', 'omega', 'linarith', 'nlinarith', 'ring', 'ring_nf', 'norm_num', 'aesop', 'grind', 'induction', 'cases', 'rcases', 'obtain', 'calc', 'all_goals', 'any_goals', 'first', 'repeat', 'try', 'focus', 'theorem', 'lemma', 'def', 'abbrev', 'opaque', 'axiom', 'instance', 'structure', 'class', 'inductive'}

def _strip_comments_and_strings(source: str) -> str:
    """
    Remove Lean line comments, nested block comments, and string contents while
    preserving newlines. This prevents theorem names mentioned only in comments
    or strings from becoming graph dependencies.
    """
    result: list[str] = []
    i = 0
    n = len(source)
    block_depth = 0
    in_string = False
    escaped = False
    while i < n:
        ch = source[i]
        nxt = source[i + 1] if i + 1 < n else ''
        if block_depth > 0:
            if ch == '/' and nxt == '-':
                block_depth += 1
                result.extend([' ', ' '])
                i += 2
                continue
            if ch == '-' and nxt == '/':
                block_depth -= 1
                result.extend([' ', ' '])
                i += 2
                continue
            result.append('\n' if ch == '\n' else ' ')
            i += 1
            continue
        if in_string:
            if escaped:
                escaped = False
                result.append(' ')
                i += 1
                continue
            if ch == '\\':
                escaped = True
                result.append(' ')
                i += 1
                continue
            if ch == '"':
                in_string = False
            result.append('\n' if ch == '\n' else ' ')
            i += 1
            continue
        if ch == '-' and nxt == '-':
            while i < n and source[i] != '\n':
                result.append(' ')
                i += 1
            continue
        if ch == '/' and nxt == '-':
            block_depth = 1
            result.extend([' ', ' '])
            i += 2
            continue
        if ch == '"':
            in_string = True
            result.append(' ')
            i += 1
            continue
        result.append(ch)
        i += 1
    return ''.join(result)

def extract_source_identifiers(body: str | None) -> list[str]:
    """
    Extract identifiers that are literally present in the source body.

    This does not claim that every token is a global declaration. Lean is asked
    to resolve each token afterwards; unresolved/local/tactic identifiers are
    discarded.
    """
    if not body:
        return []
    cleaned = _strip_comments_and_strings(body)
    result: list[str] = []
    seen: set[str] = set()
    for match in _IDENTIFIER_RE.finditer(cleaned):
        identifier = match.group('identifier')
        if identifier in _SOURCE_WORDS_TO_IGNORE:
            continue
        if identifier not in seen:
            seen.add(identifier)
            result.append(identifier)
        if '.' in identifier:
            first, *rest = identifier.split('.')
            terminal = rest[-1]
            if first and first[0].islower() and (terminal not in seen) and (terminal not in _SOURCE_WORDS_TO_IGNORE):
                seen.add(terminal)
                result.append(terminal)
    return result
_OPEN_RE = re.compile('\n    (?m)^[ \\t]*\n    open\n    [ \\t]+\n    (?!scoped\\b)\n    (?P<names>[^-\\n]+?)\n    [ \\t]*$\n    ', re.VERBOSE)

def source_open_namespaces(source: str, node: dict[str, Any]) -> tuple[str, ...]:
    """
    Approximate active `open Foo Bar` commands before the declaration.

    The fully qualified declaration namespace is reconstructed separately.
    """
    start_line = node.get('sourceStartLine')
    if start_line is None:
        prefix = source
    else:
        lines = source.splitlines(keepends=True)
        index = max(0, start_line - 1)
        prefix = ''.join(lines[:index])
    names: list[str] = []
    seen: set[str] = set()
    for match in _OPEN_RE.finditer(prefix):
        for item in match.group('names').split():
            if _lean_identifier_is_safe(item) and item not in seen:
                seen.add(item)
                names.append(item)
    return tuple(names)

def declaration_namespace(full_name: str) -> str | None:
    if '.' not in full_name:
        return None
    return full_name.rsplit('.', 1)[0]
_EXPANDABLE_CATEGORIES = {'THEOREM', 'DEFINITION', 'INSTANCE'}

def _dedupe_preserve_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result

class _GraphBuilder:

    def __init__(self, *, project_path: Path, environment_module: str, blueprint_module: str, timeout: int, max_depth: int, max_nodes: int | None) -> None:
        self.project_path = project_path
        self.environment_module = environment_module
        self.blueprint_module = blueprint_module
        self.timeout = timeout
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, str]] = []
        self.edge_keys: set[tuple[str, str, str]] = set()
        self.query_cache: dict[tuple[str, str | None, tuple[str, ...]], dict[str, Any] | None] = {}
        self.source_cache: dict[Path, str] = {}
        self.enriched: set[str] = set()

    def _query(self, identifier: str, *, namespace: str | None=None, opens: tuple[str, ...]=(), required: bool=False) -> dict[str, Any] | None:
        key = (identifier, namespace, opens)
        if key in self.query_cache:
            result = self.query_cache[key]
            if required and result is None:
                raise FLGraphExtractionError(f'Required Lean name could not be resolved: {identifier}')
            return result
        result = _run_decl_query(self.project_path, self.environment_module, identifier, blueprint_module=self.blueprint_module, namespace=namespace, opens=opens, timeout=self.timeout, required=required)
        self.query_cache[key] = result
        return result

    def _register_query_result(self, result: dict[str, Any], *, depth: int) -> str:
        node = dict(result['node'])
        name = node['name']
        if name not in self.nodes and self.max_nodes is not None and (len(self.nodes) >= self.max_nodes):
            raise FLGraphExtractionError(f'AutoFaith source traversal reached max_nodes={self.max_nodes}. Increase max_nodes or pass max_nodes=None.')
        current = self.nodes.get(name, {})
        previous_depth = current.get('depth')
        current.update(node)
        current['statementUses'] = list(result.get('statementUses', []))
        current['depth'] = depth if previous_depth is None else min(previous_depth, depth)
        self.nodes[name] = current
        return name

    def _query_full_name(self, full_name: str) -> dict[str, Any] | None:
        return self._query(full_name)

    def _add_edge(self, source: str, target: str, kind: str) -> None:
        if source == target:
            return
        key = (source, target, kind)
        if key in self.edge_keys:
            return
        self.edge_keys.add(key)
        self.edges.append({'source': source, 'target': target, 'kind': kind})

    def _source_for_node(self, node: dict[str, Any]) -> tuple[Path | None, str | None]:
        module_name = node['moduleName']
        source_path = resolve_lean_source_file(self.project_path, module_name)
        if source_path is None:
            return (None, None)
        if source_path not in self.source_cache:
            self.source_cache[source_path] = source_path.read_text(encoding='utf-8')
        return (source_path, self.source_cache[source_path])

    def _enrich_source(self, name: str) -> tuple[str | None, tuple[str, ...]]:
        """
        Fill source_path/source_declaration/proof/definition for one node.

        Return `(body, opens)` for source dependency mining.
        """
        node = self.nodes[name]
        source_path, source = self._source_for_node(node)
        node['source_path'] = str(source_path) if source_path is not None else None
        declaration = None
        opens: tuple[str, ...] = ()
        if source is not None:
            declaration = extract_source_declaration(source, node)
            opens = source_open_namespaces(source, node)
        normalized = normalize_assignment_layout(declaration)
        node['source_declaration'] = normalized
        body = extract_source_body(normalized, node['category'])
        if node['category'] == 'THEOREM':
            node['proof'] = body
            node['definition'] = None
        elif node['category'] in {'DEFINITION', 'INSTANCE'}:
            node['proof'] = None
            node['definition'] = body
        else:
            node['proof'] = None
            node['definition'] = None
        self.nodes[name] = node
        return (body, opens)

    def build(self, root_name: str) -> dict[str, Any]:
        root_result = self._query(root_name, required=True)
        assert root_result is not None
        root = self._register_query_result(root_result, depth=0)
        frontier: deque[tuple[str, int]] = deque([(root, 0)])
        queued_depth: dict[str, int] = {root: 0}
        expanded_depth: dict[str, int] = {}
        while frontier:
            current_name, current_depth = frontier.popleft()
            queued_depth.pop(current_name, None)
            previous_expanded = expanded_depth.get(current_name)
            if previous_expanded is not None and previous_expanded <= current_depth:
                continue
            expanded_depth[current_name] = current_depth
            body, opens = self._enrich_source(current_name)
            if current_depth >= self.max_depth:
                continue
            current = self.nodes[current_name]
            next_depth = current_depth + 1
            for dependency_name in current.get('statementUses', []):
                if not _lean_identifier_is_safe(dependency_name):
                    continue
                dependency_result = self._query_full_name(dependency_name)
                if dependency_result is None:
                    continue
                resolved_dependency = self._register_query_result(dependency_result, depth=next_depth)
                self._add_edge(current_name, resolved_dependency, 'STATEMENT_USES')
                old_queued = queued_depth.get(resolved_dependency)
                old_expanded = expanded_depth.get(resolved_dependency)
                should_queue = next_depth <= self.max_depth and (old_expanded is None or next_depth < old_expanded) and (old_queued is None or next_depth < old_queued)
                if should_queue:
                    frontier.append((resolved_dependency, next_depth))
                    queued_depth[resolved_dependency] = next_depth
            category = current['category']
            if category not in _EXPANDABLE_CATEGORIES or not body:
                continue
            edge_kind = 'PROOF_USES' if category == 'THEOREM' else 'DEFINITION_USES'
            source_identifiers = extract_source_identifiers(body)
            namespace = declaration_namespace(current_name)
            for source_identifier in source_identifiers:
                dependency_result = self._query(source_identifier, namespace=namespace, opens=opens)
                if dependency_result is None:
                    continue
                resolved_dependency = self._register_query_result(dependency_result, depth=next_depth)
                self._add_edge(current_name, resolved_dependency, edge_kind)
                old_queued = queued_depth.get(resolved_dependency)
                old_expanded = expanded_depth.get(resolved_dependency)
                should_queue = next_depth <= self.max_depth and (old_expanded is None or next_depth < old_expanded) and (old_queued is None or next_depth < old_queued)
                if should_queue:
                    frontier.append((resolved_dependency, next_depth))
                    queued_depth[resolved_dependency] = next_depth
        for name in list(self.nodes):
            node = self.nodes[name]
            if 'source_declaration' not in node:
                self._enrich_source(name)
        return {'root': root, 'max_depth': self.max_depth, 'nodes': list(self.nodes.values()), 'edges': self.edges}

def build_raw_source_graph(project_path: str | Path, theorem_module: str, root_name: str, *, max_depth: int=2, blueprint_module: str='FLproof.BlueprintGraph', max_nodes: int | None=None, timeout: int=180) -> dict[str, Any]:
    """
    Build a depth-limited SOURCE-LEVEL dependency graph.

    Depth:
        0 = root only
        1 = root + direct dependencies
        2 = root + direct dependencies + their direct dependencies
        ...

    PROOF_USES and DEFINITION_USES still come only from identifiers
    literally present in the original Lean source proof/body.

    No theorem proof term or definition kernel value is traversed.
    """
    if max_depth < 0:
        raise ValueError('max_depth must be >= 0')
    project_path = Path(project_path).resolve()
    builder = _GraphBuilder(project_path=project_path, environment_module=theorem_module, blueprint_module=blueprint_module, timeout=timeout, max_depth=max_depth, max_nodes=max_nodes)
    return builder.build(root_name)

def _field_names(cls: type) -> set[str]:
    return {field.name for field in fields(cls)}

def _required_fields(cls: type) -> set[str]:
    return {field.name for field in fields(cls) if field.default is MISSING and field.default_factory is MISSING}

def _node_category(category_name: str) -> NodeCategory:
    try:
        return NodeCategory[category_name]
    except KeyError as exc:
        raise FLGraphExtractionError(f'NodeCategory is missing {category_name!r}.\nExpected categories include THEOREM, DEFINITION, INSTANCE, AXIOM, INDUCTIVE, CONSTRUCTOR, RECURSOR.') from exc

def _edge_type(edge_name: str) -> EdgeType:
    try:
        return EdgeType[edge_name]
    except KeyError as exc:
        raise FLGraphExtractionError(f'EdgeType is missing {edge_name!r}.\nAdd DEFINITION_USES alongside STATEMENT_USES and PROOF_USES.') from exc

def _make_fl_node(raw: dict[str, Any], node_id: int) -> FLNode:
    available = _field_names(FLNode)
    values: dict[str, Any] = {'id': node_id, 'category': _node_category(raw['category']), 'name': raw['name'], 'statement': raw['statement'], 'proof': raw.get('proof'), 'definition': raw.get('definition'), 'directory': raw.get('source_path') or raw['moduleName'], 'module_name': raw['moduleName'], 'source_declaration': raw.get('source_declaration'), 'depth': raw.get('depth', 0), 'evidence': None}
    kwargs = {key: value for key, value in values.items() if key in available}
    missing = _required_fields(FLNode) - kwargs.keys()
    if missing:
        raise FLGraphExtractionError(f'Cannot construct FLNode; missing required fields: {sorted(missing)}')
    return FLNode(**kwargs)

def _make_fl_edge(source: FLNode, target: FLNode, kind: str) -> FLEdge:
    available = _field_names(FLEdge)
    values: dict[str, Any] = {'source': source, 'target': target, 'type': _edge_type(kind), 'evidence': None, 'explicit': True, 'confidence': 1.0}
    kwargs = {key: value for key, value in values.items() if key in available}
    missing = _required_fields(FLEdge) - kwargs.keys()
    if missing:
        raise FLGraphExtractionError(f'Cannot construct FLEdge; missing required fields: {sorted(missing)}')
    return FLEdge(**kwargs)

def convert_to_fl_graph(raw_graph: dict[str, Any]) -> FLGraph:
    root_name = raw_graph['root']
    raw_by_name = {node['name']: node for node in raw_graph['nodes']}
    if root_name not in raw_by_name:
        raise FLGraphExtractionError(f'Root {root_name!r} is missing from graph nodes.')
    ordered_names = [root_name, *sorted((name for name in raw_by_name if name != root_name))]
    node_by_name: dict[str, FLNode] = {}
    for node_id, name in enumerate(ordered_names):
        node_by_name[name] = _make_fl_node(raw_by_name[name], node_id)
    edges: list[FLEdge] = []
    for raw_edge in raw_graph['edges']:
        source_name = raw_edge['source']
        target_name = raw_edge['target']
        if source_name not in node_by_name or target_name not in node_by_name:
            continue
        edges.append(_make_fl_edge(node_by_name[source_name], node_by_name[target_name], raw_edge['kind']))
    return FLGraph(root_id=node_by_name[root_name].id, nodes=[node_by_name[name] for name in ordered_names], edges=edges)

def build_fl_graph_from_module(project_path: str | Path, theorem_module: str, root_name: str, *, max_depth: int=1, blueprint_module: str='FLproof.BlueprintGraph', max_nodes: int | None=None, timeout: int=180) -> FLGraph:
    raw = build_raw_source_graph(project_path=project_path, theorem_module=theorem_module, root_name=root_name, max_depth=max_depth, blueprint_module=blueprint_module, max_nodes=max_nodes, timeout=timeout)
    return convert_to_fl_graph(raw)

def _json_default(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return asdict(obj)
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

def graph_to_jsonable(graph: FLGraph) -> dict[str, list[dict[str, Any]]]:
    return {'nodes': [asdict(node) for node in graph.nodes], 'edges': [asdict(edge) for edge in graph.edges]}

def write_raw_source_graph(raw_graph: dict[str, Any], output_path: str | Path) -> None:
    Path(output_path).write_text(json.dumps(raw_graph, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
if __name__ == '__main__':
    lean_project = Path('/Users/vietbachhoang/AutoFaith/faithful_metric/FLproof')
    theorem_module = 'FLproof.proof'
    root_name = 'main'
    raw_graph = build_raw_source_graph(project_path=lean_project, theorem_module=theorem_module, root_name=root_name, max_depth=2, max_nodes=None)
    raw_output = lean_project.parent / 'fl_graph_raw.json'
    write_raw_source_graph(raw_graph, raw_output)
    graph = convert_to_fl_graph(raw_graph)
    output_path = lean_project.parent / 'fl_graph.json'
    output_path.write_text(json.dumps(graph_to_jsonable(graph), indent=2, ensure_ascii=False, default=_json_default), encoding='utf-8')
    print(f'Wrote raw graph JSON to {raw_output}')
    print(f'Wrote FL graph JSON to {output_path}')
