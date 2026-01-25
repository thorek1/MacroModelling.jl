#!/opt/homebrew/bin/python3
"""Repo-local static analysis for MacroModelling.jl constants usage.

Generates a markdown report mapping constants subfields to:
- functions where they are referenced
- possible get_*/plot_* entrypoints that can (transitively) reach those functions

This is heuristic (regex-based) and intended for developer guidance.
"""

from __future__ import annotations

import dataclasses
import os
import re
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
STRUCTURES = REPO_ROOT / "src" / "structures.jl"
MAIN_MODULE = REPO_ROOT / "src" / "MacroModelling.jl"

SEARCH_DIRS = [
    REPO_ROOT / "src",
    REPO_ROOT / "ext",
    REPO_ROOT / "test",
    REPO_ROOT / "benchmark",
    REPO_ROOT / "docs",
    REPO_ROOT / "models",
]

JL_FILE_RE = re.compile(r".*\\.jl$")


@dataclasses.dataclass(frozen=True)
class Field:
    top: str
    sub: Optional[str] = None

    def label(self) -> str:
        return f"{self.top}.{self.sub}" if self.sub else self.top


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def iter_julia_files() -> Iterable[Path]:
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        for p in root.rglob("*.jl"):
            # skip generated docs build output
            if "docs/build" in str(p):
                continue
            yield p


def strip_line_comment(line: str) -> str:
    # crude: Julia line comment marker
    if "#" not in line:
        return line
    return line.split("#", 1)[0]


def scrub_strings_and_comments(text: str) -> str:
    # Remove block comments first
    text = re.sub(r"#=([\s\S]*?)=#", " ", text)
    # Remove double-quoted strings
    text = re.sub(r"\"(?:\\\\.|[^\"])*\"", '""', text)
    # Remove triple-quoted strings
    text = re.sub(r"\"\"\"([\s\S]*?)\"\"\"", '""""""', text)
    # Remove char literals 'a'
    text = re.sub(r"'(?:\\\\.|[^'])*'", "''", text)
    # Remove line comments
    lines = [strip_line_comment(l) for l in text.splitlines()]
    return "\n".join(lines)


def parse_exported_symbols() -> Tuple[Set[str], Set[str]]:
    """Parse exported symbols from src/MacroModelling.jl.

    Returns (exported_functions, exported_macros).
    """
    txt = read_text(MAIN_MODULE)
    exported: Set[str] = set()
    exported_macros: Set[str] = set()

    for raw in txt.splitlines():
        line = strip_line_comment(raw).strip()
        if not line.startswith("export "):
            continue
        payload = line[len("export ") :].strip()
        # split by commas
        for item in [p.strip() for p in payload.split(",")]:
            if not item:
                continue
            if item.startswith("@"):
                exported_macros.add(item)
            else:
                # ignore obvious non-function reexports (types/constants) but keep anyway
                exported.add(item)

    return exported, exported_macros


STRUCT_BLOCK_RE = re.compile(
    r"^\s*(?:mutable\s+)?struct\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:[^\r\n]*)(?:\r?\n)(?P<body>[\s\S]*?)^\s*end\b",
    re.MULTILINE,
)

FIELD_LINE_RE = re.compile(
    r"^\s*(?P<field>[A-Za-z_][A-Za-z0-9_]*|[\u0080-\uFFFF][^:#=\s]*)\s*::",
    re.MULTILINE,
)


def extract_struct_fields(struct_name: str, text: str) -> List[str]:
    for m in STRUCT_BLOCK_RE.finditer(text):
        if m.group("name") == struct_name:
            body = m.group("body")
            fields = []
            for fm in FIELD_LINE_RE.finditer(body):
                fld = fm.group("field")
                # ignore obvious commented lines
                if fld.startswith("#"):
                    continue
                fields.append(fld)
            return fields
    raise RuntimeError(f"Struct {struct_name} not found in src/structures.jl")


def extract_constants_schema() -> Dict[str, List[str]]:
    text = read_text(STRUCTURES)
    tops = extract_struct_fields("constants", text)
    schema: Dict[str, List[str]] = {}
    for top in tops:
        # Only expand known sub-structs; others (if added) remain empty.
        if top in {"post_model_macro", "post_parameters_macro", "post_complete_parameters", "second_order", "third_order"}:
            schema[top] = extract_struct_fields(top, text)
        else:
            schema[top] = []
    return schema


def find_enclosing_function_name(text: str, line_index_0: int) -> str:
    """Best-effort: scan upward for a Julia function definition line."""
    lines = text.splitlines()
    # search up to 300 lines back
    start = max(0, line_index_0 - 300)

    # Common Julia forms:
    #   function foo(...)
    #   function Base.show(io::IO, ::MIME"text/plain", x)
    #   foo(args...) = expr
    #   MacroModelling.foo(args...) = expr
    fn_re = re.compile(
        r"^\s*(?:function\s+)?(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*!?)(?:\s*\(|\s*=)",
    )

    for i in range(line_index_0, start - 1, -1):
        line = lines[i]
        if line.lstrip().startswith("#"):
            continue
        m = fn_re.match(line)
        if not m:
            continue
        name = m.group("name")
        # filter out false positives like "if (" or "for (" (rare with this regex)
        if name in {"if", "for", "while", "let", "begin"}:
            continue
        return name
    return "<top-level>"


def collect_defined_functions(file_text: str) -> Set[str]:
    """Return function names defined in a file (best effort)."""
    defs: Set[str] = set()

    # function foo(
    fn1 = re.compile(r"^\s*function\s+(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*!?)\s*\(", re.MULTILINE)
    # foo(args) =
    fn2 = re.compile(r"^\s*(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*!?)\s*\([^\n]*\)\s*=", re.MULTILINE)

    for m in fn1.finditer(file_text):
        defs.add(m.group("name"))
    for m in fn2.finditer(file_text):
        defs.add(m.group("name"))

    return defs


@dataclasses.dataclass(frozen=True)
class FunctionBlock:
    name: str
    start_line: int
    end_line: int
    body: str


def extract_function_blocks(file_text: str) -> List[FunctionBlock]:
    """Extract function blocks with approximate start/end via keyword nesting.

    Handles:
    - multi-line `function name(...) ... end`
    - one-line `name(args...) = expr`

    This is intentionally heuristic, but far less over-approximating than attributing
    all calls in a file to every function defined in that file.
    """
    cleaned = scrub_strings_and_comments(file_text)
    lines = cleaned.splitlines()
    raw_lines = file_text.splitlines()

    fn_start_re = re.compile(
        r"^\s*function\s+(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*!?)\s*\("
    )
    oneliner_re = re.compile(
        r"^\s*(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*!?)\s*\([^\n]*\)\s*="
    )

    # tokens that open a block requiring `end`
    openers_re = re.compile(
        r"\b(function|if|for|while|let|begin|try|struct|mutable\s+struct|macro|quote)\b"
    )
    end_re = re.compile(r"\bend\b")

    blocks: List[FunctionBlock] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m1 = fn_start_re.match(line)
        m2 = oneliner_re.match(line)
        if m1:
            name = m1.group("name")
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                depth += len(openers_re.findall(lines[j]))
                depth -= len(end_re.findall(lines[j]))
                j += 1
            end_line = j if j > i else i + 1
            body = "\n".join(raw_lines[i:end_line])
            blocks.append(FunctionBlock(name=name, start_line=i + 1, end_line=end_line, body=body))
            i = end_line
            continue
        if m2:
            name = m2.group("name")
            body = raw_lines[i]
            blocks.append(FunctionBlock(name=name, start_line=i + 1, end_line=i + 1, body=body))
            i += 1
            continue
        i += 1

    return blocks


def strip_module_prefix(name: str) -> str:
    return name.split(".")[-1]


def collect_calls(file_text: str) -> Set[str]:
    """Collect called function-like tokens (best effort)."""
    calls: Set[str] = set()

    # Exclude common Julia keywords / macros-ish
    excluded = {
        "if",
        "for",
        "while",
        "let",
        "begin",
        "return",
        "do",
        "try",
        "catch",
        "finally",
        "quote",
        "struct",
        "mutable",
        "module",
        "using",
        "import",
        "export",
        "elseif",
        "else",
        "end",
        "in",
        "where",
        "local",
        "global",
        "const",
    }

    # Matches foo( or Mod.foo(
    call_re = re.compile(r"\b(?P<name>(?:[A-Za-z_][A-Za-z0-9_]*\.)*[A-Za-z_][A-Za-z0-9_]*!?)\s*\(")

    # Remove strings to avoid matching inside them
    scrubbed = re.sub(r"\"(?:\\\\.|[^\"])*\"", '""', file_text)

    for m in call_re.finditer(scrubbed):
        name = m.group("name")
        if strip_module_prefix(name) in excluded:
            continue
        # ignore macros like @something(...) - call_re doesn't match '@'
        calls.add(name)

    return calls


def build_call_graph(files: List[Path]) -> Tuple[Dict[str, Set[str]], Set[str]]:
    """Build caller -> callees graph for functions defined in the repo (per-function attribution)."""
    blocks_by_file: Dict[Path, List[FunctionBlock]] = {}
    defined: Set[str] = set()

    for p in files:
        txt = read_text(p)
        blocks = extract_function_blocks(txt)
        blocks_by_file[p] = blocks
        for b in blocks:
            defined.add(b.name)

    base_to_full: Dict[str, Set[str]] = defaultdict(set)
    for fn in defined:
        base_to_full[strip_module_prefix(fn)].add(fn)

    graph: Dict[str, Set[str]] = {fn: set() for fn in defined}
    for p, blocks in blocks_by_file.items():
        for b in blocks:
            calls = collect_calls(b.body)
            for c in calls:
                if c in defined:
                    graph[b.name].add(c)
                else:
                    for cand in base_to_full.get(strip_module_prefix(c), set()):
                        graph[b.name].add(cand)

    return graph, defined


def compute_entrypoints(defined_functions: Set[str]) -> Set[str]:
    eps = set()
    for fn in defined_functions:
        base = strip_module_prefix(fn)
        if base.startswith("get_") or base.startswith("plot_"):
            eps.add(fn)
    return eps


def compute_export_entrypoints(defined_functions: Set[str]) -> Tuple[Set[str], Set[str]]:
    exported, exported_macros = parse_exported_symbols()
    eps: Set[str] = set()
    for name in exported:
        if name in defined_functions:
            eps.add(name)
        else:
            # allow matching on basename
            base = strip_module_prefix(name)
            for fn in defined_functions:
                if strip_module_prefix(fn) == base:
                    eps.add(fn)

    return eps, exported_macros


def reverse_reachability(call_graph: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    """Return callee -> callers."""
    rev: Dict[str, Set[str]] = defaultdict(set)
    for caller, callees in call_graph.items():
        for callee in callees:
            rev[callee].add(caller)
    return rev


def entrypoints_reaching(target: str, rev_graph: Dict[str, Set[str]], entrypoints: Set[str], max_depth: int = 6) -> Set[str]:
    """Traverse callers up to max_depth and collect entrypoints."""
    found: Set[str] = set()
    q = deque([(target, 0)])
    seen = {target}
    while q:
        node, depth = q.popleft()
        if depth > max_depth:
            continue
        for caller in rev_graph.get(node, set()):
            if caller in entrypoints:
                found.add(caller)
            if caller not in seen:
                seen.add(caller)
                q.append((caller, depth + 1))
    return found


def main() -> int:
    schema = extract_constants_schema()

    julia_files = sorted(iter_julia_files())
    # Keep call graph limited to src/ext/test for entrypoint reasoning (docs/models are noisy).
    cg_files = [p for p in julia_files if any(part in str(p) for part in ("/src/", "/ext/", "/test/"))]

    call_graph, defined = build_call_graph(cg_files)
    rev = reverse_reachability(call_graph)
    entrypoints_get_plot = compute_entrypoints(defined)
    entrypoints_exported, exported_macros = compute_export_entrypoints(defined)

    # Field -> list of (file, line, function)
    usage: Dict[Field, List[Tuple[Path, int, str]]] = defaultdict(list)
    # Field -> set(function)
    usage_fns: Dict[Field, Set[str]] = defaultdict(set)
    # Lines that assign into constants.* (possible "intermediate" processing)
    constants_mutations: List[Tuple[Path, int, str]] = []
    # Field -> set(function) that assigns into it (heuristic)
    mutated_by: Dict[Field, Set[str]] = defaultdict(set)

    # Precompile regexes for each field
    field_res: Dict[Field, re.Pattern] = {}
    # Map subfield name -> owning Field(s); used for aliasing heuristic
    sub_to_fields: Dict[str, List[Field]] = defaultdict(list)
    for top, subs in schema.items():
        # top-level access: something.<top>
        field_res[Field(top)] = re.compile(rf"\b{re.escape(top)}\b")
        for sub in subs:
            field_res[Field(top, sub)] = re.compile(rf"\b{re.escape(top)}\s*\.\s*{re.escape(sub)}\b")
            sub_to_fields[sub].append(Field(top, sub))

    # Prefer per-function scanning for better attribution
    for p in julia_files:
        txt = read_text(p)
        blocks = extract_function_blocks(txt)
        if not blocks:
            continue

        # record constants mutations inside each function body
        mutation_re = re.compile(r"\.constants\.[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\s*=")

        for b in blocks:
            body = b.body
            scrubbed_body = scrub_strings_and_comments(body)

            # Determine which top-level constants substructs are referenced in this function.
            # Used to enable aliasing detection for unique subfield names.
            has_top: Dict[str, bool] = {}
            for top in schema.keys():
                has_top[top] = bool(
                    re.search(rf"\bconstants\s*\.\s*{re.escape(top)}\b", scrubbed_body)
                    or re.search(rf"\.constants\s*\.\s*{re.escape(top)}\b", scrubbed_body)
                )
            for m in mutation_re.finditer(scrubbed_body):
                # approximate line number: count newlines before match within body
                rel_line = scrubbed_body[: m.start()].count("\n")
                constants_mutations.append((p, b.start_line + rel_line, b.name))

            # Also detect assignments into subfields of constants substructs after aliasing.
            # Example: `so = m.constants.second_order; so.kron_states = ...`
            for sub_name, owners in sub_to_fields.items():
                if len(owners) != 1:
                    continue
                owner = owners[0]
                if not has_top.get(owner.top, False):
                    continue
                assign_rx = re.compile(rf"\.\s*{re.escape(sub_name)}\s*=")
                for m in assign_rx.finditer(scrubbed_body):
                    rel_line = scrubbed_body[: m.start()].count("\n")
                    constants_mutations.append((p, b.start_line + rel_line, b.name))
                    mutated_by[owner].add(b.name)

            body_lines = body.splitlines()
            for field, rx in field_res.items():
                if field.sub is None:
                    rx2 = re.compile(rf"\bconstants\s*\.\s*{re.escape(field.top)}\b")
                    for i, line in enumerate(body_lines):
                        if rx2.search(line) is None:
                            continue
                        usage[field].append((p, b.start_line + i, b.name))
                        usage_fns[field].add(b.name)
                else:
                    for i, line in enumerate(body_lines):
                        if rx.search(line) is None:
                            continue
                        usage[field].append((p, b.start_line + i, b.name))
                        usage_fns[field].add(b.name)

            # Aliasing heuristic: if this function references constants.<top>, and a subfield name is unique
            # across all constants substructs, then any `.subfield` access in this function is attributed.
            for sub_name, owners in sub_to_fields.items():
                if len(owners) != 1:
                    continue
                owner = owners[0]
                if not has_top.get(owner.top, False):
                    continue
                # look for `.sub_name` occurrences
                sub_rx = re.compile(rf"\.\s*{re.escape(sub_name)}\b")
                if not sub_rx.search(scrubbed_body):
                    continue
                # record a synthetic usage at function start (we don't know exact line reliably)
                usage[owner].append((p, b.start_line, b.name))
                usage_fns[owner].add(b.name)

    out = REPO_ROOT / "analysis" / "CONSTANTS_USAGE.md"
    audit_out = REPO_ROOT / "analysis" / "CONSTANTS_AUDIT.md"

    def rel(p: Path) -> str:
        return str(p.relative_to(REPO_ROOT)).replace(os.sep, "/")

    with out.open("w", encoding="utf-8") as f:
        f.write("# Constants usage report\n\n")
        f.write("Heuristic static scan (regex-based). For each `constants` field/subfield this lists:\n")
        f.write("- functions where it is referenced (best-effort enclosing `function` detection)\n")
        f.write("- possible entrypoints that can reach those functions (call graph heuristic)\n\n")
        f.write("Entrypoints used:\n")
        f.write(f"- exported API (from src/MacroModelling.jl): {len(entrypoints_exported)} functions, {len(exported_macros)} macros\n")
        f.write(f"- get_/plot_ subset: {len(entrypoints_get_plot)}\n\n")
        f.write("Generated by `analysis/constants_usage_report.py`.\n\n")

        for top in schema.keys():
            f.write(f"## constants.{top}\n\n")

            # First: top-level usage
            top_field = Field(top)
            top_usages = usage.get(top_field, [])
            if top_usages:
                f.write("### Direct uses of `constants.%s`\n\n" % top)
                f.write("| Function | File:Line | Entrypoints (get_/plot_) |\n")
                f.write("|---|---:|---|\n")
                for (fp, ln, fn) in sorted(top_usages, key=lambda x: (rel(x[0]), x[1], x[2])):
                    eps = sorted({strip_module_prefix(e) for e in entrypoints_reaching(fn, rev, entrypoints_exported)} or {strip_module_prefix(e) for e in entrypoints_reaching(strip_module_prefix(fn), rev, entrypoints_exported)})
                    eps_str = ", ".join(eps) if eps else "(none found)"
                    f.write(f"| `{fn}` | {rel(fp)}:{ln} | {eps_str} |\n")
                f.write("\n")

            # Now: subfields
            subs = schema[top]
            if not subs:
                f.write("(No subfields expanded for this top-level field.)\n\n")
                continue

            f.write("### Subfields\n\n")
            for sub in subs:
                fld = Field(top, sub)
                u = usage.get(fld, [])
                f.write(f"#### {top}.{sub}\n\n")
                if not u:
                    f.write("- (no references found)\n\n")
                    continue

                # Group by function
                by_fn: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
                for fp, ln, fn in u:
                    by_fn[fn].append((fp, ln))

                f.write("| Function | Locations | Entrypoints (get_/plot_) |\n")
                f.write("|---|---:|---|\n")
                for fn in sorted(by_fn.keys()):
                    locs = sorted(by_fn[fn], key=lambda x: (rel(x[0]), x[1]))
                    loc_str = "; ".join(f"{rel(fp)}:{ln}" for fp, ln in locs[:6])
                    if len(locs) > 6:
                        loc_str += f"; (+{len(locs)-6} more)"

                    eps = entrypoints_reaching(fn, rev, entrypoints_exported)
                    if not eps:
                        # try without module prefix
                        eps = entrypoints_reaching(strip_module_prefix(fn), rev, entrypoints_exported)
                    eps_str = ", ".join(sorted({strip_module_prefix(e) for e in eps})) if eps else "(none found)"
                    f.write(f"| `{fn}` | {loc_str} | {eps_str} |\n")
                f.write("\n")

    # Build an audit summary aimed at identifying unused-by-exported fields and intermediate mutations.
    all_fields: List[Field] = []
    for top, subs in schema.items():
        all_fields.append(Field(top))
        for sub in subs:
            all_fields.append(Field(top, sub))

    with audit_out.open("w", encoding="utf-8") as f:
        f.write("# Constants audit\n\n")
        f.write("This file answers two questions (heuristically):\n")
        f.write("1. Which `constants` fields/subfields are not needed by any exported/user-facing API entrypoint?\n")
        f.write("2. Which functions appear to mutate `.constants.*` after construction (potential further-processing of constants)?\n\n")
        f.write("Entrypoints:\n")
        f.write(f"- Exported functions: {len(entrypoints_exported)}\n")
        f.write(f"- Exported macros: {len(exported_macros)} (not included in call graph)\n")
        f.write(f"- get_/plot_ subset: {len(entrypoints_get_plot)}\n\n")

        f.write("## Unused by exported entrypoints (candidates)\n\n")
        unused: List[str] = []
        for fld in all_fields:
            fns = usage_fns.get(fld, set())
            if not fns:
                unused.append(f"- `{fld.label()}` (no references found)")
                continue
            reachable = False
            for fn in fns:
                if entrypoints_reaching(fn, rev, entrypoints_exported):
                    reachable = True
                    break
            if not reachable:
                unused.append(f"- `{fld.label()}` (referenced, but no exported entrypoint reaches the referencing functions)")
        if unused:
            f.write("\n".join(unused))
            f.write("\n\n")
        else:
            f.write("- (none found)\n\n")

        f.write("## Used by exported entrypoints\n\n")
        f.write("| Field | Used by exported? | Used by get_/plot_? | #Readers | #Mutators |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for fld in all_fields:
            fns = usage_fns.get(fld, set())
            used_exported = any(entrypoints_reaching(fn, rev, entrypoints_exported) for fn in fns)
            used_get_plot = any(entrypoints_reaching(fn, rev, entrypoints_get_plot) for fn in fns)
            muts = mutated_by.get(fld, set())
            f.write(f"| `{fld.label()}` | {"yes" if used_exported else "no"} | {"yes" if used_get_plot else "no"} | {len(fns)} | {len(muts)} |\n")

        f.write("\n## Post-construction mutations of constants (review)\n\n")
        if not constants_mutations:
            f.write("- (none found)\n")
        else:
            f.write("These are places where code assigns into `.constants.* = ...` inside a function body.\n")
            f.write("This can indicate that constants hold intermediate data that can be further processed without new inputs, or that lazy caches are populated.\n\n")
            f.write("| Function | Location |\n")
            f.write("|---|---:|\n")
            for fp, ln, fn in sorted(constants_mutations, key=lambda x: (rel(x[0]), x[1], x[2])):
                f.write(f"| `{fn}` | {rel(fp)}:{ln} |\n")

        f.write("\n## Notes\n\n")
        f.write("- This audit is static and heuristic; macros and dynamic dispatch can hide true reachability.\n")
        f.write("- If a field is only used during model construction via macros, it may appear as unused-by-exported here.\n")

    print(f"Wrote {out}")
    print(f"Wrote {audit_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
