"""A stdlib-only parser for the SUBSET of YAML GitHub workflow files use, exposing only what C3
reads: trigger events (and which carry a paths filter), job keys, job display names,
`continue-on-error`, `if`, step counts and step-level `continue-on-error`.

Why not PyYAML: the core is stdlib-only by contract (PyYAML is optional). Why not a silent
fallback: a scanner that can skip is not a gate. Anything this parser cannot handle — tabs,
anchors, flow mappings as structure, a non-mapping document, a workflow with no jobs — raises
WorkflowParseError, which the runner turns into CRASHED. checks/tests/test_workflow_yaml.py checks
this parser against PyYAML over EVERY real workflow in the repo (scope derived from the
filesystem). NEW module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_TRUE = {"true", "yes", "on"}

# NINTH CYCLE (verifier 8, FILED). A NUL byte inside a `run:` value returned PROVEN: this parser read
# a file that real YAML REFUSES to load, so C3 compared a workflow GitHub would never run and reported
# the gate intact. No gate change was hidden, so it is not an escape — but a parser that models YAML
# must never be MORE PERMISSIVE than YAML, or every divergence is a place where what C3 compares and
# what Actions executes are two different documents.
#
# DERIVED FROM PyYAML, not guessed: this is `yaml.reader.Reader.NON_PRINTABLE`, YAML 1.2's c-printable
# set. Verified against PyYAML 6.0.3 rather than assumed — NUL, BEL, ESC, DEL, VT, FF, U+009F and lone
# surrogates are all rejected by `yaml.safe_load`, and U+00A0 is ACCEPTED, so refusing that would make
# this parser stricter than YAML, which is the same error pointing the other way.
#
# It also closes a line-breaking divergence for free. `str.splitlines()` breaks on \x0b \x0c \x1c \x1d
# \x1e, which YAML does not; every one of those is non-printable and is now refused before the split.
# What remains — \n, \r, \x85, \u2028, \u2029 — is exactly PyYAML's own `scan_line_break` set.
_NON_PRINTABLE = re.compile(
    "[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010FFFF]")


class WorkflowParseError(ValueError):
    pass


@dataclass(frozen=True)
class Job:
    key: str
    name: str | None
    continue_on_error: bool                      # a CONSTANT truthy value in any spelling (true / True / 'true' / ${{ true }} / 1)
    if_expr: str | None
    step_count: int
    step_continue_on_error: tuple[int, ...]
    needs: tuple[str, ...] = ()
    continue_on_error_expr: str | None = None    # a NON-constant expression (e.g. ${{ matrix.experimental }}) — C3 observes it

    @property
    def contexts(self) -> tuple[str, ...]:
        """The check-run context names this job can produce: its display name if set, else its key."""
        return (self.name,) if self.name else (self.key,)


@dataclass(frozen=True)
class Workflow:
    name: str | None
    events: frozenset[str]
    paths_filtered_events: tuple[str, ...]
    jobs: dict[str, Job] = field(default_factory=dict)


# ----------------------------------------------------------------------------- line scanning

@dataclass
class _Line:
    indent: int
    content: str
    no: int

    @property
    def is_seq_item(self) -> bool:
        return self.content == "-" or self.content.startswith("- ")


def _strip_comment(s: str) -> str:
    out, quote = [], None
    for i, ch in enumerate(s):
        if quote:
            out.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            out.append(ch)
            continue
        if ch == "#" and (i == 0 or s[i - 1] in " \t"):
            break
        out.append(ch)
    return "".join(out).rstrip()


def _lines(text: str) -> list[_Line]:
    out = []
    for no, raw in enumerate(text.splitlines(), 1):
        if "\t" in raw[: len(raw) - len(raw.lstrip())]:
            raise WorkflowParseError(f"line {no}: tab indentation is not supported")
        if raw.strip().startswith("#") or not raw.strip():
            continue
        if raw.startswith(("---", "...")) and raw.strip() in ("---", "..."):
            continue
        content = _strip_comment(raw.strip())
        if not content:
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        out.append(_Line(indent, content, no))
    return out


def _unquote(v: str) -> str:
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
        return v[1:-1]
    return v


def _scalar(v: str):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_unquote(x) for x in inner.split(",") if x.strip()] if inner else []
    if v.startswith("&") or v.startswith("*"):
        raise WorkflowParseError(f"anchors/aliases are not supported: {v!r}")
    if v == "~" or v == "null":
        return None
    return _unquote(v)


def _split_key(content: str) -> tuple[str, str] | None:
    quote = None
    for i, ch in enumerate(content):
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"') and i == 0:
            quote = ch
            continue
        if ch == ":" and (i + 1 == len(content) or content[i + 1] == " "):
            return _unquote(content[:i]), content[i + 1:].strip()
    return None


class _Parser:
    def __init__(self, lines: list[_Line]):
        self.lines, self.i, self.n = lines, 0, len(lines)

    def parse_block(self, indent: int):
        ln = self.lines[self.i]
        if ln.is_seq_item:
            return self.parse_sequence(indent)
        return self.parse_mapping(indent)

    def parse_mapping(self, indent: int) -> dict:
        out: dict = {}
        while self.i < self.n:
            ln = self.lines[self.i]
            if ln.indent < indent:
                break
            if ln.indent > indent:
                raise WorkflowParseError(f"line {ln.no}: unexpected indent {ln.indent} (expected {indent})")
            if ln.is_seq_item:
                break
            kv = _split_key(ln.content)
            if kv is None:
                raise WorkflowParseError(f"line {ln.no}: expected `key: value`, got {ln.content!r}")
            key, rest = kv
            self.i += 1
            if rest == "":
                out[key] = self._nested(indent)
            elif rest[0] in "|>" and re.fullmatch(r"[|>][-+]?[0-9]?", rest):
                self._skip_block_scalar(indent)
                out[key] = "<block scalar>"
            elif rest.startswith("{") and not rest.startswith("${{"):
                out[key] = rest                      # flow mapping kept as text; C3 does not read these
            else:
                out[key] = _scalar(rest)
        return out

    def _nested(self, parent_indent: int):
        if self.i >= self.n:
            return None
        nxt = self.lines[self.i]
        if nxt.indent > parent_indent:
            return self.parse_block(nxt.indent)
        if nxt.indent == parent_indent and nxt.is_seq_item:
            return self.parse_sequence(parent_indent)
        return None

    def _skip_block_scalar(self, key_indent: int) -> None:
        while self.i < self.n and self.lines[self.i].indent > key_indent:
            self.i += 1

    def parse_sequence(self, indent: int) -> list:
        out: list = []
        while self.i < self.n:
            ln = self.lines[self.i]
            if ln.indent != indent or not ln.is_seq_item:
                break
            body = ln.content[1:]
            pad = len(body) - len(body.lstrip(" "))
            body = body.strip()
            self.i += 1
            if body == "":
                out.append(self._nested(indent))
                continue
            kv = _split_key(body)
            if kv is not None and not body.startswith(("'", '"', "[")):
                item_indent = indent + 1 + pad
                self.lines.insert(self.i, _Line(item_indent, body, ln.no))
                self.n += 1
                out.append(self.parse_mapping(item_indent))
            else:
                out.append(_scalar(body))
        return out


def parse_workflow(text: str) -> Workflow:
    if not isinstance(text, str):
        raise WorkflowParseError(f"workflow text must be str, got {type(text).__name__}")
    bad = _NON_PRINTABLE.search(text)
    if bad is not None:
        raise WorkflowParseError(
            f"character U+{ord(bad.group()):04X} at offset {bad.start()} is outside YAML's printable set, so "
            f"this file is not loadable YAML and GitHub would never run it — refused rather than read")
    lines = _lines(text)
    if not lines:
        raise WorkflowParseError("empty workflow")
    p = _Parser(lines)
    if lines[0].indent != 0 or lines[0].is_seq_item:
        raise WorkflowParseError("top level is not a mapping")
    doc = p.parse_mapping(0)
    if p.i != p.n:
        raise WorkflowParseError(f"line {p.lines[p.i].no}: could not parse {p.lines[p.i].content!r}")
    if not isinstance(doc, dict):
        raise WorkflowParseError("top level is not a mapping")

    on = doc.get("on")
    events: set[str] = set()
    paths_filtered: list[str] = []
    if isinstance(on, str):
        events = {on}
    elif isinstance(on, list):
        events = {str(e) for e in on}
    elif isinstance(on, dict):
        events = set(on)
        for ev, cfg in on.items():
            if isinstance(cfg, dict) and ("paths" in cfg or "paths-ignore" in cfg):
                paths_filtered.append(ev)
    elif on is not None:
        raise WorkflowParseError(f"unsupported `on:` shape: {type(on).__name__}")

    jobs_raw = doc.get("jobs")
    if not isinstance(jobs_raw, dict) or not jobs_raw:
        raise WorkflowParseError("workflow has no jobs — a gate with no jobs would pass by not running")
    jobs: dict[str, Job] = {}
    for key, j in jobs_raw.items():
        if not isinstance(j, dict):
            raise WorkflowParseError(f"job {key!r} is not a mapping")
        steps = j.get("steps")
        steps = steps if isinstance(steps, list) else []
        coe_steps = tuple(i for i, s in enumerate(steps)
                          if isinstance(s, dict) and truthy_constant(s.get("continue-on-error")) is not False)
        needs = j.get("needs")
        needs_t = tuple(needs) if isinstance(needs, list) else ((str(needs),) if needs else ())
        name = j.get("name")
        if_expr = j.get("if")
        coe = truthy_constant(j.get("continue-on-error"))
        jobs[str(key)] = Job(
            key=str(key), name=str(name) if name is not None else None,
            continue_on_error=coe is True,
            if_expr=None if if_expr is None else str(if_expr),
            step_count=len(steps), step_continue_on_error=coe_steps, needs=needs_t,
            continue_on_error_expr=str(j.get("continue-on-error")) if coe is None else None,
        )
    wf_name = doc.get("name")
    return Workflow(name=str(wf_name) if wf_name is not None else None, events=frozenset(events),
                    paths_filtered_events=tuple(paths_filtered), jobs=jobs)


def truthy_constant(value) -> bool | None:
    """True / False for a constant in any GitHub spelling (`true`, `True`, `'true'`, `${{ true }}`, `1`, `yes`,
    `on`; and the false forms), None for a non-constant expression."""
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    e = str(value).strip()
    e = re.sub(r"^\$\{\{\s*(.*?)\s*\}\}$", r"\1", e)
    e = _unquote(e).strip().lower()
    if e in ("true", "yes", "on", "1"):
        return True
    if e in ("false", "no", "off", "0", ""):
        return False
    return None


def is_constant_false(expr: str | None) -> bool:
    """`if: false` in its mechanically-certain spellings: false, 0, '${{ false }}', quoted forms."""
    if expr is None:
        return False
    e = expr.strip()
    e = re.sub(r"^\$\{\{\s*(.*?)\s*\}\}$", r"\1", e)
    e = _unquote(e).strip().lower()
    return e in ("false", "0", "!true")
