"""C2's token tier for JavaScript / TypeScript test files — no parser dependency (stdlib only), so
it is a TOKENIZER-level pass, not a grammar: comments and string contents are blanked, then
member chains that start at a test head (`test`, `it`, `describe`, the `x`/`f` prefixed forms, and
simple aliases `const t = it` / `const { it: t } = ...`) are followed through `.skip`, `['skip']`,
`.concurrent`, `.each(...)`, `.failing`, `.only`, `.todo` in any order and across newlines.
Assertion tautologies (`expect(<literal>)`, `expect.anything()`), `.not.` loosenings against a
deleted `toBe` on the same X, and `process.exit(` are reported too.

What a tokenizer cannot do is stated in `NOT_COVERED_JS`; unstated = not covered. NEW module.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NOT_COVERED_JS: tuple[str, ...] = (
    "jest heads reached through an alias defined outside the diff hunks when `files_after` is absent",
    "a test title built dynamically (test.each tables, template literals with expressions): the target is the file",
    "jest heads re-exported from a wrapper module (`import { it } from './my-jest'`) are matched by name only",
    "a `skip` decided by a runtime expression inside the test body (no token to match)",
    "a computed member on a test head whose expression is NOT a constant string — a variable, a call, a template with a "
    "substitution (`it[names[0]]`, `it[mk('skip')]`): constant literals and `+`-joined constant literals (escapes decoded) "
    "ARE resolved, but a non-constant name is left unresolved rather than guessed",
)

_HEADS = ("test", "it", "describe", "xit", "xtest", "xdescribe", "fit", "fdescribe", "ftest", "bench")
_MODS = {"skip", "only", "todo", "failing", "concurrent", "each", "sequential", "skipIf", "runIf", "todoIf", "fails", "shuffle"}
_ALIAS_RE = re.compile(r"\b(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(test|it|describe)\s*[;,\n]")
_DESTRUCT_RE = re.compile(r"\b(?:const|let|var)\s*\{([^}]*)\}")
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")


@dataclass(frozen=True)
class JsFinding:
    kind: str                # jest-skip | jest-todo | jest-only | jest-failing | constant-true | assertion-loosened | runner-escape
    lineno: int
    snippet: str
    target: str | None


def strip_js(text: str) -> str:
    """Blank comment bodies and string/template contents (same length, newlines kept) so tokens
    inside them are not matched. Regex literals are not modelled (a `/` is treated as division)."""
    out = list(text)
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
            continue
        if ch == "/" and nxt == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                if out[k] != "\n":
                    out[k] = " "
            i = j
            continue
        if ch in ("'", '"', "`"):
            q = ch
            j = i + 1
            while j < n and text[j] != q:
                if text[j] == "\\":
                    j += 1
                j += 1
            for k in range(i + 1, min(j, n)):
                if out[k] != "\n":
                    out[k] = " "
            i = j + 1
            continue
        i += 1
    return "".join(out)


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _match_paren(s: str, i: int) -> int:
    """index just past the `)` matching the `(` at i, or -1."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return -1


def _first_string(original: str, start: int, end: int) -> str | None:
    m = re.search(r"""(['"`])(.*?)\1""", original[start:end], re.S)
    return m.group(2)[:80] if m else None


def _aliases(stripped: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for m in _ALIAS_RE.finditer(stripped):
        out[m.group(1)] = m.group(2)
    for m in _DESTRUCT_RE.finditer(stripped):
        for part in m.group(1).split(","):
            if ":" in part:
                a, b = (p.strip() for p in part.split(":", 1))
                if a in _HEADS and _IDENT.fullmatch(b or ""):
                    out[b] = a
    return out


_JS_STR_RE = re.compile(r"""\s*(?:'((?:[^'\\]|\\.)*)'|"((?:[^"\\]|\\.)*)"|`((?:[^`\\$]|\\.)*)`)\s*""")
_JS_ESCAPE_RE = re.compile(r"\\(x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f]{1,6}\}|u[0-9A-Fa-f]{4}|.)", re.S)
_JS_SIMPLE_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "b": "\b", "f": "\f", "v": "\v", "0": "\0",
                      "\\": "\\", "'": "'", '"': '"', "`": "`", "\n": ""}


def _unescape_js(s: str) -> str:
    def one(m: "re.Match[str]") -> str:
        body = m.group(1)
        if body[0] == "x":
            return chr(int(body[1:], 16))
        if body[0] == "u":
            return chr(int(body[2:-1] if body[1] == "{" else body[1:], 16))
        return _JS_SIMPLE_ESCAPES.get(body, body)
    return _JS_ESCAPE_RE.sub(one, s)


def resolve_computed_member(expr: str) -> str | None:
    """The string a computed member `it[<expr>]` names, when the expression is CONSTANT: one string
    literal, or several joined with `+`. Escapes (\\x73, \\u0073, \\u{73}) are decoded. None when the
    expression is not constant — a variable, a call, a template with a substitution — because the
    audit must not GUESS what a computed name resolves to.

    Verifier 3's V3b-jest-bracket-concat: `it['sk'+'ip']` was read verbatim and quote-stripped, which
    produced the literal `sk'+'ip` and missed the skip."""
    s = expr.strip()
    if not s:
        return None
    parts: list[str] = []
    pos = 0
    want_string = True
    while pos < len(s):
        if want_string:
            m = _JS_STR_RE.match(s, pos)
            if not m:
                return None
            parts.append(_unescape_js(next(g for g in m.groups() if g is not None)))
            pos, want_string = m.end(), False
            continue
        if s[pos] != "+":
            return None
        pos, want_string = pos + 1, True
    if want_string or not parts:
        return None
    return "".join(parts)


def audit_js(source: str, *, added_linenos: set[int] | None, deleted_lines: list[str]) -> list[JsFinding]:
    stripped = strip_js(source)
    aliases = _aliases(stripped)
    heads = set(_HEADS) | set(aliases)
    head_re = re.compile(r"(?<![\w$.])(" + "|".join(sorted(map(re.escape, heads), key=len, reverse=True)) + r")(?![\w$])")
    out: list[JsFinding] = []

    def touched(line: int) -> bool:
        return added_linenos is None or line in added_linenos

    for m in head_re.finditer(stripped):
        head = aliases.get(m.group(1), m.group(1))
        pos = m.end()
        mods: list[str] = []
        called = False
        last_call = None
        while True:
            k = pos
            while k < len(stripped) and stripped[k] in " \t\r\n":
                k += 1
            if k < len(stripped) and stripped[k] == ".":
                k += 1
                while k < len(stripped) and stripped[k] in " \t\r\n":
                    k += 1
                im = _IDENT.match(stripped, k)
                if not im:
                    break
                mods.append(im.group(0))
                pos = im.end()
                continue
            if k < len(stripped) and stripped[k] == "[":
                end = _match_paren(stripped, k)
                if end < 0:
                    break
                raw = source[k + 1:end - 1]
                resolved = resolve_computed_member(raw)
                mods.append(resolved if resolved is not None else raw.strip().strip("'\"`"))
                pos = end
                continue
            if k < len(stripped) and stripped[k] == "(":
                end = _match_paren(stripped, k)
                if end < 0:
                    break
                called = True
                last_call = (k, end)
                pos = end
                continue
            break
        if not called:
            continue
        line = _line_of(stripped, m.start())
        if not touched(line):
            continue
        title = _first_string(source, last_call[0], last_call[1]) if last_call else None
        kind = None
        if "skip" in mods or head.startswith("x") or "skipIf" in mods:
            kind = "jest-skip"
        elif "todo" in mods or "todoIf" in mods:
            kind = "jest-todo"
        elif "only" in mods or head in ("fit", "fdescribe", "ftest"):
            kind = "jest-only"
        elif "failing" in mods or "fails" in mods:
            kind = "jest-failing"
        if kind:
            chain = head + "".join(f".{x}" for x in mods)
            out.append(JsFinding(kind, line, chain[:120], title if kind != "jest-only" else None))

    # assertion shapes on added lines
    for m in re.finditer(r"expect\s*\(\s*(true|false|null|undefined|NaN|-?\d+(?:\.\d+)?|'[^']*'|\"[^\"]*\"|`[^`]*`|\[\s*\]|\{\s*\})\s*\)\s*\.", source):
        line = _line_of(source, m.start())
        if touched(line):
            out.append(JsFinding("constant-true", line, source[m.start():m.start() + 80].splitlines()[0].strip(), None))
    for m in re.finditer(r"expect\s*\(([^()]*(?:\([^()]*\))?[^()]*)\)\s*\.\s*(?:not\s*\.\s*)?(?:toEqual|toStrictEqual|toBe|toMatchObject)\s*\(\s*expect\s*\.\s*(anything|any)\s*\(", source):
        line = _line_of(source, m.start())
        if touched(line):
            out.append(JsFinding("assertion-loosened", line, f"expect({m.group(1).strip()}) matched against expect.{m.group(2)}() — matches anything", None))
    strong = {}
    for dl in deleted_lines:
        sm = re.search(r"expect\s*\((.+?)\)\s*\.\s*(toBe|toEqual|toStrictEqual)\s*\(", dl)
        if sm:
            strong[re.sub(r"\s+", "", sm.group(1))] = dl.strip()
    if strong:
        for m in re.finditer(r"expect\s*\((.+?)\)\s*\.\s*not\s*\.\s*(toBe|toEqual|toStrictEqual|toBeNull|toBeUndefined)\s*\(", source):
            x = re.sub(r"\s+", "", m.group(1))
            line = _line_of(source, m.start())
            if x in strong and touched(line):
                out.append(JsFinding("assertion-loosened", line, f"{strong[x]}  ->  {source[m.start():m.start() + 80].splitlines()[0].strip()}", None))
    for m in re.finditer(r"\bprocess\s*\.\s*exit\s*\(", stripped):
        line = _line_of(stripped, m.start())
        if touched(line):
            out.append(JsFinding("runner-escape", line, "process.exit(...) — ends the runner from inside a test", None))
    seen = set()
    dedup = []
    for f in out:
        k = (f.kind, f.lineno, f.target)
        if k in seen:
            continue
        seen.add(k)
        dedup.append(f)
    return dedup
