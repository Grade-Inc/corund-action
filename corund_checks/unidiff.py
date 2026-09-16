"""A small unified-diff parser (git flavour), stdlib only. Shared by C2 and the Action's
entrypoint. It refuses text that is not a diff (ValueError) rather than returning an empty result
a caller could mistake for "nothing changed" — the distinguish test: an empty diff parses to no
files; a non-diff raises; a COMBINED diff (`diff --cc`, `@@@` hunks) raises too, because its
double-prefixed lines would otherwise be read as an empty change.

Paths: git quotes paths with non-ASCII bytes, spaces in some forms, or control characters as
C-style strings (`"a/tests/test_caf\\303\\251.py"`) unless core.quotePath=false; both forms are
decoded here so a quoted path is the same path as the raw one. A leading `./` is not identity.
Mode-only changes, binary patches and symlinks are flagged on the FileDiff so a caller can say what
a diff contained. A rename recorded with no hunks (git rename detection) is kept as a FileDiff with
status `renamed` and no lines. NEW module."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_COMBINED_HUNK_RE = re.compile(r"^@@@ ")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "a": "\a", "b": "\b", "f": "\f", "v": "\v", "\\": "\\", '"': '"'}


@dataclass(frozen=True)
class DiffLine:
    side: str                 # "+" added, "-" deleted, " " context
    text: str
    old_lineno: int | None
    new_lineno: int | None


@dataclass(frozen=True)
class FileDiff:
    path: str                          # the NEW path (old path for deletions)
    old_path: str | None
    status: str                        # added | deleted | modified | renamed
    lines: tuple[DiffLine, ...] = ()
    binary: bool = False
    hunk_count: int = 0
    mode_change: bool = False
    symlink: bool = False

    @property
    def added_count(self) -> int:
        return sum(1 for l in self.lines if l.side == "+")

    @property
    def deleted_count(self) -> int:
        return sum(1 for l in self.lines if l.side == "-")

    @property
    def text_line_count(self) -> int:
        return self.added_count + self.deleted_count

    @property
    def mode_only(self) -> bool:
        return self.mode_change and self.hunk_count == 0 and not self.binary

    def added(self) -> list[DiffLine]:
        return [l for l in self.lines if l.side == "+"]

    def deleted(self) -> list[DiffLine]:
        return [l for l in self.lines if l.side == "-"]

    def new_side_text(self) -> str:
        """The NEW side as the hunks show it (context + added lines, in order). Complete for an
        added file; a fragment for a modified one."""
        return "\n".join(l.text for l in self.lines if l.side != "-") + "\n"

    def added_new_linenos(self) -> set[int]:
        return {l.new_lineno for l in self.lines if l.side == "+" and l.new_lineno is not None}


@dataclass
class _Pending:
    old_path: str | None = None
    new_path: str | None = None
    rename_from: str | None = None
    rename_to: str | None = None
    new_file: bool = False
    deleted_file: bool = False
    binary: bool = False
    mode_change: bool = False
    symlink: bool = False
    lines: list[DiffLine] = field(default_factory=list)
    hunks: int = 0

    def finish(self) -> FileDiff:
        if self.rename_from and self.rename_to:
            return FileDiff(self.rename_to, self.rename_from, "renamed", tuple(self.lines), self.binary, self.hunks,
                            self.mode_change, self.symlink)
        if self.deleted_file or (self.new_path in (None, "/dev/null") and self.old_path):
            path = self.old_path or self.new_path or ""
            return FileDiff(path, path, "deleted", tuple(self.lines), self.binary, self.hunks, self.mode_change, self.symlink)
        if self.new_file or self.old_path in (None, "/dev/null"):
            path = self.new_path or self.old_path or ""
            return FileDiff(path, None, "added", tuple(self.lines), self.binary, self.hunks, self.mode_change, self.symlink)
        path = self.new_path or self.old_path or ""
        return FileDiff(path, self.old_path, "modified", tuple(self.lines), self.binary, self.hunks, self.mode_change, self.symlink)


def unquote_git_path(p: str) -> str:
    """Decode git's C-style quoting (`"a/tests/test_caf\\303\\251.py"` -> `a/tests/test_café.py`).
    Octal escapes are bytes of the UTF-8 encoding; undecodable bytes are replaced, never dropped."""
    p = p.strip()
    if len(p) < 2 or p[0] != '"' or p[-1] != '"':
        return p
    body = p[1:-1]
    out = bytearray()
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            if nxt in "01234567":
                oct_digits = body[i + 1:i + 4]
                m = re.match(r"[0-7]{1,3}", oct_digits)
                digits = m.group(0)
                out.append(int(digits, 8) & 0xFF)
                i += 1 + len(digits)
                continue
            if nxt in _ESCAPES:
                out += _ESCAPES[nxt].encode("utf-8")
                i += 2
                continue
            out += nxt.encode("utf-8")
            i += 2
            continue
        out += ch.encode("utf-8")
        i += 1
    return out.decode("utf-8", errors="replace")


def _norm_path(p: str | None) -> str | None:
    if p is None:
        return None
    while p.startswith("./"):
        p = p[2:]
    return p


def _strip_ab(p: str) -> str | None:
    p = p.strip()
    if p == "/dev/null":
        return None
    p = unquote_git_path(p) if p.startswith('"') else p.split("\t", 1)[0]
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return _norm_path(p)


def _split_git_header(rest: str) -> tuple[str, str] | None:
    """`diff --git <a> <b>` where either side may be quoted, and an unquoted path may contain
    spaces (git only quotes for non-ASCII/control chars by default): split at the LAST ` b/`
    when unquoted, or read quoted tokens."""
    rest = rest.strip()
    if rest.startswith('"'):
        end = _quoted_end(rest, 0)
        if end is None:
            return None
        a = unquote_git_path(rest[:end + 1])
        b_part = rest[end + 1:].strip()
        b = unquote_git_path(b_part) if b_part.startswith('"') else b_part
    else:
        idx = rest.rfind(" b/")
        if idx < 0:
            return None
        a, b_part = rest[:idx], rest[idx + 1:]
        b = unquote_git_path(b_part) if b_part.startswith('"') else b_part
    if a.startswith("a/"):
        a = a[2:]
    if b.startswith("b/"):
        b = b[2:]
    return _norm_path(a) or "", _norm_path(b) or ""


def _quoted_end(s: str, start: int) -> int | None:
    i = start + 1
    while i < len(s):
        if s[i] == "\\":
            i += 2
            continue
        if s[i] == '"':
            return i
        i += 1
    return None


def parse_unified_diff(text: str) -> tuple[FileDiff, ...]:
    if not isinstance(text, str):
        raise TypeError(f"diff must be str, got {type(text).__name__}")
    if text.startswith("﻿"):
        text = text[1:]
    if not text.strip():
        return ()
    out: list[FileDiff] = []
    cur: _Pending | None = None
    old_no = new_no = 0
    in_hunk = False
    saw_any_header = False

    for raw in text.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("diff --cc ") or line.startswith("diff --combined ") or _COMBINED_HUNK_RE.match(line):
            raise ValueError("combined diff (`diff --cc` / `@@@` hunks) is not a unified diff — refused rather than read as empty")
        if line.startswith("diff --git "):
            hdr = _split_git_header(line[len("diff --git "):])
            if hdr is None:
                raise ValueError(f"unreadable `diff --git` header: {line[:120]!r}")
            if cur is not None:
                out.append(cur.finish())
            cur = _Pending(old_path=hdr[0], new_path=hdr[1])
            in_hunk = False
            saw_any_header = True
            continue
        if line.startswith("--- ") and not in_hunk:
            if cur is None:
                cur = _Pending()
                saw_any_header = True
            cur.old_path = _strip_ab(line[4:])
            continue
        if line.startswith("+++ ") and not in_hunk:
            if cur is None:
                cur = _Pending()
                saw_any_header = True
            cur.new_path = _strip_ab(line[4:])
            continue
        if cur is not None and not in_hunk:
            if line.startswith("rename from "):
                cur.rename_from = _norm_path(unquote_git_path(line[len("rename from "):].strip())); continue
            if line.startswith("rename to "):
                cur.rename_to = _norm_path(unquote_git_path(line[len("rename to "):].strip())); continue
            if line.startswith("new file mode"):
                cur.new_file = True
                if line.rstrip().endswith("120000"):
                    cur.symlink = True
                continue
            if line.startswith("deleted file mode"):
                cur.deleted_file = True
                if line.rstrip().endswith("120000"):
                    cur.symlink = True
                continue
            if line.startswith("old mode ") or line.startswith("new mode "):
                cur.mode_change = True
                if line.rstrip().endswith("120000"):
                    cur.symlink = True
                continue
            if line.startswith("Binary files") or line.startswith("GIT binary patch"):
                cur.binary = True; continue
        hm = _HUNK_RE.match(line)
        if hm:
            if cur is None:
                raise ValueError("hunk header before any file header — not a unified diff")
            old_no, new_no = int(hm.group(1)), int(hm.group(3))
            in_hunk = True
            cur.hunks += 1
            continue
        if in_hunk and cur is not None:
            if line.startswith("\\"):
                continue                      # "\ No newline at end of file"
            if line.startswith("+"):
                cur.lines.append(DiffLine("+", line[1:], None, new_no)); new_no += 1
            elif line.startswith("-"):
                cur.lines.append(DiffLine("-", line[1:], old_no, None)); old_no += 1
            elif line.startswith(" ") or line == "":
                cur.lines.append(DiffLine(" ", line[1:] if line else "", old_no, new_no)); old_no += 1; new_no += 1
            else:
                in_hunk = False               # anything else ends the hunk (e.g. a new header we did not match)
            continue
    if cur is not None:
        out.append(cur.finish())
    if not saw_any_header:
        raise ValueError("text is not a unified diff (no `diff --git`, `---`/`+++` headers found)")
    return tuple(out)
