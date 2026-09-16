"""FOURTEENTH CYCLE (verifier 13). The SUBJECT TRACE: a pytest plugin the Action loads into its own
test runs, which records -- per test, during the CALL phase only -- every function in the execution
tree the test ENTERED. That set is the test's SUBJECT, and it is what lets the runtime silencing rule
(`corund_checks/runtime_silencing.py`) clear a base test only by a REPLACEMENT that exercises the same
code, rather than by any live witness anywhere (the thirteenth cycle's escape).

HOW IT IS LOADED, and why this way. The plugin is written as a single file into the phase's report
directory (random name, OUTSIDE the tree, deleted after), that directory is prepended to PYTHONPATH,
and `-p corund_subject_trace` goes on the command line beside `-p no:cacheprovider` -- never through
PYTEST_PLUGINS (which the runner drops on purpose) and never as a file inside the tree. It is loaded
in EVERY phase, so the plugin list and the environment are identical across phases; WHAT it traces is
decided per phase by a targets file, so a phase that needs nothing traces nothing. If the collect-only
pre-pass shows pytest could not import the plugin (a wrapper that scrubs PYTHONPATH), the phase runs
WITHOUT it and the probe says the trace is unavailable -- fail closed at the core: a rename then cannot
be told from a deletion and is accused, with the reason on the receipt.

WHAT IT RECORDS. `sys.setprofile` for the duration of the call phase, `call` events only, and only
frames whose code object lives under the tree root. The function is spelled `path::qualname` with
`co_qualname` where the interpreter has it (3.11+) and `co_name` before that -- the same spelling
`checks/corund_checks/subjects.py` gives a parsed def, so the two compare as strings. Threads the test
starts are not traced (setprofile is per-thread); fixtures are not traced (setup is not the call
phase). Both are stated residuals.

WHAT IT CANNOT DO. It cannot change an outcome: it sets a profile function and unsets it in a
`finally`; an exception inside the profiler is swallowed into the sidecar's `errors` list and never
reaches the test. It writes ONE sidecar per process (xdist workers append their worker id), which the
runner merges. It reads nothing from the tree and imports nothing but the standard library and pytest.
"""
from __future__ import annotations

import glob
import json
import os

MODULE_NAME = "corund_subject_trace"
OUT_VAR = "CORUND_SUBJECT_OUT"
ROOT_VAR = "CORUND_SUBJECT_ROOT"
TARGETS_VAR = "CORUND_SUBJECT_TARGETS"

# The plugin, as text. Valid on every Python this Action supports running tests under (3.8+): no
# walrus, no match, no f-string nesting. Kept deliberately short and dependency-free.
PLUGIN_SOURCE = r'''"""Corund subject trace (written by the Corund Action into a directory outside the tree; not part of the repo)."""
import json
import os
import sys

import pytest

_OUT = os.environ.get("CORUND_SUBJECT_OUT") or ""
_ROOT = os.environ.get("CORUND_SUBJECT_ROOT") or ""
_TARGETS_FILE = os.environ.get("CORUND_SUBJECT_TARGETS") or ""
_SELF = os.path.realpath(__file__)
_CWD = os.getcwd()
_root = (os.path.realpath(_ROOT).rstrip(os.sep) + os.sep) if _ROOT else None
_targets = None            # None = trace every test; a set = only these node ids
_subjects = {}
_errors = []
_written = False


def _load_targets():
    global _targets
    if not _TARGETS_FILE:
        _targets = set()
        return
    try:
        with open(_TARGETS_FILE, "r", encoding="utf-8") as fh:
            lines = [ln.rstrip("\n") for ln in fh]
    except OSError as exc:
        _errors.append("targets file unreadable: %r" % (exc,))
        _targets = set()
        return
    if lines and lines[0].strip() == "*":
        _targets = None
    else:
        _targets = set(ln for ln in lines if ln.strip())


_load_targets()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if not _OUT or _root is None or (_targets is not None and item.nodeid not in _targets):
        yield
        return
    acc = set()
    root = _root

    def prof(frame, event, arg):
        if event == "call":
            try:
                code = frame.f_code
                acc.add((code.co_filename, getattr(code, "co_qualname", None) or code.co_name))
            except Exception as exc:  # never into the test
                _errors.append("profiler: %r" % (exc,))

    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        yield
    finally:
        sys.setprofile(old)
        _subjects[item.nodeid] = acc


def _write():
    global _written
    if _written or not _OUT:
        return
    _written = True
    out = {}
    for nodeid, acc in _subjects.items():
        fns = set()
        for fn, q in acc:
            if not fn or fn.startswith("<"):
                continue
            path = fn if os.path.isabs(fn) else os.path.join(_CWD, fn)
            try:
                path = os.path.realpath(path)
            except (OSError, ValueError):
                continue
            if _root is None or not path.startswith(_root) or path == _SELF:
                continue
            rel = path[len(_root):].replace(os.sep, "/")
            fns.add(rel + "::" + q)
        out[nodeid] = sorted(fns)
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    dest = _OUT + ("." + worker if worker else "")
    tmp = dest + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"subjects": out, "errors": _errors, "all": _targets is None,
                       "targets": (sorted(_targets) if _targets is not None else None)}, fh)
        os.replace(tmp, dest)
    except OSError as exc:
        try:
            with open(dest + ".error", "w", encoding="utf-8") as fh:
                fh.write(repr(exc))
        except OSError:
            pass


def pytest_sessionfinish(session, exitstatus):
    _write()


def pytest_unconfigure(config):
    _write()
'''


def install(directory: str) -> str:
    """Write the plugin module into `directory` (outside the tree) and return its path."""
    path = os.path.join(directory, MODULE_NAME + ".py")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(PLUGIN_SOURCE)
    return path


def write_targets(directory: str, targets: "set[str] | None") -> str:
    """The targets file for one phase: `*` alone means every test; otherwise one node id per line; an
    EMPTY file means trace nothing (the plugin is still loaded, so every phase runs the same plugin list)."""
    path = os.path.join(directory, "subject-targets.txt")
    with open(path, "w", encoding="utf-8") as fh:
        if targets is None:
            fh.write("*\n")
        else:
            for t in sorted(targets):
                fh.write(t + "\n")
    return path


def read(out_path: str) -> tuple[dict[str, list[str]], list[str], bool]:
    """(node id -> [path::qualname], errors, found). Merges the process sidecar and any xdist worker
    sidecars (`<out>.gw0`, ...). `found` is False when no sidecar exists at all -- the plugin never ran
    to session end, or was not loaded."""
    merged: dict[str, list[str]] = {}
    errors: list[str] = []
    found = False
    for p in sorted(glob.glob(glob.escape(out_path) + "*")):
        if p.endswith(".tmp"):
            continue
        if p.endswith(".error"):
            try:
                errors.append(open(p, encoding="utf-8").read()[:200])
            except OSError:
                pass
            continue
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            errors.append(f"{os.path.basename(p)}: unreadable sidecar ({exc})")
            continue
        found = True
        subs = data.get("subjects") or {}
        if isinstance(subs, dict):
            for k, v in subs.items():
                if isinstance(k, str) and isinstance(v, list):
                    merged[k] = sorted(set(merged.get(k, [])) | {x for x in v if isinstance(x, str)})
        for e in data.get("errors") or []:
            errors.append(str(e)[:200])
    return merged, errors, found
