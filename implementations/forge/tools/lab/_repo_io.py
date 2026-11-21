#!/usr/bin/env python3
"""Shared repo-internal write contract for the offline lab tools.

Every output these harnesses produce lives inside the project tree; this
helper refuses writes resolving outside the repo root. Containment for
buggy invocations, not a policy change — no documented flow writes
elsewhere. (Mirrors the guard in tools/metrics.py.)
"""
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))  # tools/lab/_repo_io.py -> repo root


def repo_open_w(path, mode="w"):
    ap = os.path.realpath(os.path.abspath(str(path)))
    if not ap.startswith(_REPO_ROOT + os.sep):
        raise ValueError(f"refusing write outside repo root: {ap}")
    return open(ap, mode)
