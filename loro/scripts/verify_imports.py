#!/usr/bin/env python3
"""Import-smoke test for LoRO modules touched by the Wave 0 fixes.

Purpose
-------
This script is a minimal "did we syntactically break anything" gate for the
``feat/loro-training-pipeline`` PR. It is NOT a training or functional check:
we have no training data on this machine, and full verification is out of
scope for this PR.

What it does
------------
For each module modified in the Wave 0 fixes, attempt an
``importlib.import_module`` and report OK / FAIL. For ``train``, additionally
assert that the newly added stubs ``load_judge_data`` and
``evaluate_judge_ndcg`` exist as callable attributes (they are allowed to
raise ``NotImplementedError`` when invoked - we only care that they are
defined, because ``finetune_judge`` imports them by name).

How to run
----------
    cd ~/workspaces/cogos-dev/research
    python3 loro/scripts/verify_imports.py

Exit code 0 means every module imported clean. Exit code 1 means something
regressed - read the per-module trace printed above the summary line.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

# Make bare imports like ``import train`` resolve to ``loro/train.py`` no
# matter where the script is invoked from. We prepend the ``loro/`` directory
# (parent of this script's directory) to sys.path.
LORO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LORO_DIR))

# Modules touched by the Wave 0 fixes. Order matches the PR description so
# the output is easy to correlate with the review notes.
MODULES = [
    "finetune_judge",  # imports fixed - expects load_judge_data / evaluate_judge_ndcg from train
    "train",           # added stubs load_judge_data / evaluate_judge_ndcg
    "train_mamba",     # padding bug fixed
    "prepare",         # 3 edge-case guards added
    "eval_response",   # format string guard
    "trm_export",      # weights_only=True
]

# Attributes that MUST exist on ``train`` as callables. The stubs are allowed
# to raise NotImplementedError when invoked; this check only verifies that
# they are defined, because ``finetune_judge`` imports them by name at module
# load time.
TRAIN_REQUIRED_CALLABLES = ("load_judge_data", "evaluate_judge_ndcg")


def _trim_traceback(exc: BaseException, max_lines: int = 12) -> str:
    """Return the last ``max_lines`` lines of the formatted traceback."""
    lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
    flat = "".join(lines).rstrip().splitlines()
    if len(flat) <= max_lines:
        return "\n".join(flat)
    return "\n".join(["  ...(traceback trimmed)..."] + flat[-max_lines:])


def verify_module(name: str) -> tuple[bool, str]:
    """Import ``name`` and run any module-specific assertions.

    Returns
    -------
    (ok, detail) where ``ok`` is True on pass, False on failure, and
    ``detail`` is an empty string on pass or a short reason on failure.
    """
    try:
        module = importlib.import_module(name)
    except BaseException as exc:  # noqa: BLE001 - we want to report anything
        return False, _trim_traceback(exc)

    if name == "train":
        missing = []
        not_callable = []
        for attr in TRAIN_REQUIRED_CALLABLES:
            if not hasattr(module, attr):
                missing.append(attr)
                continue
            if not callable(getattr(module, attr)):
                not_callable.append(attr)
        if missing or not_callable:
            parts = []
            if missing:
                parts.append(f"missing attrs: {', '.join(missing)}")
            if not_callable:
                parts.append(f"not callable: {', '.join(not_callable)}")
            return False, "; ".join(parts)

    return True, ""


def main() -> int:
    results: list[tuple[str, bool, str]] = []
    for name in MODULES:
        ok, detail = verify_module(name)
        results.append((name, ok, detail))
        if ok:
            print(f"OK {name}")
        else:
            # Print the failure line first, then the detail indented below so
            # the summary is still easy to grep out.
            print(f"FAIL {name}: {detail.splitlines()[0] if detail else 'unknown'}")
            if detail and "\n" in detail:
                for line in detail.splitlines()[1:]:
                    print(f"    {line}")

    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"{passed}/{total} modules imported clean")
    return 0 if passed == total else 1


if __name__ == "__main__":
    # Guard against accidental invocation from inside ``loro/`` vs the repo
    # root - both should work. We just need ``LORO_DIR`` on sys.path, which is
    # set above.
    sys.exit(main())
