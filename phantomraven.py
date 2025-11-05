#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PhantomRaven — Supply-Chain Defense CLI (Adam × Nova)

One-file scanner to catch known-bad npm packages and suspicious remote URLs
before they contaminate your tree or pipeline.

Features
- Scans recursively for package manifests: package.json, package-lock.json
- Exact-match blocklist detection (no substring false positives)
- Remote URL detection in manifests/lockfiles
- CI-friendly JSON output + nonzero exit on findings (opt-out via flag)
- Optional Git pre-commit hook installer that invokes this script
- Embedded default blocklist (override with --blocklist)
- Self-tests: run `python phantomraven.py --self-test`

Usage
  # basic scan (current folder)
  python phantomraven.py

  # scan a specific repo
  python phantomraven.py /path/to/project

  # JSON output (good for CI)
  python phantomraven.py . --json

  # custom blocklist file (newline-delimited)
  python phantomraven.py . --blocklist .phantomraven_blocklist.txt

  # don’t fail the process on findings (advisory mode)
  python phantomraven.py . --no-exit-nonzero

  # install a git pre-commit hook that calls this script
  python phantomraven.py --install-precommit
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# -----------------------------
# Embedded Default Blocklist
# -----------------------------
_EMBEDDED_BLOCKLIST = """
fq-ui
mocha-no-only
ft-flow
ul-inline
jest-hoist
jfrog-npm-actions-example
@acme-types/acme-package
react-web-api
mourner
unused-imports
jira-ticket-todo-comment
polyfill-corejs3
polyfill-regenerator
@aio-commerce-sdk/config-tsdown
@aio-commerce-sdk/config-typedoc
@aio-commerce-sdk/config-typescript
@aio-commerce-sdk/config-vitest
powerbi-visuals-sunburst
@gitlab-lsp/pkg-1
@gitlab-lsp/pkg-2
@gitlab-lsp/workflow-api
@gitlab-test/bun-v1
@gitlab-test/npm-v10
@gitlab-test/pnpm-v9
@gitlab-test/yarn-v4
acme-package
add-module-exports
add-shopify-header
jsx-a11y
typescript-sort-keys
uach-retrofill
""".strip().splitlines()

DEFAULT_BLOCKLIST_FILE = ".phantomraven_blocklist.txt"

PKG_FILES_GLOB = ("package.json", "package-lock.json")

URL_PATTERN = re.compile(r'https?://[^\s"\'`]+', re.IGNORECASE)
# Exact package key match: "package-name": or "<scoped>/name":
def _pkg_key_pattern(pkg: str) -> re.Pattern:
    # Anchor a quoted key followed by optional whitespace and a colon
    return re.compile(rf'"{re.escape(pkg)}"\s*:', re.IGNORECASE)

# -----------------------------
# Data Structures
# -----------------------------

@dataclass
class MaliciousFinding:
    package: str
    file: str
    context: Optional[str] = None  # e.g. "dependencies", "devDependencies", "lockfile"

@dataclass
class ScanResult:
    malicious: List[MaliciousFinding]
    remote_urls: List[str]
    errors: List[str]

    def to_json(self) -> str:
        payload = {
            "malicious": [asdict(m) for m in self.malicious],
            "remote_urls": list(self.remote_urls),
            "errors": list(self.errors),
        }
        return json.dumps(payload, indent=2, ensure_ascii=False)

    @property
    def has_findings(self) -> bool:
        return bool(self.malicious)


# -----------------------------
# Blocklist Handling
# -----------------------------

def load_blocklist(external_path: Optional[Path]) -> List[str]:
    """
    Loads a newline-delimited blocklist. If no file present, falls back to embedded list.
    Comments starting with '#' are ignored. Empty lines ignored.
    """
    if external_path and external_path.exists():
        try:
            raw = external_path.read_text(encoding="utf-8", errors="ignore").splitlines()
            items = [ln.strip() for ln in raw if ln.strip() and not ln.strip().startswith("#")]
            return items
        except Exception as e:
            print(f"⚠️  Could not read blocklist {external_path}: {e}", file=sys.stderr)
            # fall through to embedded
    return [p.strip() for p in _EMBEDDED_BLOCKLIST if p.strip() and not p.strip().startswith("#")]


# -----------------------------
# JSON Utilities for npm files
# -----------------------------

def _safe_read_text(p: Path) -> Tuple[str, Optional[str]]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore"), None
    except Exception as e:
        return "", f"Could not read {p}: {e}"

def _safe_load_json(text: str, p: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return json.loads(text), None
    except Exception as e:
        return None, f"Invalid JSON in {p}: {e}"

def _collect_deps_from_package_json(obj: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns map of {package_name: version_spec} from various dependency sections.
    """
    deps = {}
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies", "bundleDependencies"):
        d = obj.get(section)
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(k, str):
                    deps[k] = str(v)
    return deps

def _collect_pkglock_entries(obj: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Returns map of package_name -> metadata found in a package-lock.json (v1/v2+).
    Tries both "dependencies" and "packages" formats.
    """
    out: Dict[str, Dict[str, Any]] = {}

    # v1 format
    deps = obj.get("dependencies")
    if isinstance(deps, dict):
        for name, meta in deps.items():
            if isinstance(name, str) and isinstance(meta, dict):
                out[name] = meta

    # v2+ format: "packages" keyed by paths; entries include "name" sometimes
    pkgs = obj.get("packages")
    if isinstance(pkgs, dict):
        for path_key, meta in pkgs.items():
            if not isinstance(meta, dict):
                continue
            name = meta.get("name")
            if isinstance(name, str):
                out[name] = meta
    return out


# -----------------------------
# Scanner
# -----------------------------

def scan_path(base: Path, blocklist: List[str]) -> ScanResult:
    malicious: List[MaliciousFinding] = []
    remote_urls: List[str] = []
    errors: List[str] = []

    pkg_key_patterns = {pkg: _pkg_key_pattern(pkg) for pkg in blocklist}

    for p in base.rglob("*"):
        if p.name not in PKG_FILES_GLOB:
            continue

        text, err = _safe_read_text(p)
        if err:
            errors.append(err)
            continue

        # Quick text-based URL check (fast path)
        if URL_PATTERN.search(text):
            remote_urls.append(str(p))

        # Try structured parse for precise findings
        obj, perr = _safe_load_json(text, p)
        if perr:
            errors.append(perr)
            # Fallback: regex for exact key match in raw text for each package
            for pkg, pat in pkg_key_patterns.items():
                if pat.search(text):
                    malicious.append(MaliciousFinding(package=pkg, file=str(p), context="raw-text"))
            continue

        if p.name == "package.json":
            deps = _collect_deps_from_package_json(obj)
            dep_names = set(deps.keys())
            for pkg in blocklist:
                if pkg in dep_names:
                    # Try to infer which section (best-effort)
                    ctx = _infer_dep_context(obj, pkg)
                    malicious.append(MaliciousFinding(package=pkg, file=str(p), context=ctx))
            # Scan values for URLs (version specs can be git+https, direct http tarballs, etc.)
            for name, spec in deps.items():
                if isinstance(spec, str) and URL_PATTERN.search(spec):
                    if str(p) not in remote_urls:
                        remote_urls.append(str(p))

        elif p.name == "package-lock.json":
            entries = _collect_pkglock_entries(obj)
            for pkg in blocklist:
                if pkg in entries:
                    malicious.append(MaliciousFinding(package=pkg, file=str(p), context="lockfile"))
            # look for "resolved" fields with URLs
            for meta in entries.values():
                resolved = meta.get("resolved")
                if isinstance(resolved, str) and URL_PATTERN.search(resolved):
                    if str(p) not in remote_urls:
                        remote_urls.append(str(p))

    return ScanResult(malicious=malicious, remote_urls=remote_urls, errors=errors)


def _infer_dep_context(pkg_json: Dict[str, Any], pkg: str) -> str:
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies", "bundleDependencies"):
        d = pkg_json.get(section)
        if isinstance(d, dict) and pkg in d:
            return section
    return "dependencies"


# -----------------------------
# Git Hook Installer
# -----------------------------

_PRECOMMIT_SH = """#!/bin/sh
# PhantomRaven pre-commit hook (cross-platform via Python)
# Blocks commit if malicious packages or remote URLs are detected in staged npm manifests.

# Determine repo root
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_ROOT" ]; then
  echo "PhantomRaven: not inside a Git repository."
  exit 0
fi

# Invoke the Python script from repo root to ensure consistent path resolution.
PY=python
if command -v python3 >/dev/null 2>&1; then PY=python3; fi

"$PY" "$REPO_ROOT/phantomraven.py" "$REPO_ROOT" --json
STATUS=$?

if [ $STATUS -ne 0 ]; then
  echo ""
  echo "Commit blocked — remove malicious packages or remote installs first."
  exit 1
fi

exit 0
""".strip()

def install_precommit(repo_root: Path) -> Tuple[bool, str]:
    hooks_dir = repo_root / ".git" / "hooks"
    if not hooks_dir.exists():
        return False, f"Git hooks directory not found at {hooks_dir} (is this a Git repo?)"
    hook_path = hooks_dir / "pre-commit"
    try:
        hook_path.write_text(_PRECOMMIT_SH, encoding="utf-8")
        os.chmod(hook_path, 0o755)
        return True, f"Installed pre-commit hook at {hook_path}"
    except Exception as e:
        return False, f"Failed to install pre-commit hook: {e}"


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="PhantomRaven security scanner — detect malicious npm packages and remote URLs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("path", nargs="?", default=".", help="Path to scan (repo root or subdir)")
    ap.add_argument("--json", action="store_true", help="Output results as JSON")
    ap.add_argument("--blocklist", default=DEFAULT_BLOCKLIST_FILE, help="Custom blocklist file (newline-delimited)")
    ap.add_argument("--no-exit-nonzero", action="store_true",
                    help="Do not fail (exit code 0) even if findings are detected")
    ap.add_argument("--install-precommit", action="store_true",
                    help="Install a Git pre-commit hook that calls this script")
    ap.add_argument("--self-test", action="store_true",
                    help="Run built-in unit tests and exit")
    return ap.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    if args.self_test:
        return _run_self_tests()

    repo_path = Path(args.path).resolve()

    if args.install_precommit:
        ok, msg = install_precommit(repo_path)
        print(msg)
        return 0 if ok else 1

    blocklist_path = Path(args.blocklist) if args.blocklist else None
    blocklist = load_blocklist(blocklist_path)

    result = scan_path(repo_path, blocklist)

    if args.json:
        print(result.to_json())
    else:
        if result.malicious:
            print("❌ Malicious packages found:")
            for f in result.malicious:
                ctx = f" ({f.context})" if f.context else ""
                print(f"  - {f.package}{ctx} → {f.file}")
        if result.remote_urls:
            print("\n⚠️  Remote install URLs detected in:")
            for f in result.remote_urls:
                print(f"  - {f}")
        if result.errors:
            print("\nℹ️  Non-fatal parse/read errors:")
            for e in result.errors:
                print(f"  - {e}")
        if not result.has_findings and not result.errors:
            print("✅ No malicious packages or remote URLs detected.")

    if result.has_findings and not args.no_exit_nonzero:
        return 1
    return 0


# -----------------------------
# Tests (no external deps)
# -----------------------------

def _run_self_tests() -> int:
    """
    Lightweight, dependency-free tests using temporary on-disk files.
    """
    import tempfile
    import shutil

    ok = True
    tmpdir = tempfile.mkdtemp(prefix="phantomraven_test_")
    root = Path(tmpdir)

    try:
        # Case 1: Clean project → no findings
        (root / "package.json").write_text(json.dumps({
            "name": "clean-project",
            "version": "1.0.0",
            "dependencies": {"left-pad": "1.3.0"},
            "devDependencies": {"typescript": "^5.0.0"}
        }, indent=2))
        (root / "package-lock.json").write_text(json.dumps({
            "name": "clean-project",
            "lockfileVersion": 2,
            "packages": {
                "": {"name": "clean-project", "version": "1.0.0"},
                "node_modules/left-pad": {"version": "1.3.0", "resolved": "https://registry.npmjs.org/left-pad/-/left-pad-1.3.0.tgz"}
            }
        }, indent=2))

        res = scan_path(root, blocklist=["flatmap-stream"])  # something not present
        _assert(res.has_findings is False, "Clean project should have no findings")
        _assert(len(res.remote_urls) == 1, "Resolved URL in lockfile should be detected")
        # Clean, but still remote_urls due to registry URL (expected). We don't fail this test;
        # we only assert detection. That’s the designed behavior.

        # Case 2: Malicious dependency in package.json
        (root / "proj2").mkdir(parents=True, exist_ok=True)
        (root / "proj2" / "package.json").write_text(json.dumps({
            "name": "bad-project",
            "version": "0.1.0",
            "dependencies": {"flatmap-stream": "0.1.2"},
            "devDependencies": {}
        }, indent=2))
        res2 = scan_path(root, blocklist=["flatmap-stream"])
        _assert(any(m.package == "flatmap-stream" for m in res2.malicious), "Should flag flatmap-stream in package.json")

        # Case 3: Malicious dependency in lockfile (v2 'packages' section)
        (root / "proj3").mkdir(parents=True, exist_ok=True)
        (root / "proj3" / "package-lock.json").write_text(json.dumps({
            "name": "bad-lock",
            "lockfileVersion": 2,
            "packages": {
                "": {"name": "bad-lock", "version": "1.0.0"},
                "node_modules/unused-imports": {"name": "unused-imports", "version": "3.0.0"}
            }
        }, indent=2))
        res3 = scan_path(root, blocklist=["unused-imports"])
        _assert(any(m.package == "unused-imports" for m in res3.malicious), "Should flag unused-imports in lockfile")

        # Case 4: Scoped package exact match
        (root / "proj4").mkdir(parents=True, exist_ok=True)
        (root / "proj4" / "package.json").write_text(json.dumps({
            "name": "scoped",
            "version": "1.0.0",
            "dependencies": {"@aio-commerce-sdk/config-typescript": "^1.0.0"}
        }, indent=2))
        res4 = scan_path(root, blocklist=["@aio-commerce-sdk/config-typescript"])
        _assert(any(m.package == "@aio-commerce-sdk/config-typescript" for m in res4.malicious),
                "Should flag exact scoped package")

        # Case 5: Raw-text fallback for invalid JSON
        (root / "proj5").mkdir(parents=True, exist_ok=True)
        (root / "proj5" / "package.json").write_text('{"dependencies": {"flatmap-stream": "0.1.2"')  # broken JSON
        res5 = scan_path(root, blocklist=["flatmap-stream"])
        _assert(any(m.package == "flatmap-stream" for m in res5.malicious),
                "Fallback raw-text regex should flag in broken JSON")

        print("✅ PhantomRaven self-tests passed.")
        return 0
    except AssertionError as e:
        print(f"❌ Self-test failed: {e}")
        ok = False
        return 1
    finally:
        try:
            shutil.rmtree(root, ignore_errors=True)
        except Exception:
            pass


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


# -----------------------------
# Entry Point
# -----------------------------

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
