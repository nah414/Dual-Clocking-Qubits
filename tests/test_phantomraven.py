"""Tests for the PhantomRaven npm manifest scanner."""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import json
from pathlib import Path

import pytest

import phantomraven as pr


@pytest.fixture()
def temp_repo(tmp_path: Path) -> Path:
    """Return an empty temporary repo directory."""
    return tmp_path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2))


def test_load_blocklist_prefers_external_file(tmp_path: Path) -> None:
    blocklist_file = tmp_path / "custom_blocklist.txt"
    blocklist_file.write_text("# comment\nfq-ui\n  unused-imports  \n\n")

    blocklist = pr.load_blocklist(blocklist_file)
    assert blocklist == ["fq-ui", "unused-imports"]


def test_scan_path_detects_malicious_dependency_and_remote_spec(temp_repo: Path) -> None:
    pkg = {
        "name": "example",
        "version": "0.1.0",
        "dependencies": {
            "flatmap-stream": "0.1.2",
            "safe-package": "https://example.com/safe-package.tgz",
        },
    }
    write_json(temp_repo / "package.json", pkg)

    result = pr.scan_path(temp_repo, ["flatmap-stream"])

    assert [m.package for m in result.malicious] == ["flatmap-stream"]
    assert str(temp_repo / "package.json") in result.remote_urls


def test_scan_path_detects_lockfile_entries_and_resolved_urls(temp_repo: Path) -> None:
    lock = {
        "name": "example",
        "lockfileVersion": 2,
        "packages": {
            "": {"name": "example", "version": "0.1.0"},
            "node_modules/unused-imports": {
                "name": "unused-imports",
                "version": "1.0.0",
                "resolved": "https://registry.npmjs.org/unused-imports/-/unused-imports.tgz",
            },
        },
    }
    write_json(temp_repo / "package-lock.json", lock)

    result = pr.scan_path(temp_repo, ["unused-imports"])

    assert [m.package for m in result.malicious] == ["unused-imports"]
    assert str(temp_repo / "package-lock.json") in result.remote_urls


def test_scan_path_uses_raw_text_fallback_for_invalid_json(temp_repo: Path) -> None:
    broken = '{"dependencies": {"flatmap-stream": "0.1.2"}'
    (temp_repo / "package.json").write_text(broken)

    result = pr.scan_path(temp_repo, ["flatmap-stream"])

    assert any(m.package == "flatmap-stream" for m in result.malicious)


def test_install_precommit_writes_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path
    hooks_dir = repo_root / ".git" / "hooks"
    hooks_dir.mkdir(parents=True)

    ok, message = pr.install_precommit(repo_root)

    hook_path = hooks_dir / "pre-commit"
    assert ok is True
    assert hook_path.exists()
    assert "PhantomRaven pre-commit hook" in hook_path.read_text()
    assert "pre-commit" in message


def test_main_json_output_reports_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo_root = tmp_path / "proj"
    repo_root.mkdir()
    write_json(repo_root / "package.json", {
        "name": "bad",
        "version": "0.0.1",
        "dependencies": {"flatmap-stream": "0.1.2"},
    })

    blocklist_path = tmp_path / "blocklist.txt"
    blocklist_path.write_text("flatmap-stream\n")

    captured = []

    def fake_print(*values: object, **_: object) -> None:  # type: ignore[override]
        captured.append(" ".join(str(v) for v in values))

    monkeypatch.setattr("builtins.print", fake_print)

    exit_code = pr.main([str(repo_root), "--json", "--blocklist", str(blocklist_path)])

    assert exit_code == 1
    assert captured, "main() should have produced JSON output"
    payload = json.loads("\n".join(captured))
    assert payload["malicious"][0]["package"] == "flatmap-stream"
