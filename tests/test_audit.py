"""The author audit: what counts as reaching for the answers, in every harness's transcript.

And the session sweep: what a harness kept of an agent that saw the answers
leaves its store for the task's private directory.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from pinnforge.task import audit

OPERATOR = "/data/runs/cavity_solution.mat"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_PRIVATE_DIR", raising=False)
    root = tmp_path / "ws"
    (root / "tasks" / "heat").mkdir(parents=True)
    (root / "tasks" / "ldc" / "private").mkdir(parents=True)
    (root / "tasks" / "ldc" / "eval.py").write_text("x = 1\n", encoding="utf-8")
    (root / "pinnforge.yaml").write_text("version: 1\n", encoding="utf-8")
    return root


@pytest.fixture
def scope(ws, tmp_path):
    return audit.Scope(ws, "heat", operator=OPERATOR, home=tmp_path / "home")


def _claude(*calls, results=None):
    lines = []
    for i, (name, args) in enumerate(calls):
        lines.append(json.dumps({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "id": f"t{i}", "name": name, "input": args}]}}))
        if results and i in results:
            lines.append(json.dumps({"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": f"t{i}", "content": results[i]}]}}))
    return lines


def _hits(scope, lines, runtime="claude_code"):
    return audit.categories(audit.audit(runtime, lines, scope))


# ─────────────────────────────── what the old audit missed ───────────────────────────────


@pytest.mark.parametrize("command", [
    "cd private; ls",
    "cd private&&ls",
    "ls ../ldc/private;python x.py",
    "bash -lc 'cd .. && cat ldc/private/ref_values.npy'",
    "ls ../*/priv*",
    "python -c \"import os; print(os.listdir(os.path.join('..', 'ldc', 'private')))\"",
])
def test_paths_into_a_private_directory_are_caught(scope, command):
    assert audit.PRIVATE in _hits(scope, _claude(("Bash", {"command": command})))


@pytest.mark.parametrize("command", [
    "find / -name ref_values.npy",
    "python -c \"import glob; print(glob.glob('**/ref_values.npy', recursive=True))\"",
    "find .. -name '*.npy'",
    "locate ref_values",
])
def test_looking_for_the_answers_without_naming_private_is_caught(scope, command):
    assert _hits(scope, _claude(("Bash", {"command": command}))), command


def test_a_glob_over_the_parents_is_caught(scope, ws):
    lines = _claude(("Glob", {"pattern": "**/*.npy", "path": str(ws / "tasks")}))
    assert _hits(scope, lines)
    lines = _claude(("Glob", {"pattern": "../**/*.npy"}))
    assert _hits(scope, lines)


def test_copying_a_template_task_that_holds_answers_is_caught(scope, ws):
    lines = _claude(("Bash", {"command": f"cp -r {ws}/tasks/ldc/* ."}))
    assert audit.PRIVATE in _hits(scope, lines)
    lines = _claude(("Bash", {"command": f"cp -r {ws}/tasks/ldc ./copy"}))
    assert audit.PRIVATE in _hits(scope, lines)


def test_the_configured_private_dir_and_the_sealed_copy_are_caught(ws, tmp_path, monkeypatch):
    monkeypatch.setenv("FORGE_PRIVATE_DIR", str(tmp_path / "srv-private"))
    scope = audit.Scope(ws, "heat", home=tmp_path / "home")
    lines = _claude(("Read", {"file_path": str(tmp_path / "srv-private" / "ldc" / "x.npy")}))
    assert audit.PRIVATE in _hits(scope, lines)
    lines = _claude(("Bash", {"command": f"ls {ws}/.pinnforge/sealed/heat/private"}))
    assert audit.SEALED in _hits(scope, lines)
    lines = _claude(("Bash", {"command": "ls ~/elsewhere/.pinnforge/sealed/heat"}))
    assert audit.SEALED in _hits(scope, lines)


def test_harness_session_stores_and_other_authoring_records_are_caught(scope, tmp_path):
    home = tmp_path / "home"
    lines = _claude(("Bash", {"command": "cat ~/.claude/projects/-x-private-reference/s.jsonl"}))
    assert audit.SESSIONS in _hits(scope, lines)
    lines = _claude(("Read", {"file_path": str(home / ".codex" / "sessions" / "r.jsonl")}))
    assert audit.SESSIONS in _hits(scope, lines)
    lines = _claude(("Read", {"file_path": "/ws/tasks/laplace/.authoring/author.log"}))
    assert audit.AUTHORING in _hits(scope, lines)


def test_output_that_came_out_of_a_private_directory_is_caught(scope):
    lines = _claude(("Bash", {"command": "grep -rn u_max .."}),
                    results={0: "../ks/private/reference/report.md:3: u_max = 0.98\n"})
    assert audit.PRIVATE in _hits(scope, lines)
    lines = _claude(("Grep", {"pattern": "u_max", "path": ".."}),
                    results={0: [{"type": "text",
                                  "text": "Found 1 file\n/ws/tasks/ks/private/notes.md"}]})
    assert audit.PRIVATE in _hits(scope, lines)


def test_the_operator_data_is_caught_as_a_path(scope):
    assert audit.OPERATOR in _hits(scope, _claude(("Read", {"file_path": OPERATOR})))
    lines = _claude(("Bash", {"command": "find ~ -name cavity_solution.mat"}))
    assert audit.OPERATOR in _hits(scope, lines)


# ─────────────────────────────── other harnesses ───────────────────────────────


def test_opencode_grep_and_glob_lines_are_calls(scope, ws):
    lines = ["\x1b[0m", '✱ Grep "u_max" in ../ldc/private · 3 matches', "\x1b[0m"]
    assert audit.PRIVATE in _hits(scope, lines, "opencode")
    lines = [f'✱ Glob "**/*.npy" in {ws / "tasks"}']
    assert _hits(scope, lines, "opencode")
    lines = ["$ grep -rn u_max ..", "../ks/private/reference/report.md:3: u_max=0.98", "\x1b[0m"]
    assert audit.PRIVATE in _hits(scope, lines, "opencode")
    lines = ["→ Read problem.md", "← Write baseline.py",
             '✱ Grep "rRMSE" in kb1/INDEX.md · 100 matches', "$ ls -la", "total 0", "\x1b[0m"]
    assert _hits(scope, lines, "opencode") == []


def test_cursor_tool_calls_are_read(scope):
    def ev(sub, cid, body):
        return json.dumps({"type": "tool_call", "subtype": sub, "call_id": cid,
                           "tool_call": body})

    lines = [ev("started", "c1", {"readToolCall": {"args": {"path": "../ldc/private/x.npy"}}})]
    assert audit.PRIVATE in _hits(scope, lines, "cursor_agent")
    shell = {"shellToolCall": {"args": {"command": "cd .. && ls ldc/private"}}}
    lines = [ev("started", "c2", shell)]
    assert audit.PRIVATE in _hits(scope, lines, "cursor_agent")
    lines = [ev("completed", "c3", {"readToolCall": {"args": {"path": "eval.py"},
                                                     "result": {"success": {"content": "x"}}}})]
    assert _hits(scope, lines, "cursor_agent") == []


def test_codex_commands_and_their_output_are_read(scope):
    lines = [json.dumps({"type": "item.started", "item": {
        "id": "i1", "type": "command_execution",
        "command": "bash -lc 'cat ../ldc/private/ref_values.npy'"}})]
    assert audit.PRIVATE in _hits(scope, lines, "codex")
    lines = [json.dumps({"type": "item.completed", "item": {
        "id": "i2", "type": "command_execution", "command": "bash -lc 'grep -rl u_max ..'",
        "aggregated_output": "../ks/private/reference/report.md\n"}})]
    assert audit.PRIVATE in _hits(scope, lines, "codex")


# ─────────────────────────────── what is not a reach ───────────────────────────────


@pytest.mark.parametrize("call", [
    ("WebSearch", {"query": "cavity_solution.mat dataset"}),
    ("Grep", {"pattern": "cavity_solution", "path": "."}),
    ("Bash", {"command": "ls /private/tmp /private/var/folders"}),
    ("Read", {"file_path": "/home/me/private/ws/tasks/heat/eval.py"}),
    ("Bash", {"command": "ls -la && cat eval.py"}),
    ("Write", {"file_path": "baseline.py", "content": "# answers live in private/"}),
    ("Read", {"file_path": "../ldc/eval_grid.csv"}),
    ("Read", {"file_path": "/ws/tasks/heat/.authoring/intake.md"}),
    ("Bash", {"command": "find . -name '*.npy'"}),
    ("Bash", {"command": "grep -n REF_VALUES eval.py ../ldc/eval.py"}),
    ("Bash", {"command": "cat > eval.py <<'EOF'\nREF_VALUES = \"ref_values.npy\"\nEOF"}),
    ("Bash", {"command": "python - <<'PY'\nopen('eval.py','w').write("
                         "'REF_VALUES = \"ref_values.npy\"')\nPY"}),
    ("Bash", {"command": "ls -la .. && cp eval.py /tmp/eval.backup.py"}),
    ("Bash", {"command": "mv tmp_baseline.py baseline.py; ls .."}),
    ("Bash", {"command": "cp -r ../ldc/eval.py ../ldc/baseline.py ."}),
    # a stand-in reference in a temp directory, to try the metric (seen in a real author)
    ("Bash", {"command": "python -c \"import tempfile, os, numpy as np; g = np.loadtxt("
                         "'eval_grid.csv'); d = tempfile.mkdtemp(); np.save(os.path.join(d, "
                         "'ref_values.npy'), g[:, :1]); print(ev.score_predictions({}, d))\""}),
])
def test_ordinary_work_is_not_flagged(scope, call):
    assert _hits(scope, _claude(call)) == [], call


def test_a_listing_that_shows_private_is_not_a_reach(scope):
    lines = _claude(("Bash", {"command": "ls -R .."}),
                    results={0: "../ldc/private:\nref_values.npy\n"})
    assert _hits(scope, lines) == []


def test_a_workspace_under_a_directory_named_private(tmp_path, monkeypatch):
    monkeypatch.delenv("FORGE_PRIVATE_DIR", raising=False)
    root = tmp_path / "private" / "ws"
    (root / "tasks" / "heat").mkdir(parents=True)
    scope = audit.Scope(root, "heat", home=tmp_path / "home")
    lines = _claude(("Read", {"file_path": str(root / "tasks" / "heat" / "eval.py")}),
                    ("Bash", {"command": f"ls {root}/tasks && cat {root}/tasks/heat/problem.md"}))
    assert _hits(scope, lines) == []


def test_hits_carry_the_call_but_only_the_category_is_public(scope):
    hits = audit.audit("claude_code", _claude(("Read", {"file_path": OPERATOR})), scope)
    assert hits and OPERATOR in hits[0].detail
    assert all(OPERATOR not in c for c in audit.categories(hits))


# ─────────────────────────────── harness session stores ───────────────────────────────


def test_claude_sessions_of_the_workdir_move_and_others_stay(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    home = tmp_path / "home"
    workdir = tmp_path / "ws" / "tasks" / "heat" / "private" / "reference"
    workdir.mkdir(parents=True)
    projects = home / ".claude" / "projects"
    mine = projects / audit.claude_slug(str(workdir))
    mine.mkdir(parents=True)
    (mine / "abc123de.jsonl").write_text('{"x": 1}\n', encoding="utf-8")
    other = projects / audit.claude_slug(str(tmp_path / "elsewhere"))
    other.mkdir()
    (other / "s.jsonl").write_text("{}\n", encoding="utf-8")
    backups = home / ".claude" / "file-history" / "abc123de"
    backups.mkdir(parents=True)
    dest = tmp_path / "private-logs" / "harness-sessions"
    moved, warning = audit.sweep_sessions("claude_code", workdir, time.time(), dest, home=home)
    assert warning == "" and len(moved) == 2
    assert not mine.exists() and not backups.exists()
    assert (other / "s.jsonl").is_file(), "a session of another directory is not touched"
    assert list(dest.rglob("abc123de.jsonl"))
    assert audit.claude_slug("/a/b_c.d") == "-a-b-c-d"


def test_codex_rollouts_naming_the_workdir_move(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    home = tmp_path / "home"
    workdir = tmp_path / "ws" / "private" / "reference"
    workdir.mkdir(parents=True)
    day = home / ".codex" / "sessions" / "2026" / "10" / "01"
    day.mkdir(parents=True)
    started = time.time()
    mine = day / "rollout-a.jsonl"
    mine.write_text(json.dumps({"type": "session_meta", "payload": {"cwd": str(workdir)}}) + "\n")
    theirs = day / "rollout-b.jsonl"
    theirs.write_text(json.dumps({"payload": {"cwd": "/somewhere/else"}}) + "\n")
    old = day / "rollout-c.jsonl"
    old.write_text(json.dumps({"payload": {"cwd": str(workdir)}}) + "\n")
    os.utime(old, (started - 3600, started - 3600))
    moved, _ = audit.sweep_sessions("codex", workdir, started, tmp_path / "dest", home=home)
    assert len(moved) == 1 and not mine.exists()
    assert theirs.is_file() and old.is_file()


def test_opencode_sessions_are_left_with_a_warning(tmp_path):
    log = tmp_path / "reference.log"
    log.write_text("timestamp=x level=INFO message=created id=ses_abcdef123 \n")
    store = tmp_path / "home" / ".local" / "share" / "opencode"
    store.mkdir(parents=True)
    (store / "opencode.db").write_bytes(b"db")
    moved, warning = audit.sweep_sessions("opencode", tmp_path, time.time(), tmp_path / "d",
                                          log=log, home=tmp_path / "home")
    assert moved == [] and "ses_abcdef123" in warning and "OpenCode" in warning
    assert (store / "opencode.db").is_file()


def test_the_stores_follow_the_harnesses_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cc"))
    stores = audit.harness_stores(Path("/home/x"))
    assert stores["claude_projects"] == tmp_path / "cc" / "projects"
