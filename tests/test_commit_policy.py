"""Exercise the commit gate against real Git histories and event payloads."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_commit_policy.py"
SIGNOFF = "Signed-off-by: Test Author <author@example.com>"


def git(repo: Path, *args: str, message: str | None = None) -> str:
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "author@example.com",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "author@example.com",
    }
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", "-c", f"core.hooksPath={os.devnull}", *args],
        input=message,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    ).stdout.strip()


def commit(repo: Path, message: str) -> str:
    git(repo, "commit", "--allow-empty", "--cleanup=verbatim", "-F", "-", message=message)
    return git(repo, "rev-parse", "HEAD")


@pytest.fixture
def history(tmp_path: Path) -> tuple[Path, str]:
    git(tmp_path, "init", "-q", "-b", "main")
    # Adoption must not require rewriting unsigned legacy commits on main.
    base = commit(tmp_path, "legacy initial commit")
    return tmp_path, base


def check(repo: Path, base: str, *, title: str | None = None, head: str = "HEAD") -> subprocess.CompletedProcess[str]:
    args = [sys.executable, "-I", str(SCRIPT), "--repo", str(repo), "--base", base, f"--head={head}"]
    if title is not None:
        args.extend(["--title", title])
    return subprocess.run(args, capture_output=True, text=True)


def test_valid_range_with_scope_breaking_change_and_body(history):
    repo, base = history
    commit(repo, f"feat(deployment): validate device configuration\n\n{SIGNOFF}\n")
    commit(
        repo,
        "fix(model_services)!: require protocol version\n\n"
        "Reject peers without an explicit version.\n\n"
        f"BREAKING CHANGE: peers must send a protocol version.\n{SIGNOFF}\n",
    )
    result = check(repo, base, title="feat(deployment): validate device configuration")
    assert result.returncode == 0, result.stderr
    assert "2 commit(s) and the PR title" in result.stdout


def test_unsigned_earlier_commit_is_not_hidden_by_signed_head(history):
    repo, base = history
    unsigned = commit(repo, "fix: correct device state\n")
    signed = commit(repo, f"docs: explain device state\n\n{SIGNOFF}\n")
    result = check(repo, base)
    assert result.returncode == 1
    assert unsigned in result.stderr
    assert signed not in result.stderr
    assert "missing author sign-off" in result.stderr


@pytest.mark.parametrize(
    ("message", "error"),
    [
        ("fix: correct state\n\nSigned-off-by: Other Person <other@example.com>\n", "missing author sign-off"),
        (f"fix: correct state\n\n{SIGNOFF}\n\nThis is still message prose.\n", "missing author sign-off"),
        ("fix: correct state\n\nSigned-off-by: Test Author\n", "must have the form"),
        ("fix: correct state\n\nsigned-off-by: Test Author <author@example.com>\n", "exact trailer spelling"),
    ],
)
def test_signoff_must_be_a_valid_final_author_trailer(history, message, error):
    repo, base = history
    commit(repo, message)
    result = check(repo, base)
    assert result.returncode == 1
    assert error in result.stderr


@pytest.mark.parametrize("coauthor_signed", [False, True])
def test_each_named_coauthor_requires_their_own_signoff(history, coauthor_signed):
    repo, base = history
    message = f"feat: coordinate device setup\n\nCo-authored-by: Second Author <second@example.com>\n{SIGNOFF}\n"
    if coauthor_signed:
        message += "Signed-off-by: Second Author <SECOND@example.com>\n"
    commit(repo, message)
    result = check(repo, base)
    assert result.returncode == (0 if coauthor_signed else 1), result.stderr
    if not coauthor_signed:
        assert "Second Author <second@example.com>" in result.stderr


@pytest.mark.parametrize("merge_signed", [False, True])
def test_authored_merges_checked_and_updated_base_history_excluded(history, merge_signed):
    repo, _ = history
    git(repo, "switch", "-c", "upstream")
    base = commit(repo, "unsigned upstream change")
    git(repo, "switch", "main")
    commit(repo, f"feat: coordinate device setup\n\n{SIGNOFF}\n")
    message = "chore(merge): sync main"
    if merge_signed:
        message += f"\n\n{SIGNOFF}"
    git(repo, "merge", "--no-ff", "--cleanup=verbatim", "-m", message, "upstream")
    merge_sha = git(repo, "rev-parse", "HEAD")
    result = check(repo, base)
    assert result.returncode == (0 if merge_signed else 1), result.stderr
    if merge_signed:
        assert "2 commit(s)" in result.stdout
    else:
        assert merge_sha in result.stderr
        assert base not in result.stderr


@pytest.mark.parametrize(
    ("message", "error"),
    [
        (f"update: correct state\n\n{SIGNOFF}\n", "use <type>"),
        (f"fix: {'x' * 68}\n\n{SIGNOFF}\n", "at most 72 characters"),
        (f"fix: correct state\nBody without an empty line.\n\n{SIGNOFF}\n", "empty line"),
        (f"fix: correct state\n\nBody with trailing whitespace. \n\n{SIGNOFF}\n", "trailing whitespace"),
        (f"fix: correct state.\n\n{SIGNOFF}\n", "full stop"),
    ],
)
def test_message_rules_reject_invalid_real_commits(history, message, error):
    repo, base = history
    commit(repo, message)
    result = check(repo, base)
    assert result.returncode == 1
    assert error in result.stderr


@pytest.mark.parametrize("title_valid", [False, True])
def test_event_uses_actual_head_and_treats_title_as_data(history, title_valid):
    repo, base = history
    head = commit(repo, f"ci: validate commit policy\n\n{SIGNOFF}\n")
    # A temporary unsigned merge at HEAD must not enter the PR contribution range.
    commit(repo, "GitHub temporary test merge")
    title = "ci: $(touch SENTINEL)" if title_valid else "Update commit policy"
    event = repo / "event.json"
    event.write_text(json.dumps({"pull_request": {"base": {"sha": base}, "head": {"sha": head}, "title": title}}))
    result = subprocess.run(
        [sys.executable, "-I", str(SCRIPT), "--repo", str(repo), "--event", str(event)],
        capture_output=True,
        text=True,
        cwd=repo,
    )
    assert result.returncode == (0 if title_valid else 1), result.stderr
    assert not (repo / "SENTINEL").exists()
    if not title_valid:
        assert "FAIL PR title" in result.stderr


@pytest.mark.parametrize("head", ["main", "missing-object", "--help"])
def test_uninspectable_or_empty_range_fails_closed(history, head):
    repo, base = history
    result = check(repo, base, head=head)
    assert result.returncode == 2
    assert "could not inspect the range" in result.stderr
