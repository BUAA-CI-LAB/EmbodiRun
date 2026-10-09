"""Check Conventional Commit subjects and author DCO sign-offs in a Git range."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

TYPES = ("feat", "fix", "docs", "style", "refactor", "perf", "test", "build", "ci", "chore", "revert")
SUBJECT = re.compile(r"([a-z]+)(?:\(([a-z0-9][a-z0-9._/-]*)\))?(!)?: ([^\s].*)")
IDENTITY = re.compile(r"([^<>\r\n]+) <([^<>\s@]+@[^<>\s@]+)>")


def git(repo: Path, *args: str, message: str | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        input=message,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout


def subject_errors(subject: str) -> list[str]:
    errors = []
    match = SUBJECT.fullmatch(subject)
    if match is None or match[1] not in TYPES:
        errors.append("use <type>[(scope)][!]: <description>; types: " + ", ".join(TYPES))
    if len(subject) > 72:
        errors.append("subject must be at most 72 characters, including type and scope")
    if subject != subject.strip() or any(ord(char) < 32 for char in subject):
        errors.append("subject must be one line without control characters or surrounding whitespace")
    if subject.endswith((".", "。")):
        errors.append("subject must not end with a full stop")
    return errors


def commit_errors(repo: Path, message: str, author_name: str, author_email: str) -> list[str]:
    lines = message.splitlines()
    errors = subject_errors(lines[0] if lines else "")
    if len(lines) > 1 and lines[1] != "":
        errors.append("separate the subject from the body/trailers with an empty line")
    if any(line != line.rstrip() for line in lines):
        errors.append("commit message lines must not have trailing whitespace")

    trailers = git(repo, "interpret-trailers", "--parse", "--no-divider", message=message)
    identities = set()
    coauthors = set()
    for line in trailers.splitlines():
        key, _, value = line.partition(": ")
        if key.lower() not in {"signed-off-by", "co-authored-by"}:
            continue
        identity = IDENTITY.fullmatch(value)
        if identity is None:
            errors.append(f"{key} must have the form Name <email@example.com>")
            continue
        name, email = identity.groups()
        if key.lower() == "signed-off-by":
            if key != "Signed-off-by":
                errors.append("use the exact trailer spelling Signed-off-by")
            identities.add((name, email.lower()))
        else:
            coauthors.add((name, email.lower()))

    for name, email in sorted({(author_name, author_email.lower()), *coauthors}):
        if (name, email) not in identities:
            errors.append(f"missing author sign-off in the final trailers: Signed-off-by: {name} <{email}>")
    return errors


def check_range(repo: Path, base: str, head: str, *, title: str | None = None) -> list[tuple[str, list[str]]]:
    """Read every introduced commit, including authored merges, without checking out code."""
    base = git(repo, "rev-parse", "--verify", "--end-of-options", f"{base}^{{commit}}").strip()
    head = git(repo, "rev-parse", "--verify", "--end-of-options", f"{head}^{{commit}}").strip()
    git(repo, "merge-base", base, head)
    commits = git(repo, "rev-list", "--reverse", f"{base}..{head}").splitlines()
    if not commits:
        raise ValueError("the range contains no new commits; supply the PR base and actual head")
    results = []
    if title is not None:
        results.append(("PR title", subject_errors(title)))
    for commit in commits:
        name, email, message = git(repo, "show", "-s", "--format=%an%x00%ae%x00%B", commit).split("\0", 2)
        results.append((commit, commit_errors(repo, message, name, email)))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", help="base revision, usually origin/main")
    parser.add_argument("--head", help="actual contribution head, usually HEAD")
    parser.add_argument("--title", help="also check the proposed PR/squash subject")
    parser.add_argument("--event", type=Path, help="GitHub pull_request event JSON containing the base, head and title")
    args = parser.parse_args(argv)
    if args.event and any(value is not None for value in (args.base, args.head, args.title)):
        parser.error("use --event or --base/--head/--title, not both")
    if not args.event and (args.base is None or args.head is None):
        parser.error("supply --base and --head, or --event")
    try:
        if args.event:
            pr = json.loads(args.event.read_text(encoding="utf-8"))["pull_request"]
            args.base, args.head, args.title = pr["base"]["sha"], pr["head"]["sha"], pr["title"]
            if not all(isinstance(value, str) and value for value in (args.base, args.head, args.title)):
                raise ValueError("pull_request base/head SHA and title must be nonempty strings")
        results = check_range(args.repo, args.base, args.head, title=args.title)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(f"Commit policy could not inspect the range: {error}", file=sys.stderr)
        return 2
    failed = False
    for label, errors in results:
        if errors:
            failed = True
            print(f"FAIL {label}", file=sys.stderr)
            for error in errors:
                print("  " + error, file=sys.stderr)
    if failed:
        print("See CONTRIBUTING.md#commit-messages-and-sign-off for the rules and repair commands.", file=sys.stderr)
        return 1
    count = sum(label != "PR title" for label, _ in results)
    print(f"Commit policy passed for {count} commit(s)" + (" and the PR title." if args.title is not None else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
