"""Rename-tolerant repository identity for managed checkouts."""

import pytest

from embodirun.deployment.source import repository_identity


@pytest.mark.parametrize(
    ("reference", "expected"),
    [
        ("https://github.com/BUAA-CI-LAB/EmbodiRun.git", "EmbodiRun"),
        ("git@github.com:BUAA-CI-LAB/EmbodiRun.git", "EmbodiRun"),
        ("https://github.com/BUAA-CI-LAB/EmbodiRun-internal.git", "EmbodiRun"),
        ("https://github.com/BUAA-CI-LAB/EmbodiInfer.git", "EmbodiInfer"),
        ("git@github.com:BUAA-CI-LAB/EmbodiInfer-internal.git", "EmbodiInfer"),
    ],
)
def test_renamed_repositories_share_one_identity(reference, expected) -> None:
    assert repository_identity(reference) == expected


def test_unrelated_origins_are_not_rewritten() -> None:
    assert repository_identity("https://example.com/other/repo.git") == ("https://example.com/other/repo")
