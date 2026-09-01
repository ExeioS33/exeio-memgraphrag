"""Containment of the library's path handling.

Every library route takes a caller-supplied path. `_safe_path` is the only place
one is built, so it is the whole perimeter — a hole here reads any file the server
process can reach.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException

from memgraphrag.api.routers.library import _safe_path

pytestmark = pytest.mark.offline


@pytest.fixture()
def library(tmp_path):
    root = tmp_path / "library"
    (root / "sub").mkdir(parents=True)
    (root / "a.pdf").write_bytes(b"%PDF-1.4 a")
    (root / "sub" / "b.pdf").write_bytes(b"%PDF-1.4 b")
    (tmp_path / "secret.txt").write_text("not for the browser")
    return root.resolve()


def test_a_plain_relative_path_resolves(library) -> None:
    assert _safe_path(library, "a.pdf") == library / "a.pdf"
    assert _safe_path(library, "sub/b.pdf") == library / "sub" / "b.pdf"


@pytest.mark.parametrize(
    "attempt",
    [
        "../secret.txt",
        "sub/../../secret.txt",
        "../../etc/passwd",
        "/etc/passwd",
        "\\etc\\passwd",
    ],
)
def test_traversal_and_absolute_paths_are_refused(library, attempt: str) -> None:
    with pytest.raises(HTTPException) as exc:
        _safe_path(library, attempt)
    assert exc.value.status_code == 400


def test_empty_path_is_refused(library) -> None:
    with pytest.raises(HTTPException) as exc:
        _safe_path(library, "   ")
    assert exc.value.status_code == 400


def test_null_byte_is_refused(library) -> None:
    """A NUL truncates the path in some C-level calls; refuse before it gets there."""
    with pytest.raises(HTTPException):
        _safe_path(library, "a.pdf\x00.png")


def test_a_symlink_escaping_the_root_is_refused(library, tmp_path) -> None:
    """`..` rejection alone misses this: the path has no `..` in it at all."""
    escape = library / "escape.txt"
    escape.symlink_to(tmp_path / "secret.txt")
    with pytest.raises(HTTPException) as exc:
        _safe_path(library, "escape.txt")
    assert exc.value.status_code == 400


def test_a_symlink_staying_inside_the_root_is_allowed(library) -> None:
    alias = library / "alias.pdf"
    alias.symlink_to(library / "sub" / "b.pdf")
    assert _safe_path(library, "alias.pdf") == (library / "sub" / "b.pdf").resolve()
