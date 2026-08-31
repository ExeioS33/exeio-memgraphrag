"""The read-only guard on the Cypher console.

These are the functions standing between a browser text box and a Neo4j instance
shared with another project's data, so they are tested directly rather than only
through the route.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from memgraphrag.api.routers.cypher import (
    DEFAULT_LIMIT,
    apply_limit,
    find_write_violation,
    strip_literals_and_comments,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    "statement,keyword",
    [
        ("MATCH (n) DETACH DELETE n", "DETACH"),
        ("CREATE (n:Passage {content: 'x'})", "CREATE"),
        ("MATCH (n) SET n.content = 'x'", "SET"),
        ("MERGE (n:Entity {entity_id: 'a'})", "MERGE"),
        ("MATCH (n) REMOVE n.label", "REMOVE"),
        ("DROP INDEX foo", "DROP"),
        ("LOAD CSV FROM 'file:///etc/passwd' AS line RETURN line", "LOAD CSV"),
    ],
)
def test_write_statements_are_rejected(statement: str, keyword: str) -> None:
    violation = find_write_violation(strip_literals_and_comments(statement))
    assert violation is not None
    assert keyword.split()[0].upper() in violation.upper()


@pytest.mark.parametrize(
    "statement",
    [
        "MATCH p=()-[:ENTITY_TO_TYPE]->() RETURN p LIMIT 25",
        "MATCH (n:Passage) RETURN n.entity_id, n.content LIMIT 10",
        "MATCH (a:Entity)-[r:ENTITY_RELATION]-(b:Entity) RETURN a, r, b LIMIT 5",
        "MATCH (n) RETURN count(n) AS total",
    ],
)
def test_read_statements_are_allowed(statement: str) -> None:
    assert find_write_violation(strip_literals_and_comments(statement)) is None


def test_a_write_keyword_inside_a_string_literal_is_not_a_violation() -> None:
    """Scanning the raw text would reject a legitimate search for the word."""
    statement = "MATCH (n:Passage) WHERE n.content CONTAINS 'DELETE the invoice' RETURN n"
    assert find_write_violation(strip_literals_and_comments(statement)) is None


def test_a_write_keyword_inside_a_comment_is_not_a_violation() -> None:
    statement = "// CREATE was here\nMATCH (n) RETURN n"
    assert find_write_violation(strip_literals_and_comments(statement)) is None


def test_property_names_are_not_mistaken_for_keywords() -> None:
    """`created_at` and `n.set` contain write keywords as substrings."""
    statement = "MATCH (n) RETURN n.created_at, n.setting, n.dropped LIMIT 5"
    assert find_write_violation(strip_literals_and_comments(statement)) is None


def test_limit_is_injected_when_absent() -> None:
    statement = "MATCH (n:Passage) RETURN n"
    rewritten, applied = apply_limit(statement, strip_literals_and_comments(statement), 50)
    assert applied is True
    assert rewritten.rstrip().upper().endswith("LIMIT 50")


def test_an_existing_limit_is_left_alone() -> None:
    statement = "MATCH (n:Passage) RETURN n LIMIT 7"
    rewritten, applied = apply_limit(statement, strip_literals_and_comments(statement), 50)
    assert applied is False
    assert rewritten == statement


def test_default_limit_is_bounded() -> None:
    assert 0 < DEFAULT_LIMIT <= 5000
