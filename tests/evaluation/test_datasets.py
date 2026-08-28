"""Dataset loaders against miniatures of both real shapes, and clean degradation.

The fixtures below reproduce the record structure verified in the research
checkout (flat lists for HotpotQA / 2WikiMultihopQA, a ``{source, questions}``
wrapper for MuSiQue / medical). They are written here rather than read from
``MemGraphRAG/dataset/`` so the suite stays hermetic: no test may need the 58 MB
research tree to pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memgraphrag.evaluation.datasets import (
    DATASET_ROOT_ENV,
    DatasetFormatError,
    DatasetUnavailableError,
    available_datasets,
    dataset_root,
    load_corpus,
    load_questions,
    sample_examples,
)

pytestmark = pytest.mark.offline

HOTPOT = [
    {
        "_id": "5abe953b",
        "answer": "superhero roles as the Marvel Comics",
        "question": "What is one of the stars of The Newcomers known for?",
        "supporting_facts": [["The Newcomers (film)", 0], ["Chris Evans (actor)", 1]],
        "context": [["The Newcomers (film)", ["A 2000 film."]]],
        "type": "bridge",
        "level": "hard",
    },
    {
        "_id": "5ac0d3e1",
        "answer": "yes",
        "question": "Are both films documentaries?",
        # Same title twice: gold documents are a set of documents, not of sentences.
        "supporting_facts": [["Doc A", 0], ["Doc A", 3]],
        "context": [],
        "type": "comparison",
        "level": "medium",
    },
]

MUSIQUE = [
    {
        "source": "musique",
        "questions": {
            "type1": [
                {
                    "id": "2hop__13548_13529",
                    "question": "When was the person Messi was compared to signed?",
                    "answer": "June 1982",
                    "answer_aliases": ["1982"],
                    "answerable": True,
                    "paragraphs": [
                        {
                            "idx": 0,
                            "title": "Lionel Messi",
                            "paragraph_text": "…",
                            "is_supporting": False,
                        },
                        {
                            "idx": 1,
                            "title": "FC Barcelona",
                            "paragraph_text": "…",
                            "is_supporting": True,
                        },
                    ],
                }
            ]
        },
    }
]

MEDICAL = [
    {
        "source": "nccn",
        "questions": {
            "type1": [
                {
                    "id": "type1_0",
                    "source": "60_Basal Cell Skin Cancer_processed",
                    "question": "What is the most common type of skin cancer?",
                    "answer": "Basal cell carcinoma (BCC) is the most common type of skin cancer.",
                    "evidence": "BCC is the most common type of skin cancer",
                }
            ],
            "type2": [
                {
                    "id": "type2_0",
                    "source": "60_Basal Cell Skin Cancer_processed",
                    "question": "How is it treated?",
                    "answer": "Usually with surgery.",
                    "evidence": "Treatment usually involves surgery",
                }
            ],
        },
    }
]


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A dataset root holding a miniature of each of the four datasets."""
    (tmp_path / "hotpotqa").mkdir()
    (tmp_path / "hotpotqa" / "hotpotqa.json").write_text(json.dumps(HOTPOT), encoding="utf-8")
    (tmp_path / "hotpotqa" / "hotpotqa_corpus.json").write_text(
        json.dumps(
            [{"idx": 0, "title": "Doc A", "text": "Body A"}, {"title": "Doc B", "text": ""}]
        ),
        encoding="utf-8",
    )
    (tmp_path / "2wikimutlhopqa").mkdir()
    (tmp_path / "2wikimutlhopqa" / "2wikimultihopqa.json").write_text(
        json.dumps(HOTPOT), encoding="utf-8"
    )
    (tmp_path / "musique").mkdir()
    (tmp_path / "musique" / "musique.json").write_text(json.dumps(MUSIQUE), encoding="utf-8")
    (tmp_path / "medical").mkdir()
    (tmp_path / "medical" / "question.json").write_text(json.dumps(MEDICAL), encoding="utf-8")
    (tmp_path / "medical" / "medical.txt").write_text(
        "About basal cell skin cancer…", encoding="utf-8"
    )
    return tmp_path


def test_flat_dataset_yields_questions_gold_answers_and_unique_gold_docs(root: Path) -> None:
    examples = load_questions("hotpotqa", root=root)
    assert [example.id for example in examples] == ["5abe953b", "5ac0d3e1"]
    assert examples[0].gold_answers == ["superhero roles as the Marvel Comics"]
    assert examples[0].gold_docs == ["The Newcomers (film)", "Chris Evans (actor)"]
    # Two supporting sentences from one document are one gold document.
    assert examples[1].gold_docs == ["Doc A"]
    assert examples[0].question_type == "bridge"


def test_misspelled_research_directory_is_found_by_the_canonical_name(root: Path) -> None:
    """The checkout spells it "2wikimutlhopqa"; callers should not have to."""
    assert load_questions("2wikimultihopqa", root=root)[0].dataset == "2wikimultihopqa"
    assert load_questions("2wiki", root=root)[0].dataset == "2wikimultihopqa"


def test_grouped_dataset_reads_aliases_and_supporting_paragraphs(root: Path) -> None:
    example = load_questions("musique", root=root)[0]
    assert example.gold_answers == ["June 1982", "1982"]
    assert example.gold_docs == ["FC Barcelona"]


def test_medical_records_fall_back_to_their_source_document(root: Path) -> None:
    examples = load_questions("medical", root=root)
    assert len(examples) == 2
    assert examples[0].gold_docs == ["60_Basal Cell Skin Cancer_processed"]
    assert examples[0].metadata["evidence"].startswith("BCC")


def test_question_type_filter_selects_one_group(root: Path) -> None:
    examples = load_questions("medical", root=root, question_types=["type2"])
    assert [example.id for example in examples] == ["type2_0"]


def test_limit_truncates_in_file_order_so_two_runs_see_the_same_questions(root: Path) -> None:
    assert [e.id for e in load_questions("hotpotqa", root=root, limit=1)] == ["5abe953b"]


def test_corpus_loader_skips_empty_documents(root: Path) -> None:
    docs = load_corpus("hotpotqa", root=root)
    assert [doc.title for doc in docs] == ["Doc A"]
    assert docs[0].to_chunk() == "Doc A\nBody A"


def test_medical_corpus_is_the_single_text_file(root: Path) -> None:
    """The medical set ships no corpus JSON; the loader must not invent a split."""
    docs = load_corpus("medical", root=root)
    assert len(docs) == 1 and docs[0].title == "medical"


def test_missing_checkout_names_the_path_and_the_override(tmp_path: Path) -> None:
    with pytest.raises(DatasetUnavailableError) as excinfo:
        load_questions("hotpotqa", root=tmp_path / "nowhere")
    message = str(excinfo.value)
    assert "nowhere" in message and DATASET_ROOT_ENV in message


def test_unknown_dataset_lists_the_known_ones(root: Path) -> None:
    with pytest.raises(DatasetUnavailableError) as excinfo:
        load_questions("triviaqa", root=root)
    assert "hotpotqa" in str(excinfo.value)


def test_root_comes_from_the_environment_when_not_passed(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(DATASET_ROOT_ENV, str(root))
    assert dataset_root() == root
    assert sorted(available_datasets()) == ["2wikimultihopqa", "hotpotqa", "medical", "musique"]


def test_available_datasets_is_empty_without_a_checkout(tmp_path: Path) -> None:
    assert available_datasets(tmp_path) == []


def test_record_without_a_question_is_a_format_error_not_a_silent_skip(tmp_path: Path) -> None:
    (tmp_path / "hotpotqa").mkdir()
    (tmp_path / "hotpotqa" / "hotpotqa.json").write_text(
        json.dumps([{"_id": "x", "answer": "y"}]), encoding="utf-8"
    )
    with pytest.raises(DatasetFormatError):
        load_questions("hotpotqa", root=tmp_path)


def test_sampling_is_seeded_and_keeps_dataset_order(root: Path) -> None:
    examples = load_questions("hotpotqa", root=root)
    first = sample_examples(examples, 1, seed=3)
    assert first == sample_examples(examples, 1, seed=3)
    assert sample_examples(examples, 5, seed=3) == examples
