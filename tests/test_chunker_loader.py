import pytest

from app.chunker import chunk_documents, chunk_documents_by_paragraph, chunk_with_strategy
from app.errors import ConfigurationError
from app.loader import load_documents


def doc(text, source="a.md"):
    return {"source": source, "content": text}


def test_short_document_is_a_single_chunk():
    chunks = chunk_documents([doc("hello world")], chunk_size=500, overlap=50)
    assert chunks == [{"source": "a.md", "content": "hello world", "chunk_index": 0}]


def test_windows_advance_by_size_minus_overlap():
    text = "".join(str(i % 10) for i in range(25))
    chunks = chunk_documents([doc(text)], chunk_size=10, overlap=3)
    assert [c["content"] for c in chunks] == [text[0:10], text[7:17], text[14:24], text[21:25]]
    # consecutive chunks share exactly `overlap` characters
    assert chunks[0]["content"][-3:] == chunks[1]["content"][:3]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2, 3]


def test_no_redundant_tail_chunk_when_window_reaches_end():
    # Originally a 10-char text with size 10 / overlap 3 produced a second
    # chunk text[7:10] fully contained in the first one.
    chunks = chunk_documents([doc("0123456789")], chunk_size=10, overlap=3)
    assert [c["content"] for c in chunks] == ["0123456789"]


def test_whitespace_only_and_empty_inputs():
    assert chunk_documents([doc("   \n  ")]) == []
    assert chunk_documents([doc("")]) == []
    assert chunk_documents([]) == []


@pytest.mark.parametrize("size,overlap", [(10, 10), (10, 11), (0, 0), (10, -1)])
def test_invalid_parameters_raise(size, overlap):
    with pytest.raises(ValueError):
        chunk_documents([doc("x")], chunk_size=size, overlap=overlap)


def test_invalid_parameters_raise_even_without_documents():
    with pytest.raises(ValueError):
        chunk_documents([], chunk_size=10, overlap=10)


def test_paragraph_strategy_keeps_heading_context():
    text = "# Refund Policy\n\nFirst rule.\n\nSecond rule.\n"
    chunks = chunk_documents_by_paragraph([doc(text)], chunk_size=500)
    assert [c["content"] for c in chunks] == ["# Refund Policy\nFirst rule.", "# Refund Policy\nSecond rule."]
    assert [c["chunk_index"] for c in chunks] == [0, 1]


def test_paragraph_strategy_splits_long_paragraphs():
    chunks = chunk_documents_by_paragraph([doc("x" * 25)], chunk_size=10, overlap=0)
    assert all(len(c["content"]) <= 10 for c in chunks) and "".join(c["content"] for c in chunks) == "x" * 25


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunk_with_strategy([doc("x")], "semantic", 10, 0)


def test_loader_reads_real_knowledge_base_sorted():
    docs = load_documents()
    assert [d["source"] for d in docs] == ["account_policy.md", "refund_policy.md", "shipping_policy.md"]
    assert all(d["content"].strip() for d in docs)


def test_loader_skips_empty_files_and_non_markdown(tmp_path):
    (tmp_path / "b.md").write_text("B", encoding="utf-8")
    (tmp_path / "a.md").write_text("A", encoding="utf-8")
    (tmp_path / "empty.md").write_text("  \n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    assert [d["source"] for d in load_documents(tmp_path)] == ["a.md", "b.md"]


def test_loader_errors_are_actionable(tmp_path):
    with pytest.raises(ConfigurationError, match="not found"):
        load_documents(tmp_path / "missing")
    with pytest.raises(ConfigurationError, match="No non-empty .md"):
        load_documents(tmp_path)
