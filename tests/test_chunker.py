from backend.ingestion.chunker import FixedChunker


def test_fixed_chunker_respects_size_and_overlap():
    chunker = FixedChunker(chunk_size=5, overlap=2)
    chunks = chunker.split(
        "one two three four five six seven eight nine ten eleven twelve",
        3,
        {"document_name": "sample.pdf", "ingestion_timestamp": "now"},
    )
    original_tokens = chunker.encoder.encode(
        "one two three four five six seven eight nine ten eleven twelve"
    )

    assert len(chunks) > 1
    assert chunks[0].text == chunker.encoder.decode(original_tokens[:5]).strip()
    assert chunks[1].text == chunker.encoder.decode(original_tokens[3:8]).strip()
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert all(chunk.metadata["document_name"] == "sample.pdf" for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)


def test_chunker_rejects_invalid_overlap():
    try:
        FixedChunker(chunk_size=5, overlap=5)
    except ValueError:
        return
    raise AssertionError("Expected overlap equal to chunk size to fail")
