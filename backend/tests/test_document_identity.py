from src.document_identity import document_id_for, chunk_id_for


def test_document_id_is_stable_for_same_content():
    assert document_id_for("same document") == document_id_for("same document")


def test_document_id_changes_when_content_changes():
    assert document_id_for("document A") != document_id_for("document B")


def test_chunk_id_is_stable_and_position_sensitive():
    doc_id = document_id_for("document")
    first = chunk_id_for(doc_id, 0, "first chunk")
    assert first == chunk_id_for(doc_id, 0, "first chunk")
    assert first != chunk_id_for(doc_id, 1, "first chunk")
    assert first != chunk_id_for(doc_id, 0, "changed chunk")
