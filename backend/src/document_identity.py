import hashlib


def document_id_for(text):
    """Stable content identity, independent of filename and upload order."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_id_for(document_id, chunk_index, chunk_text):
    """Stable chunk identity within a content-addressed document."""
    payload = "{}:{}:{}".format(document_id, chunk_index, chunk_text)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
