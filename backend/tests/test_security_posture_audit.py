from pathlib import Path

from evaluation.security_posture_audit import inspect_sources


def test_audit_flags_wildcard_cors_pickle_and_missing_auth(tmp_path: Path):
    (tmp_path / "api.py").write_text(
        'app.add_middleware(CORSMiddleware, allow_origins=["*"])\n',
        encoding="utf-8",
    )
    (tmp_path / "conversation_store.py").write_text(
        "pickle.loads(payload)\npickle.dumps(records)\n"
        "def store(session_id: str):\n    pass\n",
        encoding="utf-8",
    )

    report = inspect_sources(tmp_path)
    finding_ids = {item["id"] for item in report["findings"]}

    assert {
        "CORS_WILDCARD",
        "NO_VISIBLE_API_AUTH",
        "PICKLE_DESERIALIZATION",
        "PICKLE_SERIALIZATION",
        "SESSION_OWNERSHIP_NOT_MODELED",
    }.issubset(finding_ids)


def test_audit_does_not_flag_controls_that_are_present(tmp_path: Path):
    (tmp_path / "api.py").write_text(
        "from fastapi.security import HTTPBearer\n"
        "security = HTTPBearer()\n"
        "allow_origins=[\"https://example.test\"]\n",
        encoding="utf-8",
    )
    (tmp_path / "conversation_store.py").write_text(
        "def store(record):\n    return record\n",
        encoding="utf-8",
    )

    report = inspect_sources(tmp_path)
    assert report["findings"] == []


def test_current_backend_source_passes_static_security_audit():
    backend_root = Path(__file__).resolve().parents[1]
    report = inspect_sources(backend_root)
    assert report["findings"] == []
    assert report["status"] == "passed_static_checks"
