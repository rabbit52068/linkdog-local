import pytest

from app.auth import require_token
from app.main import app


@pytest.fixture(autouse=True)
def _bypass_api_token(request):
    """Route tests exercise behaviour, not auth; tests/test_auth.py opts out."""
    if request.node.get_closest_marker("real_auth"):
        yield
        return
    app.dependency_overrides[require_token] = lambda: None
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_token, None)


def pytest_configure(config):
    config.addinivalue_line("markers", "real_auth: run with the real token check")


@pytest.fixture(autouse=True)
def _isolate_chat_history(tmp_path, monkeypatch):
    """Never let a test read or write the real data/history.json."""
    import app.main as main

    monkeypatch.setattr(main, "HISTORY_PATH", tmp_path / "history.json")
