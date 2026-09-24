"""HttpPolicy: request-size limit and opt-in CORS."""
import pytest
from flask import Flask, request

from src.server.http_policy import CORS_ORIGINS_ENV, FLASK_DEBUG_ENV, MAX_CONTENT_LENGTH_ENV, HttpPolicy

ORIGIN = "https://app.example.com"


def _client(policy: HttpPolicy):
    app = Flask(__name__)
    policy.apply(app)
    # The limit is enforced when the body is read, as every endpoint does.
    app.add_url_rule("/echo", "echo", lambda: request.get_data() or "ok", methods=["POST"])
    return app.test_client()


def test_oversized_body_is_rejected_with_413():
    client = _client(HttpPolicy(max_content_length=10))
    assert client.post("/echo", data=b"x" * 11).status_code == 413
    assert client.post("/echo", data=b"x" * 10).status_code == 200


def test_no_cors_headers_by_default(monkeypatch):
    monkeypatch.delenv(CORS_ORIGINS_ENV, raising=False)
    client = _client(HttpPolicy.from_environment(default_max_bytes=1024))
    response = client.post("/echo", headers={"Origin": ORIGIN})
    assert "Access-Control-Allow-Origin" not in response.headers


def test_configured_origin_only(monkeypatch):
    monkeypatch.setenv(CORS_ORIGINS_ENV, f" {ORIGIN}/ ")
    client = _client(HttpPolicy.from_environment(default_max_bytes=1024))
    assert client.post("/echo", headers={"Origin": ORIGIN}).headers["Access-Control-Allow-Origin"] == ORIGIN
    assert "Access-Control-Allow-Origin" not in client.post("/echo", headers={"Origin": "https://evil.example"}).headers


def test_max_bytes_from_environment(monkeypatch):
    monkeypatch.setenv(MAX_CONTENT_LENGTH_ENV, "2048")
    assert HttpPolicy.from_environment(default_max_bytes=1024).max_content_length == 2048
    monkeypatch.setenv(MAX_CONTENT_LENGTH_ENV, "0")
    with pytest.raises(ValueError):
        HttpPolicy.from_environment(default_max_bytes=1024)


@pytest.mark.parametrize("value,expected", [(None, False), ("false", False), ("true", True), ("1", True)])
def test_debug_is_opt_in(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv(FLASK_DEBUG_ENV, raising=False)
    else:
        monkeypatch.setenv(FLASK_DEBUG_ENV, value)
    assert HttpPolicy.debug_enabled() is expected
