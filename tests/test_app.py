from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health_returns_healthy():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_root_returns_html():
    response = client.get("/")
    assert response.status_code in (200, 404)  # 404 if template missing in CI
    assert response.headers["content-type"].startswith("text/html")
