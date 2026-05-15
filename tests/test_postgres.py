from app.core.config import settings
from app.db import postgres


def test_database_url_supports_postgres_alias(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgres://user:pass@localhost/db")

    assert postgres._database_url() == "postgresql+psycopg://user:pass@localhost/db"


def test_database_url_supports_postgresql_scheme(monkeypatch):
    monkeypatch.setattr(settings, "DATABASE_URL", "postgresql://user:pass@localhost/db")

    assert postgres._database_url() == "postgresql+psycopg://user:pass@localhost/db"
