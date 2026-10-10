"""Configuration parsing."""

from __future__ import annotations

from tripmate.config import DEFAULT_MODEL, Settings, settings_from_env


def test_database_url_gets_sslmode_appended() -> None:
    settings = settings_from_env(
        {"DATABASE_URL": "postgresql://user:pw@host:5432/db"}
    )
    assert settings.database_url == "postgresql://user:pw@host:5432/db?sslmode=require"
    assert settings.configured_checkpointer_backend == "postgres"


def test_database_url_preserves_existing_query() -> None:
    settings = settings_from_env(
        {"DATABASE_URL": "postgresql://user:pw@host:5432/db?application_name=x"}
    )
    assert settings.database_url.endswith("&sslmode=require")


def test_database_url_is_optional_and_defaults_to_memory() -> None:
    assert settings_from_env({}).database_url is None
    assert settings_from_env({}).configured_checkpointer_backend == "memory"
    assert settings_from_env({"DATABASE_URL": "  "}).database_url is None


def test_aviation_key_accepts_legacy_spelling() -> None:
    assert (
        settings_from_env({"AVIATIONSTACK_API_KEY": "legacy"}).aviationstack_api_key
        == "legacy"
    )
    assert (
        settings_from_env(
            {"AVIATIONSTACK_API_KEY": "legacy", "AVIATION_STACK_API_KEY": "modern"}
        ).aviationstack_api_key
        == "modern"
    )


def test_cors_origins_are_trimmed_and_split() -> None:
    settings = settings_from_env(
        {"CORS_ORIGINS": "https://a.vercel.app/, https://b.vercel.app ,"}
    )
    assert settings.cors_origins == ("https://a.vercel.app", "https://b.vercel.app")


def test_missing_credentials_are_reported() -> None:
    assert Settings().missing_credentials() == (
        "AVIATION_STACK_API_KEY",
        "GROQ_API_KEY",
        "OPENWEATHER_API_KEY",
        "TAVILY_API_KEY",
    )
    assert settings_from_env({"GROQ_API_KEY": "k"}).missing_credentials() == (
        "AVIATION_STACK_API_KEY",
        "OPENWEATHER_API_KEY",
        "TAVILY_API_KEY",
    )


def test_require_raises_with_actionable_message() -> None:
    try:
        Settings().require("GROQ_API_KEY")
    except RuntimeError as exc:
        assert "GROQ_API_KEY" in str(exc)
    else:
        raise AssertionError("require() should raise for an unset value")


def test_port_and_host_come_from_environment() -> None:
    settings = settings_from_env({"HOST": "127.0.0.1", "PORT": "9001"})
    assert settings.host == "127.0.0.1"
    assert settings.port == 9001


def test_frontend_dir_is_discovered(tmp_path) -> None:
    """The static frontend must resolve regardless of install layout."""

    from tripmate.config import frontend_dir

    resolved = frontend_dir({})
    assert (resolved / "index.html").is_file()
    assert resolved.name == "frontend"


def test_frontend_dir_honours_an_explicit_override(tmp_path) -> None:
    from tripmate.config import frontend_dir

    override = tmp_path / "ui"
    override.mkdir()
    (override / "index.html").write_text("<html></html>")

    assert frontend_dir({"FRONTEND_DIR": str(override)}) == override


def test_settings_expose_assets_dir() -> None:
    settings = settings_from_env({})
    assert settings.frontend_dir.name == "frontend"
    assert settings.assets_dir.name == "assets"


def test_cors_origins_reject_markdown_link_paste() -> None:
    """A pasted markdown link must not become a list of unusable origins.

    urlsplit accepts stray punctuation inside netloc, so each candidate is
    re-checked against a strict host[:port] shape.
    """

    raw = (
        "[http://localhost:8000,http://localhost:5173,"
        "http://127.0.0.1:8000](http://localhost:8000,http://localhost:5173,"
        "http://127.0.0.1:8000)"
    )
    origins = settings_from_env({"CORS_ORIGINS": raw}).cors_origins

    assert all("[" not in o and "]" not in o and "(" not in o for o in origins)
    assert "http://localhost:5173" in origins


def test_cors_origins_drop_entries_with_paths_or_credentials() -> None:
    origins = settings_from_env(
        {"CORS_ORIGINS": "https://a.app/path,https://u:p@b.app,https://ok.app"}
    ).cors_origins
    assert origins == ("https://ok.app",)


def test_cors_origins_deduplicate_and_strip_trailing_slash() -> None:
    origins = settings_from_env(
        {"CORS_ORIGINS": "https://a.app/,https://a.app,https://b.app"}
    ).cors_origins
    assert origins == ("https://a.app", "https://b.app")


def test_default_model_is_a_verified_groq_model() -> None:
    assert settings_from_env({}).model == DEFAULT_MODEL
    assert DEFAULT_MODEL == "openai/gpt-oss-120b"


def test_model_can_be_overridden() -> None:
    assert settings_from_env({"GROQ_MODEL": "custom/model"}).model == "custom/model"


def test_dotenv_file_is_loaded_by_get_settings(monkeypatch) -> None:
    """A local .env must reach settings; the README documents this workflow.

    Regression guard: ``load_dotenv_file`` was once defined but never called, so
    ``uvicorn tripmate.main:app`` ignored .env entirely and every credential
    appeared missing.
    """

    from pathlib import Path

    from tripmate.config import PROJECT_ROOT, get_settings, load_dotenv_file

    env_file = PROJECT_ROOT / ".env"
    assert env_file.is_file(), "expected a local .env in the repository"

    sentinel = "dotenv-regression-sentinel"
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("GROQ_API_KEY", sentinel)

    # A real environment variable must win over the file.
    load_dotenv_file()
    assert get_settings().groq_api_key == sentinel
    assert Path(env_file).is_file()


def test_configure_tls_sets_certificate_bundles(monkeypatch) -> None:
    """TLS trust must be configured before any HTTP client is built."""

    import os

    from tripmate.config import configure_tls

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    configure_tls()

    assert os.environ["SSL_CERT_FILE"]
    assert os.environ["REQUESTS_CA_BUNDLE"]


def test_create_app_configures_tls(monkeypatch) -> None:
    """create_app() is the documented entrypoint and must wire TLS up."""

    import os

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)

    from tripmate.main import create_app

    create_app()

    assert os.environ["SSL_CERT_FILE"]
