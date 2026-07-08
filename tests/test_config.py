"""Tests for eetools._config global project-id state.

get_project() resolves an explicit configure() value first, then falls back to the
EE_PROJECT / GOOGLE_CLOUD_PROJECT environment variables (which a local .env may
supply). Env-dependent tests clear/set those vars via monkeypatch so they don't
depend on the developer's real .env.
"""

from eetools import _config


def test_default_project_is_none(monkeypatch):
    # No explicit project and no environment variables -> None.
    monkeypatch.delenv("EE_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    _config._state["project_id"] = None
    assert _config.get_project() is None


def test_configure_sets_project():
    _config.configure(project="my-gee-project")
    assert _config.get_project() == "my-gee-project"


def test_configure_none_is_ignored():
    _config.configure(project="first-project")
    _config.configure(project=None)
    # Passing None must not wipe a previously configured project.
    assert _config.get_project() == "first-project"


def test_configure_overrides_previous():
    _config.configure(project="first-project")
    _config.configure(project="second-project")
    assert _config.get_project() == "second-project"


# --------------------------------------------------------------------------- #
# Environment-variable fallback
# --------------------------------------------------------------------------- #
def test_env_var_fallback_when_project_unset(monkeypatch):
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.setenv("EE_PROJECT", "env-project")
    _config._state["project_id"] = None
    assert _config.get_project() == "env-project"


def test_google_cloud_project_fallback(monkeypatch):
    monkeypatch.delenv("EE_PROJECT", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gcp-project")
    _config._state["project_id"] = None
    assert _config.get_project() == "gcp-project"


def test_ee_project_precedes_google_cloud_project(monkeypatch):
    monkeypatch.setenv("EE_PROJECT", "ee-project")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "gcp-project")
    _config._state["project_id"] = None
    assert _config.get_project() == "ee-project"


def test_explicit_project_overrides_env(monkeypatch):
    monkeypatch.setenv("EE_PROJECT", "env-project")
    _config.configure(project="explicit-project")
    assert _config.get_project() == "explicit-project"
