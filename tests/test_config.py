"""Tests for eetools._config global project-id state."""

from eetools import _config


def test_default_project_is_none():
    # The autouse reset fixture clears any prior configure() call.
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
