import os

# Load a local .env (if present) into the process environment so EE_PROJECT /
# GOOGLE_CLOUD_PROJECT can be supplied without exporting them in the shell. The .env
# file is gitignored and never committed. python-dotenv is a dev-only convenience;
# guard the import so `import eetools` still works for consumers that don't install it
# (they set the project explicitly or via a real environment variable).
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

# Environment variables consulted, in order, when no project has been set explicitly
# via configure() / initialize(project=...).
_PROJECT_ENV_VARS = ("EE_PROJECT", "GOOGLE_CLOUD_PROJECT")

_state: dict = {"project_id": None}


def configure(project: str | None = None) -> None:
    """Set the global eetools project ID used for EE asset paths and exports.

    Args:
        project: GEE project ID string to store globally. Ignored if None.

    Returns:
        None. Updates the module-level project ID in place.
    """
    if project is not None:
        _state["project_id"] = project


def get_project() -> str | None:
    """Return the currently configured GEE project ID.

    Resolution order: a value set explicitly via configure() / initialize(project=...)
    takes precedence; otherwise the EE_PROJECT (then GOOGLE_CLOUD_PROJECT) environment
    variable is used, which may be supplied via a local .env file.

    Args:
        None.

    Returns:
        The project ID string, or None if neither an explicit value nor a matching environment variable is set.
    """
    if _state["project_id"] is not None:
        return _state["project_id"]
    for var in _PROJECT_ENV_VARS:
        value = os.environ.get(var)
        if value:
            return value
    return None
