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

    Args:
        None.

    Returns:
        The project ID string, or None if configure has not been called.
    """
    return _state["project_id"]
