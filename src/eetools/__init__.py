"""eetools — reusable Google Earth Engine utilities for environmental remote sensing."""

import ee

from eetools._config import configure, get_project

__version__ = "0.1.0"
__all__ = ["configure", "initialize", "get_project", "__version__"]


def initialize(project: str | None = None, **kwargs) -> None:
    """Configure eetools and initialize the Earth Engine API.

    Args:
        project: Optional GEE project ID to register before initializing. If provided, overrides any previously configured project.
        **kwargs: Additional keyword arguments forwarded to ee.Initialize.

    Returns:
        None. Initializes the Earth Engine session with the resolved project ID.
    """
    if project is not None:
        configure(project=project)

    resolved = get_project()
    if resolved is not None:
        ee.Initialize(project=resolved, **kwargs)
    else:
        ee.Initialize(**kwargs)
