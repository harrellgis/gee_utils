"""Resolution and pre-flight checks for user-uploaded Earth Engine assets.

The VECEA and SoilGrids WRB layers are uploaded assets, not catalog datasets, so code
running under another Cloud project fails unless that account can read them. Asset IDs
resolve as: explicit argument -> environment variable -> constant default.
"""

import os

import ee

from eetools.constants import UPLOADED_ASSETS, VECEA_REPRODUCE_STEPS


class SiteSelectionAssetError(RuntimeError):
    """Raised when a required uploaded asset cannot be read by the current session."""


def resolve_asset_id(key: str, asset_id: str | None = None) -> str:
    """Resolve the Earth Engine ID of an uploaded asset.

    Args:
        key: Asset key in ``constants.UPLOADED_ASSETS`` ('vecea' or 'soilgrids_wrb').
        asset_id: Optional explicit asset ID; wins over the environment and the default.

    Returns:
        The asset ID string: ``asset_id`` if given, else the key's environment variable
        (``EETOOLS_VECEA_ASSET`` / ``EETOOLS_SOILGRIDS_WRB_ASSET``) if set, else the
        constant default.

    Raises:
        KeyError: If ``key`` is not a known uploaded asset.
    """
    if key not in UPLOADED_ASSETS:
        raise KeyError(
            f"Unknown uploaded asset key '{key}'. Expected one of "
            f"{sorted(UPLOADED_ASSETS)}."
        )
    if asset_id:
        return asset_id
    default, env_var = UPLOADED_ASSETS[key]
    return os.environ.get(env_var) or default


def check_required_assets(
    asset_overrides: dict[str, str] | None = None,
    keys: tuple[str, ...] = ("vecea", "soilgrids_wrb"),
) -> dict[str, str]:
    """Pre-flight check that every required uploaded asset is readable.

    Calls ``ee.data.getAsset`` once per asset, so a permission or not-found problem is
    reported up front with the remedies, instead of as an opaque error deep inside a
    reduction.

    Args:
        asset_overrides: Optional ``{key: asset_id}`` overrides (e.g. ``{"vecea": ...}``).
        keys: Asset keys to check (default both VECEA and SoilGrids WRB).

    Returns:
        Dict mapping each key to the resolved, readable asset ID.

    Raises:
        SiteSelectionAssetError: If an asset cannot be read, naming the asset, the
            session's project, and the remedies (share it, or copy it with
            ``ee.data.copyAsset`` and point the override at the copy).
    """
    overrides = asset_overrides or {}
    unknown = sorted(set(overrides) - set(UPLOADED_ASSETS))
    if unknown:
        raise KeyError(
            f"Unknown asset_overrides key(s) {unknown}; expected {sorted(UPLOADED_ASSETS)}."
        )
    resolved: dict[str, str] = {}
    for key in keys:
        asset_id = resolve_asset_id(key, overrides.get(key))
        try:
            ee.data.getAsset(asset_id)
        except Exception as exc:  # noqa: BLE001 - EE raises a generic EEException
            try:
                from eetools._config import get_project

                project = get_project() or "<default project>"
            except Exception:  # noqa: BLE001
                project = "<unknown>"
            hint = (
                f" To rebuild VECEA yourself: {VECEA_REPRODUCE_STEPS}"
                if key == "vecea"
                else ""
            )
            raise SiteSelectionAssetError(
                f"Cannot read the '{key}' asset '{asset_id}' from Earth Engine project "
                f"'{project}' ({exc}). Remedies: (a) ask the owner to share it "
                f"('Anyone can read' or your account), or (b) copy it with "
                f"ee.data.copyAsset(src, dst) and pass asset_overrides={{'{key}': dst}} "
                f"(or set the {UPLOADED_ASSETS[key][1]} environment variable).{hint}"
            ) from exc
        resolved[key] = asset_id
    return resolved
