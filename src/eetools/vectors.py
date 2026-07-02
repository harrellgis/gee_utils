from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ee
import geopandas as gpd
import pandas as pd
from shapely.geometry import mapping

# --------------------------------------------------------------------------- #
# Private helpers                                                              #
# --------------------------------------------------------------------------- #


def _read_vector_gdf(path: str | Path, layer: str | None = None) -> gpd.GeoDataFrame:
    """Read a local vector file into a WGS84 GeoDataFrame, validating it is non-empty
    and has a CRS.

    Shared read/validate/reproject backend for the vector readers below.

    Args:
        path: File path (str or Path) to the vector file.
        layer: Optional layer name to read; reads the default layer if None.

    Returns:
        geopandas.GeoDataFrame reprojected to EPSG:4326. Raises ValueError if the
        file is empty or has no CRS defined.
    """
    path = Path(path)
    gdf = gpd.read_file(path, layer=layer)

    if gdf.empty:
        raise ValueError(f"Vector file '{path.name}' contains no features.")

    if gdf.crs is None:
        raise ValueError(f"Vector file '{path.name}' has no CRS defined.")

    return gdf.to_crs("EPSG:4326")


def _to_ee_property(value: Any) -> str | int | float | bool | None:
    """Coerce a GeoDataFrame cell to a value Earth Engine can store as a property.

    Nulls/NaN/NaT become None; numpy scalars are unwrapped to native Python; strings,
    numbers, and bools pass through; anything else (e.g. timestamps) is stringified.
    """
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        # pd.isna over an array-like cell is ambiguous — treat as a non-null object.
        pass
    if hasattr(value, "item"):
        try:
            value = value.item()  # numpy scalar -> Python scalar
        except (TypeError, ValueError):
            pass
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


# --------------------------------------------------------------------------- #
# Vector → EE conversion                                                       #
# --------------------------------------------------------------------------- #


def vector_file_to_ee_geometry(
    path: str | Path, layer: str | None = None
) -> ee.Geometry:
    """Read a local vector file and dissolve its features into a single ee.Geometry
    in WGS84.

    Accepts any format geopandas can read (GeoPackage, GeoJSON, Shapefile, ...). All
    valid features are unioned into one geometry. Use :func:`vector_file_to_features`
    instead when you need each record (and its attributes) preserved individually.

    Args:
        path: File path (str or Path) to the vector file.
        layer: Optional layer name to read; reads the default layer if None.

    Returns:
        ee.Geometry representing the union of all valid features. Raises ValueError
        if the file is empty, has no CRS, or contains no valid geometries.
    """
    gdf = _read_vector_gdf(path, layer=layer)

    geometries = [
        ee.Geometry(mapping(geom))
        for geom in gdf.geometry
        if geom is not None and not geom.is_empty
    ]

    if not geometries:
        raise ValueError(f"No valid geometries found in '{Path(path).name}'.")

    if len(geometries) == 1:
        return geometries[0]

    return ee.FeatureCollection([ee.Feature(g) for g in geometries]).geometry()


def vector_file_to_features(
    path: str | Path, layer: str | None = None, keep_properties: bool = True
) -> ee.FeatureCollection:
    """Read a local vector file into an ee.FeatureCollection, preserving one feature
    per record.

    Keeps each feature individually and, by default, carries its non-geometry columns
    through as feature properties.

    Args:
        path: File path (str or Path) to the vector file.
        layer: Optional layer name to read; reads the default layer if None.
        keep_properties: If True (default), each feature carries its non-geometry columns as properties (nulls dropped to None, numpy/timestamps coerced to native types); if False, features carry geometry only.

    Returns:
        ee.FeatureCollection with one feature per valid record, in file order. Raises ValueError if the file is empty, has no CRS, or contains no valid geometries.
    """
    gdf = _read_vector_gdf(path, layer=layer)
    geometry_col = gdf.geometry.name

    features = []
    for _, row in gdf.iterrows():
        geom = row[geometry_col]
        if geom is None or geom.is_empty:
            continue
        if keep_properties:
            properties = {
                str(key): _to_ee_property(value)
                for key, value in row.items()
                if key != geometry_col
            }
        else:
            properties = None
        features.append(ee.Feature(ee.Geometry(mapping(geom)), properties))

    if not features:
        raise ValueError(f"No valid geometries found in '{Path(path).name}'.")

    return ee.FeatureCollection(features)


def vector_files_to_feature_collection(
    sites: Sequence[tuple[str | Path, str, str]],
    layer: str | None = None,
) -> ee.FeatureCollection:
    """Combine multiple vector files into one ee.FeatureCollection, one feature per
    file, with caller-supplied site identity.

    Each entry pairs a file with the identity to attach to it. The file's features
    are read, reprojected to WGS84, and dissolved into a single geometry that becomes
    one feature in the output collection; the supplied ``site_id`` / ``site_name`` and
    the resolved filename (``source_file``) are stored as properties, matching the
    features built by :func:`build_site_feature` / :func:`load_site_feature`.

    Identity is taken entirely from ``sites`` — nothing is inferred from filenames or
    file attributes, so it does not depend on per-file conventions. ``source_file`` is
    set automatically as provenance only.

    Any format geopandas can read (.geojson, .gpkg, .shp, ...) is accepted.

    Args:
        sites: Sequence of ``(path, site_id, site_name)`` tuples, one per output feature. ``path`` is a str or Path to the vector file; ``site_id`` is a short identifier; ``site_name`` is the human-readable name. Order is preserved in the output collection.
        layer: Optional layer name passed to the reader for each file; reads the default layer if None.

    Returns:
        ee.FeatureCollection with one feature per input file, each carrying site_id, site_name, and source_file properties. Raises ValueError if sites is empty or any file yields no valid geometry.
    """
    if not sites:
        raise ValueError("No files provided to build a FeatureCollection.")

    features = [
        load_site_feature(path, site_id=site_id, site_name=site_name, layer=layer)
        for path, site_id, site_name in sites
    ]
    return ee.FeatureCollection(features)


def merge_vector_files(
    paths: Sequence[str | Path],
    layer: str | None = None,
    keep_properties: bool = True,
) -> ee.FeatureCollection:
    """Read multiple vector files and merge all individual features into one flat
    ee.FeatureCollection.

    Unlike :func:`vector_files_to_feature_collection`, this function preserves every
    record from every file as its own feature — it does not dissolve per file and does
    not impose a schema. All attribute columns are carried through when
    ``keep_properties=True``.

    Use :func:`fc_select_properties` afterwards to rename or subset columns to match
    the schema expected by downstream functions such as
    ``image_collection_to_sample_fc``.

    Args:
        paths: Sequence of file paths (str or Path) in any format geopandas can read
            (GeoPackage, GeoJSON, Shapefile, …). Must be non-empty.
        layer: Optional layer name passed to each reader; reads the default layer if None.
        keep_properties: If True (default), non-geometry columns are carried through as
            feature properties; if False, features carry geometry only.

    Returns:
        ee.FeatureCollection containing one feature per valid record across all input
        files, in file order. Raises ValueError if paths is empty or any file yields no
        valid geometries.
    """
    if not paths:
        raise ValueError("No files provided to merge.")

    fcs = [
        vector_file_to_features(p, layer=layer, keep_properties=keep_properties)
        for p in paths
    ]
    result = fcs[0]
    for fc in fcs[1:]:
        result = result.merge(fc)
    return result


# --------------------------------------------------------------------------- #
# Site builders                                                                #
# --------------------------------------------------------------------------- #


def build_site_feature(
    geometry: ee.Geometry,
    site_id: str,
    site_name: str,
    source_file: str | None = None,
) -> ee.Feature:
    """Build an ee.Feature from an ee.Geometry with standard site metadata properties.

    Args:
        geometry: Site boundary as ee.Geometry.
        site_id: Short identifier string for the site.
        site_name: Human-readable site name.
        source_file: Optional filename of the source boundary file, stored as a feature property.

    Returns:
        ee.Feature with properties site_id, site_name, and optionally source_file.
    """
    properties = {"site_id": site_id, "site_name": site_name}
    if source_file is not None:
        properties["source_file"] = source_file
    return ee.Feature(geometry, properties)


def load_site_feature(
    path: str | Path,
    site_id: str,
    site_name: str,
    layer: str | None = None,
) -> ee.Feature:
    """Load a local boundary file and convert it to a metadata-rich ee.Feature.

    Accepts any format geopandas can read (.gpkg, .geojson, .shp, ...); all of the
    file's features are dissolved into a single boundary geometry.

    Args:
        path: File path (str or Path) to the boundary file.
        site_id: Short identifier string for the site.
        site_name: Human-readable site name.
        layer: Optional layer name to read; reads the default layer if None.

    Returns:
        ee.Feature with the file geometry and properties site_id, site_name, and source_file.
    """
    path = Path(path).resolve()
    geometry = vector_file_to_ee_geometry(path, layer=layer)
    return build_site_feature(
        geometry=geometry,
        site_id=site_id,
        site_name=site_name,
        source_file=path.name,
    )


def get_sites_geometry(sites_fc: ee.FeatureCollection) -> ee.Geometry:
    """Return the merged geometry of all features in a site FeatureCollection.

    Args:
        sites_fc: ee.FeatureCollection of site features.

    Returns:
        ee.Geometry representing the union of all site geometries.
    """
    return ee.FeatureCollection(sites_fc).geometry()


# --------------------------------------------------------------------------- #
# FeatureCollection utilities                                                  #
# --------------------------------------------------------------------------- #


def fc_select_properties(
    fc: ee.FeatureCollection,
    properties: list[str],
    new_names: list[str] | None = None,
    retain_geometry: bool = True,
) -> ee.FeatureCollection:
    """Select and optionally rename properties on every feature in a collection.

    Thin typed wrapper around ``ee.FeatureCollection.select``. The primary use case
    is normalising a source column to ``site_name`` so the result is compatible with
    :func:`~eetools.utils.clip_image_to_fc`, ``image_collection_to_sample_fc``, and
    ``image_collection_to_region_stats_fc``::

        fc = vector_file_to_features("regions.gpkg")
        # file has "region_label"; downstream expects "site_name"
        fc = fc_select_properties(fc, ["region_label"], new_names=["site_name"])

    Args:
        fc: ee.FeatureCollection to transform.
        properties: Property names to keep (in desired output order).
        new_names: Optional list of output names for the selected properties; must have
            the same length as ``properties`` if provided. When None the original names
            are preserved.
        retain_geometry: If True (default) geometry is preserved on each feature; if
            False only the selected properties are kept.

    Returns:
        ee.FeatureCollection with only the selected (and optionally renamed) properties
        on each feature.

    Raises:
        ValueError: If ``new_names`` is provided but its length differs from
            ``properties``.
    """
    if new_names is not None and len(new_names) != len(properties):
        raise ValueError(
            f"new_names length ({len(new_names)}) must match properties length "
            f"({len(properties)})."
        )
    if new_names is not None:
        return ee.FeatureCollection(fc).select(
            propertySelectors=properties,
            newProperties=new_names,
            retainGeometry=retain_geometry,
        )
    return ee.FeatureCollection(fc).select(
        propertySelectors=properties,
        retainGeometry=retain_geometry,
    )


def buffer_feature_collection(
    fc: ee.FeatureCollection,
    distance_m: float,
    max_error: float = 1.0,
) -> ee.FeatureCollection:
    """Buffer every feature's geometry by a fixed distance, preserving all properties.

    Runs server-side via ``ee.Feature.buffer``. Useful for expanding point or line site
    geometries to polygon AOIs before passing to ``sampleRegions`` or
    ``reduceRegions``.

    Args:
        fc: ee.FeatureCollection whose feature geometries will be buffered.
        distance_m: Buffer radius in metres (positive expands, negative shrinks).
        max_error: Maximum permitted error in metres for the buffer approximation
            (default 1.0); maps to EE's ``maxError`` parameter.

    Returns:
        ee.FeatureCollection with the same features and properties but with each
        feature's geometry replaced by its buffered equivalent.
    """

    def _buffer(feature: ee.Feature) -> ee.Feature:
        return ee.Feature(feature).buffer(distance_m, maxError=max_error)

    return ee.FeatureCollection(fc).map(_buffer)
