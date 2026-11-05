"""Stage 0: focal geometry, candidate grids, tiers, exclusions and the search ladder.

The candidate grid is built client-side in the focal UTM zone (exact, deterministic
and testable without Earth Engine): a cell is identified by its size and integer
column/row ``(ix, iy)``, so its corners are ``(ix * size, iy * size)`` in UTM metres.
Every grid shares the projection origin, so fine cells nest exactly inside coarse
cells and the 250 m fraction-image pixels nest inside both.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shapely
from pyproj import Geod, Transformer
from shapely.geometry import mapping, shape
from shapely.ops import transform as shp_transform

from eetools.constants import (
    ALLOWED_K,
    FRACTION_SCALE_M,
    SCREEN_CELL_STEP_M,
    SEARCH_LADDERS,
)
from eetools.sensors.assets import check_required_assets  # noqa: F401 (re-export)

_GEOD = Geod(ellps="WGS84")


# --------------------------------------------------------------------------- #
# Focal geometry                                                               #
# --------------------------------------------------------------------------- #
def load_focal_geometry(focal_site: Any) -> shapely.Geometry:
    """Return the focal site as a single shapely geometry in EPSG:4326.

    Args:
        focal_site: ee.Geometry, ee.Feature, ee.FeatureCollection, shapely geometry,
            or a path (str/Path) to a vector file (all features are dissolved).

    Returns:
        shapely (Multi)Polygon in lon/lat.

    Raises:
        ValueError: If the geometry is empty or not polygonal.
    """
    if isinstance(focal_site, shapely.Geometry):
        geom = focal_site
    elif isinstance(focal_site, (str, Path)):
        import geopandas as gpd

        gdf = gpd.read_file(focal_site)
        if gdf.crs is not None:
            gdf = gdf.to_crs(4326)
        geom = shapely.union_all(gdf.geometry.values)
    else:
        import ee

        if isinstance(focal_site, ee.FeatureCollection):
            obj = focal_site.geometry()
        elif isinstance(focal_site, ee.Feature):
            obj = focal_site.geometry()
        else:
            obj = ee.Geometry(focal_site)
        geom = shape(obj.transform("EPSG:4326", 0.1).getInfo())
    geom = shapely.make_valid(geom)
    if geom.is_empty or geom.area == 0:
        raise ValueError("The focal site geometry is empty or has no area.")
    return geom


def geometry_hash(geom: shapely.Geometry | None) -> str | None:
    """Stable short hash of a geometry (coordinates rounded to ~1 cm).

    Args:
        geom: shapely geometry or None.

    Returns:
        Hex digest string, or None for None.
    """
    import hashlib

    if geom is None:
        return None
    wkb = shapely.to_wkb(shapely.set_precision(geom, 1e-7), hex=True)
    return hashlib.sha256(wkb.encode()).hexdigest()[:16]


def utm_projection_for(geom_ll: shapely.Geometry) -> str:
    """Return the EPSG code of the UTM zone holding the geometry's centroid.

    Args:
        geom_ll: shapely geometry in lon/lat.

    Returns:
        'EPSG:326zz' (north) or 'EPSG:327zz' (south).
    """
    c = geom_ll.centroid
    zone = int(math.floor((c.x + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return f"EPSG:{(32700 if c.y < 0 else 32600) + zone}"


def to_utm(geom_ll: shapely.Geometry, crs: str) -> shapely.Geometry:
    """Project a lon/lat geometry to a metric CRS.

    Args:
        geom_ll: shapely geometry in EPSG:4326.
        crs: Target CRS string.

    Returns:
        Projected shapely geometry.
    """
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return shp_transform(t.transform, geom_ll)


def to_lonlat(geoms: Any, crs: str) -> Any:
    """Project geometries (scalar or array) from a metric CRS to EPSG:4326.

    Args:
        geoms: shapely geometry or array of geometries.
        crs: Source CRS string.

    Returns:
        Geometry or array in lon/lat.
    """
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def _fn(coords: np.ndarray) -> np.ndarray:
        x, y = t.transform(coords[:, 0], coords[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geoms, _fn)


def geodesic_area_m2(geom_ll: shapely.Geometry) -> float:
    """Geodesic (WGS84) area of a lon/lat geometry in square metres.

    Args:
        geom_ll: shapely geometry in EPSG:4326.

    Returns:
        Area in m^2 (0 for empty geometries).
    """
    if geom_ll is None or geom_ll.is_empty:
        return 0.0
    return abs(_GEOD.geometry_area_perimeter(geom_ll)[0])


# --------------------------------------------------------------------------- #
# Cell size                                                                    #
# --------------------------------------------------------------------------- #
def derive_cell_size(
    focal_area_m2: float,
    screen_cell_m: float = 2000.0,
    min_m: float = FRACTION_SCALE_M,
) -> tuple[float, float]:
    """Derive the (screening, fine) cell sizes for a focal site (spec §5.0).

    ``side = sqrt(focal area)``. If ``side > screen_cell_m`` the screening cell grows
    to ``ceil(side / 500) * 500`` and fine = screen (k = 1, no refinement). Otherwise
    screen = screen_cell_m and k is the allowed value (ALLOWED_K = 1, 2, 4, 8) nearest
    to ``screen / side`` whose fine cell is a whole multiple of 250 m and >= min_m
    (ties go to the smaller k, i.e. larger cells).

    Args:
        focal_area_m2: Geodesic focal-site area in m^2.
        screen_cell_m: Minimum screening cell size in metres (multiple of 250).
        min_m: Smallest permitted fine cell (default 250 m).

    Returns:
        Tuple (screen_m, fine_m); screen_m % fine_m == 0.

    Raises:
        ValueError: If the area is not positive or screen_cell_m is not a positive
            multiple of 250 m.
    """
    if focal_area_m2 <= 0:
        raise ValueError("focal_area_m2 must be positive.")
    if screen_cell_m <= 0 or screen_cell_m % FRACTION_SCALE_M:
        raise ValueError(
            f"screen_cell_m must be a positive multiple of {FRACTION_SCALE_M} m."
        )
    side = math.sqrt(focal_area_m2)
    if side > screen_cell_m:
        grown = float(math.ceil(side / SCREEN_CELL_STEP_M) * SCREEN_CELL_STEP_M)
        return grown, grown
    screen = float(screen_cell_m)
    ratio = screen / side
    options = [
        k
        for k in ALLOWED_K
        if (screen / k) % FRACTION_SCALE_M == 0 and screen / k >= min_m
    ]
    k = min(options, key=lambda kk: (abs(kk - ratio), kk))
    return screen, screen / k


# --------------------------------------------------------------------------- #
# Grids                                                                        #
# --------------------------------------------------------------------------- #
def make_cell_id(size: float, ix: int, iy: int) -> str:
    """Return the canonical cell identifier ``'<size>_<ix>_<iy>'``.

    Args:
        size: Cell size in metres.
        ix: Column index (x0 = ix * size).
        iy: Row index (y0 = iy * size).

    Returns:
        Cell id string.
    """
    return f"{int(round(size))}_{int(ix)}_{int(iy)}"


def cell_polygons(df: pd.DataFrame) -> np.ndarray:
    """UTM polygons for rows with columns ix, iy, size.

    Args:
        df: DataFrame with integer ix, iy and float size columns.

    Returns:
        Array of shapely Polygons in the grid CRS.
    """
    size = df["size"].to_numpy(float)
    x0 = df["ix"].to_numpy(float) * size
    y0 = df["iy"].to_numpy(float) * size
    return shapely.box(x0, y0, x0 + size, y0 + size)


def assign_tiers(
    dist_m: Sequence[float] | np.ndarray,
    tier_distances_km: Sequence[float] = (25.0, 50.0, 150.0),
) -> np.ndarray:
    """Assign search tiers from distance to the focal site.

    Args:
        dist_m: Distances (m) from cell centroid to the focal boundary.
        tier_distances_km: Increasing outer limits of tiers 1, 2, 3 (km).

    Returns:
        Integer array: 1, 2 or 3, or 0 beyond the last tier.

    Raises:
        ValueError: If tier distances are not strictly increasing and positive.
    """
    limits = np.asarray(tier_distances_km, float) * 1000.0
    if len(limits) == 0 or np.any(limits <= 0) or np.any(np.diff(limits) <= 0):
        raise ValueError("tier_distances_km must be positive and strictly increasing.")
    d = np.asarray(dist_m, float)
    tier = np.searchsorted(limits, d, side="left") + 1
    tier[d > limits[-1]] = 0
    return tier.astype(int)


def build_candidate_grid(
    focal_utm: shapely.Geometry,
    max_distance_m: float,
    cell_size_m: float,
    tier_distances_km: Sequence[float] = (25.0, 50.0, 150.0),
) -> pd.DataFrame:
    """Build the candidate grid around a focal site in its UTM projection.

    Cells (aligned to the projection origin) whose centroid lies within
    ``max_distance_m`` of the focal boundary are kept.

    Args:
        focal_utm: Focal geometry in the grid CRS (metres).
        max_distance_m: Search radius from the focal boundary (m).
        cell_size_m: Cell size (m).
        tier_distances_km: Tier limits (km) used to label each cell.

    Returns:
        DataFrame indexed by cell_id with columns ix, iy, size, cx, cy (UTM centroid),
        dist_to_focal_m and tier.
    """
    s = float(cell_size_m)
    minx, miny, maxx, maxy = focal_utm.bounds
    ix = np.arange(
        math.floor((minx - max_distance_m) / s), math.ceil((maxx + max_distance_m) / s)
    )
    iy = np.arange(
        math.floor((miny - max_distance_m) / s), math.ceil((maxy + max_distance_m) / s)
    )
    mx, my = np.meshgrid(ix, iy)
    gx, gy = mx.ravel(), my.ravel()
    cx = (gx + 0.5) * s
    cy = (gy + 0.5) * s
    shapely.prepare(focal_utm)
    dist = shapely.distance(shapely.points(cx, cy), focal_utm)
    keep = dist <= max_distance_m
    df = pd.DataFrame(
        {
            "ix": gx[keep].astype(int),
            "iy": gy[keep].astype(int),
            "size": s,
            "cx": cx[keep],
            "cy": cy[keep],
            "dist_to_focal_m": dist[keep],
        }
    )
    df["tier"] = assign_tiers(df["dist_to_focal_m"], tier_distances_km)
    df.index = pd.Index(
        [make_cell_id(s, a, b) for a, b in zip(df["ix"], df["iy"], strict=True)],
        name="cell_id",
    )
    return df


def apply_exclusions(
    cells: pd.DataFrame,
    exclusion_utm: shapely.Geometry,
    crs: str,
    exclude_max_share: float = 0.5,
) -> pd.DataFrame:
    """Drop cells mostly inside the exclusion zone and compute effective areas.

    A cell is removed only if more than ``exclude_max_share`` of its area lies in
    the exclusion zone (focal site + buffer, plus any user exclusions); cells that
    merely touch it are kept, and their excluded pixels are masked out of every
    reduction separately.

    Args:
        cells: Grid DataFrame (ix, iy, size, ...).
        exclusion_utm: Exclusion zone in the grid CRS.
        crs: Grid CRS string (for geodesic areas).
        exclude_max_share: Maximum excluded share for a kept cell.

    Returns:
        Copy of the kept rows with added columns focal_buffer_share, area_ha and
        effective_area_ha (geodesic, area outside the exclusion zone).
    """
    out = cells.copy()
    polys = cell_polygons(out)
    planar = shapely.area(polys)
    shapely.prepare(exclusion_utm)
    hits = shapely.intersects(polys, exclusion_utm)
    inter = np.empty(len(polys), dtype=object)
    inter[:] = None
    if hits.any():
        inter[hits] = shapely.intersection(polys[hits], exclusion_utm)
    share = np.zeros(len(polys))
    if hits.any():
        share[hits] = shapely.area(inter[hits]) / planar[hits]
    out["focal_buffer_share"] = share
    keep = share <= exclude_max_share
    out = out.loc[keep].copy()
    polys, inter = polys[keep], inter[keep]
    polys_ll = to_lonlat(polys, crs)
    area = np.array([geodesic_area_m2(p) for p in polys_ll])
    eff = area.copy()
    idx = np.flatnonzero([g is not None and not g.is_empty for g in inter])
    if len(idx):
        inter_ll = to_lonlat(np.array(list(inter[idx]), dtype=object), crs)
        eff[idx] = area[idx] - np.array([geodesic_area_m2(g) for g in inter_ll])
    out["area_ha"] = area / 1e4
    out["effective_area_ha"] = np.clip(eff, 0, None) / 1e4
    return out


def refine_cells(coarse: pd.DataFrame, k: int) -> pd.DataFrame:
    """Subdivide coarse cells into k x k nested fine cells (no clipping, no slivers).

    Args:
        coarse: Coarse cells (index cell_id; columns ix, iy, size, tier, ...).
        k: Subdivision factor (from ALLOWED_K).

    Returns:
        DataFrame of k^2 fine cells per coarse cell, indexed by fine cell_id, with
        ix, iy, size, cx, cy, parent_id and the parent's tier.

    Raises:
        ValueError: If k is not in ALLOWED_K.
    """
    if k not in ALLOWED_K:
        raise ValueError(f"k must be one of {ALLOWED_K}.")
    rows = []
    for cid, row in coarse.iterrows():
        fine = float(row["size"]) / k
        for a in range(k):
            for b in range(k):
                fx = int(row["ix"]) * k + a
                fy = int(row["iy"]) * k + b
                rows.append(
                    {
                        "cell_id": make_cell_id(fine, fx, fy),
                        "ix": fx,
                        "iy": fy,
                        "size": fine,
                        "cx": (fx + 0.5) * fine,
                        "cy": (fy + 0.5) * fine,
                        "parent_id": cid,
                        "tier": int(row["tier"]),
                    }
                )
    cols = ["ix", "iy", "size", "cx", "cy", "parent_id", "tier"]
    if not rows:
        return pd.DataFrame(columns=cols, index=pd.Index([], name="cell_id"))
    return pd.DataFrame(rows).set_index("cell_id")[cols]


def add_focal_distance(
    cells: pd.DataFrame, focal_utm: shapely.Geometry
) -> pd.DataFrame:
    """Add dist_to_focal_m (cell centroid to focal boundary) to a cell table.

    Args:
        cells: DataFrame with cx, cy columns (grid CRS).
        focal_utm: Focal geometry in the grid CRS.

    Returns:
        Copy with dist_to_focal_m set.
    """
    out = cells.copy()
    out["dist_to_focal_m"] = shapely.distance(
        shapely.points(out["cx"].to_numpy(float), out["cy"].to_numpy(float)), focal_utm
    )
    return out


# --------------------------------------------------------------------------- #
# Search ladder                                                                #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rung:
    """One rung of the search ladder (spec §5.1).

    Attributes:
        index: Position in the ladder (0-based).
        max_tier: Tiers 1..max_tier form the pool.
        relax_substrate: Substrate gate relaxed to 'related'.
        relax_class: Class gate relaxed to 'group'.
    """

    index: int
    max_tier: int
    relax_substrate: bool
    relax_class: bool

    @property
    def label(self) -> str:
        """Human-readable rung label, e.g. 'T1-2 related + group'."""
        tiers = "T1" if self.max_tier == 1 else f"T1-{self.max_tier}"
        if self.relax_substrate and self.relax_class:
            level = "related + group"
        elif self.relax_substrate:
            level = "substrate related"
        elif self.relax_class:
            level = "class group"
        else:
            level = "strict"
        return f"{tiers} {level}"


def search_rungs(relaxation: str = "before_tier3") -> list[Rung]:
    """Return the ordered search-ladder rungs for a relaxation mode.

    Args:
        relaxation: 'before_tier3' (default), 'after_tier3' or 'off'.

    Returns:
        List of Rung objects in search order.

    Raises:
        ValueError: If relaxation is unknown.
    """
    if relaxation not in SEARCH_LADDERS:
        raise ValueError(
            f"relaxation must be one of {sorted(SEARCH_LADDERS)}, got {relaxation!r}."
        )
    return [
        Rung(i, tier, sub, cls)
        for i, (tier, sub, cls) in enumerate(SEARCH_LADDERS[relaxation])
    ]


def geojson_of(geom: shapely.Geometry) -> dict[str, Any]:
    """Return a GeoJSON mapping of a shapely geometry.

    Args:
        geom: shapely geometry.

    Returns:
        GeoJSON-like dict.
    """
    return dict(mapping(geom))


def cell_geodataframe(cells: pd.DataFrame, crs: str) -> Any:
    """Turn a cell table into an EPSG:4326 GeoDataFrame of cell polygons.

    Args:
        cells: DataFrame with ix, iy, size columns.
        crs: Grid CRS string.

    Returns:
        geopandas.GeoDataFrame in EPSG:4326 with the same index and columns.
    """
    import geopandas as gpd

    if len(cells) == 0:
        return gpd.GeoDataFrame(cells.copy(), geometry=[], crs=4326)
    polys = to_lonlat(cell_polygons(cells), crs)
    return gpd.GeoDataFrame(cells.copy(), geometry=list(polys), crs=4326)
