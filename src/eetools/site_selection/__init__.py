"""TGBS reference / degraded site selection (``eetools.site_selection``).

Entry points: :func:`identify_reference_sites` and :func:`identify_degraded_sites`
(also re-exported from :mod:`eetools.workflows`). Building blocks live in
``domain`` (grid, tiers, exclusions, search ladder), ``predictors`` (Earth Engine
stacks and memoised reductions), ``similarity`` (MESS, envelope gate, embeddings),
``scoring`` (percentiles, scores, thinning, spanning, SOP matrix, layer register)
and ``types`` (configuration and result containers).
"""

from eetools.sensors.assets import SiteSelectionAssetError, check_required_assets
from eetools.site_selection.domain import (
    assign_tiers,
    build_candidate_grid,
    derive_cell_size,
    search_rungs,
    utm_projection_for,
)
from eetools.site_selection.scoring import (
    anchor_check,
    anchor_constraint,
    baseline_window,
    build_layer_register,
    build_selection_matrix,
    degradation_score,
    degraded_exclusions,
    dw_built_filter,
    focal_baseline_pct,
    forest_stratum,
    intactness_score,
    pathway_gate,
    percentile_normalise,
    rank_candidates,
    select_spanning_set,
    thin_spatially,
)
from eetools.site_selection.similarity import (
    compute_mess,
    drop_collinear,
    embedding_query_similarity,
    envelope_gate,
    mean_embedding_vector,
    resolve_seed_source,
)
from eetools.site_selection.types import (
    DegradationType,
    DegradedWeights,
    EnvelopeTolerances,
    ReferenceWeights,
    ScreeningTables,
    ScreeningTablesMismatch,
    SiteSelectionResult,
)
from eetools.site_selection.workflows import (
    export_site_selection,
    identify_degraded_sites,
    identify_reference_sites,
    resolve_signature_mode,
    result_to_feature_collection,
)

__all__ = [
    "DegradationType",
    "DegradedWeights",
    "EnvelopeTolerances",
    "ReferenceWeights",
    "ScreeningTables",
    "ScreeningTablesMismatch",
    "SiteSelectionAssetError",
    "SiteSelectionResult",
    "anchor_check",
    "anchor_constraint",
    "assign_tiers",
    "baseline_window",
    "build_candidate_grid",
    "build_layer_register",
    "build_selection_matrix",
    "check_required_assets",
    "compute_mess",
    "degradation_score",
    "degraded_exclusions",
    "dw_built_filter",
    "forest_stratum",
    "derive_cell_size",
    "drop_collinear",
    "embedding_query_similarity",
    "envelope_gate",
    "export_site_selection",
    "focal_baseline_pct",
    "identify_degraded_sites",
    "identify_reference_sites",
    "intactness_score",
    "mean_embedding_vector",
    "pathway_gate",
    "percentile_normalise",
    "rank_candidates",
    "resolve_seed_source",
    "resolve_signature_mode",
    "result_to_feature_collection",
    "search_rungs",
    "select_spanning_set",
    "thin_spatially",
    "utm_projection_for",
]
