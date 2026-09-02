"""Data: strain access, the strain -> tile representation, and tile datasets.

The representation module owns the three lossy steps between strain and network
(phase discarded, amplitude discarded, resolution) that Phase 3 exists to ablate.
"""

from .injections import InjectionParameters, ParameterSampler, rescale_to_network_snr
from .representation import (
    TileSpec,
    make_tile,
    min_max_norm,
    noise_referenced_norm,
    notch,
    notch_lines_for,
    reference_asd,
    whiten_spectral,
    qtransform,
    whiten,
)
from .strain import fetch_strain, load_reference_psd, load_segments, total_livetime
from .tiles import CachedTileDataset, GeneratedTileDataset, balanced_sampler_weights

__all__ = [
    "CachedTileDataset",
    "GeneratedTileDataset",
    "InjectionParameters",
    "ParameterSampler",
    "TileSpec",
    "balanced_sampler_weights",
    "fetch_strain",
    "load_reference_psd",
    "load_segments",
    "make_tile",
    "min_max_norm",
    "noise_referenced_norm",
    "notch",
    "notch_lines_for",
    "reference_asd",
    "whiten_spectral",
    "qtransform",
    "rescale_to_network_snr",
    "total_livetime",
    "whiten",
]
