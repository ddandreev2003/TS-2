"""Feature engineering package."""

from src.features.assembly import assemble_feature_matrix, get_feature_groups, prepare_supervised
from src.features.autoregressive import build_autoregressive_features
from src.features.calendar import build_calendar_features, build_tax_features
from src.features.macro import build_macro_features

__all__ = [
    "build_autoregressive_features",
    "build_calendar_features",
    "build_tax_features",
    "build_macro_features",
    "assemble_feature_matrix",
    "prepare_supervised",
    "get_feature_groups",
]
