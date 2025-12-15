import re
from typing import Dict, Iterable, Tuple

# Maps from legacy ASM identifiers to descriptive, snake_case names.
COMPONENT_NAME_MAP: Dict[str, str] = {
    'S_I': 'soluble_inert_cod',
    'S_F': 'readily_biodegradable_cod',
    'S_A': 'acetate',
    'X_I': 'particulate_inert_cod',
    'X_S': 'slowly_biodegradable_cod',
    'X_H': 'heterotrophic_biomass',
    'X_AUT': 'nitrifying_biomass',
    'X_PAO': 'phosphate_accumulating_biomass',
    'X_PP': 'stored_polyphosphate',
    'X_PHA': 'stored_polyhydroxyalkanoate',
    'S_O2': 'dissolved_oxygen',
    'S_N2': 'dinitrogen',
    'S_NH4': 'ammonia_nitrogen',
    'S_NO3': 'nitrate_nitrite_nitrogen',
    'S_PO4': 'orthophosphate',
    'S_ALK': 'alkalinity_as_caco3',
    'X_MeOH': 'metal_hydroxide_solids',
    'X_MeP': 'metal_phosphate_solids',
    'H2O': 'water',
}

COMPOSITE_NAME_MAP: Dict[str, str] = {
    'COD': 'chemical_oxygen_demand',
    'BOD': 'biochemical_oxygen_demand',
    'TN': 'total_nitrogen',
    'TKN': 'total_kjeldahl_nitrogen',
    'TP': 'total_phosphorus',
    'TSS': 'total_suspended_solids',
    'VSS': 'volatile_suspended_solids',
    'TOC': 'total_organic_carbon',
    'TC': 'total_carbon',
}

STREAM_PREFIX_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r'^Target_Effluent_', 'effluent'),
    (r'^Target_Wastage_', 'wastage'),
    (r'^inf_', 'influent'),
    (r'^influent_', 'influent'),
    (r'^effluent_', 'effluent'),
    (r'^wastage_', 'wastage'),
)

CONCENTRATION_SUFFIX = '_mg_L'
_REVERSE_COMPONENT_MAP = {v: k for k, v in COMPONENT_NAME_MAP.items()}
_REVERSE_COMPOSITE_MAP = {v: k for k, v in COMPOSITE_NAME_MAP.items()}
_DESCRIPTIVE_BASE_NAMES = set(_REVERSE_COMPONENT_MAP.keys()) | set(_REVERSE_COMPOSITE_MAP.keys())


def _strip_units(name: str) -> str:
    """Remove trailing unit annotations like '(mg/L)' from a column name."""
    base = name.split('(')[0].strip()
    return re.sub(r'\s+', '_', base)


def canonical_base_name(token: str) -> str:
    """
    Convert a component/composite token to the descriptive canonical form.
    Returns None when the token is not a known compound or composite.
    """
    cleaned = re.sub(r'[^A-Za-z0-9_]', '_', token).strip('_')
    cleaned_upper = cleaned.upper()
    cleaned_lower = cleaned.lower()

    if cleaned_upper in COMPONENT_NAME_MAP:
        return COMPONENT_NAME_MAP[cleaned_upper]
    if cleaned_upper in COMPOSITE_NAME_MAP:
        return COMPOSITE_NAME_MAP[cleaned_upper]
    if cleaned_lower in _DESCRIPTIVE_BASE_NAMES:
        return cleaned_lower
    return None


def legacy_identifier(descriptive_name: str) -> str:
    """
    Map a descriptive compound/composite name back to its legacy ASM identifier.
    """
    base = descriptive_name.replace(CONCENTRATION_SUFFIX, '')
    if base.startswith(('influent_', 'effluent_', 'wastage_')):
        base = base.split('_', 1)[1]
    base = base.strip('_')
    if base in _REVERSE_COMPONENT_MAP:
        return _REVERSE_COMPONENT_MAP[base]
    if base in _REVERSE_COMPOSITE_MAP:
        return _REVERSE_COMPOSITE_MAP[base]
    return base


def split_prefix(raw_name: str) -> Tuple[str, str]:
    """
    Identify the stream prefix (influent/effluent/wastage) and return the remaining token.
    """
    name = _strip_units(raw_name)
    for pattern, prefix in STREAM_PREFIX_PATTERNS:
        if re.match(pattern, name):
            remainder = re.sub(pattern, '', name)
            return prefix, remainder
    return '', name


def normalize_stream_column(name: str) -> str:
    """
    Normalize a column name that represents a concentration for a specific stream.
    Examples:
        'Target_Effluent_S_NH4 (mg/L)' -> 'effluent_ammonia_nitrogen_mg_L'
        'inf_COD' -> 'influent_chemical_oxygen_demand_mg_L'
    """
    prefix, remainder = split_prefix(name)
    base_name = canonical_base_name(remainder)
    if not base_name:
        return name.strip()
    prefix_part = f"{prefix}_" if prefix else ''
    return f"{prefix_part}{base_name}{CONCENTRATION_SUFFIX}"


def build_rename_map(columns: Iterable[str]) -> Dict[str, str]:
    """Generate a rename map for all recognizable concentration columns."""
    rename_map = {}
    for col in columns:
        normalized = normalize_stream_column(col)
        if normalized != col:
            rename_map[col] = normalized
    return rename_map


def rename_concentration_columns(df):
    """
    Rename recognizable concentration columns in a DataFrame to descriptive names.
    Returns the same DataFrame instance for convenience.
    """
    rename_map = build_rename_map(df.columns)
    if rename_map:
        df.rename(columns=rename_map, inplace=True)
    return df


def strip_prefix_and_units(name: str) -> str:
    """Return the descriptive base name without stream prefix or units."""
    normalized = normalize_stream_column(name)
    base = normalized.replace(CONCENTRATION_SUFFIX, '')
    for prefix in ('influent_', 'effluent_', 'wastage_'):
        if base.startswith(prefix):
            return base[len(prefix):]
    return base


def stream_prefix(name: str) -> str:
    """Return the canonical stream prefix ('influent', 'effluent', 'wastage', or '')."""
    normalized = normalize_stream_column(name)
    for prefix in ('influent_', 'effluent_', 'wastage_'):
        if normalized.startswith(prefix):
            return prefix[:-1]
    return ''
