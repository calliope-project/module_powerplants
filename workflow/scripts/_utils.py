"""General utilities shared across rules."""

from typing import Literal

import geopandas as gpd
import pandas as pd
from pandas.api.types import is_list_like
from pyproj import CRS

# Average year where powerplant datasets were last updated.
# MUST BE ADJUSTED WHENEVER DATASOURCES ARE UPDATED!
DATASET_YEAR = 2024


def check_crs(
    crs: int | str, how: Literal["projected", "geographic", "geocentric"]
) -> CRS:
    """Helper to verify user-provided CRS codes."""
    parsed = CRS.from_user_input(crs)
    correct = False
    match how:
        case "projected":
            if parsed.is_projected:
                correct = True
        case "geographic":
            if parsed.is_geographic:
                correct = True
        case "geocentric":
            if parsed.is_geocentric:
                correct = True
    if not correct:
        raise ValueError(f"{crs!r} is not {how!r}.")
    return parsed


def listify(item) -> list:
    """Avoids ambiguity in YAML list parameters."""
    return item if is_list_like(item) else [item]


EIA_CAT_MAPPING = {
    "bioenergy": "biomass and waste",
    "fossil": "fossil fuels",
    "geothermal": "geothermal",
    "hydropower": ["hydropower", "pumped storage"],
    "nuclear": "nuclear",
    "solar": "solar",
    "wind": "wind",
}
EIA_CAT_MAPPING = {k: listify(v) for k, v in EIA_CAT_MAPPING.items()}

DATE_SOURCE_METADATA = {
    "observed": {
        "label": "Observed date (from powerplant data)",
        "applies_to": {"start_year", "end_year"},
    },
    "derived_from_end_year": {
        "label": "Start date derived from observed end date",
        "applies_to": {"start_year"},
    },
    "imputed_capacity_profile": {
        "label": "Start date imputed from historical commissioning-profile",
        "applies_to": {"start_year"},
    },
    "imputed_construction_window": {
        "label": "Start date imputed within construction window",
        "applies_to": {"start_year"},
    },
    "imputed_pre_construction_window": {
        "label": "Start date imputed within pre-construction window",
        "applies_to": {"start_year"},
    },
    "imputed_announced_window": {
        "label": "Start date imputed within announced window",
        "applies_to": {"start_year"},
    },
    "derived_from_imputed_retirement_end_year": {
        "label": "Start date derived from retirement-profile end date",
        "applies_to": {"start_year"},
    },
    "derived_from_start_year_lifetime": {
        "label": "End date derived from start date and lifetime",
        "applies_to": {"end_year"},
    },
    "imputed_retirement_capacity_profile": {
        "label": "End date imputed from retirement-profile",
        "applies_to": {"end_year"},
    },
    "derived_from_start_year_lifetime_capped_to_retired_status": {
        "label": "End date derived from start date but capped to retired status",
        "applies_to": {"end_year"},
    },
    "derived_from_start_year_lifetime_with_retirement_delay": {
        "label": "End date derived from start date, lifetime, and retirement delay",
        "applies_to": {"end_year"},
    },
    "observed_adjusted_with_retirement_delay": {
        "label": "Observed end date adjusted with retirement delay",
        "applies_to": {"end_year"},
    },
}


def date_source_types_for(year_column: str) -> set[str]:
    """Return source types valid for a date column."""
    return {
        source_type
        for source_type, metadata in DATE_SOURCE_METADATA.items()
        if year_column in metadata["applies_to"]
    }


def date_source_labels() -> dict[str, str]:
    """Return date-source display labels."""
    return {
        source_type: metadata["label"]
        for source_type, metadata in DATE_SOURCE_METADATA.items()
    }


def get_eia_stats_in_cat_yr(
    stats: pd.DataFrame, year: int, category: str
) -> pd.DataFrame:
    """Get EIA statistics for a given year and category."""
    stats = stats[stats["year"] == year]
    stats = stats[stats["category"].isin(EIA_CAT_MAPPING[category])]
    return stats


def get_point_col(
    raw: pd.DataFrame, lon_col: str, lat_col: str, crs: str = "EPSG:4326"
) -> gpd.GeoSeries:
    """Converts latitude / longitude columns to a point geometry."""
    return gpd.points_from_xy(
        raw[lon_col], raw[lat_col], crs=check_crs(crs, "geographic")
    )


def get_combined_text_col(
    raw: pd.DataFrame,
    cols: list[str],
    sep: str = "-",
    prefix: str = "",
    suffix: str = "",
):
    """Vectorised combination of string columns with prefix, suffix and separators.

    Form: {prefix}col1{sep}col2{sep}...coln{suffix}.
    """
    return prefix + raw[cols].astype(str).agg(sep.join, axis="columns") + suffix


def check_single_category(df: pd.DataFrame) -> str:
    """Quick validation for single-category datasets."""
    categories = df["category"].unique()
    if len(categories) != 1:
        raise ValueError(
            f"Cannot compute dataset with ambiguous category. Found '{categories}'"
        )
    return categories[0]


def filter_years(
    powerplants_df: pd.DataFrame, year: int, how: Literal["operating", "future", "past"]
) -> pd.DataFrame:
    """Filter powerplants based on start/end year.

    Assumptions:
    - A powerplant comes online on January 1 of `start_year`.
    - A powerplant goes offline on January 1 of `end_year`.
    - `end_year` > `start_year`.

    Args:
        powerplants_df: Powerplant dataset to filter.
        year: Reference year.
        how:
            - "operating": plants active during `year`
            - "future": plants not yet online in `year`
            - "past": plants already offline by `year`

    Returns:
        A copy of the filtered dataframe.
    """
    match how:
        case "operating":
            mask = (powerplants_df["start_year"] <= year) & (
                year < powerplants_df["end_year"]
            )
        case "future":
            mask = powerplants_df["start_year"] > year
        case "past":
            mask = year >= powerplants_df["end_year"]
        case _:
            raise ValueError(f"Invalid request {how!r}.")

    return powerplants_df.loc[mask].copy()


def ensure_positive_capacity(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with non-positive capacity."""
    return df[df["output_capacity_mw"] > 0].copy()


def get_adjusted_capacity(
    operating_plants: pd.DataFrame, expected_capacity: pd.Series
) -> pd.Series:
    """Adjust powerplant capacity to the total expected capacity per country.

    Args:
        operating_plants (pd.DataFrame): dataframe with all operating plants to adjust.
        expected_capacity (pd.Series): expected category capacity per country.

    Returns:
        pd.Series: adjusted powerplant capacity.
    """
    adjusted_cap_mw = (
        operating_plants["output_capacity_mw"]
        / operating_plants.groupby("country_id")["output_capacity_mw"].transform("sum")
    ) * operating_plants["country_id"].map(expected_capacity)
    return adjusted_cap_mw


def adjust_aggregated_capacity(plants, stats, year):
    """Adjust capacity to national statistics in the given year."""
    category = check_single_category(plants)
    stats = get_eia_stats_in_cat_yr(stats, year, category)
    expected_capacity = stats.groupby(["country_id"])["capacity_mw"].sum()

    adjusted = ensure_positive_capacity(plants)
    adjusted = adjusted[adjusted["country_id"].isin(expected_capacity.index)]

    if adjusted.empty:
        return adjusted.reset_index(drop=True)

    adjusted["output_capacity_mw"] = get_adjusted_capacity(adjusted, expected_capacity)
    return adjusted.reset_index(drop=True)
