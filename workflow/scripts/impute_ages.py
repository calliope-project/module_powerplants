"""Imputation of missing values."""

import math
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import _utils
import geopandas as gpd
import numpy as np
import pandas as pd
from cmap import Colormap
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if TYPE_CHECKING:
    snakemake: Any

HISTORICAL = {"operating", "retired"}
CURRENT = {"operating"}
PLANNED = {"construction", "pre-construction", "announced"}

SCENARIO_MAP = {
    "historical": HISTORICAL,
    "construction": HISTORICAL | {"construction"},
    "pre_construction": HISTORICAL | {"construction", "pre-construction"},
    "announced": HISTORICAL | PLANNED,
}

# Harmonise powerplant categories with the category names used by the
# annual reference-capacity dataset.
REFERENCE_CATEGORY_MAP = _utils.EIA_CAT_MAPPING

# formatting preferences for the visualisations
DATE_SOURCE_LABELS = {
    "observed": "Observed date (from powerplant data)",
    "derived_from_end_year": "Start date derived from observed end date",
    "imputed_capacity_profile": "Start date imputed from historical commissioning-profile",
    "derived_from_imputed_retirement_end_year": "Start date derived from retirement-profile end date",
    "derived_from_start_year_lifetime": "End date derived from start date and lifetime",
    "derived_from_start_year_lifetime_capped_to_retired_status": "End date derived from start date but capped to retired status",
    "derived_from_start_year_lifetime_with_retirement_delay": "End date derived from start date, lifetime, and retirement delay",
    "observed_adjusted_with_retirement_delay": "Observed end date adjusted with retirement delay",
    "imputed_retirement_capacity_profile": "End date imputed from retirement-profile",
    "imputed_construction_window": "Start date imputed within construction window",
    "imputed_pre_construction_window": "Start date imputed within pre-construction window",
    "imputed_announced_window": "Start date imputed within announced window",
}

DATE_SOURCE_COLORS = {
    "observed": "#aaaaaa",
    "derived_from_end_year": "#0c2c84",
    "imputed_capacity_profile": "#225ea8",
    "imputed_construction_window": "#1d91c0",
    "imputed_pre_construction_window": "#41b6c4",
    "imputed_announced_window": "#7fcdbb",
    "derived_from_imputed_retirement_end_year": "#c7e9b4",
    "derived_from_start_year_lifetime": "#7a0177",
    "imputed_retirement_capacity_profile": "#c51b8a",
    "derived_from_start_year_lifetime_capped_to_retired_status": "#f768a1",
    "derived_from_start_year_lifetime_with_retirement_delay": "#fa9fb5",
    "observed_adjusted_with_retirement_delay": "#fcc5c0",
}

FALLBACK_DATE_SOURCE_COLORS = [
    "#1F77B4",
    "#FF7F0E",
    "#2CA02C",
    "#D62728",
    "#9467BD",
    "#8C564B",
    "#E377C2",
    "#7F7F7F",
    "#BCBD22",
    "#17BECF",
]


DATE_SOURCE_ORDER = [
    "observed",
    "derived_from_end_year",
    "imputed_capacity_profile",
    "imputed_construction_window",
    "imputed_pre_construction_window",
    "imputed_announced_window",
    "derived_from_imputed_retirement_end_year",
    "derived_from_start_year_lifetime",
    "imputed_retirement_capacity_profile",
    "derived_from_start_year_lifetime_capped_to_retired_status",
    "derived_from_start_year_lifetime_with_retirement_delay",
    "observed_adjusted_with_retirement_delay",
]


def _initial_year_source_type(year: pd.Series) -> pd.Series:
    """Label whether year values were originally present or missing."""
    source_type = pd.Series("observed", index=year.index, dtype="object")
    source_type.loc[year.isna()] = "missing_unresolved"
    return source_type


def _build_reference_addition_profile(
    reference_capacity_df: pd.DataFrame,
    country_id: str,
    category: str,
    years: pd.Index,
    smoothing_window: int = 1,
) -> pd.DataFrame:
    """Build an annual commissioning profile from capacity stock data."""
    capacity_stock = (
        reference_capacity_df.loc[
            (reference_capacity_df["country_id"] == country_id)
            & (reference_capacity_df["category"] == category)
            & (reference_capacity_df["year"] <= _utils.DATASET_YEAR),
            ["year", "capacity_mw"],
        ]
        .sort_values("year")
        .set_index("year")["capacity_mw"]
    )

    # Positive annual stock changes provide the temporal commissioning
    # profile. Negative changes represent retirements or revisions and do
    # not contribute commissioning weight hence clipped to 0.
    reference_stock = capacity_stock.reindex(years)
    reference_positive_change = (
        capacity_stock.diff().clip(lower=0.0).reindex(years).fillna(0.0)
    )

    profile_basis = reference_positive_change.copy()

    if smoothing_window > 1:
        profile_basis = profile_basis.rolling(
            window=smoothing_window, center=True, min_periods=1
        ).mean()

    profile_fallback_used = profile_basis.sum() <= 0

    if profile_fallback_used:
        profile_basis.loc[:] = 1.0

    profile = pd.DataFrame(
        {
            "reference_stock_mw": reference_stock,
            "reference_positive_change_mw": reference_positive_change,
            "reference_profile_basis_mw": profile_basis,
            "reference_profile_weight": profile_basis / profile_basis.sum(),
        },
        index=years,
    )
    profile["profile_fallback_used"] = profile_fallback_used

    return profile


def _build_reference_retirement_profile(
    reference_capacity_df: pd.DataFrame, country_id: str, category: str, years: pd.Index
) -> pd.DataFrame:
    """Build an annual retirement profile from capacity-stock reductions."""
    capacity_stock = (
        reference_capacity_df.loc[
            (reference_capacity_df["country_id"] == country_id)
            & (reference_capacity_df["category"] == category)
            & (reference_capacity_df["year"] < _utils.DATASET_YEAR),
            ["year", "capacity_mw"],
        ]
        .sort_values("year")
        .set_index("year")["capacity_mw"]
    )

    reference_stock = capacity_stock.reindex(years)

    # Negative annual stock changes provide the retirement profile.
    # Positive changes represent net additions and carry no retirement weight.
    reference_negative_change = (
        (-capacity_stock.diff()).clip(lower=0.0).reindex(years).fillna(0.0)
    )

    profile_basis = reference_negative_change.copy()
    profile_fallback_used = profile_basis.sum() <= 0

    if profile_fallback_used:
        profile_basis.loc[:] = 1.0

    profile = pd.DataFrame(
        {
            "reference_stock_mw": reference_stock,
            "reference_negative_change_mw": reference_negative_change,
            "reference_profile_basis_mw": profile_basis,
            "reference_profile_weight": (profile_basis / profile_basis.sum()),
        },
        index=years,
    )
    profile["profile_fallback_used"] = profile_fallback_used

    return profile


def _build_clipped_residual_profile(
    dated_df: pd.DataFrame,
    profile: pd.DataFrame,
    allocatable_years: pd.Index,
    missing_capacity_mw: float,
    year_col: str,
) -> pd.DataFrame:
    """Add observed capacity and a feasible residual target to a profile."""
    result = profile.copy()

    observed_capacity = (
        dated_df.groupby(year_col)["output_capacity_mw"]
        .sum()
        .reindex(result.index, fill_value=0.0)
    )

    # The reference data determine the temporal shape, whilst the plant
    # dataset determines the total capacity represented by the profile.
    total_capacity_mw = observed_capacity.sum() + missing_capacity_mw

    result["observed_mw"] = observed_capacity
    result["target_final_mw"] = result["reference_profile_weight"] * total_capacity_mw
    result["raw_residual_mw"] = result["target_final_mw"] - result["observed_mw"]

    # Observed dates remain fixed. Years that already exceed the scaled
    # target therefore receive no additional imputed capacity.
    residual_target = (
        result["raw_residual_mw"]
        .clip(lower=0.0)
        .reindex(allocatable_years, fill_value=0.0)
    )

    residual_fallback_used = residual_target.sum() <= 0

    if residual_fallback_used:
        residual_target = result["reference_profile_weight"].reindex(
            allocatable_years, fill_value=0.0
        )

    if residual_target.sum() <= 0:
        residual_target = pd.Series(1.0, index=allocatable_years, dtype=float)

    # Rescale the feasible residual so that all missing plant capacity is
    # allocated while preserving its relative annual shape.
    residual_target = residual_target / residual_target.sum() * missing_capacity_mw

    result["residual_target_mw"] = residual_target.reindex(result.index, fill_value=0.0)
    result["residual_fallback_used"] = residual_fallback_used

    return result


def _allocate_start_years_by_residual_target(
    undated_df: pd.DataFrame, residual_target: pd.Series, lifetimes: dict[str, int]
) -> pd.Series:
    """Assign whole plants to feasible years with the largest deficits.

    Plants with the narrowest feasible commissioning windows are processed
    first. Within equal feasible windows, larger plants are processed first
    because they are harder to fit into the residual target.
    """
    capacities = undated_df["output_capacity_mw"]
    earliest_feasible_year = _utils.DATASET_YEAR - undated_df["technology"].map(
        lifetimes
    )

    # Allocate the least flexible plants first, then the largest capacity
    # blocks, to reduce poor fits caused by indivisible plants.
    order = pd.DataFrame(
        {
            "earliest_feasible_year": earliest_feasible_year,
            "capacity": capacities,
            "powerplant_id": undated_df["powerplant_id"],
            "row_order": np.arange(len(undated_df)),
        },
        index=undated_df.index,
    ).sort_values(
        ["earliest_feasible_year", "capacity", "powerplant_id", "row_order"],
        ascending=[False, False, True, True],
    )

    remaining_target = residual_target.copy()
    assigned_years = pd.Series(np.nan, index=undated_df.index, dtype=float)

    for plant_index in order.index:
        plant = undated_df.loc[plant_index]
        plant_capacity = capacities.loc[plant_index]
        lifetime_years = lifetimes[plant["technology"]]

        earliest_year = _utils.DATASET_YEAR - lifetime_years
        feasible_target = remaining_target.loc[
            (remaining_target.index >= earliest_year)
            & (remaining_target.index <= _utils.DATASET_YEAR)
        ]

        # Assign the plant to the feasible year with the largest remaining
        # deficit. Original target size and year provide deterministic ties.
        target_ranking = pd.DataFrame(
            {
                "year": feasible_target.index,
                "remaining_target": feasible_target.to_numpy(),
                "original_target": residual_target.loc[
                    feasible_target.index
                ].to_numpy(),
            }
        ).sort_values(
            ["remaining_target", "original_target", "year"],
            ascending=[False, False, True],
        )

        assigned_year = target_ranking.iloc[0]["year"]

        assigned_years.loc[plant_index] = assigned_year
        remaining_target.loc[assigned_year] -= plant_capacity

    return assigned_years

def _allocate_years_by_target(
    undated_df: pd.DataFrame,
    target: pd.Series,
) -> pd.Series:
    """Assign whole plants to years with the largest remaining deficits."""
    order = (
        undated_df[["output_capacity_mw", "powerplant_id"]]
        .assign(row_order=np.arange(len(undated_df)))
        .sort_values(
            ["output_capacity_mw", "powerplant_id", "row_order"],
            ascending=[False, True, True],
        )
    )

    remaining_target = target.copy()
    assigned_years = pd.Series(
        np.nan,
        index=undated_df.index,
        dtype=float,
    )

    for plant_index in order.index:
        plant_capacity = undated_df.loc[
            plant_index,
            "output_capacity_mw",
        ]

        target_ranking = pd.DataFrame(
            {
                "year": remaining_target.index,
                "remaining_target": remaining_target.to_numpy(),
                "original_target": target.to_numpy(),
            }
        ).sort_values(
            [
                "remaining_target",
                "original_target",
                "year",
            ],
            ascending=[False, False, True],
        )

        assigned_year = target_ranking.iloc[0]["year"]

        assigned_years.loc[plant_index] = assigned_year
        remaining_target.loc[assigned_year] -= plant_capacity

    return assigned_years

def _complete_capacity_profile(
    profile: pd.DataFrame,
    undated_df: pd.DataFrame,
    assigned_years: pd.Series,
    country_id: str,
    category: str,
    reference_category: str,
) -> pd.DataFrame:
    """Add realised imputed capacity and allocation diagnostics."""
    assignments = pd.DataFrame(
        {
            "start_year": assigned_years,
            "output_capacity_mw": undated_df["output_capacity_mw"],
        },
        index=undated_df.index,
    )

    imputed_capacity = (
        assignments.groupby("start_year")["output_capacity_mw"]
        .sum()
        .reindex(profile.index, fill_value=0.0)
    )

    result = profile.copy()
    result["imputed_mw"] = imputed_capacity
    result["final_mw"] = result["observed_mw"] + result["imputed_mw"]
    result["allocation_error_mw"] = result["imputed_mw"] - result["residual_target_mw"]

    # One minus the normalised absolute difference between the target and
    # realised imputed profiles. A value of one indicates an exact match.
    missing_capacity_mw = undated_df["output_capacity_mw"].sum()
    allocation_match = 1.0 - (
        result["allocation_error_mw"].abs().sum() / (2.0 * missing_capacity_mw)
    )
    final_profile_error_mw = result["final_mw"] - result["target_final_mw"]

    total_capacity_mw = result["target_final_mw"].sum()

    final_profile_match = 1.0 - (
        final_profile_error_mw.abs().sum() / (2.0 * total_capacity_mw)
    )

    result["country_id"] = country_id
    result["category"] = category
    result["reference_category"] = reference_category
    result["missing_capacity_mw"] = missing_capacity_mw
    result["number_imputed"] = len(undated_df)
    result["years_used"] = assigned_years.nunique()
    result["allocation_match"] = allocation_match
    result["final_profile_match"] = final_profile_match

    result.index.name = "year"

    column_order = [
        "country_id",
        "category",
        "reference_category",
        "year",
        "reference_stock_mw",
        "reference_positive_change_mw",
        "reference_profile_basis_mw",
        "reference_profile_weight",
        "target_final_mw",
        "observed_mw",
        "raw_residual_mw",
        "residual_target_mw",
        "imputed_mw",
        "final_mw",
        "allocation_error_mw",
        "missing_capacity_mw",
        "number_imputed",
        "years_used",
        "allocation_match",
        "final_profile_match",
        "profile_fallback_used",
        "residual_fallback_used",
    ]

    return result.reset_index()[column_order]


def _complete_retirement_profile(
    profile: pd.DataFrame,
    undated_df: pd.DataFrame,
    assigned_end_years: pd.Series,
    country_id: str,
    category: str,
    reference_category: str,
) -> pd.DataFrame:
    """Add realised retirement capacity and allocation diagnostics."""
    assignments = pd.DataFrame(
        {
            "end_year": assigned_end_years,
            "output_capacity_mw": undated_df["output_capacity_mw"],
        },
        index=undated_df.index,
    )

    imputed_capacity = (
        assignments.groupby("end_year")["output_capacity_mw"]
        .sum()
        .reindex(profile.index, fill_value=0.0)
    )

    result = profile.copy()
    result["imputed_mw"] = imputed_capacity
    result["final_mw"] = result["observed_mw"] + result["imputed_mw"]
    result["allocation_error_mw"] = result["imputed_mw"] - result["residual_target_mw"]

    missing_capacity_mw = undated_df["output_capacity_mw"].sum()

    allocation_match = 1.0 - (
        result["allocation_error_mw"].abs().sum() / (2.0 * missing_capacity_mw)
    )

    final_profile_error_mw = result["final_mw"] - result["target_final_mw"]
    total_capacity_mw = result["target_final_mw"].sum()

    final_profile_match = 1.0 - (
        final_profile_error_mw.abs().sum() / (2.0 * total_capacity_mw)
    )

    result["country_id"] = country_id
    result["category"] = category
    result["reference_category"] = reference_category
    result["missing_capacity_mw"] = missing_capacity_mw
    result["number_imputed"] = len(undated_df)
    result["years_used"] = assigned_end_years.nunique()
    result["allocation_match"] = allocation_match
    result["final_profile_match"] = final_profile_match

    result.index.name = "year"

    column_order = [
        "country_id",
        "category",
        "reference_category",
        "profile_source",
        "year",
        "reference_stock_mw",
        "reference_negative_change_mw",
        "reference_profile_basis_mw",
        "reference_profile_weight",
        "target_final_mw",
        "observed_mw",
        "raw_residual_mw",
        "residual_target_mw",
        "imputed_mw",
        "final_mw",
        "allocation_error_mw",
        "missing_capacity_mw",
        "number_imputed",
        "years_used",
        "allocation_match",
        "final_profile_match",
        "profile_fallback_used",
        "residual_fallback_used",
    ]

    return result.reset_index()[column_order]

def _complete_planned_commissioning_profile(
    undated_df: pd.DataFrame,
    dated_df: pd.DataFrame,
    assigned_years: pd.Series,
    target: pd.Series,
    country_id: str,
    category: str,
    technology: str,
    status: str,
    lower_offset: int,
    upper_offset: int,
) -> pd.DataFrame:
    """Build diagnostics for flat planned commissioning allocation."""
    assignments = pd.DataFrame(
        {
            "year": assigned_years,
            "output_capacity_mw": undated_df["output_capacity_mw"],
        },
        index=undated_df.index,
    )

    imputed_capacity = (
        assignments.groupby("year")["output_capacity_mw"]
        .sum()
        .reindex(target.index, fill_value=0.0)
    )

    known_capacity = (
        dated_df.groupby("start_year")["output_capacity_mw"]
        .sum()
        .reindex(target.index, fill_value=0.0)
    )

    result = pd.DataFrame(
        {
            "target_imputed_mw": target,
            "known_mw": known_capacity,
            "imputed_mw": imputed_capacity,
        }
    )

    result["final_mw"] = (
        result["known_mw"]
        + result["imputed_mw"]
    )
    result["allocation_error_mw"] = (
        result["imputed_mw"]
        - result["target_imputed_mw"]
    )

    missing_capacity_mw = undated_df["output_capacity_mw"].sum()

    allocation_match = 1.0 - (
        result["allocation_error_mw"].abs().sum()
        / (2.0 * missing_capacity_mw)
    )

    result["country_id"] = country_id
    result["category"] = category
    result["technology"] = technology
    result["status"] = status
    result["lower_offset"] = lower_offset
    result["upper_offset"] = upper_offset
    result["missing_capacity_mw"] = missing_capacity_mw
    result["number_imputed"] = len(undated_df)
    result["years_used"] = assigned_years.nunique()
    result["allocation_match"] = allocation_match

    result.index.name = "year"

    column_order = [
        "country_id",
        "category",
        "technology",
        "status",
        "year",
        "lower_offset",
        "upper_offset",
        "target_imputed_mw",
        "known_mw",
        "imputed_mw",
        "final_mw",
        "allocation_error_mw",
        "missing_capacity_mw",
        "number_imputed",
        "years_used",
        "allocation_match",
    ]

    return result.reset_index()[column_order]

def _impute_start_years_by_capacity_profile(
    prepared_df: pd.DataFrame,
    reference_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
    smoothing_window: int = 1,
) -> tuple[pd.Series, pd.DataFrame]:
    """Impute operating plant dates against a reference capacity profile."""
    start_year = prepared_df["start_year"].copy()
    lifetime = prepared_df["technology"].map(lifetimes)

    result = pd.Series(np.nan, index=prepared_df.index, dtype=float)
    profiles = []

    missing_mask = start_year.isna() & prepared_df["status"].isin(CURRENT)

    if not missing_mask.any():
        return result, pd.DataFrame()

    grouped_missing = prepared_df.loc[missing_mask].groupby(
        ["country_id", "category"], dropna=False
    )

    for (country_id, category), undated_group in grouped_missing:
        reference_category = REFERENCE_CATEGORY_MAP.get(category, category)

        group_mask = (
            prepared_df["country_id"].eq(country_id)
            & prepared_df["category"].eq(category)
            & prepared_df["status"].isin(HISTORICAL)
        )

        dated_group = prepared_df.loc[group_mask & start_year.notna()].copy()
        dated_group["start_year"] = start_year.loc[dated_group.index]

        # this int() is necessary as the subtraction creates a float, was causing errors
        earliest_allocatable_year = int(
            (_utils.DATASET_YEAR - lifetime.loc[undated_group.index]).min()
        )

        allocatable_years = pd.Index(
            range(earliest_allocatable_year, _utils.DATASET_YEAR + 1), name="start_year"
        )

        # this int() is necessary, was causing errors
        reference_first_year = int(
            reference_capacity_df.loc[
                (reference_capacity_df["country_id"] == country_id)
                & (reference_capacity_df["category"] == reference_category),
                "year",
            ].min()
        )

        # the int() is necessary, was causing errors
        observed_first_year = (
            int(dated_group["start_year"].min())
            if not dated_group.empty
            else _utils.DATASET_YEAR
        )

        first_profile_year = min(
            reference_first_year, observed_first_year, earliest_allocatable_year
        )

        profile_years = pd.Index(
            range(first_profile_year, _utils.DATASET_YEAR + 1), name="start_year"
        )

        reference_profile = _build_reference_addition_profile(
            reference_capacity_df=reference_capacity_df,
            country_id=country_id,
            category=reference_category,
            years=profile_years,
            smoothing_window=smoothing_window,
        )

        missing_capacity_mw = undated_group["output_capacity_mw"].sum()

        allocation_profile = _build_clipped_residual_profile(
            dated_df=dated_group,
            profile=reference_profile,
            allocatable_years=allocatable_years,
            missing_capacity_mw=missing_capacity_mw,
            year_col="start_year",
        )

        assigned_years = _allocate_start_years_by_residual_target(
            undated_df=undated_group,
            residual_target=allocation_profile["residual_target_mw"],
            lifetimes=lifetimes,
        )

        result.loc[undated_group.index] = assigned_years

        profiles.append(
            _complete_capacity_profile(
                profile=allocation_profile,
                undated_df=undated_group,
                assigned_years=assigned_years,
                country_id=country_id,
                category=category,
                reference_category=reference_category,
            )
        )

    profile_diagnostics = pd.concat(profiles, ignore_index=True)

    return result, profile_diagnostics


def _impute_retired_dates_by_capacity_profile(
    prepared_df: pd.DataFrame,
    reference_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Impute dates for retired plants with neither date available."""
    imputed_start_year = pd.Series(np.nan, index=prepared_df.index, dtype=float)
    imputed_end_year = pd.Series(np.nan, index=prepared_df.index, dtype=float)
    profiles = []

    missing_mask = (
        prepared_df["status"].eq("retired")
        & prepared_df["start_year"].isna()
        & prepared_df["end_year"].isna()
    )

    if not missing_mask.any():
        return (imputed_start_year, imputed_end_year, pd.DataFrame())

    grouped_missing = prepared_df.loc[missing_mask].groupby(
        ["country_id", "category"], dropna=False
    )

    for (country_id, category), undated_group in grouped_missing:
        reference_category = REFERENCE_CATEGORY_MAP.get(category, category)

        group_mask = (
            prepared_df["country_id"].eq(country_id)
            & prepared_df["category"].eq(category)
            & prepared_df["status"].eq("retired")
        )

        dated_group = prepared_df.loc[
            group_mask & prepared_df["end_year"].notna()
        ].copy()

        reference_first_year = int(
            reference_capacity_df.loc[
                (reference_capacity_df["country_id"] == country_id)
                & (reference_capacity_df["category"] == reference_category)
                & (reference_capacity_df["year"] < _utils.DATASET_YEAR),
                "year",
            ].min()
        )

        observed_first_year = (
            int(dated_group["end_year"].min())
            if not dated_group.empty
            else reference_first_year
        )

        first_profile_year = min(reference_first_year, observed_first_year)

        profile_years = pd.Index(
            range(first_profile_year, _utils.DATASET_YEAR), name="end_year"
        )

        # Missing retirements are allocated only within the period covered
        # by the reference capacity series.
        allocatable_years = pd.Index(
            range(reference_first_year, _utils.DATASET_YEAR), name="end_year"
        )

        reference_profile = _build_reference_retirement_profile(
            reference_capacity_df=reference_capacity_df,
            country_id=country_id,
            category=reference_category,
            years=profile_years,
        )

        reference_profile["profile_source"] = "negative_capacity_change"

        # Where the reference stock contains no reductions, use the timing of
        # already dated retired plants as the retirement-profile basis.
        if reference_profile["profile_fallback_used"].iloc[0]:
            observed_retirement_basis = (
                dated_group.groupby("end_year")["output_capacity_mw"]
                .sum()
                .reindex(profile_years, fill_value=0.0)
            )

            if observed_retirement_basis.sum() > 0:
                reference_profile["reference_profile_basis_mw"] = (
                    observed_retirement_basis
                )
                reference_profile["reference_profile_weight"] = (
                    observed_retirement_basis / observed_retirement_basis.sum()
                )
                reference_profile["profile_source"] = "observed_retirements"

                # Observed retirement dates may precede the reference series.
                allocatable_years = profile_years
            else:
                reference_profile["profile_source"] = "uniform"

        missing_capacity_mw = undated_group["output_capacity_mw"].sum()

        allocation_profile = _build_clipped_residual_profile(
            dated_df=dated_group,
            profile=reference_profile,
            allocatable_years=allocatable_years,
            missing_capacity_mw=missing_capacity_mw,
            year_col="end_year",
        )

        assigned_end_years = _allocate_years_by_target(
            undated_df=undated_group,
            target=allocation_profile["residual_target_mw"],
        )

        assigned_start_years = assigned_end_years - undated_group["technology"].map(
            lifetimes
        )

        imputed_end_year.loc[undated_group.index] = assigned_end_years
        imputed_start_year.loc[undated_group.index] = assigned_start_years

        profiles.append(
            _complete_retirement_profile(
                profile=allocation_profile,
                undated_df=undated_group,
                assigned_end_years=assigned_end_years,
                country_id=country_id,
                category=category,
                reference_category=reference_category,
            )
        )

    retirement_profiles = pd.concat(profiles, ignore_index=True)

    return (imputed_start_year, imputed_end_year, retirement_profiles)

def _impute_planned_start_years(
    prepared_df: pd.DataFrame,
    planned_commissioning_year_windows: dict[
        str,
        dict[str, list[int]],
    ],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Impute missing planned start years using flat capacity targets."""
    imputed_start_year = pd.Series(
        np.nan,
        index=prepared_df.index,
        dtype=float,
    )
    source_type = pd.Series(
        pd.NA,
        index=prepared_df.index,
        dtype="object",
    )
    profiles = []

    missing_mask = (
        prepared_df["status"].isin(PLANNED)
        & prepared_df["start_year"].isna()
    )

    if not missing_mask.any():
        return (
            imputed_start_year,
            source_type,
            pd.DataFrame(),
        )

    grouped_missing = prepared_df.loc[missing_mask].groupby(
        [
            "country_id",
            "category",
            "technology",
            "status",
        ],
        dropna=False,
    )

    for (
        country_id,
        category,
        technology,
        status,
    ), undated_group in grouped_missing:
        lower_offset, upper_offset = (
            planned_commissioning_year_windows[
                technology
            ][status]
        )

        years = pd.Index(
            range(
                _utils.DATASET_YEAR + lower_offset,
                _utils.DATASET_YEAR + upper_offset + 1,
            ),
            name="year",
        )

        missing_capacity_mw = (
            undated_group["output_capacity_mw"].sum()
        )

        flat_target = pd.Series(
            missing_capacity_mw / len(years),
            index=years,
            dtype=float,
            name="target_imputed_mw",
        )

        assigned_years = _allocate_years_by_target(
            undated_df=undated_group,
            target=flat_target,
        )

        imputed_start_year.loc[
            undated_group.index
        ] = assigned_years

        source_type.loc[
            undated_group.index
        ] = f"imputed_{status.replace('-', '_')}_window"

        group_mask = (
            prepared_df["country_id"].eq(country_id)
            & prepared_df["category"].eq(category)
            & prepared_df["technology"].eq(technology)
            & prepared_df["status"].eq(status)
        )

        dated_group = prepared_df.loc[
            group_mask
            & prepared_df["start_year"].notna()
        ].copy()

        profiles.append(
            _complete_planned_commissioning_profile(
                undated_df=undated_group,
                dated_df=dated_group,
                assigned_years=assigned_years,
                target=flat_target,
                country_id=country_id,
                category=category,
                technology=technology,
                status=status,
                lower_offset=lower_offset,
                upper_offset=upper_offset,
            )
        )

    profile_diagnostics = pd.concat(
        profiles,
        ignore_index=True,
    )

    return (
        imputed_start_year,
        source_type,
        profile_diagnostics,
    )

def _impute_start_years_by_group_average(
    prepared_df: pd.DataFrame, group_cols: list[str] | None = None
) -> pd.Series:
    """Impute missing start years using rounded group-average start years."""
    if group_cols is None:
        group_cols = ["country_id", "category", "technology", "status"]

    start_year = prepared_df["start_year"].copy()

    averages = (
        prepared_df.groupby(group_cols, dropna=False)["start_year"]
        .transform("mean")
        .round()
    )

    result = pd.Series(np.nan, index=prepared_df.index, dtype=float)
    mask_na = start_year.isna()
    result.loc[mask_na] = averages.loc[mask_na]

    return result


def _impute_remaining_start_years(
    prepared_df: pd.DataFrame,
    reference_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
    method: str = "group_average",
) -> tuple[pd.Series, pd.DataFrame]:
    """Impute start years that remain missing after direct backfilling."""
    if method == "group_average":
        return (_impute_start_years_by_group_average(prepared_df), pd.DataFrame())

    if method == "capacity_profile":
        return _impute_start_years_by_capacity_profile(
            prepared_df,
            reference_capacity_df=reference_capacity_df,
            lifetimes=lifetimes,
        )

    raise ValueError(
        "Unknown start-year imputation method "
        f"{method!r}. Expected 'group_average' or 'capacity_profile'."
    )


def _impute_start_year(
        prepared_df: pd.DataFrame,
        reference_capacity_df: pd.DataFrame,
        lifetimes: dict[str, int],
        planned_commissioning_year_windows: dict[
            str,
            dict[str, list[int]],
        ],
        method: str = "capacity_profile",
    ) -> tuple[
        pd.Series,
        pd.Series,
        pd.DataFrame,
        pd.DataFrame,
    ]:
    """Impute missing powerplant start years and track source labels."""
    start_year = prepared_df["start_year"].copy()
    start_year_source_type = _initial_year_source_type(start_year)
    lifetime = prepared_df["technology"].map(lifetimes)

    # First, preserve the direct deterministic backfill from known end year.
    direct_backfill_mask = start_year.isna() & prepared_df["end_year"].notna()
    start_year.loc[direct_backfill_mask] = (
        prepared_df.loc[direct_backfill_mask, "end_year"]
        - lifetime.loc[direct_backfill_mask]
    )
    start_year_source_type.loc[direct_backfill_mask] = "derived_from_end_year"

    historical_profile_diagnostics = pd.DataFrame()
    planned_profile_diagnostics = pd.DataFrame()

    # Impute planned projects independently of historical start-year
    # methods. Only the missing planned capacity is distributed across
    # the configured technology- and status-specific window.
    planned_imputation_df = prepared_df.copy()
    planned_imputation_df["start_year"] = start_year

    (
        planned_start_year,
        planned_source_type,
        planned_profile_diagnostics,
    ) = _impute_planned_start_years(
        prepared_df=planned_imputation_df,
        planned_commissioning_year_windows=(
            planned_commissioning_year_windows
        ),
    )

    planned_imputed_mask = (
        start_year.isna()
        & planned_start_year.notna()
    )

    start_year.loc[planned_imputed_mask] = (
        planned_start_year.loc[planned_imputed_mask]
    )
    start_year_source_type.loc[planned_imputed_mask] = (
        planned_source_type.loc[planned_imputed_mask]
    )

    # Apply the configured historical method only to historical plants
    # that still have no start year.
    historical_missing_mask = (
        start_year.isna()
        & prepared_df["status"].isin(HISTORICAL)
    )

    if historical_missing_mask.any():
        imputation_df = prepared_df.copy()
        imputation_df["start_year"] = start_year

        (
            historical_start_year,
            historical_profile_diagnostics,
        ) = _impute_remaining_start_years(
            imputation_df,
            reference_capacity_df=reference_capacity_df,
            lifetimes=lifetimes,
            method=method,
        )

        historical_imputed_mask = (
            historical_missing_mask
            & historical_start_year.notna()
        )

        start_year.loc[historical_imputed_mask] = (
            historical_start_year.loc[
                historical_imputed_mask
            ]
        )
        start_year_source_type.loc[
            historical_imputed_mask
        ] = f"imputed_{method}"

    return (
        start_year,
        start_year_source_type,
        historical_profile_diagnostics,
        planned_profile_diagnostics,
    )


def _impute_end_year(
    df: pd.DataFrame, lifetimes: dict[str, int], delay: dict[str, int]
) -> tuple[pd.Series, pd.Series]:
    """Impute end_year using lifetime and track source labels.

    Old plants operating beyond lifetime will be retired with a given delay.
    """
    ref_year = _utils.DATASET_YEAR

    end_year = df["end_year"].copy()
    end_year_source_type = _initial_year_source_type(end_year)

    expected_end = df["start_year"] + df["technology"].map(lifetimes)

    lifetime_fill_mask = end_year.isna() & expected_end.notna()
    result = end_year.copy()
    result.loc[lifetime_fill_mask] = expected_end.loc[lifetime_fill_mask]
    end_year_source_type.loc[lifetime_fill_mask] = "derived_from_start_year_lifetime"

    # Cap lifetime-derived end years so that plants recorded as retired
    # end before the dataset year.
    retired_lifetime_fill_mask = lifetime_fill_mask & df["status"].eq("retired")
    retired_end_capped_mask = retired_lifetime_fill_mask & result.ge(ref_year)
    result.loc[retired_end_capped_mask] = ref_year - 1
    end_year_source_type.loc[retired_end_capped_mask] = (
        "derived_from_start_year_lifetime_capped_to_retired_status"
    )

    # Plants operating beyond expected lifetime will be retired after a delay
    # of >=1 yr.
    needs_delay = (result <= ref_year) & (df["status"] == "operating")
    delayed_end = result + df["technology"].map(delay).fillna(0).astype(int)
    delayed_end = delayed_end.clip(lower=ref_year + 1)

    result.loc[needs_delay] = delayed_end.loc[needs_delay]

    end_year_source_type.loc[needs_delay & lifetime_fill_mask] = (
        "derived_from_start_year_lifetime_with_retirement_delay"
    )

    end_year_source_type.loc[needs_delay & ~lifetime_fill_mask] = (
        "observed_adjusted_with_retirement_delay"
    )

    return result, end_year_source_type


def _reconcile_status_from_observed_dates(
    df: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Correct status before imputation where observed dates are decisive.

    This function deliberately uses only the dates already present in the
    input data. Observed retirement years are treated as the strongest signal:
    if an observed end year is on or before the dataset year, the plant is
    treated as retired before any lifetime or retirement-delay logic is
    applied.

    Future observed start years are treated as planned. If the original status
    already records a planned-development stage (construction,
    pre-construction, announced), that stage is preserved because planned
    start-year imputation still needs it. Otherwise the status is corrected to
    the generic temporal status "planned".
    """
    ref_year = _utils.DATASET_YEAR

    corrected_status = df["status"].copy()
    correction_source = pd.Series(
        "unchanged_original_status",
        index=df.index,
        dtype="object",
    )

    observed_start_year = df["start_year"]
    observed_end_year = df["end_year"]

    retired_from_observed_end = observed_end_year.notna() & observed_end_year.le(
        ref_year
    )
    corrected_status.loc[retired_from_observed_end] = "retired"
    correction_source.loc[retired_from_observed_end] = "observed_end_year"

    operating_from_observed_dates = (
        ~retired_from_observed_end
        & observed_start_year.notna()
        & observed_start_year.le(ref_year)
        & observed_end_year.notna()
        & observed_end_year.gt(ref_year)
    )
    corrected_status.loc[operating_from_observed_dates] = "operating"
    correction_source.loc[operating_from_observed_dates] = (
        "observed_start_and_end_year"
    )

    planned_from_observed_start = (
        ~retired_from_observed_end
        & observed_start_year.notna()
        & observed_start_year.gt(ref_year)
    )

    planned_stage_already_known = (
        planned_from_observed_start
        & corrected_status.isin(PLANNED)
    )
    correction_source.loc[planned_stage_already_known] = (
        "observed_start_year_confirms_planned_status"
    )

    generic_planned_needed = (
        planned_from_observed_start
        & ~corrected_status.isin(PLANNED)
    )
    corrected_status.loc[generic_planned_needed] = "planned"
    correction_source.loc[generic_planned_needed] = "observed_start_year"

    return corrected_status, correction_source


def _impute_status(df: pd.DataFrame) -> pd.Series:
    """Derive final temporal status after start/end years are complete."""
    status = df["status"].copy()
    ref_year = _utils.DATASET_YEAR
    status.loc[ref_year < df["start_year"]] = "planned"
    status.loc[(df["start_year"] <= ref_year) & (ref_year < df["end_year"])] = (
        "operating"
    )
    status.loc[df["end_year"] <= ref_year] = "retired"

    if status.isna().any():
        raise ValueError("Entries with ambiguous states were left in the dataframe.")

    return status


def _build_age_imputation_diagnostics(
    original_df: pd.DataFrame,
    aged_df: pd.DataFrame,
    status_after_observed_date_correction: pd.Series,
    status_observed_date_correction_source: pd.Series,
    status_final: pd.Series,
    start_year_source_type: pd.Series,
    end_year_source_type: pd.Series,
    lifetimes: dict[str, int],
    start_year_imputation_method: str,
) -> pd.DataFrame:
    """Build row-level diagnostics for powerplant age imputation."""
    passthrough_cols = [
        "powerplant_id",
        "name",
        "country_id",
        "category",
        "technology",
        "output_capacity_mw",
    ]

    diagnostics = original_df[passthrough_cols].copy()

    diagnostics["lifetime"] = original_df["technology"].map(lifetimes)

    diagnostics["status_original"] = original_df["status"]
    diagnostics["status_after_observed_date_correction"] = (
        status_after_observed_date_correction
    )
    diagnostics["status_observed_date_correction_source"] = (
        status_observed_date_correction_source
    )
    diagnostics["status_final"] = status_final

    diagnostics["start_year_original"] = original_df["start_year"]
    diagnostics["start_year"] = aged_df["start_year"]
    diagnostics["start_year_source_type"] = start_year_source_type

    diagnostics["end_year_original"] = original_df["end_year"]
    diagnostics["end_year"] = aged_df["end_year"]
    diagnostics["end_year_source_type"] = end_year_source_type

    # TODO: these source types should be reflected in scripts/_schemas.
    diagnostics["retained_after_time_imputation"] = (
        diagnostics["start_year"].notna() & diagnostics["end_year"].notna()
    )

    diagnostics["start_year_imputation_method"] = start_year_imputation_method

    return diagnostics.reset_index(drop=True)

CAPACITY_DATE_EVENT_COLUMNS = [
    "powerplant_id",
    "name",
    "country_id",
    "category",
    "technology",
    "status_original",
    "status_final",
    "year",
    "event_type",
    "source_type",
    "source_label",
    "output_capacity_mw",
    "capacity_change_mw",
]


def _build_capacity_date_events(
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Convert plant dates into annual commissioning and retirement events."""
    if diagnostics.empty:
        return pd.DataFrame(columns=CAPACITY_DATE_EVENT_COLUMNS)

    retained = diagnostics.loc[
        diagnostics["retained_after_time_imputation"]
    ].copy()

    if retained.empty:
        return pd.DataFrame(columns=CAPACITY_DATE_EVENT_COLUMNS)

    common_cols = [
        "powerplant_id",
        "name",
        "country_id",
        "category",
        "technology",
        "status_original",
        "status_final",
        "output_capacity_mw",
    ]

    start_events = retained[
        common_cols
        + [
            "start_year",
            "start_year_source_type",
        ]
    ].rename(
        columns={
            "start_year": "year",
            "start_year_source_type": "source_type",
        }
    )
    start_events["event_type"] = "commissioning"
    start_events["capacity_change_mw"] = (
        start_events["output_capacity_mw"]
    )

    end_events = retained[
        common_cols
        + [
            "end_year",
            "end_year_source_type",
        ]
    ].rename(
        columns={
            "end_year": "year",
            "end_year_source_type": "source_type",
        }
    )
    end_events["event_type"] = "retirement"
    end_events["capacity_change_mw"] = (
        -end_events["output_capacity_mw"]
    )

    events = pd.concat(
        [
            start_events,
            end_events,
        ],
        ignore_index=True,
    )

    events["source_label"] = (
        events["source_type"]
        .map(DATE_SOURCE_LABELS)
        .fillna(events["source_type"])
    )

    return (
        events[CAPACITY_DATE_EVENT_COLUMNS]
        .sort_values(
            [
                "country_id",
                "year",
                "event_type",
                "source_type",
                "powerplant_id",
            ]
        )
        .reset_index(drop=True)
    )

def _save_age_imputation_diagnostics(
    diagnostics: pd.DataFrame, output_path_str: str
) -> None:
    """Save row-level age-imputation diagnostics if requested."""
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(output_path, index=False)


def _save_age_profile_diagnostics(
    diagnostics: pd.DataFrame, output_path_str: str | None
) -> None:
    """Save annual age-imputation profile diagnostics."""
    if output_path_str is None:
        return

    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_parquet(output_path, index=False)

def _save_capacity_date_events(
    events: pd.DataFrame,
    output_path_str: str,
) -> None:
    """Save annual capacity-date events."""
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output_path, index=False)

def plot_capacity_date_events(
    events_df: pd.DataFrame,
    commissioning_profile_df: pd.DataFrame,
    retirement_profile_df: pd.DataFrame,
    planned_profile_df: pd.DataFrame,
    output_path: str,
    cat: str,
) -> None:
    """Plot commissioning and retirement events by date-source method."""
    display_category = cat.replace("_", " ")
    suptitle = f"Capacity-date imputation for {display_category}"

    country_sets = []

    for dataframe in [
        events_df,
        commissioning_profile_df,
        retirement_profile_df,
        planned_profile_df,
    ]:
        if not dataframe.empty:
            country_sets.extend(
                dataframe["country_id"].unique()
            )

    countries = sorted(set(country_sets))

    if not countries:
        _plots.plot_empty(suptitle, output_path)
        return

    present_source_types = (
        events_df["source_type"]
        .unique()
        .tolist()
    )

    source_types = [
        source_type
        for source_type in DATE_SOURCE_ORDER
        if source_type in present_source_types
    ]

    source_types.extend(
        sorted(
            set(present_source_types)
            - set(source_types)
        )
    )

    source_colors = {}
    fallback_index = 0

    for source_type in source_types:
        if source_type in DATE_SOURCE_COLORS:
            source_colors[source_type] = DATE_SOURCE_COLORS[source_type]
        else:
            source_colors[source_type] = FALLBACK_DATE_SOURCE_COLORS[
                fallback_index % len(FALLBACK_DATE_SOURCE_COLORS)
            ]
            fallback_index += 1

    n_countries = len(countries)
    cols = 2 if n_countries > 1 else 1
    rows = math.ceil(n_countries / cols)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 7, rows * 4.5),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axes_flat = np.array(axes).ravel()

    for ax, country in zip(axes_flat, countries):
        country_events = events_df.loc[
            events_df["country_id"].eq(country)
        ].copy()

        commissioning_target = (
            commissioning_profile_df.loc[
                commissioning_profile_df["country_id"].eq(country)
            ].copy()
            if not commissioning_profile_df.empty
            else pd.DataFrame()
        )

        retirement_target = (
            retirement_profile_df.loc[
                retirement_profile_df["country_id"].eq(country)
            ].copy()
            if not retirement_profile_df.empty
            else pd.DataFrame()
        )

        planned_target = (
            planned_profile_df.loc[
                planned_profile_df["country_id"].eq(country)
            ].copy()
            if not planned_profile_df.empty
            else pd.DataFrame()
        )

        year_values = set(
            country_events["year"]
            .astype(int)
        )

        if not commissioning_target.empty:
            year_values.update(
                commissioning_target["year"].astype(int)
            )

        if not retirement_target.empty:
            year_values.update(
                retirement_target["year"].astype(int)
            )

        if not planned_target.empty:
            year_values.update(
                planned_target["year"].astype(int)
            )

        years = np.array(sorted(year_values))

        if len(years) == 0:
            _plots.draw_empty(
                ax,
                country,
                f"No date events for {country}",
            )
            continue

        country_events["year"] = (
            country_events["year"].astype(int)
        )

        annual_events = (
            country_events.groupby(
                [
                    "year",
                    "event_type",
                    "source_type",
                ],
                as_index=False,
            )["capacity_change_mw"]
            .sum()
        )

        positive_bottom = np.zeros(len(years))
        negative_bottom = np.zeros(len(years))

        for source_type in source_types:
            commissioning_values = (
                annual_events.loc[
                    annual_events["source_type"].eq(source_type)
                    & annual_events["event_type"].eq("commissioning")
                ]
                .set_index("year")["capacity_change_mw"]
                .reindex(years, fill_value=0.0)
                .to_numpy()
            )

            retirement_values = (
                annual_events.loc[
                    annual_events["source_type"].eq(source_type)
                    & annual_events["event_type"].eq("retirement")
                ]
                .set_index("year")["capacity_change_mw"]
                .reindex(years, fill_value=0.0)
                .to_numpy()
            )

            if commissioning_values.any():
                ax.bar(
                    years,
                    commissioning_values,
                    bottom=positive_bottom,
                    color=source_colors[source_type],
                    width=0.9,
                )

            if retirement_values.any():
                ax.bar(
                    years,
                    retirement_values,
                    bottom=negative_bottom,
                    color=source_colors[source_type],
                    width=0.9,
                )

            positive_bottom += commissioning_values
            negative_bottom += retirement_values

        if not commissioning_target.empty:
            commissioning_target = (
                commissioning_target.sort_values("year")
            )

            ax.plot(
                commissioning_target["year"],
                commissioning_target["target_final_mw"],
                color="0.15",
                linewidth=2,
            )

        if not retirement_target.empty:
            retirement_target = (
                retirement_target.sort_values("year")
            )

            ax.plot(
                retirement_target["year"],
                -retirement_target["target_final_mw"],
                color="0.15",
                linewidth=2,
                linestyle="--",
            )
        
        if not planned_target.empty:
            planned_target = planned_target.groupby("year",as_index=False)["target_imputed_mw"].sum().sort_values("year")

            ax.plot(
                planned_target["year"],
                planned_target["target_imputed_mw"],
                color="0.35",
                linewidth=2,
                linestyle=":",
            )

        ax.axhline(
            0,
            color="0.35",
            linewidth=0.8,
        )
        ax.set_title(country)
        ax.set_xlabel("Year")
        ax.set_ylabel("Annual capacity event (MW)")
        ax.locator_params(axis="x", nbins=12)
        ax.tick_params(axis="x", rotation=45)
        ax.minorticks_off()

    for ax in axes_flat[n_countries:]:
        ax.set_visible(False)

    legend_handles = [
        Patch(
            facecolor=source_colors[source_type],
            label=DATE_SOURCE_LABELS.get(
                source_type,
                source_type,
            ),
        )
        for source_type in source_types
    ]

    if not commissioning_profile_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=2,
                label="Commissioning target",
            )
        )

    if not retirement_profile_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=2,
                linestyle="--",
                label="Retirement target",
            )
        )

    if not planned_profile_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                linewidth=2,
                linestyle=":",
                label="Planned imputation target",
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    fig.suptitle(suptitle, fontsize=14)

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)

def impute(
    relocated_gdf: gpd.GeoDataFrame,
    reference_capacity_df: pd.DataFrame,
    imputation: dict,
    technology_mapping: dict,
    age_imputation_output_path: str,
    age_profile_output_path: str,
    retirement_profile_output_path: str,
    planned_profile_output_path: str,
) -> gpd.GeoDataFrame:
    """Add automatic and user imputations to fill missing data.

    Args:
        relocated_gdf: Relocated powerplants with country identifiers.
        reference_capacity_df: Annual category-level capacity stock data.
        imputation: Imputation configuration.
        technology_mapping: Technology-mapping configuration.
        age_imputation_output_path: Path for row-level imputation diagnostics.
        age_profile_output_path: Path for annual profile diagnostics.
        retirement_profile_output_path: Path for annual retirement-profile diagnostics.
        planned_profile_output_path: Path for planned commissioning-profile diagnostics.
    """
    if relocated_gdf.empty:
        imputed = relocated_gdf
        _save_age_imputation_diagnostics(pd.DataFrame(), age_imputation_output_path)
        _save_age_profile_diagnostics(pd.DataFrame(), age_profile_output_path)
        _save_age_profile_diagnostics(pd.DataFrame(), retirement_profile_output_path)
        _save_age_profile_diagnostics(pd.DataFrame(), planned_profile_output_path)
    else:
        _utils.check_single_category(relocated_gdf)
        if (relocated_gdf.geometry.geom_type != "Point").any():
            raise ValueError(
                "Polygon powerplant geometries detected. Only Points are supported."
            )

        lifetimes = imputation["lifetime_years"]
        planned_commissioning_year_windows = imputation["planned_commissioning_year_windows"]
        retirement_delay_years = imputation["retirement_delay_years"]
        scenario = SCENARIO_MAP[imputation["scenario"]]
        start_year_imputation_method = imputation["start_year_imputation_method"]

        (
            status_after_observed_date_correction,
            status_observed_date_correction_source,
        ) = _reconcile_status_from_observed_dates(relocated_gdf)

        # Get facilities within the requested scenario after correcting only
        # the statuses that are already contradicted by observed dates.
        scenario_gdf = relocated_gdf.copy()
        scenario_gdf["status"] = status_after_observed_date_correction

        imputed = scenario_gdf[scenario_gdf["status"].isin(scenario)].copy()
        original = relocated_gdf.loc[imputed.index].copy()

        if not imputed.empty:
            (
                imputed["start_year"],
                start_year_source_type,
                profile_diagnostics,
                planned_profile_diagnostics,
            ) = _impute_start_year(
                prepared_df=imputed,
                reference_capacity_df=reference_capacity_df,
                lifetimes=lifetimes,
                planned_commissioning_year_windows=(
                    planned_commissioning_year_windows
                ),
                method=start_year_imputation_method,
            )

            imputed["end_year"], end_year_source_type = _impute_end_year(
                imputed, lifetimes, retirement_delay_years
            )

            retired_start_year, retired_end_year, retirement_profile_diagnostics = (
                _impute_retired_dates_by_capacity_profile(
                    prepared_df=imputed,
                    reference_capacity_df=reference_capacity_df,
                    lifetimes=lifetimes,
                )
            )

            _save_age_profile_diagnostics(
                retirement_profile_diagnostics, retirement_profile_output_path
            )

            retirement_imputed_mask = retired_end_year.notna()

            imputed.loc[retirement_imputed_mask, "start_year"] = retired_start_year.loc[
                retirement_imputed_mask
            ]

            imputed.loc[retirement_imputed_mask, "end_year"] = retired_end_year.loc[
                retirement_imputed_mask
            ]

            start_year_source_type.loc[retirement_imputed_mask] = (
                "derived_from_imputed_retirement_end_year"
            )
            end_year_source_type.loc[retirement_imputed_mask] = (
                "imputed_retirement_capacity_profile"
            )

            has_complete_dates = imputed[["start_year", "end_year"]].notna().all(axis=1)

            status_final = pd.Series(pd.NA, index=imputed.index, dtype="object")

            if has_complete_dates.any():
                status_final.loc[has_complete_dates] = _impute_status(
                    imputed.loc[has_complete_dates]
                )


            diagnostics = _build_age_imputation_diagnostics(
                original_df=original,
                aged_df=imputed,
                status_after_observed_date_correction=(
                    status_after_observed_date_correction.loc[imputed.index]
                ),
                status_observed_date_correction_source=(
                    status_observed_date_correction_source.loc[imputed.index]
                ),
                status_final=status_final,
                start_year_source_type=start_year_source_type,
                end_year_source_type=end_year_source_type,
                lifetimes=lifetimes,
                start_year_imputation_method=start_year_imputation_method,
            )
            _save_age_imputation_diagnostics(diagnostics, age_imputation_output_path)
            _save_age_profile_diagnostics(profile_diagnostics, age_profile_output_path)
            _save_age_profile_diagnostics(planned_profile_diagnostics, planned_profile_output_path)

            # Drop projects with insufficient date data.
            imputed = imputed.loc[has_complete_dates].copy()

            # Update the powerplant status.
            imputed["status"] = status_final.loc[imputed.index]
        else:
            diagnostics = _build_age_imputation_diagnostics(
                original_df=original,
                aged_df=imputed,
                status_after_observed_date_correction=(
                    status_after_observed_date_correction.loc[imputed.index]
                ),
                status_observed_date_correction_source=(
                    status_observed_date_correction_source.loc[imputed.index]
                ),
                status_final=pd.Series(pd.NA, index=imputed.index, dtype="object"),
                start_year_source_type=pd.Series(
                    pd.NA, index=imputed.index, dtype="object"
                ),
                end_year_source_type=pd.Series(
                    pd.NA, index=imputed.index, dtype="object"
                ),
                lifetimes=lifetimes,
                start_year_imputation_method=start_year_imputation_method,
            )
            _save_age_imputation_diagnostics(diagnostics, age_imputation_output_path)
            _save_age_profile_diagnostics(pd.DataFrame(), age_profile_output_path)
            _save_age_profile_diagnostics(pd.DataFrame(), retirement_profile_output_path)
            _save_age_profile_diagnostics(pd.DataFrame(), planned_profile_output_path)

    schema = _schemas.build_schema(technology_mapping, "impute")
    return schema.validate(imputed)


def explore(imputed: gpd.GeoDataFrame, output_path: str, colormap="tab20"):
    """Create a HTML map for users to explore."""
    if imputed.empty:
        with open(output_path, "w") as f:
            f.write("No data")
    else:
        explorer = imputed.explore(
            column="technology", legend=True, popup=True, cmap=colormap
        )
        explorer.save(output_path)


def plot_powerplant_capacity_buildup(
    df: pd.DataFrame, output_path: str, colormap: str, cat: str = "powerplant"
):
    """Plot stacked bar charts of active powerplant capacity over time per country.

    Input should be a powerplant capacity file of a single category.
    """
    suptitle = f"Active {cat} capacity by technology per country"

    if df.empty:
        _plots.plot_empty(suptitle, output_path)
        return

    # Year range (x-axis)
    start_year = df["start_year"].astype(int).min()
    end_year = df["end_year"].astype(int).max()
    years = list(range(start_year, end_year + 1))

    # Layout (per country in alphabetical order)
    countries = sorted(df["country_id"].unique())
    n_countries = len(countries)
    cols = 2 if n_countries > 1 else 1
    rows = math.ceil(n_countries / cols)

    # Tech type color range
    tech_types = sorted(df["technology"].unique())
    cmap = Colormap(colormap).to_mpl()
    colors = [cmap(i) for i in np.linspace(0, 1, len(tech_types))]

    # Figure (always 2 columns, flexible rows)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 5, rows * 4),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axes_flat = np.array(axes).ravel()

    # Plot per country
    for ax, country in zip(axes_flat, countries):
        country_df = df[df["country_id"] == country]
        if country_df.empty:
            _plots.draw_empty(ax, country, f"No data for {country}")
            continue

        cap_mw = pd.DataFrame(0.0, index=years, columns=tech_types)
        for year in years:
            active = country_df[
                (country_df["start_year"] <= year) & (year < country_df["end_year"])
            ]
            cap_mw.loc[year] = (
                active.groupby("technology")["output_capacity_mw"]
                .sum()
                .reindex(tech_types, fill_value=0)
            )

        cap_mw.plot(kind="bar", stacked=True, ax=ax, color=colors, legend=False, rot=45)
        ax.set_title(country)
        ax.set_ylabel("Capacity (MW)")
        ax.locator_params(axis="x", nbins=10)
        ax.minorticks_off()

    # Hide extra axes
    for ax in axes_flat[n_countries:]:
        ax.set_visible(False)

    # Add details
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        title="Technology",
        frameon=False,
    )
    fig.suptitle(suptitle, fontsize=14)

    fig.savefig(output_path, bbox_inches="tight")


def main() -> None:
    """Main snakemake process."""
    imputed_gdf = impute(
        relocated_gdf=gpd.read_parquet(snakemake.input.relocated),
        reference_capacity_df=pd.read_parquet(snakemake.input.category_capacity),
        imputation=snakemake.params.imputation,
        technology_mapping=snakemake.params.tech_map,
        age_imputation_output_path=snakemake.output.age_imputation,
        age_profile_output_path=snakemake.output.age_profile,
        retirement_profile_output_path=snakemake.output.retirement_profile,
        planned_profile_output_path=snakemake.output.planned_profile,
    )
    imputed_gdf.to_parquet(snakemake.output.aged)

    age_diagnostics_df = pd.read_parquet(
        snakemake.output.age_imputation
    )

    capacity_date_events_df = _build_capacity_date_events(
        age_diagnostics_df
    )

    _save_capacity_date_events(
        capacity_date_events_df,
        snakemake.output.capacity_date_events,
    )

    plot_powerplant_capacity_buildup(
        imputed_gdf,
        snakemake.output.histogram,
        "seaborn:tab20",
        snakemake.wildcards.category,
    )
    explore(imputed_gdf, snakemake.output.explorer)

    commissioning_profile_df = pd.read_parquet(
        snakemake.output.age_profile
    )
    retirement_profile_df = pd.read_parquet(
        snakemake.output.retirement_profile
    )

    planned_profile_df = pd.read_parquet(
        snakemake.output.planned_profile
    )

    plot_capacity_date_events(
        events_df=capacity_date_events_df,
        commissioning_profile_df=commissioning_profile_df,
        retirement_profile_df=retirement_profile_df,
        output_path=snakemake.output.capacity_date_plot,
        cat=snakemake.wildcards.category,
        planned_profile_df=planned_profile_df,
    )


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
