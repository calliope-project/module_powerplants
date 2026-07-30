"""Imputation of missing values."""

import math
import sys
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

# Status Setup
OPERATING = "operating"
RETIRED = "retired"
HISTORICAL = {OPERATING, RETIRED}
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


def _reference_categories(category: str) -> list[str]:
    """Return reference-capacity categories matching a powerplant category."""
    return _utils.listify(REFERENCE_CATEGORY_MAP.get(category, category))


def _reference_capacity_stock(
    reference_capacity_df: pd.DataFrame,
    country_id: str,
    categories: list[str],
    max_year: int,
) -> pd.Series:
    """Return annual reference-capacity stock for mapped categories.

    Multiple reference categories may map to one powerplant category, such as
    hydropower and pumped storage. These are summed by year before deriving
    commissioning or retirement profiles.
    """
    return (
        reference_capacity_df.loc[
            reference_capacity_df["country_id"].eq(country_id)
            & reference_capacity_df["category"].isin(categories)
            & reference_capacity_df["year"].le(max_year),
            ["year", "capacity_mw"],
        ]
        .groupby("year", as_index=True)["capacity_mw"]
        .sum()
        .sort_index()
    )


def _initial_year_source_type(year: pd.Series) -> pd.Series:
    """Label whether year values were originally present or missing."""
    source_type = pd.Series("observed", index=year.index, dtype="object")
    source_type.loc[year.isna()] = "missing_unresolved"
    return source_type


def _build_reference_addition_profile(
    reference_capacity_df: pd.DataFrame,
    country_id: str,
    categories: list[str],
    years: pd.Index,
    smoothing_window: int = 1,
) -> pd.DataFrame:
    """Build an annual commissioning profile from capacity stock data."""
    capacity_stock = _reference_capacity_stock(
        reference_capacity_df=reference_capacity_df,
        country_id=country_id,
        categories=categories,
        max_year=_utils.DATASET_YEAR,
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

    # If the reference series contains no positive capacity changes, there is no
    # commissioning pattern to follow. Use a uniform profile so missing capacity can
    # still be allocated across the candidate years.
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
    reference_capacity_df: pd.DataFrame,
    country_id: str,
    categories: list[str],
    years: pd.Index,
) -> pd.DataFrame:
    """Build an annual retirement profile from capacity-stock reductions."""
    capacity_stock = _reference_capacity_stock(
        reference_capacity_df=reference_capacity_df,
        country_id=country_id,
        categories=categories,
        max_year=_utils.DATASET_YEAR - 1,
    )

    reference_stock = capacity_stock.reindex(years)

    # Negative annual stock changes provide the retirement profile.
    # Positive changes represent net additions and carry no retirement weight.
    reference_negative_change = (
        (-capacity_stock.diff()).clip(lower=0.0).reindex(years).fillna(0.0)
    )

    profile_basis = reference_negative_change.copy()
    profile_fallback_used = profile_basis.sum() <= 0

    # If the reference series contains no negative capacity changes, there is no
    # retirement pattern to follow. Use a uniform profile so missing retirements can
    # still be allocated across the candidate years.
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


def _allocate_years_by_target(undated_df: pd.DataFrame, target: pd.Series) -> pd.Series:
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
    assigned_years = pd.Series(np.nan, index=undated_df.index, dtype=float)

    for plant_index in order.index:
        plant_capacity = undated_df.loc[plant_index, "output_capacity_mw"]

        target_ranking = pd.DataFrame(
            {
                "year": remaining_target.index,
                "remaining_target": remaining_target.to_numpy(),
                "original_target": target.to_numpy(),
            }
        ).sort_values(
            ["remaining_target", "original_target", "year"],
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
    """Return the commissioning-profile target used in the diagnostic plot."""
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

    return pd.DataFrame(
        {
            "country_id": country_id,
            "category": category,
            "reference_category": reference_category,
            "year": profile.index,
            "target_final_mw": profile["target_final_mw"],
            "observed_mw": profile["observed_mw"],
            "imputed_mw": imputed_capacity,
        }
    )


def _complete_retirement_profile(
    profile: pd.DataFrame,
    undated_df: pd.DataFrame,
    assigned_end_years: pd.Series,
    country_id: str,
    category: str,
    reference_category: str,
) -> pd.DataFrame:
    """Return the retirement-profile target used in the diagnostic plot."""
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

    return pd.DataFrame(
        {
            "country_id": country_id,
            "category": category,
            "reference_category": reference_category,
            "profile_source": profile["profile_source"],
            "year": profile.index,
            "target_final_mw": profile["target_final_mw"],
            "observed_mw": profile["observed_mw"],
            "imputed_mw": imputed_capacity,
        }
    )


def _complete_planned_commissioning_profile(
    undated_df: pd.DataFrame,
    assigned_years: pd.Series,
    target: pd.Series,
    country_id: str,
    category: str,
    technology: str,
    status: str,
) -> pd.DataFrame:
    """Return the planned commissioning profile used in the diagnostic plot."""
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

    return pd.DataFrame(
        {
            "country_id": country_id,
            "category": category,
            "technology": technology,
            "status": status,
            "year": target.index,
            "target_imputed_mw": target,
            "imputed_mw": imputed_capacity,
        }
    )


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

    missing_mask = start_year.isna() & prepared_df["status"].eq(OPERATING)

    if not missing_mask.any():
        return result, pd.DataFrame()

    grouped_missing = prepared_df.loc[missing_mask].groupby(
        ["country_id", "category"], dropna=False
    )

    for (country_id, category), undated_group in grouped_missing:
        reference_categories = _reference_categories(category)
        reference_category_label = "+".join(reference_categories)

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
                & (reference_capacity_df["category"].isin(reference_categories)),
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
            categories=reference_categories,
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
                reference_category=reference_category_label,
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
        prepared_df["status"].eq(RETIRED)
        & prepared_df["start_year"].isna()
        & prepared_df["end_year"].isna()
    )

    if not missing_mask.any():
        return (imputed_start_year, imputed_end_year, pd.DataFrame())

    grouped_missing = prepared_df.loc[missing_mask].groupby(
        ["country_id", "category"], dropna=False
    )

    for (country_id, category), undated_group in grouped_missing:
        reference_categories = _reference_categories(category)
        reference_category_label = "+".join(reference_categories)

        group_mask = (
            prepared_df["country_id"].eq(country_id)
            & prepared_df["category"].eq(category)
            & prepared_df["status"].eq(RETIRED)
        )

        dated_group = prepared_df.loc[
            group_mask & prepared_df["end_year"].notna()
        ].copy()

        reference_first_year = int(
            reference_capacity_df.loc[
                (reference_capacity_df["country_id"] == country_id)
                & (reference_capacity_df["category"].isin(reference_categories))
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
            categories=reference_categories,
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
            undated_df=undated_group, target=allocation_profile["residual_target_mw"]
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
                reference_category=reference_category_label,
            )
        )

    retirement_profiles = pd.concat(profiles, ignore_index=True)

    return (imputed_start_year, imputed_end_year, retirement_profiles)


def _impute_planned_start_years(
    prepared_df: pd.DataFrame,
    planned_commissioning_year_windows: dict[str, dict[str, list[int]]],
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Impute missing planned start years using flat capacity targets."""
    imputed_start_year = pd.Series(np.nan, index=prepared_df.index, dtype=float)
    source_type = pd.Series(pd.NA, index=prepared_df.index, dtype="object")
    profiles = []

    missing_mask = (
        prepared_df["status"].isin(PLANNED) & prepared_df["start_year"].isna()
    )

    if not missing_mask.any():
        return (imputed_start_year, source_type, pd.DataFrame())

    grouped_missing = prepared_df.loc[missing_mask].groupby(
        ["country_id", "category", "technology", "status"], dropna=False
    )

    for (country_id, category, technology, status), undated_group in grouped_missing:
        lower_offset, upper_offset = planned_commissioning_year_windows[technology][
            status
        ]

        years = pd.Index(
            range(
                _utils.DATASET_YEAR + lower_offset,
                _utils.DATASET_YEAR + upper_offset + 1,
            ),
            name="year",
        )

        missing_capacity_mw = undated_group["output_capacity_mw"].sum()

        flat_target = pd.Series(
            missing_capacity_mw / len(years),
            index=years,
            dtype=float,
            name="target_imputed_mw",
        )

        assigned_years = _allocate_years_by_target(
            undated_df=undated_group, target=flat_target
        )

        imputed_start_year.loc[undated_group.index] = assigned_years

        source_type.loc[undated_group.index] = (
            f"imputed_{status.replace('-', '_')}_window"
        )

        profiles.append(
            _complete_planned_commissioning_profile(
                undated_df=undated_group,
                assigned_years=assigned_years,
                target=flat_target,
                country_id=country_id,
                category=category,
                technology=technology,
                status=status,
            )
        )

    profile_diagnostics = pd.concat(profiles, ignore_index=True)

    return (imputed_start_year, source_type, profile_diagnostics)


def _impute_remaining_start_years(
    prepared_df: pd.DataFrame,
    reference_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
    method: str = "capacity_profile",
) -> tuple[pd.Series, pd.DataFrame]:
    """Impute start years that remain missing after direct backfilling."""
    if method != "capacity_profile":
        raise ValueError(
            "Unknown start-year imputation method "
            f"{method!r}. Expected 'capacity_profile'."
        )

    return _impute_start_years_by_capacity_profile(
        prepared_df, reference_capacity_df=reference_capacity_df, lifetimes=lifetimes
    )


def _impute_start_year(
    prepared_df: pd.DataFrame,
    reference_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
    planned_commissioning_year_windows: dict[str, dict[str, list[int]]],
    method: str = "capacity_profile",
) -> tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
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

    (planned_start_year, planned_source_type, planned_profile_diagnostics) = (
        _impute_planned_start_years(
            prepared_df=planned_imputation_df,
            planned_commissioning_year_windows=(planned_commissioning_year_windows),
        )
    )

    planned_imputed_mask = start_year.isna() & planned_start_year.notna()

    start_year.loc[planned_imputed_mask] = planned_start_year.loc[planned_imputed_mask]
    start_year_source_type.loc[planned_imputed_mask] = planned_source_type.loc[
        planned_imputed_mask
    ]

    # Apply the configured historical method only to historical plants
    # that still have no start year.
    historical_missing_mask = start_year.isna() & prepared_df["status"].isin(HISTORICAL)

    if historical_missing_mask.any():
        imputation_df = prepared_df.copy()
        imputation_df["start_year"] = start_year

        (historical_start_year, historical_profile_diagnostics) = (
            _impute_remaining_start_years(
                imputation_df,
                reference_capacity_df=reference_capacity_df,
                lifetimes=lifetimes,
                method=method,
            )
        )

        historical_imputed_mask = (
            historical_missing_mask & historical_start_year.notna()
        )

        start_year.loc[historical_imputed_mask] = historical_start_year.loc[
            historical_imputed_mask
        ]
        start_year_source_type.loc[historical_imputed_mask] = f"imputed_{method}"

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
    retired_lifetime_fill_mask = lifetime_fill_mask & df["status"].eq(RETIRED)
    retired_end_capped_mask = retired_lifetime_fill_mask & result.ge(ref_year)
    result.loc[retired_end_capped_mask] = ref_year - 1
    end_year_source_type.loc[retired_end_capped_mask] = (
        "derived_from_start_year_lifetime_capped_to_retired_status"
    )

    # Plants operating beyond expected lifetime will be retired after a delay
    # of >=1 yr.
    needs_delay = (result <= ref_year) & (df["status"].eq(OPERATING))
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
    start-year imputation still needs it.
    """
    ref_year = _utils.DATASET_YEAR

    corrected_status = df["status"].copy()
    correction_source = pd.Series(
        "unchanged_original_status", index=df.index, dtype="object"
    )

    observed_start_year = df["start_year"]
    observed_end_year = df["end_year"]

    retired_from_observed_end = observed_end_year.notna() & observed_end_year.le(
        ref_year
    )
    corrected_status.loc[retired_from_observed_end] = RETIRED
    correction_source.loc[retired_from_observed_end] = "observed_end_year"

    operating_from_observed_dates = (
        ~retired_from_observed_end
        & observed_start_year.notna()
        & observed_start_year.le(ref_year)
        & observed_end_year.notna()
        & observed_end_year.gt(ref_year)
    )
    corrected_status.loc[operating_from_observed_dates] = OPERATING
    correction_source.loc[operating_from_observed_dates] = "observed_start_and_end_year"

    planned_from_observed_start = (
        ~retired_from_observed_end
        & observed_start_year.notna()
        & observed_start_year.gt(ref_year)
        & corrected_status.isin(PLANNED)
    )

    correction_source.loc[planned_from_observed_start] = (
        "observed_start_year_confirms_planned_status"
    )

    return corrected_status, correction_source


def _impute_status(df: pd.DataFrame) -> pd.Series:
    """Derive final temporal status after start/end years are complete."""
    status = df["status"].copy()
    ref_year = _utils.DATASET_YEAR
    status.loc[ref_year < df["start_year"]] = "planned"
    status.loc[(df["start_year"] <= ref_year) & (ref_year < df["end_year"])] = OPERATING
    status.loc[df["end_year"] <= ref_year] = RETIRED

    if status.isna().any():
        raise ValueError("Entries with ambiguous states were left in the dataframe.")

    return status


CAPACITY_DATE_EVENT_COLUMNS = [
    "powerplant_id",
    "name",
    "country_id",
    "category",
    "technology",
    "status",
    "year",
    "event_type",
    "source_type",
    "source_label",
    "output_capacity_mw",
    "capacity_change_mw",
]


def _build_capacity_date_events(imputed: pd.DataFrame) -> pd.DataFrame:
    """Convert imputed plant dates into annual commissioning and retirement events."""
    if imputed.empty:
        return pd.DataFrame(columns=CAPACITY_DATE_EVENT_COLUMNS)

    common_cols = [
        "powerplant_id",
        "name",
        "country_id",
        "category",
        "technology",
        "status",
        "output_capacity_mw",
    ]

    start_events = imputed[
        common_cols + ["start_year", "start_year_source_type"]
    ].rename(columns={"start_year": "year", "start_year_source_type": "source_type"})
    start_events["event_type"] = "commissioning"
    # Keep the plant capacity unchanged and create a signed event value for plotting:
    # commissioning adds capacity, retirement removes capacity. This signed value is
    # used only in the capacity-date-events diagnostic table.
    start_events["capacity_change_mw"] = start_events["output_capacity_mw"]

    end_events = imputed[common_cols + ["end_year", "end_year_source_type"]].rename(
        columns={"end_year": "year", "end_year_source_type": "source_type"}
    )
    end_events["event_type"] = "retirement"
    # Retirements are represented as negative events so they can be plotted below
    # zero without altering the original plant capacity.
    end_events["capacity_change_mw"] = -end_events["output_capacity_mw"]

    events = pd.concat([start_events, end_events], ignore_index=True)

    events["source_label"] = events["source_type"].map(_utils.date_source_labels())

    return (
        events[CAPACITY_DATE_EVENT_COLUMNS]
        .sort_values(
            ["country_id", "year", "event_type", "source_type", "powerplant_id"]
        )
        .reset_index(drop=True)
    )


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
            country_sets.extend(dataframe["country_id"].unique())

    countries = sorted(set(country_sets))

    if not countries:
        _plots.plot_empty(suptitle, output_path)
        return

    present_source_types = events_df["source_type"].unique().tolist()

    unknown_source_types = set(present_source_types) - set(_utils.DATE_SOURCE_METADATA)

    if unknown_source_types:
        raise ValueError(
            f"Missing DATE_SOURCE_METADATA entries for: {sorted(unknown_source_types)}"
        )

    source_types = [
        source_type
        for source_type in _utils.DATE_SOURCE_METADATA
        if source_type in present_source_types
    ]

    source_colors = _plots.get_time_imputation_colours()

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
        country_events = events_df.loc[events_df["country_id"].eq(country)].copy()

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
            planned_profile_df.loc[planned_profile_df["country_id"].eq(country)].copy()
            if not planned_profile_df.empty
            else pd.DataFrame()
        )

        year_values = set(country_events["year"].astype(int))

        if not commissioning_target.empty:
            year_values.update(commissioning_target["year"].astype(int))

        if not retirement_target.empty:
            year_values.update(retirement_target["year"].astype(int))

        if not planned_target.empty:
            year_values.update(planned_target["year"].astype(int))

        years = np.array(sorted(year_values))

        if len(years) == 0:
            _plots.draw_empty(ax, country, f"No date events for {country}")
            continue

        country_events["year"] = country_events["year"].astype(int)

        annual_events = country_events.groupby(
            ["year", "event_type", "source_type"], as_index=False
        )["capacity_change_mw"].sum()

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
            commissioning_target = commissioning_target.sort_values("year")

            ax.plot(
                commissioning_target["year"],
                commissioning_target["target_final_mw"],
                color="0.15",
                linewidth=2,
            )

        if not retirement_target.empty:
            retirement_target = retirement_target.sort_values("year")

            ax.plot(
                retirement_target["year"],
                -retirement_target["target_final_mw"],
                color="0.15",
                linewidth=2,
                linestyle=":",
            )

        if not planned_target.empty:
            planned_target = (
                planned_target.groupby("year", as_index=False)["target_imputed_mw"]
                .sum()
                .sort_values("year")
            )

            ax.plot(
                planned_target["year"],
                planned_target["target_imputed_mw"],
                color="0.35",
                linewidth=2,
                linestyle="--",
            )

        ax.axhline(0, color="0.35", linewidth=0.8)
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
            label=_utils.DATE_SOURCE_METADATA[source_type]["label"],
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
                label="Commissioning profile for historic assets",
            )
        )

    if not retirement_profile_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.15",
                linewidth=2,
                linestyle=":",
                label="Retirement profile for historic assets",
            )
        )

    if not planned_profile_df.empty:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color="0.35",
                linewidth=2,
                linestyle="--",
                label="Commissioning profile for planned assets",
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
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Add automatic and user imputations to fill missing data.

    Args:
        relocated_gdf: Relocated powerplants with country identifiers.
        reference_capacity_df: Annual category-level capacity stock data.
        imputation: Imputation configuration.
        technology_mapping: Technology-mapping configuration.
    """
    age_profile_diagnostics = pd.DataFrame()
    retirement_profile_diagnostics = pd.DataFrame()
    planned_profile_diagnostics = pd.DataFrame()

    _utils.check_single_category(relocated_gdf)

    lifetimes = imputation["lifetime_years"]
    planned_commissioning_year_windows = imputation[
        "planned_commissioning_year_windows"
    ]
    retirement_delay_years = imputation["retirement_delay_years"]
    scenario = SCENARIO_MAP[imputation["scenario"]]
    start_year_imputation_method = imputation["start_year_imputation_method"]

    status_after_observed_date_correction, _ = _reconcile_status_from_observed_dates(
        relocated_gdf
    )

    # Get facilities within the requested scenario after correcting only
    # the statuses that are already contradicted by observed dates.
    scenario_gdf = relocated_gdf.copy()
    scenario_gdf["status"] = status_after_observed_date_correction

    imputed = scenario_gdf[scenario_gdf["status"].isin(scenario)].copy()

    if not imputed.empty:
        (
            imputed["start_year"],
            start_year_source_type,
            age_profile_diagnostics,
            planned_profile_diagnostics,
        ) = _impute_start_year(
            prepared_df=imputed,
            reference_capacity_df=reference_capacity_df,
            lifetimes=lifetimes,
            planned_commissioning_year_windows=(planned_commissioning_year_windows),
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

        # Drop projects with insufficient date data.
        imputed = imputed.loc[has_complete_dates].copy()

        # Update the powerplant status.
        imputed["status"] = status_final.loc[imputed.index]

        # Pass date-source provenance forward.
        imputed["start_year_source_type"] = start_year_source_type.loc[imputed.index]
        imputed["end_year_source_type"] = end_year_source_type.loc[imputed.index]
    else:
        imputed["start_year_source_type"] = pd.Series(dtype="object")
        imputed["end_year_source_type"] = pd.Series(dtype="object")

    schema = _schemas.build_schema(technology_mapping, "impute")
    return (
        schema.validate(imputed),
        age_profile_diagnostics,
        retirement_profile_diagnostics,
        planned_profile_diagnostics,
    )


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
    relocated_gdf = gpd.read_parquet(snakemake.input.relocated)
    reference_capacity_df = pd.read_parquet(snakemake.input.category_capacity)
    if relocated_gdf.empty:
        imputed_gdf = relocated_gdf
        imputed_gdf["start_year_source_type"] = pd.Series(dtype="object")
        imputed_gdf["end_year_source_type"] = pd.Series(dtype="object")
        commissioning_profile_df = pd.DataFrame()
        retirement_profile_df = pd.DataFrame()
        planned_profile_df = pd.DataFrame()
    else:
        (
            imputed_gdf,
            commissioning_profile_df,
            retirement_profile_df,
            planned_profile_df,
        ) = impute(
            relocated_gdf=relocated_gdf,
            reference_capacity_df=reference_capacity_df,
            imputation=snakemake.params.imputation,
            technology_mapping=snakemake.params.tech_map,
        )
    imputed_gdf.to_parquet(snakemake.output.aged)

    capacity_date_events_df = _build_capacity_date_events(imputed_gdf)

    capacity_date_events_df.to_parquet(
        snakemake.output.capacity_date_events, index=False
    )

    plot_powerplant_capacity_buildup(
        imputed_gdf,
        snakemake.output.histogram,
        "seaborn:tab20",
        snakemake.wildcards.category,
    )
    explore(imputed_gdf, snakemake.output.explorer)

    plot_capacity_date_events(
        events_df=capacity_date_events_df,
        commissioning_profile_df=commissioning_profile_df,
        retirement_profile_df=retirement_profile_df,
        planned_profile_df=planned_profile_df,
        output_path=snakemake.output.capacity_date_plot,
        cat=snakemake.wildcards.category,
    )


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
