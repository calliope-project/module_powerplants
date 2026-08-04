"""Impute powerplant dates and produce time-imputation diagnostics."""

import heapq
import math
import sys
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

import _plots
import _schemas
import _utils
import geopandas as gpd
import numpy as np
import pandas as pd
from _schemas import HISTORICAL, OPERATING, PLANNED, RETIRED, SCENARIO_MAP
from _utils import DATASET_YEAR, EIA_CAT_MAPPING
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.typing import ColorType

if TYPE_CHECKING:
    snakemake: Any


PROFILE_COLUMNS = [
    "country_id",
    "category",
    "reference_category",
    "event_type",
    "cohort",
    "technology",
    "status",
    "profile_source",
    "year",
    "target_mw",
    "observed_mw",
    "imputed_mw",
]


def empty_profiles() -> pd.DataFrame:
    """Return an empty normalized profile table."""
    return pd.DataFrame(columns=PROFILE_COLUMNS)


def _initialise_year_source(year: pd.Series) -> pd.Series:
    """Label whether year values were originally observed or missing."""
    source_type = pd.Series("observed", index=year.index, dtype="object")
    source_type.loc[year.isna()] = "missing_unresolved"
    return source_type


def _has_no_dates(plants: pd.DataFrame) -> pd.Series:
    """Return powerplants without an observed start or end year."""
    return plants[["start_year", "end_year"]].isna().all(axis=1)


def _reference_capacity_stock(
    reference_capacity_df: pd.DataFrame,
    country_id: str,
    categories: list[str],
    max_year: int,
) -> pd.Series:
    """Return annual reference stock summed across mapped categories.

    One powerplant category may map to multiple reference categories. For
    example, hydropower combines the hydropower and pumped-storage stocks.
    These are summed by year before deriving event profiles.
    """
    return (
        reference_capacity_df.loc[
            reference_capacity_df["country_id"].eq(country_id)
            & reference_capacity_df["category"].isin(categories)
            & reference_capacity_df["year"].le(max_year),
            ["year", "capacity_mw"],
        ]
        .groupby("year")["capacity_mw"]
        .sum(min_count=1)
        .sort_index()
    )


def _first_reported_year(capacity_stock: pd.Series) -> int | None:
    """Return the first reported stock year, if one exists."""
    reported_years = capacity_stock.index[capacity_stock.notna()]
    return None if reported_years.empty else int(reported_years.min())


def _build_reference_profile(
    reference_capacity_df: pd.DataFrame,
    country_id: str,
    categories: list[str],
    years: pd.Index,
    event_type: Literal["commissioning", "retirement"],
    *,
    smoothing_window: int = 1,
) -> pd.DataFrame:
    """Build a commissioning or retirement profile from annual stock changes."""
    capacity_stock = _reference_capacity_stock(
        reference_capacity_df, country_id, categories, years.max()
    )

    # Annual stock changes are only a proxy for gross events.
    # - Positive net changes provide commissioning weight
    # - Negative net changes provide retirement weight.
    # Changes in the opposite direction are clipped out.
    direction = 1 if event_type == "commissioning" else -1
    change = (direction * capacity_stock.diff()).clip(lower=0).reindex(years).fillna(0)
    basis = change.copy()
    if smoothing_window > 1:
        basis = basis.rolling(smoothing_window, center=True, min_periods=1).mean()

    fallback_used = basis.sum() <= 0
    if fallback_used:
        # A uniform 'flat' profile keeps dates imputable when reference history is
        # absent, or contains no changes in the requested direction.
        basis.loc[:] = 1.0

    return pd.DataFrame(
        {
            "reference_stock_mw": capacity_stock.reindex(years),
            "reference_change_mw": change,
            "reference_profile_basis_mw": basis,
            "reference_profile_weight": basis / basis.sum(),
            "profile_fallback_used": fallback_used,
        },
        index=years,
    )


def _build_residual_profile(
    dated_df: pd.DataFrame,
    profile: pd.DataFrame,
    allocatable_years: pd.Index,
    missing_capacity_mw: float,
    year_col: str,
) -> pd.DataFrame:
    """Scale a reference profile and subtract capacity with observed dates."""
    result = profile.copy()
    observed = (
        dated_df.groupby(year_col)["output_capacity_mw"]
        .sum()
        .reindex(result.index, fill_value=0.0)
    )

    # Reference data determine the temporal shape, while the plant dataset
    # determines the total capacity represented by the scaled target.
    total_capacity_mw = observed.sum() + missing_capacity_mw
    result["observed_mw"] = observed
    result["target_final_mw"] = result["reference_profile_weight"] * total_capacity_mw

    # Exclude observed overcapacity from the initial residual target. If no
    # positive residual remains, fall back to the reference or uniform profile.
    residual = (
        (result["target_final_mw"] - observed)
        .clip(lower=0)
        .reindex(allocatable_years, fill_value=0.0)
    )
    if residual.sum() <= 0:
        residual = result["reference_profile_weight"].reindex(
            allocatable_years, fill_value=0.0
        )
    if residual.sum() <= 0:
        residual = pd.Series(1.0, index=allocatable_years)

    # Rescale the feasible residual so every missing capacity block is
    # allocated while preserving the residual profile's relative shape.
    residual = residual / residual.sum() * missing_capacity_mw
    result["residual_target_mw"] = residual.reindex(result.index, fill_value=0.0)
    return result


def _allocate_years_by_target(
    plants: pd.DataFrame,
    target: pd.Series,
    *,
    earliest_year: pd.Series | None = None,
    prefer_later_years: bool = False,
) -> pd.Series:
    """Greedily allocate whole plants using an incremental priority queue.

    Earlier/later mean lower/higher calendar-year values, respectively.
    """
    if target.empty:
        raise ValueError("Cannot allocate dates against an empty target profile.")
    if earliest_year is None:
        earliest_year = pd.Series(target.index.min(), index=plants.index)

    # Allocate the least flexible plants first, then the largest capacity
    # blocks, to reduce poor fits caused by indivisible plants.
    order = pd.DataFrame(
        {
            "earliest_year": earliest_year,
            "capacity": plants["output_capacity_mw"],
            "powerplant_id": plants["powerplant_id"],
            "row_order": np.arange(len(plants)),
        },
        index=plants.index,
    ).sort_values(
        ["earliest_year", "capacity", "powerplant_id", "row_order"],
        ascending=[False, False, True, True],
    )

    remaining = target.copy()
    years_desc = sorted(target.index, reverse=True)
    next_year = 0
    heap: list[tuple[float, float, int, int]] = []
    assigned = pd.Series(np.nan, index=plants.index, dtype=float)

    def push(year: int) -> None:
        """Add a year to the allocation queue with its current priority."""
        year_tie = -year if prefer_later_years else year
        # heapq uses a min-heap algorithm
        # negating values prioritize the largest deficit and then the largest target.
        heapq.heappush(heap, (-remaining.loc[year], -target.loc[year], year_tie, year))

    for plant_index in order.index:
        lower_bound = earliest_year.loc[plant_index]

        # Plants are ordered by decreasing earliest year
        # As next_year increments, so newly feasible plants are added to the queue
        while next_year < len(years_desc) and years_desc[next_year] >= lower_bound:
            push(years_desc[next_year])
            next_year += 1

        if not heap:
            raise ValueError(f"No allocatable year at or after {lower_bound}.")

        # Heap priority is deterministic:
        # largest remaining deficit, then original target, then the configured year tie.
        _, _, _, assigned_year = heapq.heappop(heap)
        assigned.loc[plant_index] = assigned_year
        remaining.loc[assigned_year] -= plants.loc[plant_index, "output_capacity_mw"]
        # Reinsert the year with its reduced remaining capacity.
        push(assigned_year)

    return assigned


def _profile_rows(
    profile: pd.DataFrame,
    plants: pd.DataFrame,
    assigned_years: pd.Series,
    *,
    country_id: str,
    category: str,
    event_type: str,
    cohort: str,
    profile_source: str,
    reference_category: str | None = None,
    technology: str | None = None,
    status: str | None = None,
) -> pd.DataFrame:
    """Return normalized diagnostic rows for an imputed plant cohort."""
    imputed = (
        pd.DataFrame(
            {"year": assigned_years, "output_capacity_mw": plants["output_capacity_mw"]}
        )
        .groupby("year")["output_capacity_mw"]
        .sum()
        .reindex(profile.index, fill_value=0.0)
    )
    return pd.DataFrame(
        {
            "country_id": country_id,
            "category": category,
            "reference_category": reference_category,
            "event_type": event_type,
            "cohort": cohort,
            "technology": technology,
            "status": status,
            "profile_source": profile_source,
            "year": profile.index,
            "target_mw": profile["target_final_mw"],
            "observed_mw": profile["observed_mw"],
            "imputed_mw": imputed,
        }
    )[PROFILE_COLUMNS]


def _remaining_target(
    target: pd.Series, plants: pd.DataFrame, assigned_years: pd.Series
) -> pd.Series:
    """Subtract assigned plant capacity from an annual target."""
    assigned_capacity = (
        plants.assign(year=assigned_years)
        .groupby("year")["output_capacity_mw"]
        .sum()
        .reindex(target.index, fill_value=0.0)
    )
    return target - assigned_capacity


def _allocate_historical_start_years(
    plants: pd.DataFrame, target: pd.Series, earliest_plant_start_years: pd.Series
) -> pd.Series:
    """Allocate retired and operating starts against one commissioning target."""
    assigned = pd.Series(np.nan, index=plants.index, dtype=float)

    # A retired plant needs at least one operating year before its end year.
    retired = plants.loc[plants["status"].eq(RETIRED)]
    if not retired.empty:
        latest_retired_start_year = DATASET_YEAR - 1
        retired_years = _allocate_years_by_target(
            retired, target.loc[target.index <= latest_retired_start_year]
        )
        assigned.loc[retired.index] = retired_years
        target = _remaining_target(target, retired, retired_years)

    # Operating plants must be commissioned late enough to remain active in the
    # dataset year under their configured lifetime assumption.
    if not earliest_plant_start_years.empty:
        operating = plants.loc[earliest_plant_start_years.index]
        assigned.loc[operating.index] = _allocate_years_by_target(
            operating, target, earliest_year=earliest_plant_start_years
        )

    return assigned


def _impute_historical_start_years(
    plants: pd.DataFrame,
    reference_capacity_df: pd.DataFrame,
    lifetimes: Mapping[str, int],
) -> tuple[pd.Series, pd.DataFrame]:
    """Impute historical start years from commissioning profiles."""
    result = pd.Series(np.nan, index=plants.index, dtype=float)
    profiles: list[pd.DataFrame] = []
    missing = _has_no_dates(plants) & plants["status"].isin(HISTORICAL)

    for (country_id, category), undated in plants.loc[missing].groupby(
        ["country_id", "category"]
    ):
        # Harmonise plant categories with the annual reference-capacity categories.
        categories = EIA_CAT_MAPPING[category]
        group = plants["country_id"].eq(country_id) & plants["category"].eq(category)
        dated = plants.loc[
            group & plants["status"].isin(HISTORICAL) & plants["start_year"].notna()
        ]

        operating = undated["status"].eq(OPERATING)
        lifetime = undated["technology"].map(lifetimes)
        earliest_operating_year = DATASET_YEAR - lifetime.loc[operating] + 1
        retired = undated["status"].eq(RETIRED)

        stock = _reference_capacity_stock(
            reference_capacity_df, country_id, categories, DATASET_YEAR
        )
        reference_first = _first_reported_year(stock)
        first_year = min(
            pd.concat(
                [
                    earliest_operating_year,
                    DATASET_YEAR - lifetime.loc[retired],
                    dated["start_year"],
                ]
            ).min(),
            reference_first if reference_first is not None else DATASET_YEAR,
        )
        years = pd.Index(range(int(first_year), DATASET_YEAR + 1), name="start_year")
        last_allocatable_year = DATASET_YEAR if operating.any() else DATASET_YEAR - 1
        allocatable_years = years[years <= last_allocatable_year]

        profile = _build_reference_profile(
            reference_capacity_df, country_id, categories, years, "commissioning"
        )
        allocation = _build_residual_profile(
            dated,
            profile,
            allocatable_years,
            undated["output_capacity_mw"].sum(),
            "start_year",
        )
        assigned = _allocate_historical_start_years(
            undated, allocation["residual_target_mw"], earliest_operating_year
        )
        result.loc[undated.index] = assigned
        source = (
            "uniform"
            if bool(profile["profile_fallback_used"].iloc[0])
            else "positive_capacity_change"
        )
        profiles.append(
            _profile_rows(
                allocation,
                undated,
                assigned,
                country_id=country_id,
                category=category,
                event_type="commissioning",
                cohort="historical",
                profile_source=source,
                reference_category="+".join(categories),
            )
        )

    return result, pd.concat(
        profiles, ignore_index=True
    ) if profiles else empty_profiles()


def _impute_retired_end_years(
    plants: pd.DataFrame,
    retirement_linked: pd.Series,
    reference_capacity_df: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    """Impute retirement years for plants whose two dates were missing."""
    end_result = pd.Series(np.nan, index=plants.index, dtype=float)
    profiles: list[pd.DataFrame] = []

    for (country_id, category), undated in plants.loc[retirement_linked].groupby(
        ["country_id", "category"]
    ):
        categories = EIA_CAT_MAPPING[category]
        group = plants["country_id"].eq(country_id) & plants["category"].eq(category)
        dated = plants.loc[
            group & plants["status"].eq(RETIRED) & plants["end_year"].notna()
        ]
        stock = _reference_capacity_stock(
            reference_capacity_df, country_id, categories, DATASET_YEAR
        )
        reference_first = _first_reported_year(stock)
        observed_first = int(dated["end_year"].min()) if not dated.empty else None
        earliest_end = undated["start_year"] + 1
        first_year = int(
            min(
                year
                for year in (reference_first, observed_first, earliest_end.min())
                if year is not None
            )
        )
        years = pd.Index(range(first_year, DATASET_YEAR + 1), name="end_year")
        profile = _build_reference_profile(
            reference_capacity_df, country_id, categories, years, "retirement"
        )
        # Missing retirements are normally restricted to the period covered by
        # the country's reference series.
        allocatable_years = pd.Index(
            range(reference_first or first_year, DATASET_YEAR + 1), name="end_year"
        )
        profile_source = "negative_capacity_change"

        if bool(profile["profile_fallback_used"].iloc[0]):
            # When stock history contains no reductions, observed retirement
            # timing is more informative than a uniform fallback. It may also
            # legitimately precede the reference-series coverage.
            observed_basis = (
                dated.groupby("end_year")["output_capacity_mw"]
                .sum()
                .reindex(years, fill_value=0.0)
            )
            if observed_basis.sum() > 0:
                profile["reference_profile_weight"] = (
                    observed_basis / observed_basis.sum()
                )
                profile_source = "observed_retirements"
                allocatable_years = years
            else:
                profile_source = "uniform"

        allocation = _build_residual_profile(
            dated,
            profile,
            allocatable_years,
            undated["output_capacity_mw"].sum(),
            "end_year",
        )
        assigned_end = _allocate_years_by_target(
            undated,
            allocation["residual_target_mw"],
            earliest_year=earliest_end,
            prefer_later_years=True,
        )
        end_result.loc[undated.index] = assigned_end
        profiles.append(
            _profile_rows(
                allocation,
                undated,
                assigned_end,
                country_id=country_id,
                category=category,
                event_type="retirement",
                cohort="historical",
                profile_source=profile_source,
                reference_category="+".join(categories),
            )
        )

    combined = pd.concat(profiles, ignore_index=True) if profiles else empty_profiles()
    return end_result, combined


def _impute_planned_from_commissioning_windows(
    plants: pd.DataFrame, windows: Mapping[str, Mapping[str, list[int]]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign start years to fully undated planned plants using flat windows."""
    result = plants.copy()
    profiles: list[pd.DataFrame] = []
    missing = result["status"].isin(PLANNED) & _has_no_dates(result)

    for (country_id, category, technology, status), undated in result.loc[
        missing
    ].groupby(["country_id", "category", "technology", "status"]):
        # Planned projects follow technology-specific future windows
        lower, upper = windows[technology][status]
        years = pd.Index(
            range(DATASET_YEAR + lower, DATASET_YEAR + upper + 1), name="year"
        )
        target = pd.Series(
            undated["output_capacity_mw"].sum() / len(years), index=years
        )
        assigned = _allocate_years_by_target(undated, target)
        result.loc[undated.index, "start_year"] = assigned
        result.loc[undated.index, "start_year_source_type"] = (
            f"imputed_{status.replace('-', '_')}_window"
        )

        profile = pd.DataFrame(
            {"target_final_mw": target, "observed_mw": 0.0}, index=years
        )
        profiles.append(
            _profile_rows(
                profile,
                undated,
                assigned,
                country_id=country_id,
                category=category,
                event_type="commissioning",
                cohort="planned",
                profile_source="uniform_window",
                technology=technology,
                status=status,
            )
        )

    combined = pd.concat(profiles, ignore_index=True) if profiles else empty_profiles()
    return result, combined


def _impute_years_from_lifetimes(
    plants: pd.DataFrame, lifetimes: Mapping[str, int], delays: Mapping[str, int]
) -> pd.DataFrame:
    """Complete missing start/end year counterparts using configured lifetimes."""
    result = plants.copy()
    lifetime = result["technology"].map(lifetimes)

    expected_start = result["end_year"] - lifetime
    start_mask = result["start_year"].isna() & expected_start.notna()
    result.loc[start_mask, "start_year"] = expected_start.loc[start_mask]
    result.loc[start_mask, "start_year_source_type"] = "derived_from_end_year"

    expected_end = result["start_year"] + lifetime
    end_mask = result["end_year"].isna() & expected_end.notna()
    result.loc[end_mask, "end_year"] = expected_end.loc[end_mask]
    result.loc[end_mask, "end_year_source_type"] = "derived_from_start_year_lifetime"

    # Only derived dates may be adjusted: known start/end years remain sacred.
    # A plant marked retired must be offline by the start of the dataset year.
    retired_cap = (
        end_mask & result["status"].eq(RETIRED) & result["end_year"].gt(DATASET_YEAR)
    )
    result.loc[retired_cap, "end_year"] = DATASET_YEAR
    result.loc[retired_cap, "end_year_source_type"] = (
        "derived_from_start_year_lifetime_capped_to_retired_status"
    )

    # An operating plant that outlived its assumed lifetime receives the
    # configured extension, with at least one year beyond the dataset year.
    delayed = (
        end_mask & result["status"].eq(OPERATING) & result["end_year"].le(DATASET_YEAR)
    )
    result.loc[delayed, "end_year"] = (
        result.loc[delayed, "end_year"] + result.loc[delayed, "technology"].map(delays)
    ).clip(lower=DATASET_YEAR + 1)
    result.loc[delayed, "end_year_source_type"] = (
        "derived_from_start_year_lifetime_with_retirement_delay"
    )
    return result


def _impute_capacity_profile_dates(
    plants: pd.DataFrame,
    reference_capacity_df: pd.DataFrame,
    lifetimes: Mapping[str, int],
    delays: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Complete fully undated historical plants using capacity profiles."""
    result = plants.copy()
    undated = _has_no_dates(result) & result["status"].isin(HISTORICAL)
    retired = undated & result["status"].eq(RETIRED)

    start_year, commissioning_profiles = _impute_historical_start_years(
        result, reference_capacity_df, lifetimes
    )
    start_imputed = undated & start_year.notna()
    result.loc[start_imputed, "start_year"] = start_year.loc[start_imputed]
    result.loc[start_imputed, "start_year_source_type"] = "imputed_capacity_profile"
    result.loc[retired & start_year.notna(), "start_year_source_type"] = (
        "imputed_capacity_profile_retirement_linked"
    )

    # Profile-imputed commissioning years set the earliest feasible retirement year.
    end_year, retirement_profiles = _impute_retired_end_years(
        result, retired, reference_capacity_df
    )
    end_imputed = retired & end_year.notna()
    result.loc[end_imputed, "end_year"] = end_year.loc[end_imputed]
    result.loc[end_imputed, "end_year_source_type"] = (
        "imputed_retirement_capacity_profile"
    )

    # Complete counterparts which the capacity profiles left unresolved with lifetimes.
    completed = _impute_years_from_lifetimes(result.loc[undated], lifetimes, delays)
    date_columns = [
        "start_year",
        "end_year",
        "start_year_source_type",
        "end_year_source_type",
    ]
    result.loc[undated, date_columns] = completed[date_columns]

    profile_parts = [
        profile
        for profile in (commissioning_profiles, retirement_profiles)
        if not profile.empty
    ]
    profiles = (
        pd.concat(profile_parts, ignore_index=True).reindex(columns=PROFILE_COLUMNS)
        if profile_parts
        else empty_profiles()
    )
    return result, profiles


def _adjust_scenario_status_to_reference_year(plants: pd.DataFrame) -> pd.Series:
    """Adjust powerplant status to the given year without altering observed dates."""
    status = plants["status"].copy()

    # Observed retirement years are the strongest status signal.
    retired = plants["end_year"].notna() & plants["end_year"].le(DATASET_YEAR)
    status.loc[retired] = RETIRED

    # Ensure powerplants active in the given year are marked as operating
    operating = (
        ~retired
        & plants["start_year"].notna()
        & plants["start_year"].le(DATASET_YEAR)
        & plants["end_year"].notna()
        & plants["end_year"].gt(DATASET_YEAR)
    )
    status.loc[operating] = OPERATING

    # Find powerplants known to be operating after the provided year.
    # Treat them as in construction for scenario eligibility.
    future_historical = (
        ~retired
        & plants["start_year"].notna()
        & plants["start_year"].gt(DATASET_YEAR)
        & status.isin(HISTORICAL)
    )
    status.loc[future_historical] = "construction"
    return status


def _simplify_status_for_users(plants: pd.DataFrame) -> pd.Series:
    """Derive final temporal status from completed start and end years."""
    status = plants["status"].copy()
    status.loc[plants["start_year"].gt(DATASET_YEAR)] = "planned"
    status.loc[
        plants["start_year"].le(DATASET_YEAR) & plants["end_year"].gt(DATASET_YEAR)
    ] = OPERATING
    status.loc[plants["end_year"].le(DATASET_YEAR)] = RETIRED
    return status


def impute_time(
    plants: pd.DataFrame, reference_capacity_df: pd.DataFrame, imputation: Mapping
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing dates and return plants plus normalized profile diagnostics."""
    if plants.empty:
        return _schemas.PlantSchema.empty(), empty_profiles()

    _utils.check_single_category(plants)

    scenario_plants = plants.assign(
        status=_adjust_scenario_status_to_reference_year(plants),
        start_year_source_type=_initialise_year_source(plants["start_year"]),
        end_year_source_type=_initialise_year_source(plants["end_year"]),
    )
    result = scenario_plants.loc[
        scenario_plants["status"].isin(SCENARIO_MAP[imputation["scenario"]])
    ].copy()

    if result.empty:
        # Scenario removed all available powerplants
        return _schemas.PlantSchema.empty(), empty_profiles()

    # Observed dates should be immutable!
    # Start by filling in start/end years that have a known counterpart.
    lifetimes = imputation["lifetime_years"]
    delays = imputation["retirement_delay_years"]
    result = _impute_years_from_lifetimes(result, lifetimes, delays)

    # Fill unknown planned plants using flat commissioning windows.
    # Then fill end years with the lifetime.
    result, planned_profiles = _impute_planned_from_commissioning_windows(
        plants=result, windows=imputation["planned_commissioning_year_windows"]
    )
    result = _impute_years_from_lifetimes(result, lifetimes, delays)

    # Remaining historical powerplants are filled with user-specified methods.
    match imputation["method"]:
        case "capacity_profile":
            result, method_profiles = _impute_capacity_profile_dates(
                result, reference_capacity_df, lifetimes, delays
            )
        case method:
            raise ValueError(f"Unknown time imputation method: {method}")

    result["status"] = _simplify_status_for_users(result)
    profile_parts = [i for i in (planned_profiles, method_profiles) if not i.empty]
    profiles = (
        pd.concat(profile_parts, ignore_index=True).reindex(columns=PROFILE_COLUMNS)
        if profile_parts
        else empty_profiles()
    )
    return result, profiles


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
    common = [
        "powerplant_id",
        "name",
        "country_id",
        "category",
        "technology",
        "status",
        "output_capacity_mw",
    ]
    events = []
    for year_col, source_col, event_type, sign in (
        ("start_year", "start_year_source_type", "commissioning", 1),
        ("end_year", "end_year_source_type", "retirement", -1),
    ):
        event = imputed[common + [year_col, source_col]].rename(
            columns={year_col: "year", source_col: "source_type"}
        )
        event["event_type"] = event_type

        # Keep plant capacity unchanged and add a signed event value solely for
        # diagnostics: commissioning is positive and retirement is negative.
        event["capacity_change_mw"] = sign * event["output_capacity_mw"]
        events.append(event)

    result = pd.concat(events, ignore_index=True)
    result["source_label"] = result["source_type"].map(_utils.date_source_labels())
    return (
        result[CAPACITY_DATE_EVENT_COLUMNS]
        .sort_values(
            ["country_id", "year", "event_type", "source_type", "powerplant_id"]
        )
        .reset_index(drop=True)
    )


def _time_imputation_colours() -> dict[str, ColorType]:
    """Return colours for time imputation source types."""
    start_sources = _utils.date_source_types_for("start_year")
    end_sources = _utils.date_source_types_for("end_year")
    ordered = _utils.DATE_SOURCE_METADATA
    shared = [source for source in ordered if source in start_sources & end_sources]
    start_only = [source for source in ordered if source in start_sources - end_sources]
    end_only = [source for source in ordered if source in end_sources - start_sources]
    return (
        _plots.get_colour_dict(shared, "colorbrewer:Greys", value_range=(0.4, 0.45))
        | _plots.get_colour_dict(
            start_only, "colorbrewer:Purples", value_range=(0.2, 0.9)
        )
        | _plots.get_colour_dict(end_only, "colorbrewer:Reds", value_range=(0.2, 0.9))
    )


def _profile_target(
    profiles: pd.DataFrame, country: str, event_type: str, cohort: str
) -> pd.DataFrame:
    """Aggregate normalized diagnostic targets for plotting."""
    if profiles.empty:
        return pd.DataFrame(columns=["year", "target_mw"])
    return (
        profiles.loc[
            profiles["country_id"].eq(country)
            & profiles["event_type"].eq(event_type)
            & profiles["cohort"].eq(cohort)
        ]
        .groupby("year", as_index=False)["target_mw"]
        .sum()
        .sort_values("year")
    )


def plot_capacity_date_events(
    events: pd.DataFrame, profiles: pd.DataFrame, output_path: str, category: str
) -> None:
    """Plot commissioning and retirement events by date provenance."""
    title = f"Capacity-date imputation for {category.replace('_', ' ')}"
    countries = sorted(
        set(events["country_id"] if not events.empty else [])
        | set(profiles["country_id"] if not profiles.empty else [])
    )
    if not countries:
        _plots.plot_empty(title, output_path)
        return

    present_sources = set(events["source_type"])
    source_types = [
        source for source in _utils.DATE_SOURCE_METADATA if source in present_sources
    ]
    colours = _time_imputation_colours()
    cols = 2 if len(countries) > 1 else 1
    rows = math.ceil(len(countries) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 7, rows * 4.5), constrained_layout=True
    )
    axes_flat = np.array(axes).ravel()

    profile_kinds = (
        ("commissioning", "historical", 1, "0.15", "-"),
        ("retirement", "historical", -1, "0.15", ":"),
        ("commissioning", "planned", 1, "0.35", "--"),
    )
    any_profile = {kind[:2]: False for kind in profile_kinds}

    for ax, country in zip(axes_flat, countries, strict=False):
        country_events = events.loc[events["country_id"].eq(country)]
        targets = {
            (event_type, cohort): _profile_target(profiles, country, event_type, cohort)
            for event_type, cohort, *_ in profile_kinds
        }
        year_values = set(country_events["year"].astype(int))
        for target in targets.values():
            year_values.update(target["year"].astype(int))
        years = np.array(sorted(year_values))
        if not len(years):
            _plots.draw_empty(ax, country, f"No date events for {country}")
            continue

        annual = (
            country_events.assign(year=country_events["year"].astype(int))
            .groupby(["year", "event_type", "source_type"])["capacity_change_mw"]
            .sum()
        )
        positive_bottom = np.zeros(len(years))
        negative_bottom = np.zeros(len(years))
        for source in source_types:
            values = {}
            for event_type in ("commissioning", "retirement"):
                try:
                    series = annual.xs((event_type, source), level=(1, 2))
                except KeyError:
                    series = pd.Series(dtype=float)
                values[event_type] = series.reindex(years, fill_value=0).to_numpy()

            if values["commissioning"].any():
                ax.bar(
                    years,
                    values["commissioning"],
                    bottom=positive_bottom,
                    color=colours[source],
                    width=0.9,
                )
            if values["retirement"].any():
                ax.bar(
                    years,
                    values["retirement"],
                    bottom=negative_bottom,
                    color=colours[source],
                    width=0.9,
                )
            positive_bottom += values["commissioning"]
            negative_bottom += values["retirement"]

        for event_type, cohort, sign, colour, linestyle in profile_kinds:
            target = targets[event_type, cohort]
            if not target.empty:
                any_profile[event_type, cohort] = True
                ax.plot(
                    target["year"],
                    sign * target["target_mw"],
                    color=colour,
                    linewidth=2,
                    linestyle=linestyle,
                )

        ax.axhline(0, color="0.35", linewidth=0.8)
        ax.set(title=country, xlabel="Year", ylabel="Annual capacity event (MW)")
        ax.locator_params(axis="x", nbins=12)
        ax.tick_params(axis="x", rotation=45)
        ax.minorticks_off()

    for ax in axes_flat[len(countries) :]:
        ax.set_visible(False)

    handles: list[Patch | Line2D] = [
        Patch(
            facecolor=colours[source],
            label=_utils.DATE_SOURCE_METADATA[source]["label"],
        )
        for source in source_types
    ]
    profile_labels = {
        ("commissioning", "historical"): "Commissioning profile for historic assets",
        ("retirement", "historical"): "Retirement profile for historic assets",
        ("commissioning", "planned"): "Commissioning profile for planned assets",
    }
    for event_type, cohort, _, colour, linestyle in profile_kinds:
        if any_profile[event_type, cohort]:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=colour,
                    linewidth=2,
                    linestyle=linestyle,
                    label=profile_labels[event_type, cohort],
                )
            )
    fig.legend(
        handles=handles, loc="center left", bbox_to_anchor=(1, 0.5), frameon=False
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _capacity_profile_diagnostics(
    imputed: pd.DataFrame, profiles: pd.DataFrame, output_path: str, category: str
) -> pd.DataFrame:
    """Build capacity-profile diagnostics and render their plot."""
    events = _schemas.CapacityDateEventSchema.validate(
        _build_capacity_date_events(imputed)
    )
    plot_capacity_date_events(events, profiles, output_path, category)
    return events


def explore(
    imputed: gpd.GeoDataFrame, output_path: str, colormap: str = "tab20"
) -> None:
    """Create an HTML map for users to explore."""
    if imputed.empty:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write("No data")
    else:
        imputed.explore(
            column="technology", legend=True, popup=True, cmap=colormap
        ).save(output_path)


def plot_powerplant_capacity_buildup(
    plants: pd.DataFrame, output_path: str, colormap: str, category: str = "powerplant"
) -> None:
    """Plot active powerplant capacity over time per country and technology."""
    title = f"Active {category} capacity by technology per country"
    if plants.empty:
        _plots.plot_empty(title, output_path)
        return

    years = pd.Index(
        range(int(plants["start_year"].min()), int(plants["end_year"].max()) + 1)
    )
    countries = sorted(plants["country_id"].unique())
    technologies = sorted(plants["technology"].unique())
    colours = _plots.get_colour_dict(technologies, colormap)
    cols = 2 if len(countries) > 1 else 1
    rows = math.ceil(len(countries) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 5, rows * 4), constrained_layout=True
    )
    axes_flat = np.array(axes).ravel()

    for ax, country in zip(axes_flat, countries, strict=False):
        country_plants = plants.loc[plants["country_id"].eq(country)]

        # Represent each lifecycle as a positive commissioning event and a
        # negative retirement event. Cumulative annual changes then give active
        # capacity without rescanning every plant for every plotted year.
        starts = country_plants.rename(columns={"start_year": "year"}).assign(
            capacity_change_mw=country_plants["output_capacity_mw"]
        )
        ends = country_plants.rename(columns={"end_year": "year"}).assign(
            capacity_change_mw=-country_plants["output_capacity_mw"]
        )
        changes = (
            pd.concat([starts, ends])
            .groupby(["year", "technology"])["capacity_change_mw"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=years, columns=technologies, fill_value=0)
        )
        changes.cumsum().plot(
            kind="bar",
            stacked=True,
            ax=ax,
            color=[colours[technology] for technology in technologies],
            legend=False,
            rot=45,
        )
        ax.set_title(country)
        ax.set_ylabel("Capacity (MW)")
        ax.locator_params(axis="x", nbins=10)
        ax.minorticks_off()

    for ax in axes_flat[len(countries) :]:
        ax.set_visible(False)
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles[::-1],
        labels[::-1],
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        title="Technology",
        frameon=False,
    )
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Main snakemake process."""
    imputed, profiles = impute_time(
        plants=gpd.read_parquet(snakemake.input.relocated),
        reference_capacity_df=pd.read_parquet(snakemake.input.category_capacity),
        imputation=snakemake.params.imputation,
    )
    schema = _schemas.build_schema(snakemake.params.tech_map, "impute")
    imputed = schema.validate(imputed)
    imputed.to_parquet(snakemake.output.aged)

    plot_powerplant_capacity_buildup(
        imputed,
        snakemake.output.histogram,
        "seaborn:tab20",
        snakemake.wildcards.category,
    )
    explore(imputed, snakemake.output.explorer)

    match snakemake.params.imputation["method"]:
        case "capacity_profile":
            diagnostics = _capacity_profile_diagnostics(
                imputed,
                profiles,
                snakemake.output.capacity_date_plot,
                snakemake.wildcards.category,
            )
        case method:
            raise ValueError(f"Unknown time imputation method: {method}")
    diagnostics.to_parquet(snakemake.output.capacity_date_events, index=False)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
