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

if TYPE_CHECKING:
    snakemake: Any

HISTORICAL = {"operating", "retired"}
CURRENT = {"operating"}
SCENARIO_MAP = {
    "historical": HISTORICAL,
    "construction": HISTORICAL | {"construction"},
    "pre_construction": HISTORICAL | {"construction", "pre-construction"},
    "announced": HISTORICAL | {"construction", "pre-construction", "announced"},
}
#this map accounts for the difference in naming between the EIA dataset and the parquet naming for fossil fuels.
STATISTICS_CATEGORY_MAP = {
    "fossil": "fossil fuels",
}

def _initial_year_source_type(year: pd.Series) -> pd.Series:
    """Label whether year values were originally present or missing."""
    source_type = pd.Series("observed", index=year.index, dtype="object")
    source_type.loc[year.isna()] = "missing_unresolved"
    return source_type

def _build_addition_weights(
    category_capacity_df: pd.DataFrame,
    country_id: str,
    category: str,
    years: pd.Index,
    smoothing_window: int = 1,
) -> pd.Series:
    """Build normalised annual addition weights from capacity stocks."""
    capacity_stock = (
        category_capacity_df.loc[
            (category_capacity_df["country_id"] == country_id)
            & (category_capacity_df["category"] == category)
            & (category_capacity_df["year"] <= _utils.DATASET_YEAR),
            ["year", "capacity_mw"],
        ]
        .sort_values("year")
        .set_index("year")["capacity_mw"]
    )

    additions = capacity_stock.diff().clip(lower=0)
    additions = additions.reindex(years).fillna(0.0)

    if smoothing_window > 1:
        additions = additions.rolling(
            window=smoothing_window,
            center=True,
            min_periods=1,
        ).mean()

    if additions.sum() <= 0:
        additions.loc[:] = 1.0

    return additions / additions.sum()

def _build_clipped_residual_target(
    dated_df: pd.DataFrame,
    year_weights: pd.Series,
    allocatable_years: pd.Index,
    missing_capacity_mw: float,
) -> pd.Series:
    """Build a feasible target for the missing powerplant capacity."""
    observed_capacity = (
        dated_df.groupby("start_year")["output_capacity_mw"]
        .sum()
        .reindex(year_weights.index, fill_value=0.0)
    )

    total_capacity_mw = observed_capacity.sum() + missing_capacity_mw
    target_final_capacity = year_weights * total_capacity_mw

    residual_target = (
        target_final_capacity - observed_capacity
    ).clip(lower=0.0)

    residual_target = residual_target.reindex(
        allocatable_years,
        fill_value=0.0,
    )

    if residual_target.sum() <= 0:
        residual_target = year_weights.reindex(
            allocatable_years,
            fill_value=0.0,
        )

    if residual_target.sum() <= 0:
        residual_target = pd.Series(
            1.0,
            index=allocatable_years,
            dtype=float,
        )

    return residual_target / residual_target.sum() * missing_capacity_mw

def _allocate_start_years_by_residual_target(
    undated_df: pd.DataFrame,
    residual_target: pd.Series,
    lifetimes: dict[str, int],
) -> pd.Series:
    """Assign whole plants to years that have the largest deficits.

    Plants with the narrowest feasible commissioning windows are processed
    first. Within equal feasible windows, larger plants are processed first
    because they are harder to fit into the residual target.
    """
    capacities = undated_df["output_capacity_mw"].astype(float)

    earliest_feasible_year = (
        _utils.DATASET_YEAR
        - undated_df["technology"].map(lifetimes)
    )

    order = pd.DataFrame(
        {
            "earliest_feasible_year": earliest_feasible_year,
            "capacity": capacities,
            "powerplant_id": undated_df["powerplant_id"],
            "row_order": np.arange(len(undated_df)),
        },
        index=undated_df.index,
    ).sort_values(
        [
            "earliest_feasible_year",
            "capacity",
            "powerplant_id",
            "row_order",
        ],
        ascending=[False, False, True, True],
    )

    remaining_target = residual_target.copy()
    assigned_years = pd.Series(
        np.nan,
        index=undated_df.index,
        dtype=float,
    )

    for plant_index in order.index:
        plant = undated_df.loc[plant_index]
        plant_capacity = capacities.loc[plant_index]
        lifetime_years = lifetimes[plant["technology"]]

        earliest_year = _utils.DATASET_YEAR - lifetime_years
        feasible_target = remaining_target.loc[
            (remaining_target.index >= earliest_year)
            & (remaining_target.index <= _utils.DATASET_YEAR)
        ]

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

def _impute_start_years_by_capacity_profile(
    prepared_df: pd.DataFrame,
    category_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
    smoothing_window: int = 1,
) -> pd.Series:
    """Impute operating plant dates against a target profile."""
    start_year = prepared_df["start_year"].copy()
    lifetime = prepared_df["technology"].map(lifetimes)

    

    result = pd.Series(
        np.nan,
        index=prepared_df.index,
        dtype=float,
    )

    missing_mask = (
        start_year.isna()
        & lifetime.notna()
        & prepared_df["status"].isin(CURRENT)
    )

    if not missing_mask.any():
        return result

    grouped_missing = prepared_df.loc[missing_mask].groupby(
        ["country_id", "category"],
        dropna=False,
    )

    for (country_id, category), undated_group in grouped_missing:

        statistics_category = STATISTICS_CATEGORY_MAP.get(category, category)
        
        group_mask = (
            prepared_df["country_id"].eq(country_id)
            & prepared_df["category"].eq(category)
            & prepared_df["status"].isin(HISTORICAL)
        )

        dated_group = prepared_df.loc[
            group_mask & start_year.notna()
        ].copy()
        dated_group["start_year"] = start_year.loc[dated_group.index]

        earliest_allocatable_year = int(
            (
                _utils.DATASET_YEAR
                - lifetime.loc[undated_group.index]
            ).min()
        )

        allocatable_years = pd.Index(
            range(
                earliest_allocatable_year,
                _utils.DATASET_YEAR + 1,
            ),
            name="start_year",
        )

        profile_first_year = category_capacity_df.loc[
            (category_capacity_df["country_id"] == country_id)
            & (category_capacity_df["category"] == statistics_category),
            "year",
        ].min()

        observed_first_year = (
            dated_group["start_year"].min()
            if not dated_group.empty
            else _utils.DATASET_YEAR
        )

        first_profile_year = int(
            min(
                profile_first_year,
                observed_first_year,
                earliest_allocatable_year,
            )
        )

        profile_years = pd.Index(
            range(
                first_profile_year,
                _utils.DATASET_YEAR + 1,
            ),
            name="start_year",
        )

        year_weights = _build_addition_weights(
            category_capacity_df=category_capacity_df,
            country_id=country_id,
            category=statistics_category,
            years=profile_years,
            smoothing_window=smoothing_window,
        )

        missing_capacity_mw = undated_group[
            "output_capacity_mw"
        ].sum()

        residual_target = _build_clipped_residual_target(
            dated_df=dated_group,
            year_weights=year_weights,
            allocatable_years=allocatable_years,
            missing_capacity_mw=missing_capacity_mw,
        )

        result.loc[undated_group.index] = (
            _allocate_start_years_by_residual_target(
                undated_df=undated_group,
                residual_target=residual_target,
                lifetimes=lifetimes,
            )
        )

    return result

def _impute_start_years_by_group_average(
    prepared_df: pd.DataFrame,
    group_cols: list[str] | None = None,
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
    category_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
    method: str = "group_average",
) -> pd.Series:
    """Impute start years that remain missing after direct backfilling."""
    if method == "group_average":
        return _impute_start_years_by_group_average(prepared_df)

    if method == "capacity_profile":
        return _impute_start_years_by_capacity_profile(
            prepared_df,
            category_capacity_df=category_capacity_df,
            lifetimes=lifetimes,
        )

    raise ValueError(
        "Unknown start-year imputation method "
        f"{method!r}. Expected 'group_average' or 'capacity_profile'."
    )
def _impute_start_year(
    prepared_df: pd.DataFrame,
    category_capacity_df: pd.DataFrame,
    lifetimes: dict[str, int],
    method: str = "group_average",
) -> tuple[pd.Series, pd.Series]:
    """Impute missing powerplant start years and track source labels."""
    start_year = prepared_df["start_year"].copy()
    start_year_source_type = _initial_year_source_type(start_year)

    lifetime = prepared_df["technology"].map(lifetimes)

    # First, preserve the direct deterministic backfill from known end year.
    direct_backfill_mask = (
        start_year.isna()
        & prepared_df["end_year"].notna()
        & lifetime.notna()
    )
    start_year.loc[direct_backfill_mask] = (
        prepared_df.loc[direct_backfill_mask, "end_year"]
        - lifetime.loc[direct_backfill_mask]
    )
    start_year_source_type.loc[direct_backfill_mask] = "derived_from_end_year"

    # Then impute only the start years that remain missing.
    remaining_missing_mask = start_year.isna()

    if remaining_missing_mask.any():
        imputation_df = prepared_df.copy()
        imputation_df["start_year"] = start_year

        imputed_start_year = _impute_remaining_start_years(
            imputation_df,
            category_capacity_df=category_capacity_df,
            lifetimes=lifetimes,
            method=method,
        )

        fallback_imputed_mask = (
            remaining_missing_mask
            & imputed_start_year.notna()
        )

        start_year.loc[fallback_imputed_mask] = imputed_start_year.loc[
            fallback_imputed_mask
        ]
        start_year_source_type.loc[fallback_imputed_mask] = f"imputed_{method}"

    return start_year, start_year_source_type

def _impute_end_year(
    df: pd.DataFrame,
    lifetimes: dict[str, int],
    delay: dict[str, int],
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
    end_year_source_type.loc[lifetime_fill_mask] = (
        "derived_from_start_year_lifetime"
    )

    # Plants operating beyond expected lifetime will be retired after a delay
    # of >=1 yr.
    needs_delay = (result <= ref_year) & (df["status"] == "operating")
    delayed_end = result + df["technology"].map(delay).fillna(0).astype(int)
    delayed_end = delayed_end.clip(lower=ref_year + 1)

    result.loc[needs_delay] = delayed_end.loc[needs_delay]

    end_year_source_type.loc[
        needs_delay & lifetime_fill_mask
    ] = "derived_from_start_year_lifetime_with_retirement_delay"

    end_year_source_type.loc[
        needs_delay & ~lifetime_fill_mask
    ] = "observed_adjusted_with_retirement_delay"

    return result, end_year_source_type

def _impute_status(df: pd.DataFrame) -> pd.Series:
    """Impute powerplant status.

    Must be called after start/end years are complete.
    """
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
    status_final: pd.Series,
    start_year_source_type: pd.Series,
    end_year_source_type: pd.Series,
    lifetimes: dict[str, int],
    start_year_imputation_method: str,
) -> pd.DataFrame:
    """Build row-level diagnostics for powerplant age imputation."""
    diagnostics = pd.DataFrame(index=original_df.index)

    passthrough_cols = [
        "powerplant_id",
        "name",
        "country_id",
        "category",
        "technology",
        "output_capacity_mw",
    ]

    for col in passthrough_cols:
        if col in original_df.columns:
            diagnostics[col] = original_df[col]

    diagnostics["lifetime"] = original_df["technology"].map(lifetimes)

    diagnostics["status_original"] = original_df["status"]
    diagnostics["status_final"] = status_final.reindex(original_df.index)

    diagnostics["start_year_original"] = original_df["start_year"]
    diagnostics["start_year"] = aged_df["start_year"].reindex(original_df.index)
    diagnostics["start_year_source_type"] = start_year_source_type.reindex(
        original_df.index
    )

    diagnostics["end_year_original"] = original_df["end_year"]
    diagnostics["end_year"] = aged_df["end_year"].reindex(original_df.index)
    diagnostics["end_year_source_type"] = end_year_source_type.reindex(
        original_df.index
    ) 
    
    # TODO: these source types should be reflected in scripts/_schemas

    diagnostics["retained_after_time_imputation"] = (
        diagnostics["start_year"].notna()
        & diagnostics["end_year"].notna()
    )

    diagnostics["start_year_imputation_method"] = start_year_imputation_method

    return diagnostics.reset_index(drop=True)

def _save_age_imputation_diagnostics(
    diagnostics: pd.DataFrame,
    output_path_str: str | None,
) -> None:
    """Save row-level age-imputation diagnostics if requested."""
    if output_path_str is None:
        return

    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(output_path, index=False)

def impute(
    relocated_gdf: gpd.GeoDataFrame,
    category_capacity_df: pd.DataFrame,
    imputation: dict,
    technology_mapping: dict,
    age_imputation_output_path: str | None = None,
) -> gpd.GeoDataFrame:
    """Add automatic and user imputations to fill missing data.

    Args:
        relocated_gdf: Relocated powerplants with country identifiers.
        category_capacity_df: Annual category-capacity stock data.
        imputation: Imputation configuration.
        technology_mapping: Technology-mapping configuration.
        age_imputation_output_path: Path for row-level imputation diagnostics.
    """
    if relocated_gdf.empty:
        imputed = relocated_gdf
        _save_age_imputation_diagnostics(
            pd.DataFrame(),
            age_imputation_output_path,
        )
    else:
        _utils.check_single_category(relocated_gdf)
        if (relocated_gdf.geometry.geom_type != "Point").any():
            raise ValueError(
                "Polygon powerplant geometries detected. Only Points are supported."
            )

        lifetimes = imputation["lifetime_years"]
        retirement_delay_years = imputation["retirement_delay_years"]
        scenario = SCENARIO_MAP[imputation["scenario"]]
        start_year_imputation_method = imputation.get(
            "start_year_imputation_method",
            "group_average",
        )

        # Get facilities within the requested scenario.
        imputed = relocated_gdf[relocated_gdf["status"].isin(scenario)].copy()
        original = imputed.copy()

        if not imputed.empty:
            imputed["start_year"], start_year_source_type = _impute_start_year(
                prepared_df=imputed,
                category_capacity_df=category_capacity_df,
                lifetimes=lifetimes,
                method=start_year_imputation_method,
            )

            imputed["end_year"], end_year_source_type = _impute_end_year(
                imputed,
                lifetimes,
                retirement_delay_years,
            )

            has_complete_dates = imputed[["start_year", "end_year"]].notna().all(
                axis=1
            )

            status_final = pd.Series(pd.NA, index=imputed.index, dtype="object")

            if has_complete_dates.any():
                status_final.loc[has_complete_dates] = _impute_status(
                    imputed.loc[has_complete_dates]
                )

            diagnostics = _build_age_imputation_diagnostics(
                original_df=original,
                aged_df=imputed,
                status_final=status_final,
                start_year_source_type=start_year_source_type,
                end_year_source_type=end_year_source_type,
                lifetimes=lifetimes,
                start_year_imputation_method=start_year_imputation_method,
            )
            _save_age_imputation_diagnostics(
                diagnostics,
                age_imputation_output_path,
            )

            # Drop projects with insufficient date data.
            imputed = imputed.loc[has_complete_dates].copy()

            # Update the powerplant status.
            imputed["status"] = status_final.loc[imputed.index]
        else:
            diagnostics = _build_age_imputation_diagnostics(
                original_df=original,
                aged_df=imputed,
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
            _save_age_imputation_diagnostics(
                diagnostics,
                age_imputation_output_path,
            )

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
    tech_types = sorted(df["technology"].dropna().unique())
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
        category_capacity_df=pd.read_parquet(snakemake.input.category_capacity),
        imputation=snakemake.params.imputation,
        technology_mapping=snakemake.params.tech_map,
        age_imputation_output_path=snakemake.output.age_imputation,
    )
    imputed_gdf.to_parquet(snakemake.output.aged)

    plot_powerplant_capacity_buildup(
        imputed_gdf,
        snakemake.output.histogram,
        "seaborn:tab20",
        snakemake.wildcards.category,
    )
    explore(imputed_gdf, snakemake.output.explorer)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
