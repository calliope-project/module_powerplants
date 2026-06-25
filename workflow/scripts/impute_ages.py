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

if TYPE_CHECKING:
    snakemake: Any

HISTORICAL = {"operating", "retired"}
SCENARIO_MAP = {
    "historical": HISTORICAL,
    "construction": HISTORICAL | {"construction"},
    "pre_construction": HISTORICAL | {"construction", "pre-construction"},
    "announced": HISTORICAL | {"construction", "pre-construction", "announced"},
}


def _impute_start_year(
    prepared_df: pd.DataFrame, lifetimes: dict[str, int]
) -> pd.Series:
    """Fill start year using reasonable assumptions and user settings."""
    start_year = prepared_df["start_year"].copy()

    # Impute using the lifetime if possible.
    mask_life = start_year.isna() & prepared_df["end_year"].notna()
    start_year[mask_life] = prepared_df.loc[mask_life, "end_year"] - prepared_df.loc[
        mask_life, "technology"
    ].map(lifetimes)
    # Impute using country averages.
    averages = (
        prepared_df.groupby(["country_id", "category", "technology", "status"])[
            "start_year"
        ]
        .transform("mean")
        .round()
    )
    mask_na = start_year.isna()
    start_year.loc[mask_na] = averages[mask_na]

    return start_year


def _impute_end_year(
    df: pd.DataFrame, lifetimes: dict[str, int], delay: dict[str, int]
) -> pd.Series:
    """Impute end_year using lifetime.

    Old plants operating beyond lifetime will retired with a given delay.
    """
    ref_year = _utils.DATASET_YEAR

    # Impute expected end year only if no data is present.
    expected_end = df["start_year"] + df["technology"].map(lifetimes)
    result = df["end_year"].copy().fillna(expected_end)

    # Plants operating beyond expected lifetime will be retired after a delay of >=1 yr
    needs_delay = (result <= ref_year) & (df["status"] == "operating")
    delayed_end = result + df["technology"].map(delay).fillna(0).astype(int)
    delayed_end = delayed_end.clip(lower=ref_year + 1)
    result.loc[needs_delay] = delayed_end.loc[needs_delay]

    return result


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


def impute(
    relocated_gdf: gpd.GeoDataFrame, imputation: dict, technology_mapping: dict
) -> gpd.GeoDataFrame:
    """Add automatic and user imputations to fill missing data.

    Args:
        relocated_gdf (gpd.GeoDataFrame): relocated powerplants (must have country_id).
        imputation (str): imputation configuration.
        technology_mapping (str): technology mapping configuration.
    """
    if relocated_gdf.empty:
        imputed = relocated_gdf
    else:
        _utils.check_single_category(relocated_gdf)
        if (relocated_gdf.geometry.geom_type != "Point").any():
            raise ValueError(
                "Polygon powerplant geometries detected. Only Points are supported."
            )

        lifetimes = imputation["lifetime_years"]
        retirement_delay_years = imputation["retirement_delay_years"]
        scenario = SCENARIO_MAP[imputation["scenario"]]

        # Get facilities within the requested scenario
        imputed = relocated_gdf[relocated_gdf["status"].isin(scenario)].copy()

        if not imputed.empty:
            # Adjust project dates
            imputed["start_year"] = _impute_start_year(imputed, lifetimes)
            imputed["end_year"] = _impute_end_year(
                imputed, lifetimes, retirement_delay_years
            )
            # Drop projects with insufficient date data
            imputed = imputed.dropna(subset=["start_year", "end_year"])
            # Update the powerplant status
            imputed["status"] = _impute_status(imputed)

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
        imputation=snakemake.params.imputation,
        technology_mapping=snakemake.params.tech_map,
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
