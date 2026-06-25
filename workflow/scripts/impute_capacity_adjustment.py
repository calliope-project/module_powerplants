"""Adjustment of powerplant capacity for generic cases."""

import sys
from typing import TYPE_CHECKING, Any

import _plots
import _schemas
import _utils
import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    snakemake: Any


def adjustment(plants: pd.DataFrame, stats: pd.DataFrame, year: int) -> pd.DataFrame:
    """Adjust operating plant capacity to national statistics in the given year.

    Rules:
    - Past plants are kept unchanged.
    - Future plants are kept unchanged.
    - Operating plants are adjusted to match the statistics.
    - If a country's expected operating capacity is missing or zero in that year,
      operating plants in that country are removed.
    """
    category = _utils.check_single_category(plants)
    cat_stats = _utils.get_eia_stats_in_cat_yr(stats, year, category)
    expected_capacity = cat_stats.groupby("country_id")["capacity_mw"].sum()

    # isolate powerplants operating in the requested year
    adjusted = plants.copy()
    operating = _utils.filter_years(adjusted, year, how="operating")
    non_operating = adjusted.drop(index=operating.index)

    # empty placeholder for currently operating powerplants
    adjusted_operating = operating.head(0).copy()
    if not operating.empty and not expected_capacity.empty:
        positive_expected = expected_capacity[expected_capacity > 0]
        operating_to_adjust = operating[
            operating["country_id"].isin(positive_expected.index)
        ].copy()

        if not operating_to_adjust.empty:
            operating_to_adjust["output_capacity_mw"] = _utils.get_adjusted_capacity(
                operating_to_adjust, positive_expected
            )
            adjusted_operating = operating_to_adjust

    adjusted = pd.concat([non_operating, adjusted_operating], ignore_index=True)
    return adjusted


def adjust_powerplant_capacity(
    stats_file: str, unadjusted_file: str, year: int, output_file: str
):
    """Adjust powerplant capacity in the given year.

    Also appends future projects without altering their values.
    """
    stats = pd.read_parquet(stats_file)
    plants = gpd.read_parquet(unadjusted_file)

    if plants.empty:
        adjusted_plants = plants
    else:
        adjusted_plants = adjustment(plants, stats, year)

    _schemas.PlantSchema.validate(adjusted_plants).to_parquet(output_file)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    adjust_powerplant_capacity(
        stats_file=snakemake.input.stats,
        unadjusted_file=snakemake.input.unadjusted,
        year=_utils.DATASET_YEAR,
        output_file=snakemake.output.adjusted,
    )
    _plots.plot_capacity_adjustment(
        stats_file=snakemake.input.stats,
        unadjusted_file=snakemake.input.unadjusted,
        adjusted_file=snakemake.output.adjusted,
        year=_utils.DATASET_YEAR,
        output_file=snakemake.output.plot,
        is_disagg=True,
    )
