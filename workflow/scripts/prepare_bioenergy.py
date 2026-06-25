"""Prepare a bioenergy dataset using our schemas."""

import sys
from typing import TYPE_CHECKING, Any

import _gem as gem
import _schemas
import _utils
import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    snakemake: Any


def _start_year(gem_df: pd.DataFrame):
    """Return retirement year.

    GBPT has a separate column for fossil powerplants that were converted to bioenergy.
    """
    return gem_df.apply(
        lambda x: (
            pd.to_numeric(x["unit_conversion_year"], errors="coerce")
            if pd.notna(x["unit_conversion_year"])
            else pd.to_numeric(x["start_year"], errors="coerce")
        ),
        axis="columns",
    )


def main():
    """Obtain bioenergy power locations using GEM-GBPT data."""
    raw_df = gem.read_gem_dataset(snakemake.input.gem_gbpt, ["Data"])

    fuel_settings = snakemake.params.fuel_settings
    fuels_df, fuel_class = gem.get_unique_fuel_dataset(
        raw_fuels=raw_df["fuel"],
        mapping=fuel_settings["mapping"],
        ignored=fuel_settings["ignored"],
        default="bioenergy: unknown",
        class_prefix="b"
    )
    _schemas.FuelSchema.validate(fuels_df).to_parquet(snakemake.output.fuels)

    technology_mapping = snakemake.params.technology_mapping
    crs = snakemake.params.geo_crs
    bioenergy_df = gpd.GeoDataFrame(
        {
            "powerplant_id": _utils.get_combined_text_col(
                raw_df, ["gem_location_id", "gem_phase_id"], prefix="GEM_"
            ),
            "name": _utils.get_combined_text_col(raw_df, ["project_name", "unit_name"]),
            "category": "bioenergy",
            "technology": technology_mapping["unknown"],
            "output_capacity_mw": raw_df["capacity_(mw)"],
            "start_year": _start_year(raw_df),
            "end_year": gem.year_col(raw_df, "end"),
            "status": gem.status_col(raw_df),
            "geometry": _utils.get_point_col(raw_df, "longitude", "latitude", crs),
            "ccs": False,
            "chp": False,
            "fuel_class": fuel_class,
        },
        crs=crs,
    ).reset_index(drop=True)
    schema = _schemas.build_schema(technology_mapping, "prepare")
    schema.validate(bioenergy_df).to_parquet(snakemake.output.plants)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
