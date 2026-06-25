"""Prepare a clean wind power dataset using our schemas.

Uses the Global Wind Power Tracker (GEM-GWPT) dataset.
A free alternative to the WEMI dataset.
"""

import sys
from typing import TYPE_CHECKING, Any

import _gem as gem
import _schemas
import _utils
import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    snakemake: Any

INTERNAL_TECH_MAPPING = {
    "Onshore": "onshore",
    "Offshore floating": "offshore",
    "Offshore hard mount": "offshore",
    "Offshore mount unknown": "offshore",
}


def _technology(gem_df: pd.DataFrame):
    return gem_df["installation_type"].apply(lambda x: INTERNAL_TECH_MAPPING[x])


def prepare_gem_gwpt(
    gem_gwpt_path: str, tech_mapping: dict, crs: str
) -> gpd.GeoDataFrame:
    """Obtain wind power locations using GEM-GWPT data."""
    raw_df = gem.read_gem_dataset(gem_gwpt_path, gem.GEM_GWPT_SHEETS)
    # Remove unknown installation types to avoid misplacement
    raw_df = raw_df[raw_df["installation_type"] != "Unknown"]
    raw_df = raw_df.dropna(subset=["installation_type"])

    wind_df = gpd.GeoDataFrame(
        {
            "powerplant_id": _utils.get_combined_text_col(
                raw_df, ["gem_location_id", "gem_phase_id"], prefix="GEM_"
            ),
            "name": _utils.get_combined_text_col(
                raw_df, ["project_name", "phase_name"]
            ),
            "category": "wind",
            "technology": _technology(raw_df).map(tech_mapping),
            "output_capacity_mw": raw_df["capacity_(mw)"],
            "start_year": gem.year_col(raw_df, "start"),
            "end_year": gem.year_col(raw_df, "end"),
            "status": gem.status_col(raw_df),
            "geometry": _utils.get_point_col(raw_df, "longitude", "latitude", crs),
        },
        crs=crs,
    )
    schema = _schemas.build_schema(tech_mapping, "prepare")
    return schema.validate(wind_df)


def main():
    """Main snakemake process."""
    gem_gwpt = prepare_gem_gwpt(
        snakemake.input.gem_gwpt, snakemake.params.tech_map, snakemake.params.geo_crs
    )
    gem_gwpt.to_parquet(snakemake.output.path)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
