"""Prepare a geothermal power dataset using our schemas."""

import sys
from typing import TYPE_CHECKING, Any

import _gem as gem
import _schemas
import _utils
import geopandas as gpd

if TYPE_CHECKING:
    snakemake: Any


def main(
    gem_ggpt_path: str,
    technology_mapping: dict[str, str],
    crs: str,
    output_plants_path: str,
):
    """Obtain geothermal power plants using GEM-GGPT data."""
    raw_df = gem.read_gem_dataset(
        gem_ggpt_path, ["Data"], ["unit_capacity_(mw)", "latitude", "longitude"]
    )

    geo_df = gpd.GeoDataFrame(
        {
            "powerplant_id": _utils.get_combined_text_col(
                raw_df, ["gem_location_id", "gem_unit_id"], prefix="GEM_"
            ),
            "name": _utils.get_combined_text_col(raw_df, ["project_name", "unit_name"]),
            "category": "geothermal",
            "technology": gem.technology_col(
                raw_df, technology_mapping, col="technology"
            ),
            "output_capacity_mw": raw_df["unit_capacity_(mw)"],
            "start_year": gem.year_col(raw_df, "start"),
            "end_year": gem.year_col(raw_df, "end"),
            "status": gem.status_col(raw_df),
            "geometry": _utils.get_point_col(raw_df, "longitude", "latitude", crs),
        },
        crs=crs,
    ).reset_index(drop=True)
    schema = _schemas.build_schema(technology_mapping, "prepare")
    schema.validate(geo_df).to_parquet(output_plants_path)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main(
        gem_ggpt_path=snakemake.input.gem_ggpt,
        technology_mapping=snakemake.params.technology_mapping,
        crs=snakemake.params.geo_crs,
        output_plants_path=snakemake.output.plants,
    )
