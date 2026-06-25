"""Prepare an oil and gas dataset using our schemas."""

import sys
from typing import TYPE_CHECKING, Any

import _gem as gem
import _schemas
import _utils
import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    snakemake: Any

STATUS_MAPPING_COAL = {
    "announced": "announced",
    "pre-permit": "pre-construction",
    "permitted": "pre-construction",
    "construction": "construction",
    "operating": "operating",
    "mothballed": "retired",
    "retired": "retired",
}


def _get_coal_ccs(gem_df: pd.DataFrame) -> pd.Series:
    """Get CCS status of coal powerplants.

    Can be inferred from technology and coal consumption nomenclature.
    """
    return (
        gem_df[["combustion_technology", "coal_type"]]
        .fillna("")
        .apply(
            lambda x: (
                True
                if ("CCS" in x["combustion_technology"] or "CCS" in x["coal_type"])
                else False
            ),
            axis="columns",
        )
    )


def _get_coal_fuel(gem_df: pd.DataFrame) -> pd.Series:
    """Harmonise fuel nomenclature with other GEM datasets."""
    # bituminous -> coal: bituminous
    coal_type = "coal: " + gem_df["coal_type"].str.replace("with CCS", "").fillna(
        "unknown"
    )
    # bioenergy - paper mill wastes -> bioenergy: paper mill wastes
    alternate_fuel = gem_df["alternate_fuel"].str.replace(" - ", ": ")
    return coal_type.combine(
        alternate_fuel, lambda a, b: f"{a.strip()}, {b.strip()}" if pd.notna(b) else a
    )


def _get_end_year(
    gem_df: pd.DataFrame, *, planned_ret_col: str, ret_col: str = "retired_year"
) -> pd.Series:
    """Return retirement year.

    Fossil GEM datasets have two separate columns for this value.
    """
    return gem_df.apply(
        lambda x: (
            pd.to_numeric(x[ret_col], errors="coerce")
            if pd.notna(x[ret_col])
            else pd.to_numeric(x[planned_ret_col], errors="coerce")
        ),
        axis="columns",
    )


def prepare_gem_gcpt(
    gem_gcpt_path: str,
    technology_mapping: dict[str, str],
    fuel_settings: dict,
    crs: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Obtain coal power locations using GEM-GCPT data."""
    raw_df = gem.read_gem_dataset(gem_gcpt_path, ["Units"])

    # Create fuel lookups
    fuels_df, fuel_class = gem.get_unique_fuel_dataset(
        raw_fuels=_get_coal_fuel(raw_df),
        mapping=fuel_settings["mapping"],
        ignored=fuel_settings["ignored"],
        default="coal: unknown",
        class_prefix="c",
    )

    coal_df = gpd.GeoDataFrame(
        {
            "powerplant_id": _utils.get_combined_text_col(
                raw_df, ["gem_location_id", "gem_unit/phase_id"], prefix="GEM_"
            ),
            "name": _utils.get_combined_text_col(raw_df, ["plant_name", "unit_name"]),
            "category": "fossil",
            "technology": gem.technology_col(
                raw_df, technology_mapping, col="combustion_technology"
            ),
            "output_capacity_mw": raw_df["capacity_(mw)"],
            "start_year": gem.year_col(raw_df, "start"),
            "end_year": _get_end_year(raw_df, planned_ret_col="planned_retirement"),
            "status": gem.status_col(raw_df, mapping=STATUS_MAPPING_COAL),
            "geometry": _utils.get_point_col(raw_df, "longitude", "latitude", crs),
            "ccs": _get_coal_ccs(raw_df),
            "chp": False,  # Not specified in GCPT
            "fuel_class": fuel_class,
        },
        crs=crs,
    ).reset_index(drop=True)
    coal_df = _utils.ensure_positive_capacity(coal_df)
    schema = _schemas.build_schema(technology_mapping, "prepare")
    return schema.validate(coal_df), _schemas.FuelSchema.validate(fuels_df)


def prepare_gem_gogpt(
    gem_gogpt_path: str,
    technology_mapping: dict[str, str],
    fuel_settings: dict,
    crs: str,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Obtain oil and gas power plants using GEM-GOGPT data."""
    raw_df = gem.read_gem_dataset(
        gem_gogpt_path, ["Gas & Oil Units", "sub-threshold units"]
    )

    fuels_df, fuel_class = gem.get_unique_fuel_dataset(
        raw_fuels=raw_df["fuel"],
        mapping=fuel_settings["mapping"],
        ignored=fuel_settings["ignored"],
        default="fossil gas: unknown",
        class_prefix="og",
    )

    oil_gas_df = gpd.GeoDataFrame(
        {
            "powerplant_id": _utils.get_combined_text_col(
                raw_df, ["gem_location_id", "gem_unit_id"], prefix="GEM_"
            ),
            "name": _utils.get_combined_text_col(raw_df, ["plant_name", "unit_name"]),
            "category": "fossil",
            "technology": gem.technology_col(
                raw_df, technology_mapping, col="turbine/engine_technology"
            ),
            "output_capacity_mw": raw_df["capacity_(mw)"],
            "start_year": gem.year_col(raw_df, "start"),
            "end_year": _get_end_year(raw_df, planned_ret_col="planned_retire"),
            "status": gem.status_col(raw_df),
            "geometry": _utils.get_point_col(raw_df, "longitude", "latitude", crs),
            "ccs": raw_df["ccs_attachment?"] == "yes",
            "chp": raw_df["chp"] == "yes",
            "fuel_class": fuel_class,
        },
        crs=crs,
    ).reset_index(drop=True)
    oil_gas_df = _utils.ensure_positive_capacity(oil_gas_df)
    schema = _schemas.build_schema(technology_mapping, "prepare")
    return schema.validate(oil_gas_df), _schemas.FuelSchema.validate(fuels_df)


def main() -> None:
    """Main snakemake process."""
    # Oil and gas
    og_plants, og_fuels = prepare_gem_gogpt(
        gem_gogpt_path=snakemake.input.gem_gogpt,
        technology_mapping=snakemake.params.technology_mapping["oil_gas"],
        fuel_settings=snakemake.params.fuel_settings,
        crs=snakemake.params.geo_crs,
    )
    og_plants.to_parquet(snakemake.output.og_plants)
    og_fuels.to_parquet(snakemake.output.og_fuels)

    # Coal
    coal_plants, coal_fuels = prepare_gem_gcpt(
        gem_gcpt_path=snakemake.input.gem_gcpt,
        technology_mapping=snakemake.params.technology_mapping["coal"],
        fuel_settings=snakemake.params.fuel_settings,
        crs=snakemake.params.geo_crs,
    )
    coal_plants.to_parquet(snakemake.output.coal_plants)
    coal_fuels.to_parquet(snakemake.output.coal_fuels)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
