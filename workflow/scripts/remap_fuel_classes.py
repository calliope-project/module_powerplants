"""Fuel class harmonisation."""

import sys
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pandas as pd

if TYPE_CHECKING:
    snakemake: Any


def remap(
    plants_files: tuple[str, ...], old_class_files: tuple[str, ...], new_class_file: str
) -> gpd.GeoDataFrame:
    """Re-map powerplant fuel classes to a different class dataset."""
    new_fuel_df = pd.read_parquet(new_class_file)

    new_fuels_by_class = new_fuel_df.groupby("fuel_class")["fuel"].apply(
        lambda values: tuple(sorted(values))
    )
    new_class_by_fuels = {
        fuel_combo: fuel_class for fuel_class, fuel_combo in new_fuels_by_class.items()
    }

    remapped = []

    for plants_file, old_class_file in zip(plants_files, old_class_files):
        plants_df = gpd.read_parquet(plants_file)
        old_fuel_df = pd.read_parquet(old_class_file)

        old_fuels_by_class = old_fuel_df.groupby("fuel_class")["fuel"].apply(
            lambda values: tuple(sorted(values))
        )

        plants_df["fuel_class"] = plants_df["fuel_class"].apply(
            lambda fuel_class: new_class_by_fuels[old_fuels_by_class[fuel_class]]
        )

        remapped.append(plants_df)

    return pd.concat(remapped, ignore_index=True, sort=False, axis="index")


def main() -> None:
    """Main snakemake process."""
    remapped_gdf = remap(
        snakemake.input.plants, snakemake.input.old_classes, snakemake.input.new_classes
    )
    remapped_gdf.to_parquet(snakemake.output.remapped)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
