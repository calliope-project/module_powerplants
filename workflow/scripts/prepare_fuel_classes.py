"""Prepare unique fuel classes by all prepared categories that require them."""

import sys
from typing import TYPE_CHECKING, Any

import _schemas
import pandas as pd

if TYPE_CHECKING:
    snakemake: Any


def prepare(fuel_class_files: tuple[str], prefix: str = "fc") -> pd.DataFrame:
    """Prepare harmonised fuel classes across categories.

    Args:
        fuel_class_files (tuple[str]): fuel class datasets.
        output_file (str): harmonised fuel class dataset.
        prefix (str): prefix to append to each class name.
    """
    fuel_df = pd.concat([pd.read_parquet(i) for i in fuel_class_files], axis="index")

    # rebuild sorted tuples of fuels
    grouped = fuel_df.groupby("fuel_class")["fuel"].apply(lambda x: tuple(sorted(x)))
    # remove duplicates and resort
    fuel_combinations = sorted(set(grouped.values))
    fuel_class_df = pd.DataFrame(
        [
            (f"{prefix}{i}", fuel)
            for i, comb in enumerate(fuel_combinations)
            for fuel in comb
        ],
        columns=["fuel_class", "fuel"],
    ).reset_index(drop=True)
    return _schemas.FuelSchema.validate(fuel_class_df)


def main() -> None:
    """Main snakemake process."""
    combined_fuels = prepare(snakemake.input.category_fuels)
    combined_fuels.to_parquet(snakemake.output.fuel_classes)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
