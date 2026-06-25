"""General utilities for processing GEM datasets."""

import re
from typing import Literal

import pandas as pd

GEM_GWPT_SHEETS = ["Data", "Below Threshold"]

GEM_GSPT_SHEETS = ["20 MW+", "1-20 MW"]
GSPT_CAPACITY_RATING_MAPPING = {"MWac": "AC", "MWp/dc": "DC"}

_INVALID_STATUS_VALUES = ["cancelled", "shelved"]
_DROPPED_NA_COLUMNS = ["capacity_(mw)", "latitude", "longitude"]


_STATUS_MAPPING = {
    "announced": "announced",
    "pre-construction": "pre-construction",
    "construction": "construction",
    "operating": "operating",
    "mothballed": "retired",
    "retired": "retired",
}


def read_gem_dataset(
    path: str, sheets: list[str], dropped_na_cols: list[str] | None = None
) -> pd.DataFrame:
    """Get a GEM dataset for a type/category of powerplant."""
    if dropped_na_cols is None:
        dropped_na_cols = _DROPPED_NA_COLUMNS

    gem_df = pd.concat([pd.read_excel(path, sheet) for sheet in sheets], axis="index")
    # Harmonise column names
    gem_df.columns = gem_df.columns.str.strip().str.lower().str.replace(" ", "_")
    # Get raw dataset without cancelled projects
    pattern = "|".join(map(re.escape, _INVALID_STATUS_VALUES))
    mask = gem_df["status"].str.contains(pattern, na=False)
    gem_df = gem_df[~mask]
    # Remove rows with problematic empty values
    gem_df = gem_df.dropna(subset=dropped_na_cols)
    gem_df = gem_df.reset_index(drop=True)

    return gem_df


def year_col(gem_df: pd.DataFrame, option: Literal["start", "end"]):
    """Get start/end year, ensuring typing is respected."""
    mapping = {"start": "start_year", "end": "retired_year"}
    return gem_df[mapping[option]].apply(lambda x: pd.to_numeric(x, errors="coerce"))


def status_col(gem_df: pd.DataFrame, mapping: dict | None = None):
    """Get standardised plant status."""
    if mapping is None:
        mapping = _STATUS_MAPPING
    return gem_df["status"].map(mapping)


def technology_col(
    gem_df: pd.DataFrame, mapping: dict[str, str], col: str = "technology"
) -> pd.Series:
    """Remap technology names, cleaning CCS specifics and other inconsistencies."""
    return (
        gem_df[col]
        .fillna("unknown")
        .replace("unknown type", "unknown")
        .apply(lambda x: mapping[x.replace("/CCS", "").strip()])
    )


def output_capacity_mw_gspt(
    gem_df: pd.DataFrame, dc_ac_ratio: float, default_rating: Literal["AC", "DC"]
):
    """Obtain capacity, applying DC-to-AC ratios where necessary."""
    rating = (
        gem_df["capacity_rating"]
        .fillna("unknown")
        .replace("unknown", default_rating)
        .replace(GSPT_CAPACITY_RATING_MAPPING)
    )
    invalid_ratings = set(rating.unique()) - set(GSPT_CAPACITY_RATING_MAPPING.values())
    assert not invalid_ratings, f"GEM GSPT: invalid capacity ratings {invalid_ratings}."

    cap_df = pd.concat([gem_df["capacity_(mw)"], rating], axis="columns")
    return cap_df.apply(
        lambda x: (
            x["capacity_(mw)"]
            if x["capacity_rating"] == "AC"
            else x["capacity_(mw)"] / dc_ac_ratio
        ),
        axis="columns",
    )


def _remap_fuel_col(
    cell: str, mapping: dict, default: str, ignored: list[str]
) -> tuple[str, ...]:
    """Find and replace fuel names using pattern matching."""
    fuels = set()
    if pd.isna(cell):
        fuels.add(default)
    else:
        for value in cell.split(","):
            # removes shares within brackets (e.g., fossil gas: LNG [50%])
            value = re.sub(r"\s*\[.*?\]", "", value).strip()
            if ":" not in value:
                # removes ambiguous cases (e.g., fossil gas -> fossil gas: unknown)
                value = value + ": unknown"
            if value in ["other: other", "other: unknown"]:
                # Too ambiguous to map usefully
                continue
            try:
                if value in ignored:
                    continue
                fuels.add(mapping[value])
            except KeyError:
                raise KeyError(f"No mapped fuel for '{value}'.")
    if len(fuels) == 0:
        # Handle edge cases where only ambiguous fuels are given.
        fuels.add(default)
    return tuple(sorted(fuels))


def get_unique_fuel_dataset(
    raw_fuels: pd.Series,
    mapping: dict[str, str],
    ignored: list[str],
    default: str,
    class_prefix: str = "f",
) -> tuple[pd.DataFrame, dict]:
    """Get a row with fuel values and convert it to a unique fuel class database.

    Example [(oil, coal), (coal), ...] returns

    - class dataframe:
        id  fuel_class  fuel
        0   f1          coal
        1   f1          oil
        3   f2          coal
    - class series (same index as raw_fuels):
        f1, f2, ...

    Args:
        raw_fuels (pd.Series[str]): series with fuel values.
        mapping (dict[str, str]): fuel mapping (for renaming).
        ignored (list[str]): fuels to exclude, if detected.
        default (str): default fuel, if missing or unknown.
        class_prefix (str, optional): prefix for the fuel class. Defaults to "f".

    Returns:
        tuple[pd.DataFrame, dict]: clas dataframe and class series.
    """
    fuels = raw_fuels.apply(
        _remap_fuel_col, mapping=mapping, default=mapping[default], ignored=ignored
    )
    fuel_combs = sorted(set(fuels))
    fuel_class_df = pd.DataFrame(
        [
            (f"{class_prefix}{i}", fuel)
            for i, comb in enumerate(fuel_combs)
            for fuel in comb
        ],
        columns=["fuel_class", "fuel"],
    ).reset_index(drop=True)
    combo_to_class = {combo: f"{class_prefix}{i}" for i, combo in enumerate(fuel_combs)}
    class_series = fuels.apply(lambda x: combo_to_class[x])
    return fuel_class_df, class_series
