"""Processing of large-scale solar powerplant datasets.

- Tranzition Zero Solar Asset Mapper (TZ-SAM) dataset.
- GEM Global Solar Power Tracker (GEM-GSPT) dataset.
"""

import sys
from typing import TYPE_CHECKING, Any

import _gem as gem
import _schemas
import _utils
import geopandas as gpd
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    snakemake: Any


def _start_year_tz_sam(df: pd.DataFrame):
    """Assume installation occurred in the middle of the detection window."""
    before = pd.to_datetime(df["constructed_before"])
    after = pd.to_datetime(df["constructed_after"])
    delta = (before - after) / 2
    return (after + delta).dt.year


def fill_tz_with_gem(
    gem_df: gpd.GeoDataFrame, tz_sam_df: gpd.GeoDataFrame, proj_crs="epsg:3857"
) -> gpd.GeoDataFrame:
    """Fills in GEM data that is ambiguous in TZ-SAM's satellite-based approach.

    Handles cases where multiple points fall within the same polygon.
    """
    filled_tz_sam = tz_sam_df.copy()

    within = gpd.sjoin(
        gem_df.to_crs(proj_crs),
        tz_sam_df.to_crs(proj_crs)[["geometry"]],
        how="inner",
        predicate="within",
    )

    # Assume start/end years are the mean of the value stated in GEM.
    start_year = within.groupby("index_right")["start_year"].mean().round().dropna()
    end_year = within.groupby("index_right")["end_year"].mean().round().dropna()

    # Assume the status is the mode (most common category).
    status = within.groupby("index_right")["status"].agg(pd.Series.mode)
    # Drop ambiguous cases (two status appear in equal amounts).
    status = status[status.apply(lambda x: isinstance(x, str))]

    filled_tz_sam.loc[start_year.index, "start_year"] = start_year
    filled_tz_sam.loc[end_year.index, "end_year"] = end_year
    filled_tz_sam.loc[status.index, "status"] = status

    return filled_tz_sam


def get_gem_v_tz_mismatch(
    gem_df: gpd.GeoDataFrame,
    tz_sam_df: gpd.GeoDataFrame,
    valid_status: list[str],
    buffer=1000,
    proj_crs="epsg:8857",
) -> gpd.GeoDataFrame:
    """Estimation of future projects missed by TZ-SAM's satellite-based approach.

    To avoid double counting:
    - Only projects with known pre-operation states are accepted.
    - Only projects outside a buffer are accepted (default 1 km).
    """
    _utils.check_crs(proj_crs, "projected")
    if gem_df.crs != tz_sam_df.crs:
        raise ValueError("GEM and TZ-SAM CRS mismatch.")

    # Filter future projects
    if valid_status:
        future_gem_df = gem_df[gem_df["status"].isin(valid_status)]
    else:
        raise ValueError(f"Must specify valid status to get. Got {valid_status!r}.")

    # Buffer around the points using a projected CRS
    future_gem_buffered = future_gem_df.copy()
    future_gem_buffered["geometry"] = (
        future_gem_buffered["geometry"].to_crs(proj_crs).buffer(buffer)
    )

    intersecting = gpd.sjoin(
        future_gem_buffered,
        tz_sam_df.to_crs(proj_crs)[["geometry"]],
        how="inner",
        predicate="intersects",
    )

    return future_gem_df.loc[~future_gem_df.index.isin(intersecting.index)]


def prepare_solar_utility_pv(
    tz_sam_path: str, gem_gspt_path: str, tech_name: str, dc_ac_ratio: float = 1.25
):
    """Obtain utility-scale PV locations by combinging GEM-GSPT and TZ-SAM data.

    - TZ-SAM is the primary source for current facilities
    - GEM-GSPT provides project status and near-future facilities.
    """
    raw_tz_df = gpd.read_file(tz_sam_path)

    tz_df = gpd.GeoDataFrame(
        {
            "powerplant_id": "TZ-SAM_" + raw_tz_df["cluster_id"],
            "name": "TZ-SAM cluster " + raw_tz_df["cluster_id"],
            "category": "solar",
            "technology": tech_name,
            "output_capacity_mw": raw_tz_df["capacity_mw"],
            "start_year": _start_year_tz_sam(raw_tz_df),
            "end_year": np.nan,
            "status": "operating",
            "geometry": raw_tz_df["geometry"],
        }
    )

    # Get only Utility PV facilities.
    raw_gem_df = gem.read_gem_dataset(gem_gspt_path, gem.GEM_GSPT_SHEETS)
    raw_gem_df["technology_type"] = raw_gem_df["technology_type"].fillna("Assumed PV")
    raw_gem_df = raw_gem_df[raw_gem_df["technology_type"].isin(["Assumed PV", "PV"])]
    gem_df = gpd.GeoDataFrame(
        {
            "powerplant_id": _utils.get_combined_text_col(
                raw_gem_df, ["gem_location_id", "gem_phase_id"], prefix="GEM_"
            ),
            "name": _utils.get_combined_text_col(
                raw_gem_df, ["project_name", "phase_name"]
            ),
            "category": "solar",
            "technology": tech_name,
            "output_capacity_mw": gem.output_capacity_mw_gspt(
                raw_gem_df, dc_ac_ratio, "AC"
            ),
            "start_year": gem.year_col(raw_gem_df, "start"),
            "end_year": gem.year_col(raw_gem_df, "end"),
            "status": gem.status_col(raw_gem_df),
            "geometry": _utils.get_point_col(raw_gem_df, "longitude", "latitude"),
        }
    )

    filled_tz_df = fill_tz_with_gem(gem_df, tz_df)
    gem_mismatch_df = get_gem_v_tz_mismatch(
        gem_df,
        tz_df,
        valid_status=["announced", "pre-construction", "construction", "retired"],
    )

    utility_pv = pd.concat([filled_tz_df, gem_mismatch_df], ignore_index=True)
    # Convert to points to make further processing easier.
    utility_pv["geometry"] = utility_pv["geometry"].representative_point()
    utility_pv = _utils.filter_noncontributing_powerplants(utility_pv)
    schema = _schemas.build_schema({"utility_pv": tech_name}, "prepare")
    return schema.validate(utility_pv)


def prepare_solar_csp(
    gem_gspt_path: str, tech_name: str, dc_ac_ratio: float = 1
) -> gpd.GeoDataFrame:
    """Obtain CSP power locations using GEM-GSPT data."""
    raw_df = gem.read_gem_dataset(gem_gspt_path, gem.GEM_GSPT_SHEETS)
    raw_df = raw_df[raw_df["technology_type"] == "Solar Thermal"]

    csp_df = gpd.GeoDataFrame(
        {
            "powerplant_id": _utils.get_combined_text_col(
                raw_df, ["gem_location_id", "gem_phase_id"], prefix="GEM_"
            ),
            "name": _utils.get_combined_text_col(
                raw_df, ["project_name", "phase_name"]
            ),
            "category": "solar",
            "technology": tech_name,
            "output_capacity_mw": gem.output_capacity_mw_gspt(
                raw_df, dc_ac_ratio, "AC"
            ),
            "start_year": gem.year_col(raw_df, "start"),
            "end_year": gem.year_col(raw_df, "end"),
            "status": gem.status_col(raw_df),
            "geometry": _utils.get_point_col(raw_df, "longitude", "latitude"),
        }
    ).reset_index(drop=True)
    csp_df = _utils.filter_noncontributing_powerplants(csp_df)
    schema = _schemas.build_schema({"csp": tech_name}, "prepare")
    return schema.validate(csp_df)


def main() -> None:
    """Main snakemake process."""
    utility_pv_gdf = prepare_solar_utility_pv(
        snakemake.input.tz_sam,
        snakemake.input.gem_gspt,
        snakemake.params.utility_pv_name,
        snakemake.params.dc_ac_ratio,
    )
    csp_gdf = prepare_solar_csp(snakemake.input.gem_gspt, snakemake.params.csp_name)

    crs = _utils.check_crs(snakemake.params.geo_crs, "geographic")

    # Combine into one large category
    large_solar_gdf = pd.concat(
        [utility_pv_gdf, csp_gdf], ignore_index=True, sort=False, axis="index"
    )
    large_solar_gdf.to_crs(crs).to_parquet(snakemake.output.large_solar)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
