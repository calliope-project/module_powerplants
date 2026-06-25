"""Functions for proxying missing data."""

import math
import sys
from typing import TYPE_CHECKING, Any

import _schemas
import _utils
import geopandas as gpd
import gregor
import pandas as pd
import rioxarray as rxr
from cmap import Colormap
from matplotlib import pyplot as plt

if TYPE_CHECKING:
    snakemake: Any


def _get_borders_gdf(shapes_file: str) -> gpd.GeoDataFrame:
    """Constructs country borders by removing marine regions and dissolving land regions.

    Args:
        shapes_file (str): Path to shapefile with national regions.

    Returns:
        gpd.GeoDataFrame: dataframe with only land borders.
    """
    shapes = _schemas.ShapeSchema.validate(gpd.read_parquet(shapes_file))
    if "shape_class" in shapes.columns:
        shapes = shapes[shapes["shape_class"] == "land"]
    shapes = shapes[["country_id", "geometry"]].dissolve("country_id").reset_index()
    if not shapes["country_id"].is_unique:
        raise ValueError(f"Borders file contains duplicate countries: {shapes_file}")
    shapes["geometry"] = shapes.buffer(0)
    return shapes


def proxy_rooftop_pv_capacity(
    shapes_file: str,
    proxy_file: str,
    aggregated_unadjusted_file: str,
    stats_file: str,
    output_file: str,
    category: str,
    year: int,
):
    """Produce a proxy raster used to disaggregate missing capacity.

    Will only be computed for regions with missing capacity.
    If unadjusted capacity exceeds statistics, the country will be dismissed.

    Args:
        shapes_file (str): shapefile at national level.
        proxy_file (str): proxy raster file.
        aggregated_unadjusted_file (str): file with unadjusted capacity.
        stats_file (str): country_statistics.
        output_file (str): output file location.
        category (str): category to process.
        year (int): year to use for adjustment.
    """
    borders_df = _get_borders_gdf(shapes_file).set_index("country_id")
    stats_df = pd.read_parquet(stats_file)
    area_potential_da = rxr.open_rasterio(proxy_file).squeeze()  # type: ignore[union-attr]

    agg_unadjusted_df = pd.read_parquet(aggregated_unadjusted_file)
    unadj_cap_mw = agg_unadjusted_df.groupby("country_id")["output_capacity_mw"].sum()

    stats_df = stats_df[stats_df["year"] == year]
    stats_df = stats_df[stats_df["category"].isin(_utils.EIA_CAT_MAPPING[category])]
    total_cap_mw = stats_df.set_index("country_id")["capacity_mw"]

    missing_cap_mw = total_cap_mw - unadj_cap_mw
    missing_cap_mw = missing_cap_mw.dropna()
    missing_cap_mw = missing_cap_mw.where(missing_cap_mw >= 0, 0)
    borders_df["output_capacity_mw"] = missing_cap_mw

    proxy = gregor.disaggregate.disaggregate_polygon_to_raster(
        borders_df, column="output_capacity_mw", proxy=area_potential_da.chunk("auto"),
    )
    proxy.attrs |= {
        "name": "output_capacity_mw",
        "long_name": "Output capacity (MW)",
        "unit": "mw",
    }
    proxy.attrs
    proxy.rio.to_raster(output_file)


def plot(proxy_file: str, shapes_file: str, output_file: str, pixels: int = 500_000):
    """Plot a figure of the generated proxy.

    Args:
        proxy_file (str): proxy file generated.
        shapes_file (str): shapes used for the proxy.
        output_file (str): output image file location.
        pixels (int): pixel count.
    """
    shapes_gdf = gpd.read_parquet(shapes_file)

    area_potential_da = rxr.open_rasterio(proxy_file).squeeze()  # type: ignore[union-attr]

    # Compute a coarsening factor to avoid memory limits
    nx, ny = area_potential_da.sizes["x"], area_potential_da.sizes["y"]
    factor = math.ceil(math.sqrt((nx * ny) / pixels))
    pixel_count = (nx // factor) * (ny // factor)

    # Coarsen the proxy data
    coarse = area_potential_da.coarsen(x=factor, y=factor, boundary="trim").mean()

    fig, ax = plt.subplots(dpi=300)
    im = coarse.plot.imshow(
        ax=ax,
        cmap=Colormap("seaborn:rocket").to_matplotlib(),
        add_colorbar=False,
        alpha=1,
    )
    shapes_gdf.to_crs(area_potential_da.rio.crs).geometry.boundary.plot(
        ax=ax, color="lightgrey", linewidth=0.5, alpha=0.7
    )
    ax.set_axis_off()
    ax.set_title(f"Aggregation proxy (coarsened ~{pixel_count:.1e} pixels)")
    fig.colorbar(im, ax=ax, location="right", label="Proxied capacity")

    fig.savefig(output_file, bbox_inches="tight")


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    proxy_rooftop_pv_capacity(
        shapes_file=snakemake.input.shapes,
        proxy_file=snakemake.input.proxy,
        aggregated_unadjusted_file=snakemake.input.agg_unadj,
        stats_file=snakemake.input.stats,
        output_file=snakemake.output.proxy,
        category=snakemake.params.category,
        year=_utils.DATASET_YEAR,
    )
    plot(
        proxy_file=snakemake.output.proxy,
        shapes_file=snakemake.input.shapes,
        output_file=snakemake.output.plot,
    )
