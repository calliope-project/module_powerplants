"""Plot functions used in one or more rules."""

from collections.abc import Collection

import _schemas
import _utils
import geopandas as gpd
import numpy as np
import pandas as pd
from cmap import Colormap
from matplotlib import pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.axes import Axes
from matplotlib.patches import Patch


def draw_empty(ax: Axes, title: str = "", message="No data available"):
    """Helper to render an empty-data placeholder."""
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, alpha=0.7)
    if title:
        ax.set_title(title)
    ax.set_axis_off()


def plot_empty(title: str, output_path: str) -> None:
    fig, ax = plt.subplots()
    draw_empty(ax, "")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output_path)


def plot_capacity_adjustment(
    stats_file: str,
    unadjusted_file: str,
    adjusted_file: str,
    year: int,
    output_file: str,
    is_disagg: bool,
    *,
    country_col: str = "country_id",
    dis_tech_col: str = "technology",
    dis_cap_col: str = "output_capacity_mw",
    eia_cat_col: str = "category",
    eia_cap_col: str = "capacity_mw",
    bar_width: float = 0.5,
    row_height: float = 3.0,
    fig_width: float = 12.0,
    target_ticks: int = 12,
) -> None:
    """Plot adjustment per country."""
    suptitle = f"Unadjusted vs Adjusted vs EIA Capacity by Country - {year}"
    df_udj = pd.read_parquet(unadjusted_file)
    df_adj = pd.read_parquet(adjusted_file)
    df_eia = pd.read_parquet(stats_file)

    # Handle the no-data case
    if df_udj.empty and df_adj.empty:
        plot_empty(suptitle, output_file)
        return

    if is_disagg:
        df_udj = _utils.filter_years(df_udj, year, how="operating")
        df_adj = _utils.filter_years(df_adj, year, how="operating")

    category_dis = _utils.check_single_category(df_udj)
    category_adj = _utils.check_single_category(df_adj)
    if category_dis != category_adj:
        raise ValueError(
            f"Input datasets are not of the same category: {category_dis} vs {category_adj}"
        )
    df_eia = _utils.get_eia_stats_in_cat_yr(df_eia, year, category_dis)
    df_udj = df_udj[df_udj["country_id"].isin(df_eia["country_id"].unique())]

    # aggregate total capacities.
    agg_dis = (
        df_udj.groupby([country_col, dis_tech_col])[[dis_cap_col]].sum().reset_index()
    )
    agg_adj = (
        df_adj.groupby([country_col, dis_tech_col])[[dis_cap_col]].sum().reset_index()
    )
    agg_eia = (
        df_eia.groupby([country_col, eia_cat_col])[[eia_cap_col]].sum().reset_index()
    )
    countries = sorted(set(agg_dis[country_col]).union(agg_adj[country_col]))
    techs = sorted(set(agg_dis[dis_tech_col]).union(agg_adj[dis_tech_col]))
    cats = sorted(agg_eia[eia_cat_col].unique())

    # Avoid overlaps when techs / cats have the same name
    tech_slices = [("tech", t) for t in techs]
    cat_slices = [("cat", c) for c in cats]
    slices = tech_slices + cat_slices

    cmap = plt.get_cmap("tab20")
    positions = np.linspace(0, 1, len(slices))
    colours = {k: cmap(p) for k, p in zip(slices, positions)}

    tech_handles = [Patch(color=colours[("tech", t)], label=t) for t in techs[::-1]]
    cat_handles = [Patch(color=colours[("cat", c)], label=c) for c in cats[::-1]]

    # build lookup dicts
    dis = agg_dis.set_index([country_col, dis_tech_col])[dis_cap_col].to_dict()
    adj = agg_adj.set_index([country_col, dis_tech_col])[dis_cap_col].to_dict()
    eia = agg_eia.set_index([country_col, eia_cat_col])[eia_cap_col].to_dict()

    # figure layout (per country rows)
    nrows = len(countries)
    fig, axes = plt.subplots(
        nrows,
        3,  # tech-legend | bars | cat-legend
        figsize=(fig_width, row_height * nrows),
        gridspec_kw={"width_ratios": [1.2, 4, 1.2]},
        squeeze=False,
        tight_layout=True,
    )

    xpos = {"Unadjusted": 0, "Adjusted": 1, "EIA": 2}

    # loop per country
    for (ax_tech, ax_bar, ax_cat), country in zip(axes, countries):
        bottoms = {k: 0.0 for k in xpos}

        for grp, label in slices:
            colour = colours[(grp, label)]

            if grp == "tech":  # from df_dis / df_adj
                dis_val = dis.get((country, label), 0.0)
                adj_val = adj.get((country, label), 0.0)
                eia_val = 0.0
            else:  # grp == 'cat' (EIA dataframe)
                dis_val = 0.0
                adj_val = 0.0
                eia_val = eia.get((country, label), 0.0)

            ax_bar.bar(
                xpos["Unadjusted"],
                dis_val,
                bar_width,
                bottom=bottoms["Unadjusted"],
                color=colour,
            )
            ax_bar.bar(
                xpos["Adjusted"],
                adj_val,
                bar_width,
                bottom=bottoms["Adjusted"],
                color=colour,
            )
            ax_bar.bar(
                xpos["EIA"], eia_val, bar_width, bottom=bottoms["EIA"], color=colour
            )

            bottoms["Unadjusted"] += dis_val
            bottoms["Adjusted"] += adj_val
            bottoms["EIA"] += eia_val

        tot_dis, tot_adj, tot_eia = (
            bottoms["Unadjusted"],
            bottoms["Adjusted"],
            bottoms["EIA"],
        )

        # cosmetics
        ax_bar.axhline(tot_dis, ls=":", lw=0.8, color="grey")
        ax_bar.axhline(tot_adj, ls="--", lw=0.8, color="grey")

        ax_bar.set_xticks([xpos[k] for k in ("Unadjusted", "Adjusted", "EIA")])
        ax_bar.set_xticklabels(["Unadjusted", "Adjusted", "EIA statistics"])
        ax_bar.set_title(country, pad=4)
        ax_bar.set_ylabel("Capacity (MW)")
        ax_bar.yaxis.set_major_locator(mticker.MaxNLocator(nbins=target_ticks))
        ax_bar.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

        # percentage axis
        if tot_adj:
            sec = ax_bar.secondary_yaxis(
                "right",
                functions=(
                    lambda mw, tot=tot_adj: mw / tot * 100,
                    lambda pct, tot=tot_adj: pct * tot / 100,
                ),
            )
            pct_max = max(100, tot_dis / tot_adj * 100, tot_eia / tot_adj * 100)
            sec.set_ylabel(r"% of adjusted total")
            sec.set_yticks(np.linspace(0, pct_max, target_ticks))
            sec.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
            sec.set_ylim(0, pct_max)

        # legends
        for ax, handles, title in [
            (ax_tech, tech_handles, "Technology"),
            (ax_cat, cat_handles, "EIA category"),
        ]:
            ax.axis("off")
            ax.legend(
                handles,
                [h.get_label() for h in handles],
                ncol=1,
                frameon=False,
                title=title,
                loc="center",
            )
    fig.suptitle(suptitle, y=0.999)

    fig.savefig(output_file, bbox_inches="tight")


def plot_capacity_aggregation(
    aggregated_file: str,
    shapes_file: str,
    output_file: str,
    category: str,
    crs: int | str,
):
    """Plot aggregated capacity per region."""
    shapes = _schemas.ShapeSchema.validate(gpd.read_parquet(shapes_file))
    shapes = shapes.to_crs(_utils.check_crs(crs, "projected"))
    agg = _schemas.AggregatedPlantSchema.validate(pd.read_parquet(aggregated_file))

    title = f"Aggregated {category} capacity"

    if agg.empty:
        plot_empty(title, output_file)
    else:
        cap_by_shape = agg.groupby("shape_id")["output_capacity_mw"].sum()

        shapes = shapes.set_index("shape_id")
        shapes["output_capacity_mw"] = cap_by_shape.replace(0, np.nan)

        fig, ax = plt.subplots(dpi=200)

        ax = shapes.plot(
            ax=ax,
            column="output_capacity_mw",
            cmap="plasma",
            edgecolor="grey",
            linewidth=0.5,
            legend=True,
            legend_kwds={"label": "Capacity ($MW$)"},
            missing_kwds={"color": "lightgrey", "alpha": 0.2},
        )
        ax.set_axis_off()
        ax.set_title(title + f" in year {agg.attrs['year']}")
        fig.savefig(output_file, bbox_inches="tight")

def get_colour_dict(
    sources: Collection[str],
    colormap: str,
    *,
    value_range: tuple[float, float] = (0.0, 1.0),
) -> dict[str, str]:
    """Return deterministic colours for a collection of source types."""
    sorted_sources = sorted(sources)

    if not sorted_sources:
        return {}

    cmap = Colormap(colormap).to_mpl()
    values = np.linspace(value_range[0], value_range[1], len(sorted_sources))

    return {
        source: colour
        for source, colour in zip(sorted_sources, cmap(values), strict=True)
    }


def get_time_imputation_colours() -> dict[str, str]:
    """Return colours for time-imputation source types."""
    start_year_sources = _utils.date_source_types_for("start_year")
    end_year_sources = _utils.date_source_types_for("end_year")

    observed_sources = start_year_sources & end_year_sources
    start_only_sources = start_year_sources - end_year_sources
    end_only_sources = end_year_sources - start_year_sources

    return (
        get_colour_dict(observed_sources, "colorbrewer:Greys", value_range=(0.3,0.4))  #forces observed values to a light grey
        | get_colour_dict(start_only_sources, "colorbrewer:Purples", value_range=(0.1,1))  #prevents anything to pale in the plot
        | get_colour_dict(end_only_sources, "colorbrewer:Reds",value_range=(0.1,1)) #prevents anything to pale in the plot
    )
