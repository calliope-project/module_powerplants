"""Adjust powerplant point locations to valid placement shapes.

First, this will:
- combine powerplant categories and user provided files
- apply optional exclusions
- assign country IDs to in-scope powerplants
- split or reject powerplants falling in overlapping shapes

Then, configured technologies are moved (per country) to the nearest shape matching
their requested shape class.
- Plants already inside a valid shape are kept in place.
- Plants outside are nudged just inside the nearest valid shape while minimizing
movement from their original location.
"""

import sys
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import hypot
from typing import TYPE_CHECKING, Any, Literal

import _schemas
import _utils
import geopandas as gpd
import pandas as pd
from matplotlib import pyplot as plt
from pyproj import CRS
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points

if TYPE_CHECKING:
    snakemake: Any


type OnError = Literal["raise", "ignore", "drop"]
type CountryOverlapMethod = Literal["raise", "split_capacity"]
type PolygonLike = Polygon | MultiPolygon


@dataclass(frozen=True)
class CountryAssignmentResult:
    """Assigned powerplants and split-created IDs."""

    powerplants: gpd.GeoDataFrame
    split_powerplant_ids: pd.Index


@dataclass(frozen=True)
class LocationAdjustmentResult:
    """Adjusted powerplants and forced-moved IDs."""

    powerplants: gpd.GeoDataFrame
    moved_powerplant_ids: pd.Index


def _nearest_part(point: Point, geometry: PolygonLike) -> Polygon:
    """Return the polygon part closest to point."""
    if isinstance(geometry, MultiPolygon):
        return min(geometry.geoms, key=point.distance)
    return geometry


def _line_strings(geometry: BaseGeometry) -> Iterator[LineString]:
    """Yield non-empty LineStrings from a possibly multipart geometry."""
    if geometry.is_empty:
        return

    if isinstance(geometry, LineString) and geometry.length > 0:
        yield geometry
        return

    for part in getattr(geometry, "geoms", []):
        yield from _line_strings(part)


def _entry_cut(ray: LineString, polygon: Polygon, contact: Point) -> LineString | None:
    """Return the first interior segment entered from contact."""
    cut = min(
        _line_strings(ray.intersection(polygon)),
        key=lambda line: line.distance(contact),
        default=None,
    )

    if cut is not None:
        coords = list(cut.coords)
        if Point(coords[-1]).distance(contact) < Point(coords[0]).distance(contact):
            cut = LineString(coords[::-1])

    return cut


def _polygon_span(polygon: Polygon) -> float:
    """Return a conservative span length for extending a ray."""
    minx, miny, maxx, maxy = polygon.bounds
    return hypot(maxx - minx, maxy - miny)


def _inward_ray(
    point: Point, polygon: Polygon, inner_distance: float
) -> tuple[LineString, Point, Point]:
    """Build a straight ray from the nearest contact point into the polygon."""
    contact = nearest_points(point, polygon)[1]
    fallback = polygon.representative_point()

    dx = contact.x - point.x
    dy = contact.y - point.y
    length = hypot(dx, dy)

    if length == 0:
        dx = fallback.x - contact.x
        dy = fallback.y - contact.y
        length = hypot(dx, dy)

    if length == 0:
        return LineString([contact, fallback]), contact, fallback

    extension = _polygon_span(polygon) + inner_distance
    end = Point(
        contact.x + extension * dx / length, contact.y + extension * dy / length
    )

    return LineString([contact, end]), contact, fallback


def _point_on_cut(cut: LineString, inner_distance: float) -> Point:
    """Return inner-distance point, or midpoint if the cut is too short."""
    distance = inner_distance if inner_distance < cut.length else cut.length / 2
    return cut.interpolate(distance)


def _place_slightly_inside(
    point: Point, polygon: PolygonLike, inner_distance: float
) -> Point:
    """Place point just inside polygon using the nearest inward cut."""
    polygon = _nearest_part(point, polygon)
    ray, contact, fallback = _inward_ray(point, polygon, inner_distance)

    cut = _entry_cut(ray, polygon, contact)
    if cut is None:
        cut = _entry_cut(LineString([contact, fallback]), polygon, contact)

    placed = _point_on_cut(cut, inner_distance) if cut is not None else fallback
    return placed if placed.within(polygon) else fallback


def _raise_country_overlaps(overlaps: gpd.GeoDataFrame) -> None:
    """Raise an error for powerplants assigned to multiple countries."""
    duplicate_ids = overlaps["powerplant_id"].drop_duplicates()
    raise ValueError(
        "Found powerplants intersecting multiple countries: "
        f"{', '.join(map(str, duplicate_ids))}. "
        "Please adjust your shapes, or enable the 'split_capacity' correction."
    )


def _split_shape_overlaps(
    overlaps: gpd.GeoDataFrame,
    placement_shapes: gpd.GeoDataFrame,
    inner_distance: float,
) -> gpd.GeoDataFrame:
    """Split overlapping rows and move them into precomputed exclusive shapes."""
    overlaps = overlaps.copy()

    counts = overlaps["powerplant_id"].map(overlaps["powerplant_id"].value_counts())
    overlaps["output_capacity_mw"] /= counts

    placement_geometry = placement_shapes.set_index("shape_id")["geometry"]
    overlaps["geometry"] = overlaps.apply(
        lambda row: _place_slightly_inside(
            row.geometry, placement_geometry.loc[row["shape_id"]], inner_distance
        ),
        axis="columns",
    )

    suffixes = overlaps.groupby("powerplant_id").cumcount().astype(str)
    overlaps["powerplant_id"] += "_duplicate_" + suffixes

    return overlaps


def assign_country_id(
    prepared_powerplants: gpd.GeoDataFrame,
    shapes: gpd.GeoDataFrame,
    exclusive_shapes: gpd.GeoDataFrame,
    projected_crs: CRS,
    inner_distance: float,
    overlap_method: CountryOverlapMethod,
) -> CountryAssignmentResult:
    """Assign country_id, dropping out-of-scope plants and handling shape overlaps."""
    output_columns = [*prepared_powerplants.columns, "country_id"]
    shapes = shapes.to_crs(projected_crs)[["shape_id", "country_id", "geometry"]]
    exclusive_shapes = exclusive_shapes[["shape_id", "geometry"]]
    split_powerplant_ids = pd.Index([], dtype="object", name="powerplant_id")

    assigned = (
        gpd.sjoin(
            prepared_powerplants.to_crs(projected_crs),
            shapes,
            predicate="intersects",
            how="inner",
        )
        .drop(columns="index_right")
        .reset_index(drop=True)
    )

    overlaps = assigned[assigned["powerplant_id"].duplicated(keep=False)]
    if not overlaps.empty:
        if overlap_method == "raise":
            _raise_country_overlaps(overlaps)
        elif overlap_method == "split_capacity":
            split_rows = _split_shape_overlaps(
                overlaps=overlaps,
                placement_shapes=exclusive_shapes,
                inner_distance=inner_distance,
            )
            assigned.loc[overlaps.index] = split_rows
            split_powerplant_ids = pd.Index(
                split_rows["powerplant_id"], name="powerplant_id"
            )
        else:
            raise ValueError(f"Invalid overlap method {overlap_method!r}.")

    return CountryAssignmentResult(
        powerplants=assigned.to_crs(prepared_powerplants.crs)[output_columns],
        split_powerplant_ids=split_powerplant_ids,
    )


def _has_target_shape(
    candidates: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame
) -> pd.Series:
    """Return whether each candidate has a matching shape class in-country."""
    available = pd.MultiIndex.from_frame(
        shapes[["country_id", "shape_class"]].drop_duplicates()
    )
    requested = pd.MultiIndex.from_frame(
        candidates[["country_id", "target_shape_class"]]
    )

    return pd.Series(requested.isin(available), index=candidates.index)


def _raise_missing_shapes(missing: gpd.GeoDataFrame) -> None:
    """Raise a compact error describing missing country/class combinations."""
    summary = (
        missing[["technology", "country_id", "target_shape_class"]]
        .value_counts()
        .rename("count")
        .reset_index()
    )
    details = "; ".join(
        f"technology={row.technology!r}, country_id={row.country_id!r}, "
        f"shape_class={row.target_shape_class!r} ({row['count']} plants)"
        for _, row in summary.iterrows()
    )
    raise ValueError(f"No matching shapes found: {details}.")


def _already_correct_index(
    candidates: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame
) -> pd.Index:
    """Return candidate rows already inside a valid target shape."""
    if candidates.empty:
        return pd.Index([])

    shape_lookup = shapes[["shape_id", "country_id", "shape_class", "geometry"]].rename(
        columns={"country_id": "shape_country_id", "shape_class": "matched_shape_class"}
    )
    inside = gpd.sjoin(candidates, shape_lookup, how="inner", predicate="within")
    inside = inside[
        (inside["country_id"] == inside["shape_country_id"])
        & (inside["target_shape_class"] == inside["matched_shape_class"])
    ]

    return inside.index.unique()


def _adjust_group(
    group: gpd.GeoDataFrame, eligible_shapes: gpd.GeoDataFrame, inner_distance: float
) -> gpd.GeoDataFrame:
    """Adjust one country/class group to each point's nearest eligible shape."""
    matched = gpd.sjoin_nearest(
        group,
        eligible_shapes[["shape_id", "geometry"]],
        how="left",
        distance_col="_distance",
    )

    geometry_by_shape_id = eligible_shapes.set_index("shape_id")["geometry"]
    matched["matched_geometry"] = matched["shape_id"].map(geometry_by_shape_id)

    matched = matched.sort_values(["powerplant_id", "_distance", "shape_id"]).loc[
        lambda df: ~df.index.duplicated(keep="first")
    ]

    matched["geometry"] = matched.apply(
        lambda row: _place_slightly_inside(
            row.geometry, row.matched_geometry, inner_distance
        ),
        axis=1,
    )

    return matched[["geometry"]]


def _adjust_candidates(
    candidates: gpd.GeoDataFrame, shapes: gpd.GeoDataFrame, inner_distance: float
) -> gpd.GeoDataFrame:
    """Adjust all misplaced candidates to nearest valid shapes."""
    adjusted_parts = []

    for (country_id, shape_class), group in candidates.groupby(
        ["country_id", "target_shape_class"], sort=False
    ):
        eligible_shapes = shapes.loc[
            (shapes["country_id"] == country_id)
            & (shapes["shape_class"] == shape_class),
            ["shape_id", "geometry"],
        ]
        adjusted_parts.append(_adjust_group(group, eligible_shapes, inner_distance))

    adjusted = (
        pd.concat(adjusted_parts)
        if adjusted_parts
        else gpd.GeoDataFrame(
            columns=["geometry"], geometry="geometry", crs=candidates.crs
        )
    )

    return gpd.GeoDataFrame(adjusted, geometry="geometry", crs=candidates.crs)


def adjust_powerplant_location(
    powerplants: gpd.GeoDataFrame,
    exclusive_shapes: gpd.GeoDataFrame,
    forced_shape_class: Mapping[str, str],
    projected_crs: CRS,
    inner_distance: float,
    on_error: OnError = "raise",
) -> LocationAdjustmentResult:
    """Adjust configured technologies to nearest valid exclusive shape in-country."""
    plants = powerplants.to_crs(projected_crs).copy()
    plants["target_shape_class"] = plants["technology"].map(forced_shape_class)

    candidates = plants[plants["target_shape_class"].notna()].copy()

    has_target = _has_target_shape(candidates, exclusive_shapes)
    missing = candidates[~has_target]

    if not missing.empty and on_error == "raise":
        _raise_missing_shapes(missing)

    drop_index = missing.index if on_error == "drop" else pd.Index([])
    valid = candidates[has_target & ~candidates.index.isin(drop_index)]

    already_correct = _already_correct_index(valid, exclusive_shapes)
    to_adjust = valid[~valid.index.isin(already_correct)]
    moved_powerplant_ids = pd.Index(to_adjust["powerplant_id"], name="powerplant_id")

    adjusted = _adjust_candidates(to_adjust, exclusive_shapes, inner_distance)

    result = plants.drop(index=drop_index)
    if not adjusted.empty:
        result.loc[adjusted.index, "geometry"] = adjusted["geometry"]

    result = result.drop(columns="target_shape_class")
    result = result[powerplants.columns]
    result = result.to_crs(powerplants.crs)

    return LocationAdjustmentResult(
        powerplants=result, moved_powerplant_ids=moved_powerplant_ids
    )


def combine_powerplants(
    file_paths: list[str], geo_crs: CRS, excluded: list[str]
) -> gpd.GeoDataFrame:
    """Combine internal category files and user files into a final dataset."""
    combined = pd.concat(
        (gpd.read_parquet(f).to_crs(geo_crs) for f in file_paths),
        ignore_index=True,
        sort=False,
    )
    combined = combined[~combined["powerplant_id"].isin(_utils.listify(excluded))]
    if not combined.empty:
        _utils.check_single_category(combined)
    return combined


def plot(
    adjustment: LocationAdjustmentResult,
    assignment: CountryAssignmentResult,
    shapes: gpd.GeoDataFrame,
    projected_crs: CRS,
):
    """Plot final powerplants, highlighting split and forced-moved plants."""
    fig, ax = plt.subplots(layout="constrained")

    adjusted = adjustment.powerplants.to_crs(projected_crs).set_index(
        "powerplant_id", drop=False
    )
    shapes = shapes.to_crs(projected_crs)

    split = adjusted.loc[adjusted.index.intersection(assignment.split_powerplant_ids)]
    moved = adjusted.loc[adjusted.index.intersection(adjustment.moved_powerplant_ids)]

    highlighted = split.index.union(moved.index)
    other = adjusted.loc[adjusted.index.difference(highlighted)]

    if not shapes.empty:
        shapes.boundary.plot(
            ax=ax, linewidth=0.5, color="black", label=f"Shapes (n={len(shapes)})"
        )

    layers = [
        (other, "Valid plants"),
        (split, "Split due to overlaps"),
        (moved, "Forced shape class"),
    ]

    for data, label in layers:
        if not data.empty:
            data.plot(
                ax=ax,
                markersize=8 if label == "Valid plants" else 28,
                marker=".",
                alpha=0.5 if label == "Valid plants" else 0.9,
                label=f"{label} (n={len(data)})",
            )

    ax.set_aspect("equal")
    ax.set_axis_off()

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(1, 1), borderaxespad=0
        )

    return fig, ax


def main() -> None:
    """Main snakemake process."""
    # Open both shape files and ensure they match
    shapes = _schemas.ShapeSchema.validate(gpd.read_parquet(snakemake.input.shapes))
    exclusive = _schemas.ShapeSchema.validate(
        gpd.read_parquet(snakemake.input.exclusive_shapes)
    )
    mismatch = set(shapes["shape_id"].unique()) ^ set(exclusive["shape_id"].unique())
    if mismatch:
        raise ValueError(
            f"Mismatched IDs between user and exclusive shapes {sorted(mismatch)}"
        )

    projected_crs = _utils.check_crs(snakemake.params.crs["projected"], "projected")
    geographic_crs = _utils.check_crs(snakemake.params.crs["geographic"], "geographic")
    if not projected_crs.equals(exclusive.crs):
        exclusive = exclusive.to_crs(projected_crs)

    location_cnf = snakemake.params.location_cnf
    inner_distance = location_cnf["inner_distance"]

    combined = combine_powerplants(
        file_paths=[snakemake.input.internal, *snakemake.input.user],
        geo_crs=geographic_crs,
        excluded=snakemake.params.excluded,
    )
    schema = _schemas.build_schema(snakemake.params.tech_mapping, "prepare")
    combined = schema.validate(combined)

    assignment = assign_country_id(
        prepared_powerplants=combined,
        shapes=shapes,
        exclusive_shapes=exclusive,
        projected_crs=projected_crs,
        inner_distance=inner_distance,
        overlap_method=location_cnf["on_overlap"],
    )
    assigned = schema.validate(assignment.powerplants)

    adjustment = adjust_powerplant_location(
        powerplants=assigned,
        exclusive_shapes=exclusive,
        forced_shape_class=location_cnf.get("shape_class", {}),
        projected_crs=projected_crs,
        inner_distance=inner_distance,
        on_error=location_cnf["on_shape_class_error"],
    )
    adjusted = schema.validate(adjustment.powerplants)
    adjusted.to_parquet(snakemake.output.relocated)

    fig, _ = plot(
        adjustment=adjustment,
        assignment=assignment,
        shapes=shapes,
        projected_crs=projected_crs,
    )
    fig.savefig(snakemake.output.plot, dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
