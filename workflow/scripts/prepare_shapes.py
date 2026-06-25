"""Precompute non-overlapping / exclusive shapes.

For each input shape, subtract all other shapes that intersect it to ensure exclusivity.
"""

import sys
from typing import TYPE_CHECKING, Any

import _schemas
import _utils
import geopandas as gpd
from shapely import make_valid, union_all
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    snakemake: Any


type PolygonLike = Polygon | MultiPolygon


def _polygonal_geometry(geometry: BaseGeometry) -> PolygonLike | None:
    """Return valid polygonal geometry, or None if no polygonal area remains."""
    result = None

    if not geometry.is_empty:
        if not geometry.is_valid:
            geometry = make_valid(geometry)

        if isinstance(geometry, Polygon | MultiPolygon):
            result = geometry
        else:
            polygons = [
                part
                for part in getattr(geometry, "geoms", [])
                if isinstance(part, Polygon)
            ]

            if polygons:
                merged = union_all(polygons)
                if isinstance(merged, Polygon | MultiPolygon):
                    result = merged

    return result


def build_exclusive_shapes(shapes: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Subtract intersecting neighbor shapes from each shape.

    Raises if any shape_id has no exclusive polygonal area left.
    """
    shapes = shapes.reset_index(drop=True)
    sindex = shapes.sindex

    exclusive_geometries: list[PolygonLike] = []
    removed_shape_ids: list[str] = []

    for i, shape in shapes.iterrows():
        neighbor_positions = [
            j for j in sindex.query(shape.geometry, predicate="intersects") if j != i
        ]

        exclusive = (
            shape.geometry.difference(
                union_all(shapes.geometry.iloc[neighbor_positions].to_list())
            )
            if neighbor_positions
            else shape.geometry
        )
        exclusive = _polygonal_geometry(exclusive)

        if exclusive is None:
            removed_shape_ids.append(shape["shape_id"])
            continue

        exclusive_geometries.append(exclusive)

    if removed_shape_ids:
        raise ValueError(
            "The following shapes have no exclusive placement area after removing "
            "overlaps: "
            f"{', '.join(map(str, removed_shape_ids))}. "
            "This indicates overlapping input shapes that fully cover one or more "
            "shape_id geometries."
        )

    shapes["geometry"] = exclusive_geometries
    return shapes[["shape_id", "country_id", "shape_class", "geometry"]]


def main() -> None:
    """Main Snakemake process."""
    projected_crs = _utils.check_crs(snakemake.params.crs, "projected")

    shapes = _schemas.ShapeSchema.validate(gpd.read_parquet(snakemake.input.shapes))
    shapes = shapes.to_crs(projected_crs)

    exclusive_shapes = build_exclusive_shapes(shapes)
    exclusive_shapes = _schemas.ShapeSchema.validate(exclusive_shapes)

    exclusive_shapes.to_parquet(snakemake.output.exclusive)


if __name__ == "__main__":
    sys.stderr = open(snakemake.log[0], "w", buffering=1)
    main()
