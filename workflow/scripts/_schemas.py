"""Schemas for key files."""

from typing import Literal

import _utils
import pandas as pd
import pandera.geopandas as gpa
import pandera.pandas as pa
from pandera.typing.geopandas import GeoSeries
from pandera.typing.pandas import Index, Series
from shapely.geometry import Point

# A diverse set of statuses to diminish oversimplification during gap filling.
OPERATING = "operating"
RETIRED = "retired"
HISTORICAL = {OPERATING, RETIRED}
PLANNED = {"construction", "pre-construction", "announced"}
SCENARIO_MAP = {
    "historical": HISTORICAL,
    "construction": HISTORICAL | {"construction"},
    "pre_construction": HISTORICAL | {"construction", "pre-construction"},
    "announced": HISTORICAL | PLANNED,
}
# Status categorisation shown to users (and accepted in user imputed files).
IMPUTED_STATUS = {"planned", "operating", "retired"}


class EIASchema(pa.DataFrameModel):
    class Config:
        coerce = True
        strict = True

    year: Series[int]
    "Sample year"
    category: Series[str]
    "Human readable name"
    capacity_mw: Series[float] = pa.Field(ge=0, nullable=True)
    "Electrical capacity in Megawatt"
    country_id: Series[str]
    "Country ISO-3 code"


class ShapeSchema(gpa.GeoDataFrameModel):
    class Config:
        coerce = True
        strict = False

    shape_id: Series[str] = gpa.Field(unique=True)
    "Unique ID for this shape."
    country_id: Series[str]
    "ISO alpha-3 code."
    shape_class: Series[str] = gpa.Field(isin=["land", "maritime"])
    "Shape classifier"
    geometry: GeoSeries
    "Shape polygon."

    @gpa.check("geometry", element_wise=True)
    def geom_not_empty(cls, geom):
        return (geom is not None) and (not geom.is_empty) and geom.is_valid


class AggregatedPlantSchema(pa.DataFrameModel):
    class Config:
        coerce = True
        strict = True

    index: Index[int] = pa.Field(unique=True)

    shape_id: Series[str]
    country_id: Series[str]
    category: Series[str]
    technology: Series[str]
    output_capacity_mw: Series[float] = pa.Field(gt=0)
    chp: Series[bool] | None
    ccs: Series[bool] | None
    fuel_class: Series[str] | None


class PlantSchema(gpa.GeoDataFrameModel):
    class Config:
        coerce = True
        strict = True

    index: Index[int] = gpa.Field(unique=True)

    # Identifiers
    powerplant_id: Series[str] = gpa.Field(unique=True)
    "Unique ID for the powerplant."
    name: Series[str]
    "Human readable powerplant name."
    # Technology characteristics
    category: Series[str]
    "General category of the powerplant."
    technology: Series[str]
    "Subcategory of the powerplant, if necessary."
    output_capacity_mw: Series[float] = gpa.Field(gt=0)
    "Powerplant gross output capacity in Megawatts."
    # Temporal aspects
    start_year: Series[float]
    "Installation year."
    end_year: Series[float]
    "Expected decommissioning year."
    status: Series[str]
    "Known state of the project."
    start_year_source_type: Series[str] | None = gpa.Field(
        isin=_utils.date_source_types_for("start_year")
    )
    "Source/provenance label for the start year."
    end_year_source_type: Series[str] | None = gpa.Field(
        isin=_utils.date_source_types_for("end_year")
    )
    "Source/provenance label for the end year."
    # Location / size
    geometry: GeoSeries[Point] = gpa.Field()
    "Powerplant point data."
    country_id: Series[str] | None = gpa.Field(
        str_length={"min_value": 3, "max_value": 3}
    )
    # Combustion specifics
    ccs: Series[bool] | None
    """Identifier for known CCS-enabled powerplants."""
    chp: Series[bool] | None
    """Identifier for known CHP-enabled powerplants."""
    fuel_class: Series[str] | None
    """Unique ID in the fuel consumption look-up table."""
    # Hydropower specifics
    reservoir_km3: Series[float] | None = gpa.Field(nullable=True, ge=0)
    """Reservoir volume."""

    @gpa.check("geometry", element_wise=True)
    def geom_not_empty(cls, geom):
        return (geom is not None) and (not geom.is_empty) and geom.is_valid

    @gpa.dataframe_check
    def end_after_start(cls, plants: pd.DataFrame):
        """Require ordered dates wherever both years are known."""
        known_dates = plants[["start_year", "end_year"]].notna().all(axis="columns")
        return ~known_dates | plants["end_year"].gt(plants["start_year"])


class FuelSchema(pa.DataFrameModel):
    class Config:
        strict = True
        coerce = True

    fuel_class: Series[str]
    "ID of the fuel consumption class."
    fuel: Series[str]
    "Fuel consumed."


class CapacityDateEventSchema(pa.DataFrameModel):
    """Annual commissioning and retirement events used by diagnostics."""

    class Config:
        coerce = True
        strict = True

    powerplant_id: Series[str]
    name: Series[str]
    country_id: Series[str] = pa.Field(str_length={"min_value": 3, "max_value": 3})
    category: Series[str]
    technology: Series[str]
    status: Series[str] = pa.Field(isin={"planned", "operating", "retired"})
    year: Series[float]
    event_type: Series[str] = pa.Field(isin={"commissioning", "retirement"})
    source_type: Series[str] = pa.Field(isin=set(_utils.DATE_SOURCE_METADATA))
    source_label: Series[str]
    output_capacity_mw: Series[float] = pa.Field(gt=0)
    capacity_change_mw: Series[float] = pa.Field(ne=0)


def build_schema(
    tech_mapping: dict[str, str], stage: Literal["prepare", "impute"]
) -> pa.DataFrameSchema:
    """Construct an inflexible schema applicable to each processing stage."""
    schema = PlantSchema.to_schema()
    if stage == "prepare":
        status_set = SCENARIO_MAP["announced"]
        # Years can be empty during preparation stages
        year_overrides: list[tuple[str, dict]] = [
            ("start_year", {"nullable": True}),
            ("end_year", {"nullable": True}),
        ]
    elif stage == "impute":
        status_set = IMPUTED_STATUS
        year_overrides = []
    else:
        raise ValueError(f"Incorrect stage given: '{stage}'.")

    techs = set(tech_mapping.values())
    overrides = [
        *year_overrides,
        ("technology", {"checks": pa.Check.isin(techs)}),
        ("status", {"checks": pa.Check.isin(status_set)}),
    ]
    for col, kwargs in overrides:
        schema = schema.update_column(col, **kwargs)

    return schema
