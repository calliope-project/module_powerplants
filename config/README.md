We recommend consulting the following before using this module:
- `config/config.yaml`: a generic example configuration of this module.
- `workflow/internal/config.schema.yaml`: a schematic overview of all the configuration options of this module.
- `INTERFACE.yaml`: lists module input and output files, and their default locations.
- `tests/integration/Snakefile`: an example of how to call this module from another workflow.

## Configuration overview

The main configuration groups are:

- `crs.projected`: projected coordinate reference system to use for distance and area operations.
Adapt this to your [region of interest](https://epsg.io/) to get accurate area estimates.
- `category`:
    - `[CATEGORY_NAME].technology_mapping`: rename / regroup source technology labels to the names used in module outputs.
    - `[CATEGORY_NAME].excluded_ids`: drop specific powerplants during processing.
    Useful if you wish to correct powerplant data via `<imputed_powerplants>` files.
    - `wind.source`: selects either the open [GEM wind dataset](https://globalenergymonitor.org/projects/global-wind-power-tracker) (`gem`) or a user-provided [WEMI file](https://www.thewindpower.net/index.php) (`wemi`).
    - `solar.dc_ac_ratio`: converts utility PV capacity from DC to AC where needed.
    We recommend using the 1.25 default.
- `fuel_mapping`: optional overrides for combustion fuel names.
- `imputation`:
    - `location`: controls shape overlaps and technology-to-`shape_class` handling.
    - `time`: controls future installations scenarios, technology lifetimes, and retirement delays.

This data module is part of the [Modelblocks](https://www.modelblocks.org/) project.
Please consult the [Modelblocks documentation](https://modelblocks.readthedocs.io/) for more details.
