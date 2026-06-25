"""Collection of auxiliary functions for this module."""

# Options for both powerplants and aggregated capacities
ADJUSTMENTS = ("adjusted", "unadjusted")
# Names after original database processing
PREPARED_FUEL_CAT = ("bioenergy", "fossil_coal", "fossil_oil_gas")
PREPARED_PLANT_CAT = (
    "bioenergy",
    "fossil_coal",
    "fossil_oil_gas",
    "geothermal",
    "hydropower",
    "nuclear",
    "large_solar",
    "wind",
)
# Fossil categories are harmonised during fuel processing
COMBINED_FUEL_CAT = ("bioenergy", "fossil")
# Final categories seen by the user
IMPUTED_CAT = {
    "bioenergy",
    "fossil",
    "geothermal",
    "hydropower",
    "nuclear",
    "large_solar",
    "wind",
}
# Intermediate categories that require proxies for aggregation.
IMPUTED_CAT_WITHOUT_ADJUSTMENT = {"large_solar"}


def additional_config_validation():
    """Ensures technology mapping and lifetime-related names match."""
    lifetime_set = set(config["imputation"]["time"]["lifetime_years"].keys())
    mismatch = lifetime_set ^ set(
        config["imputation"]["time"]["retirement_delay_years"]
    )
    if mismatch:
        raise ValueError(
            f"Technologies in lifetimes and retirement delays mismatch: {mismatch}."
        )
    # Check category technology names
    tech_map_set = set()
    for cat in ["bioenergy", "geothermal", "hydropower", "nuclear", "solar", "wind"]:
        tech_map_set |= set(config["category"][cat]["technology_mapping"].values())
    for fossil_cat in ["coal", "oil_gas"]:
        tech_map_set |= set(
            config["category"]["fossil"]["technology_mapping"][fossil_cat].values()
        )
    mismatch = lifetime_set ^ tech_map_set
    if mismatch:
        raise ValueError(
            f"Technology mapping does not match lifetime technologies for {mismatch}"
        )


def get_excluded_powerplant_ids(category):
    """Handle cases where the naming between files and configuration mismatch."""
    if category == "large_solar":
        config_cat = "solar"
    else:
        config_cat = category
    return config["category"][config_cat].get("excluded_ids", [])


def get_technology_mapping(filename: str):
    if "solar" in filename:
        mapping = config["category"]["solar"]["technology_mapping"]
    elif "fossil" in filename:
        tech = filename.removeprefix("fossil_").removesuffix(".parquet")
        mapping = dict()
        for tech in ["coal", "oil_gas"]:
            mapping |= config["category"]["fossil"]["technology_mapping"][tech]
    else:
        mapping = config["category"][filename]["technology_mapping"]
    return mapping


def get_files_to_remap(category: str, prefix: str):
    """Produce a list of fuel category files to combine."""
    to_combine = []
    if category == "fossil":
        to_combine += [
            f"<resources>/automatic/temp/{prefix}_fossil_coal.parquet",
            f"<resources>/automatic/temp/{prefix}_fossil_oil_gas.parquet",
        ]
    else:
        to_combine.append(f"<resources>/automatic/temp/{prefix}_{category}.parquet")

    return to_combine

def get_fuel_settings() -> dict:
    """Standardised access to fuel configuration."""
    combined = internal["fuels"]
    combined["mapping"] = combined["mapping"] | config.get("fuel_mapping", {})
    return combined
