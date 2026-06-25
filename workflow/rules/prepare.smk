"""Rules in this file focus on parsing and cleaning data into shared schemas."""


rule prepare_hydropower:
    input:
        glohydrores_path=rules.download_glohydrores.output.path,
    output:
        output_path="<resources>/automatic/prepared/hydropower.parquet",
    log:
        "<logs>/prepare_hydropower.log",
    conda:
        "../envs/powerplants.yaml"
    params:
        technology_mapping=config["category"]["hydropower"]["technology_mapping"],
        geo_crs=internal["crs"]["geographic"],
    message:
        "Preparing hydropower powerplants using the GloHydroRes dataset."
    script:
        "../scripts/prepare_hydropower.py"


rule prepare_large_solar:
    input:
        tz_sam=rules.download_tz_sam.output.path,
        gem_gspt=rules.download_gem.output.path.format(dataset="GSPT"),
    output:
        large_solar="<resources>/automatic/prepared/large_solar.parquet",
    log:
        "<logs>/prepare_large_solar.log",
    conda:
        "../envs/powerplants.yaml"
    params:
        dc_ac_ratio=config["category"]["solar"]["dc_ac_ratio"]["utility_pv"],
        utility_pv_name=config["category"]["solar"]["technology_mapping"]["utility_pv"],
        csp_name=config["category"]["solar"]["technology_mapping"]["csp"],
        geo_crs=internal["crs"]["geographic"],
    message:
        "Preparing utility PV powerplants using the TZ-SAM and GEM-GSPT datasets."
    script:
        "../scripts/prepare_large_solar.py"


if config["category"]["wind"]["source"] == "gem":

    rule prepare_wind_gem:
        input:
            gem_gwpt=rules.download_gem.output.path.format(dataset="GWPT"),
        output:
            path="<resources>/automatic/prepared/wind.parquet",
        log:
            "<logs>/prepare_wind_gem.log",
        conda:
            "../envs/powerplants.yaml"
        params:
            tech_map=config["category"]["wind"]["technology_mapping"],
            geo_crs=internal["crs"]["geographic"],
        message:
            "Preparing wind powerplants using the Global Wind Power Tracker (GEM-GWPT) dataset."
        script:
            "../scripts/prepare_wind_gwpt.py"

elif config["category"]["wind"]["source"] == "wemi":

    rule prepare_wind_wemi:
        input:
            wemi="<wemi>",
        output:
            path="<resources>/automatic/prepared/wind.parquet",
        log:
            "<logs>/prepare_wind_wemi.log",
        conda:
            "../envs/powerplants.yaml"
        params:
            geo_crs=internal["crs"]["geographic"],
            tech_map=config["category"]["wind"]["technology_mapping"],
        message:
            "Preparing wind powerplants using the Wind Energy Market Intelligence dataset."
        script:
            "../scripts/prepare_wind_wemi.py"

else:
    raise ValueError(
        f"Incorrect wind source configuration value '{config['wind']['source']}'"
    )


rule prepare_bioenergy:
    input:
        gem_gbpt=rules.download_gem.output.path.format(dataset="GBPT"),
    output:
        plants=temp("<resources>/automatic/temp/plants_bioenergy.parquet"),
        fuels=temp("<resources>/automatic/temp/fuels_bioenergy.parquet"),
    log:
        "<logs>/prepare_bioenergy.log",
    conda:
        "../envs/powerplants.yaml"
    params:
        geo_crs=internal["crs"]["geographic"],
        fuel_settings=get_fuel_settings(),
        technology_mapping=config["category"]["bioenergy"]["technology_mapping"],
    message:
        "Preparing bioenergy powerplants using the Global Bioenergy Power Tracker (GBPT) dataset."
    script:
        "../scripts/prepare_bioenergy.py"


rule prepare_fossil:
    input:
        gem_gcpt=rules.download_gem.output.path.format(dataset="GCPT"),
        gem_gogpt=rules.download_gem.output.path.format(dataset="GOGPT"),
    output:
        og_plants=temp("<resources>/automatic/temp/plants_fossil_oil_gas.parquet"),
        og_fuels=temp("<resources>/automatic/temp/fuels_fossil_oil_gas.parquet"),
        coal_plants=temp("<resources>/automatic/temp/plants_fossil_coal.parquet"),
        coal_fuels=temp("<resources>/automatic/temp/fuels_fossil_coal.parquet"),
    log:
        "<logs>/prepare_fossil.log",
    conda:
        "../envs/powerplants.yaml"
    params:
        geo_crs=internal["crs"]["geographic"],
        fuel_settings=get_fuel_settings(),
        technology_mapping=config["category"]["fossil"]["technology_mapping"],
    message:
        "Preparing fossil powerplants using the GOGPT and GCPT datasets."
    script:
        "../scripts/prepare_fossil.py"


rule prepare_nuclear:
    input:
        gem_gnpt=rules.download_gem.output.path.format(dataset="GNPT"),
    output:
        plants="<resources>/automatic/prepared/nuclear.parquet",
    log:
        "<logs>/prepare_nuclear.log",
    conda:
        "../envs/powerplants.yaml"
    params:
        geo_crs=internal["crs"]["geographic"],
        technology_mapping=config["category"]["nuclear"]["technology_mapping"],
    message:
        "Preparing nuclear powerplants using the Global Nuclear Power Tracker (GNPT) dataset."
    script:
        "../scripts/prepare_nuclear.py"


rule prepare_geothermal:
    input:
        gem_ggpt=rules.download_gem.output.path.format(dataset="GGPT"),
    output:
        plants="<resources>/automatic/prepared/geothermal.parquet",
    log:
        "<logs>/prepare_geothermal.log",
    conda:
        "../envs/powerplants.yaml"
    params:
        geo_crs=internal["crs"]["geographic"],
        technology_mapping=config["category"]["geothermal"]["technology_mapping"],
    message:
        "Preparing geothermal powerplants using the Global Geothermal Power Tracker (GGPT) dataset."
    script:
        "../scripts/prepare_geothermal.py"


rule prepare_statistics:
    input:
        shapes="<shapes>",
        eia_bulk=rules.download_eia.output.path,
    output:
        total="<resources>/automatic/shapes/{shapes}/statistics/total_capacity.parquet",
        categories="<resources>/automatic/shapes/{shapes}/statistics/category_capacity.parquet",
        plot="<resources>/automatic/shapes/{shapes}/statistics/category_capacity.pdf",
    log:
        "<logs>/{shapes}/prepare_statistics.log",
    conda:
        "../envs/powerplants.yaml"
    message:
        "Get EIA annual country capacity statistics."
    script:
        "../scripts/prepare_statistics.py"


rule prepare_fuel_classes:
    input:
        category_fuels=[
            rules.prepare_fossil.output.og_fuels,
            rules.prepare_fossil.output.coal_fuels,
            rules.prepare_bioenergy.output.fuels,
        ],
    output:
        fuel_classes="<results>/fuel_classes.parquet",
    log:
        "<logs>/prepare_fuels.log",
    conda:
        "../envs/powerplants.yaml"
    message:
        "Get a harmonised dataset of fuel class combinations."
    script:
        "../scripts/prepare_fuel_classes.py"


rule remap_fuel_classes:
    input:
        plants=lambda wc: get_files_to_remap(wc.category, "plants"),
        old_classes=lambda wc: get_files_to_remap(wc.category, "fuels"),
        new_classes=rules.prepare_fuel_classes.output.fuel_classes,
    output:
        remapped="<resources>/automatic/prepared/{category}.parquet",
    log:
        "<logs>/{category}/remap_fuel_classes.log",
    wildcard_constraints:
        category="|".join(COMBINED_FUEL_CAT),
    conda:
        "../envs/powerplants.yaml"
    message:
        "Remap fuel classes of combustion plants to harmonised ones."
    script:
        "../scripts/remap_fuel_classes.py"


rule prepare_shapes:
    input:
        shapes="<shapes>",
    output:
        exclusive="<resources>/automatic/shapes/{shapes}/exclusive_shapes.parquet",
    log:
        "<logs>/{shapes}/prepare_shapes.log",
    conda:
        "../envs/powerplants.yaml"
    params:
        crs=config["crs"]["projected"],
    message:
        "Prepare checks to ensure exclusive area."
    script:
        "../scripts/prepare_shapes.py"
