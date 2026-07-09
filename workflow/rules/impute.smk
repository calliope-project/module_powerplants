"""Rules related to internal and user-provided imputations."""


rule impute_location:
    input:
        internal="<resources>/automatic/prepared/{category}.parquet",
        exclusive_shapes=branch(
            config["imputation"]["location"]["remove_shape_overlaps"],
            then=rules.prepare_shapes.output.exclusive,
            otherwise="<shapes>",
        ),
        shapes="<shapes>",
        user=branch(
            exists("<imputed_powerplants>"),
            then=["<imputed_powerplants>"],
            otherwise=[],
        ),
    output:
        relocated=temp(
            "<resources>/automatic/shapes/{shapes}/impute_location/{category}.parquet"
        ),
        plot=report(
            "<resources>/automatic/shapes/{shapes}/impute_location/{category}_relocation.png",
            caption="../report/impute_location.rst",
            category="Powerplants module",
            subcategory="{category}",
        ),
    log:
        "<logs>/{shapes}/{category}/impute_location.log",
    wildcard_constraints:
        category="|".join(IMPUTED_CAT),
    conda:
        "../envs/module.yaml"
    params:
        crs=internal["crs"] | config["crs"],
        location_cnf=config["imputation"]["location"],
        excluded=lambda wc: get_excluded_powerplant_ids(f"{wc.category}"),
        tech_mapping=lambda wc: get_technology_mapping(f"{wc.category}"),
    message:
        "Imputation of user-configured location adjustment for {wildcards.shapes}-{wildcards.category}."
    script:
        "../scripts/impute_location.py"


rule impute_time:
    input:
        relocated=rules.impute_location.output.relocated,
        category_capacity=(
            "<resources>/automatic/shapes/{shapes}/"
            "statistics/category_capacity.parquet"
        ),
    output:
        aged=workflow.pathvars.apply("<powerplants>").format(
            shapes="{shapes}",
            adjustment="unadjusted",
            category="{category}",
        ),
        histogram=report(
            "<results>/{shapes}/powerplants/unadjusted/{category}_histogram.pdf",
            caption="../report/impute_time_histogram.rst",
            category="Powerplants module",
            subcategory="{category}",
        ),
        explorer=report(
            "<results>/{shapes}/powerplants/unadjusted/{category}_explorer.html",
            caption="../report/impute_time_explorer.rst",
            category="Powerplants module",
            subcategory="{category}",
        ),
        age_imputation=report(
            "<resources>/automatic/shapes/{shapes}/impute_time/{category}_age_imputation.parquet",
            caption="../report/impute_time_age_imputation.rst",
            category="Powerplants module",
            subcategory="{category}",
        ),
        age_profile=(
            "<resources>/automatic/shapes/{shapes}/impute_time/{category}_age_profile.parquet"
        ),
        retirement_profile=(
            "<resources>/automatic/shapes/{shapes}/impute_time/{category}_retirement_profile.parquet"
        ),
        age_profile_plot=report(
            "<results>/{shapes}/powerplants/unadjusted/{category}_age_profile.pdf",
            caption="../report/impute_time_profile.rst",
            category="Powerplants module",
            subcategory="{category}",
        ),
    log:
        "<logs>/{shapes}/{category}/impute_time.log",
    wildcard_constraints:
        dataset="|".join(IMPUTED_CAT),
    conda:
        "../envs/module.yaml"
    params:
        imputation=config["imputation"]["time"],
        tech_map=lambda wc: get_technology_mapping(wc.category),
    message:
        "National-level imputation of missing powerplant ages in {wildcards.shapes}-{wildcards.category} dataset."
    script:
        "../scripts/impute_ages.py"


rule impute_capacity_adjustment:
    input:
        unadjusted=rules.impute_time.output.aged,
        stats=rules.prepare_statistics.output.categories,
    output:
        adjusted=workflow.pathvars.apply("<powerplants>").format(
            shapes="{shapes}",
            adjustment="adjusted",
            category="{category}",
        ),
        plot=report(
            "<results>/{shapes}/powerplants/adjusted/{category}_adjustment.pdf",
            caption="../report/impute_capacity_adjustment.rst",
            category="Powerplants module",
            subcategory="{category}",
        ),
    log:
        "<logs>/{shapes}/{category}/impute_capacity_adjustment.log",
    wildcard_constraints:
        category="|".join(IMPUTED_CAT - IMPUTED_CAT_WITHOUT_ADJUSTMENT),
    conda:
        "../envs/module.yaml"
    message:
        "National-level adjustment of powerplant capacity in {wildcards.shapes}-{wildcards.category} to EIA statistics."
    script:
        "../scripts/impute_capacity_adjustment.py"
