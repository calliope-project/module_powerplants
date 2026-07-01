"""Rules to used to download automatic resource files."""


rule download_eia:
    output:
        path="<resources>/automatic/downloads/EIA-INTL.txt",
    log:
        "<logs>/download_eia.log",
    conda:
        "../envs/module.yaml"
    params:
        retry=internal["curl"]["retry"],
        url=internal["resources"]["automatic"]["EIA"],
    message:
        "Download the EIA International energy statistics in bulk."
    shell:
        """
        curl -fsSL --retry {params.retry:q} -o {output.path:q} {params.url:q}
        """


rule download_tz_sam:
    output:
        path="<resources>/automatic/downloads/TZ-SAM.gpkg",
    log:
        "<logs>/download_tz_sam.log",
    conda:
        "../envs/module.yaml"
    params:
        retry=internal["curl"]["retry"],
        url=internal["resources"]["automatic"]["TZ-SAM"],
    message:
        "Download the Transition Zero - Solar Asset Mapper dataset."
    shell:
        """
        curl -fsSL --retry {params.retry:q} -o {output.path:q} {params.url:q}
        """


rule download_glohydrores:
    output:
        path="<resources>/automatic/downloads/GloHydroRes.csv",
    log:
        "<logs>/download_glohydrores.log",
    conda:
        "../envs/module.yaml"
    params:
        retry=internal["curl"]["retry"],
        url=internal["resources"]["automatic"]["GloHydroRes"],
    message:
        "Download the GloHydroRes dataset."
    shell:
        """
        curl -fsSL --retry {params.retry:q} -o {output.path:q} {params.url:q}
        """


rule download_gem:
    output:
        path="<resources>/automatic/downloads/GEM_{dataset}.xlsx",
    log:
        "<logs>/download_gem_{dataset}.log",
    conda:
        "../envs/module.yaml"
    params:
        retry=internal["curl"]["retry"],
        url=lambda wc: internal["resources"]["automatic"]["GEM"][wc.dataset],
    message:
        "Download the GEM {wildcards.dataset} dataset."
    shell:
        """
        curl -fsSL --retry {params.retry:q} -o {output.path:q} {params.url:q}
        """
