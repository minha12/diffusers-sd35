# Snakefile for SD3 ControlNet training
configfile: "config/config.yaml"

# Include rules
include: "workflow/rules/preprocess.smk"
include: "workflow/rules/train.smk"
# include: "workflow/rules/validate.smk"
# include: "workflow/rules/deploy.smk"

# Default target rule
rule all:
    input:
        expand("results/models/{dataset_name}/final_model", 
               dataset_name=config["dataset_name"])

# Clean rule
rule clean:
    shell:
        "rm -rf results/models/*/checkpoints/* logs/rule_logs/*"