configfile: "config.yaml"

rule all:
    input:
        summary_pretrain=config['output_folder'] + config['adata_name'] + "/pretrain_models/summary.pdf",
        summary_train=config['output_folder'] + config['adata_name'] + "/train_models/summary.pdf"

include: config["rules"] + "rules.smk"
