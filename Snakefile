configfile: "config.yaml"

rule all:
    input:
        summary_pretrain=config['output_folder'] + config['adata_name'] + "/pretrain_models/summary.pdf",
        summary_train=config['output_folder'] + config['adata_name'] + "/train_models/summary.pdf",
        model_eval= config['output_folder'] + config['adata_name'] + "/train_models/eval_models.csv",
        bar = config['output_folder'] + config['adata_name'] + "/plots/barplot_celltype_dist.png",
        heat = config['output_folder'] + config['adata_name'] + "/plots/heatmap_celltype_dist.png",
        hist = config['output_folder'] + config['adata_name'] + "/plots/histogram_unique_celltype_dist.png"

include: config["rules"] + "rules.smk"
