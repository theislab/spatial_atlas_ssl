configfile: "config.yaml"

rule all:
    input:
        #dataset=expand("snake_output/pretrain_models/model_{k_hop}_{radius}_{masking_mode}_{pretrained}.pt",k_hop=
        #config['k_hop'],radius=config['radius'],masking_mode=config['masking_mode'],pretrained=config[
         #   'pretrain_structure']),
        #datasett=expand("snake_output/models/model_{k_hop}_{radius}_{masking_mode}_{pretrained}.pt",k_hop=config[
          #  'k_hop'],radius=config['radius'],masking_mode=config['masking_mode'],pretrained=config[
           # 'pretrain_structure']),
        summary_pretrain=config['output_folder'] + config['adata_name'] + "/pretrain_models/summary.pdf",
        summary_train=config['output_folder'] + config['adata_name'] + "/train_models/summary.pdf"

include: config["rules"] + "rules.smk"
