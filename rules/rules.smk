# we calculate splice variants for all genes for each cell (each cell gets different variants
rule create_dataset:
    input:
        adata = config['adata']
    params:
        radius = config['radius'],
        k_hop = config['k_hop'],
        basename= "snake_output/datasets/dataset_{k_hop}_{radius}_"
    output:
        trainset = "snake_output/datasets/dataset_{k_hop}_{radius}_train.pt",
        valset= "snake_output/datasets/dataset_{k_hop}_{radius}_val.pt",
        testset = "snake_output/datasets/dataset_{k_hop}_{radius}_test.pt",
    shell:
        """
        python scripts/create_dataset.py {input.adata} {wildcards.radius} {wildcards.k_hop} {params.basename}
        """

# we train the model on the dataset
rule pretraining:
    input:
        trainset = "snake_output/datasets/dataset_{k_hop}_{radius}_train.pt",
        valset = "snake_output/datasets/dataset_{k_hop}_{radius}_val.pt",
        adata = config['adata']
    params:
        pretrain_patience = config['pretrain_patience'],
        pretrain_batch_size = config['pretrain_batch_size'],
        pretrain_lr = config['pretrain_lr'],
        pretrain_epochs = config['pretrain_epochs'],
    output:
        model = "snake_output/pretrain_models/model_{k_hop}_{radius}_{masking_mode}_{pretrained}.pt",
        summary_pdf = "snake_output/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_{pretrained}.pdf",
        summary_txt= "snake_output/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_{pretrained}.csv",
    shell:
        """
        python scripts/pretrain_model.py {input.trainset} {input.valset} {output.model} {wildcards.masking_mode} {params} {input.adata} {output.summary_pdf} {output.summary_txt}
        """

rule combine_results:
    input:
        files = expand("snake_output/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_{pretrained}.csv",k_hop=config[
    'k_hop'],radius=config['radius'],masking_mode=config['masking_mode'],pretrained=config['pretrain_structure'])
    params:
        folder = "snake_output/pretrain_models"
    output:
        "snake_output/pretrain_models/summary.pdf"
    shell:
        """
        python scripts/combine_results.py {params.folder} {output}
        """