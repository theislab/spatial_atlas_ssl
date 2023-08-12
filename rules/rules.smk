# we calculate splice variants for all genes for each cell (each cell gets different variants
rule create_dataset:
    input:
        adata=config['adata']
    params:
        radius=config['radius'],
        k_hop=config['k_hop'],
        pre_basename="snake_output/pretrain_datasets/dataset_{k_hop}_{radius}_",
        train_basename="snake_output/train_datasets/dataset_{k_hop}_{radius}_",
    output:
        trainset="snake_output/pretrain_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset="snake_output/pretrain_datasets/dataset_{k_hop}_{radius}_val.pt",
        testset="snake_output/pretrain_datasets/dataset_{k_hop}_{radius}_test.pt",
        down_trainset="snake_output/train_datasets/dataset_{k_hop}_{radius}_train.pt",
        down_valset="snake_output/train_datasets/dataset_{k_hop}_{radius}_val.pt",
        down_testset="snake_output/train_datasets/dataset_{k_hop}_{radius}_test.pt",
    shell:
        """
        python scripts/create_dataset.py {input.adata} {wildcards.radius} {wildcards.k_hop} {params.pre_basename} {params.train_basename}
        """

# we train the model on the dataset
rule pretraining:
    input:
        trainset="snake_output/pretrain_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset="snake_output/pretrain_datasets/dataset_{k_hop}_{radius}_val.pt",
        adata=config['adata']
    params:
        pretrain_patience=config['pretrain_patience'],
        pretrain_batch_size=config['pretrain_batch_size'],
        pretrain_lr=config['pretrain_lr'],
        pretrain_epochs=config['pretrain_epochs'],
    output:
        model="snake_output/pretrain_models/model_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.pt",
        summary_pdf="snake_output/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.pdf",
        summary_txt="snake_output/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.csv",
    shell:
        """
        python scripts/pretrain_model.py {input.trainset} {input.valset} {output.model} {wildcards.masking_mode} {params} {input.adata} {output.summary_pdf} {output.summary_txt}
        """

rule combine_results_pretrain:
    input:
        files=expand("snake_output/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.csv",
            k_hop=config['k_hop'],radius=config['radius'],
            masking_mode=config['masking_mode'],
            pretrained=config['pretrain_structure'])
    params:
        folder="snake_output/pretrain_models"
    output:
        "snake_output/pretrain_models/summary.pdf"
    shell:
        """
        python scripts/combine_results.py {params.folder} {output}
        """

rule train:
    input:
        trainset="snake_output/train_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset="snake_output/train_datasets/dataset_{k_hop}_{radius}_val.pt",
        adata=config['adata'],
        model="snake_output/pretrain_models/model_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.pt"
    params:
        train_patience=config['train_patience'],
        train_batch_size=config['train_batch_size'],
        train_lr=config['train_lr'],
        train_epochs=config['train_epochs'],
    output:
        model="snake_output/train_models/model_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.pt",
        summary_pdf="snake_output/train_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.pdf",
        summary_txt="snake_output/train_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.csv",
    shell:
        """
        python scripts/train_model.py {input.trainset} {input.valset} {output.model} {params} {input.adata} {output.summary_pdf} {output.summary_txt} {input.model}
        """

rule train_no_pre:
    input:
        trainset="snake_output/train_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset="snake_output/train_datasets/dataset_{k_hop}_{radius}_val.pt",
        adata=config['adata'],
    params:
        train_patience=config['train_patience'],
        train_batch_size=config['train_batch_size'],
        train_lr=config['train_lr'],
        train_epochs=config['train_epochs'],
    output:
        model="snake_output/train_models/model_{k_hop}_{radius}_no_pre.pt",
        summary_pdf="snake_output/train_models/summary_{k_hop}_{radius}_no_pre.pdf",
        summary_txt="snake_output/train_models/summary_{k_hop}_{radius}_no_pre.csv",
    shell:
        """
        python scripts/train_model.py {input.trainset} {input.valset} {output.model} {params} {input.adata} {output.summary_pdf} {output.summary_txt} None
        """


rule combine_results_train:
    input:
        files=expand("snake_output/train_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}.csv",
            k_hop=config['k_hop'],
            radius=config['radius'],
            masking_mode=config['masking_mode'],
            pretrained=config['pretrain_structure']),
        no_pre=expand("snake_output/train_models/summary_{k_hop}_{radius}_no_pre.csv",
            k_hop=config['k_hop'],
            radius=config['radius'])
    params:
        folder="snake_output/train_models"
    output:
        "snake_output/train_models/summary.pdf"
    shell:
        """
        python scripts/combine_results.py {params.folder} {output}
        """
