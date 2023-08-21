# we calculate splice variants for all genes for each cell (each cell gets different variants
rule create_dataset:
    input:
        adata=config['adata_folder'] + config['adata_name']
    params:
        radius=config['radius'],
        k_hop=config['k_hop'],
        pre_basename=config['output_folder'] + config['adata_name'] + "/pretrain_datasets/dataset_{k_hop}_{radius}_",
        train_basename=config['output_folder'] + config['adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_",
    output:
        trainset=config['output_folder'] + config[
            'adata_name'] + "/pretrain_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset=config['output_folder'] + config['adata_name'] + "/pretrain_datasets/dataset_{k_hop}_{radius}_val.pt",
        testset=config['output_folder'] + config['adata_name'] + "/pretrain_datasets/dataset_{k_hop}_{radius}_test.pt",
        down_trainset=config['output_folder'] + config[
            'adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_train.pt",
        down_valset=config['output_folder'] + config['adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_val.pt",
        down_testset=config['output_folder'] + config[
            'adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_test.pt",
    cluster:
        "sbatch --job-name {rule}"
    shell:
        """
        python scripts/create_dataset.py {input.adata} {wildcards.radius} {wildcards.k_hop} {params.pre_basename} {params.train_basename}
        """

# we train the model on the dataset
rule pretraining:
    input:
        trainset=config['output_folder'] + config[
            'adata_name'] + "/pretrain_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset=config['output_folder'] + config['adata_name'] + "/pretrain_datasets/dataset_{k_hop}_{radius}_val.pt",
        adata=config['adata_folder'] + config['adata_name']
    params:
        pretrain_patience=config['pretrain_patience'],
        pretrain_batch_size=config['pretrain_batch_size'],
        pretrain_lr=config['pretrain_lr'],
        pretrain_epochs=config['pretrain_epochs'],
    #       pretrain_bottleneck=config['bottleneck'],
    output:
        model=config['output_folder'] + config[
            'adata_name'] + "/pretrain_models/model_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.pt",
        summary_pdf=config['output_folder'] + config[
            'adata_name'] + "/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.pdf",
        summary_txt=config['output_folder'] + config[
            'adata_name'] + "/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.csv",
    shell:
        """
        python scripts/pretrain_model.py {input.trainset} {input.valset} {output.model} {wildcards.masking_mode} {params} {wildcards.neck} {input.adata} {output.summary_pdf} {output.summary_txt} {wildcards.pretrained} {wildcards.layers}
        """

rule combine_results_pretrain:
    input:
        files=expand(config['output_folder'] + config[
            'adata_name'] + "/pretrain_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.csv",
            k_hop=config['k_hop'],
            radius=config['radius'],
            masking_mode=config['masking_mode'],
            pretrained=config['pretrain_structure'],
            neck=config['bottleneck'],
            layers=config['num_hidden_layers']),
    params:
        folder=config['output_folder'] + config['adata_name'] + "/pretrain_models"
    output:
        config['output_folder'] + config['adata_name'] + "/pretrain_models/summary.pdf"
    shell:
        """
        python scripts/combine_results.py {params.folder} {output}
        """

rule train:
    input:
        trainset=config['output_folder'] + config['adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset=config['output_folder'] + config['adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_val.pt",
        adata=config['adata_folder'] + config['adata_name'],
        model=config['output_folder'] + config[
            'adata_name'] + "/pretrain_models/model_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.pt"
    params:
        train_patience=config['train_patience'],
        train_batch_size=config['train_batch_size'],
        train_lr=config['train_lr'],
        train_epochs=config['train_epochs'],
    output:
        model=config['output_folder'] + config[
            'adata_name'] + "/train_models/model_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.pt",
        summary_pdf=config['output_folder'] + config[
            'adata_name'] + "/train_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.pdf",
        summary_txt=config['output_folder'] + config[
            'adata_name'] + "/train_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.csv",
    shell:
        """
        python scripts/train_model.py {input.trainset} {input.valset} {output.model} {params} {input.adata} {output.summary_pdf} {output.summary_txt} None {input.model} 
        """

rule train_no_pre:
    input:
        trainset=config['output_folder'] + config['adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_train.pt",
        valset=config['output_folder'] + config['adata_name'] + "/train_datasets/dataset_{k_hop}_{radius}_val.pt",
        adata=config['adata_folder'] + config['adata_name'],
    params:
        train_patience=config['train_patience'],
        train_batch_size=config['train_batch_size'],
        train_lr=config['train_lr'],
        train_epochs=config['train_epochs'],
    output:
        model=config['output_folder'] + config['adata_name'] + "/train_models/model_{k_hop}_{radius}_n{neck}_m{model}_lay{layers}_no_pre.pt",
        summary_pdf=config['output_folder'] + config[
            'adata_name'] + "/train_models/summary_{k_hop}_{radius}_n{neck}_m{model}_lay{layers}_no_pre.pdf",
        summary_txt=config['output_folder'] + config[
            'adata_name'] + "/train_models/summary_{k_hop}_{radius}_n{neck}_m{model}_lay{layers}_no_pre.csv"
    shell:
        """
        python scripts/train_model.py {input.trainset} {input.valset} {output.model} {params} {input.adata} {output.summary_pdf} {output.summary_txt} {wildcards.model} None {wildcards.neck} {wildcards.layers}
        """


rule combine_results_train:
    input:
        files=expand(config['output_folder'] + config[
            'adata_name'] + "/train_models/summary_{k_hop}_{radius}_{masking_mode}_pre_{pretrained}_lay{layers}_n{neck}.csv",
            k_hop=config['k_hop'],
            radius=config['radius'],
            masking_mode=config['masking_mode'],
            pretrained=config['pretrain_structure'],
            neck=config['bottleneck'],
            layers=config['num_hidden_layers']),
        no_pre=expand(config['output_folder'] + config[
            'adata_name'] + "/train_models/summary_{k_hop}_{radius}_n{neck}_m{model}_lay{layers}_no_pre.csv",
            k_hop=config['k_hop'],
            radius=config['radius'],
            neck=config['bottleneck'],
            model=config['pretrain_structure'],
            layers=config['num_hidden_layers']),
    params:
        folder=config['output_folder'] + config['adata_name'] + "/train_models"
    output:
        config['output_folder'] + config['adata_name'] + "/train_models/summary.pdf"
    shell:
        """
        python scripts/combine_results.py {params.folder} {output}
        """
