# Rules for model training

rule train_model:
    input:
        processed_dir = "data/processed/{dataset_name}",
        cache_dir = "data/cache/{dataset_name}"
    output:
        model_dir = directory("results/models/{dataset_name}/sd3.5-controlnet-out-{dataset_name}"),
        checkpoint = "results/models/{dataset_name}/sd3.5-controlnet-out-{dataset_name}/checkpoint-{max_steps}"
    params:
        model_dir = config["pretrained_model_dir"],
        script_path = "workflow/scripts/train.py",
        accelerate_config = "config/accelerate_config.yaml",
        script_params = lambda wildcards, input: (
            f"--pretrained_model_name_or_path={config['pretrained_model_dir']} "
            f"--output_dir={wildcards.output_dir} "
            f"--train_data_dir={input.processed_dir} "
            f"--dataset_cache_dir={input.cache_dir} "
            f"--dataset_script_path={config['dataset_script_path'].format(dataset_name=wildcards.dataset_name)} "
            f"--resolution={config['resolution']} "
            f"--learning_rate={config['learning_rate']} "
            f"--dataset_preprocess_batch_size={config['dataset_preprocess_batch_size']} "
            f"--max_train_steps={wildcards.max_steps} "
            f"--train_batch_size={config['train_batch_size']} "
            f"--gradient_accumulation_steps={config['gradient_accumulation_steps']} "
            f"{'--use_8bit_adam' if config['use_8bit_adam'] else ''} "
            f"{'--gradient_checkpointing' if config['gradient_checkpointing'] else ''} "
            f"--mixed_precision {config['mixed_precision']}"
        ),
        max_steps = lambda wildcards: wildcards.max_steps
    log:
        "logs/rule_logs/train_model_{dataset_name}_{max_steps}.log"
    resources:
        gpu = 4,
    shell:
        "accelerate launch --config_file {params.accelerate_config} {params.script_path} "
        "{params.script_params} "
        "&> {log}"