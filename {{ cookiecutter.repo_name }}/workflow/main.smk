# be careful to not execute container jobs when the absolute
# current working directory differs from os.path.realpath,
# e.g. because of a symlink; snakemake will mount the current
# directory incorrectly in this case
# note that snakemake mounts all directories from sys.path into
# the container, so effectively this only replaces the OS, not
# the python environment
container: '{{ cookiecutter.container_path }}'

envvars:
    "WANDB_API_KEY"

from pathlib import Path
from mltools.snakemake import load_hydra_config

LOADER_THREADS = 4
HIDDEN_CONV_CHANNELS = [4, 8]

exp_group, exp_name = config['exp_group'], config['exp_name']
exp_path = f'experiments/{exp_group}/{exp_name}'
experiment_id = f'{exp_group}/{exp_name}'

hydra_cfg = load_hydra_config('main')
if 'resume_training' in config:
    hydra_cfg.resume_training = config['resume_training']
hydra_cfg.stage = config['stage']
hydra_cfg.wandb.api_key = os.environ['WANDB_API_KEY']
hydra_cfg.datamodule.train_loader_factory.num_workers = LOADER_THREADS - 1
hydra_cfg.datamodule.predict_loader_factory.num_workers = LOADER_THREADS - 1
config = hydra_cfg

### ALL ###
rule all:
    input:
        f'{exp_path}/results/roc_curves.pdf',

### PLOTTING ###
rule plot_roc_curves:
    group: 'plotting'
    input:
        f'{exp_path}/predictions/prediction_1.npz',
        f'{exp_path}/predictions/prediction_2.npz',
    output:
        roc_curves = f'{exp_path}/results/roc_curves.pdf',
    log: f'{exp_path}/logs/plot_roc_curves.log'
    params:
        pred_ids = [f'{n} conv channels'
            for n in HIDDEN_CONV_CHANNELS],
    script:
        'scripts/plot_roc.py'

### PREDICTING ###
rule predict:
    input:
        model = exp_path + '/train_output/model_{i}.ckpt',
        dataset = f'{exp_path}/data.npz',
    output:
        prediction = exp_path + '/predictions/prediction_{i}.npz'
    log: exp_path + '/logs/predict_{i}.log'
    threads: LOADER_THREADS
    resources:
        runtime = 15,
        mem_mb = 16000,
        slurm_extra = '--gpus=1'
    script:
        'scripts/predict.py'

### TRAINING ###
rule train_model:
    input:
        dataset = f'{exp_path}/data.npz',
    output:
        model = exp_path + '/train_output/model_{i}.ckpt',
    log: exp_path + '/logs/train_model_{i}.log'
    params:
        hidden_conv_channels = lambda wildcards:
            HIDDEN_CONV_CHANNELS[int(wildcards.i) - 1],
        checkpoints_path = exp_path + '/train_output/checkpoints_{i}',
        wandb_run_id_path = exp_path + '/train_output/wandb_run_id_{i}',
        wandb_run_name = lambda wildcards:
            f'{experiment_id}/{HIDDEN_CONV_CHANNELS[int(wildcards.i) - 1]}_channels',
        wandb_save_dir = f'{exp_path}/train_output',
        trainer_default_root_dir = f'{exp_path}/train_output',
    threads: LOADER_THREADS # this also requests CPUs from slurm
    resources:
        runtime = 60,
        mem_mb = 16000, # this also requests memory from slurm
        slurm_extra = '--gpus=1'
    script:
        'scripts/train.py'

### DATA ACQUISITION ###
rule acquire_data:
    output:
        dataset = f'{exp_path}/data.npz',
    log: f'{exp_path}/logs/acquire_data.log'
    script:
        'scripts/acquire_data.py'
