# be careful to not execute container jobs when the absolute
# current working directory differs from os.path.realpath,
# e.g. because of a symlink; snakemake will mount the current
# directory incorrectly in this case
container: '/home/users/a/ackersch/scratch/'
    'projects/mnist_jona/experiment_env.sif'

envvars:
    "WANDB_API_KEY"

from pathlib import Path
from mltools.snakemake import load_hydra_config

LOADER_THREADS = 4
HIDDEN_CONV_CHANNELS = [4, 8]

experiment_path = Path(os.path.realpath('.'))
experiment_id = f'{experiment_path.parent.name}/{experiment_path.name}'

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
        'results/roc_curves.pdf',

### PLOTTING ###
rule plot_roc_curves:
    group: 'plotting'
    input:
        'predictions/prediction_1.npz',
        'predictions/prediction_2.npz',
    output:
        roc_curves = 'results/roc_curves.pdf',
    params:
        pred_ids = [f'{n} conv channels'
            for n in HIDDEN_CONV_CHANNELS],
    script:
        'scripts/plot_roc.py'

### PREDICTING ###
rule predict:
    input:
        model = 'train_output/model_{i}.ckpt',
        dataset = 'data.npz',
    output:
        prediction = 'predictions/prediction_{i}.npz'
    threads: LOADER_THREADS
    resources:
        runtime = 15,
        mem_mb = 1000,
        slurm_extra = '--gpus=1'
    script:
        'scripts/predict.py'

### TRAINING ###
rule train_model:
    input:
        dataset = 'data.npz',
    output:
        model = 'train_output/model_{i}.ckpt',
    params:
        hidden_conv_channels = lambda wildcards:
            HIDDEN_CONV_CHANNELS[int(wildcards.i) - 1],
        checkpoints_path = 'train_output/checkpoints_{i}',
        wandb_run_id_path = 'train_output/wandb_run_id_{i}',
        wandb_run_name = f'{experiment_id}/{{i}}',
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
        dataset = 'data.npz',
    script:
        'scripts/acquire_data.py'
