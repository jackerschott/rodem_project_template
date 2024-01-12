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

experiment_path = Path(os.path.realpath('.'))
experiment_id = f'{experiment_path.parent.name}/{experiment_path.name}'
hydra_cfg = load_hydra_config('main')
if 'resume_training' in config:
    hydra_cfg.resume_training = config['resume_training']
hydra_cfg.stage = config['stage']
hydra_cfg.wandb.api_key = os.environ['WANDB_API_KEY']
config = hydra_cfg

### ALL ###
rule all:
    input:
        'results/roc.pdf',
        'results/predictions.pdf',

### PLOTTING ###
rule plot_roc:
    group: 'plotting'
    input:
        prediction_1 = 'predictions/1.npz',
        prediction_2 = 'predictions/2.npz',
    output:
        plots = 'plots/roc.pdf',
    script:
        'scripts/plot_roc.py'

rule plot_predictions:
    group: 'plotting'
    input:
        prediction_1 = 'predictions/1.npz',
    output:
        plots = 'plots/predictions.pdf',
    script:
        'scripts/plot_predictions.py'

### PREDICTING ###
rule predict:
    input:
    input:
        model = 'train_output/model_{i}.ckpt',
        dataset = 'data.npz',
    output:
        plots = 'plots/predictions.pdf'
    script:
        'scripts/predict.py'

### TRAINING ###
rule train_model:
    input:
        dataset = 'data.npz',
    output:
        model = 'train_output/model_{i}.ckpt',
        model = 'train_output/{dataset_fit}/model.ckpt'
    params:
        checkpoints_path = 'train_output/{dataset_fit}/checkpoints',
        wandb_run_id_path = 'train_output/{dataset_fit}/wandb_run_id',
        wandb_run_name = lambda wildcards:
            f'{experiment_id}/{wildcards.dataset_fit}',
    resources:
        runtime = 60,
        mem_mb = 16000,
        slurm_extra = '--gpus=1'
    script:
        'scripts/train.py'

### DATA ACQUISITION ###
rule acquire_data:
    group: 'data_acquisition'
    output:
        dataset = 'data.npz',
    script:
        'scripts/acquire_data.py'
