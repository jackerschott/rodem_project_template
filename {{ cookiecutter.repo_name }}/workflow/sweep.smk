container: config['container_path']

envvars:
    "WANDB_API_KEY"

import os
from mltools.snakemake import load_hydra_config

LOADER_THREADS = 4

exp_group, exp_name = config['exp_group'], config['exp_name']
exp_path = os.path.join(config['experiments_base_path'], exp_group, exp_name)
experiment_id = f'{exp_group}/{exp_name}'

hydra_cfg = load_hydra_config('sweep')
hydra_cfg.stage = config['stage']
hydra_cfg.wandb.api_key = os.environ['WANDB_API_KEY']
hydra_cfg.wandb.name = experiment_id
hydra_cfg.wandb.save_dir = '.'
hydra_cfg.datamodule.train_loader_factory.num_workers = LOADER_THREADS - 1
hydra_cfg.datamodule.predict_loader_factory.num_workers = LOADER_THREADS - 1
config = hydra_cfg

### STOP SWEEP ###
rule stop_sweep:
    input:
        # we need to generate a DAG that has job_count parallel train_sweep
        # invocations; to the best of my knowledge this doesn't work without having
        # an file with a wildcard in train_sweep and collecting all output files
        # here; technically one could do this with the wandb run directories, but
        # they have unpredictable names and I don't want to consider internal wandb
        # stuff anything other than external state in the workflow
        expand(exp_path + '/job_completion_markers/{job_idx}',
                job_idx=range(config.sweep.job_count)),
        sweep_id = f'{exp_path}/sweep_id',
    script:
        'scripts/stop_sweep.py'

### TRAINING ###
rule train_sweep:
    input:
        dataset = f'{exp_path}/data.npz',
        sweep_id = f'{exp_path}/sweep_id',
    output:
        job_completion_marker = exp_path + '/job_completion_markers/{job_idx}',
    params:
        checkpoints_base_path = f'{exp_path}/train_output/checkpoints',
        trainer_default_root_dir = f'{exp_path}/train_output',
    threads: LOADER_THREADS # this also requests CPUs from slurm
    resources:
        runtime = 60,
        mem_mb = 16000,
        slurm_extra = '--gpus=1'
    script:
        'scripts/train_sweep.py'

rule init_sweep:
    output:
        sweep_id = f'{exp_path}/sweep_id',
    params:
        sweep_name = experiment_id,
    script:
        'scripts/init_sweep.py'

### DATA ACQUISITION ###
rule acquire_data:
    output:
        dataset = f'{exp_path}/data.npz',
    script:
        'scripts/acquire_data.py'
