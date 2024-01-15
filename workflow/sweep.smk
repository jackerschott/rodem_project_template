container: '/home/users/a/ackersch/scratch/'
    'projects/template/experiment_env.sif'

envvars:
    "WANDB_API_KEY"

from pathlib import Path
from mltools.snakemake import load_hydra_config

experiment_path = Path(os.path.realpath('.'))
experiment_id = f'{experiment_path.parent.name}/{experiment_path.name}'
hydra_cfg = load_hydra_config('sweep')
hydra_cfg.stage = config['stage']
hydra_cfg.wandb.api_key = os.environ['WANDB_API_KEY']
hydra_cfg.wandb.name = experiment_id
hydra_cfg.wandb.save_dir = '.'
hydra_cfg.trainer.callbacks[0].dirpath = 'checkpoints'
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
        expand('job_completion_markers/{job_idx}',
                job_idx=range(config.sweep.job_count)),
        sweep_id = 'sweep_id',
    script:
        'scripts/stop_sweep.py'

### TRAINING ###
rule train_sweep:
    input:
        dataset = 'data.npz',
        sweep_id = 'sweep_id',
    output:
        job_completion_marker = 'job_completion_markers/{job_idx}',
    params:
        checkpoints_base_path = 'train_output/checkpoints',
        trainer_default_root_dir = 'train_output',
    resources:
        runtime = 60,
        mem_mb = 16000,
        slurm_extra = '--gpus=1'
    script:
        'scripts/train_sweep.py'

rule init_sweep:
    output:
        sweep_id = 'sweep_id',
    params:
        sweep_name = experiment_id,
    script:
        'scripts/init_sweep.py'

### DATA ACQUISITION ###
rule acquire_data:
    output:
        dataset = 'data.npz',
    script:
        'scripts/acquire_data.py'
