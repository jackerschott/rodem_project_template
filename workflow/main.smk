# be careful to not execute container jobs when the absolute
# current working directory differs from os.path.realpath,
# e.g. because of a symlink; snakemake will mount the current
# directory incorrectly in this case
# note that snakemake mounts all directories from sys.path into
# the container, so effectively this only replaces the OS, not
# the python environment
container: config['container_path']

envvars:
    "WANDB_API_KEY"

HIDDEN_CONV_CHANNELS = [4, 8]

exp_group, exp_name = config['exp_group'], config['exp_name']
exp_path = os.path.join(config['experiments_base_path'], exp_group, exp_name)
experiment_id = f'{exp_group}/{exp_name}'

#hydra_cfg = load_hydra_config('main')
#if 'resume_training' in config:
#    hydra_cfg.resume_training = config['resume_training']
#hydra_cfg.stage = config['stage']
#hydra_cfg.wandb.api_key = os.environ['WANDB_API_KEY']
#hydra_cfg.datamodule.train_loader_factory.num_workers = LOADER_THREADS - 1
#hydra_cfg.datamodule.predict_loader_factory.num_workers = LOADER_THREADS - 1
#config = hydra_cfg

### ALL ###
#rule all:
#    input:
#        f'{exp_path}/results/roc_curves.pdf',

### PLOTTING ###
#rule plot_roc_curves:
#    group: 'plotting'
#    input:
#        f'{exp_path}/predictions/prediction_1.npz',
#        f'{exp_path}/predictions/prediction_2.npz',
#    output:
#        roc_curves = f'{exp_path}/results/roc_curves.pdf',
#    log: f'{exp_path}/logs/plot_roc_curves.log'
#    params:
#        pred_ids = [f'{n} conv channels'
#            for n in HIDDEN_CONV_CHANNELS],
#    script:
#        'scripts/plot_roc.py'

rule all:
    input:
        model = expand(exp_path + '/train_output/model_{hidden_conv_channels}.ckpt',
                hidden_conv_channels=range(2))

### PLOT ###
rule plot:
    input:
        dataset = f'{exp_path}/data.npz',
        prediction_1 = f'{exp_path}/predictions/prediction_1.h5',
        prediction_2 = f'{exp_path}/predictions/prediction_2.h5'
    output:
        plot = f'{exp_path}/plots/roc_curves.pdf'
    params:

    log: exp_path + '/logs/plot.log'
    wrapper:
        'https://raw.githubusercontent.com/sambklein/hydra_snakmake/main/'

### PREDICTING ###
rule predict:
    input:
        model = exp_path + '/train_output/model_{i}.ckpt',
        dataset = f'{exp_path}/data.npz',
    output:
        prediction = exp_path + '/predictions/prediction_{i}.h5'
    params:
    'scripts/predict.py',
    f'hydra.run.dir={exp_path}/hydra/predict',
    'dataset.load_path'
    threads: 4
    resources:
        runtime = 15,
        mem_mb = 16000,
        gpu=1,
    log: exp_path + '/logs/predict_{i}.log'
    wrapper:
        'https://raw.githubusercontent.com/sambklein/hydra_snakmake/main/'

rule train:
    input:
        dataset = f'{exp_path}/data.npz',
    output:
        model = exp_path + '/train_output/model_{i}.ckpt',
    params:
        'scripts/train.py',
        f'hydra.run.dir={exp_path}/hydra/predict'
        'dataset.load_path={input.dataset}',
        'model_save_path={output.model}',
        lambda wc: 'model.hidden_conv_channels'
            f'={HIDDEN_CONV_CHANNELS[int(wc.i) - 1]}',
        'trainer.callbacks.model_checkpoint.dirpath'
            f'={exp_path}/train_output/checkpoints_{{i}}',
        f'trainer.default_root_dir={exp_path}/train_output',
        f'wandb.api_key={os.environ["WANDB_API_KEY"]}',
        f'wandb.run_id_path={exp_path}/train_output/wandb_run_id_{{i}}',
        lambda wc: 'wandb.run_name='
            f'{experiment_id}/{HIDDEN_CONV_CHANNELS[int(wc.i) - 1]}_channels',
        f'wandb.save_dir={exp_path}/train_output',
        'datamodule.train_loader_factory.num_workers={threads-1}',
        'datamodule.predict_loader_factory.num_workers={threads-1}',
        'auto_resume=true',
    threads: 4
    resources:
        runtime = 60,
        mem_mb = 16000, # this also requests memory from slurm
        gpu = 1,
    log: f'{exp_path}/logs/train_model_{{i}}.log'
    wrapper:
        'https://raw.githubusercontent.com/sambklein/hydra_snakmake/main/'

### DATA ACQUISITION ###
rule acquire_data:
    output:
        dataset = f'{exp_path}/data.npz',
    params:
        'scripts/acquire_data.py',
        f'hydra.run.dir={exp_path}/hydra/predict'
        'save_path={output.dataset}'
    log: f'{exp_path}/logs/acquire_data.log'
    cache: True
    wrapper:
        'https://raw.githubusercontent.com/sambklein/hydra_snakmake/main/'
