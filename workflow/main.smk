container: config["container_path"]  # note that this only replaces the OS, not the python environment


envvars:
    "WANDB_API_KEY",


hidden_conv_channels = lambda i: [4, 8][int(i) - 1]

exp_group, exp_name = config["exp_group"], config["exp_name"]
exp_path = os.path.join(config["experiments_base_path"], exp_group, exp_name)
exp_id = f"{exp_group}/{exp_name}"

model = "cnn"
datamodule = "mnist"


rule all:
    input:
        f"{exp_path}/plots/rocs.pdf",
        f"{exp_path}/plots/paper.pdf",


rule plot:
    input:
        # we're only loading the dataset, so just take the first checkpoint
        ckpt=f"{exp_path}/train_output/result_1.ckpt",
        prediction_1=f"{exp_path}/predictions/prediction_1.h5",
        prediction_2=f"{exp_path}/predictions/prediction_2.h5",
    output:
        plot=f"{exp_path}/plots/{{plot_target}}.pdf",
    params:
        "scripts/plot.py",
        #f"plot_def.load_predict_set={datamodule}",
        "plot_def={plot_target}",
        "io.checkpoint_path={input.ckpt}",
        f"+io.prediction_paths.channels_{hidden_conv_channels(1)}={{input.prediction_1}}",
        f"+io.prediction_paths.channels_{hidden_conv_channels(2)}={{input.prediction_2}}",
        "io.output_path={output.plot}",
        f"hydra.run.dir={exp_path}/hydra/predict",
    log:
        f"{exp_path}/logs/plot_{{plot_target}}.log",
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.2/"


rule predict:
    input:
        ckpt=f"{exp_path}/train_output/result_{{i}}.ckpt",
    output:
        prediction=f"{exp_path}/predictions/prediction_{{i}}.h5",
    params:
        "scripts/predict.py",
        f"load_model={model}",
        f"load_datamodule={datamodule}",
        "io.checkpoint_path={input.ckpt}",
        "io.predictions_save_path={output.prediction}",
        f"hydra.run.dir={exp_path}/hydra/predict",
    threads: 4
    resources:
        runtime=15,
        mem_mb=16000,
        gpu=1,
    log:
        f"{exp_path}/logs/predict_{{i}}.log",
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.2/"


rule train:
    output:
        ckpt=f"{exp_path}/train_output/result_{{i}}.ckpt",
    params:
        "scripts/train.py",
        f"model={model}",
        lambda wc: f"model.hidden_conv_channels={hidden_conv_channels(wc.i)}",
        f"datamodule={datamodule}",
        "datamodule.dev_loader_conf.num_workers={threads-1}",
        # careful, thread count here and in predict should be the same!
        "datamodule.predict_loader_conf.num_workers={threads-1}",
        f"io.dataset_path={exp_path}/data.npz",
        "io.result_path={output.ckpt}",
        f"io.checkpoints_path={exp_path}/train_output/checkpoints_{{i}}",
        f"io.trainer_root={exp_path}/train_output",
        f"io.logging_dir={exp_path}/train_output",
        lambda wc: f"id={exp_id}/{hidden_conv_channels(wc.i)}_channels",
        f"hydra.run.dir={exp_path}/hydra/train_{{i}}",
    threads: 4
    resources:
        runtime=60,
        mem_mb=16000,  # this also requests memory from slurm
        gpu=1,
    log:
        f"{exp_path}/logs/train_model_{{i}}.log",
    wrapper:
        "https://raw.githubusercontent.com/sambklein/hydra_snakmake/v0.0.2/"
