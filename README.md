<div align="center">

# RODEM Project Template: MNIST Example

[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.1-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB_0.16-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![snakemake](https://img.shields.io/badge/-Snakemake_7.32.4-039475)](https://snakemake.readthedocs.io/)
[![invoke](https://img.shields.io/badge/-Invoke_2.2.0-yellow)](https://www.pyinvoke.org/)
</div>

This is an example application of the
[RODEM machine learning template](https://gitlab.cern.ch/rodem/projects/projecttemplate/), using a CNN
classifier for MNIST digit classification.
It relies on pytorch and lightning for machine learning, wandb for experiment
tracking, hydra for configuration management, snakemake for workflow management
and invoke as a project cli.
Additionally, [MLTools](https://gitlab.cern.ch/mleigh/mltools/) is used for certain helper functions.

## Usage

After cloning the repo, start by setting up a virtual environment on an HPC
cluster with slurm installed:
```
module load GCCcore/12.3.0 Python/3.11.3
python -m venv <env_path>
source <env_path>/bin/activate
pip install --upgrade pip
pip install -r requirements_workflow.txt
```
We start by loading the python module here (this is cluster specific), to make sure that we have a
newer python version, which is needed for some of the packages we want to use.

Next, make your wandb api key available by adding the following line to your .bashrc:
```
WANDB_API_KEY=<your-api-key>
```
This method is strictly for convenience and it should be noted that while other cluster users have
no access to your bashrc, for any admin it is easy to retreive your key.
So you might want to use a different method.
Be warned though that WandB itself also saves the key in plain text in $HOME/.netrc, so it is not
an easy task to circumvent this problem.

Now we can start the actual workflow.
This could be done with snakemake directly, but is a bit more convenient with invoke.
The latter simply provides a python wrapper around snakemake in `tasks.py` which neatly
documents the actually used commands and flags.
To run the main workflow with invoke without a container or slurm on the login node, simply type
```
invoke experiment-run <experiment_name>
```
where `<experiment_name>` is used to identify the current run/experiment and to define
the overall output directory, which will be `experiments/initial_testing/<experiment_name>`.
Under the hood, this command will run
```
snakemake \
    --snakefile workflow/main.smk \
    --workflow-profile workflow/profiles/test \
    --configfile config/common/default.yaml config/common/private.yaml \
    --config experiment_name=<experiment_name>
```
Check out tasks.py for details on the experiment-run subcommand. Note that all
invoke flags directly translate to shell flags (i.e. use `--profile` to define the
`profile` argument).

The invoke command provided above runs snakemake with the `test` profile, which
neither invokes slurm, nor uses a container and runs everything sequentially.
One can use a container and parallelization with the `test-in-container`
profile, while the `slurm` profile instructs snakemake to automatically launch slurm
jobs for each workflow step, using parallelization and containers as well.

The example workflow defined in `workflow/main.smk` looks as follows

![DAG](dag.png)

As you can see, this workflow will train two models, which differ by the number of hidden channels
in the convolution layers, predict labels from both models and plot ROC curves.
The all rule is simply the last rule and defines all output files that should be present.
The resulting plots can be found in `experiments/initial_testing/<experiment_name>/plots/rocs.pdf`
and `experiments/initial_testing/<experiment_group>/plots/paper.pdf`, where the
latter are optimised for a corresponding latex document.

## Scripts

There are three hydra scripts that get called by snakemake: `scripts/train.py`, `scripts/predict.py`, `scripts/plot.py`.
These scripts should be generic enough to never be changed for any kind of ML project.
However you might want to add additional scripts. This includes plotting scripts in particular, since the plotting
script is only meant to be used for prediction vs truth plots in any kind of
summarizing metric.

## Configuration

All configuration is handled by hydra and is stored as `config/<workflow_step_name>.yaml`.
There are two special files, `config/common/default.yaml` and `config/common/private.yaml`, which are
used by snakemake and hydra.
Note that `config/common/private.yaml` should never be pushed, since its user specific.

## Docker and Gitlab

This project is setup to use the CERN GitLab CI/CD to automatically build a Docker image based
on the `docker/Dockerfile` and `requirements.txt`.
It will also run the pre-commit as part of the pipeline.
To edit this behaviour change `.gitlab-ci.yml`

## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you have any improvements or suggestions.
Please use the provided `pre-commit` before making merge requests!

## License

This project is licensed under the MIT License. See the LICENSE file for details.
