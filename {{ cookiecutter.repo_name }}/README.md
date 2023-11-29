<div align="center">

# {{ cookiecutter.project_name }}

[![python](https://img.shields.io/badge/-Python_3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![pytorch](https://img.shields.io/badge/-PyTorch_2.1-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![lightning](https://img.shields.io/badge/-Lightning_2.1-792EE5?logo=lightning&logoColor=white)](https://lightning.ai/)
[![hydra](https://img.shields.io/badge/-Hydra_1.3-89b8cd&logoColor=white)](https://hydra.cc/)
[![wandb](https://img.shields.io/badge/-WandB_0.16-orange?logo=weightsandbiases&logoColor=white)](https://wandb.ai)
</div>

This project is generated from the RODEM template for training deep learning models using PyTorch, Lightning, Hydra, and WandB. It is loosely based on the PyTorch Lightning Hydra template by ashleve.

The example network in this template is a generative diffusion model for MNIST using the EDM framework.

## Submodules

The example project in this template relies on a custom submodule called `mltools`.
This is a collection of useful functions, layers and networks for deep learning developed by the RODEM group at UNIGE.

[MLTools Repo](https://gitlab.cern.ch/mleigh/mltools/-/tree/master)

## Configuration

All job configuration is handled by hydra and omegaconf and are stored in the `configs` directory.
The main configuration file that composes training is `train.yaml`.
It sets the seed for reproducibility, the project and network names, the checkpoint path for resuming training, precision and compile options for PyTorch, and tags for the loggers.
This file composes the training config using yaml files for the `trainer`, `model`, `datamodule`, `loggers`, `paths`, `callbacks`.
The `experiment` folder is used to overwite any of the config values before composition.
Ideally trainings should always be run using `python train.py experiment=...`

## Usage

To quickly set up the repo with a single command run

```
chmod +777 setup_git.sh; ./setup_git.sh
```

To try out this template, follow these steps:

1. Make the project into a repository and add the mltools submodule:
```
git init
git submodule add https://gitlab.cern.ch/mleigh/mltools.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Install the required packages for mltools:
```
pip install -r mltools/requirements.txt
```
4. Run the training script with the desired configuration:
```
python main.py experiment=generator.yaml
```

## Docker and Gitlab

This project is setup to use the CERN GitLab CI/CD to automatically build a Docker image based on the `docker/Dockerfile` and `requirements.txt`.
It will also run the pre-commit as part of the pipeline.
To edit this behaviour change `.gitlab-ci`

## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you have any improvements or suggestions.
Please use the provided `pre-commit` before making merge requests!

## License

This project is licensed under the MIT License. See the [LICENSE](https://gitlab.cern.ch/rodem/projects/projecttemplate/blob/main/LICENSE) file for details.