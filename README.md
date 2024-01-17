# The RODEM HEP project starter

## Introduction

This repository serves as a template for a typical research machine learning workflow.
It is loosely based on the PyTorch Lightning Hydra template by ashleve.
The basic usage and goals off this template are described in the following.
The README in the project itself provides more detailed usage instructions.

## Typical workflow
1) Research and development:
    * Collect data.
    * Get data loader.
    * Develop models.
    * Test models.
2) Start writing paper: 
    * Settle on models and evaluation. 
    * Run pipeline.
3) New requests: 
    * Update models.
    * Extend evaluation. 
    * Rerun models.
    * Rerun evaluation.
    * Update paper.    
4) Iterate 3 until done.

In the above workflow by far the most exhausting, time consuming, and boring, is step 4.
This template uses snakemake to build flexible automated workflows, which can combine
data generation, model training, model evaluation and result plotting into a
single command that automatically launches slurm jobs.

## Using this template
Start by setting up a new virtual environment on an HPC cluster with slurm and
install cookiecutter:
```
python3 -m venv <env_path>
source <env_path>/bin/activate
pip install --upgrade pip
pip install cookiecutter
```
This environment can and should be reused for installing the
workflow requirements as described in the project readme.

Now we can run
```
cookiecutter https://${USERNAME}:${ACCESS_TOKEN}@gitlab.cern.ch/rodem/projects/projecttemplate
```
And follow the prompts to create your new project.
To use a specific branch:
```
cookiecutter -c <branch> https://${USERNAME}:${ACCESS_TOKEN}@gitlab.cern.ch/rodem/projects/projecttemplate
```
You can also clone the repository and then define a new instance as follows
```
git clone https://gitlab.cern.ch/rodem/projects/projecttemplate
cookiecutter projecttemplate
```
Note that cookiecutter will ask you for a container path.
The corresponding container is supposed to be build by the provided Dockerfile
and gitlab CI and pulled to the provided path later.
Naturally the container path can always be changed by editing `workflow/main.smk` and `workflow/sweep.smk`.

Afterwards you will be left with a new repository using the values you entered.
This repository will not by default be a git repository, so you want to
initialize one, setup a gitlab repository and push.
This will also automatically launch a gitlab pipeline to build a container,
using the provided Dockerfile.

After the pipeline is done you want to pull the container
```
apptainer pull --docker-login <container_path> <gitlab_repo_url>/docker-image:latest
```
where `<gitlab_repo_url>` should be replaced with the url to your newly created
gitlab repo and `<container_path>` with the path you entered previously.

This concludes the setup.
For instructions on how to run experiments, you can check out the README inside the project.

## Contributing

Contributions are welcome! Please submit a pull request or create an issue if you have any improvements or suggestions.
Note that the `master` branch is nothing but a few commits on top of the `example` branch that
"cookiecutterizes" everything. 
So for almost all development you want to use the example branch and rebase the
master branch onto it.
Also, please use the provided `pre-commit` before making merge requests!
