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
This repository defines a cookiecutter template for quickly building new base repositories.
After cloning the repo
```
git clone https://gitlab.cern.ch/rodem/projects/projecttemplate/ <repo_name>
cd <repo_name>
```
start by setting up a new virtual environment on an HPC cluster with slurm installed:
```
python3 -m venv <env_path>
source <env_path>/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
This will install cookiecutter, snakemake and some other packages needed for the
workflows defined in the template.

Now we can run
```
cookiecutter git clone https://gitlab.cern.ch/rodem/projects/projecttemplate
```
And follow the prompts to create your new project.
Note that you might need to add your username and access token here:
```
cookiecutter git clone https://${USERNAME}:${ACCESS_TOKEN}@gitlab.cern.ch/rodem/projects/projecttemplate
```

You can also clone the repository and then define a new instance as follows
```
git clone https://gitlab.cern.ch/rodem/projects/projecttemplate
cookiecutter projecttemplate
```
This will define a new repository with the names that you defined.
This repository will not by default by a git repo, and you will need to add this manually and set up a remote on gitlab.
