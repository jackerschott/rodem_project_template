# The RODEM HEP project starter

## Using this template
This repository defines a cookiecutter template for quickly building new base repositories.
To use this on the cluster you first need to make a virtualenv on the login node and install cookiecutter. 

```
python3 -m venv ~/venvs/cluster
source ~/venvs/cluster/bin/activate
pip install cookiecutter==2.5.0
```

You can then run
```
cookiecutter https://${USERNAME}:${ACCESS_TOKEN}@gitlab.cern.ch/rodem/projects/projecttemplate
```
where `${USERNAME}` and `${ACCESS_TOKEN}` should be replaced by your gitlab username and accesstoken.
Then follow the prompts to create your new project.

If you want to clone this repo from your own branch, run
```
cookiecutter -c ${BRANCH_NAME} https://${USERNAME}:${ACCESS_TOKEN}@gitlab.cern.ch/rodem/projects/projecttemplate
```

You can also clone the repository and then define a new instance as follows
```
git clone https://gitlab.cern.ch/rodem/projects/projecttemplate
cookiecutter projecttemplate
rm -rf projecttemplate
```
This will define a new repository with the names that you defined.
This repository will not by default by a git repo, and you will need to add this manually and set up a remote on gitlab.


## Introduction

This README describes typical workflows and the goals the template should serve.
The README in the project itself provides instruction on how the template is designed to be used.

## Typical workflow
1) Research and development:
    * Collect data.
    * Get data loader.
    * Develop models.
    * Test models.
1) Start writing paper: 
    * Settle on models and evaluation. 
    * Run pipeline.
1) New requests: 
    * Update models.
    * Extend evaluation. 
    * Rerun models.
    * Rerun evaluation.
    * Update paper.    
9) Iterate 3. until done.

In the above workflow by far the most exhausting, time consuming, and boring, is step 4.
We should aim to cut down on this by building flexible automated workflows such that once the code has been fixed the full model training, model evaluation and result plotting can be done easily (ideally with a single command that launches all of the jobs straight to slurm).
Further, everything is ideally packaged such that our workflows can be easily reproduced by other groups.

### Orchestration
Note that at present this workflow does not make use of any workflow manager to orchestrate the full suite of jobs.
This is a missing piece, but not suitable tool has been found that can be installed on the UNIGE cluster and does not add overhead when working with SLURM and hydra.
