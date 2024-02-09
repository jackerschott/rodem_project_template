import glob
import logging
import os
from pathlib import Path
from typing import Optional

from invoke import Context, task

# setup logger
logging.basicConfig(
    level=logging.INFO, format="[%(filename)s] %(levelname)s: %(message)s"
)
log = logging.getLogger(__name__)


@task
def experiment_run(
    ctx: Context,
    name: str,
    group: Optional[str] = None,
    workflow: str = "main",
    profile: str = "test",
    stage: Optional[str] = None,
    snakemake_args: str = "",
):
    """Run an experiment through a snakemake workflow.

    All output will be gathered in experiments/<group>/<name>/. Persistent defaults for
    group and stage can be found in config/common/default.yaml and
    config/common/private.yaml.

    Args
    ----
        ctx (Context): invoke context
        name (str): name of the experiment, used for identification and to determine the
                    output directory.
        group (Optional[str], optional): group of the experiment, used for
                                         identification and to determine the output
                                         directory (default: None).
        workflow (str, optional): name of the workflow, available workflows are located
                                  in the workflow directory under `<workflow>.smk`
                                  (default: "main").
        profile (str, optional): workflow profile, sets hardware specific snakemake
                                 flags , available profiles are located in the
                                 workflow/profiles directory (default: "test").
        stage (Optional[str], optional): experiment stage, should be either "debug" or
                                         "experiment". Used for wandb tags
                                         (default:None).
        snakemake_args (str, optional): anything additional one might want to pass to
                                        the snakemake command (default: "").
    """
    workflows = [Path(p).stem for p in glob.glob("workflow/*.smk")]
    if workflow not in workflows:
        log.error(
            f'Workflow "{workflow}" not found.'
            f' Available workflows: {", ".join(workflows)}'
        )
        return

    profiles = os.listdir("workflow/profiles")
    if profile not in profiles:
        log.error(
            f'Profile "{profile}" not found.'
            f' Available profiles: {", ".join(profiles)}'
        )
        return

    snakemake_cfg = [f"experiment_name={name}"]
    if stage:
        snakemake_cfg.append(f"stage={stage}")
    if group:
        snakemake_cfg.append(f"experiment_group={group}")
    snakemake_cfg = " ".join(snakemake_cfg)

    snakemake_cmd = [
        "snakemake",
        f"--snakefile workflow/{workflow}.smk",
        f"--workflow-profile workflow/profiles/{profile}",
        "--configfile config/common/default.yaml config/common/private.yaml",
        f"--config {snakemake_cfg}",
    ]
    snakemake_cmd = " ".join(snakemake_cmd)

    ctx.run(f"{snakemake_cmd} {snakemake_args}", pty=True)


@task
def pre_commit(ctx: Context):
    ctx.run("pre-commit run --all-files", pty=True)


@task
def pull_container(ctx: Context, login: bool = False):
    apptainer_pull_args = []
    if login:
        apptainer_pull_args.append("--docker-login")

    url = (
        "docker://gitlab-registry.cern.ch/rodem/"
        "projects/projecttemplate/docker-image:latest"
    )
    ctx.run(f"apptainer pull {apptainer_pull_args} {url}", pty=True)
