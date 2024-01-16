import glob
import logging
import os
from pathlib import Path

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
    group: str,
    workflow: str = "main",
    profile: str = "test",
    stage: str = "debug",
    snakemake_args: str = "",
):
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

    snakemake_cmd = [
        "snakemake",
        f"--snakefile workflow/{workflow}.smk",
        f"--workflow-profile workflow/profiles/{profile}",
        f"--config exp_name={name} exp_group={group} stage={stage}",
    ]
    snakemake_cmd = " ".join(snakemake_cmd)
    ctx.run(f"{snakemake_cmd} {snakemake_args}", pty=True)
