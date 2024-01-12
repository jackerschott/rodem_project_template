from abc import abstractmethod
import json
import logging
import netrc
import os
from pathlib import Path
import re
import socket
from typing import Literal, Union, Tuple, List

from fabric import Connection
from invoke import Context, task

# setup logger
logging.basicConfig(level=logging.INFO,
        format="[%(filename)s] %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

### CLASSES
class DevEnvironment:
    def __init__(self):
        pass

    def connection(self):
        return Connection('localhost', self.USER, self.REVERSE_PORT)

    def root_path(self):
        return self.ROOT_PATH

    def tasks_path(self):
        return self.root_path() / "tasks.py"

    def workflow_path(self):
        return self.root_path() / "workflow"

    def module_paths(self):
        return [self.root_path() / 'src']

    def results_dir(self):
        return self.root_path() / 'results'

    def get_result_dir(self, name: str, group: str):
        return self.results_dir() / group / name

    def workflow_excludes(self):
        return ['__pycache__']

    def module_excludes(self):
        return ['__pycache__']

    def setup_experiment(self, conn: 'EnvironmentConnection', name: str, group: str):
        assert self.is_local()
        assert conn.dev_env == self

        conn.exp_env.populate_experiment(conn, name, group, includes=['tasks'])

        wandb_api_key = self._get_wandb_api_key(conn)
        env = f'WANDB_API_KEY={wandb_api_key}'

        experiment_dir = conn.exp_env.get_experiment_dir(name, group)
        tmux_cmd = f'tmux new-session -s "{group}/{name}" -c "{experiment_dir}"'
        if conn.is_local('exp'):
            conn.run('dev', f'{env} exec {tmux_cmd}', pty=True)
        else:
            conn.ssh(f'{env} exec {tmux_cmd}')

    def attach_to_experiment(self, conn: 'EnvironmentConnection',
            name: str, group: str):
        assert self.is_local()
        assert conn.dev_env == self

        tmux_cmd = f'tmux attach-session -t {group}/{name}'
        if conn.is_local('exp'):
            conn.run('dev', f'{tmux_cmd}', pty=True)
        else:
            conn.ssh(f'exec {tmux_cmd}')

    @abstractmethod
    def _get_wandb_api_key(self, conn: 'EnvironmentConnection'):
        ...

class ExpEnvironment:
    def host(self):
        return f'{self.USER}@{self.HOSTNAME}'

    def connection(self):
        return Connection(self.HOSTNAME, self.USER, self.PORT)

    def get_experiment_dir(self, name: str, group: str):
        return self.EXPERIMENTS_DIR / group / name

    def get_experiment_identifiers_from_cwd(self):
        cwd = Path(os.getcwd())
        return cwd.name, cwd.parent.name

    def tasks_dest(self, experiment_dir: Path):
        return experiment_dir / "tasks.py"

    def workflow_dest(self, experiment_dir: Path):
        return experiment_dir / "workflow"

    def modules_dest(self, experiment_dir: Path):
        return experiment_dir / "modules"

    def plots_dest(self, experiment_dir: Path):
        return experiment_dir / "plots"


    def run_experiment(self, conn: 'EnvironmentConnection', workflow='main',
            profile: str = 'default', snakemake_cfg: str = '',
            other_snakemake_opts: str = '', populate=List[str], push_results=False):
        assert self.is_local()
        assert conn.exp_env == self
        exp_name, exp_group = self.get_experiment_identifiers_from_cwd()

        if len(populate) > 0:
            self.populate_experiment(conn, exp_name, exp_group, includes=populate)

        profiles = os.listdir('workflow/profiles')
        if not profile in profiles:
            log.error(f'Profile "{profile}" not found.'
                f' Available profiles: {", ".join(profiles)}')
            return

        # This PYTHONPATH doesn't work when using a container; in this case the
        # path should be set by using the singularity-args flag in the snakemake
        # profile
        experiment_dir = self.get_experiment_dir(exp_name, exp_group)
        modules_dest = self.modules_dest(experiment_dir)
        snakemake_opts = [
            f'--snakefile workflow/{workflow}.smk',
            f'--workflow-profile workflow/profiles/{profile}',
            f'--config {snakemake_cfg}',
        ]
        snakemake_opts = ' '.join(snakemake_opts)
        conn.run('exp', f'PYTHONPATH="{modules_dest}"'
                f' snakemake {snakemake_opts} {other_snakemake_opts}', pty=True)

        if push_results:
            self.push_experiment_results(conn, exp_name, exp_group)
            
    def push_experiment_results(self, conn: 'EnvironmentConnection',
            name: str, group: str):
        assert conn.exp_env == self
        result_path = conn.dev_env.get_result_dir(name, group)
        conn.makedirs('dev', result_path)

        results_full_path = result_path / 'results_full.pdf'
        conn.run('dev', f'rm -f {results_full_path}')

        plots_dir = self.plots_dest(self.get_experiment_dir(name, group))
        conn.rsync('exp', 'dev', plots_dir, result_path, delete=True, copy_contents=True)

        conn.run('dev', f'pdfunite {result_path}/*.pdf {results_full_path}')

    def populate_experiment(self, conn: 'EnvironmentConnection',
            name: str, group: str, includes = List[str]):
        experiment_dir = self.get_experiment_dir(name, group)
        conn.makedirs('exp', experiment_dir)

        if 'tasks' in includes:
            conn.rsync('dev', 'exp', conn.dev_env.tasks_path(),
                    conn.exp_env.tasks_dest(experiment_dir), delete=True)

        if 'workflow' in includes:
            log.info('uploading workflow')
            conn.makedirs('exp', conn.exp_env.workflow_dest(experiment_dir))
            conn.rsync('dev', 'exp', conn.dev_env.workflow_path(),
                    conn.exp_env.workflow_dest(experiment_dir),
                    excludes=conn.dev_env.workflow_excludes(),
                    delete=True, copy_contents=True)

        if 'main_module' in includes or 'other_modules' in includes:
            log.info('uploading modules')
            conn.makedirs('exp', conn.exp_env.modules_dest(experiment_dir))

        if 'main_module' in includes:
            conn.rsync('dev', 'exp', conn.dev_env.module_paths()[0],
                    conn.exp_env.modules_dest(experiment_dir),
                    excludes=conn.dev_env.module_excludes(), delete=True)
        if 'other_modules' in includes:
            for module_path in conn.dev_env.module_paths()[1:]:
                conn.rsync('dev', 'exp', module_path,
                        conn.exp_env.modules_dest(experiment_dir),
                        excludes=conn.dev_env.module_excludes(), delete=True)

class EnvironmentConnection:
    conn: Union[Context, Connection]
    dev_env: DevEnvironment
    exp_env: ExpEnvironment

    def __init__(self, ctx: Context, env_local: Union[DevEnvironment, ExpEnvironment],
            env_remote: Union[DevEnvironment, ExpEnvironment]):
        assert isinstance(env_local, DevEnvironment) \
            and isinstance(env_remote, ExpEnvironment) \
            or isinstance(env_local, ExpEnvironment) \
            and isinstance(env_remote, DevEnvironment)

        self.dev_env = (env_remote, env_local)[isinstance(env_local, DevEnvironment)]
        self.exp_env = (env_remote, env_local)[isinstance(env_local, ExpEnvironment)]

        if self.is_local('dev') and self.is_local('exp'):
            self.conn = ctx
        elif self.is_local('dev') and not self.is_local('exp'):
            self.conn = self.exp_env.connection()
        elif not self.is_local('dev') and self.is_local('exp'):
            self.conn = self.dev_env.connection()
        else:
            assert False
 
    def is_local(self, env: Literal['dev', 'exp', 'local', 'remote']):
        if env == 'dev':
            return self.dev_env.is_local()
        elif env == 'exp':
            return self.exp_env.is_local()
        elif env == 'local':
            return True
        elif env == 'remote':
            return False
        else:
            assert False

    def env_by_target(self, target: Literal['local', 'remote']):
        if target == 'local' and self.is_local('dev'):
            return self.dev_env
        elif target == 'local' and self.is_local('exp'):
            return self.exp_env
        elif target == 'remote' and self.is_local('dev'):
            return self.exp_env
        elif target == 'remote' and self.is_local('exp'):
            return self.dev_env

    def run(self, env: Literal['dev', 'exp', 'local', 'remote'], *args, **kwargs):
        if self.is_local(env) and type(self.conn) is Connection:
            return self.conn.local(*args, **kwargs)
        else:
            return self.conn.run(*args, **kwargs)

    def makedirs(self, env: Literal['dev', 'exp', 'local', 'remote'], path: Path):
        if self.is_local(env):
            os.makedirs(path, exist_ok=True)
        else:
            self.run(env, f'mkdir -p {path}')

    def rsync(self, env_src: Literal['dev', 'exp'],
            env_target: Literal['dev', 'exp'], *args, **kwargs):
        if self.is_local(env_src) and self.is_local(env_target):
            rsync(self.conn, 'local', *args, **kwargs)
        elif self.is_local(env_src):
            rsync(self.conn, 'to_remote', *args, **kwargs)
        elif self.is_local(env_target):
            rsync(self.conn, 'from_remote', *args, **kwargs)
        else:
            assert False

    def ssh(self, cmd: str):
        assert self.is_local('dev')
        self.run('local', f'ssh -p {self.exp_env.PORT} -tR'
                f' {self.dev_env.REVERSE_PORT}:localhost:{self.dev_env.PORT}'
                f' {self.exp_env.host()} "{cmd}"', pty=True)


def rsync(conn: Union[Context, Connection],
        flow: Literal['to_remote', 'from_remote', 'local'],
        source: Path, target: Path, excludes: Tuple[str] = (), quiet: bool = False,
        delete: bool = False, infos: List[str] = ['DEL', 'REMOVE', 'NAME'],
        copy_contents = False):
    opts = ['-r', '--checksum']
    if quiet:
        opts.append('-q')
    if delete:
        opts.append('--delete')
    if len(infos) > 0:
        opts.append(f'--info={",".join(infos)}')
    opts = ' '.join(opts)
    excludes = ' '.join([f'--exclude={exclude}' for exclude in excludes])

    if copy_contents:
        source = f'{source}/'

    if flow == 'to_remote':
        assert type(conn) is Connection
        conn.local(f'rsync -e "ssh -p {conn.port}" {opts} {excludes} ' \
                f'"{source}" "{conn.user}@{conn.host}:{target}"')
    elif flow == 'from_remote':
        assert type(conn) is Connection
        conn.local(f'rsync -e "ssh -p {conn.port}" {opts} {excludes} ' \
                f'"{conn.user}@{conn.host}:{source}" "{target}"')
    else:
        assert type(conn) is Context
        conn.run(f'rsync {opts} {excludes} "{source}" "{target}"')

### ENVIRONMENTS
class TeslaDev(DevEnvironment):
    ROOT_PATH = Path('/home/jona/studies/phd/'
            'projecttemplate/template/{{ cookiecutter.repo_name }}')
    HOSTNAME = 'tesla'
    USER = 'jona'
    PORT = 22
    REVERSE_PORT = 5000

    def is_local(self):
        return socket.gethostname() == self.HOSTNAME

    def module_paths(self):
        return [self.root_path() / m for m in [
            'src/model_indep_unfolding', 'mltools/mltools', 'normflows/normflows'
        ]]

    @abstractmethod
    def _get_wandb_api_key(self, conn: 'EnvironmentConnection'):
        if not os.path.exists(os.path.expanduser('~/.netrc')):
            conn.run('dev', 'wandb login', hide=True)

        netrc_info = netrc.netrc()
        wandb_info = netrc_info.authenticators('api.wandb.ai')
        if not wandb_info:
            conn.run('dev', 'wandb login', hide=True)

        _, _, api_key = wandb_info
        return api_key

class TeslaExp(ExpEnvironment):
    HOSTNAME = 'tesla'
    EXPERIMENTS_DIR = Path('/home/jona/studies/phd/'
            'projecttemplate/template/{{ cookiecutter.repo_name }}/experiments')

    def is_local(self):
        return socket.gethostname() == self.HOSTNAME

class Baobab(ExpEnvironment):
    HOSTNAME = 'baobab'
    USER = 'ackersch'
    PORT = 22
    # make sure there are no symlinks here; this fucks up snakemake mounts otherwise
    EXPERIMENTS_DIR = Path('/srv/beegfs/scratch/users/a/ackersch/'
            'projects/template/{{ cookiecutter.repo_name }}/experiments')

    def is_local(self):
        return self.on_login_node() or self.on_compute_node()

    def on_login_node(self):
        return socket.gethostname() == f'login2.{self.HOSTNAME}'

    def on_compute_node(self):
        return not re.match(f'^gpu[0-9]+.{self.HOSTNAME}$',
            socket.gethostname()) is None

### SETUP
@task
def experiment_setup(ctx: Context, name: str, group: str, local=False):
    if local:
        conn = EnvironmentConnection(ctx, TeslaDev(), TeslaExp())
    else:
        conn = EnvironmentConnection(ctx, TeslaDev(), Baobab())
    conn.dev_env.setup_experiment(conn, name, group)

@task
def experiment_attach(ctx: Context, name: str, group: str, local=False):
    if local:
        conn = EnvironmentConnection(ctx, TeslaDev(), TeslaExp())
    else:
        conn = EnvironmentConnection(ctx, TeslaDev(), Baobab())
    conn.dev_env.attach_to_experiment(conn, name, group)

### POPULATE
@task
def experiment_push(ctx: Context, name: str, group: str, local=False,
        include_tasks: bool = True, include_workflow: bool = True,
        include_main_module: bool = True, include_other_modules: bool = True):
    if local:
        conn = EnvironmentConnection(ctx, TeslaDev(), TeslaExp())
    else:
        conn = EnvironmentConnection(ctx, TeslaDev(), Baobab())

    locals_vars = locals()
    includes = [name.removeprefix('include_') for name in local_vars
        if name.startswith('include_') and local_vars[name]]
    conn.exp_env.populate_experiment(conn, name, group, includes=includes)

@task
def experiment_populate(ctx: Context, include_tasks: bool = True,
        include_workflow: bool = True, include_main_module: bool = True,
        include_other_modules: bool = True):
    tesla = TeslaExp()
    baobab = Baobab()
    if tesla.is_local():
        conn = EnvironmentConnection(ctx, TeslaDev(), tesla)
    elif baobab.is_local():
        conn = EnvironmentConnection(ctx, baobab, TeslaDev())
    else:
        assert False

    local_vars = locals()
    includes = [name.removeprefix('include_') for name in local_vars
        if name.startswith('include_') and local_vars[name]]

    name, group = conn.exp_env.get_experiment_identifiers_from_cwd()
    conn.exp_env.populate_experiment(conn, name, group, includes=includes)
    
### RESULTS
@task
def experiment_results_pull(ctx: Context, name: str, group: str, local=False):
    if local:
        conn = EnvironmentConnection(ctx, TeslaDev(), TeslaExp())
    else:
        conn = EnvironmentConnection(ctx, TeslaDev(), Baobab())

    conn.exp_env.push_experiment_results(conn, name, group)

@task
def experiment_results_push(ctx: Context):
    tesla = TeslaExp()
    baobab = Baobab()
    if tesla.is_local():
        conn = EnvironmentConnection(ctx, TeslaDev(), tesla)
    elif baobab.is_local():
        conn = EnvironmentConnection(ctx, baobab, TeslaDev())
    else:
        assert False

    conn.exp_env.push_experiment_results(conn, name, group)

### RUN
@task
def experiment_run(ctx: Context, workflow: str = 'main', profile: str = 'default',
        snakemake_cfg: str = '', snakemake_opts: str = '',
        populate: bool = False, populate_tasks: bool = True,
        populate_workflow: bool = True, populate_main_module: bool = True,
        populate_other_modules: bool = True, push_results: bool = False):
    tesla = TeslaExp()
    baobab = Baobab()
    if tesla.is_local():
        conn = EnvironmentConnection(ctx, TeslaDev(), tesla)
    elif baobab.is_local():
        conn = EnvironmentConnection(ctx, baobab, TeslaDev())
    else:
        assert False

    local_vars = locals()
    if not populate:
        populate_includes = []
    else:
        populate_includes = [name.removeprefix('populate_') for name in local_vars
            if name.startswith('populate_') and local_vars[name]]
    conn.exp_env.run_experiment(conn, workflow, profile, snakemake_cfg, snakemake_opts,
            populate=populate_includes, push_results=push_results)
