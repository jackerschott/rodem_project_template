git init
git submodule add https://gitlab.cern.ch/mleigh/mltools.git
pip install -r requirements.txt
pip install -r mltools/requirements.txt
python main.py experiment=generator.yaml
git add .
git commit -m "Initial commit"
# TODO: Add gitlab remote and push, currently refused as can't push to protected branch
# Also a problem to get the credentials correct, though this isn't a problem for the cookiecutter
# and this wouldn't be tracked by git
# git push --set-upstream git@gitlab.cern.ch:7999/rodem/projects/{{ cookiecutter.project_name }}.git master