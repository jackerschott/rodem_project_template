import json
import os
import shutil

with open("cookiecutter/cookiecutter.json") as f:
    cookiecutter_config = json.load(f)


def cookiecutter_string(variable_name: str):
    assert variable_name in cookiecutter_config
    return "{{ cookiecutter." + variable_name + " }}"


REPO_DIR = cookiecutter_string("repo_name")


def pack_repo_content():
    repo_content = os.listdir(".")

    excludes = [".git", "cookiecutter"]
    repo_content = [path for path in repo_content if path not in excludes]

    os.mkdir(REPO_DIR)

    for path in repo_content:
        repo_content = shutil.move(path, REPO_DIR)


def unpack_cookiecutter_dir():
    os.remove("cookiecutter/cookiecutterize.py")
    os.remove("cookiecutter/cookiecutterize.yml")
    shutil.move("cookiecutter/PROJECT_README.md", f"{REPO_DIR}/README.md")

    cookiecutter_files = os.listdir("cookiecutter")
    for file in cookiecutter_files:
        shutil.move(f"cookiecutter/{file}", ".")

    os.rmdir("cookiecutter")


def cookiecutterize_config():
    cookiecutterize_common_default_config()
    create_common_private_config()


def cookiecutterize_license():
    with open("LICENSE") as f:
        content = f.read()

    content = content.replace(
        "Copyright (c) 2023 Jona Ackerschott",
        f'Copyright (c) 2023 {cookiecutter_string("your_name")}',
    )

    with open("LICENSE", "w") as f:
        f.write(content)


def cookiecutterize_pyproject_toml():
    with open("pyproject.toml") as f:
        content = f.read()

    content = content.replace(
        'name = "digit_classification"',
        f'name = "{cookiecutter_string("project_name")}"',
    )
    content = content.replace(
        'description = "RODEM Project Template"',
        f'description = "{cookiecutter_string("project_name")}"',
    )
    content = content.replace(
        'packages = ["digit_classification"]',
        f'packages = ["{cookiecutter_string("python_package_name")}"]',
    )

    with open("pyproject.toml", "w") as f:
        f.write(content)


def cookiecutterize_common_default_config():
    with open("config/common/default.yaml") as f:
        content = f.read()

    content = content.replace("template", cookiecutter_string("project_name"))

    with open("config/common/default.yaml", "w") as f:
        f.write(content)


def create_common_private_config():
    shutil.copy("config/common/default.yaml", "config/common/private.yaml")

    with open("config/common/private.yaml") as f:
        content = f.read()

    content = content.replace(
        "container_path: null",
        f'container_path: {cookiecutter_string("container_path")}',
    )
    content = content.replace(
        "experiments_base_path: null",
        f'experiments_base_path: {cookiecutter_string("experiments_base_path")}',
    )
    content = content.replace(
        "wandb_username: null",
        f'wandb_username: {cookiecutter_string("wandb_username")}',
    )

    with open("config/common/private.yaml", "w") as f:
        f.write(content)


def cookiecutterize_package_dir():
    shutil.move("digit_classification", cookiecutter_string("python_package_name"))


def strip_cookiecutterize_from_ci_config():
    with open(".gitlab-ci.yml") as f:
        content = f.read()

    content = content.replace("include: 'cookiecutter/cookiecutterize.yml'\n\n", "")
    content = content.replace("  - cookiecutterize\n", "")

    with open(".gitlab-ci.yml", "w") as f:
        f.write(content)


pack_repo_content()
unpack_cookiecutter_dir()

os.chdir(REPO_DIR)
cookiecutterize_license()
cookiecutterize_pyproject_toml()
cookiecutterize_config()
cookiecutterize_package_dir()
strip_cookiecutterize_from_ci_config()
