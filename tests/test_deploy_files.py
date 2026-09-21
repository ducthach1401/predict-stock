"""The deployment files stay consistent with each other (the container itself is exercised by hand: see docs/DEPLOY.md)."""
from __future__ import annotations

import subprocess

import yaml

from predict_stock.config import PROJECT_ROOT


def test_deploy_script_is_valid_bash_and_only_paper_commands_are_used():
    subprocess.run(["bash", "-n", str(PROJECT_ROOT / "deploy.sh")], check=True)
    subprocess.run(["bash", "-n", str(PROJECT_ROOT / "docker" / "bootstrap.sh")], check=True)
    text = (PROJECT_ROOT / "deploy.sh").read_text() + (PROJECT_ROOT / "docker" / "bootstrap.sh").read_text()
    assert "--mode live" not in text and "live" not in [w.strip("`'\"") for w in text.split()]


def test_compose_has_the_app_next_to_mysql_and_secrets_come_only_from_the_env_file():
    doc = yaml.safe_load((PROJECT_ROOT / "docker-compose.yml").read_text())
    app = doc["services"]["app"]
    assert app["env_file"] == ".env" and app["environment"]["MYSQL_HOST"] == "mysql" and "mysql" in app["depends_on"]
    assert not any("PASSWORD" in k for k in app["environment"])                      # no secret is written into the compose file
    assert {"./artifacts:/app/artifacts", "./backups:/app/backups", "./config:/app/config:ro"} <= set(app["volumes"])


def test_the_image_installs_the_locked_versions_and_the_lock_covers_the_declared_dependencies():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "requirements.lock" in dockerfile and "mysqldump" in dockerfile and "libgomp1" in dockerfile
    import tomllib
    deps = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    lock = {line.split("==")[0].lower().replace("_", "-") for line in (PROJECT_ROOT / "requirements.lock").read_text().splitlines() if "==" in line}
    for d in deps:
        name = d.split(">")[0].split("<")[0].split("=")[0].split("[")[0].strip().lower().replace("_", "-")
        assert name in lock, f"{name} is not pinned in requirements.lock"
