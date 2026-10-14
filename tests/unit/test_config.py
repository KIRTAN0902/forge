from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from forge.cli.app import app
from forge.config import ForgeConfig, init_project, load_config
from forge.core.errors import ConfigError
from forge.policies import Permission, RiskLevel


def test_defaults_without_file(tmp_path: Path) -> None:
    config = load_config(tmp_path, env={})
    assert config.source is None
    assert config.root == tmp_path.resolve()
    assert config.limits.max_iterations == 10
    assert Permission.NETWORK not in config.policy.permissions
    assert config.db_path == tmp_path.resolve() / ".forge" / "forge.db"


def test_file_is_found_from_subdirectory(tmp_path: Path) -> None:
    (tmp_path / "forge.yaml").write_text("limits:\n  max_iterations: 4\n", encoding="utf-8")
    sub = tmp_path / "src" / "pkg"
    sub.mkdir(parents=True)
    config = load_config(sub, env={})
    assert config.limits.max_iterations == 4
    assert config.root == tmp_path.resolve()


def test_env_overrides_file(tmp_path: Path) -> None:
    (tmp_path / "forge.yaml").write_text("limits:\n  max_iterations: 4\n", encoding="utf-8")
    config = load_config(
        tmp_path,
        env={
            "FORGE__LIMITS__MAX_ITERATIONS": "7",
            "FORGE__POLICY__APPROVAL_THRESHOLD": "medium",
            "FORGE__POLICY__PERMISSIONS": "[read]",
            "UNRELATED": "1",
        },
    )
    assert config.limits.max_iterations == 7
    assert config.policy.approval_threshold == RiskLevel.MEDIUM
    assert config.policy.permissions == [Permission.READ]


def test_typos_are_rejected(tmp_path: Path) -> None:
    (tmp_path / "forge.yaml").write_text("limits:\n  max_iteration: 4\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="max_iteration"):
        load_config(tmp_path, env={})


def test_non_mapping_file_rejected(tmp_path: Path) -> None:
    (tmp_path / "forge.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(tmp_path, env={})


def test_init_template_is_valid_and_idempotent(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    first = init_project(tmp_path, name='my "app"')
    assert first.config_path in first.created
    assert first.gitignore_updated
    config = load_config(tmp_path, env={})
    assert config.project.name == 'my "app"'
    # The template must match the defaults so `forge init` never changes behaviour.
    defaults = ForgeConfig()
    assert config.limits == defaults.limits
    assert config.policy == defaults.policy

    second = init_project(tmp_path)
    assert second.config_path in second.skipped
    assert not second.gitignore_updated
    assert (tmp_path / ".gitignore").read_text(encoding="utf-8").count(".forge/") == 1


def test_cli_version_init_config(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "forge" in result.stdout

    result = runner.invoke(app, ["init", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert (tmp_path / "forge.yaml").exists()

    result = runner.invoke(app, ["config", str(tmp_path)])
    assert result.exit_code == 0, result.stdout
    assert "max_iterations: 10" in result.stdout

    (tmp_path / "forge.yaml").write_text("bogus: 1\n", encoding="utf-8")
    result = runner.invoke(app, ["config", str(tmp_path)])
    assert result.exit_code == 1
