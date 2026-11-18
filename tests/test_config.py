from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from junctionlab.config import DEFAULT_CONFIG_PATH, load_config


def test_repo_config_loads_with_decided_values():
    cfg = load_config()
    assert cfg.geometry.approach_length_m == 300
    assert cfg.geometry.queue_warning_share == 0.8
    assert len(cfg.simulation.seeds) == 5
    assert cfg.simulation.time_to_teleport_s == -1
    assert cfg.simulation.end_s == 600 + 3600 + 900
    assert cfg.driver.sigma == 0.5
    assert (cfg.control.amber_s, cfg.control.all_red_s) == (3, 2)
    assert cfg.los.signalised["E"] == 80
    assert cfg.los.unsignalised["E"] == 50
    assert cfg.llm.model == "claude-haiku-5-5"


def _write(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    mutate(data)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_rejects_non_increasing_los_thresholds(tmp_path):
    path = _write(tmp_path, lambda d: d["los"]["signalised"].update(C=15))
    with pytest.raises(ValidationError, match="increase"):
        load_config(path)


def test_rejects_unknown_keys(tmp_path):
    path = _write(tmp_path, lambda d: d["geometry"].update(typo_field=1))
    with pytest.raises(ValidationError):
        load_config(path)


def test_env_var_overrides_default_path(tmp_path, monkeypatch):
    path = _write(tmp_path, lambda d: d["geometry"].update(approach_length_m=450))
    monkeypatch.setenv("JUNCTIONLAB_CONFIG", str(path))
    assert load_config().geometry.approach_length_m == 450
