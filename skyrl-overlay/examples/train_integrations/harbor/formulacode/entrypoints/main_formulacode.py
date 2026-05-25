"""
Training entrypoint for FormulaCode RLVR.

Identical to main_harbor.py except HARBOR_DEFAULT_CONFIG points to
docker.yaml (terminus-2 xml parser, Docker environment).
"""

import sys
from pathlib import Path

import ray
import yaml

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HarborExp,
    HarborSkyRLConfig,
    _deep_merge,
    skyrl_entrypoint,
)

HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "docker.yaml"


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError(
            "trainer.algorithm.max_seq_len must be set — must match max_model_len."
        )
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
