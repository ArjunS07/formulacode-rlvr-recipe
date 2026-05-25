"""
Generation-only debug entrypoint for FormulaCode.
Runs a small number of trials without a gradient update — use this first
to confirm Docker containers start, terminus-2 connects, and rewards appear.
"""

import sys
import asyncio
from pathlib import Path

import ray
import yaml
from loguru import logger

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl.train.generators.base import GeneratorInput, TrajectoryID
from examples.train_integrations.harbor.entrypoints.main_harbor import (
    HarborSkyRLConfig,
    _deep_merge,
)
from examples.train_integrations.harbor.harbor_generator import HarborGenerator
from examples.train_integrations.harbor.dataset import HarborTaskDataset
from examples.train_integrations.harbor.entrypoints.main_harbor import HarborExp

HARBOR_DEFAULT_CONFIG = Path(__file__).parent.parent / "harbor_trial_config" / "docker.yaml"

NUM_SAMPLES_TO_TEST = 2  # Run 2 trials — enough to confirm pipeline without long wait


class FormulaCodeGenerateExp(HarborExp):
    def _setup_generator(self):
        logger.info("FormulaCode gen-debug: starting inference engine and generator")
        inference_engine_client = self.get_inference_client()
        asyncio.run(inference_engine_client.wake_up())
        return self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

    def run(self):
        generator = self._setup_generator()
        prompts, trajectory_ids = [], []
        for item in self.train_dataset:
            prompts.append(item["prompt"])
            trajectory_ids.append(TrajectoryID(instance_id=item["uid"], repetition_id=0))

        input_batch = GeneratorInput(
            prompts=prompts[:NUM_SAMPLES_TO_TEST],
            trajectory_ids=trajectory_ids[:NUM_SAMPLES_TO_TEST],
            env_classes=None,
            env_extras=None,
            sampling_params=None,
        )
        logger.info(f"Running {NUM_SAMPLES_TO_TEST} trial(s)...")
        asyncio.run(generator.generate(input_batch))
        logger.info("Gen-debug complete. Check logs above for rewards.")


@ray.remote(num_cpus=1)
def skyrl_gen_entrypoint(cfg):
    exp = FormulaCodeGenerateExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    if cfg.trainer.algorithm.max_seq_len is None:
        raise ValueError("trainer.algorithm.max_seq_len must be set.")
    initialize_ray(cfg)
    ray.get(skyrl_gen_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
