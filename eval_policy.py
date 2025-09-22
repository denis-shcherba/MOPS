import json
import logging
import os
import pathlib
import sys
import time

import hydra
import omegaconf
import numpy as np
from dotenv import load_dotenv
from rich.pretty import pprint

from mops.environments.utils import Environment, Task, Updater
from mops.policies.utils import Policy
from mops.utils import get_log_dir


load_dotenv()
log = logging.getLogger(__name__)


class StreamToLogger:
    def __init__(self, logger, log_level):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_dir = get_log_dir()
    log_file = os.path.join(log_dir, f"output.log")

    formatter = logging.Formatter("%(message)s")

    # FileHandler: only log errors to file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.ERROR)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Do NOT add a StreamHandler here!

    # Redirect stdout and stderr
    sys.stdout = StreamToLogger(logger, logging.INFO)   # print() as INFO
    sys.stderr = StreamToLogger(logger, logging.ERROR)  # errors as ERROR

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("mops", "config")),
    config_name="mops_draw_star"
)
def main(cfg: omegaconf.DictConfig):

    log.info(" ".join(sys.argv))

    setup_logger()

    # if cfg.get("seed") is not None:
    #     random.seed(cfg["seed"])
    #     np.random.seed(cfg["seed"])

    use_komo = cfg.get("use_komo")

    log.info("Setting up environment and policy...")
    pprint(cfg)
    task: Task = hydra.utils.instantiate(cfg.task)
    updater: Updater = hydra.utils.instantiate(cfg.updater)
    env: Environment = hydra.utils.instantiate(
        cfg.env, task=task, render=cfg.render and not cfg.vis_debug, use_komo=use_komo,
    )
    obs = env.reset()

    belief = updater.update(obs)

    twin_env: Environment = hydra.utils.get_class(cfg.env._target_).sample_twin(
        env, belief, task, render=cfg.vis_debug
    )

    policy_kwargs = {
        "optimizer": cfg.get("optimizer", "cma"),
        "queryLLM": cfg.get("queryLLM", True),
        "task_name": cfg.get("task", {}).get("task_name", None),
        "cma_sigma": cfg.get("cma_sigma", 1e-2),
        "cost_thresh": cfg.get("cost_thresh", 1e-2),
    }

    policy: Policy = hydra.utils.instantiate(
        cfg.policy, twin=twin_env, seed=cfg["seed"], use_komo=use_komo, **policy_kwargs
    )

    statistics = {"execution_time": 0, "planning_time": 0}
    
    action_sequence = []

    for i in range(cfg.get("max_env_steps")):
        log.info("Step " + str(i))
        goal = env.task.get_goal()
        log.info("Goal: " + str(goal))
        belief = updater.update(obs)
        # log.info("Scene: " + str(belief))
        st = time.time()
        action, step_statistics = policy.get_action(belief, goal)
        for k, v in step_statistics.items():
            statistics["step_{}_{}".format(i, k)] = v
        statistics["planning_time"] += time.time() - st
        log.info("Action: " + str(action))
        if action is None:
            break
        
        st = time.time()
        obs, reward, done, info = env.step(action, vis=False)
        for k, v in info.items():
            statistics["step_{}_{}".format(i, k)] = v
        statistics["execution_time"] += time.time() - st

        action_sequence.append(action)

        if cfg.render:
            import matplotlib.pyplot as plt
            env.render()
            plt.imshow(env.image_without_background)
            plt.axis("off")
            plt.savefig(os.path.join(get_log_dir(), "end_image.png"))

        log.info("Reward: " + str(reward))
        log.info("Done: " + str(done))
        log.info("Info: " + str(info))

    np.save(os.path.join(get_log_dir(), "action_sequence.npy"), action_sequence)
    env.close()
    log.info("Statistics: " + str(json.dumps(statistics)))


if __name__ == "__main__":
    main()
