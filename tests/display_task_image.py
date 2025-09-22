import numpy as np
import hydra
import pathlib
from mops.environments.utils import Environment, Task, Updater
from omegaconf import OmegaConf

@hydra.main(
    version_base=None,
    config_path="../mops/config",  # relative to your project root
    config_name="mops_draw_star.yaml"
)
def main(cfg):
    # Load the action sequence
    action_sequence = np.load("action_sequence.npy", allow_pickle=True)

    # Set up environment and task (no policy/updater needed)
    task: Task = hydra.utils.instantiate(cfg.task)
    env: Environment = hydra.utils.instantiate(
        cfg.env, task=task, render=cfg.render, use_komo=cfg.get("use_komo", False)
    )
    obs = env.reset()
    for i, action in enumerate(action_sequence):
        print(f"Replaying action {i}: {action}")
        obs, reward, done, info = env.step(action, vis=cfg.get("render", False))
        if done:
            print(f"Episode finished after {i+1} actions.")
            break

    env.render()
    env.close()

    

if __name__ == "__main__":
    main()