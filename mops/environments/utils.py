from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import robotic as ry


@dataclass
class Action:
    name: str = "default"
    params: List[float] = field(default_factory=list)


def parse_lisp(action_str: str) -> Action:
    assert action_str[0] == "(" and action_str[-1] == ")"
    parts = action_str[1:-1].split(" ")
    return Action(parts[0], [float(p) for p in parts[1:]])


@dataclass
class State:
    pass


class Task(ABC):
    @abstractmethod
    def get_goal(self):
        pass

    @abstractmethod
    def get_reward(self, env):
        pass

    @abstractmethod
    def get_cost(self, env):
        pass

    def setup_cfg(self, vertical_blocks: bool = True, multi: bool = False, **kwargs) -> ry.Config:
        C = ry.Config()
        C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

        C.delFrame("panda_collCameraWrist")
        C.getFrame("table").setShape(ry.ST.ssBox, size=[1., 1., .1, .02])

        names = ["red", "green", "blue"]

        # Objects
        if multi:
            for k in range(3):
                for i in range(3):
                    color = [0., 0., 0.]
                    color[i % 3] = 1.
                    size_xyz = [.04, .04, .12] if vertical_blocks else [.04, .12, .04]
                    C.addFrame(f"block_{names[i]}_{k}") \
                        .setPosition([(i % 3) * .15, (i // 3) * .1 + k * .1, .71]) \
                        .setShape(ry.ST.ssBox, size=[*size_xyz, 0.005]) \
                        .setColor(color) \
                        .setContact(1) \
                        .setMass(.1)
        else:
            for i in range(3):
                color = [0., 0., 0.]
                color[i % 3] = 1.
                size_xyz = [.04, .04, .12] if vertical_blocks else [.04, .12, .04]
                C.addFrame(f"block_{names[i]}") \
                    .setPosition([(i % 3) * .15, (i // 3) * .1, .71]) \
                    .setShape(ry.ST.ssBox, size=[*size_xyz, 0.005]) \
                    .setColor(color) \
                    .setContact(1) \
                    .setMass(.1)
        # C.view(True)

        return C


class Updater(ABC):
    def __init__(self):
        pass

    def update(self, obs):
        raise NotImplementedError


class DefaultUpdater(Updater):
    def __init__(self):
        pass

    def update(self, obs):
        return obs


class Environment(ABC):
    @abstractmethod
    def __init__(self, task: Task = None, **kwargs):
        self.task = task
        self.param_scale = 1

    @abstractmethod
    def step(self, action: Action, return_belief: bool = False, profile_stats={}):
        raise NotImplementedError

    @abstractmethod
    def sample_twin(env, obs, task) -> Environment:
        raise NotImplementedError

    def render(self):
        pass

    def reset(self):
        pass

    def close(self):
        pass
