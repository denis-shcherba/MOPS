from __future__ import annotations

import base64
import logging
import os
import pathlib
import random
import re
import time
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple

import dotenv
import numpy as np
import openai
from openai import OpenAI

from mops.environments.utils import Action, Environment

log = logging.getLogger(__name__)

env_file = os.path.join(pathlib.Path(__file__).parent.parent.parent, ".env")
dotenv.load_dotenv(env_file, override=True)
openai_api_key = os.environ.get("OPENAI_KEY")
print(openai_api_key)
openai_client = OpenAI(api_key=openai_api_key)

ENGINE = "gpt-4o-mini"  # "gpt-4-turbo-2024-04-09"  # "gpt-3.5-turbo"


def encode_image_tob64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def add_gaussian_noise(ground_plan: List[Action], std: float) -> List[Action]:
    noised_plan = []
    for action in ground_plan:
        noised_params = []
        for param in action.params:
            if isinstance(param, float) or isinstance(param, int):
                noised_params.append(param + np.random.normal(0, std, 1)[0])
            else:
                noised_params.append(param)

        noised_plan.append(Action(action.name, noised_params))
    return noised_plan


def guassian_rejection_sample(
    env: Environment,
    ground_plan: List[Action],
    max_noise: float = 1.0,
    max_attempts: int = 10000,
) -> Tuple[List[Action], int]:
    """A constraint satisfaction strategy that randomly samples input vectors
    until it finds one that satisfies the constraints.

    If none are found, it returns the most common mode of failure.

    This function also returns the number of CSP samples
    """
    violation_modes = Counter()
    for i in range(max_attempts):
        log.info(f"GCSP Sampling iter {i}")
        _ = env.reset()
        std = i / float(max_attempts) * max_noise
        noised_plan = add_gaussian_noise(ground_plan, std=std * env.param_scale)
        constraint_violated = False
        log.info(noised_plan)
        for ai, action in enumerate(noised_plan):
            _, _, _, info = env.step(action)
            if len(info["constraint_violations"]) > 0:
                violation_str = [
                    "Step {}, Action {}, Violation: {}".format(
                        ai, action.name, violation
                    )
                    for violation in info["constraint_violations"]
                ]
                violation_modes.update(violation_str)
                constraint_violated = True
                log.info(f"Constraint violation " + str(info["constraint_violations"]))
                break
        if not constraint_violated:
            return noised_plan, i

    return None, i


def parse_code(input_text):
    pattern = "```python(.*?)```"
    matches = re.findall(pattern, input_text, re.DOTALL)
    if len(matches) == 0:
        return None

    all_code = ""
    for match in matches:
        all_code += "\n" + match
    return all_code


def query_llm(messages, seed, max_retries=5):
    retry_count = 0
    backoff_factor = 60
    while True:
        try:
            st = time.time()
            output = openai_client.responses.create( #chat.completions.create(
                model=ENGINE,
                input=messages,
                temperature=0.,
                # seed=seed,
                # n=1,
                # stop=None
            )

            return str(output.output_text), time.time() - st
        except openai.RateLimitError as e:
            retry_count += 1
            if retry_count > max_retries:
                raise e
            sleep_time = backoff_factor * (2**retry_count)
            print(f"Rate limit exceeded. Retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)


@dataclass
class Sampler:
    def sample(self):
        pass


@dataclass
class ContinuousSampler(Sampler):
    min: float = 0
    max: float = 1
    shape: tuple = None

    def sample(self):
        if self.shape == None:
            return random.uniform(self.min, self.max)
        return np.random.uniform(self.min, self.max, self.shape).tolist()


@dataclass
class DiscreteSampler:
    values: List[int]

    def sample(self):
        return random.choice(self.values)


class Policy(ABC):
    @abstractmethod
    def __init__(self, twin: Environment):
        self.twin = twin

    @abstractmethod
    def get_action(self, belief, goal: str, profile_stats={}):
        pass
