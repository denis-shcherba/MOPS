from __future__ import annotations

import importlib
import logging
import math
import os
import pathlib
import random
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Union

from matplotlib import pyplot as plt
from mops.utils import get_log_dir

import numpy as np

from mops.environments.utils import Action, Environment, State
from mops.policies.utils import (
    ContinuousSampler,
    DiscreteSampler,
    Policy,
    Sampler,
    encode_image_tob64,
    parse_code,
    query_llm,
)
from mops.utils import (
    are_files_identical,
    get_log_dir,
    get_previous_log_folder,
    parse_text_prompt,
    read_file,
    save_log,
    write_prompt,
)

_, _ = Action(), State()
log = logging.getLogger(__name__)


FUNC_NAME = "gen_plan"
FUNC_DOMAIN = "gen_domain"


def rejection_sample_csp(
    env: Environment,
    initial_state: State,
    plan_gen: Callable[[List[Union[int, float]]], List[Action]],
    domains_gen: List[Sampler],
    max_attempts: int = 10,
    cost_threshold: float = 50,
) -> Union[List[Action], str]:
    """A constraint satisfaction strategy that randomly samples input vectors
    until it finds one that satisfies the constraints.

    If none are found, it returns the most common mode of failure.
    """
    lowest_cost = 1e7
    best_plan = None
    violation_modes = Counter()
    cost_history = []  
    best_state = None
    best_action = None
    info = None

    for i in range(max_attempts):

        log.info(f"CSP Sampling iter {i}")
        domains = domains_gen(initial_state)
        input_vec = {name: domain.sample() for name, domain in domains.items()}
        state = env.reset()
        ground_plan = plan_gen(initial_state, **input_vec)
        constraint_violated = False

        # Rollout the entire action plan
        for ai, action in enumerate(ground_plan):
            state, _, _, info = env.step(action, vis=False)
        cost = env.compute_cost()
        
        cost_history.append(cost)  
        if cost < lowest_cost:
            lowest_cost = cost
            best_state = state
            best_action = action
            best_plan = ground_plan

        # Terminate early if everything is already ok
        if cost < cost_threshold:
            constraint_violated = False
            break
        else:
            constraint_violated = True

        if not constraint_violated:
            print(f"Solved problem at iter {i}. Cost {cost}")
            return ground_plan, None, i
        else:
            print(f"Finished iter {i}. Cost {cost}, (lowest cost {lowest_cost})")

    return best_plan, violation_modes, i, lowest_cost, best_state, best_action, np.array(cost_history), info


def import_constants_from_class(cls):
    # Get the module name from the class
    module_name = cls.__module__

    # Dynamically import the module
    module = importlib.import_module(module_name)

    # Import all uppercase attributes (assuming these are constants)
    for attribute_name in module.__all__:
        # Importing the attribute into the global namespace
        globals()[attribute_name] = getattr(module, attribute_name)
        print(f"Imported {attribute_name}: {globals()[attribute_name]}")


class Proc3s(Policy):
    def __init__(
        self,
        twin=None,
        max_feedbacks=0,
        seed=0,
        max_csp_samples=10000,
        use_cache=False,
        **kwargs,
    ):
        self.twin = twin
        self.seed = seed
        self.max_feedbacks = max_feedbacks
        self.max_csp_samples = max_csp_samples

        self.use_cache = use_cache

        import_constants_from_class(twin.__class__)
        
        # Get environment specific prompt
        prompt_fn = "prompt_{}".format(twin.__class__.__name__)
        prompt_path = os.path.join(
            pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn)
        )

        self.prompt = parse_text_prompt(prompt_path)

        self.cost_threshold = 1
        self.plan = None

    def get_action(self, belief, goal: str):
        statistics = {}
        if self.plan is None:
            # No plan yet, we need to come up with one
            ground_plan, statistics = self.full_query_csp(belief, goal)
            if ground_plan is None:
                return None, statistics
            else:
                log.info("Found plan: {}".format(ground_plan))
                self.plan = ground_plan[1:]
                return ground_plan[0], statistics
        
        elif len(self.plan) > 0:
            next_action = self.plan[0]
            self.plan = self.plan[1:]
            return next_action, statistics
        
        else:
            return None, statistics

    def full_query_csp(self, belief, task):
        _ = self.twin.reset()
        content = "Goal: {}".format(task)
        content = "State: {}\n".format(str(belief)) + content
        chat_history = self.prompt + [{"role": "user", "content": content}]
        statistics = {}
        statistics["csp_samples"] = 0
        statistics["csp_solve_time"] = 0
        statistics["llm_query_time"] = 0

        for iter in range(self.max_feedbacks + 1):
            print(iter)
            statistics["num_feedbacks"] = iter
            st = time.time()
            input_fn = f"llm_input_{iter}.txt"
            output_fn = f"llm_output_{iter}.txt"
            write_prompt(input_fn, chat_history)  # writing the history here

            # Check if the inputs match
            parent_log_folder = os.path.join(get_log_dir(), "..")
            previous_folder = get_previous_log_folder(parent_log_folder)
            llm_query_time = 0
            if (
                self.use_cache
                and os.path.isfile(os.path.join(previous_folder, output_fn))
                and are_files_identical(
                    os.path.join(previous_folder, input_fn),
                    os.path.join(get_log_dir(), input_fn),
                )
            ):
                log.info("Loading cached LLM response")
                llm_response = read_file(os.path.join(previous_folder, output_fn))
            else:
                log.info("Querying LLM")
                
                llm_response, llm_query_time = query_llm(chat_history, seed=self.seed)
                #####################################################
                # llm_response = open("./triangle_hlvlsr_proc3s.txt", 'r').read()
                # llm_response = open("./bridge_hlvlsr_proc3s.txt", 'r').read()
                # llm_response = open("./llm_outs/PentagonProc3s.txt", 'r').read()
                # llm_query_time = 0
                #####################################################
            print(llm_response)

            statistics["llm_query_time"] += llm_query_time

            chat_history.append({"role": "assistant", "content": llm_response})
            save_log(output_fn, llm_response)

            error_message = None
            ground_plan = None

            try:
                llm_code = parse_code(llm_response)
                exec(llm_code, globals())
                func = globals()[FUNC_NAME]
                domain = globals()[FUNC_DOMAIN]
                st = time.time()
                ground_plan, failure_message, csp_samples, cost, state, best_x, cost_over_time, info = rejection_sample_csp(
                    self.twin,
                    belief,
                    func,
                    domain,
                    max_attempts=self.max_csp_samples,
                    cost_threshold=self.cost_threshold,
                )
                np.save(os.path.join(get_log_dir(), f"cost_history_proc3s_{iter}.npy"), cost_over_time)    #return None, violation_modes, i

                statistics["csp_samples"] += csp_samples
                statistics["csp_solve_time"] += time.time() - st


                #plt.show()
                # Evaluate the generated plan
                self.twin.reset()

                if ground_plan is None:
                    if error_message is not None:
                        failure_response = error_message
                    else:
                        failure_response = ""
                        for fm, count in failure_message.most_common(2):
                            failure_response += f"{count} occurences: {fm}\n"

                    save_log(f"feedback_output_{iter}.txt", failure_response)
                    chat_history.append({"role": "user", "content": failure_response})
                    continue

                for action in ground_plan:
                    self.twin.step(action, vis=False)
                cost = self.twin.compute_cost()
                state = self.twin.getState()
                self.twin.render()



                image = self.twin.image_without_background
                plt.imshow(image)

                plt.imsave(os.path.join(get_log_dir(), f"image_{iter}.png"), image)
                plt.axis('off')  # Turn off axis labels

                if ground_plan is not None and error_message is None and cost <= self.cost_threshold:
                    return ground_plan, statistics
                
                else:
                    # Feedback for plan cost being too high (csp_final_cost >= cost_threshold)
                    log.warning(f"Feedback {iter}: Plan cost {cost:.4f} >= {self.cost_threshold}")
                    
                
                    img = encode_image_tob64("result.png")

                    

                    feedback_str = (f"The best parameters that were found based on your solution are {best_x} "
                                    f"and have a cost of {cost}, which is above the target cost of <= {self.cost_threshold}. "
                                    f"Please revise your solution accordingly."
                                    f"The final state after running the best solution is: {state}."
                                    f"You got this additional info: {info}.")

                    feedback = {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"{feedback_str} Image of final state after attempt {iter}"},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{img}",
                            },
                        ],
                    }

                    print(feedback)
                    chat_history.append(feedback)

            except Exception as e:
                # Get the traceback as a string
                print(e)
                error_message = traceback.format_exc()
                log.info("Code error: " + str(error_message))

        return ground_plan, statistics