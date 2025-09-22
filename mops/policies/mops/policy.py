from __future__ import annotations

import os
import cma
import pathlib
import logging
import numpy as np
import importlib
import traceback
from typing import Callable, List, Union
import matplotlib.pyplot as plt
from mops.utils import get_log_dir
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

from mops.environments.utils import Action, State
from mops.policies.utils import (
    Policy,
    parse_code,
    query_llm,
    encode_image_tob64
)
from mops.utils import parse_text_prompt, save_log, write_prompt

_, _ = Action(), State()
log = logging.getLogger(__name__)

FUNC_NAME = "gen_plan"
FUNC_DOMAIN = "gen_initial_guess"

def reshape_like(template: list, flat_list: list) -> list:
    def helper(template):
        if isinstance(template, list):
            return [helper(item) for item in template]
        else:
            return flat_list.pop(0)
    
    # Make a copy to avoid modifying original
    flat_list = flat_list.copy()
    return helper(template)

def flatten(nested: list) -> list:
    flat = []
    for item in nested:
        if isinstance(item, list):
            flat.extend(flatten(item))
        else:
            flat.append(item)
    return flat


def bbo_on_motion_plan(
    env,
    initial_state,
    plan_gen: Callable[[List[Union[int, float]]], List],
    domains_gen: Callable,
    use_komo: bool = False,
    max_evals: int = 1000,
    cma_sigma: float = .01,
    optimizer: str = "cma"  # or "hill_climb"
) -> Union[List, str]:
    failure_message = ""

    domains = domains_gen(initial_state)
    initial_x = [v for _, v in domains.items()]

    cost_history = []
    info = None  # Future work

    def compute_cost(input_vec: np.ndarray) -> float:
        input_vec_reshaped = reshape_like(initial_x, list(input_vec))
        env.reset()
        ground_plan = plan_gen(initial_state, *input_vec_reshaped)
        if use_komo:
            env.step_komo(ground_plan, vis=False)
        else:
            for action in ground_plan:
                env.step(action, vis=False)
        cost = env.compute_cost()
        cost_history.append(cost)
        
        print(f"Input vec {np.round(input_vec, 3)}; Cost {np.round(cost, 3)}")
        
        return cost

    # Init plan eval
    env.reset()
    ground_plan = plan_gen(initial_state, *initial_x)
    for action in ground_plan:
        env.step(action, vis=False)

    flat_init = flatten(initial_x)

    if optimizer == "hill_climb":
        print(f"Running Hill Climbing with {max_evals} evaluations")
        x_best = flat_init
        y_best = compute_cost(flat_init)
        step_size = 0.05

        eval_count = 0

        while eval_count < max_evals:
            improved = False
            for dim in range(len(x_best)):
                for direction in [+1, -1]:
                    x_new = x_best.copy()
                    x_new[dim] += direction * step_size
                    y_new = compute_cost(x_new)
                    eval_count += 1  # <-- increment here

                    if y_new < y_best:
                        x_best = x_new
                        y_best = y_new
                        improved = True
                        print(f"Improved by stepping {direction * step_size} in dimension {dim}")
                        break  # Found improvement, restart dim-loop

                    if eval_count >= max_evals:
                        break  # stop if reached max_evals
                if improved or eval_count >= max_evals:
                    break

            if not improved:
                step_size *= 0.5
                print(f"No improvement, reducing step size to {step_size}")
                if step_size < 1e-4:
                    print("Step size small.")

        best_x = x_best
        best_y = y_best

    elif optimizer == "cma":
        print(f"Running CMA-ES with {max_evals} evaluations")
        bbo_options = {
            "maxfevals": max_evals,
            "ftarget": 0,
            "CMA_active": True,
        }
        best_x, es_result = cma.fmin2(compute_cost, x0=flat_init, sigma0=cma_sigma, options=bbo_options)
        best_y = es_result.result.fbest

        # Convert best_x to a list if it's a numpy array
        if isinstance(best_x, np.ndarray):
            best_x = best_x.tolist()

    elif optimizer == "random":
        print(f"Running Random Sampling with {max_evals} evaluations")
        
        # Evaluate initial point first
        x_best = flat_init
        y_best = compute_cost(flat_init)
        eval_count = 1  # Count initial evaluation
        
        # Set exploration range around initial point
        param_ranges = []
        for i in range(len(flat_init)):
            # Default exploration range is ±0.125 around initial value
            param_ranges.append((flat_init[i] - 0.025, flat_init[i] + 0.025))
            
        # Random sampling for exactly (max_evals-1) more evaluations
        while eval_count < 1000:
            # Generate random sample within parameter ranges
            x_sample = [np.random.uniform(low, high) for low, high in param_ranges]
            
            # Evaluate the sample
            y_sample = compute_cost(x_sample)
            eval_count += 1
            
            # Update best if better
            if y_sample < y_best:
                x_best = x_sample
                y_best = y_sample
                print(f"Evaluation {eval_count}/{max_evals}: Found new best solution with cost {y_best:.4f}")
        
        best_x = x_best
        best_y = y_best
        print(f"Random sampling complete: {eval_count}/{max_evals} evaluations performed")
        
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Use 'hill_climb' or 'cma'.")

    # Generate final plan
    ground_plan = plan_gen(initial_state, *reshape_like(initial_x, best_x))
    
    # Print summary information
    print(f"Optimization completed")
    print(f"Total evaluations: {len(cost_history)}")
    print(f"Best solution: {best_x}")
    print(f"Best cost: {best_y}")
    
    return ground_plan, failure_message, best_x, cost_history, info

def import_constants_from_class(cls):
    module_name = cls.__module__
    module = importlib.import_module(module_name)
    for attribute_name in module.__all__:
        globals()[attribute_name] = getattr(module, attribute_name)
        print(f"Imported {attribute_name}: {globals()[attribute_name]}")


class MOPS(Policy):
    def __init__(
        self,
        twin=None,
        max_feedbacks=0,
        seed=0,
        max_evals=50,
        use_cache=True,
        use_komo=False,
        **kwargs,
    ):
        self.twin = twin
        self.seed = seed
        self.max_feedbacks = max_feedbacks
        self.max_evals = max_evals
        self.use_komo = use_komo
        self.use_cache = use_cache

        self.queryLLM = kwargs["queryLLM"]
        self.task_name = kwargs["task_name"]
        self.optimizer = kwargs["optimizer"]
        self.cma_sigma = kwargs["cma_sigma"]
        self.cost_threshold = kwargs["cost_thresh"]

        import_constants_from_class(twin.__class__)

        prompt_fn = "prompt_{}".format(twin.__class__.__name__)
        prompt_path = os.path.join(pathlib.Path(__file__).parent, "{}.txt".format(prompt_fn))
        self.prompt = parse_text_prompt(prompt_path)
        self.plan = None

    def get_action(self, belief, goal: str):
        statistics = {}
        if self.plan is None:
            ground_plan, statistics = self.full_query_bbo(belief, goal)
            if ground_plan is None:
                return None, statistics
            elif self.use_komo:
                self.plan = ground_plan
                return self.plan, statistics
            else:
                self.plan = ground_plan[1:]
                return ground_plan[0], statistics
        elif self.use_komo:
            return self.plan, statistics
        elif len(self.plan) > 0:
            next_action = self.plan[0]
            self.plan = self.plan[1:]
            return next_action, statistics
        return None, statistics

    def full_query_bbo(self, belief, task):
        self.twin.reset()
        content = "initial={}\nGoal: {}".format(str(belief), task)
        chat_history = self.prompt + [{"role": "user", "content": content}]
        statistics = {
            "bbo_evals": 0,
            "bbo_solve_time": 0,
            "llm_query_time": 0,
            "num_bbo_evals": 0,
        }

        input_fn = "llm_input.txt"
        output_fn = "llm_output.txt"
        write_prompt(input_fn, chat_history)

        attempts = 0
        success = False
        ground_plan = None

        while attempts <= self.max_feedbacks and not success:
            
            llm_response = None
            llm_query_time = 0

            if not self.queryLLM:
                llm_response = open(f"./llm_outs_examples/{self.task_name}.txt", 'r').read()
            else:
                llm_response, llm_query_time = query_llm(chat_history, seed=self.seed)

            #####################################################

            statistics["llm_query_time"] += llm_query_time
            write_prompt("llm_input.txt", chat_history)
            chat_history.append({"role": "assistant", "content": llm_response})
            save_log(output_fn, llm_response)
            try:
                llm_code = parse_code(llm_response)
                exec(llm_code, globals())
                komo_generator = globals()[FUNC_NAME]
                if FUNC_DOMAIN in globals():
                    guess_generator = globals()[FUNC_DOMAIN]
                    ground_plan, failure_message, best_x, cost_history, info = bbo_on_motion_plan(
                        self.twin,
                        belief,
                        komo_generator,
                        guess_generator,
                        max_evals=self.max_evals,
                        use_komo=self.use_komo,
                        optimizer=self.optimizer,
                        cma_sigma=self.cma_sigma,
                    )
                else:
                    log.info("No variables provided to optimize. Continuing without ES.")
                    ground_plan = komo_generator(belief)

                np.save(os.path.join(get_log_dir(), f"cost_history_mops_{attempts}_{self.optimizer}.npy"), np.array(cost_history))

                # Evaluate the generated plan
                self.twin.reset()
                if self.use_komo:
                    self.twin.step_komo(ground_plan, vis=False)
                else:
                    for action in ground_plan:
                        self.twin.step(action, vis=False)
                cost = self.twin.compute_cost()
                state = self.twin.getState()
                self.twin.render()
                plt.imshow(self.twin.image_without_background)
                plt.axis("off")
                plt.savefig(os.path.join(get_log_dir(), f"result_{attempts}.png"))
                if cost < 2e-2:
                    success = True
                    chat_history.append({
                        "role": "user",
                        "content": f"Your code is runnable and achieved a cost of {cost} which is acceptable."
                    })
                    print(f"Attempt {attempts}: Solved with cost {cost}!")
                    break
                else:
                    print(f"Attempt {attempts}: Generated plan cost too high: {cost}")
                    img = encode_image_tob64("result.png")
                    feedback_str = (f"The best parameters that were found based on your solution are {best_x} "
                                    f"and have a cost of {cost}, which is above the target cost of <= {self.cost_threshold} "
                                    f"Please revise your solution accordingly."
                                    f"The final state after running the best solution is: {state}."
                                    f"You got this additional info: {info}.")
                    feedback = {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": f"{feedback_str} Image of final state after attempt {attempts}"},
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{img}",
                            },
                        ],
                    }
                    print(feedback)
                    chat_history.append(feedback)
                    attempts += 1

            except Exception as e:
                error_message = traceback.format_exc()
                print(error_message)
                exit()
                print(f"Attempt {attempts}: Code execution error:\n{error_message}")
                chat_history.append({
                    "role": "user",
                    "content": f"That didn't work! Error: {error_message}"
                })
                attempts += 1

        return ground_plan, statistics
