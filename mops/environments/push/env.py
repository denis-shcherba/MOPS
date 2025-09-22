from __future__ import annotations

import copy
import time
import rowan
import logging
import numpy as np
import robotic as ry
from typing import List, Tuple

import mops.environments.push.manipulation as manip
from mops.environments.push.simulator import Simulator
from mops.environments.utils import Action, Environment, State, Task
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
# Used as imports for the LLM-generated code
__all__ = ["Frame", "PushState"]


@dataclass
class Frame:
    name: str
    x_pos: float
    y_pos: float
    z_pos: float
    x_rot: float
    y_rot: float
    z_rot: float
    size: float | list[float]
    color: list[float]

    def __str__(self):
        return (
            f'Frame('
            f'name="{self.name}", '
            f'x_pos={round(self.x_pos, 2)}, '
            f'y_pos={round(self.y_pos, 2)}, '
            f'z_pos={round(self.z_pos, 2)}, '
            f'x_rot={round(self.x_rot, 2)}, '
            f'y_rot={round(self.y_rot, 2)}, '
            f'z_rot={round(self.z_rot, 2)}, '
            f'size={round(self.size, 2) if isinstance(self.size, float) else [round(s, 2) for s in self.size]}, '
            f'color="{[round(c, 2) for c in self.color]}")'
        )


@dataclass
class PushState(State):
    config: ry.Config
    frames: List[Frame] = field(default_factory=list)

    def __str__(self):
        return "PushState(frames=[{}])".format(
            ", ".join([str(o) for o in self.frames])
        )

    def getFrame(self, name: str) -> Frame:
        for f in self.frames:
            if f.name == name:
                return f
        raise ValueError("Push state {} not found".format(name))
    

def pick_place_manipulation(C: ry.Config,
                            frame_name: str,
                            pick_dir: str,
                            place_dir: str,
                            pos: Tuple[float],
                            yaw: float,
                            compute_collisions: bool=True) -> manip.ManipulationModelling:
    x, y, z = pos
    M = manip.ManipulationModelling()
    M.setup_pick_and_place_waypoints(C, "l_gripper", frame_name, accumulated_collisions=compute_collisions)
    
    M.grasp_box(1., "l_gripper", frame_name, "l_palm", pick_dir)

    if z == None:
        M.place_box(2., frame_name, "table", "l_palm", place_dir)
        M.target_relative_xy_position(2., frame_name, "table", [x, y])
    else:
        table_frame = C.getFrame("table")
        table_offset = table_frame.getPosition()[2] + table_frame.getSize()[2]*.5
        if z < table_offset:
            z += table_offset
        M.place_box(2., frame_name, "table", "l_palm", place_dir, on_table=False)
        M.target_position(2., frame_name, [x, y, z])

    if yaw != None:
        
        if place_dir == "x" or place_dir == "xNeg":
            feature = ry.FS.vectorY
        
        elif place_dir == "y" or place_dir == "yNeg":
            feature = ry.FS.vectorX
        
        elif place_dir == "z" or place_dir == "zNeg":
            feature = ry.FS.vectorY
        
        else:
            raise Exception(f"'{place_dir}' is not a valid up vector for a place motion!")

        yaw += np.pi*.5
        target = np.array([np.cos(yaw), -np.sin(yaw), .0])
        if "Neg" in place_dir: target *= -1
        M.komo.addObjective([2.], feature, [frame_name], ry.OT.eq, [1e1], target)
    
    return M


def straight_push(C: ry.Config, start: np.ndarray, end: np.ndarray) -> ry.KOMO:

    table = C.getFrame("table")
    height = table.getPosition()[2] + table.getSize()[2]*.5 + .05

    start = np.append(start, height)
    end = np.append(end, height)
    delta = end - start
    delta /= np.linalg.norm(delta)

    C.addFrame("start_frame") \
        .setPosition([start[0], start[1], height]) \
        .setShape(ry.ST.marker, [.05]) \
        .setColor([1., 0., 0.])
    C.addFrame("end_frame") \
        .setPosition([end[0], end[1], height]) \
        .setShape(ry.ST.marker, [.05]) \
        .setColor([0., 0., 1.])
    
    qHome = C.getJointState()

    komo = ry.KOMO()
    komo.setConfig(C, False)
    komo.setTiming(2, 32, 10, 2)

    komo.addControlObjective([], 1, 1e-1)
    komo.addControlObjective([], 2, 1e-1)

    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq, [1e0])

    if C.getFrame("wall", warnIfNotExist=False):
        komo.addObjective([], ry.FS.negDistance, ["l_palm", "wall"], ry.OT.ineq, [1e1], [-.01])
    komo.addObjective([], ry.FS.negDistance, ["l_palm", "table"], ry.OT.ineq, [1e1], [-.01])

    mat = np.eye(3) - np.outer(delta, delta)
    komo.addObjective([1., 2], ry.FS.positionDiff, ["l_gripper", "start_frame"], ry.OT.eq, mat)
    komo.addObjective([1.], ry.FS.positionDiff, ["l_gripper", "start_frame"], ry.OT.eq, [1e1])
    komo.addObjective([2.], ry.FS.positionDiff, ["l_gripper", "end_frame"], ry.OT.eq, [1e1])
    
    return komo


class PushEnv(Environment):
    def __init__(self, task: Task, **kwargs):

        super().__init__(task)

        self.compute_collisions = True
        
        self.base_config: ry.Config = self.task.setup_cfg()
        self.C: ry.Config = self.task.setup_cfg()
        self.initial_state = self.reset()
        self.qHome = self.C.getJointState()

        pos = [0.9819932, -0.0416917, 1.49871321]
        quat = [0.33147948, -0.75582893, -0.5163221, 0.22859457]
        
        cam = self.C.getFrame("cameraTop")
        cam.setPosition(pos)
        cam.setQuaternion(quat)
        self.C.view_setCamera(cam)
        time.sleep(.1)

    def step(self, action: Action, vis: bool=True):
        
        info = {"constraint_violations": []}
        self.path = []

        if not self.feasible:
            self.C.view()
            self.t = self.t + 1
            return self.state, False, 0, info
        
        self.feasible = False

        if action.name == "pick":
            assert self.to_be_picked == None
            self.to_be_picked = action.params
            self.feasible = True
        
        elif action.name == "place_sr":
            assert self.to_be_picked != None

            frame_name = self.to_be_picked[0]
            pick_dir = self.to_be_picked[1]
            x, y, z = action.params[:3]
            rotated, yaw = action.params[3:5]
            
            grasp_dirs = ["x", "y"] if pick_dir == None else [pick_dir]
            for grasp_dir in grasp_dirs:
                if rotated and grasp_dir == 'x':
                    place_dirs = ['y', 'yNeg']
                elif rotated and grasp_dir == 'y':
                    place_dirs = ['x', 'xNeg']
                elif not rotated:
                    place_dirs = ['z', 'zNeg']


                for place_dir in place_dirs:
                    M = pick_place_manipulation(self.C,
                                                frame_name,
                                                grasp_dir,
                                                place_dir,
                                                (x, y, z),
                                                yaw,
                                                self.compute_collisions)

                    M.solve(verbose=0)
                    if M.feasible:

                        M1 = M.sub_motion(0, accumulated_collisions=self.compute_collisions)
                        path1 = M1.solve(verbose=0)

                        M2 = M.sub_motion(1, accumulated_collisions=self.compute_collisions)
                        path2 = M2.solve(verbose=0)
                        
                        if M1.feasible and M2.feasible:
                            
                            if vis:
                                for q in path1:
                                    self.C.setJointState(q)
                                    self.C.view()
                                    time.sleep(.1)
                                self.C.attach("l_gripper", frame_name)
                                
                                for q in path2:
                                    self.C.setJointState(q)
                                    self.C.view()
                                    time.sleep(.1)
                                self.C.attach("table", frame_name)
                            
                            else:
                                self.C.setJointState(path1[-1])
                                self.C.attach("l_gripper", frame_name)
                                self.C.view()
                                self.C.setJointState(path2[-1])
                                self.C.attach("table", frame_name)
                                self.C.view()

                            self.feasible = True
                            self.to_be_picked = None
                            break

                if self.feasible:
                    break

        elif action.name == "push_motion":
            assert self.to_be_picked == None
            self.feasible = False

            start = np.array(action.params[:2])
            end = np.array(action.params[2:4])

            komo = straight_push(self.C, start, end)
            sol = ry.NLP_Solver()
            sol.setProblem(komo.nlp())
            sol.setOptions(damping=1e-1, verbose=0, stopTolerance=1e-3, maxLambda=100., stopInners=20, stopEvals=200)
            self.ret = sol.solve()

            self.C.delFrame("start_frame")
            self.C.delFrame("end_frame")
            
            if self.ret.feasible:
                self.path = komo.getPath()
                self.feasible = True

                sim = Simulator(self.C)
                xs, qs, xdots, qdots = sim.run_trajectory(self.path, 2, real_time=vis, close_gripper=False)
                del sim._sim
                del sim
                
                self.C.setJointState(qs[-1])
                self.C.setFrameState(xs[-1])

        elif action.name == "wait":
            wait_time = action.params[0]

            tau = 5e-3
            steps = int(wait_time/tau)
            sim = Simulator(self.C)
            xs, qs, _, _ = sim.step(steps)
            
            self.C.setJointState(qs[-1])
            self.C.setFrameState(xs[-1])

        else:
            raise NotImplementedError(f"action {action} not implemented")
        
        if not self.feasible:
            info["constraint_violations"].append("idk")

        self.t = self.t + 1
        self.state = self.getState()
        return self.state, False, 0, info
    
    def step_komo(self, komo: ry.KOMO, vis: bool=True):
        info = {"constraint_violations": []}

        if not self.feasible:
            self.C.view()
            self.t = self.t + 1
            return self.state, False, 0, info
        
        self.feasible = False

        sol = ry.NLP_Solver()
        sol.setProblem(komo.nlp())
        sol.setOptions(damping=1e-1, verbose=0, stopTolerance=1e-3, maxLambda=100., stopInners=20, stopEvals=200)
        ret = sol.solve()
        
        self.feasible = ret.feasible
        if not self.feasible:
            print(komo.report())
            print("KOMO not feasible")
            info["constraint_violations"].append("idk")

        C_state = komo.getPathFrames()[-1]
        q = komo.getPath()[-1]
        self.C.setFrameState(C_state)
        self.C.setJointState(q)
        
        if vis:
            komo.view_play(False, delay=.1)
        
        self.C.view()
        self.t = self.t + 1
        self.state = self.getState()
        return self.state, False, 0, info
    
    @staticmethod
    def sample_twin(real_env: PushEnv, obs, task: Task, **kwargs) -> PushEnv:
        twin = PushEnv(task)
        twin.C = ry.Config()
        twin.C.addConfigurationCopy(real_env.C)
        twin.state.frames = copy.deepcopy(obs.frames)
        twin.state.config = ry.Config()
        twin.state.config.addConfigurationCopy(obs.config)
        twin.initial_state.frames = copy.deepcopy(obs.frames)
        twin.initial_state.config = ry.Config()
        twin.initial_state.config.addConfigurationCopy(obs.config)
        return twin

    def reset(self):
        q = self.base_config.getJointState()
        C_state = self.base_config.getFrameState()
        self.C.setJointState(q)
        self.C.setFrameState(C_state)
        self.C.view()
        self.state = self.getState()
        self.t = 0
        self.feasible = True
        self.to_be_picked: List[str] = None
        return self.state
    
    def getState(self):
        state = PushState(self.C)
        state.frames = []
        
        for f in self.task.relevant_frames:
            C_frame = self.C.getFrame(f)
        
            pos = C_frame.getPosition()
            size = C_frame.getSize()
            rot = rowan.to_euler(C_frame.getQuaternion(), convention="xyz") # Rotations need further testing
            color = C_frame.getMeshColors().flatten()[:3]  # TODO: check if this is correct

            frame = Frame(f, *pos, *rot, size, color)
            state.frames.append(frame)
        
        return state

    def render(self, block: bool=True):

        self.C.view(block, f"Actions Performed Count: {self.t}")
        
        img = self.C.view_getRgb()
        return img

    def compute_cost(self):
        self.C.view()
        cost = self.task.get_cost(self)
        if not self.feasible:
            cost += 1000
        return cost
    