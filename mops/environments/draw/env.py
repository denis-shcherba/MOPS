from __future__ import annotations

import copy
from matplotlib import pyplot as plt
import rowan
import logging
import numpy as np
import robotic as ry
from typing import List

import mops.environments.push.manipulation as manip
from mops.environments.draw.tasks import isolate_red_shapes_from_rgb
from mops.environments.utils import Action, Environment, State, Task
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
# Used as imports for the LLM-generated code
__all__ = ["Frame", "DrawState"]


@dataclass
class Frame:
    name: str
    x_pos: float
    y_pos: float
    z_pos: float
    x_size: float
    y_size: float
    z_size: float
    x_rot: float
    y_rot: float
    z_rot: float


    def __str__(self):
        return (
            f''
        )



@dataclass
class DrawState(State):
    config: ry.Config
    frames: List[Frame] = field(default_factory=list)

    def __str__(self):
        return "DrawState(frames=[{}])".format(
            ", ".join([str(o) for o in self.frames])
        )

    def getFrame(self, name: str) -> Frame:
        for f in self.frames:
            if f.name == name:
                return f
        return None
    


class DrawEnv(Environment):
    def __init__(self, task: Task, **kwargs):

        super().__init__(task)

        state_dict = self.task.setup_env()


        self.C: ry.Config = state_dict["config"]
        self.C.view(False, "Working Config")
        self.initial_state = self.reset()
        self.qHome = self.C.getJointState()
        self.lines = 0
        CameraView = ry.CameraView(self.C)
        CameraView.setCamera(self.C.getFrame("cameraTop"))
        self.fx, self.fy, self.cx, self.cy = CameraView.getFxycxy()
        self.number_lines = task.number_lines

        self.whiteboard_thickness = state_dict["whiteboard_thickness"]
        self.whiteboard_tilt = state_dict["whiteboard_tilt"]
        self.whiteboard_size_x = state_dict["whiteboard_size_x"]
        self.whiteboard_size_y = state_dict["whiteboard_size_y"]
        self.draw_method = 2
        self.sphere_distance = 0.03
        self.spheres_drawn = 0

    def step(self, action: Action, vis: bool=True):
        info = {"constraint_violations": []}
        if not self.feasible:
            if vis: self.C.view()
            self.t += 1
            self.starting_points.append(np.array([np.nan, np.nan]))
            self.vectors.append(np.array([np.nan, np.nan]))
            self.state = self.getState()
            return self.state, False, 0, info

        self.feasible = False
        overall_feasible = False 

        # Initialize projection results for this step
        proj_vector = np.array([0., 0.])
        start_point_proj = np.array([0., 0.])

        if action.name == "draw_line":
            
            p1 = np.array(action.params[:2])
            p2 = np.array(action.params[2:4])
            
            if p1[0]>0 and p1[0]<self.whiteboard_size_x and p1[1]>0 and p1[1]<self.whiteboard_size_y and p2[0]>0 and p2[0]<self.whiteboard_size_x and p2[1]>0 and p2[1]<self.whiteboard_size_y:

                # Use penEnd frame if available, otherwise pen
                pen_contact_frame = "penEnd" if self.C.getFrame("penEnd", warnIfNotExist=False) else "pen"

                # --- Calculate World Coordinates (Needed for Cylinder method if used later, and projection) ---
                whiteboard_frame = self.C.getFrame("whiteboard")
                whiteboard_pos = whiteboard_frame.getPosition()
                wb_width = self.whiteboard_size_x
                wb_height = self.whiteboard_size_y
                cos_tilt = np.cos(self.whiteboard_tilt)
                sin_tilt = np.sin(self.whiteboard_tilt)
                R_wb = np.array([[1, 0, 0], [0, cos_tilt, -sin_tilt], [0, sin_tilt, cos_tilt]])
                x0_c = p1[0] - wb_width / 2
                y0_c = p1[1] - wb_height / 2
                x1_c = p2[0] - wb_width / 2
                y1_c = p2[1] - wb_height / 2
                z_offset = self.whiteboard_thickness / 2 + 0.001
                local_p0 = np.array([x0_c, y0_c, z_offset])
                local_p1 = np.array([x1_c, y1_c, z_offset])
                world_p0 = whiteboard_pos + R_wb @ local_p0
                world_p1 = whiteboard_pos + R_wb @ local_p1
                # --- End World Coordinate Calculation ---

                motion_feasible_current_step = True
                last_sphere_pos = None # Initialize for sphere drawing

                for i in range(2): # 0: move to start, 1: move to end (drawing motion)
                    target_x_rel = p1[0] if i == 0 else p2[0]
                    target_y_rel = p1[1] if i == 0 else p2[1]

                    # --- IK ---
                    if i == 1:
                        if self.C.getFrame("tmp", warnIfNotExist=False): self.C.delFrame("tmp")
                        self.C.addFrame("tmp").setPosition(self.C.getFrame(pen_contact_frame).getPosition())

                    man_ik = manip.ManipulationModelling()
                    man_ik.setup_inverse_kinematics(self.C, accumulated_collisions=False)

                    man_ik.komo.addObjective([1], ry.FS.positionRel, ["l_gripper", "whiteboard"], ry.OT.eq, scale=[0, 0, 1], target=[0, 0, .1])
                    man_ik.komo.addObjective([1], ry.FS.positionRel, ["l_gripper", "whiteboard"], ry.OT.eq, scale=np.diag([1,1,0]), target=[target_x_rel - wb_width/2, target_y_rel - wb_height/2, 0])
                    man_ik.komo.addObjective([1], ry.FS.negDistance, [pen_contact_frame, "whiteboard"], ry.OT.eq, scale=[1], target=[0.001]) # penEnd touches

                    ret_ik = man_ik.solve(verbose=0)
                    feasible_ik = man_ik.feasible
                    path_ik = man_ik.path

                    if not feasible_ik:
                        print(f'  -- IK infeasible for {"start" if i==0 else "end"} point')
                        motion_feasible_current_step = False
                        if i == 1 and self.C.getFrame("tmp", warnIfNotExist=False): self.C.delFrame("tmp")
                        break

                    target_q = path_ik[0]

                    # --- Path Planning ---
                    man_path = manip.ManipulationModelling()
                    man_path.setup_point_to_point_motion(self.C, target_q, accumulated_collisions=False)

                    if i == 1:
                        q_current = self.C.getJointState()
                        self.C.setJointState(target_q)
                        target_pos_world = self.C.getFrame(pen_contact_frame).getPosition()
                        self.C.setJointState(q_current)
                        delta = target_pos_world - self.C.getFrame(pen_contact_frame).getPosition()
                        if np.linalg.norm(delta) > 1e-6:
                            delta /= np.linalg.norm(delta)
                            projection_matrix = np.eye(3) - np.outer(delta, delta)
                            man_path.komo.addObjective([], ry.FS.positionDiff, [pen_contact_frame, "tmp"], ry.OT.eq, scale=1e1 * projection_matrix)

                    if i == 0:
                        man_path.komo.addObjective([0, .8], ry.FS.negDistance, [pen_contact_frame, "whiteboard"], ry.OT.ineq, scale=[1], target=[-.05])

                    ret_path = man_path.solve(verbose=0)
                    feasible_path = man_path.feasible
                    path_motion = man_path.path

                    if not feasible_path:
                        print(f'  -- Path planning infeasible for {"start" if i==0 else "end"} point')
                        motion_feasible_current_step = False
                        if i == 1 and self.C.getFrame("tmp", warnIfNotExist=False): self.C.delFrame("tmp")
                        break

                    # --- Animate & Draw Spheres ---
                    # Place first sphere at the start of the drawing motion (i=1)
                    if i == 1 and self.draw_method == 1 and self.sphere_distance > 1e-6:
                        self.C.setJointState(path_motion[0]) # Ensure config is at the start
                        current_pen_pos_start = self.C.getFrame(pen_contact_frame).getPosition() # Get pen position at start

                        # --- Calculate Sphere Position near Whiteboard Surface for the first sphere ---
                        # whiteboard_frame, whiteboard_pos, R_wb, z_offset are available from outside
                        local_pen_pos_start = R_wb.T @ (current_pen_pos_start - whiteboard_pos)
                        local_sphere_pos_start = np.array([local_pen_pos_start[0], local_pen_pos_start[1], z_offset])
                        world_sphere_pos_start = whiteboard_pos + R_wb @ local_sphere_pos_start
                        # --- End Sphere Position Calculation ---

                        self.C.addFrame(f"sphere_{self.spheres_drawn}").setShape(ry.ST.sphere, [0.005]).setPosition(world_sphere_pos_start).setColor([1, 0, 0])
                        
                        self.spheres_drawn += 1
                        last_sphere_pos = world_sphere_pos_start # Initialize last_sphere_pos with the placed sphere's position


                    # Animate and potentially draw spheres along the path
                    for t in range(path_motion.shape[0]):
                        self.C.setJointState(path_motion[t])
                        current_pen_pos = self.C.getFrame(pen_contact_frame).getPosition() # Get current pen tip center

                        # --- Start Replacement Block ---
                        # Draw spheres during the drawing motion (i=1) based on distance
                        if i == 1 and self.draw_method == 1 and last_sphere_pos is not None:
                            # Calculate distance based on actual pen tip movement since the last step
                            dist_moved_step = np.linalg.norm(current_pen_pos - last_sphere_pos)

                            # Check if enough distance has accumulated *since the last sphere was placed*
                            # This requires tracking accumulated distance separately.
                            # Let's stick to the simpler approach for now: place when step moves far enough.
                            # A more accurate method would track total distance and place spheres at intervals.

                            # Simplified check: Place if distance since last sphere placement threshold is met
                            # Note: This is still approximate equidistance.
                            dist_since_last_sphere = np.linalg.norm(current_pen_pos - last_sphere_pos) # Re-calculate or use a dedicated variable

                            if dist_since_last_sphere >= self.sphere_distance:
                                # --- Calculate Sphere Position near Whiteboard Surface ---
                                whiteboard_frame = self.C.getFrame("whiteboard")
                                whiteboard_pos = whiteboard_frame.getPosition()
                                # R_wb and z_offset are calculated outside the loop

                                local_pen_pos = R_wb.T @ (current_pen_pos - whiteboard_pos)
                                local_sphere_pos = np.array([local_pen_pos[0], local_pen_pos[1], z_offset])
                                world_sphere_pos = whiteboard_pos + R_wb @ local_sphere_pos
                                # --- End Sphere Position Calculation ---

                                self.C.addFrame(f"sphere_{self.spheres_drawn}").setShape(ry.ST.sphere, [0.005]).setPosition(world_sphere_pos).setColor([1, 0, 0])
                                self.spheres_drawn += 1
                                # Update last_sphere_pos *only when a sphere is placed* to mark the position of the last sphere
                                last_sphere_pos = world_sphere_pos # Update with the placed sphere's position


                        if vis:
                            self.C.view(False, f"Moving segment {i+1} - step {t}")

                    # Set final state for this segment
                    self.C.setJointState(path_motion[-1])

                    if i == 1 and self.C.getFrame("tmp", warnIfNotExist=False):
                        self.C.delFrame("tmp")
            else:
                motion_feasible_current_step = False
            # --- End Motion Loop ---
            overall_feasible = motion_feasible_current_step

            # --- Add Cylinder Marker (if method=2 and feasible) ---
            if overall_feasible and self.draw_method == 2:
                line_vector = world_p1 - world_p0
                line_length = np.linalg.norm(line_vector)
                if line_length >= 1e-6:
                    center_pos = (world_p0 + world_p1) / 2
                    z_axis = np.array([0., 0., 1.])
                    vector_norm = line_vector / line_length
                    axis = np.cross(z_axis, vector_norm)
                    dot_product = np.clip(np.dot(z_axis, vector_norm), -1.0, 1.0)
                    angle = np.arccos(dot_product)
                    if np.linalg.norm(axis) > 1e-6:
                        axis = axis / np.linalg.norm(axis)
                        quat = rowan.from_axis_angle(axis, angle)
                    elif dot_product < -0.9999: quat = np.array([0., 1., 0., 0.])
                    else: quat = np.array([1., 0., 0., 0.])
                    self.C.addFrame(f"line_{self.lines}") \
                        .setShape(ry.ST.cylinder, [line_length, 0.002]) \
                        .setPosition(center_pos) \
                        .setQuaternion(quat) \
                        .setColor([1, 0, 0])
                else: print("Warning: Skipping near-zero length line for cylinder.")


            # --- Calculate Camera Projection ---
            if overall_feasible:
                self.lines += 1
                cam_frame = self.C.getFrame("cameraTop")
                cam_pos = cam_frame.getPosition()
                cam_quat = cam_frame.getQuaternion()
                R_cam = rowan.to_matrix(cam_quat)
                R_cam_inv = R_cam.T
                t_cam = -R_cam_inv @ cam_pos
                p_cam0 = R_cam_inv @ world_p0 + t_cam
                if p_cam0[2] > 1e-6:
                    u0 = self.fx * p_cam0[0] / p_cam0[2] + self.cx
                    v0 = self.fy * p_cam0[1] / p_cam0[2] + self.cy
                    start_point_proj = np.array([u0, v0])
                else: start_point_proj = np.array([np.nan, np.nan])
                p_cam1 = R_cam_inv @ world_p1 + t_cam
                if p_cam1[2] > 1e-6:
                    u1 = self.fx * p_cam1[0] / p_cam1[2] + self.cx
                    v1 = self.fy * p_cam1[1] / p_cam1[2] + self.cy
                    end_point_proj = np.array([u1, v1])
                    if not np.isnan(start_point_proj).any(): proj_vector = end_point_proj - start_point_proj
                    else: proj_vector = np.array([np.nan, np.nan])
                else: proj_vector = np.array([np.nan, np.nan])
            else:
                start_point_proj = np.array([np.nan, np.nan])
                proj_vector = np.array([np.nan, np.nan])

        else:
            raise NotImplementedError(f"Action '{action.name}' not implemented.")

        # --- Finalize Step ---
        self.feasible = overall_feasible

        if not self.feasible:
            info["constraint_violations"].append(f"Motion planning failed for action {action.name}")

        if self.C.getFrame("tmp", warnIfNotExist=False):
            self.C.delFrame("tmp")

        
        self.starting_points.append(start_point_proj)
        self.vectors.append(proj_vector)
        self.t += 1
        self.state = self.getState()
        return self.state, False, 0, info

    
    
    @staticmethod
    def sample_twin(real_env: DrawEnv, obs, task: Task, **kwargs) -> DrawEnv:
        twin = DrawEnv(task)
        twin.C = ry.Config()
        twin.C.addConfigurationCopy(real_env.C)
        twin.state.frames = copy.deepcopy(obs.frames)
        twin.state.config = ry.Config()
        twin.state.config.addConfigurationCopy(obs.config)
        twin.state.config.view()
        twin.initial_state.frames = copy.deepcopy(obs.frames)
        twin.initial_state.config = ry.Config()
        twin.initial_state.config.addConfigurationCopy(obs.config)
        twin.initial_state.config.view()
        return twin

    def reset(self):
        q = self.C.getJointState()
        C_state = self.C.getFrameState()
        self.C.setJointState(q)

        for frame in self.C.getFrameNames():
            if "sphere" in frame or "line" in frame:
                self.C.delFrame(frame)
        
        if self.C.getFrame("tmp", warnIfNotExist=False):
            self.C.delFrame("tmp")

        self.starting_points = []
        self.vectors = []
        self.C.setFrameState(C_state)
        self.C.view()
        self.state = self.getState()
        self.t = 0
        self.feasible = True

        return self.state
    
    def getState(self):

        state = DrawState(self.C)
        state.frames = []


        return state

    def render(self):
        to_stay = ["world", "table", "whiteboard", "cameraTop"]

        C_copy = ry.Config()
        C_copy.addConfigurationCopy(self.C)
        for frame in C_copy.getFrameNames():
            if frame not in to_stay and "line_" not in frame:
                C_copy.delFrame(frame)

        CameraView = ry.CameraView(C_copy)
        CameraView.setCamera(C_copy.getFrame("cameraTop"))
        image, _ = CameraView.computeImageAndDepth(C_copy)
        plt.imshow(image)
        image = isolate_red_shapes_from_rgb(image, background_color=(255, 255, 255))
        
        plt.imshow(image)
        plt.axis("off")
        plt.savefig("result.png")
        print(self.task.get_cost(self))

        self.image_without_background = isolate_red_shapes_from_rgb(image, background_color=(255, 255, 255))

    def compute_cost(self):
        self.C.view()
        cost = self.task.get_cost(self)

        return cost
    