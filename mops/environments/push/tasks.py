from __future__ import annotations

import rowan
import numpy as np
import robotic as ry
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from mops.environments.utils import Task
from mops.environments.push.env import PushEnv
from mops.environments.push.simulator import Simulator


class BuildPlanarTriangle(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = ["block_red", "block_green", "block_blue", "l_gripper", "table"]

    def get_goal(self):
        return self.goal_str

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):

        red_block_error = np.abs(env.C.eval(ry.FS.positionRel, ["block_red", "block_green"])[0][0]-env.C.eval(ry.FS.positionRel, ["block_red", "block_blue"])[0][0])
        green_block_error = np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][0]-env.C.eval(ry.FS.positionRel, ["block_green", "block_blue"])[0][0])
        blue_block_error = np.abs(env.C.eval(ry.FS.positionRel, ["block_blue", "block_red"])[0][0]-env.C.eval(ry.FS.positionRel, ["block_blue", "block_green"])[0][0])

        red_block_error += np.abs(env.C.eval(ry.FS.positionRel, ["block_red", "block_green"])[0][1]+env.C.eval(ry.FS.positionRel, ["block_red", "block_blue"])[0][1])
        green_block_error += np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][1]+env.C.eval(ry.FS.positionRel, ["block_green", "block_blue"])[0][1])
        blue_block_error += np.abs(env.C.eval(ry.FS.positionRel, ["block_blue", "block_red"])[0][1]+env.C.eval(ry.FS.positionRel, ["block_blue", "block_green"])[0][1])

        # Distance of one cm between triangle sides
        red_block_error += 30*(env.C.eval(ry.FS.negDistance, ["block_red", "block_green"])[0]+.01)**2
        green_block_error += 30*(env.C.eval(ry.FS.negDistance, ["block_green", "block_blue"])[0]+.01)**2
        blue_block_error += 30*(env.C.eval(ry.FS.negDistance, ["block_blue", "block_green"])[0]+.01)**2

        total_cost = red_block_error + green_block_error + blue_block_error
    
        return total_cost[0]
    

class TestTask(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):

        red_block_error = 30*(env.C.eval(ry.FS.negDistance, ["block_red", "block_green"])[0]+.04)**2
        green_block_error = 30*(env.C.eval(ry.FS.negDistance, ["block_green", "block_blue"])[0]+.04)**2

        blue_block_error = 10*(env.C.eval(ry.FS.positionDiff, ["block_blue", "block_green"])[0][1])**2
        green_block_error += 10*(env.C.eval(ry.FS.positionDiff, ["block_green", "block_red"])[0][1])**2

        total_cost = red_block_error + green_block_error + blue_block_error
        return total_cost[0]


class BuildPlanarI(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str

    def get_goal(self):
        return self.goal_str

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):
        red_block_error = np.abs(env.C.eval(ry.FS.scalarProductXX, ["block_red", "block_green"])[0][0])
        green_block_error = np.abs(env.C.eval(ry.FS.scalarProductXX, ["block_green", "block_blue"])[0][0])
        blue_block_error = np.abs(env.C.eval(ry.FS.scalarProductXY, ["block_blue", "block_red"])[0][0])
        
        # alignment things
        green_block_error += 10 * (np.abs(np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][0])-.08)+np.abs(env.C.eval(ry.FS.positionRel, ["block_green", "block_red"])[0][1]))

        total_cost = red_block_error + green_block_error + blue_block_error

        if total_cost<.01:
            env.C.view(True)


        return total_cost


class BuildBridge(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = ["block_red", "block_green", "block_blue", "l_gripper", "table"]

    def get_goal(self):
        return self.goal_str

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):

        red_block = env.C.getFrame("block_red")
        green_block = env.C.getFrame("block_green")
        blue_block = env.C.getFrame("block_blue")

        red_block_error = 0
        green_block_error = 0
        blue_block_error = 0

        # Positions
        green_block_error += np.abs(np.linalg.norm(green_block.getPosition() - red_block.getPosition()) - 0.12)
        blue_block_error += np.abs((blue_block.getPosition()[2] - red_block.getPosition()[2]) - .06 - .02)

        # Rotations
        blue_block_error += np.abs(env.C.eval(ry.FS.scalarProductZZ, ["block_blue", "table"])[0][0])

        total_cost = red_block_error + green_block_error + blue_block_error
        
        return total_cost
    

class BuildMultiBridge(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = ["l_gripper", "table"]
        for i in range(3):
            self.relevant_frames.extend([f"block_red_{i}", f"block_green_{i}", f"block_blue_{i}"])

    def get_goal(self):
        return self.goal_str

    def setup_cfg(self, vertical_blocks: bool = True, multi: bool = True, **kwargs) -> ry.Config:
        return Task.setup_cfg(self, multi=True, vertical_blocks=vertical_blocks, **kwargs)

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):

        total_cost = 0
        for i in range(3):
            red_block = env.C.getFrame(f"block_red_{i}")
            green_block = env.C.getFrame(f"block_green_{i}")
            blue_block = env.C.getFrame(f"block_blue_{i}")

            red_block_error = 0
            green_block_error = 0
            blue_block_error = 0

            # Positions
            green_block_error += np.abs(np.linalg.norm(green_block.getPosition() - red_block.getPosition()) - 0.12)
            blue_block_error += np.abs((blue_block.getPosition()[2] - red_block.getPosition()[2]) - .06 - .02)

            # Rotations
            blue_block_error += np.abs(env.C.eval(ry.FS.scalarProductZZ, [f"block_blue_{i}", "table"])[0][0])

            total_cost += red_block_error + green_block_error + blue_block_error
        
        return total_cost
    

class PlaceRed(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = ["block_red", "block_green", "block_blue"]

    def get_goal(self):
        return self.goal_str

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):
        red_block = env.C.getFrame("block_red")
        total_cost = np.linalg.norm(red_block.getPosition()[:2] - np.array([.3, .3]))**2
        return total_cost
    

def big_red_block_config() -> ry.Config:
    C = ry.Config()
    C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

    C.delFrame("panda_collCameraWrist")
    C.getFrame("table").setShape(ry.ST.ssBox, size=[2., 2., .1, .02])

    C.addFrame("big_red_block") \
        .setPosition([-.2, .3, .7]) \
        .setQuaternion(rowan.from_euler(0., 0., -np.pi * 1.5, convention="xyz")) \
        .setShape(ry.ST.ssBox, size=[.1, .2, .1, 0.005]) \
        .setColor([.8, .2, .25]) \
        .setContact(1) \
        .setMass(.1)

    C.addFrame("target_pose") \
        .setPosition([.4, .3, .7]) \
        .setQuaternion(rowan.from_euler(0., 0., np.pi * 1.2, convention="xyz")) \
        .setShape(ry.ST.ssBox, size=[.1, .2, .1, 0.005]) \
        .setColor([0., 1., 0., .1])
    return C


class PushRed(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = ["big_red_block", "target_pose"]

    def get_goal(self):
        return self.goal_str

    def setup_cfg(self):
        C = big_red_block_config()
        return C

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):

        box_frame = env.C.getFrame("big_red_block")
        target_frame = env.C.getFrame("target_pose")

        pos_diff, _ = env.C.eval(ry.FS.positionDiff, ["big_red_block", "target_pose"])
        pos_cost = (np.linalg.norm(pos_diff)*10)**2
        
        q_diff = rowan.multiply(rowan.conjugate(box_frame.getQuaternion()), target_frame.getQuaternion())
        angle = 2 * np.arccos(np.clip(q_diff[0], -1.0, 1.0))
        rot_cost = angle**2

        total_cost = pos_cost*.5 + rot_cost*.05
        
        return total_cost
    

class PushRedWithWall(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = ["big_red_block", "target_pose", "wall"]

    def get_goal(self):
        return self.goal_str

    def setup_cfg(self):
        C = big_red_block_config()
        C.addFrame("wall") \
            .setPosition([.0, .4, .8]) \
            .setShape(ry.ST.ssBox, [.1, .4, .3, .001]) \
            .setColor([.7, .7, .7]) \
            .setContact(1)
        C.getFrame("big_red_block") \
            .setPosition([-.3, .3, .7])
        C.getFrame("target_pose") \
            .setPosition([.4, .1, .7])
        self.box_initial_pos = C.getFrame("big_red_block").getPosition()
        return C

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):

        box_frame = env.C.getFrame("big_red_block")
        target_frame = env.C.getFrame("target_pose")
        gripper_frame = env.C.getFrame("l_gripper")
        wall_frame = env.C.getFrame("wall")

        box_pos = box_frame.getPosition()

        # Minimise box distance to target
        pos_cost = (np.linalg.norm(box_pos - target_frame.getPosition()) * 4)**2
        
        # Minimise box angle diff to target
        q_diff = rowan.multiply(rowan.conjugate(box_frame.getQuaternion()), target_frame.getQuaternion())
        angle = 2 * np.arccos(np.clip(q_diff[0], -1.0, 1.0))
        rot_cost = angle

        # Maximise box distance to wall (as a circle for less local minima(?))
        wall_dist_cost = -np.log(np.linalg.norm(wall_frame.getPosition()[:2]-box_pos[:2]))

        # Maximise box distance to its starting position
        initial_pos_dist_cost = -np.log(np.max([np.linalg.norm(box_pos-self.box_initial_pos), .001]))

        # Minimise the end distance between the gripper and the box
        endeff_dist_cost = np.linalg.norm(box_pos-gripper_frame.getPosition() * 4)**2

        # Maximise the distance between the gripper and the wall
        endeff_wall_dist_cost = -np.linalg.norm(wall_frame.getPosition()-gripper_frame.getPosition())

        total_cost = pos_cost * 2. + \
                     rot_cost * 0. + \
                     wall_dist_cost * .01 + \
                     initial_pos_dist_cost * .01 + \
                     endeff_dist_cost * .7 + \
                     endeff_wall_dist_cost * .2
        
        return total_cost
    

class PushLine(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = [f"block{i}" for i in range(6)]
        self.cost_threshold = 1.  # How low the cost has to be for the task to be considered fulfilled.

    def get_goal(self):
        return self.goal_str

    def setup_cfg(self):
        
        # np.random.seed(30)
        # are_colls = True
        # while are_colls:
        #     C = ry.Config()
        #     C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
        #     for f in self.relevant_frames:
        #         pos_xy = np.random.uniform(-.3, .3, (2,)) + np.array([.0, .25])
        #         color = np.random.uniform(0., 1., (3,))
        #         mass = np.random.uniform(1., 10.)
        #         C.addFrame(f) \
        #             .setPosition([*pos_xy, .7]) \
        #             .setShape(ry.ST.ssBox, [.08, .08, .08, .001]) \
        #             .setColor(color) \
        #             .setContact(1) \
        #             .setMass(mass)
                
        #     collisions = C.getCollisions()
        #     are_colls = False
        #     for c in collisions:
        #         if "block" in c[0] and "block" in c[1]:
        #             del C
        #             are_colls = True
        #             break

        np.random.seed(100)
        are_colls = True
        while are_colls:
            C = ry.Config()
            C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
            
            C.addFrame("block0") \
                .setShape(ry.ST.ssBox, [.075, .069, .059, .001]) \
                .setColor([.9, .9, .0]) \
                .setContact(1) \
                .setMass(.22)
            
            C.addFrame("block1") \
                .setShape(ry.ST.ssBox, [.072, .075, .059, .001]) \
                .setColor([.95, .9, .0]) \
                .setContact(1) \
                .setMass(.22)
            
            C.addFrame("block2") \
                .setShape(ry.ST.ssBox, [.06, .06, .06, .001]) \
                .setColor([1., .0, .0]) \
                .setContact(1) \
                .setMass(.02)
            
            C.addFrame("block3") \
                .setShape(ry.ST.ssBox, [.05, .06, .05, .001]) \
                .setColor([.0, .7, .2]) \
                .setContact(1) \
                .setMass(.01)
            
            C.addFrame("block4") \
                .setShape(ry.ST.ssBox, [.09, .06, .059, .001]) \
                .setColor([1., .8, .0]) \
                .setContact(1) \
                .setMass(.02)
            
            C.addFrame("block5") \
                .setShape(ry.ST.ssBox, [.06, .06, .06, .001]) \
                .setColor([1., .0, .0]) \
                .setContact(1) \
                .setMass(.04)

            for f in self.relevant_frames:
                pos_xy = np.random.uniform(-.3, .3, (2,)) + np.array([.0, .25])
                C.getFrame(f).setPosition([*pos_xy, .7]) \
                
            collisions = C.getCollisions()
            are_colls = False
            for c in collisions:
                if "block" in c[0] and "block" in c[1]:
                    del C
                    are_colls = True
                    break

        return C

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):
    
        poss = []
        for f in self.relevant_frames:
            pos = env.C.getFrame(f).getPosition()[:2]
            poss.append(pos)
        poss = np.array(poss)

        # Fit line y = mx + b
        X = poss[:, 0].reshape(-1, 1)
        y = poss[:, 1]
        model = LinearRegression()
        model.fit(X, y)

        y_pred = model.predict(X)
        lr_mse = mean_squared_error(y, y_pred)

        # Project each point onto the regression line
        slope = model.coef_[0]
        intercept = model.intercept_

        # Unit direction vector of the line
        direction = np.array([1, slope])
        direction = direction / np.linalg.norm(direction)

        # Project each point onto the line
        projected = []
        for point in poss:
            # Vector from origin to point
            vec = point - np.array([0, intercept])
            # Scalar projection onto direction vector
            t = np.dot(vec, direction)
            proj_point = np.array([0, intercept]) + t * direction
            projected.append(proj_point)
        projected = np.array(projected)

        # Sort projected points along the line (by t)
        t_values = np.dot(projected - np.array([0, intercept]), direction)
        sorted_indices = np.argsort(t_values)
        sorted_projected = projected[sorted_indices]

        # Compute distances between successive projected points
        distances = np.linalg.norm(np.diff(sorted_projected, axis=0), axis=1)

        # Compute MSE of distances
        mean_distance = np.mean(distances)
        mse = np.mean((distances - mean_distance) ** 2)
        cost = lr_mse*1e4 + mse*1e2

        return cost
    

class PushCircle(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = [f"block{i}" for i in range(6)]
        self.cost_threshold = 1.  # How low the cost has to be for the task to be considered fulfilled.

    def get_goal(self):
        return self.goal_str

    def setup_cfg(self):
        
        # np.random.seed(30)
        # are_colls = True
        # while are_colls:
        #     C = ry.Config()
        #     C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
        #     for f in self.relevant_frames:
        #         pos_xy = np.random.uniform(-.3, .3, (2,)) + np.array([.0, .25])
        #         color = np.random.uniform(0., 1., (3,))
        #         mass = np.random.uniform(1., 10.)
        #         C.addFrame(f) \
        #             .setPosition([*pos_xy, .7]) \
        #             .setShape(ry.ST.ssBox, [.08, .08, .08, .001]) \
        #             .setColor(color) \
        #             .setContact(1) \
        #             .setMass(mass)
                
        #     collisions = C.getCollisions()
        #     are_colls = False
        #     for c in collisions:
        #         if "block" in c[0] and "block" in c[1]:
        #             del C
        #             are_colls = True
        #             break

        # return C
    
        np.random.seed(100)
        are_colls = True
        while are_colls:
            C = ry.Config()
            C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))
            
            C.addFrame("block0") \
                .setShape(ry.ST.ssBox, [.075, .069, .059, .001]) \
                .setColor([.9, .9, .0]) \
                .setContact(1) \
                .setMass(.22)
            
            C.addFrame("block1") \
                .setShape(ry.ST.ssBox, [.072, .075, .059, .001]) \
                .setColor([.95, .9, .0]) \
                .setContact(1) \
                .setMass(.22)
            
            C.addFrame("block2") \
                .setShape(ry.ST.ssBox, [.06, .06, .06, .001]) \
                .setColor([1., .0, .0]) \
                .setContact(1) \
                .setMass(.02)
            
            C.addFrame("block3") \
                .setShape(ry.ST.ssBox, [.05, .06, .05, .001]) \
                .setColor([.0, .7, .2]) \
                .setContact(1) \
                .setMass(.01)
            
            C.addFrame("block4") \
                .setShape(ry.ST.ssBox, [.09, .06, .059, .001]) \
                .setColor([1., .8, .0]) \
                .setContact(1) \
                .setMass(.02)
            
            C.addFrame("block5") \
                .setShape(ry.ST.ssBox, [.06, .06, .06, .001]) \
                .setColor([1., .0, .0]) \
                .setContact(1) \
                .setMass(.04)

            for f in self.relevant_frames:
                pos_xy = np.random.uniform(-.3, .3, (2,)) + np.array([.0, .25])
                C.getFrame(f).setPosition([*pos_xy, .7]) \
                
            collisions = C.getCollisions()
            are_colls = False
            for c in collisions:
                if "block" in c[0] and "block" in c[1]:
                    del C
                    are_colls = True
                    break

        return C

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):
    
        # Circle Radius Cost
        poss = []
        center = np.array([0., 0.])
        for f in self.relevant_frames:
            b = env.C.getFrame(f)
            pos = b.getPosition()[:2]
            center += pos
            poss.append(pos)
        center /= len(poss)

        rad_cost = 0
        radius = .2
        for pos in poss:
            rad_cost += (radius - np.linalg.norm(pos-center))**2

        # Nearest Neighbour Cost (Want to maximise)
        all_dists = []
        for i, p0 in enumerate(poss):
            dists = []
            for j, p1 in enumerate(poss):
                if i != j:
                    dist = np.linalg.norm(p1-p0)
                    dists.append(dist)
            all_dists.append(min(dists))
        
        neig_cost = 0
        for dist in all_dists:
            neig_cost += (.2 - dist)**2
        
        print(f"Rad Component: {rad_cost}")
        print(f"Neig Component: {neig_cost}")
        cost = rad_cost *1e3 + neig_cost *1e0
        return cost
    

class PourBeer(Task):
    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        self.relevant_frames = ["beer_bottle", "empty_glass"]
        self.density = 400

    def get_goal(self):
        return self.goal_str
    
    def setup_cfg(self):
        
        C = ry.Config()
        C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaSingle.g'))

        beer_bottle_pos = np.array([-.3, .3, .7])

        C.addFrame("empty_glass_base") \
            .setShape(ry.ST.ssBox, [.08, .08, .01, .001]) \
            .setPosition([.0, .4, .7]) \
            .setColor([.8, .8, .8]) \
            .setMass(.3)
        C.addFrame("empty_glass_side0", "empty_glass_base") \
            .setShape(ry.ST.ssBox, [.08, .005, .07, .001]) \
            .setRelativePosition([.0, .04, .04]) \
            .setColor([.8, .8, .8])
        C.addFrame("empty_glass_side1", "empty_glass_base") \
            .setShape(ry.ST.ssBox, [.08, .005, .07, .001]) \
            .setRelativePosition([.0, -.04, .04]) \
            .setColor([.8, .8, .8])
        C.addFrame("empty_glass_side2", "empty_glass_base") \
            .setShape(ry.ST.ssBox, [.005, .08, .07, .001]) \
            .setRelativePosition([.04, .0, .04]) \
            .setColor([.8, .8, .8])
        C.addFrame("empty_glass_side3", "empty_glass_base") \
            .setShape(ry.ST.ssBox, [.005, .08, .07, .001]) \
            .setRelativePosition([-.04, .0, .04]) \
            .setColor([.8, .8, .8])
        C.addFrame("empty_glass", "empty_glass_base") \
            .setShape(ry.ST.ssBox, [.08, .08, .07, .001]) \
            .setRelativePosition([.0, .0, .04]) \
            .setColor([1., 1., .0, .3]) \
            .setContact(1)

        C.addFrame("beer_bottle_base") \
            .setShape(ry.ST.ssBox, [.07, .07, .01, .001]) \
            .setPosition(beer_bottle_pos) \
            .setColor([.5, .0, .0]) \
            .setMass(.5)
        C.addFrame("beer_bottle_side0", "beer_bottle_base") \
            .setShape(ry.ST.ssBox, [.07, .005, .1, .001]) \
            .setRelativePosition([.0, .035, .05]) \
            .setColor([.5, .0, .0])
        C.addFrame("beer_bottle_side1", "beer_bottle_base") \
            .setShape(ry.ST.ssBox, [.07, .005, .1, .001]) \
            .setRelativePosition([.0, -.035, .05]) \
            .setColor([.5, .0, .0])
        C.addFrame("beer_bottle_side2", "beer_bottle_base") \
            .setShape(ry.ST.ssBox, [.005, .07, .1, .001]) \
            .setRelativePosition([.035, .0, .05]) \
            .setColor([.5, .0, .0])
        C.addFrame("beer_bottle_side3", "beer_bottle_base") \
            .setShape(ry.ST.ssBox, [.005, .07, .1, .001]) \
            .setRelativePosition([-.035, .0, .05]) \
            .setColor([.5, .0, .0])
        C.addFrame("beer_bottle", "beer_bottle_base") \
            .setShape(ry.ST.ssBox, [.07, .07, .1, .001]) \
            .setRelativePosition([.0, .0, .055]) \
            .setColor([1., 1., .0, .8]) \
            .setContact(1)
        
        beer_count = int(self.density * .8)
        for i in range(beer_count):
            pos = np.random.uniform(-.02, .02, (3,))
            pos[2] = np.random.uniform(.0, .5) + .1
            pos += beer_bottle_pos
            C.addFrame(f"beer_particle{i}") \
                .setShape(ry.ST.sphere, [.005]) \
                .setColor([1., 1., 0.]) \
                .setPosition(pos) \
                .setMass(.01) \
                .setContact(1)
        
        foam_count = int(self.density * .2)
        for i in range(foam_count):
            pos = np.random.uniform(-.02, .02, (3,))
            pos[2] = np.random.uniform(.0, .5) + .7
            pos += beer_bottle_pos
            C.addFrame(f"foam_particle{i}") \
                .setShape(ry.ST.sphere, [.005]) \
                .setColor([1., 1., 1.]) \
                .setPosition(pos) \
                .setMass(.01) \
                .setContact(1)
        
        sim = Simulator(C)
        x, q = sim.step(300)
        C.setFrameState(x)
        C.setJointState(q)

    def get_reward(self, env: PushEnv):
        return 0
    
    def get_cost(self, env: PushEnv):

        fp = "foam_particle"
        bp = "beer_particle"
        egi = "empty_glass"

        in_bottle = 0
        cps = env.C.getCollisions()
        for cp in cps:
            if ((fp in cp[0] or fp in cp[1] or
                 bp in cp[0] or bp in cp[1]) and
                (egi in cp[0] or egi in cp[1])):
                in_bottle += 1

        cost = self.density - in_bottle
        return cost
