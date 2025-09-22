from __future__ import annotations

import numpy as np
import robotic as ry
from mops.environments.utils import  Task
import cv2

def isolate_red_shapes_from_rgb(image_rgb: np.ndarray, background_color=(255, 255, 255)) -> np.ndarray:
    """
    Keeps only red shapes in an RGB image and replaces everything else with a solid background color.

    Args:
        image_rgb (np.ndarray): Input image in RGB format.
        background_color (tuple): RGB tuple for background (default: white).

    Returns:
        np.ndarray: Image with only red parts retained, rest filled with background color.
    """
    if image_rgb is None or not isinstance(image_rgb, np.ndarray):
        raise ValueError("Input must be a valid NumPy image array in RGB format.")

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    result_rgb = np.full_like(image_rgb, background_color, dtype=np.uint8)
    result_rgb[red_mask > 0] = image_rgb[red_mask > 0]

    return result_rgb


def create_config(ON_REAL=False) -> ry.Config:
    

    WHITEBOARD_OFFSET = 0.125  # base offset of whiteboard in cms to table
   
    if ON_REAL:
        WHITEBOARD_OFFSET+= .02+ 0.01

    WHITEBOARD_TILT = 40  # tilt of the whiteboard in degrees
    WHITEBOARD_OFFSET_X = 0
    WHITEBOARD_OFFSET_Y = 0.4


    WHITEBOARD_SIZE_X = .64
    WHITEBOARD_SIZE_Y = .48
    PEN_SIZE = .1

    C = ry.Config()
    C.addFile(ry.raiPath('scenarios/pandaMops.g'))
    if not ON_REAL:
        C.getFrame('l_panda_finger_joint1').setJointState([.01])
        C.getFrame("table").setColor([1, 1, 1]).setShape(ry.ST.box, [2.5, 2.5, .01])

    C.delFrame("panda_collCameraWrist")
    C.delFrame("cameraWrist")

    WHITEBOARD_TILT = np.deg2rad(WHITEBOARD_TILT)  # Convert to radians
    quaternion_wb = [np.cos(WHITEBOARD_TILT / 2), np.sin(WHITEBOARD_TILT / 2), 0, 0]
    table_height = C.getFrame("table").getSize()[2] / 2
    WHITEBOARD_THICKNESS = 0.005 
    whiteboard_height_adjustment = (.7) * np.sin(WHITEBOARD_TILT) / 2

    C.addFrame("whiteboard", "table")\
        .setColor([1, 1, 1])\
        .setShape(ry.ST.box, [WHITEBOARD_SIZE_X, WHITEBOARD_SIZE_Y, WHITEBOARD_THICKNESS])\
        .setRelativePosition([WHITEBOARD_OFFSET_X, WHITEBOARD_OFFSET_Y, table_height + WHITEBOARD_THICKNESS / 2 + whiteboard_height_adjustment + WHITEBOARD_OFFSET])\
        .setQuaternion(quaternion_wb)

    # Pen with whom the robot will draw
    C.addFrame("pen", "l_gripper").setColor([1, 0, 0]).setShape(ry.ST.cylinder, [.1, .01]).setRelativePosition([0, 0, -.05])
    C.addFrame("penEnd", "pen").setRelativePosition([0, 0, -PEN_SIZE / 2])

    C.getFrame("cameraTop").setQuaternion([0, 1, 0, 0]).setPosition([0, .45, 1.5])

    return {
        "config": C,
        "whiteboard_thickness": WHITEBOARD_THICKNESS,
        "whiteboard_tilt": WHITEBOARD_TILT,
        "whiteboard_size_x": WHITEBOARD_SIZE_X,
        "whiteboard_size_y": WHITEBOARD_SIZE_Y,
        "whiteboard_offset_x": WHITEBOARD_OFFSET_X,
        "whiteboard_offset_y": WHITEBOARD_OFFSET_Y
    }

 
def compute_vector_cost(vec_list_1, vec_list_2):
    """
    Computes the pairwise summed MSE between two lists of 2D points,
    invariant to translation (centers both lists before comparison).
    Each list must have the same number of 2D points.
    """
    if len(vec_list_1) != len(vec_list_2):
        return np.nan
    assert all(len(vec) == 2 for vec in vec_list_1 + vec_list_2), "All vectors must be 2D."

    X = np.array(vec_list_1)
    Y = np.array(vec_list_2)

    mse = np.mean(np.sum((X - Y) ** 2, axis=1))
    return mse


def compute_vector_cost_star(starting_point_list, vec_list):
    """
    Computes a cost based on how closely the drawn vectors form a 5-pointed star.
    Assumes 10 vectors drawn sequentially to form the star outline.
    """
    if len(vec_list) != 10 or len(starting_point_list) != 10:
        print("Star cost: Incorrect number of vectors/points.")
        return 123456789

    if np.any(np.isnan(vec_list)) or np.any(np.isnan(starting_point_list)):
        return 1e10

    # --- 1. Calculate Vertices and Center ---
    vertices = np.array(starting_point_list)
    end_points = np.array([s + v for s, v in zip(starting_point_list, vec_list)])
    center = np.mean(vertices, axis=0) # Approximate center

    # --- 2. Connectivity Cost ---
    cost_connectivity = 0
    # Check connection between end of vector i and start of vector i+1
    for i in range(9):
        connection_gap = np.linalg.norm(end_points[i] - vertices[i+1])
        cost_connectivity += connection_gap**2
    # Check connection between end of last vector and start of first vector
    final_connection_gap = np.linalg.norm(end_points[9] - vertices[0])
    cost_connectivity += final_connection_gap**2
    
    # Add a scaling factor to make it significant
    cost_connectivity *= 100 
    print(f"Star cost_connectivity: {cost_connectivity:.4f}")


    # --- 3. Radius Consistency Cost ---
    # Assume vertices 0, 2, 4, 6, 8 are outer; 1, 3, 5, 7, 9 are inner
    outer_indices = range(0, 10, 2)
    inner_indices = range(1, 10, 2)

    outer_radii = [np.linalg.norm(vertices[i] - center) for i in outer_indices]
    inner_radii = [np.linalg.norm(vertices[i] - center) for i in inner_indices]

    # Cost is variance (spread) of radii. Low variance means points are equidistant.
    # Add small epsilon to avoid division by zero if all radii are identical (mean=0)
    mean_outer_radius = np.mean(outer_radii)
    mean_inner_radius = np.mean(inner_radii)
    
    # Avoid division by zero or large costs for tiny stars
    cost_outer_radius_consistency = np.var(outer_radii) / (mean_outer_radius**2 + 1e-6) if mean_outer_radius > 1e-3 else np.var(outer_radii)
    cost_inner_radius_consistency = np.var(inner_radii) / (mean_inner_radius**2 + 1e-6) if mean_inner_radius > 1e-3 else np.var(inner_radii)

    # Add scaling factors
    cost_outer_radius_consistency *= 500 
    cost_inner_radius_consistency *= 500
    print(f"Star cost_outer_radius_consistency: {cost_outer_radius_consistency:.4f}")
    print(f"Star cost_inner_radius_consistency: {cost_inner_radius_consistency:.4f}")

    # Maybe delete
    # --- 3b. Outer-to-Inner Radius Ratio Cost ---
    # A good star should have outer points significantly farther from center than inner points
    outer_to_inner_ratio = mean_outer_radius / (mean_inner_radius + 1e-6)
    ideal_ratio = 2.0  # Outer radius should be about twice the inner radius
    min_acceptable_ratio = 1.5  # Below this is penalized heavily

    if outer_to_inner_ratio < min_acceptable_ratio:
        # Exponential penalty for having outer and inner points too close together
        ratio_penalty = 300 * (min_acceptable_ratio - outer_to_inner_ratio)**2
        print(f"Star outer-to-inner ratio: {outer_to_inner_ratio:.2f} (too small, ideal: {ideal_ratio:.1f})")
        print(f"Star ratio penalty: {ratio_penalty:.4f}")
        
        # Add to total cost
        cost_outer_radius_consistency += ratio_penalty
    else:
        # Still add a smaller cost for deviating from ideal ratio
        ratio_cost = 100 * (outer_to_inner_ratio - ideal_ratio)**2
        print(f"Star outer-to-inner ratio: {outer_to_inner_ratio:.2f} (acceptable, ideal: {ideal_ratio:.1f})")
        print(f"Star ratio cost: {ratio_cost:.4f}")
        
        # Add to total cost
        cost_outer_radius_consistency += ratio_cost


    # --- 4. Angle Consistency Cost ---
    vectors_from_center = [v - center for v in vertices]
    cost_angle_consistency = 0
    ideal_angle_rad = np.pi / 5.0  # 36 degrees

    for i in range(10):
        v1 = vectors_from_center[i]
        v2 = vectors_from_center[(i + 1) % 10] # Wrap around for the last point

        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)

        if norm_v1 < 1e-6 or norm_v2 < 1e-6: # Avoid division by zero if a point is at the center
            continue 

        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)

        # Penalize deviation from the ideal 36 degrees
        angle_diff = angle_rad - ideal_angle_rad
        cost_angle_consistency += angle_diff**2

    # Add scaling factor
    cost_angle_consistency *= 100 
    print(f"Star cost_angle_consistency: {cost_angle_consistency:.4f}")

    # --- 5. Total Cost ---
    total_cost = (cost_connectivity + 
                  cost_outer_radius_consistency + 
                  cost_inner_radius_consistency + 
                  cost_angle_consistency)

    # --- penalize vanishing lengths ---
    avg_size = (np.mean([np.linalg.norm(v) for v in vec_list]) + 1e-6)
    min_length_threshold = 35.0  # below this = heavy penalty
    if avg_size < min_length_threshold:
        total_cost += 7e2  # Massive penalty for being too small


    print(f"Star total model-based cost: {total_cost:.4f}")
    return total_cost

def compute_vector_cost_pentagon(starting_point_list, vec_list):
    """
    Computes a cost based on how closely the drawn vectors form an equilateral pentagon.
    Assumes 5 vectors drawn sequentially to form the pentagon outline.
    """
    if len(vec_list) != 5 or len(starting_point_list) != 5:
        print("Pentagon cost: Incorrect number of vectors/points.")
        return 123456789


    if np.any(np.isnan(vec_list)) or np.any(np.isnan(starting_point_list)):
        print("Pentagon cost: NaN detected in input vectors or starting points.")
        return 1e10

    # --- 1. Calculate endpoints and check connectivity ---
    start_points = np.array(starting_point_list)
    end_points = np.array([s + v for s, v in zip(starting_point_list, vec_list)])
    
    # --- 2. Connectivity Cost ---
    cost_connectivity = 0
    # Check connection between end of vector i and start of vector i+1
    for i in range(4):
        connection_gap = np.linalg.norm(end_points[i] - start_points[i+1])
        cost_connectivity += connection_gap**2
    # Check connection between end of last vector and start of first vector
    final_connection_gap = np.linalg.norm(end_points[4] - start_points[0])
    cost_connectivity += final_connection_gap**2
    
    # Add a scaling factor to make it significant
    cost_connectivity *= 500
    print(f"Pentagon cost_connectivity: {cost_connectivity:.4f}")

    # --- 3. Side Length Equality Cost ---
    lengths = np.array([np.linalg.norm(v) for v in vec_list])
    mean_length = np.mean(lengths)
    
    # Variance of lengths normalized by mean length squared
    if mean_length > 1e-6:
        cost_side_length = 500 * np.var(lengths) / (mean_length**2)
    else:
        cost_side_length = 1e10  # Penalize very small pentagons
    
    print(f"Pentagon cost_side_length: {cost_side_length:.4f}")

   
    # --- 4. Calculate Center of Shape ---
    # Approximate center - mean of start points
    center = np.mean(start_points, axis=0)
    
    # --- 5. Regular Shape Cost (all vertices should be equidistant from center) ---
    radii = [np.linalg.norm(p - center) for p in start_points]
    mean_radius = np.mean(radii)
    
    if mean_radius > 1e-6:
        cost_radius = 300 * np.var(radii) / (mean_radius**2)
    else:
        cost_radius = 1e10
    
    print(f"Pentagon cost_radius: {cost_radius:.4f}")
    
    # --- 6. Angle Cost (108° for each internal angle in a regular pentagon) ---
    cost_angle = 0
    ideal_angle_rad = 3 * np.pi / 5  # 108 degrees for internal angles
    
    for i in range(5):
        # Get the two vectors forming this vertex
        v1 = -vec_list[i-1]  # Previous vector (reversed direction)
        v2 = vec_list[i]     # Current vector
        
        # Normalize vectors
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            continue  # Skip if vectors are too small
            
        v1_norm = v1 / norm_v1
        v2_norm = v2 / norm_v2
        
        # Calculate angle using dot product
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_rad = np.arccos(dot_product)
        angle_deg = np.degrees(angle_rad)
        
        # Penalize deviation from 108 degrees
        angle_diff = angle_rad - ideal_angle_rad
        cost_angle += 100 * angle_diff**2
        
        print(f"Pentagon angle {i+1}: {angle_deg:.1f}° (ideal: 108°)")
    
    print(f"Pentagon cost_angle: {cost_angle:.4f}")

    # --- 7. Angular Spacing Cost (vertices should be equally spaced around center) ---
    vectors_from_center = [p - center for p in start_points]
    cost_angular_spacing = 0
    ideal_angle_rad = 2 * np.pi / 5  # 72 degrees between consecutive vertices
    
    for i in range(5):
        v1 = vectors_from_center[i]
        v2 = vectors_from_center[(i + 1) % 5]  # Wrap around for the last point
        
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 < 1e-6 or norm_v2 < 1e-6:
            continue  # Skip if points are too close to center
            
        # Calculate angle using dot product
        dot_product = np.dot(v1, v2)
        cos_theta = np.clip(dot_product / (norm_v1 * norm_v2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        
        # Penalize deviation from ideal spacing (72 degrees)
        angle_diff = angle_rad - ideal_angle_rad
        cost_angular_spacing += 100 * angle_diff**2
    
    print(f"Pentagon cost_angular_spacing: {cost_angular_spacing:.4f}")

    # --- 8. Shape Closure Cost ---
    # Check if the pentagon is closed (first point should be close to end of last vector)
    closure_dist = np.linalg.norm(start_points[0] - end_points[4])
    cost_closure = 200 * closure_dist**2
    print(f"Pentagon cost_closure: {cost_closure:.4f}")



    # --- 9. Total Cost ---
    total_cost = (cost_connectivity + 
                  cost_side_length + 
                  cost_radius +
                  cost_angle +
                  cost_angular_spacing +
                  cost_closure)


    # --- penalize vanishing lengths ---
    avg_size = (np.mean([np.linalg.norm(v) for v in vec_list]) + 1e-6)
    min_length_threshold = 30.0  # below this = heavy penalty
    if avg_size < min_length_threshold:
        total_cost += 7e2  # Massive penalty for being too small

    print(f"Pentagon total cost: {total_cost:.4f}")
    return total_cost


def compute_vector_cost_hash(starting_point_list, vec_list):
    """
    Computes a cost based on how closely the drawn vectors form a hash (#).
    Assumes 4 vectors: 2 horizontal and 2 vertical.
    """
    if len(vec_list) != 4 or len(starting_point_list) != 4:
        print("Hash cost: Incorrect number of vectors/points.")
        return 123456789


    if np.any(np.isnan(vec_list)) or np.any(np.isnan(starting_point_list)):
        print("Hash cost: NaN detected in input.")
        return 1e10

    orientations = []
    for v in vec_list:
        if abs(v[0]) > abs(v[1]):
            orientations.append('h')
        else:
            orientations.append('v')

    if orientations.count('h') != 2 or orientations.count('v') != 2:
        print("Hash cost: Need exactly 2 horizontal and 2 vertical lines.")
        return 1e10

    h_indices = [i for i, o in enumerate(orientations) if o == 'h']
    v_indices = [i for i, o in enumerate(orientations) if o == 'v']

    h_vecs = [vec_list[i] for i in h_indices]
    v_vecs = [vec_list[i] for i in v_indices]
    h_starts = [starting_point_list[i] for i in h_indices]
    v_starts = [starting_point_list[i] for i in v_indices]

    cost = 0

    # 1. Straightness cost
    for v in h_vecs:
        cost += 100 * (abs(v[1]) / (np.linalg.norm(v) + 1e-6))**2
    for v in v_vecs:
        cost += 100 * (abs(v[0]) / (np.linalg.norm(v) + 1e-6))**2

    # 2. Parallelism cost
    h_angles = [np.arctan2(v[1], v[0]) for v in h_vecs]
    v_angles = [np.arctan2(v[1], v[0]) for v in v_vecs]
    h_angle_diff = min(abs(h_angles[0] - h_angles[1]), np.pi - abs(h_angles[0] - h_angles[1]))
    v_angle_diff = min(abs(v_angles[0] - v_angles[1]), np.pi - abs(v_angles[0] - v_angles[1]))
    cost += 50 * (h_angle_diff)**2
    cost += 50 * (v_angle_diff)**2

    # 3. Perpendicularity cost
    for hv in h_vecs:
        for vv in v_vecs:
            dot = np.dot(hv, vv) / (np.linalg.norm(hv) * np.linalg.norm(vv) + 1e-6)
            cost += 50 * (dot)**2  # dot should be ~0 for 90 degrees

    # 4. Spacing cost
    h_y = [h_starts[i][1] for i in range(2)]
    v_x = [v_starts[i][0] for i in range(2)]
    avg_size = (np.mean([np.linalg.norm(v) for v in h_vecs + v_vecs]) + 1e-6)

    h_spacing = abs(h_y[0] - h_y[1]) / avg_size
    v_spacing = abs(v_x[0] - v_x[1]) / avg_size
    target_spacing = 0.33  # maybe 1/3 of average size
    cost += 300 * ((h_spacing - target_spacing)**2 + (v_spacing - target_spacing)**2)

    # Also penalize if the spacings aren't symmetric
    cost += 100 * (h_spacing - v_spacing)**2

    # 5. Improved intersection cost - check where the lines actually intersect
    h_ends = [h_starts[i] + h_vecs[i] for i in range(2)]
    v_ends = [v_starts[i] + v_vecs[i] for i in range(2)]
    h_lengths = [np.linalg.norm(v) for v in h_vecs]
    v_lengths = [np.linalg.norm(v) for v in v_vecs]
    
    intersection_cost = 0
    
    # For each horizontal-vertical line pair, calculate the precise intersection point
    for h_idx in range(2):
        for v_idx in range(2):
            h_start = h_starts[h_idx]
            h_end = h_starts[h_idx] + h_vecs[h_idx]
            v_start = v_starts[v_idx]
            v_end = v_starts[v_idx] + v_vecs[v_idx]
            
            # Line intersection formula
            denom = (v_end[1] - v_start[1]) * (h_end[0] - h_start[0]) - (v_end[0] - v_start[0]) * (h_end[1] - h_start[1])
            
            if abs(denom) < 1e-6:  # Lines are parallel
                intersection_cost += 300
                continue
                
            ua = ((v_end[0] - v_start[0]) * (h_start[1] - v_start[1]) - (v_end[1] - v_start[1]) * (h_start[0] - v_start[0])) / denom
            ub = ((h_end[0] - h_start[0]) * (h_start[1] - v_start[1]) - (h_end[1] - h_start[1]) * (h_start[0] - v_start[0])) / denom
            
            # Check if intersection is within line segments
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                # Calculate the intersection point
                intersect_x = h_start[0] + ua * (h_end[0] - h_start[0])
                intersect_y = h_start[1] + ua * (h_end[1] - h_start[1])
                intersection_point = np.array([intersect_x, intersect_y])
                
                # For a proper hash, intersection should occur at ~1/3 or ~2/3 of each line
                target_fractions = [1/3, 2/3]
                
                # Calculate fraction along horizontal line
                h_fraction = ua
                min_h_fraction_diff = min(abs(h_fraction - target_fractions[0]), abs(h_fraction - target_fractions[1]))
                
                # Calculate fraction along vertical line
                v_fraction = ub
                min_v_fraction_diff = min(abs(v_fraction - target_fractions[0]), abs(v_fraction - target_fractions[1]))
                
                # Add cost based on how far intersection is from ideal 1/3 or 2/3 points
                intersection_cost += 150 * (min_h_fraction_diff**2 + min_v_fraction_diff**2)
            else:
                # Lines don't actually intersect within segments
                intersection_cost += 300
    
    cost += intersection_cost
    print(f"Hash intersection cost: {intersection_cost:.4f}")

    # 6. Length consistency
    h_lens = [np.linalg.norm(v) for v in h_vecs]
    v_lens = [np.linalg.norm(v) for v in v_vecs]
    cost += 50 * ((h_lens[0] - h_lens[1]) / (np.mean(h_lens) + 1e-6))**2
    cost += 50 * ((v_lens[0] - v_lens[1]) / (np.mean(v_lens) + 1e-6))**2

    # 7. Total horizontal vs vertical length balance
    h_total = np.sum(h_lens)
    v_total = np.sum(v_lens)
    cost += 20 * ((h_total - v_total) / (h_total + v_total + 1e-6))**2

    # 8. Deviation from Standard Length Cost with Barrier
    min_length_threshold = 75.0  # below this = heavy penalty
    if avg_size < min_length_threshold:
        cost += 7e2  # Massive penalty for being too small

    print(f"Total Hash Cost: {cost:.4f}")
    return cost



class Draw(Task):
    groundtruth_file = None

    def __init__(self, goal_str: str, **kwargs):
        self.goal_str = goal_str
        
        self.relevant_frames = ["cameraTop", "whiteboard"]

    def get_goal(self):
        return self.goal_str

    def setup_env(self, ON_REAL=False):
        return create_config(ON_REAL)

    def get_reward(self):
        return 0

    def get_cost(self):
        pass

class Star(Draw):
    number_lines = 10
    groundtruth_file = "groundtruths/groundtruth.npy"

    def get_cost(self, env):
        cost = compute_vector_cost_star(env.starting_points, env.vectors)
        print(f"Cost: {cost}")
        return cost


class Square(Draw):
    number_lines = 4
    groundtruth_file = "groundtruths/groundtruth_square.npy"

class Hexagon(Draw):
    number_lines = 6
    groundtruth_file = "groundtruths/groundtruth_hexagon.npy"
    
class Pentagon(Draw):
    number_lines = 5

    def get_cost(self, env):
        cost = compute_vector_cost_pentagon(env.starting_points, env.vectors)
        print(f"Cost: {cost}")
        return cost
    
class HashSymbol(Draw):
    number_lines = 4
    
    def get_cost(self, env):
        cost = compute_vector_cost_hash(env.starting_points, env.vectors)
        print(f"Hash cost: {cost}")
        return cost