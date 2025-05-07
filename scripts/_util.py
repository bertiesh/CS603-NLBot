#!/usr/bin/env python3
import hashlib
import contextlib
import tempfile
import os
import json
from dataclasses import asdict, is_dataclass
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Quaternion
import tf.transformations
import yaml

@contextlib.contextmanager
def change_dir(path):
    """
    Context manager to temporarily change the working directory.
    
    Args:
        path: Directory to change to
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

@contextlib.contextmanager
def create_tempdir():
    """
    Context manager to create and use a temporary directory.
    """
    with tempfile.TemporaryDirectory() as dirname:
        with change_dir(dirname):
            yield dirname

def dataclass_2_str(obj):
    """
    Convert a dataclass to a JSON string.
    
    Args:
        obj: Dataclass object
    
    Returns:
        JSON string representation
    """
    if is_dataclass(obj):
        return json.dumps(asdict(obj))
    return json.dumps(obj)

def hash_string(text):
    """
    Generate a hash from a string.
    
    Args:
        text: String to hash
    
    Returns:
        Hash string
    """
    return hashlib.md5(text.encode()).hexdigest()

def save_targets(targets_dict, filename):
    """
    Save a dictionary of named target locations to a YAML file.
    
    Args:
        targets_dict: Dictionary of named targets (name: PoseStamped)
        filename: Path to save the file
    """
    targets_data = {}
    
    for name, pose in targets_dict.items():
        # Convert PoseStamped to a dictionary
        pose_dict = {
            'position': {
                'x': pose.pose.position.x,
                'y': pose.pose.position.y,
                'z': pose.pose.position.z
            },
            'orientation': {
                'x': pose.pose.orientation.x,
                'y': pose.pose.orientation.y,
                'z': pose.pose.orientation.z,
                'w': pose.pose.orientation.w
            },
            'frame_id': pose.header.frame_id
        }
        targets_data[name] = pose_dict
    
    # Save as YAML
    with open(filename, 'w') as f:
        yaml.dump(targets_data, f, default_flow_style=False)
    
    return True

def load_targets(filename):
    """
    Load a dictionary of named target locations from a YAML file.
    
    Args:
        filename: Path to the YAML file
    
    Returns:
        Dictionary of named targets (name: PoseStamped)
    """
    if not os.path.exists(filename):
        rospy.logwarn(f"Targets file {filename} does not exist")
        return {}
    
    with open(filename, 'r') as f:
        targets_data = yaml.safe_load(f)
    
    if not targets_data:
        return {}
    
    targets_dict = {}
    
    for name, pose_dict in targets_data.items():
        # Create a PoseStamped from the dictionary
        pose = PoseStamped()
        pose.header.frame_id = pose_dict.get('frame_id', 'map')
        pose.header.stamp = rospy.Time.now()
        
        # Set position
        pose.pose.position.x = pose_dict['position']['x']
        pose.pose.position.y = pose_dict['position']['y']
        pose.pose.position.z = pose_dict['position']['z']
        
        # Set orientation
        pose.pose.orientation.x = pose_dict['orientation']['x']
        pose.pose.orientation.y = pose_dict['orientation']['y']
        pose.pose.orientation.z = pose_dict['orientation']['z']
        pose.pose.orientation.w = pose_dict['orientation']['w']
        
        targets_dict[name] = pose
    
    return targets_dict

def create_pose_from_xy_yaw(x, y, yaw, frame_id='map'):
    """
    Create a PoseStamped from x, y coordinates and a yaw angle.
    
    Args:
        x: X coordinate
        y: Y coordinate
        yaw: Yaw angle in radians
        frame_id: Frame ID for the pose
    
    Returns:
        PoseStamped object
    """
    pose = PoseStamped()
    pose.header.frame_id = frame_id
    pose.header.stamp = rospy.Time.now()
    
    # Set position
    pose.pose.position.x = x
    pose.pose.position.y = y
    pose.pose.position.z = 0.0
    
    # Convert yaw to quaternion
    q = tf.transformations.quaternion_from_euler(0, 0, yaw)
    pose.pose.orientation = Quaternion(q[0], q[1], q[2], q[3])
    
    return pose

def get_yaw_from_pose(pose):
    """
    Extract the yaw angle from a pose.
    
    Args:
        pose: Pose or PoseStamped object
    
    Returns:
        Yaw angle in radians
    """
    if hasattr(pose, 'pose'):
        # It's a PoseStamped
        orientation = pose.pose.orientation
    else:
        # It's just a Pose
        orientation = pose.orientation
    
    q = [orientation.x, orientation.y, orientation.z, orientation.w]
    euler = tf.transformations.euler_from_quaternion(q)
    return euler[2]  # yaw

def distance_between_poses(pose1, pose2):
    """
    Calculate the Euclidean distance between two poses.
    
    Args:
        pose1: First pose (Pose or PoseStamped)
        pose2: Second pose (Pose or PoseStamped)
    
    Returns:
        Distance in meters
    """
    # Extract positions
    if hasattr(pose1, 'pose'):
        # It's a PoseStamped
        pos1 = pose1.pose.position
    else:
        # It's just a Pose
        pos1 = pose1.position
        
    if hasattr(pose2, 'pose'):
        # It's a PoseStamped
        pos2 = pose2.pose.position
    else:
        # It's just a Pose
        pos2 = pose2.position
    
    # Calculate distance
    return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

def parse_nl_coordinates(text):
    """
    Parse natural language text to extract coordinates.
    
    Args:
        text: Natural language text that might contain coordinates
        
    Returns:
        Tuple of (x, y) coordinates or None if not found
    """
    import re
    
    # Look for patterns like "x: 1.2, y: 3.4" or "coordinates (1.2, 3.4)"
    patterns = [
        r'x\s*[\:\=]\s*(-?\d+\.?\d*)\s*,?\s*y\s*[\:\=]\s*(-?\d+\.?\d*)',
        r'coordinates\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',
        r'location\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',
        r'position\s*\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)',
        r'\(\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*\)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                x = float(match.group(1))
                y = float(match.group(2))
                return (x, y)
            except ValueError:
                continue
    
    return None