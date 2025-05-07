#!/usr/bin/env python3

import os
import json
import time
import argparse
import subprocess
import numpy as np
from dataclasses import dataclass, asdict
from typing import Tuple
import rospy
import rospkg
from nav_msgs.msg import Odometry
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState

from _util import *

@dataclass
class TaskConfig:
    instruction: str
    start_position: Tuple[float, float, float]  # x, y, theta
    target_position: Tuple[float, float, float]  # x, y, theta
    time_limit: float = 300.0  # seconds
    task_id: str = ""


class NavigationEvaluationRunner:
    def __init__(self, tasks_file: str, output_dir: str):
        """
        Initialize the evaluation runner.
        
        Args:
            tasks_file: Path to JSON file with task definitions
            output_dir: Directory to save evaluation data
        """
        self.tasks_file = tasks_file
        self.output_dir = output_dir
        self.tasks = []
        self.current_robot_position = None
        
        # ROS setup
        try:
            rospy.init_node("navigation_evaluator", anonymous=True)
        except rospy.exceptions.ROSException:
            pass  # Node already initialized
            
        # Subscribe to robot's position
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # Setup services for controlling the robot's position
        rospy.wait_for_service('/gazebo/set_model_state')
        self.set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.get_model_state = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        
    def odom_callback(self, msg):
        """Callback to update robot position from odometry."""
        pose = msg.pose.pose
        self.current_robot_position = (
            pose.position.x,
            pose.position.y,
            2 * np.arctan2(pose.orientation.z, pose.orientation.w)  # Convert quaternion to theta
        )
        
    def load_tasks(self):
        """Load task definitions from JSON file."""
        with open(self.tasks_file, 'r') as f:
            task_data = json.load(f)
            
        for i, task_def in enumerate(task_data.get("tasks", [])):
            task = TaskConfig(
                instruction=task_def.get("instruction", ""),
                start_position=tuple(task_def.get("start_position", [0, 0, 0])),
                target_position=tuple(task_def.get("target_position", [0, 0, 0])),
                time_limit=task_def.get("time_limit", 300.0),
                task_id=task_def.get("id", f"task_{i+1}")
            )
            self.tasks.append(task)
            
        print(f"Loaded {len(self.tasks)} tasks for evaluation")
        
    def set_robot_position(self, position: Tuple[float, float, float]):
        """
        Set the robot's position in the simulation.
        
        Args:
            position: Tuple of (x, y, theta)
        """
        model_state = ModelState()
        model_state.model_name = "triton"  
        model_state.pose.position.x = position[0]
        model_state.pose.position.y = position[1]
        model_state.pose.position.z = 0.0
        
        # Convert theta to quaternion
        theta = position[2]
        model_state.pose.orientation.z = np.sin(theta / 2)
        model_state.pose.orientation.w = np.cos(theta / 2)
        
        try:
            self.set_model_state(model_state)
            rospy.sleep(1.0)  # Give time for the position to update
            return True
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
            return False
            
    def get_current_position(self):
        """Get the current robot position."""
        if self.current_robot_position:
            return self.current_robot_position
            
        try:
            # Fallback if odometry isn't available
            model_state = self.get_model_state("triton", "")
            pose = model_state.pose
            return (
                pose.position.x,
                pose.position.y,
                2 * np.arctan2(pose.orientation.z, pose.orientation.w)
            )
        except:
            return (0, 0, 0)
    
    def execute_task(self, task: TaskConfig):
        """
        Execute a single navigation task and collect evaluation data.
        
        Args:
            task: The task configuration to execute
            
        Returns:
            Path to the task output directory
        """
        # Create task directory
        task_dir = os.path.join(self.output_dir, task.task_id)
        os.makedirs(task_dir, exist_ok=True)
        
        # Save task configuration
        with open(os.path.join(task_dir, "task_config.json"), 'w') as f:
            json.dump(asdict(task), f, indent=2)
        
        # Set robot to start position
        print(f"Setting robot to start position: {task.start_position}")
        if not self.set_robot_position(task.start_position):
            print(f"Failed to set robot position for task: {task.task_id}")
            return task_dir
            
        # Modify the script to use the specific instruction
        rospack = rospkg.RosPack()
        script_path = os.path.join(rospack.get_path("NLBot"), "scripts")
        tmp_script_path = os.path.join(task_dir, "nl_command_executor_modified.py")
        
        with open(os.path.join(script_path, "main.py"), 'r') as f:
            script_content = f.read()
            
        # Replace the instruction
        script_content = script_content.replace(
            '# === Input natural language instruction ===\nnatural_language_instruction = """Go forward, cross the corridor until you leave it. \\\nThen you will be in a room with a big table, \\\nturn left after you take the exit from the corridor and fully enter the room. \\\nThen walk forward following the wall on your right hand for a distance, \\\nand there will be a door on the left. \\\nEnter that door, and the process ends."""',
            f'# === Input natural language instruction ===\nnatural_language_instruction = """{task.instruction}"""'
        )
        
        # Change the save path to the task directory
        script_content = script_content.replace(
            'SAVE_ROOT = "src/NLBot/scripts/tmp"',
            f'SAVE_ROOT = "{task_dir}"'
        )
        
        # Write the modified script
        with open(tmp_script_path, 'w') as f:
            f.write(script_content)
        os.chmod(tmp_script_path, 0o755)  # Make executable
        
        # Run the navigation script with timeout
        print(f"Executing task: {task.task_id}")
        print(f"Instruction: {task.instruction}")
        
        start_time = time.time()
        process = subprocess.Popen(["python3", tmp_script_path])
        
        try:
            # Monitor execution and timeout
            running = True
            while running and (time.time() - start_time) < task.time_limit:
                # Check if process is still running
                if process.poll() is not None:
                    running = False
                    break
                    
                # Record robot position periodically
                if (time.time() - start_time) % 5 < 0.1:  # Every ~5 seconds
                    current_pos = self.get_current_position()
                    step_dirs = [d for d in os.listdir(task_dir) if d.startswith("step_")]
                    if step_dirs:
                        latest_step = sorted(step_dirs)[-1]
                        pos_file = os.path.join(task_dir, latest_step, "position.json")
                        with open(pos_file, 'w') as f:
                            json.dump({
                                "x": current_pos[0],
                                "y": current_pos[1],
                                "theta": current_pos[2],
                                "timestamp": time.time()
                            }, f)
                
                time.sleep(0.1)
                
            # If timeout, terminate the process
            if running:
                print(f"Task timed out after {task.time_limit} seconds")
                process.terminate()
                process.wait(timeout=5)
                
        except KeyboardInterrupt:
            print("Evaluation interrupted by user")
            process.terminate()
            process.wait(timeout=5)
            
        # Record final position
        final_pos = self.get_current_position()
        with open(os.path.join(task_dir, "final_position.json"), 'w') as f:
            json.dump({
                "x": final_pos[0],
                "y": final_pos[1],
                "theta": final_pos[2],
                "timestamp": time.time()
            }, f)
            
        # Calculate distance to target
        target_dist = np.sqrt(
            (final_pos[0] - task.target_position[0])**2 +
            (final_pos[1] - task.target_position[1])**2
        )
        
        # Record execution summary
        execution_time = time.time() - start_time
        summary = {
            "task_id": task.task_id,
            "execution_time": execution_time,
            "distance_to_target": target_dist,
            "steps_taken": len([d for d in os.listdir(task_dir) if d.startswith("step_")])
        }
        
        with open(os.path.join(task_dir, "execution_summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"Task {task.task_id} completed in {execution_time:.2f} seconds")
        print(f"Distance to target: {target_dist:.2f} meters")
        
        return task_dir
        
    def run_evaluation(self):
        """Run the complete evaluation on all tasks."""
        if not self.tasks:
            self.load_tasks()
            
        if not self.tasks:
            print("No tasks to evaluate!")
            return
            
        os.makedirs(self.output_dir, exist_ok=True)
        
        results = {}
        for i, task in enumerate(self.tasks):
            print(f"\n[{i+1}/{len(self.tasks)}] Evaluating task: {task.task_id}")
            task_dir = self.execute_task(task)
            results[task.task_id] = task_dir
            
        # Save overall results
        with open(os.path.join(self.output_dir, "evaluation_summary.json"), 'w') as f:
            json.dump({
                "tasks_executed": len(self.tasks),
                "task_paths": results
            }, f, indent=2)
            
        print(f"\nEvaluation complete! Results saved to: {self.output_dir}")


def create_example_tasks_file(output_file: str = "navigation_tasks.json"):
    """Create an example tasks file with common navigation scenarios."""
    tasks = {
        "tasks": [
            {
                "id": "corridor_navigation",
                "instruction": "Go forward and navigate through the corridor until you reach the open area.",
                "start_position": [0, 0.5, 0],
                "target_position": [3, 0.5, 0],
                "time_limit": 180.0
            },
            {
                "id": "room_exploration",
                "instruction": "Go forward, then turn right when you see a door. Enter the room and stop.",
                "start_position": [0, 0.5, 0],
                "target_position": [6.5, -1, -1.07],
                "time_limit": 240.0
            },
            {
                "id": "object_approach",
                "instruction": "Navigate to the table in front of you and stop when you're close to it.",
                "start_position": [-3, 1, 1.07],
                "target_position": [-3, 1.7, 1.07],
                "time_limit": 120.0
            }
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(tasks, f, indent=2)
        
    print(f"Example tasks file created: {output_file}")
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run robot navigation evaluation')
    parser.add_argument('--tasks-file', type=str, default='navigation_tasks.json',
                        help='Path to JSON file with task definitions')
    parser.add_argument('--output-dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation data')
    parser.add_argument('--create-example', action='store_true',
                        help='Create an example tasks file')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    if args.create_example:
        create_example_tasks_file(args.tasks_file)
        return
        
    if not os.path.exists(args.tasks_file):
        print(f"Tasks file not found: {args.tasks_file}")
        print("Use --create-example to create an example tasks file")
        return
        
    print(f"Starting navigation evaluation with tasks from: {args.tasks_file}")
    runner = NavigationEvaluationRunner(args.tasks_file, args.output_dir)
    runner.run_evaluation()


if __name__ == "__main__":
    main()