#!/usr/bin/env python3

import os
import rospy
from openai import OpenAI
from geometry_msgs.msg import Twist, PoseStamped, PoseWithCovarianceStamped
import base64
import numpy as np
import cv2
import tf
from nav_msgs.msg import OccupancyGrid, Path
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from std_msgs.msg import Header
from _sensor import save_view, save_radar
from _util import *

# === Configuration ===
GAP_SEC = 1
SAVE_ROOT = "src/NLBot/scripts/tmp"
LIN_V = 0.3
ANG_V = 0.1

class SLAMRobotController:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("nl_slam_controller")
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        self.goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
        
        # Subscribers
        self.map_sub = rospy.Subscriber("/map", OccupancyGrid, self.map_callback)
        self.amcl_pose_sub = rospy.Subscriber("/amcl_pose", PoseWithCovarianceStamped, self.pose_callback)
        self.path_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.path_callback)
        
        # Action client for move_base
        self.move_base_client = actionlib.SimpleActionClient("move_base", MoveBaseAction)
        
        # Initialize state variables
        self.current_map = None
        self.current_pose = None
        self.current_path = None
        self.step = 0
        self.message_history = []
        self.navigation_mode = "manual"  # "manual" or "slam"
        self.targets = {}  # Dictionary to store named locations
        
        # TF listener for coordinate transformations
        self.tf_listener = tf.TransformListener()
        
        # Configure OpenAI client
        self.client = OpenAI()
        
        # Initialize system prompt
        self.initialize_system_prompt()
        
    def initialize_system_prompt(self):
        """Initialize the system prompt for GPT-4o"""
        self.message_history = [{
            "role": "system",
            "content": f"""
            You are helping a robot understand and execute navigation instructions using natural language. 
            The robot can operate in two modes:
            1. Manual Mode: You provide direct movement commands (left, right, forward, stop)
            2. SLAM Mode: You identify targets and the robot navigates to them autonomously

            In every round, you will receive:
            - The robot's current camera view (view.jpg)
            - A visualization of the LiDAR scan (radar.jpg)
            - In SLAM mode, you'll also receive a map visualization with the robot's position and path

            Your task is to decide the robot's next action:

            MANUAL MODE COMMANDS:
            A - Go left (linear velocity: 0, angular velocity: {ANG_V} * PI)
            B - Go forward (linear velocity: {LIN_V}, angular velocity: 0)
            C - Go right (linear velocity: 0, angular velocity: -{ANG_V} * PI)
            D - Stop (linear and angular velocity: 0, task completed)

            SLAM MODE COMMANDS:
            E - Mark current location as a named target (e.g., "E - Mark as kitchen")
            F - Navigate to a previously marked target (e.g., "F - Navigate to kitchen")
            G - Create a new navigation goal at coordinates (e.g., "G - Navigate to x:1.5 y:2.3")
            H - Switch to Manual Mode
            I - Switch to SLAM Mode

            You may provide a reasoning trace before you make the decision.
            Please analyze:
            - What does the robot currently see?
            - Where might it be in the environment?
            - Is the robot close to hitting obstacles?
            - Which part of the instruction has been completed?
            - What action should the robot take now?

            Stay away from obstacles shown on the radar! Safety takes priority over completing instructions!

            The response includes a natural language analysis and the choice, split by "----".
            Response example:
            ```
            ...(some analysis)
            ...The robot should turn left.
            ----
            A - Go left
            ```

            Or for SLAM mode:
            ```
            ...(some analysis)
            ...The robot should mark this location.
            ----
            E - Mark as livingroom
            ```
            """
        }]
    
    def map_callback(self, msg):
        """Callback for receiving map updates"""
        self.current_map = msg
        
    def pose_callback(self, msg):
        """Callback for receiving pose updates"""
        self.current_pose = msg
        
    def path_callback(self, msg):
        """Callback for receiving path updates"""
        self.current_path = msg
    
    def save_map_visualization(self, step_dir):
        """Save a visualization of the current map with robot position and path"""
        if self.current_map is None or self.current_pose is None:
            rospy.logwarn("Map or pose not available for visualization")
            return False
        
        # Create map visualization
        map_data = np.array(self.current_map.data).reshape(
            self.current_map.info.height, self.current_map.info.width)
        
        # Convert to RGB image
        map_vis = np.zeros((map_data.shape[0], map_data.shape[1], 3), dtype=np.uint8)
        
        # Unknown space (gray)
        map_vis[map_data == -1] = [128, 128, 128]
        # Free space (white)
        map_vis[map_data == 0] = [255, 255, 255]
        # Occupied space (black)
        map_vis[map_data > 0] = [0, 0, 0]
        
        # Convert robot pose to map coordinates
        x = self.current_pose.pose.pose.position.x
        y = self.current_pose.pose.pose.position.y
        
        # Get orientation as Euler angles
        quaternion = (
            self.current_pose.pose.pose.orientation.x,
            self.current_pose.pose.pose.orientation.y,
            self.current_pose.pose.pose.orientation.z,
            self.current_pose.pose.pose.orientation.w
        )
        _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
        
        # Convert world coordinates to map coordinates
        map_x = int((x - self.current_map.info.origin.position.x) / self.current_map.info.resolution)
        map_y = int((y - self.current_map.info.origin.position.y) / self.current_map.info.resolution)
        
        # Draw robot position (red dot)
        if 0 <= map_x < map_vis.shape[1] and 0 <= map_y < map_vis.shape[0]:
            cv2.circle(map_vis, (map_x, map_y), 3, (0, 0, 255), -1)
            
            # Draw orientation line
            end_x = int(map_x + 10 * np.cos(yaw))
            end_y = int(map_y + 10 * np.sin(yaw))
            cv2.line(map_vis, (map_x, map_y), (end_x, end_y), (0, 0, 255), 2)
        
        # Draw path if available
        if self.current_path is not None and len(self.current_path.poses) > 0:
            path_points = []
            for pose in self.current_path.poses:
                px = pose.pose.position.x
                py = pose.pose.position.y
                map_px = int((px - self.current_map.info.origin.position.x) / self.current_map.info.resolution)
                map_py = int((py - self.current_map.info.origin.position.y) / self.current_map.info.resolution)
                path_points.append((map_px, map_py))
            
            if len(path_points) > 1:
                for i in range(len(path_points) - 1):
                    cv2.line(map_vis, path_points[i], path_points[i + 1], (0, 255, 0), 2)
        
        # Draw markers for saved targets
        for name, target_pose in self.targets.items():
            tx = target_pose.pose.position.x
            ty = target_pose.pose.position.y
            map_tx = int((tx - self.current_map.info.origin.position.x) / self.current_map.info.resolution)
            map_ty = int((ty - self.current_map.info.origin.position.y) / self.current_map.info.resolution)
            
            if 0 <= map_tx < map_vis.shape[1] and 0 <= map_ty < map_vis.shape[0]:
                cv2.circle(map_vis, (map_tx, map_ty), 5, (255, 0, 0), -1)
                cv2.putText(map_vis, name, (map_tx + 5, map_ty + 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Save the visualization
        cv2.imwrite(f"{step_dir}/map.jpg", map_vis)
        return True
    
    def encode_image_base64(self, path):
        """Encode an image to base64"""
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"
    
    def send_to_gpt_with_images(self, step_dir):
        """Send the current state to GPT-4o and get a decision"""
        view_path = os.path.join(step_dir, "view.jpg")
        radar_path = os.path.join(step_dir, "radar.jpg")
        
        view_b64 = self.encode_image_base64(view_path)
        radar_b64 = self.encode_image_base64(radar_path)
        
        content_items = [
            {"type": "text", "text": "Here are the current camera view and LiDAR radar scan."},
            {"type": "image_url", "image_url": {"url": view_b64}},
            {"type": "image_url", "image_url": {"url": radar_b64}}
        ]
        
        # If in SLAM mode and map visualization exists, include it
        map_path = os.path.join(step_dir, "map.jpg")
        if self.navigation_mode == "slam" and os.path.exists(map_path):
            map_b64 = self.encode_image_base64(map_path)
            content_items.append({"type": "text", "text": "Here is the current SLAM map with robot position and path:"})
            content_items.append({"type": "image_url", "image_url": {"url": map_b64}})
            
            # Add information about saved targets
            if self.targets:
                targets_info = "Saved targets:\n"
                for name in self.targets:
                    targets_info += f"- {name}\n"
                content_items.append({"type": "text", "text": targets_info})
        
        # Add current mode information
        content_items.append({"type": "text", "text": f"Current mode: {self.navigation_mode.upper()}"})
            
        response = self.client.chat.completions.create(
            model="gpt-4o",
            tools=[],
            tool_choice="none",
            messages=[
                *self.message_history,
                {
                    "role": "user",
                    "content": content_items
                }
            ],
        )
        return response.choices[0].message.content.strip()
    
    def get_letter(self, gpt_reply):
        """Extract action letter from GPT-4o reply"""
        try:
            result = gpt_reply.split("----")[-1].strip()
            letter = result.split(" ")[0].upper()
            remaining = result[len(letter):].strip()
            return letter, remaining
        except:
            rospy.logwarn("Failed to parse GPT-4o reply for action letter")
            return None, ""
    
    def execute_manual_action(self, letter):
        """Execute a manual control action"""
        twist = Twist()
        if letter == "A":
            twist.linear.x = 0
            twist.angular.z = ANG_V * np.pi
            rospy.loginfo("Manual action: Turn left")
        elif letter == "B":
            twist.linear.x = LIN_V
            twist.angular.z = 0.0
            rospy.loginfo("Manual action: Go forward")
        elif letter == "C":
            twist.linear.x = 0
            twist.angular.z = -ANG_V * np.pi
            rospy.loginfo("Manual action: Turn right")
        elif letter == "D":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            rospy.loginfo("Manual action: Stop (task complete)")
        else:
            rospy.logwarn(f"Invalid manual action: {letter}")
            return False
            
        self.cmd_vel_pub.publish(twist)
        return True
    
    def execute_slam_action(self, letter, params):
        """Execute a SLAM-based action"""
        if letter == "E":  # Mark current location
            if not params.startswith("- Mark as"):
                target_name = params.strip()
            else:
                target_name = params.replace("- Mark as", "").strip()
                
            if not target_name:
                rospy.logwarn("No target name provided for marking location")
                return False
                
            if self.current_pose is None:
                rospy.logwarn("Cannot mark location: robot pose not available")
                return False
                
            # Create a copy of the current pose as a PoseStamped
            pose_stamped = PoseStamped()
            pose_stamped.header = Header()
            pose_stamped.header.stamp = rospy.Time.now()
            pose_stamped.header.frame_id = "map"
            pose_stamped.pose = self.current_pose.pose.pose
            
            # Store the target
            self.targets[target_name] = pose_stamped
            rospy.loginfo(f"Marked current location as '{target_name}'")
            return True
            
        elif letter == "F":  # Navigate to named target
            target_name = params.replace("- Navigate to", "").strip()
            
            if not target_name or target_name not in self.targets:
                rospy.logwarn(f"Target '{target_name}' not found")
                return False
                
            # Get the target pose
            target_pose = self.targets[target_name]
            
            # Create a goal for move_base
            goal = MoveBaseGoal()
            goal.target_pose = target_pose
            
            rospy.loginfo(f"Navigating to target '{target_name}'")
            
            # Send the goal to move_base
            self.move_base_client.send_goal(goal)
            return True
            
        elif letter == "G":  # Navigate to coordinates
            # Extract coordinates from params
            import re
            coords = re.search(r"x:(-?\d+\.?\d*)\s+y:(-?\d+\.?\d*)", params)
            
            if not coords:
                rospy.logwarn("Could not parse coordinates from params")
                return False
                
            x = float(coords.group(1))
            y = float(coords.group(2))
            
            # Create a goal pose
            goal_pose = PoseStamped()
            goal_pose.header.frame_id = "map"
            goal_pose.header.stamp = rospy.Time.now()
            goal_pose.pose.position.x = x
            goal_pose.pose.position.y = y
            goal_pose.pose.orientation.w = 1.0  # Default orientation (no rotation)
            
            # Create a goal for move_base
            goal = MoveBaseGoal()
            goal.target_pose = goal_pose
            
            rospy.loginfo(f"Navigating to coordinates: x={x}, y={y}")
            
            # Send the goal to move_base
            self.move_base_client.send_goal(goal)
            return True
            
        elif letter == "H":  # Switch to Manual Mode
            self.navigation_mode = "manual"
            rospy.loginfo("Switched to Manual Mode")
            return True
            
        elif letter == "I":  # Switch to SLAM Mode
            self.navigation_mode = "slam"
            rospy.loginfo("Switched to SLAM Mode")
            
            # Make sure move_base is available
            rospy.loginfo("Waiting for move_base action server...")
            try:
                self.move_base_client.wait_for_server(rospy.Duration(5.0))
                rospy.loginfo("move_base action server ready")
            except:
                rospy.logwarn("move_base action server not available - SLAM navigation might not work properly")
                
            return True
            
        else:
            rospy.logwarn(f"Invalid SLAM action: {letter}")
            return False
    
    def main_loop(self, natural_language_instruction=None):
        """Main control loop for the robot"""
        if natural_language_instruction:
            # Add the instruction to the system prompt
            self.message_history[0]["content"] += f"\n\nCurrent instruction: \"{natural_language_instruction}\""
        
        # Wait for sensors to stabilize
        rospy.sleep(2.0)
        
        hz = 10
        rate = rospy.Rate(hz)
        
        # Initialize robot in manual mode
        self.navigation_mode = "manual"
        twist = Twist()
        
        # Main loop
        while not rospy.is_shutdown():
            self.cmd_vel_pub.publish(twist)
            self.step += 1
            rate.sleep()
            
            # Only process at specified intervals
            if self.step % (GAP_SEC * hz) != 0:
                continue
                
            rospy.loginfo(f"[Step {self.step}] Processing...")
            
            # Create directory for this step
            step_dir = os.path.join(SAVE_ROOT, f"step_{self.step:03d}")
            os.makedirs(step_dir, exist_ok=True)
            
            # Save sensor data
            save_view(step_dir)
            save_radar(step_dir)
            
            # In SLAM mode, also save map visualization
            if self.navigation_mode == "slam":
                self.save_map_visualization(step_dir)
                
            # Get decision from GPT-4o
            rospy.loginfo("Sending state to GPT-4o...")
            gpt_reply = self.send_to_gpt_with_images(step_dir)
            rospy.loginfo(f"GPT response: {gpt_reply}")
            
            # Save response
            with open(f"{step_dir}/reply.txt", "w") as f:
                f.write(gpt_reply)
                
            # Add to message history
            self.message_history.append({"role": "assistant", "content": gpt_reply})
            
            # Parse action
            letter, params = self.get_letter(gpt_reply)
            if letter is None:
                rospy.logwarn("Could not parse action from GPT response")
                continue
                
            # Execute action based on current mode
            if letter in ["A", "B", "C", "D"]:
                # Manual mode actions
                success = self.execute_manual_action(letter)
                if letter == "D" and success:  # Task complete
                    break
            elif letter in ["E", "F", "G", "H", "I"]:
                # SLAM mode actions
                self.execute_slam_action(letter, params)
            else:
                rospy.logwarn(f"Unknown action: {letter}")
            
            # Update twist for next iteration if in manual mode
            if self.navigation_mode == "manual" and letter in ["A", "B", "C", "D"]:
                if letter == "A":
                    twist.linear.x = 0
                    twist.angular.z = ANG_V * np.pi
                elif letter == "B":
                    twist.linear.x = LIN_V
                    twist.angular.z = 0.0
                elif letter == "C":
                    twist.linear.x = 0
                    twist.angular.z = -ANG_V * np.pi
                elif letter == "D":
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

def main():
    try:
        # Natural language instruction
        instruction = """Go forward, cross the corridor until you leave it. \
        Then you will be in a room with a big table, \
        turn left after you take the exit from the corridor and fully enter the room. \
        Then walk forward following the wall on your right hand for a distance, \
        and there will be a door on the left. \
        Enter that door, and the process ends."""

        # Create and start the controller
        controller = SLAMRobotController()
        controller.main_loop(instruction)
        
    except rospy.ROSInterruptException:
        pass

if __name__ == "__main__":
    main()