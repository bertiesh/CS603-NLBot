import rospy
from geometry_msgs.msg import Twist
import time
import tf
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import math

class RobotController:
    """ROS-based interface to control the Triton robot."""
    
    def __init__(self):
        """Initialize ROS node and publishers/subscribers for robot control."""
        # Initialize ROS node
        rospy.init_node('nl_bot_controller', anonymous=True)
        
        # Publishers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        
        # Subscribers
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Robot state variables
        self.position = {'x': 0.0, 'y': 0.0, 'orientation': 0.0}
        self.position_history = [self.position.copy()]
        self.current_image = None
        
        # Control parameters
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.5  # rad/s
        
        # Wait for subscribers to connect
        time.sleep(1)
        rospy.loginfo("Robot controller initialized")
    
    def odom_callback(self, msg):
        """Callback for odometry updates."""
        # Extract position from odometry message
        self.position['x'] = msg.pose.pose.position.x
        self.position['y'] = msg.pose.pose.position.y
        
        # Extract orientation from quaternion
        quaternion = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        )
        euler = tf.transformations.euler_from_quaternion(quaternion)
        self.position['orientation'] = euler[2]  # yaw (rotation around z-axis)
    
    def image_callback(self, msg):
        """Callback for camera image updates."""
        try:
            # Convert ROS Image message to OpenCV image
            self.current_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to convert image: {e}")
    
    def get_current_position(self):
        """Get the current position and orientation of the robot."""
        return self.position.copy()
    
    def get_current_image(self):
        """Get the current camera image."""
        return self.current_image.copy() if self.current_image is not None else None
    
    def move_forward(self, distance, timeout=30):
        """
        Move the robot forward by the specified distance in meters.
        
        Args:
            distance: Distance to move in meters
            timeout: Maximum time to spend on this operation
        """
        # Calculate time needed to move the specified distance
        # time_to_move = abs(distance) / self.linear_speed
        
        # Create Twist message for forward movement
        twist = Twist()
        twist.linear.x = self.linear_speed if distance > 0 else -self.linear_speed
        twist.angular.z = 0.0
        
        # Store starting position
        start_position = self.get_current_position()
        start_time = time.time()
        
        # Publish twist message continuously until distance is reached or timeout
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Check if we've reached the target distance
            current_position = self.get_current_position()
            dx = current_position['x'] - start_position['x']
            dy = current_position['y'] - start_position['y']
            distance_moved = math.sqrt(dx**2 + dy**2)
            
            # Check if we've reached the target or timed out
            if distance_moved >= abs(distance) or (time.time() - start_time) > timeout:
                break
            
            # Publish movement command
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        
        # Stop the robot
        self.stop()
        
        # Store new position in history
        self.position_history.append(self.get_current_position())
        
        return f"Moved forward {distance_moved:.2f} meters"
    
    def rotate(self, degrees, timeout=30, speed=None):
        """
        Rotate the robot by the specified angle in degrees.
        
        Args:
            degrees: Angle to rotate in degrees (positive = counter-clockwise)
            timeout: Maximum time to spend on this operation
            speed: Optional angular speed override (rad/s)
        """
        # Convert degrees to radians
        radians = math.radians(degrees)
        
        # Use specified speed or default
        angular_speed = speed if speed is not None else self.angular_speed
        angular_speed = abs(angular_speed)  # Make sure speed is positive
        
        # Create Twist message for rotation
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = angular_speed if radians > 0 else -angular_speed
        
        # Store starting orientation
        start_orientation = self.get_current_position()['orientation']
        start_time = time.time()
        
        # Publish twist message continuously until angle is reached or timeout
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            # Check if we've reached the target angle
            current_orientation = self.get_current_position()['orientation']
            # Calculate the angle rotated (handle wrapping)
            angle_rotated = abs(self.normalize_angle(current_orientation - start_orientation))
            
            # Check if we've reached the target or timed out
            if angle_rotated >= abs(radians) or (time.time() - start_time) > timeout:
                break
            
            # Publish rotation command
            self.cmd_vel_pub.publish(twist)
            rate.sleep()
        
        # Stop the robot
        self.stop()
        
        # Store new position in history
        self.position_history.append(self.get_current_position())
        
        return f"Rotated {math.degrees(angle_rotated):.2f} degrees"
    
    def normalize_angle(self, angle):
        """Normalize angle to be between -pi and pi."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def stop(self):
        """Stop all robot movement."""
        twist = Twist()  # Zero twist = stop
        self.cmd_vel_pub.publish(twist)
        
        # Publish several times to ensure robot stops
        for _ in range(3):
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.1)
        
        return "Robot stopped"
    
    def navigate_to_target(self, timeout=60):
        """
        Navigate to an identified target.
        """
        rospy.loginfo("Starting navigation to target...")
        
        # Call a ROS action server
        # for navigation, TODO: Implement actual navigation logic
        # This is a placeholder for the actual navigation logic
        
        return "Navigated toward target"
    
    def backtrack(self):
        """Return to the previous position using the same motion model as particle propagation."""
        if len(self.position_history) < 2:
            return "Cannot backtrack, no previous position"
        
        # Get current and previous positions
        current_pos = self.get_current_position()
        prev_pos = self.position_history[-2]
        
        # Calculate the differences between positions (in world frame)
        dx = prev_pos['x'] - current_pos['x']
        dy = prev_pos['y'] - current_pos['y']
        dyaw = prev_pos['orientation'] - current_pos['orientation']
        
        # Normalize angle difference
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        
        # Calculate motion parameters
        delta_trans = math.sqrt(dx**2 + dy**2)
        
        # Check if there's significant movement needed
        if delta_trans < 0.01 and abs(dyaw) < 0.01:
            rospy.loginfo("Already at previous position, no need to backtrack")
            return "Already at previous position"
        
        # Decompose motion into robot frame
        if delta_trans < 1e-5:
            # Pure rotation
            delta_rot1 = 0.0
            delta_rot2 = dyaw
        else:
            # Mixed rotation and translation
            # First compute heading of the motion in world frame
            motion_heading = math.atan2(dy, dx)
            
            # delta_rot1 is from robot's current heading to motion heading
            delta_rot1 = motion_heading - current_pos['orientation']
            
            # delta_rot2 is rotation needed after translation to reach final heading
            delta_rot2 = dyaw - delta_rot1
            
            # Normalize angles
            delta_rot1 = math.atan2(math.sin(delta_rot1), math.cos(delta_rot1))
            delta_rot2 = math.atan2(math.sin(delta_rot2), math.cos(delta_rot2))
        
        # Execute the backtracking motion
        
        # 1. First rotation to align with path
        rospy.loginfo(f"Rotating {math.degrees(delta_rot1):.2f} degrees to align with path")
        self.rotate(math.degrees(delta_rot1))
        
        # 2. Translation to previous position
        rospy.loginfo(f"Moving {delta_trans:.2f} meters toward previous position")
        self.move_forward(delta_trans)
        
        # 3. Second rotation to match previous orientation
        rospy.loginfo(f"Rotating {math.degrees(delta_rot2):.2f} degrees to match previous orientation")
        self.rotate(math.degrees(delta_rot2))
        
        # Remove the last position from history if backtrack was successful
        current_pos_after = self.get_current_position()
        
        # Calculate how close we got to the target
        dx_after = prev_pos['x'] - current_pos_after['x']
        dy_after = prev_pos['y'] - current_pos_after['y']
        delta_trans_after = math.sqrt(dx_after**2 + dy_after**2)
        
        if delta_trans_after < 0.1:  # Within 10cm of target
            if len(self.position_history) > 1:
                self.position_history.pop()
            return f"Backtracked to previous position (within {delta_trans_after:.2f}m)"
        else:
            return f"Attempted to backtrack but ended {delta_trans_after:.2f}m from target"
    
    def explore(self):
        """Perform a 360-degree scan of the environment."""
        self.rotate(360, speed=self.angular_speed/2)  # Slow rotation for better scanning
        return "Completed 360-degree scan of surroundings"