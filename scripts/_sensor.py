#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseWithCovarianceStamped
import os
import tf

latest_scan: LaserScan = None
latest_image: Image = None
latest_map: OccupancyGrid = None
latest_pose: PoseWithCovarianceStamped = None

def scan_callback(msg):
    global latest_scan
    latest_scan = msg

def camera_callback(msg):
    global latest_image
    latest_image = msg

def map_callback(msg):
    global latest_map
    latest_map = msg

def pose_callback(msg):
    global latest_pose
    latest_pose = msg

# Initialize subscribers
rospy.Subscriber("/scan", LaserScan, scan_callback)
rospy.Subscriber('/camera/image_raw', Image, camera_callback)
rospy.Subscriber('/map', OccupancyGrid, map_callback)
rospy.Subscriber('/amcl_pose', PoseWithCovarianceStamped, pose_callback)

def save_view(dir):
    """Save the current camera view as an image"""
    global latest_image
    if latest_image is None:
        rospy.logwarn("No image received yet.")
        return

    if latest_image.encoding != "rgb8":
        rospy.logerr(f"Unsupported encoding: {latest_image.encoding}")
        return

    img_array = np.frombuffer(latest_image.data, dtype=np.uint8).reshape(
        latest_image.height, latest_image.width, 3
    )

    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{dir}/view.jpg", img_bgr)


def save_radar(dir, img_size=600):
    """Save the current LiDAR scan as a radar visualization"""
    global latest_scan
    if latest_scan is None:
        rospy.logwarn("No scan received yet.")
        return

    # Compute scale: 1 meter = ? pixels
    meter_to_pixel = img_size / 10.0  # because we visualize ±5 meters
    center = img_size // 2

    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # white background

    angle = latest_scan.angle_min
    for r in latest_scan.ranges:
        if np.isinf(r) or np.isnan(r):
            angle += latest_scan.angle_increment
            continue
        if r > 5.0:
            angle += latest_scan.angle_increment
            continue

        x = r * np.sin(angle)
        y = r * np.cos(angle)

        px = int(center - x * meter_to_pixel)  # flip x to fix left-right mapping
        py = int(center - y * meter_to_pixel)

        if 0 <= px < img_size and 0 <= py < img_size:
            radius = max(1, int(0.025 * meter_to_pixel))  # ensure at least 1 pixel
            cv2.circle(img, (px, py), radius, (255, 0, 0), thickness=-1)

        angle += latest_scan.angle_increment

    # Draw green arrow pointing up (robot forward direction)
    arrow_length = int(0.5 * meter_to_pixel)
    arrow_tip = (center, center - arrow_length)
    cv2.arrowedLine(img, (center, center), arrow_tip, (0, 255, 0), 2, tipLength=0.2)

    # Draw red scale bar (1 meter)
    bar_length = int(meter_to_pixel)
    bar_start = (10, img_size - 10)
    bar_end = (10 + bar_length, img_size - 10)
    cv2.line(img, bar_start, bar_end, (0, 0, 255), 2)
    cv2.putText(img, "1m", (10, img_size - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    # Draw cross arrow for left-right in bottom right corner
    cross_center = (int(img_size * 0.8), int(img_size * 0.95))
    arrow_len = 20

    # Right arrow
    right_tip = (cross_center[0] + arrow_len, cross_center[1])
    cv2.arrowedLine(img, cross_center, right_tip, (0, 0, 0), 1, tipLength=0.3)
    cv2.putText(img, "right", (right_tip[0] + 5, right_tip[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Left arrow
    left_tip = (cross_center[0] - arrow_len, cross_center[1])
    cv2.arrowedLine(img, cross_center, left_tip, (0, 0, 0), 1, tipLength=0.3)
    cv2.putText(img, "left", (left_tip[0] - 35, left_tip[1] + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    fov_rad = 1.047
    half_fov = fov_rad / 2
    fov_length = int(5.0 * 2 * meter_to_pixel)

    left_angle = np.pi / 2 + half_fov
    left_x = int(center - np.cos(left_angle) * fov_length)
    left_y = int(center - np.sin(left_angle) * fov_length)
    cv2.line(img, (center, center), (left_x, left_y), (18,153,255), 2)

    right_angle = np.pi / 2 - half_fov
    right_x = int(center - np.cos(right_angle) * fov_length)
    right_y = int(center - np.sin(right_angle) * fov_length)
    cv2.line(img, (center, center), (right_x, right_y), (18,153,255), 2)

    cv2.imwrite(f"{dir}/radar.jpg", img)

def save_map(dir, img_size=800):
    """Save the current occupancy grid map with robot position"""
    global latest_map, latest_pose
    if latest_map is None:
        rospy.logwarn("No map received yet.")
        return False
        
    if latest_pose is None:
        rospy.logwarn("No pose received yet.")
        return False
        
    # Create map visualization
    map_data = np.array(latest_map.data).reshape(
        latest_map.info.height, latest_map.info.width)
    
    # If map is too big, resize for better visualization
    original_height = map_data.shape[0]
    original_width = map_data.shape[1]
    scale_factor = min(img_size / original_height, img_size / original_width)
    
    if scale_factor < 1:
        new_height = int(original_height * scale_factor)
        new_width = int(original_width * scale_factor)
        map_data_resized = cv2.resize(map_data.astype(np.uint8), (new_width, new_height), 
                                   interpolation=cv2.INTER_NEAREST)
    else:
        map_data_resized = map_data
        new_height = original_height
        new_width = original_width
    
    # Convert to RGB image
    map_vis = np.zeros((new_height, new_width, 3), dtype=np.uint8)
    
    # Unknown space (gray)
    map_vis[map_data_resized == -1] = [128, 128, 128]
    # Free space (white)
    map_vis[map_data_resized == 0] = [255, 255, 255]
    # Occupied space (black)
    map_vis[map_data_resized > 0] = [0, 0, 0]
    
    # Convert robot pose to map coordinates
    x = latest_pose.pose.pose.position.x
    y = latest_pose.pose.pose.position.y
    
    # Get orientation as Euler angles
    quaternion = (
        latest_pose.pose.pose.orientation.x,
        latest_pose.pose.pose.orientation.y,
        latest_pose.pose.pose.orientation.z,
        latest_pose.pose.pose.orientation.w
    )
    _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
    
    # Convert world coordinates to map coordinates
    map_x = int((x - latest_map.info.origin.position.x) / latest_map.info.resolution * scale_factor)
    map_y = int((y - latest_map.info.origin.position.y) / latest_map.info.resolution * scale_factor)
    
    # Draw robot position (red dot)
    if 0 <= map_x < map_vis.shape[1] and 0 <= map_y < map_vis.shape[0]:
        cv2.circle(map_vis, (map_x, map_y), 5, (0, 0, 255), -1)
        
        # Draw orientation line (direction the robot is facing)
        end_x = int(map_x + 15 * np.cos(yaw))
        end_y = int(map_y + 15 * np.sin(yaw))
        cv2.arrowedLine(map_vis, (map_x, map_y), (end_x, end_y), (0, 0, 255), 2)
    
    # Add scale and legend
    # Draw a scale bar of 1 meter
    scale_length = int(1.0 / latest_map.info.resolution * scale_factor)
    cv2.line(map_vis, (10, new_height - 10), (10 + scale_length, new_height - 10), (0, 0, 255), 2)
    cv2.putText(map_vis, "1m", (10, new_height - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Add legend for map colors
    legend_y = 30
    cv2.putText(map_vis, "Map Legend:", (10, legend_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
    cv2.rectangle(map_vis, (10, legend_y + 10), (30, legend_y + 30), (0, 0, 0), -1)
    cv2.putText(map_vis, "Obstacle", (35, legend_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
    cv2.rectangle(map_vis, (10, legend_y + 35), (30, legend_y + 55), (255, 255, 255), -1)
    cv2.putText(map_vis, "Free Space", (35, legend_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                
    cv2.rectangle(map_vis, (10, legend_y + 60), (30, legend_y + 80), (128, 128, 128), -1)
    cv2.putText(map_vis, "Unknown", (35, legend_y + 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.circle(map_vis, (20, legend_y + 95), 5, (0, 0, 255), -1)
    cv2.putText(map_vis, "Robot Position", (35, legend_y + 100), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save the map image
    cv2.imwrite(f"{dir}/map.jpg", map_vis)
    return True

if __name__ == "__main__":
    rospy.init_node("image_and_lidar_saver", anonymous=True)
    rospy.sleep(1.0)
    save_dir = "src/NLBot/scripts/tmp"
    os.makedirs(save_dir, exist_ok=True)
    save_view(save_dir)
    save_radar(save_dir)
    save_map(save_dir)