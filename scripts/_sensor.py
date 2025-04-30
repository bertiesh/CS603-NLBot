#!/usr/bin/env python3
import cv2
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import Image
import os

latest_scan: LaserScan = None
latest_image: Image = None

def scan_callback(msg):
    global latest_scan
    latest_scan = msg

def camera_callback(msg):
    global latest_image
    latest_image = msg

rospy.Subscriber("/scan", LaserScan, scan_callback)
rospy.Subscriber('/camera/image_raw', Image, camera_callback)

def save_view(dir):
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


if __name__ == "__main__":
    rospy.init_node("image_and_lidar_saver", anonymous=True)
    rospy.sleep(1.0)
    save_dir = "src/NLBot/scripts/tmp"
    os.makedirs(save_dir, exist_ok=True)
    save_view(save_dir)
    save_radar(save_dir)
