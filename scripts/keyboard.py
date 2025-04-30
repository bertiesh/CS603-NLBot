#!/usr/bin/python3

from __future__ import print_function

import rospy
from geometry_msgs.msg import Twist

from threading import Thread
import time
import sys
from pynput import keyboard

# Initialize ROS node and publisher
rospy.init_node("teleop_robot")
vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=2)

LIN_SPEED = 0.7
ANG_SPEED = 1.0

vel_msg = Twist()
key_state = {}

def key_update(key, state):
    if key not in key_state:
        key_state[key] = state
        return True
    if state != key_state[key]:
        key_state[key] = state
        return True
    return False

stop_display = False

def key_press(key):
    if key == keyboard.Key.esc:
        global stop_display
        stop_display = True
        print('\nPress Ctrl+C to exit')
        return False

    try:
        k = key.char
    except:
        k = key.name

    change = key_update(key, True)
    if change:
        global vel_msg, LIN_SPEED, ANG_SPEED
        if k in ['w', 'up']:
            vel_msg.linear.x += LIN_SPEED
        elif k in ['s', 'down']:
            vel_msg.linear.x -= LIN_SPEED
        elif k in ['d', 'right']:
            vel_msg.angular.z -= ANG_SPEED
        elif k in ['a', 'left']:
            vel_msg.angular.z += ANG_SPEED
        elif k == 'e':
            vel_msg.angular.z -= ANG_SPEED
        elif k == 'q':
            vel_msg.angular.z += ANG_SPEED
        elif k == 'x':
            LIN_SPEED += 0.1
        elif k == 'z':
            LIN_SPEED = max(0.1, LIN_SPEED - 0.1)  # Avoid negative speed
    return True

def key_release(key):
    try:
        k = key.char
    except:
        k = key.name

    change = key_update(key, False)
    if change:
        global vel_msg
        if k in ['w', 'up', 's', 'down']:
            vel_msg.linear.x = 0
        elif k in ['a', 'left', 'd', 'right', 'q', 'e']:
            vel_msg.angular.z = 0
    return True

rate = rospy.Rate(10)

def user_display():
    print('Use W/S or ↑/↓ to move forward/backward.\nUse A/D or ←/→ to rotate left/right.\nUse Q & E to rotate as well.\nUse X/Z to increase/decrease speed.')
    while True:
        try:
            print('\r' + ' '*80, end='')
            sys.stdout.flush()
            log_str = "\r\t\tX: {:.2f}\tTHETA: {:.2f}\tSpeed: {:.1f}".format(
                vel_msg.linear.x,
                vel_msg.angular.z,
                LIN_SPEED
            )
            print(log_str, end=' ')
            sys.stdout.flush()

            global stop_display
            if stop_display:
                exit(0) 

            if not rospy.is_shutdown():
                rate.sleep()
                vel_pub.publish(vel_msg)
            else:
                exit(0)
        except KeyboardInterrupt:
            exit(0)

# Start listener and display threads
key_listener = keyboard.Listener(on_press=key_press, on_release=key_release)
key_listener.start()

display_thread = Thread(target=user_display)
display_thread.start()

rospy.spin()
