#!/usr/bin/env python3

import os
import time
import rospy
from openai import OpenAI
import shutil
from geometry_msgs.msg import Twist
import base64
import json

from _sensor import save_view, save_radar

# === Configuration ===
GAP_SEC = 1
SAVE_ROOT = "src/NLBot/scripts/tmp"
LIN_V = 0.3
ANG_V = 0.1

# === Input natural language instruction ===
natural_language_instruction = """Go forward, cross the corridor until you leave it. \
Then you will be in a room with a big table, \
turn left after you take the exit from the corridor and fully enter the room. \
Then walk forward following the wall on your right hand for a distance, \
and there will be a door on the left. \
Enter that door, and the process ends."""

# === Conversation history ===
message_history = [{
    "role": "system",
    "content": f"""
You are helping a robot understand and execute a natural language navigation instruction. 
The instruction is:

    \"{natural_language_instruction}\"

In every round, you will receive two images:
- The first is the robot's current camera view (view.jpg).
- The second is a visualization of the LiDAR scan (radar.jpg).

Note that (very important) in the LiDAR scan visualization image, the orange lines demonstrate the FOV of the robot camera.

Your task is to decide the robot's next action for the next {GAP_SEC} seconds.

You must always choose one of the following actions:
A - Go left (linear velocity: 0, angular velocity: {ANG_V} * PI)
B - Go forward (linear velocity: {LIN_V}, angular velocity: 0)
C - Go right (linear velocity: 0, angular velocity: -{ANG_V} * PI)
D - Stop (linear and angular velocity: 0, task completed)

You may provide a reasoning trace before you make the decision.
Please analyze:
- Is the robot close to hitting some obstacles? What choice can avoid it?
- What does the robot currently see? 
- Where might it be?
- Which part of the instruction has been REALLY completed?
- What action should the robot take NOW? (not in the future but NOW)

Note:
- Stay away from obstacles shown on the radar! Not hitting obstacles takes priority over completing instructions!
- The administrator will make sure that the robot needs to work on the first part of the instruction at the start of the conversation. Therefore, at the beginning of the conversation, you do not need to guess the progress of the instruction following because it is always initial.

The response includes a natural language analysis and the choice, split by "----".
Response example:
```
...(some analysis)
...The robot should turn left.
----
A - Go left
```
"""
}]

def get_letter(gpt_reply: str) -> str:
    try:
        result = gpt_reply.split("----")[-1].strip()
        letter: str = result[:1]
        letter = letter.upper()
        assert letter in ["A", "B", "C", "D"]
        return letter
    except:
        return None


def get_twist(letter) -> Twist:
    twist = Twist()
    if letter == "A":
        twist.linear.x = 0
        twist.angular.z = ANG_V * 3.14159
    elif letter == "B":
        twist.linear.x = LIN_V
        twist.angular.z = 0.0
    elif letter == "C":
        twist.linear.x = 0
        twist.angular.z = -ANG_V * 3.14159
    elif letter == "D":
        twist.linear.x = 0.0
        twist.angular.z = 0.0
    else:
        rospy.logwarn(f"Invalid action received from GPT: {letter}")
    return twist

def send_to_gpt_with_images(step_dir):
    client = OpenAI()

    def encode_image_base64(path):
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    view_path = os.path.join(step_dir, "view.jpg")
    radar_path = os.path.join(step_dir, "radar.jpg")

    view_b64 = encode_image_base64(view_path)
    radar_b64 = encode_image_base64(radar_path)

    with open(view_path, "rb") as f1, open(radar_path, "rb") as f2:
        response = client.chat.completions.create(
            model="gpt-4o",
            tools=[],
            tool_choice="none",
            messages=[
                *message_history,
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here are the current camera view and LiDAR radar scan."},
                        {"type": "image_url", "image_url": {"url": view_b64}},
                        {"type": "image_url", "image_url": {"url": radar_b64}},
                    ]
                }
            ],
        )
        return response.choices[0].message.content.strip()

def main_loop():
    rospy.init_node("nl_command_executor")
    cmd_vel_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)

    step = 0
    rospy.sleep(2.0)  # Wait for sensors to stabilize

    hz = 10
    rate = rospy.Rate(hz)

    twist = Twist()
    while not rospy.is_shutdown():
        cmd_vel_pub.publish(twist)
        step += 1
        rate.sleep()
        if step % (GAP_SEC * hz) != 0:
            continue

        rospy.loginfo(f"[Step {step}] Saving images...")
        step_dir = os.path.join(SAVE_ROOT, f"step_{step:03d}")
        os.makedirs(step_dir, exist_ok=True)
        save_view(step_dir)
        save_radar(step_dir)

        rospy.loginfo("Sending images to GPT-4o...")
        gpt_reply = send_to_gpt_with_images(step_dir)
        rospy.loginfo(f"GPT response: {gpt_reply}")
        with open(f"{step_dir}/reply.txt", "w") as f:
            f.write(gpt_reply)
        message_history.append({"role": "assistant", "content": gpt_reply})

        final_letter = get_letter(gpt_reply)
        twist = get_twist(final_letter)

        if final_letter == "D":
            rospy.loginfo("Task marked complete by GPT. Exiting.")
            break


if __name__ == "__main__":
    try:
        main_loop()
    except rospy.ROSInterruptException:
        pass
