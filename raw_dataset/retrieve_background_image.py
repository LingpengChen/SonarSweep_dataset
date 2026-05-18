#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

IMAGE_TOPIC = '/isaacsim/sonar_rect_image'
base_dir = "./vfov12hfov60"



BAG_FILE = base_dir+'/background.bag'
OUTPUT_DIR = base_dir+'/background'


def extract_images():
    """
    Extract sonar images from the configured bag file and save them as PNGs.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created directory: {OUTPUT_DIR}")

    bridge = CvBridge()

    count_rect = 0

    print(f"Processing ROS bag: {BAG_FILE}")
    print(f"Extracting images from topic '{IMAGE_TOPIC}'...")

    with rosbag.Bag(BAG_FILE, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=IMAGE_TOPIC):
            try:
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
            except CvBridgeError as e:
                print(f"Failed to convert image: {e}")
                continue

            timestamp = t.to_sec()
            image_name = f"{timestamp}.png"
            image_path = os.path.join(OUTPUT_DIR, image_name)

            if cv2.imwrite(image_path, cv_image):
                count_rect += 1
            else:
                print(f"Failed to save image: {image_path}")

    print("-" * 30)
    print("Processing complete.")
    print(f"Saved {count_rect} images from '{IMAGE_TOPIC}' to '{OUTPUT_DIR}'.")

if __name__ == '__main__':
    extract_images()
