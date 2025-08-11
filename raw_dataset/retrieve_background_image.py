#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# --- 配置参数 ---
base_dir = "./vfov20hfov130/blue_water_visual_degraded"
# ROS bag 文件路径
# 要提取的图像话题
IMAGE_TOPIC = '/isaacsim/sonar_rect_image'
BAG_FILE = base_dir+'/background.bag'
OUTPUT_DIR = base_dir+'/background'




def extract_images():
    """
    从指定的 bag 文件中提取图像并保存为 PNG 文件。
    """
    # 检查输出目录是否存在，如果不存在则创建
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建文件夹: {OUTPUT_DIR}")

    # 初始化 cv_bridge
    bridge = CvBridge()
    
    # 记录提取的图像数量
    count = 0

    print(f"开始处理 ROS bag 文件: {BAG_FILE}")
    print(f"从话题 '{IMAGE_TOPIC}' 中提取图像...")

    # 使用 'with' 语句安全地打开 bag 文件
    with rosbag.Bag(BAG_FILE, 'r') as bag:
        # 遍历 bag 文件中指定话题的消息
        # read_messages 会返回一个生成器 (topic, msg, t)
        # topic: 消息话题名 (str)
        # msg: 消息本身
        # t: 消息的时间戳 (rospy.Time)
        for topic, msg, t in bag.read_messages(topics=[IMAGE_TOPIC]):
            try:
                # 将 ROS Image 消息转换为 OpenCV 图像 (NumPy ndarray)
                # "passthrough" 表示保持原始编码，对于灰度图(如 mono8, mono16)很有效
                # 如果你想确保输出是8位灰度图，可以使用 'mono8'
                cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            except CvBridgeError as e:
                print(e)
                continue

            # 使用消息的时间戳来命名文件
            # t.to_sec() 将时间戳转换为浮点数秒
            timestamp = t.to_sec()
            image_name = f"{timestamp}.png"
            
            # 构造完整的输出路径
            output_path = os.path.join(OUTPUT_DIR, image_name)

            # 保存图像
            cv2.imwrite(output_path, cv_image)
            
            count += 1

    print("-" * 30)
    print("处理完成！")
    print(f"总共从 bag 文件中提取并保存了 {count} 张图像到 '{OUTPUT_DIR}' 文件夹。")

if __name__ == '__main__':
    extract_images()