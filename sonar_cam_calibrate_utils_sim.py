#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
import math
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError

class ImageProcessorNode:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('image_processor_node', anonymous=True)
        
        # 初始化cv_bridge
        self.bridge = CvBridge()

        # === 相机处理部分 ===
        # OAK相机内参 K 和畸变系数 D
        self.K = np.array([
            [550.0, 0.0, 320.0],
            [0.0, 550.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        self.D = np.array([0.0, 0.0, 0.0, 0.0, 0.0]) # 畸变系数为0

        # 期望的视场角 (degrees)
        self.target_hfov_deg = 60.0
        self.target_vfov_deg = 12.0

        # 根据期望的FOV计算裁剪后图像的尺寸
        self.calculate_crop_dims()
        
        # === 话题发布者 ===
        # 声呐图像发布者
        self.sonar_rect_pub = rospy.Publisher('/isaacsim_sonar/drawn_sonar_rect/mono_flipped', Image, queue_size=10)

        # 相机图像发布者
        self.cam_pub = rospy.Publisher('/isaacsim_camera/camera/undistorted_cropped', Image, queue_size=10)

        # === 话题订阅者 ===
        # 使用lambda函数将对应的发布者传递给回调函数，避免代码重复
  
        rospy.Subscriber('/isaacsim/sonar_rect_image', Image, 
                         lambda msg: self.sonar_callback(msg, self.sonar_rect_pub))

        rospy.Subscriber('/isaacsim/camera/image_raw', Image, self.camera_callback)

        rospy.loginfo("图像处理节点已启动，正在监听话题...")

    def calculate_crop_dims(self):
        """
        根据相机内参和期望的FOV计算裁剪区域
        """
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # 将角度转换为弧度
        target_hfov_rad = math.radians(self.target_hfov_deg)
        target_vfov_rad = math.radians(self.target_vfov_deg)

        # 计算新的宽度和高度
        # 公式: f = (w/2) / tan(FOV/2)  =>  w = 2 * f * tan(FOV/2)
        new_w = int(round(2 * fx * math.tan(target_hfov_rad / 2.0)))
        new_h = int(round(2 * fy * math.tan(target_vfov_rad / 2.0)))

        # 确保尺寸是偶数，便于中心化
        self.crop_w = new_w if new_w % 2 == 0 else new_w + 1
        self.crop_h = new_h if new_h % 2 == 0 else new_h + 1
        
        # 计算裁剪区域的左上角和右下角坐标
        self.x1 = int(round(cx - self.crop_w / 2.0))
        self.y1 = int(round(cy - self.crop_h / 2.0))
        self.x2 = self.x1 + self.crop_w
        self.y2 = self.y1 + self.crop_h

        rospy.loginfo("相机裁剪尺寸计算完成: width={}, height={}".format(self.crop_w, self.crop_h))
        rospy.loginfo("裁剪区域: (x1, y1)=({}, {}), (x2, y2)=({}, {})".format(self.x1, self.y1, self.x2, self.y2))


    def sonar_callback(self, msg, publisher):
        """
        处理声呐图像的回调函数: 转灰度 -> 翻转 -> 发布
        """
        try:
            # 将ROS Image消息转换为OpenCV图像 (bgr8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # 1. 转换为灰度图
        gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # 2. 上下和左右翻转
        # cv2.flip(src, flipCode): 
        #   0: 垂直翻转 (上下)
        #   1: 水平翻转 (左右)
        #  -1: 水平和垂直翻转
        flipped_image = cv2.flip(gray_image, -1)

        try:
            # 将处理后的OpenCV图像转换回ROS Image消息
            flipped_msg = self.bridge.cv2_to_imgmsg(flipped_image, "mono8")
            
            # 3. 关键：使用原始消息的header（包含时间戳和frame_id）
            flipped_msg.header = msg.header
            
            # 发布处理后的图像
            publisher.publish(flipped_msg)
        except CvBridgeError as e:
            rospy.logerr(e)

    def camera_callback(self, msg):
        """
        处理相机图像的回调函数: 解压 -> 去畸变 -> 裁剪 -> 发布
        """
        try:
            # 1. 将ROS Image消息直接转换为OpenCV图像 (rgb8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        except CvBridgeError as e:
            rospy.logerr(e)
            return

        # 2. 去畸变
        # 因为D=[0,0]，这一步实际上不会改变图像，但代码结构是完整的
        undistorted_image = cv2.undistort(cv_image, self.K, self.D)
        
        # 获取图像尺寸以进行边界检查
        h, w, _ = undistorted_image.shape
        if self.x1 < 0 or self.y1 < 0 or self.x2 > w or self.y2 > h:
            rospy.logwarn_once("计算出的裁剪区域超出了图像边界，请检查相机内参和FOV设置！")
            # 简单处理：将超出部分裁剪掉
            x1_c, y1_c = max(0, self.x1), max(0, self.y1)
            x2_c, y2_c = min(w, self.x2), min(h, self.y2)
            cropped_image = undistorted_image[y1_c:y2_c, x1_c:x2_c]
        else:
             # 3. 以图像中心裁剪
            cropped_image = undistorted_image[self.y1:self.y2, self.x1:self.x2]

        try:
            # 将处理后的OpenCV图像转换回ROS Image消息
            cropped_msg = self.bridge.cv2_to_imgmsg(cropped_image, "rgb8")

            # 4. 关键：使用原始消息的header（包含时间戳和frame_id）
            cropped_msg.header = msg.header
            
            # 发布处理后的图像
            self.cam_pub.publish(cropped_msg)
        except CvBridgeError as e:
            rospy.logerr(e)


if __name__ == '__main__':
    try:
        ImageProcessorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass