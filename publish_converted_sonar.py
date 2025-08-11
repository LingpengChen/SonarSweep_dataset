#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import threading
from queue import Queue, Empty

from utils.sonar_pad_transform import padding_sonar_image, rect_to_sonar_map
from config.hyperparam import Min_range, Range_res, Img_height, Img_width, Hori_fov


class SonarImageConverter:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('sonar_image_converter', anonymous=True)
        
        # 初始化CvBridge
        self.bridge = CvBridge()
        
        # 预计算常量，避免重复计算
        self.top_padding_pixels = int(Min_range/Range_res)
        self.azimuth_bounds = (-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2))
        
        # 使用线程队列进行异步处理
        self.processing_queue = Queue(maxsize=2)  # 限制队列大小，避免堆积
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 订阅原始声纳图像 - 减小队列大小，使用最新数据
        self.sub = rospy.Subscriber('/isaacsim/sonar_rect_image', Image, self.sonar_callback, queue_size=1)
        
        # 发布转换后的声纳图像
        self.pub_cartesian = rospy.Publisher('/sonar_cartesian_image', Image, queue_size=1)
        self.pub_padded = rospy.Publisher('/sonar_rect_padded_image', Image, queue_size=1)
        
        # 性能监控
        self.last_process_time = rospy.Time.now()
        self.process_count = 0
        
        rospy.loginfo("Sonar image converter initialized with async processing.")
        rospy.loginfo(f"Subscribing to: /isaacsim/sonar_rect_image")
        rospy.loginfo(f"Publishing to: /sonar_cartesian_image and /sonar_rect_padded_image")
        
    def sonar_callback(self, msg):
        """轻量级回调函数，快速将数据放入处理队列"""
        try:
            # 如果队列满了，丢弃旧数据，确保处理最新的图像
            if self.processing_queue.full():
                try:
                    self.processing_queue.get_nowait()  # 移除旧数据
                except Empty:
                    pass
            
            # 将消息放入处理队列
            self.processing_queue.put_nowait(msg)
            
        except Exception as e:
            rospy.logwarn(f"Error in callback: {e}")

    def processing_worker(self):
        """异步处理工作线程"""
        while not rospy.is_shutdown():
            try:
                # 从队列获取消息，超时1秒
                msg = self.processing_queue.get(timeout=1.0)
                self.process_sonar_image(msg)
                self.processing_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Error in processing worker: {e}")

    def process_sonar_image(self, msg):
        """实际的图像处理函数"""
        start_time = rospy.Time.now()
        
        try:
            # 将ROS图像消息转换为OpenCV图像
            sonar_rect_ori = self.bridge.imgmsg_to_cv2(msg, "mono8")
            
            # 对原始声纳图像进行padding（使用预计算的值）
            sonar_rect_ori_padded = padding_sonar_image(
                sonar_rect_ori, 
                top_padding_pixels=self.top_padding_pixels
            )
            
            # 转换为笛卡尔坐标系的声纳图（使用预计算的值）
            sonar_ori = rect_to_sonar_map(
                sonar_rect_ori_padded, 
                Img_height, 
                Img_width, 
                azimuth_bounds=self.azimuth_bounds
            )
            
            # 将处理后的图像转换回ROS消息格式并发布
            self.publish_images(sonar_ori, sonar_rect_ori_padded, msg.header)
            
            # 性能监控
            self.monitor_performance(start_time)
                
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in processing: {e}")

    def publish_images(self, sonar_ori, sonar_rect_ori_padded, header):
        """发布图像消息"""
        try:
            # 发布笛卡尔坐标系的声纳图像
            # sonar_ori_rotated = cv2.rotate(sonar_ori, cv2.ROTATE_180)
            sonar_ori_flipped = cv2.flip(sonar_ori, 0)  # 0表示垂直翻转

            cartesian_msg = self.bridge.cv2_to_imgmsg(sonar_ori_flipped, "mono8")
            cartesian_msg.header = header
            self.pub_cartesian.publish(cartesian_msg)
            
            # 发布padding后的矩形声纳图像
            padded_msg = self.bridge.cv2_to_imgmsg(sonar_rect_ori_padded, "mono8")
            padded_msg.header = header
            self.pub_padded.publish(padded_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error when publishing: {e}")

    def monitor_performance(self, start_time):
        """性能监控"""
        process_duration = (rospy.Time.now() - start_time).to_sec()
        self.process_count += 1
        
        if self.process_count % 30 == 0:  # 每30帧打印一次性能信息
            rospy.loginfo(f"Processing time: {process_duration*1000:.1f}ms, Frame count: {self.process_count}")


def main():
    try:
        # 设置ROS参数优化性能
        rospy.set_param('/tcp_nodelay', True)
        
        # 创建声纳图像转换器
        converter = SonarImageConverter()
        
        # 保持节点运行
        rospy.loginfo("Sonar image converter is running with async processing. Press Ctrl+C to exit.")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Sonar image converter shutting down.")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")


if __name__ == '__main__':
    main()