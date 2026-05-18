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
        rospy.init_node('sonar_image_converter', anonymous=True)
        
        self.bridge = CvBridge()
        
        # Precompute constants used for every frame.
        self.top_padding_pixels = int(Min_range/Range_res)
        self.azimuth_bounds = (-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2))
        
        # Use a bounded worker queue to avoid processing stale frames.
        self.processing_queue = Queue(maxsize=2)
        self.processing_thread = threading.Thread(target=self.processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.sub = rospy.Subscriber('/isaacsim/sonar_rect_image', Image, self.sonar_callback, queue_size=1)
        
        self.pub_cartesian = rospy.Publisher('/sonar_cartesian_image', Image, queue_size=1)
        self.pub_padded = rospy.Publisher('/sonar_rect_padded_image', Image, queue_size=1)
        
        self.last_process_time = rospy.Time.now()
        self.process_count = 0
        
        rospy.loginfo("Sonar image converter initialized with async processing.")
        rospy.loginfo(f"Subscribing to: /isaacsim/sonar_rect_image")
        rospy.loginfo(f"Publishing to: /sonar_cartesian_image and /sonar_rect_padded_image")
        
    def sonar_callback(self, msg):
        """Queue the latest sonar image for asynchronous processing."""
        try:
            if self.processing_queue.full():
                try:
                    self.processing_queue.get_nowait()
                except Empty:
                    pass
            
            self.processing_queue.put_nowait(msg)
            
        except Exception as e:
            rospy.logwarn(f"Error in callback: {e}")

    def processing_worker(self):
        """Process queued sonar images in a background thread."""
        while not rospy.is_shutdown():
            try:
                msg = self.processing_queue.get(timeout=1.0)
                self.process_sonar_image(msg)
                self.processing_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                rospy.logerr(f"Error in processing worker: {e}")

    def process_sonar_image(self, msg):
        """Convert one rectangular sonar image and publish the outputs."""
        start_time = rospy.Time.now()
        
        try:
            sonar_rect_ori = self.bridge.imgmsg_to_cv2(msg, "mono8")
            
            sonar_rect_ori_padded = padding_sonar_image(
                sonar_rect_ori, 
                top_padding_pixels=self.top_padding_pixels
            )
            
            sonar_ori = rect_to_sonar_map(
                sonar_rect_ori_padded, 
                Img_height, 
                Img_width, 
                azimuth_bounds=self.azimuth_bounds
            )
            
            self.publish_images(sonar_ori, sonar_rect_ori_padded, msg.header)
            self.monitor_performance(start_time)
                
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
        except Exception as e:
            rospy.logerr(f"Unexpected error in processing: {e}")

    def publish_images(self, sonar_ori, sonar_rect_ori_padded, header):
        """Publish Cartesian and padded rectangular sonar images."""
        try:
            # sonar_ori_rotated = cv2.rotate(sonar_ori, cv2.ROTATE_180)
            sonar_ori_flipped = cv2.flip(sonar_ori, 0)

            cartesian_msg = self.bridge.cv2_to_imgmsg(sonar_ori_flipped, "mono8")
            cartesian_msg.header = header
            self.pub_cartesian.publish(cartesian_msg)
            
            padded_msg = self.bridge.cv2_to_imgmsg(sonar_rect_ori_padded, "mono8")
            padded_msg.header = header
            self.pub_padded.publish(padded_msg)
            
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error when publishing: {e}")

    def monitor_performance(self, start_time):
        """Log processing time every 30 frames."""
        process_duration = (rospy.Time.now() - start_time).to_sec()
        self.process_count += 1
        
        if self.process_count % 30 == 0:
            rospy.loginfo(f"Processing time: {process_duration*1000:.1f}ms, Frame count: {self.process_count}")


def main():
    try:
        rospy.set_param('/tcp_nodelay', True)
        
        converter = SonarImageConverter()
        
        rospy.loginfo("Sonar image converter is running with async processing. Press Ctrl+C to exit.")
        rospy.spin()
        
    except rospy.ROSInterruptException:
        rospy.loginfo("Sonar image converter shutting down.")
    except Exception as e:
        rospy.logerr(f"Error in main: {e}")


if __name__ == '__main__':
    main()
