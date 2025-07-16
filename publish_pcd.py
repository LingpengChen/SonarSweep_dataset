#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge
import message_filters
from std_msgs.msg import Header

# 新增导入
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf.transformations

class DepthToPointCloudNode:
    def __init__(self):
        rospy.init_node('depth_to_pointcloud_node', anonymous=True)
        
        # --- Parameters ---
        self.downsample_factor = rospy.get_param('~downsample_factor', 4)

        # ==================== NEW: Static TF Broadcaster ====================
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster()
        # A flag to ensure we only publish the static transform once.
        self.static_transform_published = False
        # ====================================================================

        # --- Subscribers using message_filters for synchronization ---
        depth_topic = "/isaacsim/camera/depth/image_raw"
        info_topic = "/isaacsim/camera/camera_info"
        
        depth_sub = message_filters.Subscriber(depth_topic, Image)
        info_sub = message_filters.Subscriber(info_topic, CameraInfo)
        
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, info_sub], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

        # --- Publisher ---
        self.cloud_pub = rospy.Publisher("/pointcloud_camera_frame", PointCloud2, queue_size=10)
        
        # --- Tools ---
        self.bridge = CvBridge()
        
        rospy.loginfo("Depth to PointCloud node started.")
        rospy.loginfo("Will publish the static transform from 'camera_link' to 'camera_link_optical' upon first message.")

    def publish_static_transform(self, parent_frame_id, child_frame_id):
        """
        Publishes the static transform from a standard camera link (body) frame
        to the camera optical frame.
        """
        t = TransformStamped()
        
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = parent_frame_id
        t.child_frame_id = child_frame_id
        
        # No translation, the origin is the same
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # The required rotation from a body frame (X-fwd, Y-left, Z-up)
        # to an optical frame (Z-fwd, X-right, Y-down).
        # This is a rotation of -90 deg around X, then -90 deg around the original Z.
        q = tf.transformations.quaternion_from_euler(-1.5707963, 0, -1.5707963, 'sxyz')
        # q = tf.transformations.quaternion_from_euler(0, 0, 0, 'sxyz')
        
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        
        self.static_broadcaster.sendTransform(t)
        rospy.loginfo("Published static transform from '%s' to '%s'", parent_frame_id, child_frame_id)

    def callback(self, depth_msg, info_msg):
        # ==================== NEW: Publish Static TF on first callback ====================
        # We do this here because we need the frame_id from the message.
        if not self.static_transform_published:
            parent_frame = info_msg.header.frame_id # e.g., "camera_link"
            optical_frame = parent_frame + "_optical"
            self.publish_static_transform(parent_frame, optical_frame)
            self.static_transform_published = True
        # ================================================================================

        try:
            # depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
            radial_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")
        except Exception as e:
            rospy.logwarn("Failed to convert depth image: %s", e)
            return

        K = np.array(info_msg.K).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        
        height, width = radial_depth_image.shape
        
        u_all, v_all = np.meshgrid(np.arange(0, width, self.downsample_factor),
                                   np.arange(0, height, self.downsample_factor))
        
        # depth_downsampled = depth_image[v_all, u_all]
        # valid = (depth_downsampled > 0) & np.isfinite(depth_downsampled)
        
        # z = depth_downsampled[valid]
        # u = u_all[valid]
        # v = v_all[valid]
        
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy

        # ==================== KEY MODIFICATION FOR RADIAL DEPTH ====================

        radial_depth_downsampled = radial_depth_image[v_all, u_all]
        
        # Filter out invalid depth values
        valid = (radial_depth_downsampled > 0) & np.isfinite(radial_depth_downsampled)
        
        r = radial_depth_downsampled[valid]
        u = u_all[valid]
        v = v_all[valid]
        
        # Pre-calculate normalized coordinates
        # This is ((u - cx) / fx) and ((v - cy) / fy)
        norm_x = (u - cx) / fx
        norm_y = (v - cy) / fy
        
        # Calculate the denominator for the Z conversion
        # This is sqrt(norm_x^2 + norm_y^2 + 1)
        denominator = np.sqrt(norm_x**2 + norm_y**2 + 1)
        
        # Step 4: Convert radial depth 'r' to planar depth 'z'
        z = r / denominator
        
        # Step 5: Calculate x and y using the newly found planar depth 'z'
        x = norm_x * z
        y = norm_y * z
        
        points_optical_frame = np.vstack((x, y, z)).T
        
        if points_optical_frame.shape[0] == 0:
            return

        header = Header()
        header.stamp = info_msg.header.stamp
        # Set the frame_id to the new optical frame.
        header.frame_id = info_msg.header.frame_id + "_optical"
        
        cloud_msg = pc2.create_cloud_xyz32(header, points_optical_frame)
        self.cloud_pub.publish(cloud_msg)

if __name__ == '__main__':
    try:
        node = DepthToPointCloudNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass