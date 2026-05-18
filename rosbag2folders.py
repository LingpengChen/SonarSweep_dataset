#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import glob
from tqdm import tqdm

import rosbag
import rospy
import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError

import tf2_ros
import tf_conversions


from denoise.sonar_denoise import SonarDenoiser, compute_average_background
from utils.sonar_pad_transform import padding_sonar_image, rect_to_sonar_map
from config.hyperparam import (
    Hori_fov,
    Img_height,
    Img_width,
    Min_range,
    Range_res,
    RIGHT_CAM_FRAME,
    SONAR_FRAME,
    SONAR_INTRINSIC_CONTENT,
    TIME_SLOP,
    TOPIC_CAM_LEFT_DEPTH,
    TOPIC_CAM_LEFT_RGB,
    TOPIC_CAM_RIGHT_DEPTH,
    TOPIC_CAM_RIGHT_INFO,
    TOPIC_CAM_RIGHT_RGB,
    TOPIC_SONAR,
    TOPIC_TF,
    TOPIC_TF_STATIC,
    WORLD_FRAME,
)


def t_body_to_cam(T_B):
    """Convert a body-frame transform into the camera coordinate convention."""
    R_C_from_B = np.array([
        [ 0., -1.,  0.],
        [ 0.,  0., -1.],
        [ 1.,  0.,  0.]
    ])

    # The frames share the same origin, so this is a pure rotation change.
    M_C_from_B = np.identity(4)
    M_C_from_B[:3, :3] = R_C_from_B

    M_B_from_C = np.linalg.inv(M_C_from_B)
    T_C = M_C_from_B @ T_B @ M_B_from_C
    
    return T_C


def msg_to_se3(msg):
    """
    Convert geometry_msgs/TransformStamped or PoseStamped into a 4x4 SE(3) matrix.
    """
    if hasattr(msg, 'transform'):
        transform = msg.transform
    elif hasattr(msg, 'pose'):
        transform = msg.pose
    else:
        raise TypeError("Input message type not supported")

    translation = [transform.translation.x, transform.translation.y, transform.translation.z]
    rotation = [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
    return tf_conversions.transformations.concatenate_matrices(
        tf_conversions.transformations.translation_matrix(translation),
        tf_conversions.transformations.quaternion_matrix(rotation)
    )

def visualize_depth(depth_img, max_depth=10.0):
    """
    Convert a float depth image into a color visualization for inspection.
    """
    depth_in_metres = np.nan_to_num(depth_img, nan=max_depth)
    depth_in_metres[depth_in_metres > max_depth] = max_depth
    
    normalized_depth = cv2.normalize(depth_in_metres, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    return colored_depth


def process_bag(bag_path, output_folder_base, sonar_denoiser: SonarDenoiser, tf_buffer: tf2_ros.Buffer, start_index=0):
    """
    Process one ROS bag into synchronized datapoint folders.
    """
    print(f"Processing bag: {bag_path}")
    
    bridge = CvBridge()
    
    tf_buffer.clear()
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=[TOPIC_TF_STATIC]):
            for transform in msg.transforms:
                tf_buffer.set_transform_static(transform, "default_authority")
        for topic, msg, t in bag.read_messages(topics=[TOPIC_TF]):
            for transform in msg.transforms:
                tf_buffer.set_transform(transform, "default_authority")
    
    try:
        T_cam_sonar_msg = tf_buffer.lookup_transform(
            RIGHT_CAM_FRAME, # Target Frame
            SONAR_FRAME,   # Source Frame
            rospy.Time(0)
        )
        T_camright2sonar = msg_to_se3(T_cam_sonar_msg)
        T_camright2sonar = t_body_to_cam(T_camright2sonar)
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print(f"  [Error] Could not find static transform from {SONAR_FRAME} to {RIGHT_CAM_FRAME}. Skipping bag. Error: {e}")
        return start_index

    messages = {
        'cam_right_rgb': [],
        'cam_right_depth': [],
        'cam_info': [],
        'cam_left_rgb': [],
        'cam_left_depth': [],
        'sonar': []
    }
    
    topics_to_read = [
        TOPIC_CAM_RIGHT_RGB, TOPIC_CAM_RIGHT_DEPTH, TOPIC_CAM_RIGHT_INFO,
        TOPIC_CAM_LEFT_RGB, TOPIC_CAM_LEFT_DEPTH, TOPIC_SONAR
    ]

    key_map = {
        TOPIC_CAM_RIGHT_RGB: 'cam_right_rgb',
        TOPIC_CAM_RIGHT_DEPTH: 'cam_right_depth',
        TOPIC_CAM_RIGHT_INFO: 'cam_info',
        TOPIC_CAM_LEFT_RGB: 'cam_left_rgb',
        TOPIC_CAM_LEFT_DEPTH: 'cam_left_depth',
        TOPIC_SONAR: 'sonar'
    }

    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=topics_to_read):
            # Use the message header stamp for synchronization rather than bag time.
            if hasattr(msg, 'header'):
                header_stamp = msg.header.stamp
                messages[key_map[topic]].append((header_stamp, msg))
            else:
                print(f"  [Warning] Message on topic {topic} does not have a header. Skipping.")

    for key in messages:
        messages[key].sort(key=lambda x: x[0])
    
    timestamps = {key: np.array([t.to_sec() for t, msg in messages[key]]) for key in messages}
    
    current_index = start_index
    for t_ref, msg_cam_right_rgb in tqdm(messages['cam_right_rgb'], desc="  Frames", leave=False):
        
        def find_nearest(key, t_ref_sec):
            if len(timestamps[key]) == 0:
                return None
            idx = np.searchsorted(timestamps[key], t_ref_sec, side="left")
            if idx == len(timestamps[key]):
                idx -= 1
            if idx > 0 and (idx == len(timestamps[key]) or \
                abs(t_ref_sec - timestamps[key][idx-1]) < abs(t_ref_sec - timestamps[key][idx])):
                idx = idx - 1
            if abs(t_ref_sec - timestamps[key][idx]) > TIME_SLOP:
                return None
            return messages[key][idx][1]

        t_ref_sec = t_ref.to_sec()
        
        msg_cam_right_depth = find_nearest('cam_right_depth', t_ref_sec)
        msg_cam_info = find_nearest('cam_info', t_ref_sec)
        msg_cam_left_rgb = find_nearest('cam_left_rgb', t_ref_sec)
        msg_cam_left_depth = find_nearest('cam_left_depth', t_ref_sec)
        msg_sonar = find_nearest('sonar', t_ref_sec)

        if not all([msg_cam_right_depth, msg_cam_info, msg_cam_left_rgb, msg_cam_left_depth, msg_sonar]):
            # print(f"  [Warn] Skipping frame at time {t_ref_sec:.3f}, couldn't find synchronized messages.")
            continue
        
        try:
            cam_right_pose_msg = tf_buffer.lookup_transform(
                WORLD_FRAME,
                RIGHT_CAM_FRAME,
                t_ref
            )
            cam_right_pose = msg_to_se3(cam_right_pose_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(f"  [Warn] Skipping frame at time {t_ref_sec:.3f}, could not get camera pose. Error: {e}")
            continue

        output_folder = f"{output_folder_base}_{current_index}"
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            cam_intrinsic = np.array(msg_cam_info.K).reshape(3, 3)
            np.savetxt(os.path.join(output_folder, 'cam_intrinsic.txt'), cam_intrinsic, fmt='%.6f')
            
            try:
                sonar_intrinsic_path = os.path.join(output_folder, 'sonar_intrinsic.txt')
                with open(sonar_intrinsic_path, 'w', encoding='utf-8') as f:
                    f.write(SONAR_INTRINSIC_CONTENT)
            except Exception as e:
                print(f"\n[Error] Failed to write sonar_intrinsic.txt in {sonar_intrinsic_path}: {e}")
                
            np.savetxt(os.path.join(output_folder, 'cam_right_pose.txt'), cam_right_pose, fmt='%.6f')
            np.savetxt(os.path.join(output_folder, 'T_camright2sonar.txt'), T_camright2sonar, fmt='%.6f')
            
            cv_cam_right_rgb = bridge.imgmsg_to_cv2(msg_cam_right_rgb, "bgr8")
            cv_cam_right_depth = bridge.imgmsg_to_cv2(msg_cam_right_depth, "32FC1")
            cv_cam_left_rgb = bridge.imgmsg_to_cv2(msg_cam_left_rgb, "bgr8")
            cv_cam_left_depth = bridge.imgmsg_to_cv2(msg_cam_left_depth, "32FC1")

            cv2.imwrite(os.path.join(output_folder, 'cam_right.png'), cv_cam_right_rgb)
            np.save(os.path.join(output_folder, 'depth_right.npy'), cv_cam_right_depth)
            cv2.imwrite(os.path.join(output_folder, 'depth_right_visualize.png'), visualize_depth(cv_cam_right_depth))
            
            cv2.imwrite(os.path.join(output_folder, 'cam_left.png'), cv_cam_left_rgb)
            np.save(os.path.join(output_folder, 'depth_left.npy'), cv_cam_left_depth)
            cv2.imwrite(os.path.join(output_folder, 'depth_left_visualize.png'), visualize_depth(cv_cam_left_depth))

            sonar_rect_ori = bridge.imgmsg_to_cv2(msg_sonar, "mono8")
            sonar_rect_ori_padded = padding_sonar_image(sonar_rect_ori, top_padding_pixels=int(Min_range/Range_res))
            sonar_ori = rect_to_sonar_map(sonar_rect_ori_padded, Img_height, Img_width, azimuth_bounds=(-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2)))
            
            sonar_rect_denoised = sonar_denoiser.process(sonar_rect_ori)
            sonar_rect_denoised_padded = padding_sonar_image(sonar_rect_denoised, top_padding_pixels=int(Min_range/Range_res))
            sonar_denoised = rect_to_sonar_map(sonar_rect_denoised_padded, Img_height, Img_width, azimuth_bounds=(-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2)))
            
            cv2.imwrite(os.path.join(output_folder, 'sonar.png'), sonar_ori)
            cv2.imwrite(os.path.join(output_folder, 'sonar_rect.png'), sonar_rect_ori_padded)
            
            cv2.imwrite(os.path.join(output_folder, 'sonar_denoise.png'), sonar_denoised)
            cv2.imwrite(os.path.join(output_folder, 'sonar_rect_denoise.png'), sonar_rect_denoised_padded)

        except CvBridgeError as e:
            print(f"  [Error] CV Bridge Error: {e}. Skipping frame.")
            continue
        except Exception as e:
            print(f"  [Error] An unexpected error occurred during saving: {e}. Skipping frame.")
            continue
        
        current_index += 1

    return current_index


def write_datapoint_list(output_dir, bag_name):
    """
    Save all datapoint folder names for one bag into ../{bag_name}.txt.
    """
    output_dir = os.path.abspath(output_dir)
    list_dir = os.path.dirname(output_dir)
    prefix = f"{bag_name}_"

    def sort_key(folder_name):
        suffix = folder_name[len(prefix):]
        return int(suffix) if suffix.isdigit() else suffix

    datapoint_names = sorted(
        name for name in os.listdir(output_dir)
        if name.startswith(prefix) and os.path.isdir(os.path.join(output_dir, name))
    )
    datapoint_names.sort(key=sort_key)

    list_path = os.path.join(list_dir, f"{bag_name}.txt")
    with open(list_path, 'w', encoding='utf-8') as f:
        for name in datapoint_names:
            f.write(f"{name}\n")

    print(f"Saved {len(datapoint_names)} datapoint names to '{list_path}'")


def main():
    try:
        import rospy
        rospy.set_param('/use_sim_time', True)
    except Exception as e:
        print(f"Could not set /use_sim_time: {e}")
    
    parser = argparse.ArgumentParser(description="Process ROS bags to extract synchronized sensor data.")
    parser.add_argument('--sonar_type', type=str, default='vfov12hfov60', help='Sonar type folder name.')
    parser.add_argument('--input_dir', type=str, default=None, help='Path to the raw dataset directory.')
    parser.add_argument('--output_dir', type=str, default=None, help='Path to the output directory.')
    args = parser.parse_args()
    args.input_dir = args.input_dir or os.path.join('raw_dataset', args.sonar_type)
    args.output_dir = args.output_dir or os.path.join('processed_dataset', args.sonar_type)

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3600.0)) 
    
    background_cache_path = f'./denoise/avg_background_{args.sonar_type}.png'
    try:
        background_dir = os.path.join(args.input_dir, 'background')
        avg_background_np_float = compute_average_background(background_dir, img_format='png')
        avg_background_np = np.clip(avg_background_np_float, 0, 255).astype(np.uint8)
        cv2.imwrite(background_cache_path, avg_background_np)
        print(f"Average background image saved to '{background_cache_path}'")
    except Exception as e:
        print(f"Cannot compute background from '{background_dir}': {e}")
        avg_background_np = cv2.imread(background_cache_path, cv2.IMREAD_GRAYSCALE)
        if avg_background_np is None:
            print(f"Cannot load default background image: {background_cache_path}")
            sys.exit(1)
        print(f"Default background image loaded from '{background_cache_path}'")

    sonar_denoiser = SonarDenoiser('./denoise/model/scunet_gray_25.pth', avg_background_np)

    bag_files = sorted(
        file_path for file_path in glob.glob(os.path.join(args.input_dir, '*.bag'))
        if os.path.basename(file_path) != 'background.bag'
    )
    if not bag_files:
        print(f"Warning: No .bag files found in {args.input_dir}")
        return

    print(f"Found {len(bag_files)} bag files in {args.input_dir}")
    for bag_file in tqdm(bag_files, desc="Bags"):
        bag_name = os.path.splitext(os.path.basename(bag_file))[0]
        output_folder_base = os.path.join(args.output_dir, bag_name)
        process_bag(bag_file, output_folder_base, sonar_denoiser, tf_buffer)
        write_datapoint_list(args.output_dir, bag_name)

    print("\nProcessing complete!")

if __name__ == '__main__':
    main()
