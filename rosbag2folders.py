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

# TF相关
import tf2_ros
import tf_conversions
from tf2_msgs.msg import TFMessage


from denoise.sonar_denoise import *
from utils.sonar_pad_transform import *
from config.hyperparam import *


# 5. 使用相似变换公式，计算在相机坐标系(C)下的等效变换 T_C
#    Robot body to camera
def T_body2cam(T_B):
    R_C_from_B = np.array([
        [ 0., -1.,  0.],
        [ 0.,  0., -1.],
        [ 1.,  0.,  0.]
    ])

    # 2. 构建4x4的齐次变换矩阵 M_C_from_B
    #    假设两个坐标系原点重合，所以没有平移部分
    M_C_from_B = np.identity(4)
    M_C_from_B[:3, :3] = R_C_from_B

    # 3. 计算其逆矩阵 M_B_from_C
    #    对于纯旋转矩阵，其逆矩阵等于其转置矩阵
    M_B_from_C = np.linalg.inv(M_C_from_B)
    T_C = M_C_from_B @ T_B @ M_B_from_C
    
    return T_C

# --- 辅助函数 ---

def msg_to_se3(msg):
    """
    将geometry_msgs/TransformStamped或PoseStamped转换为4x4 SE(3)矩阵
    """
    # 如果是TransformStamped
    if hasattr(msg, 'transform'):
        transform = msg.transform
    # 如果是PoseStamped
    elif hasattr(msg, 'pose'):
        transform = msg.pose # Pose和Transform有相同的translation和rotation字段
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
    将深度图（通常是float类型）转换为可用于保存的可视化彩色图
    """
    # 裁剪到最大深度
    depth_in_metres = np.nan_to_num(depth_img, nan=max_depth)
    depth_in_metres[depth_in_metres > max_depth] = max_depth
    
    # 归一化到0-255
    normalized_depth = cv2.normalize(depth_in_metres, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 应用伪彩色映射
    colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_JET)
    return colored_depth


def process_bag(bag_path, output_folder_base, Sonar_denoiser: SonarDenoiser, tf_buffer: tf2_ros.Buffer, start_index=0):
    """
    处理单个rosbag文件
    """
    print(f"Processing bag: {bag_path}")
    
    # 初始化CvBridge
    bridge = CvBridge()
    
    tf_buffer.clear()
    with rosbag.Bag(bag_path, 'r') as bag:
        # 先读取静态TF
        for topic, msg, t in bag.read_messages(topics=[TOPIC_TF_STATIC]):
            for transform in msg.transforms:
                tf_buffer.set_transform_static(transform, "default_authority")
        # 再读取动态TF
        for topic, msg, t in bag.read_messages(topics=[TOPIC_TF]):
            for transform in msg.transforms:
                # rospy.Time.now() 在离线处理时可能为0，使用消息时间戳
                tf_buffer.set_transform(transform, "default_authority")
    
    # 2. 查找并计算静态变换 T_camright2sonar
    try:
        # 对于静态变换，时间戳无关紧要，使用rospy.Time(0)
        T_cam_sonar_msg = tf_buffer.lookup_transform(
            RIGHT_CAM_FRAME, # Target Frame
            SONAR_FRAME,   # Source Frame
            rospy.Time(0)
        )
        T_camright2sonar = msg_to_se3(T_cam_sonar_msg)
        T_camright2sonar = T_body2cam(T_camright2sonar)
        
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
        print(f"  [Error] Could not find static transform from {SONAR_FRAME} to {RIGHT_CAM_FRAME}. Skipping bag. Error: {e}")
        return start_index

    # 3. 读取所有需要的消息到列表中
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

        # rostopic echo -b 1.bag -p /tf | tail -n 1
        # Bag Time  0  Header Time
        # 1752220909682654648,0,101133338928,world,imu_link,6.860143661499023,6.110368728637695,1.6384718894958497,-0.014497057534754276,0.012607721611857414,0.7544293403625488,0.6561000347137451
    
    with rosbag.Bag(bag_path, 'r') as bag:
        for topic, msg, t in bag.read_messages(topics=topics_to_read):
            # --- FIX: 使用 msg.header.stamp 而不是 t ---
            # 所有Image和CameraInfo消息都有header字段
            if hasattr(msg, 'header'):
                header_stamp = msg.header.stamp
                messages[key_map[topic]].append((header_stamp, msg))
            else:
                # 如果某个消息没有header，这是一个问题，需要警告
                print(f"  [Warning] Message on topic {topic} does not have a header. Skipping.")

    # 按时间排序以确保
    for key in messages:
        messages[key].sort(key=lambda x: x[0])
    
    # 提取时间戳数组（现在是正确的Header Time）
    timestamps = {key: np.array([t.to_sec() for t, msg in messages[key]]) for key in messages}
    

    # 4. 主循环：以右相机RGB为基准，查找同步的数据帧
    
    current_index = start_index
    for t_ref, msg_cam_right_rgb in tqdm(messages['cam_right_rgb'], desc="  Frames", leave=False):
        
        # 查找最近邻的消息
        def find_nearest(key, t_ref_sec):
            idx = np.searchsorted(timestamps[key], t_ref_sec, side="left")
            if idx > 0 and (idx == len(timestamps[key]) or \
                abs(t_ref_sec - timestamps[key][idx-1]) < abs(t_ref_sec - timestamps[key][idx])):
                idx = idx - 1
            if abs(t_ref_sec - timestamps[key][idx]) > TIME_SLOP:
                return None # 超过容差，同步失败
            return messages[key][idx][1]

        t_ref_sec = t_ref.to_sec()
        
        msg_cam_right_depth = find_nearest('cam_right_depth', t_ref_sec)
        msg_cam_info = find_nearest('cam_info', t_ref_sec)
        msg_cam_left_rgb = find_nearest('cam_left_rgb', t_ref_sec)
        msg_cam_left_depth = find_nearest('cam_left_depth', t_ref_sec)
        msg_sonar = find_nearest('sonar', t_ref_sec)

        # 检查是否所有消息都找到了
        if not all([msg_cam_right_depth, msg_cam_info, msg_cam_left_rgb, msg_cam_left_depth, msg_sonar]):
            # print(f"  [Warn] Skipping frame at time {t_ref_sec:.3f}, couldn't find synchronized messages.")
            continue
        
        # 5. 查找相机位姿
        try:
            cam_right_pose_msg = tf_buffer.lookup_transform(
                WORLD_FRAME,
                RIGHT_CAM_FRAME,
                t_ref # 使用参考时间戳
            )
            cam_right_pose = msg_to_se3(cam_right_pose_msg)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(f"  [Warn] Skipping frame at time {t_ref_sec:.3f}, could not get camera pose. Error: {e}")
            continue

        # 6. 创建输出目录
        output_folder = output_folder_base +str(f"_{current_index}")
        os.makedirs(output_folder, exist_ok=True)
        
        # 7. 转换并保存数据
        try:
            # 相机内参
            cam_intrinsic = np.array(msg_cam_info.K).reshape(3, 3)
            np.savetxt(os.path.join(output_folder, 'cam_intrinsic.txt'), cam_intrinsic, fmt='%.6f')
            
            # 声纳内参
            # save_sonar_intrinsics(os.path.join(output_folder, 'sonar_intrinsic.txt'))
            try:
                sonar_intrinsic_path = os.path.join(output_folder, 'sonar_intrinsic.txt')
                with open(sonar_intrinsic_path, 'w', encoding='utf-8') as f:
                    f.write(SONAR_INTRINSIC_CONTENT)
            except Exception as e:
                print(f"\n[Error] Failed to write sonar_intrinsic.txt in {sonar_intrinsic_path}: {e}")
                
            # 位姿
            np.savetxt(os.path.join(output_folder, 'cam_right_pose.txt'), cam_right_pose, fmt='%.6f')
            np.savetxt(os.path.join(output_folder, 'T_camright2sonar.txt'), T_camright2sonar, fmt='%.6f')
            
            # 图像转换
            cv_cam_right_rgb = bridge.imgmsg_to_cv2(msg_cam_right_rgb, "bgr8")
            cv_cam_right_depth = bridge.imgmsg_to_cv2(msg_cam_right_depth, "32FC1") # IsaacSim通常是32FC1
            cv_cam_left_rgb = bridge.imgmsg_to_cv2(msg_cam_left_rgb, "bgr8")
            cv_cam_left_depth = bridge.imgmsg_to_cv2(msg_cam_left_depth, "32FC1")
            

            
            # 保存图像
            cv2.imwrite(os.path.join(output_folder, 'cam_right.png'), cv_cam_right_rgb)
            np.save(os.path.join(output_folder, 'depth_right.npy'), cv_cam_right_depth)
            cv2.imwrite(os.path.join(output_folder, 'depth_right_visualize.png'), visualize_depth(cv_cam_right_depth))
            
            cv2.imwrite(os.path.join(output_folder, 'cam_left.png'), cv_cam_left_rgb)
            np.save(os.path.join(output_folder, 'depth_left.npy'), cv_cam_left_depth)
            cv2.imwrite(os.path.join(output_folder, 'depth_left_visualize.png'), visualize_depth(cv_cam_left_depth))

            # 保存声纳图像
            # 你的要求中有sonar.png和sonar_rect.png，但只有一个topic。这里都保存相同的内容。
            sonar_rect_ori = bridge.imgmsg_to_cv2(msg_sonar, "mono8") # 声纳图通常是灰度图
            sonar_rect_ori_padded = padding_sonar_image(sonar_rect_ori, top_padding_pixels=int(Min_range/Range_res))
            sonar_ori = rect_to_sonar_map(sonar_rect_ori_padded, Img_height, Img_width, azimuth_bounds=(-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2)))
            
            sonar_rect_denoised = Sonar_denoiser.process(sonar_rect_ori)
            sonar_rect_denoised_padded = padding_sonar_image(sonar_rect_denoised, top_padding_pixels=int(Min_range/Range_res))
            sonar_denoised = rect_to_sonar_map(sonar_rect_denoised_padded, Img_height, Img_width, azimuth_bounds=(-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2)))
            
            cv2.imwrite(os.path.join(output_folder, 'sonar.png'), sonar_ori)
            cv2.imwrite(os.path.join(output_folder, 'sonar_rect.png'), sonar_rect_ori_padded)
            
            # 去噪并保存
            # 使用中值滤波对声纳图像常见的椒盐噪声有较好效果
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

def main():
    try:
        import rospy
        rospy.set_param('/use_sim_time', True)
    except Exception as e:
        print(f"Could not set /use_sim_time: {e}")
        
    parser = argparse.ArgumentParser(description="Process ROS bags to extract synchronized sensor data.")
    parser.add_argument('--input_dir', type=str, default='raw_dataset/vfov12hfov60', help='Path to the raw dataset directory.')
    parser.add_argument('--output_dir', type=str, default='processed_dataset/vfov12hfov60', help='Path to the output directory.')
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        sys.exit(1)
        
    os.makedirs(args.output_dir, exist_ok=True)

    # tf_buffer 是一个在整个程序运行期间都存在的对象
    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(3600.0)) 
    
    # 查找所有场景文件夹
    scenario_folders = sorted([d for d in os.listdir(args.input_dir) if os.path.isdir(os.path.join(args.input_dir, d))])
    
    for scenario_name in tqdm(scenario_folders, desc="Scenarios"):
        
        scenario_path = os.path.join(args.input_dir, scenario_name)
        
        # get background sonar image for denoising    
        try:
            BACKGROUND_DATA_DIR = scenario_path + '/background/'
            IMAGE_FORMAT = 'png' # 背景图的格式

            # MODIFIED: 计算平均背景图，现在从图像文件读取
            avg_background_np_float = compute_average_background(BACKGROUND_DATA_DIR, img_format=IMAGE_FORMAT)
            avg_background_np = np.clip(avg_background_np_float, 0, 255).astype(np.uint8)

            # cv2.imshow('avg_background_np', avg_background_np)
            MODEL_PATH = './denoise/model/scunet_gray_25.pth'
            sonar_denoiser = SonarDenoiser(MODEL_PATH, avg_background_np)
            
        
        except Exception as e:
            print(f"Cannot find background data at {BACKGROUND_DATA_DIR}: Detailed error {e}")
        
        
        # now look into each bag in the scenerio
        bag_files = sorted(
            file_path for file_path in glob.glob(os.path.join(scenario_path, '*.bag'))
            if os.path.basename(file_path) != 'background.bag'
        )
        print(bag_files)
        # ['raw_dataset/green_water1/1.bag', 'raw_dataset/green_water1/10.bag', 'raw_dataset/green_water1/2.bag', 'raw_dataset/green_water1/3.bag', 'raw_dataset/green_water1/4.bag', 'raw_dataset/green_water1/5.bag', 'raw_dataset/green_water1/6.bag', 'raw_dataset/green_water1/7.bag', 'raw_dataset/green_water1/8.bag', 'raw_dataset/green_water1/9.bag', 'raw_dataset/green_water1/circular1.bag', 'raw_dataset/green_water1/circular2.bag', 'raw_dataset/green_water1/circular3.bag']
        if not bag_files:
            print(f"Warning: No .bag files found in {scenario_path}")
            continue
        
        for bag_file in bag_files:
            # 从完整路径中提取不带 .bag 后缀的文件名
            # os.path.basename('path/to/my_bag.bag') -> 'my_bag.bag'
            # os.path.splitext('my_bag.bag') -> ('my_bag', '.bag')
            bag_name_with_ext = os.path.basename(bag_file)
            bag_name_without_ext = os.path.splitext(bag_name_with_ext)[0]
            
            # 调用更新后的函数，传入新的 bag_name
            # 注意：process_bag现在返回的是这个bag处理的帧数，我们不再需要用它来累加
            output_folder_base = os.path.join(args.output_dir, f"{scenario_name}_{bag_name_without_ext}" )
            process_bag(bag_file, output_folder_base, sonar_denoiser, tf_buffer)

    print("\nProcessing complete!")

if __name__ == '__main__':
    main()