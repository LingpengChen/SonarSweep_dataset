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



def process_bag(bag_path, output_folder_base, Sonar_denoiser: SonarDenoiser, tf_buffer: tf2_ros.Buffer, start_index=0):
    """
    处理单个rosbag文件
    """
    print(f"Processing bag: {bag_path}")
    
    # 初始化CvBridge
    bridge = CvBridge()
    
    tf_buffer.clear()

    
    # 3. 读取所有需要的消息到列表中
    messages = {
        'cam_right_rgb': [],
        'cam_left_rgb': [],
        'sonar': [],
        'sonar_rect': []
    }
    
    topics_to_read = [
        TOPIC_CAM_RIGHT_RGB,
        TOPIC_CAM_LEFT_RGB, 
        TOPIC_SONAR, TOPIC_SONAR_RECT
    ]

    key_map = {
        TOPIC_CAM_RIGHT_RGB: 'cam_right_rgb',
        TOPIC_CAM_LEFT_RGB: 'cam_left_rgb',
        TOPIC_SONAR: 'sonar',
        TOPIC_SONAR_RECT: 'sonar_rect'
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
        
        msg_cam_left_rgb = find_nearest('cam_left_rgb', t_ref_sec)
        msg_sonar = find_nearest('sonar', t_ref_sec)
        msg_sonar_rect = find_nearest('sonar_rect', t_ref_sec)

        # 检查是否所有消息都找到了
        if not all([ msg_cam_left_rgb, msg_sonar, msg_sonar_rect]):
            # print(f"  [Warn] Skipping frame at time {t_ref_sec:.3f}, couldn't find synchronized messages.")
            continue
        
        
        # 6. 创建输出目录
        output_folder = output_folder_base +str(f"_{current_index}")
        os.makedirs(output_folder, exist_ok=True)
        
        # 7. 转换并保存数据
    
        # 相机内参
        # TODO
        # cam_intrinsic = np.array(msg_cam_info.K).reshape(3, 3)
        # np.savetxt(os.path.join(output_folder, 'cam_intrinsic.txt'), cam_intrinsic, fmt='%.6f')
        
        # 声纳内参
        # save_sonar_intrinsics(os.path.join(output_folder, 'sonar_intrinsic.txt'))
        try:
            sonar_intrinsic_path = os.path.join(output_folder, 'sonar_intrinsic.txt')
            with open(sonar_intrinsic_path, 'w', encoding='utf-8') as f:
                f.write(SONAR_INTRINSIC_CONTENT)
        except Exception as e:
            print(f"\n[Error] Failed to write sonar_intrinsic.txt in {sonar_intrinsic_path}: {e}")
            
        # 位姿
        # np.savetxt(os.path.join(output_folder, 'cam_right_pose.txt'), cam_right_pose, fmt='%.6f')
        # np.savetxt(os.path.join(output_folder, 'T_camright2sonar.txt'), T_camright2sonar, fmt='%.6f')
        
        # 图像转换
        # 处理压缩图像消息
        cv_cam_right_rgb = bridge.compressed_imgmsg_to_cv2(msg_cam_right_rgb, "bgr8")
        cv_cam_left_rgb = bridge.compressed_imgmsg_to_cv2(msg_cam_left_rgb, "bgr8")
        


        # 保存图像
        cv2.imwrite(os.path.join(output_folder, 'cam_right.png'), cv_cam_right_rgb)
        
        cv2.imwrite(os.path.join(output_folder, 'cam_left.png'), cv_cam_left_rgb)

        # 保存声纳图像
        sonar_ori = bridge.imgmsg_to_cv2(msg_sonar, "mono8") # 声纳图通常是灰度图
        sonar_ori = cv2.flip(sonar_ori, 1)  
        cv2.imwrite(os.path.join(output_folder, 'sonar.png'), sonar_ori)

        sonar_rect_ori = bridge.imgmsg_to_cv2(msg_sonar_rect, "mono8") # 声纳图通常是灰度图
        sonar_rect_ori = cv2.flip(sonar_rect_ori, 0)  # 0表示上下翻转
        sonar_rect_ori = cv2.flip(sonar_rect_ori, 1)  # 1表示左右翻转
        cv2.imwrite(os.path.join(output_folder, 'sonar_rect.png'), sonar_rect_ori)
        ## Get mapped sonar image (should be the same as sonar)
        # # sonar_rect_ori_padded = padding_sonar_image(sonar_rect_ori, top_padding_pixels=int(Min_range/Range_res))
        # sonar_remapped = rect_to_sonar_map(sonar_rect_ori, Img_height, Img_width, azimuth_bounds=(-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2)))
        # sonar_remapped = cv2.flip(sonar_remapped, 0)  # 0表示上下翻转
        # cv2.imwrite(os.path.join(output_folder, 'sonar_remapped.png'), sonar_remapped)

        # TODO: may need to remove 
        if True:
            # Crop 4/5 of the height from the bottom
            h, w = sonar_rect_ori.shape
            crop_height = int(h * 4/5)
            sonar_rect_ori_cropped = sonar_rect_ori[:crop_height, :]

            # Resize to match denoised_background_tensor size
            target_size = Sonar_denoiser.denoised_background_tensor.shape
            sonar_rect_ori_resized = cv2.resize(sonar_rect_ori_cropped, (target_size[3], target_size[2]))
            sonar_rect_ori = sonar_rect_ori_resized
            # cv2.imwrite(os.path.join(output_folder, 'a_sonar_rect_ori.png'), sonar_rect_ori)
            # cv2.imwrite(os.path.join(output_folder, 'a_sonar_rect_ori_cropped.png'), sonar_rect_ori_cropped)
            # cv2.imwrite(os.path.join(output_folder, 'a_sonar_rect_ori_resized.png'), sonar_rect_ori_resized)


        sonar_rect_denoised = Sonar_denoiser.process(sonar_rect_ori)
        sonar_denoised = rect_to_sonar_map(sonar_rect_denoised, Img_height_target, Img_width, azimuth_bounds=(-np.deg2rad(Hori_fov/2), np.deg2rad(Hori_fov/2)))
        sonar_denoised = cv2.flip(sonar_denoised, 0)  # 0表示上下翻转

        # 去噪并保存
        # 使用中值滤波对声纳图像常见的椒盐噪声有较好效果
        cv2.imwrite(os.path.join(output_folder, 'sonar_rect_denoise.png'), sonar_rect_denoised)
        cv2.imwrite(os.path.join(output_folder, 'sonar_denoise.png'), sonar_denoised)

        current_index += 1

    return current_index

def main():
    try:
        import rospy
        rospy.set_param('/use_sim_time', True)
    except Exception as e:
        print(f"Could not set /use_sim_time: {e}")
    
    sonar_type = "vfov12hfov60"
    scene_name = "one_stone"  
    # logic: sonar_type -> scene_name -> trajectory (rosbag name)
    # sonar_type = "vfov20hfov130"
    parser = argparse.ArgumentParser(description="Process ROS bags to extract synchronized sensor data.")
    parser.add_argument('--input_dir', type=str, default=f'raw_dataset/{sonar_type}', help='Path to the raw dataset directory.')
    parser.add_argument('--output_dir', type=str, default=f'processed_dataset/{sonar_type}', help='Path to the output directory.')
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
        if scenario_name != scene_name: continue
        print(f"Processing scenario: {scenario_name}")
        scenario_path = os.path.join(args.input_dir, scenario_name)
        
        try:
            BACKGROUND_DATA_DIR = scenario_path + '/background/'
            IMAGE_FORMAT = 'png' # 背景图的格式

            # MODIFIED: 计算平均背景图，现在从图像文件读取
            avg_background_np_float = compute_average_background(BACKGROUND_DATA_DIR, img_format=IMAGE_FORMAT)
            avg_background_np = np.clip(avg_background_np_float, 0, 255).astype(np.uint8)
            # MODIFIED: 保存平均背景图
            cv2.imwrite(f'./denoise/avg_background_{sonar_type}.png', avg_background_np)
            print(f"Average background image computed and saved to './denoise/avg_background_{sonar_type}.png'")
            MODEL_PATH = './denoise/model/scunet_gray_25.pth'
            sonar_denoiser = SonarDenoiser(MODEL_PATH, avg_background_np)
          
        
        except Exception as e:
            try: # try to load the default background
                avg_background_np = cv2.imread(f'./denoise/avg_background_{sonar_type}.png', cv2.IMREAD_GRAYSCALE)
                if avg_background_np is None:
                    raise FileNotFoundError("Default background image not found.")
                else:
                    print(f"Default background image loaded from './denoise/avg_background_{sonar_type}.png'")
                sonar_denoiser = SonarDenoiser('./denoise/model/scunet_gray_25.pth', avg_background_np)
            except Exception as e:
                print(f"Cannot load default background image: {e}")
                sys.exit(1)
            
        break
        
        
        # now look into each bag in the scenerio
        bag_files = sorted(
            file_path for file_path in glob.glob(os.path.join(scenario_path, '*.bag'))
            if os.path.basename(file_path) != 'background.bag'
        )
        # ['raw_dataset/green_water1/1.bag', 'raw_dataset/green_water1/10.bag', 'raw_dataset/green_water1/2.bag', 'raw_dataset/green_water1/3.bag', 'raw_dataset/green_water1/4.bag', 'raw_dataset/green_water1/5.bag', 'raw_dataset/green_water1/6.bag', 'raw_dataset/green_water1/7.bag', 'raw_dataset/green_water1/8.bag', 'raw_dataset/green_water1/9.bag', 'raw_dataset/green_water1/circular1.bag', 'raw_dataset/green_water1/circular2.bag', 'raw_dataset/green_water1/circular3.bag']
        if not bag_files:
            print(f"Warning: No .bag files found in {scenario_path}")
            continue
        else:
            print(f"Found {len(bag_files)} bag files in {scenario_path}:")
        
        for bag_file in bag_files:
            # 从完整路径中提取不带 .bag 后缀的文件名
            # os.path.basename('path/to/my_bag.bag') -> 'my_bag.bag'
            # os.path.splitext('my_bag.bag') -> ('my_bag', '.bag')
            bag_name_with_ext = os.path.basename(bag_file)
            bag_name_without_ext = os.path.splitext(bag_name_with_ext)[0]
            if bag_name_without_ext != 'stone_end':
                continue
            # 调用更新后的函数，传入新的 bag_name
            # 注意：process_bag现在返回的是这个bag处理的帧数，我们不再需要用它来累加
            output_folder_base = os.path.join(args.output_dir, f"{scenario_name}_{bag_name_without_ext}" )
            process_bag(bag_file, output_folder_base, sonar_denoiser, tf_buffer)
            print(f"Processed bag: {bag_name_with_ext} -> Output folder: {output_folder_base}")

    print("\nProcessing complete!")

if __name__ == '__main__':
    main()