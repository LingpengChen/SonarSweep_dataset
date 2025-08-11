# SonarSweep Dataset


## 录制rosbag in Oceansim
（仿真器中left cam 和 sonar 重合）
进入/raw_dataset

仿真器包含topics:
    /isaacsim/camera/camera_info          msgs    : sensor_msgs/CameraInfo
    /isaacsim/camera/depth/image_raw      msgs    : sensor_msgs/Image     
    /isaacsim/camera/image_raw            msgs    : sensor_msgs/Image     
    /isaacsim/camera2/depth/image_raw     msgs    : sensor_msgs/Image     
    /isaacsim/camera2/image_raw           msgs    : sensor_msgs/Image     
    /isaacsim/imu/data                   msgs    : sensor_msgs/Imu       
    /isaacsim/sonar_rect_image            msgs    : sensor_msgs/Image     
    /tf                                  msgs    : tf2_msgs/TFMessage    
    /tf_static                             msg     : tf2_msgs/TFMessage
里面的数据的规律是，在某一时刻，机器人会同时记录下camera(right cam) camera(left cam) sonar tf(包含imu pose，可以通过static tf解算得到cam_right_pose)

我们准备SonarSweep的目标是：对于每个图像，我们把他们按照timestamp划归到一个文件夹下面，文件夹命名规则如下{scenerio_name}_{index} 保存到raw_dataset

# STEP 1 record bag in raw_dataset
1. rosparam set use_sim_time false
2. rosbag record -O dataset.bag /isaacsim/camera/camera_info /isaacsim/camera/depth/image_raw /isaacsim/camera/image_raw /isaacsim/camera2/depth/image_raw /isaacsim/camera2/image_raw /isaacsim/imu/data /isaacsim/sonar_rect_image /tf /tf_static

3. rosbag record -O background.bag /isaacsim/sonar_rect_image

# STEP 2 
set config/hyperparam.py
    soanr fov
python3 rosbag2folders.py


### 录制结果可视化
1. rviz -d ./config/config.rviz 
2. python3 publish_pcd.py 
    project depth image to point cloud
   python3 publish_converted_sonar.py
3. rosparam set use_sim_time true
4. rosbag play -l -r raw_dataset/green_water1/1.bag 




## Main scripts
1. rosbag2folders.py:
    read rosbag and convert each image at certain timestamp into a folder with 

    ├── cam_intrinsic.txt   # 内参矩阵
    ├── cam_left.png        # (0.3,0.2,0) cam2 原始RGB图像
    ├── cam_right_pose.txt 
    ├── cam_right.png       # (0.3,0.0,0) cam1 
    ├── depth_left_visualize.png    # 深度图可视化
    ├── depth_left.npy              # 深度图
    ├── depth_right_visualize.png   # 深度图可视化
    ├── depth_right.npy             # 深度图

    ├── sonar.png   
    ├── sonar_rect.png
    ├── sonar_denoise.png   
    ├── sonar_rect_denoise.png
    ├── sonar_intrinsic.txt  # 

    └── T_camright2sonar.txt  # T_sonar_from_cam ie T_cam2sonar

    其中
    1. sonar data 
        从仿真器里面拿出的原始图像是Min_range到max_range
        我们进行了padding使得它从0~max_range，并保存为sonar_rect，为了可视化，我们转到笛卡尔坐标系，得到sonar
        为了给模型提供纯净的数据，我们又滤波得到sonar_rect_denoise.png （转到笛卡尔坐标系的是sonar_denoise.png ）
 
        sonar_rect 因为进行了补齐，所以代表的范围就是0-maxrange(5m)
        并且我们使用network_scunet对声纳图像进行了去噪声
        同时我们提供声纳内参信息 sonar.txt  
            Min_range: float = 0.1, # m
            max_range: float = 5.0, # m
            range_res: float = 0.005, # m
            img_height: float = 1000

            hori_fov: float = 60.0, # deg
            vert_fov: float = 12.0, # deg
            angular_res: float = 0.4, # deg
            img_width: float = 150

    2. camera data
        在仿真器设置的时候，left camera和sonar重合,因此我们只保存了 cam_right_pose 和 right_cam到sonar 的外参
        ├── cam_right_pose.txt 
        └── T_camright2sonar.txt 

        注意T_camright2sonar是在相机坐标系下的变换(i.e.z轴方向为相机朝向，x轴朝右)
        cam_right_pose是在世界坐标系(刚体坐标系)下，cam_right_link的pose, z轴朝天上

2. crop_and_enhance_images.py
    遍历并处理每个单个场景文件夹：根据新的FOV（sonar的FOV）裁剪图像并更新相机内参。并进行图像增强（标准灰度图 + CLAHE）

    未来可能操作：
        颜色校正优先：
            在转为灰度图之前，先对彩色图像进行颜色校正，尝试恢复一些红色分量，并平衡白平衡。例如使用“灰度世界算法”或更复杂的“暗通道先验(Dark Channel Prior)”等去雾/去水下散射算法。
        使用对比度受限的自适应直方图均衡化 (CLAHE)：
            这是一个非常强大且常用的技术，尤其适合水下图像。它不是对整张图进行均衡化，而是将图像分成许多小块，对每个小块分别进行直方图均衡，从而极大地提升局部对比度，让隐藏在蓝绿色背景中的细节显现出来。

3. raw_dataset/retrieve_background_image.py

    1. rosparam set use_sim_time false
    2. rosbag record -O background.bag /isaacsim/sonar_rect_image

    为了帮助sonar denoise, 我需要录制一段没有任何物体的声纳图，也叫做sonar_background，我们需要录制一段 background.bag 并进行平均，得到的图像会被保存在 raw_dataset/{sonar_sensor_setting}/{scenerio}/background  (e.g. raw_dataset/vfov12hfov60/green_water1/background)


### some useful command
rosbag filter dataset.bag 1.bag "t.to_sec() <= 1752220844.72 + 65.0"
rosbag play -r 10 1.bag # SonarSweep_dataset


scp -r vfov20hfov130  clp@10.26.1.168:/data2/home/clp/workspace/data/
scp -r vfov12hfov60  lingpeng@10.20.35.16:/home/lingpeng/workspace/data