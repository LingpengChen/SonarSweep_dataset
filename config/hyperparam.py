# sonar_intrinsics:
Min_range = 0.1
Max_range = 5.0
Range_res = 0.005
Img_height = 1000
# Hori_fov = 130.0
# Vert_fov = 20.0
Hori_fov = 60.0
Vert_fov = 12.0
Angular_res = 0.4
Img_width = 150

WORLD_FRAME = 'world'  # 或 'odom', 'map' 等固定坐标系
IMU_FRAME = 'imu_link' # 右相机 (/isaacsim/camera)
RIGHT_CAM_FRAME = 'camera_link' # 右相机 (/isaacsim/camera)
LEFT_CAM_FRAME = 'camera2_link' # 左相机 (/isaacsim/camera2)
SONAR_FRAME = 'sonar_link'    # 声纳

# 2. ROS Topics
TOPIC_CAM_RIGHT_RGB = '/isaacsim/camera/image_raw'
TOPIC_CAM_RIGHT_DEPTH = '/isaacsim/camera/depth/image_raw'
TOPIC_CAM_RIGHT_INFO = '/isaacsim/camera/camera_info'
TOPIC_CAM_LEFT_RGB = '/isaacsim/camera2/image_raw'
TOPIC_CAM_LEFT_DEPTH = '/isaacsim/camera2/depth/image_raw'
TOPIC_SONAR = '/isaacsim/sonar_rect_image'
TOPIC_TF = '/tf'
TOPIC_TF_STATIC = '/tf_static'

# 3. 同步容差 (秒)
TIME_SLOP = 0.05 # 50ms

SONAR_INTRINSIC_CONTENT = f"""# scalar1: max_range
# scalar2: range_res
# scalar3: img_height
# scalar4: hori_fov
# scalar5: vert_fov
# scalar6: angular_res
# scalar7: img_width
{Max_range}
{Range_res}
{Img_height}
{Hori_fov}
{Vert_fov}
{Angular_res}
{Img_width}
"""
