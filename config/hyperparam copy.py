# sonar_intrinsics:
from pygame import K_RIGHT


Min_range = 0.0
Max_range = 5.0
Range_res = 0.01147
Img_height = 436

# calibration parameters
Min_range_target = 0.0
Max_range_target = 4.0
# Range_res_target = 0.005
# Img_height_target = 800
Range_res_target = 0.007163
Img_height_target = 698


Hori_fov = 60.0
Vert_fov = 12.0
Angular_res = 0.1172
Img_width = 512

WORLD_FRAME = 'world'  # 或 'odom', 'map' 等固定坐标系
IMU_FRAME = 'imu_link' # 右相机 (/isaacsim/camera)
RIGHT_CAM_FRAME = 'camera_link' # 右相机 (/isaacsim/camera)
LEFT_CAM_FRAME = 'camera2_link' # 左相机 (/isaacsim/camera2)
SONAR_FRAME = 'sonar_link'    # 声纳

# 2. ROS Topics
TOPIC_CAM_RIGHT_RGB = '/oak_ffc_sync_publisher/CAM_A/image/compressed'
TOPIC_CAM_RIGHT_DEPTH = '/isaacsim/camera/depth/image_raw'
TOPIC_CAM_RIGHT_INFO = '/isaacsim/camera/camera_info'
TOPIC_CAM_LEFT_RGB = '/oak_ffc_sync_publisher/CAM_D/image/compressed'
TOPIC_CAM_LEFT_DEPTH = '/isaacsim/camera2/depth/image_raw'
TOPIC_SONAR_RECT = '/oculus/drawn_sonar_rect'
TOPIC_SONAR = '/oculus/drawn_sonar'
TOPIC_TF = '/tf'
TOPIC_TF_STATIC = '/tf_static'

# 3. 同步容差 (秒)
TIME_SLOP = 0.05 # 50ms
IMAGE_SIZE = [1920, 1200] 
K_LEFT = [[1.507023626208046e+03,0,9.599672345094020e+02],
          [0,1.508586917497671e+03,5.968926810336995e+02],
          [0,0,1]]
RADICAL_DISTORTION_LEFT = [0.504084949029017,-0.064630911483709]
K_RIGHT = [[1.514301011614384e+03,0,9.383946573726952e+02],
           [0,1.511843979278141e+03,5.932504087775503e+02],
           [0,0,1]]
RADICAL_DISTORTION_RIGHT = [0.538906690442389,-0.205631792033296] 

T_LEFT2RIGHT = [[0.999825249771265,-0.0137249503436375,-0.0126923464297101,155.498443388040],
                [0.0137397348324110,0.999905023895296,0.00107836713766443,-5.80778918958525],
                [0.0126763404246700,-0.00125256816710575,0.999918867441966,5.21662026942604],
                [0,0,0,1]]

SONAR_INTRINSIC_CONTENT = f"""# scalar1: max_range
# scalar2: range_res
# scalar3: img_height
# scalar4: hori_fov
# scalar5: vert_fov
# scalar6: angular_res
# scalar7: img_width
{Max_range_target}
{Range_res_target}
{Img_height_target}
{Hori_fov}
{Vert_fov}
{Angular_res}
{Img_width}
"""
