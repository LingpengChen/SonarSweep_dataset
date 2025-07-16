import numpy as np
import cv2
from pathlib import Path
import sys
from tqdm import tqdm  # 导入tqdm

SONAR_INTRINSIC_CONTENT = """# scalar1: max_range
# scalar2: range_res
# scalar3: img_height
# scalar4: hori_fov
# scalar5: vert_fov
# scalar6: angular_res
# scalar7: img_width
5.0
0.005
1000
60.0
12.0
0.4
150
"""

def process_scene(scene_dir: Path, hori_fov_deg: float, vert_fov_deg: float):
    """
    处理单个场景文件夹：根据新的FOV裁剪图像并更新相机内参。
    此版本移除了不必要的print语句以提高性能。

    Args:
        scene_dir (Path): 场景文件夹的路径。
        hori_fov_deg (float): 新的目标水平视场角（度）。
        vert_fov_deg (float): 新的目标垂直视场角（度）。
    """
    # 1. 定义文件路径
    intrinsic_path = scene_dir / 'cam_intrinsic.txt'
    files_to_crop = [
        'cam_left.png', 'cam_right.png',
        'depth_left.npy', 'depth_left_visualize.png',
        'depth_right.npy', 'depth_right_visualize.png'
    ]

    # 2. 检查并读取原始相机内参
    if not intrinsic_path.exists():
        # 这是一个关键警告，需要打印出来
        print(f"\n[Warning] 'cam_intrinsic.txt' not found in {scene_dir}. Skipping.")
        return

    try:
        intrinsics = np.genfromtxt(intrinsic_path).astype(np.float32).reshape((3, 3))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    except Exception as e:
        print(f"\n[Error] Failed to read or parse intrinsics from {intrinsic_path}: {e}. Skipping.")
        return

    # 3. 计算新的图像尺寸和裁剪框
    hori_fov_rad = np.deg2rad(hori_fov_deg)
    vert_fov_rad = np.deg2rad(vert_fov_deg)
    new_w = int(round(2 * fx * np.tan(hori_fov_rad / 2)))
    new_h = int(round(2 * fy * np.tan(vert_fov_rad / 2)))
    
    # 确保新尺寸为正数
    if new_w <= 0 or new_h <= 0:
        print(f"\n[Error] Calculated new dimensions ({new_w}x{new_h}) are invalid. Skipping {scene_dir.name}.")
        return

    x1 = int(round(cx - new_w / 2))
    y1 = int(round(cy - new_h / 2))
    x2 = x1 + new_w
    y2 = y1 + new_h
    
    # 检查原始图像尺寸以防裁剪框越界
    try:
        ref_image_path = scene_dir / 'cam_left.png'
        if not ref_image_path.exists():
             ref_image_path = scene_dir / 'depth_left_visualize.png'
        
        if not ref_image_path.exists():
            print(f"\n[Error] Cannot find a reference image in {scene_dir.name} to determine original size. Skipping.")
            return

        # 使用 imread 获取尺寸，避免解码整个图像的开销（对于某些格式）
        # 但对于png/jpeg, 完整的读取是必要的。
        ref_image = cv2.imread(str(ref_image_path), cv2.IMREAD_UNCHANGED)
        if ref_image is None:
            print(f"\n[Error] Could not read reference image: {ref_image_path}. Skipping.")
            return
            
        original_h, original_w = ref_image.shape[:2]

        if x1 < 0 or y1 < 0 or x2 > original_w or y2 > original_h:
            print(f"\n[Error] In {scene_dir.name}, calculated crop box [{x1}:{x2}, {y1}:{y2}] is out of original image bounds [{original_w}x{original_h}]. Skipping.")
            return
            
    except Exception as e:
        print(f"\n[Error] In {scene_dir.name}, could not read reference image to get dimensions: {e}. Skipping.")
        return

    # 4. 遍历文件，进行裁剪并保存
    for filename in files_to_crop:
        input_path = scene_dir / filename
        if not input_path.exists():
            continue  # 静默跳过不存在的文件

        output_path = scene_dir / f"cropped_{filename}"
        
        try:
            # rgb
            if filename.endswith('.png'):
                img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
                cropped_img = img[y1:y2, x1:x2]
                cv2.imwrite(str(output_path), cropped_img)
            
            elif filename.endswith('.npy'):
                data = np.load(input_path)
                cropped_data = data[y1:y2, x1:x2]
                np.save(output_path, cropped_data)
        except Exception as e:
            # 打印处理单个文件时发生的错误
            print(f"\n[Error] Failed to process {input_path}: {e}")

    # 5. 创建并保存新的相机内参文件
    new_cx = new_w / 2.0
    new_cy = new_h / 2.0
    
    new_intrinsics = np.array([
        [fx, 0, new_cx],
        [0, fy, new_cy],
        [0, 0, 1]
    ], dtype=np.float32)

    new_intrinsic_path = scene_dir / 'cropped_cam_intrinsic.txt'
    np.savetxt(new_intrinsic_path, new_intrinsics, fmt='%.6f', delimiter=' ')
    
    # --- 新增内容 ---
    # 6. 创建/覆盖 sonar_intrinsic.txt 文件
    # try:
    #     sonar_intrinsic_path = scene_dir / 'sonar_intrinsic.txt'
    #     with open(sonar_intrinsic_path, 'w', encoding='utf-8') as f:
    #         f.write(SONAR_INTRINSIC_CONTENT)
    # except Exception as e:
    #     print(f"\n[Error] Failed to write sonar_intrinsic.txt in {scene_dir.name}: {e}")
        
    

if __name__ == '__main__':
    # --- 配置参数 ---
    # 数据集根目录 (使用你提供的路径)
    ROOT_DATASET_DIR = Path('./processed_dataset/vfov12hfov60/')
    # 新的视场角（单位：度）
    HORI_FOV = 60.0
    VERT_FOV = 12.0
    # ----------------

    if not ROOT_DATASET_DIR.is_dir():
        print(f"Error: Root directory '{ROOT_DATASET_DIR}' not found.")
        print("Please make sure the script is in the correct location or update the ROOT_DATASET_DIR variable.")
        sys.exit(1)

    # 首先获取所有需要处理的子文件夹列表
    directories_to_process = [d for d in ROOT_DATASET_DIR.iterdir() if d.is_dir()]
    
    if not directories_to_process:
        print(f"No subdirectories found in '{ROOT_DATASET_DIR}'.")
        sys.exit(0)

    print(f"Found {len(directories_to_process)} directories to process in '{ROOT_DATASET_DIR}'.")

    # 使用tqdm来创建一个进度条
    for scene_directory in tqdm(directories_to_process, desc="Processing Scenes", unit="dir"):
        if scene_directory.is_dir():
            process_scene(scene_directory, HORI_FOV, VERT_FOV)

    print("\n--- All scenes processed. ---")