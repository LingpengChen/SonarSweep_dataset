import numpy as np
import cv2
from pathlib import Path
import sys

def process_scene(scene_dir: Path, hori_fov_deg: float, vert_fov_deg: float):
    """
    处理单个场景文件夹：根据新的FOV裁剪图像并更新相机内参。

    Args:
        scene_dir (Path): 场景文件夹的路径。
        hori_fov_deg (float): 新的目标水平视场角（度）。
        vert_fov_deg (float): 新的目标垂直视场角（度）。
    """
    print(f"--- Processing scene: {scene_dir.name} ---")

    # 1. 定义文件路径
    intrinsic_path = scene_dir / 'cam_intrinsic.txt'
    files_to_crop = [
        'cam_left.png', 'cam_right.png',
        'depth_left.npy', 'depth_left_visualize.png',
        'depth_right.npy', 'depth_right_visualize.png'
    ]

    # 2. 检查并读取原始相机内参
    if not intrinsic_path.exists():
        print(f"  [Warning] 'cam_intrinsic.txt' not found in {scene_dir}. Skipping.")
        return

    try:
        intrinsics = np.genfromtxt(intrinsic_path).astype(np.float32).reshape((3, 3))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    except Exception as e:
        print(f"  [Error] Failed to read or parse intrinsics from {intrinsic_path}: {e}. Skipping.")
        return

    # print(f"  Original intrinsics:\n{intrinsics}")
    # print(f"  Original principal point: (cx={cx}, cy={cy})")
    
    # 3. 计算新的图像尺寸和裁剪框
    # 将FOV从度转换为弧度
    hori_fov_rad = np.deg2rad(hori_fov_deg)
    vert_fov_rad = np.deg2rad(vert_fov_deg)

    # 根据FOV和焦距计算新的宽度和高度
    # 公式: new_W = 2 * fx * tan(FOV_x / 2)
    new_w = 2 * fx * np.tan(hori_fov_rad / 2)
    new_h = 2 * fy * np.tan(vert_fov_rad / 2)

    # 新的尺寸必须是整数
    new_w = int(round(new_w))
    new_h = int(round(new_h))
    
    # print(f"  Calculated new dimensions: {new_w}x{new_h}")

    # 计算以(cx, cy)为中心的裁剪框坐标
    x1 = int(round(cx - new_w / 2))
    y1 = int(round(cy - new_h / 2))
    x2 = int(round(cx + new_w / 2))
    y2 = int(round(cy + new_h / 2))
    
    # 需要检查原始图像尺寸以防裁剪框越界
    # 我们从其中一个图像文件中获取原始尺寸
    try:
        # 尝试读取一个png图像来获取原始尺寸
        ref_image_path = scene_dir / 'cam_left.png'
        if not ref_image_path.exists():
             # 如果不存在，尝试另一个
             ref_image_path = scene_dir / 'depth_left_visualize.png'
        
        if not ref_image_path.exists():
            print(f"  [Error] Cannot find a reference image to determine original size. Skipping.")
            return

        ref_image = cv2.imread(str(ref_image_path), cv2.IMREAD_UNCHANGED)
        original_h, original_w = ref_image.shape[:2]
        print(f"  Original image dimensions found: {original_w}x{original_h}")

        # 确保裁剪框不越界
        if x1 < 0 or y1 < 0 or x2 > original_w or y2 > original_h:
            print(f"  [Error] Calculated crop box [{x1}:{x2}, {y1}:{y2}] is out of original image bounds [{original_w}x{original_h}]. Skipping scene.")
            return
            
    except Exception as e:
        print(f"  [Error] Could not read reference image to get dimensions: {e}. Skipping.")
        return

    print(f"  Crop box (x1, y1, x2, y2): ({x1}, {y1}, {x2}, {y2})")

    # 4. 遍历文件，进行裁剪并保存
    for filename in files_to_crop:
        input_path = scene_dir / filename
        if not input_path.exists():
            print(f"  [Info] File '{filename}' not found. Skipping this file.")
            continue

        output_path = scene_dir / f"cropped_{filename}"
        
        try:
            if filename.endswith('.png'):
                # 读取PNG图像
                img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
                # 裁剪
                cropped_img = img[y1:y2, x1:x2]
                # 保存
                cv2.imwrite(str(output_path), cropped_img)
                # print(f"  Successfully cropped and saved '{output_path.name}'")
            
            elif filename.endswith('.npy'):
                # 读取NPY文件
                data = np.load(input_path)
                # 裁剪
                cropped_data = data[y1:y2, x1:x2]
                # 保存
                np.save(output_path, cropped_data)
                # print(f"  Successfully cropped and saved '{output_path.name}'")

        except Exception as e:
            print(f"  [Error] Failed to process {filename}: {e}")

    # 5. 创建并保存新的相机内参文件
    # 新的光心是裁剪后图像的中心
    new_cx = new_w / 2.0
    new_cy = new_h / 2.0
    
    # 焦距 fx, fy 保持不变
    new_intrinsics = np.array([
        [fx, 0, new_cx],
        [0, fy, new_cy],
        [0, 0, 1]
    ], dtype=np.float32)

    new_intrinsic_path = scene_dir / 'cropped_cam_intrinsic.txt'
    np.savetxt(new_intrinsic_path, new_intrinsics, fmt='%.6f', delimiter=' ')
    print(f"  New intrinsics saved to '{new_intrinsic_path.name}'")
    print(f"  New intrinsics:\n{new_intrinsics}")


if __name__ == '__main__':
    # --- 配置参数 ---
    # 数据集根目录
    ROOT_DATASET_DIR = Path('./processed_dataset/vfov12hfov60/')
    # 新的视场角（单位：度）
    HORI_FOV = 60.0
    VERT_FOV = 12.0
    # ----------------

    if not ROOT_DATASET_DIR.is_dir():
        print(f"Error: Root directory '{ROOT_DATASET_DIR}' not found.")
        print("Please make sure the script is in the correct location or update the ROOT_DATASET_DIR variable.")
        sys.exit(1)

    # 遍历所有子文件夹
    for scene_directory in ROOT_DATASET_DIR.iterdir():
        if scene_directory.is_dir():
            process_scene(scene_directory, HORI_FOV, VERT_FOV)

    print("\n--- All scenes processed. ---")