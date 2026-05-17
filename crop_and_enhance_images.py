import numpy as np
import cv2
from pathlib import Path
import sys
import argparse
from tqdm import tqdm  # 导入tqdm


clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


IMAGE_FILES = [
    'cam_left.png', 'cam_right.png',
    'depth_left_visualize.png', 'depth_right_visualize.png'
]
DEPTH_FILES = ['depth_left.npy', 'depth_right.npy']
CAMERA_FILES = {'cam_left.png', 'cam_right.png'}


def round_down_to_multiple(value: float, multiple: int = 4) -> int:
    return int(value) // multiple * multiple


def process_scene(scene_dir: Path, hori_fov_deg: float, vert_fov_deg: float, overwrite: bool = False):
    """
    处理单个场景文件夹：根据新的FOV裁剪图像并更新相机内参。
    此版本移除了不必要的print语句以提高性能。

    Args:
        scene_dir (Path): 场景文件夹的路径。
        hori_fov_deg (float): 新的目标水平视场角（度）。
        vert_fov_deg (float): 新的目标垂直视场角（度）。
    """
    new_intrinsic_path = scene_dir / 'cropped_cam_intrinsic.txt'
    if new_intrinsic_path.exists() and not overwrite:
        return  # 已经处理过，跳过
        
    intrinsic_path = scene_dir / 'cam_intrinsic.txt'

    if not intrinsic_path.exists():
        print(f"\n[Warning] 'cam_intrinsic.txt' not found in {scene_dir}. Skipping.")
        return

    try:
        intrinsics = np.genfromtxt(intrinsic_path).astype(np.float32).reshape((3, 3))
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    except Exception as e:
        print(f"\n[Error] Failed to read or parse intrinsics from {intrinsic_path}: {e}. Skipping.")
        return

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
    except Exception as e:
        print(f"\n[Error] In {scene_dir.name}, could not read reference image to get dimensions: {e}. Skipping.")
        return

    target_w = 2 * fx * np.tan(np.deg2rad(hori_fov_deg) / 2)
    target_h = 2 * fy * np.tan(np.deg2rad(vert_fov_deg) / 2)
    max_w = 2 * min(cx, original_w - cx)
    max_h = 2 * min(cy, original_h - cy)
    new_w = round_down_to_multiple(min(target_w, max_w))
    new_h = round_down_to_multiple(min(target_h, max_h))

    if new_w <= 0 or new_h <= 0:
        print(f"\n[Error] Calculated crop size ({new_w}x{new_h}) is invalid. Skipping {scene_dir.name}.")
        return

    x1 = max(0, int(round(cx - new_w / 2)))
    y1 = max(0, int(round(cy - new_h / 2)))
    x2 = min(original_w, x1 + new_w)
    y2 = min(original_h, y1 + new_h)

    # 4. 遍历文件，进行裁剪并保存
    for filename in IMAGE_FILES + DEPTH_FILES:
        input_path = scene_dir / filename
        if not input_path.exists():
            continue  # 静默跳过不存在的文件

        output_path = scene_dir / f"cropped_{filename}"
        
        try:
            if filename.endswith('.png'):
                img = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"\n[Warning] Could not read image: {input_path}")
                    continue
                cropped_img = img[y1:y2, x1:x2]
                if overwrite or not output_path.exists():
                    cv2.imwrite(str(output_path), cropped_img)

                if filename in CAMERA_FILES:
                    enhanced_gray_output_path = scene_dir / f"enhanced_gray_{filename}"
                    if overwrite or not enhanced_gray_output_path.exists():
                        if cropped_img.ndim == 2:
                            gray_standard = cropped_img
                        else:
                            gray_standard = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
                        cv2.imwrite(str(enhanced_gray_output_path), clahe.apply(gray_standard))
                
            elif filename.endswith('.npy'):
                data = np.load(input_path)
                cropped_data = data[y1:y2, x1:x2]
                if overwrite or not output_path.exists():
                    np.save(output_path, cropped_data)
        except Exception as e:
            # 打印处理单个文件时发生的错误
            print(f"\n[Error] Failed to process {input_path}: {e}")
            continue  # 继续处理其他文件

    # 5. 创建并保存新的相机内参文件
    new_cx = cx - x1
    new_cy = cy - y1
    
    new_intrinsics = np.array([
        [fx, 0, new_cx],
        [0, fy, new_cy],
        [0, 0, 1]
    ], dtype=np.float32)

    if overwrite or not new_intrinsic_path.exists():
        try:
            np.savetxt(new_intrinsic_path, new_intrinsics, fmt='%.6f', delimiter=' ')
        except Exception as e:
            print(f"\n[Error] Failed to save intrinsics for {scene_dir.name}: {e}")
    

if __name__ == '__main__':
    # --- 配置参数 ---
    # 数据集根目录 (使用你提供的路径)
    from config.hyperparam import Hori_fov, Vert_fov
    parser = argparse.ArgumentParser(description="Crop camera/depth data to sonar FOV and enhance RGB images.")
    parser.add_argument('--sonar_type', type=str, default='vfov12hfov60', help='Sonar type folder name.')
    parser.add_argument('--root_dir', type=Path, default=None, help='Processed dataset directory.')
    parser.add_argument('--hori_fov', type=float, default=Hori_fov, help='Target horizontal FOV in degrees.')
    parser.add_argument('--vert_fov', type=float, default=Vert_fov, help='Target vertical FOV in degrees.')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate cropped and enhanced files.')
    args = parser.parse_args()

    ROOT_DATASET_DIR = args.root_dir or Path(f'./processed_dataset/{args.sonar_type}/')
    # 新的视场角（单位：度）
    HORI_FOV = args.hori_fov
    VERT_FOV = args.vert_fov

    if not ROOT_DATASET_DIR.is_dir():
        print(f"Error: Root directory '{ROOT_DATASET_DIR}' not found.")
        print("Please make sure the script is in the correct location or update the ROOT_DATASET_DIR variable.")
        sys.exit(1)

    directories_to_process = sorted(d for d in ROOT_DATASET_DIR.iterdir() if d.is_dir())
    
    if not directories_to_process:
        print(f"No subdirectories found in '{ROOT_DATASET_DIR}'.")
        sys.exit(0)

    print(f"Found {len(directories_to_process)} directories to process in '{ROOT_DATASET_DIR}'.")

    # 使用tqdm来创建一个进度条
    for scene_directory in tqdm(directories_to_process, desc="Processing Scenes", unit="dir"):
        if scene_directory.is_dir():
            try:
                process_scene(scene_directory, HORI_FOV, VERT_FOV, overwrite=args.overwrite)
            except Exception as e:
                print(f"\n[Error] Failed to process directory {scene_directory.name}: {e}")
                continue  # 继续处理下一个目录

    print("\n--- All scenes processed. ---")
