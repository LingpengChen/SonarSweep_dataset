import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def read_tiff(image_path, is_depth=False):
    """
    使用PIL读取TIFF/TIF文件，增加depth图像处理
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    img = np.array(img)
    
    if is_depth:
        if img.dtype == np.uint16:
            # 对于深度图，保持16位精度
            return img
    else:
        # RGB图像处理
        if img.dtype == np.uint16:
            img = (img / 256).astype(np.uint8)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    return img

def undistort_image(image, K, D):
    """
    去畸变处理
    """
    h, w = image.shape[:2]
    newcameramtx0, roi0 = cv2.getOptimalNewCameraMatrix(K, D, (w,h), 1, (w,h))
    dst0 = cv2.undistort(image, K, D, None, newcameramtx0)
    x, y, w, h = roi0
    if roi0 != (0, 0, 0, 0):
        dst0 = dst0[y:y+h, x:x+w]
    return dst0, newcameramtx0

def crop_image_to_fov(image, image_intrinsics, APERTURE, AZIMUTH):
    """
    根据指定的FOV裁切图像，确保输出图像的长宽都是4的倍数
    """
    HEIGHT, WIDTH = image.shape[:2]
    FX = image_intrinsics[0,0]
    FY = image_intrinsics[1,1]
    
    vertical_fov = 2 * np.arctan2(HEIGHT/2, FY)
    horizontal_fov = 2 * np.arctan2(WIDTH/2, FX)
    
    new_height = HEIGHT
    new_width = WIDTH
    
    if vertical_fov > APERTURE:
        new_height = int(2 * FY * np.tan(APERTURE/2))
        # 调整到4的倍数
        new_height = new_height - (new_height % 4)
    if horizontal_fov > AZIMUTH:
        new_width = int(2 * FX * np.tan(AZIMUTH/2))
        # 调整到4的倍数
        new_width = new_width - (new_width % 4)
    
    # 确保起始位置也是偶数，这样裁切后的图像会更居中
    start_x = (WIDTH - new_width) // 2
    start_x = start_x - (start_x % 2)  # 确保是偶数
    end_x = start_x + new_width
    
    start_y = (HEIGHT - new_height) // 2
    start_y = start_y - (start_y % 2)  # 确保是偶数
    end_y = start_y + new_height
    
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    # 验证输出尺寸是否符合要求
    assert cropped_image.shape[0] % 4 == 0, "Height is not divisible by 4"
    assert cropped_image.shape[1] % 4 == 0, "Width is not divisible by 4"
    
    new_intrinsics = np.array([
        [FX, 0, new_width/2],
        [0, FY, new_height/2],
        [0, 0, 1]
    ])
    
    return cropped_image, new_intrinsics

def process_images(rgb_path, depth_path, output_dir):
    """
    处理RGB和深度图像
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取图像
    rgb_img = read_tiff(rgb_path)
    depth_img = read_tiff(depth_path, is_depth=True)
    
    # 原始内参和畸变系数
    original_intrinsics = np.array([[FX,0,CX],[0,FY,CY],[0,0,1]])
    distortion_coeffs = np.array([K1, K2, P1, P2])
    
    # 处理RGB图像
    undistorted_rgb, updated_intrinsics = undistort_image(rgb_img, original_intrinsics, distortion_coeffs)
    final_rgb, final_intrinsics = crop_image_to_fov(undistorted_rgb, updated_intrinsics, APERTURE, AZIMUTH)
    
    # 处理深度图像
    undistorted_depth, _ = undistort_image(depth_img, original_intrinsics, distortion_coeffs)
    final_depth, _ = crop_image_to_fov(undistorted_depth, updated_intrinsics, APERTURE, AZIMUTH)
    
    # 保存结果
    cv2.imwrite(os.path.join(output_dir, 'processed_rgb.png'), final_rgb)
    np.save(os.path.join(output_dir, 'processed_depth.npy'), final_depth)

    # 保存图像shape和内参矩阵
    with open(os.path.join(output_dir, 'image_info.txt'), 'w') as f:
        f.write(f"RGB Shape: {final_rgb.shape}\n")
        f.write(f"DEPTH Shape: {final_depth.shape}\n")
        f.write("\nFinal Intrinsics Matrix:\n")
        np.savetxt(f, final_intrinsics, fmt='%.6f')
    
    return final_rgb, final_depth, final_intrinsics

if __name__ == "__main__":
    from hyper import *
    
    # 主程序
    rgb_image_path = "/media/clp/T9/stereo_dataset/convert_sonar/data/rgb/16233051862684293.tiff"
    depth_image_path = "/media/clp/T9/stereo_dataset/convert_sonar/data/depth/16233051862684293_SeaErra_abs_depth.tif"
    output_directory = "processed_results"

    final_rgb, final_depth, final_intrinsics = process_images(rgb_image_path, depth_image_path, output_directory)