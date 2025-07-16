#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import argparse
from typing import Tuple
import math
    
def padding_sonar_image(input_image, top_padding_pixels=20):
    """
    对声纳图像进行顶部填充操作。

    根据示意图，该函数将一个 980x150 的图像在顶部填充20像素的黑边，
    使其变为 1000x150 的图像。

    :param input_image: 输入的OpenCV图像 (NumPy 数组)。
    :return: 填充后的OpenCV图像 (NumPy 数组)。
    """
    # 从示意图得知需要填充的参数
    top_padding = top_padding_pixels
    bottom_padding = 0
    left_padding = 0
    right_padding = 0
    
    # 填充值为0 (黑色)
    padding_value = 0
    
    # 使用 OpenCV 的 copyMakeBorder 函数进行填充
    # cv2.BORDER_CONSTANT 表示使用一个常数值进行填充
    padded_image = cv2.copyMakeBorder(
        input_image, 
        top_padding, 
        bottom_padding, 
        left_padding, 
        right_padding, 
        cv2.BORDER_CONSTANT, 
        value=padding_value
    )
    
    return padded_image


def rect_to_sonar_map(rect_image: np.ndarray,
                      range_pixels: int = 1000, 
                      azimuth_pixels: int = 150,
                      azimuth_bounds: Tuple[float, float] = (-0.5236, 0.5236)) -> np.ndarray:
    """
    将矩形声纳图像映射为扇形图像
    
    Args:
        range_pixels: 距离采样点数量
        azimuth_pixels: 方位角采样点数量  
        azimuth_bounds: 方位角范围(min_angle, max_angle)
        rect_image: 输入的矩形图像
        
    Returns:
        扇形声纳图像
    """
    # 计算目标图像尺寸
    azimuth_pixels = rect_image.shape[1]
    
    
    minus_width = math.floor(range_pixels * math.sin(azimuth_bounds[0]))
    plus_width = math.ceil(range_pixels * math.sin(azimuth_bounds[1]))
    width = plus_width - minus_width
    

    # 创建映射矩阵
    map_x = np.zeros((range_pixels, width), dtype=np.float32)
    map_y = np.zeros((range_pixels, width), dtype=np.float32)
    
    # 计算方位角步长
    db = (azimuth_bounds[1] - azimuth_bounds[0]) / azimuth_pixels
    origin_x = abs(minus_width)
    
    # 计算映射关系
    for x in range(width):
        for y in range(range_pixels):
            dx = x - origin_x
            dy = y
            
            range_val = math.sqrt(dx * dx + dy * dy)
            azimuth = math.atan2(dx, dy)
            
            map_x[y, x] = (azimuth - azimuth_bounds[0]) / db
            map_y[y, x] = range_val
    
    # 执行重映射
    sonar_image = cv2.remap(rect_image, map_x, map_y, cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
    return sonar_image


def process_images_in_folder(input_dir, output_dir):
    """
    遍历输入文件夹中的所有图像，应用padding函数，并保存到输出文件夹。

    :param input_dir: 包含原始图像的文件夹路径。
    :param output_dir: 保存处理后图像的文件夹路径。
    """
    # 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 获取文件夹中所有文件的列表
    try:
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    except FileNotFoundError:
        print(f"错误: 输入目录 '{input_dir}' 不存在。")
        return

    if not image_files:
        print(f"在目录 '{input_dir}' 中未找到任何图像文件。")
        return
        
    print(f"在 '{input_dir}' 中找到 {len(image_files)} 张图像，开始处理...")
    
    # 遍历所有图像文件
    for i, filename in enumerate(sorted(image_files)):
        # 构建完整的文件路径
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        # 读取图像，使用 cv2.IMREAD_GRAYSCALE 确保以灰度模式读取
        image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"\n警告: 无法读取图像 {filename}，已跳过。")
            continue
            
        # 调用函数对图像进行padding处理
        cv2.imshow("image original", image)
        padded_image = padding_sonar_image(image, top_padding_pixels=20)
        cv2.imshow("padded original", padded_image)
        sonar_image = rect_to_sonar_map(padded_image, 
                                        range_pixels= 1000, 
                                        azimuth_pixels = 150,
                                        azimuth_bounds = (-0.5236, 0.5236))
        cv2.imshow("sonar image", sonar_image)
        
        cv2.waitKey(0)
        # 验证尺寸 (可选)
        # print(f"原始尺寸: {image.shape}, 处理后尺寸: {padded_image.shape}")
        
        # 保存处理后的图像
        cv2.imwrite(output_path, padded_image)
        
        # 打印进度
        print(f"\r已处理: {i + 1}/{len(image_files)} - {filename}", end="")

    print(f"\n\n处理完成！所有图像已保存到 '{output_dir}' 目录。")


if __name__ == '__main__':
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="对声纳图像进行顶部填充处理。")
    parser.add_argument("-input_dir", default="cp", help="包含原始图像的输入目录 (例如: tmp)")
    parser.add_argument("-o", "--output_dir", default="cp_padded", help="保存处理后图像的输出目录 (默认为: tmp_padded)")
    # parser.add_argument("-input_dir", default="tmp", help="包含原始图像的输入目录 (例如: tmp)")
    # parser.add_argument("-o", "--output_dir", default="tmp_padded", help="保存处理后图像的输出目录 (默认为: tmp_padded)")

    args = parser.parse_args()

    process_images_in_folder(args.input_dir, args.output_dir)