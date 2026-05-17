#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式图像拼接器
允许用户从多张源图像中选择最佳的patch来组成最终图像
"""

import numpy as np
import cv2
from PIL import Image
import os
import sys

class InteractivePatchSelector:
    def __init__(self, source_images, patch_size=(64, 64), overlap=0):
        """
        初始化交互式patch选择器
        
        Args:
            source_images (list): 源图像文件路径列表
            patch_size (tuple): patch大小 (width, height)
            overlap (int): patch之间的重叠像素数
        """
        self.source_images = []
        self.source_names = []
        self.patch_size = patch_size
        self.overlap = overlap
        self.current_patch_row = 0
        self.current_patch_col = 0
        self.result_image = None
        self.patches_grid = None
        self.total_rows = 0
        self.total_cols = 0
        self.result_window_name = "实时拼接结果"
        
        # 加载所有源图像
        self.load_source_images(source_images)
        
        # 立即计算网格布局以初始化result_image和patches_grid
        self.calculate_grid_layout()
        
    def load_source_images(self, image_paths):
        """加载所有源图像"""
        print("正在加载源图像...")
        for i, path in enumerate(image_paths):
            try:
                img = cv2.imread(path)
                if img is None:
                    print(f"错误：无法加载图像 {path}")
                    continue
                self.source_images.append(img)
                self.source_names.append(os.path.basename(path))
                print(f"已加载: {os.path.basename(path)} - 尺寸: {img.shape[:2]}")
            except Exception as e:
                print(f"加载图像 {path} 时出错: {e}")
        
        if not self.source_images:
            raise ValueError("没有成功加载任何源图像！")
        
        # 确保所有图像尺寸一致
        base_shape = self.source_images[0].shape
        for i, img in enumerate(self.source_images):
            if img.shape != base_shape:
                print(f"警告：调整图像 {self.source_names[i]} 的尺寸以匹配第一张图像")
                self.source_images[i] = cv2.resize(img, (base_shape[1], base_shape[0]))
    
    def calculate_grid_layout(self):
        """计算patch网格布局"""
        img_height, img_width = self.source_images[0].shape[:2]
        patch_w, patch_h = self.patch_size
        
        # 计算需要多少行和列的patches
        step_w = patch_w - self.overlap
        step_h = patch_h - self.overlap
        
        self.total_cols = (img_width - self.overlap) // step_w
        self.total_rows = (img_height - self.overlap) // step_h
        
        # 调整最后一列和最后一行以覆盖整个图像
        if (self.total_cols - 1) * step_w + patch_w < img_width:
            self.total_cols += 1
        if (self.total_rows - 1) * step_h + patch_h < img_height:
            self.total_rows += 1
        
        print(f"图像将被分割为 {self.total_rows} x {self.total_cols} = {self.total_rows * self.total_cols} 个patches")
        print(f"每个patch尺寸: {patch_w} x {patch_h}")
        
        # 初始化结果图像
        self.result_image = np.zeros_like(self.source_images[0])
        # 初始化patches选择记录
        self.patches_grid = np.zeros((self.total_rows, self.total_cols), dtype=int)
    
    def extract_patch(self, image, row, col):
        """从图像中提取指定位置的patch"""
        img_height, img_width = image.shape[:2]
        patch_w, patch_h = self.patch_size
        step_w = patch_w - self.overlap
        step_h = patch_h - self.overlap
        
        # 计算patch的起始位置
        start_y = row * step_h
        start_x = col * step_w
        
        # 确保不超出图像边界
        end_y = min(start_y + patch_h, img_height)
        end_x = min(start_x + patch_w, img_width)
        start_y = max(0, end_y - patch_h)
        start_x = max(0, end_x - patch_w)
        
        return image[start_y:end_y, start_x:end_x], (start_x, start_y, end_x, end_y)
    
    def enhance_contrast(self, image, contrast_factor=3.0, brightness=0):
        """增强图像对比度"""
        # 将图像转换为浮点数以避免溢出
        enhanced = image.astype(np.float32)
        # 应用对比度和亮度调整
        enhanced = enhanced * contrast_factor + brightness
        # 确保像素值在有效范围内
        enhanced = np.clip(enhanced, 0, 255)
        return enhanced.astype(np.uint8)
    

    def show_patch_options(self, row, col):
        """显示当前位置所有源图像的patch选项"""
        print(f"\n=== Patch ({row+1}/{self.total_rows}, {col+1}/{self.total_cols}) ===")
        
        patches = []
        coords = []
        
        # 从所有源图像中提取相同位置的patch
        for i, img in enumerate(self.source_images):
            patch, coord = self.extract_patch(img, row, col)
            patches.append(patch)
            coords.append(coord)
        
        # 创建显示窗口
        window_name = f"Patch选择器 - 位置({row+1},{col+1})"
        
        # # 将所有patch水平拼接显示
        display_patches = []
        # for i, patch in enumerate(patches):
        #     # 添加边框以便区分
        #     patch_with_border = cv2.copyMakeBorder(patch, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[100, 100, 100])
        #     # 添加标签
        #     # cv2.putText(patch_with_border, f"{i+1}: {self.source_names[i]}", 
        #     #            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #     # cv2.putText(patch_with_border, f"按 '{i+1}' 选择", 
        #     #            (10, patch_with_border.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        #     display_patches.append(patch_with_border)

        for i, patch in enumerate(patches):

            
            # 增强对比度的patch
            enhanced_patch = self.enhance_contrast(patch, contrast_factor=3, brightness=10)
            enhanced_with_border = cv2.copyMakeBorder(enhanced_patch, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[150, 100, 50])
        
            display_patches.append(enhanced_with_border)
        
        # 水平拼接
        combined_display = np.hstack(display_patches)
        
        # 调整显示尺寸
        display_height = 400  # 增加高度以便更好地显示
        aspect_ratio = combined_display.shape[1] / combined_display.shape[0]
        display_width = int(display_height * aspect_ratio)
        
        # 限制最大宽度
        max_width = 1200
        if display_width > max_width:
            display_width = max_width
            display_height = int(display_width / aspect_ratio)
        
        combined_display_resized = cv2.resize(combined_display, (display_width, display_height))
        
        # 设置窗口位置
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.moveWindow(window_name, 100, 100)
        cv2.imshow(window_name, combined_display_resized)
        
        # 立即更新实时结果显示
        self.update_result_display(row, col)
        
        return patches, coords, window_name
    
    def place_patch(self, patch, coords):
        """将选择的patch放置到结果图像中"""
        if self.result_image is None:
            return
        start_x, start_y, end_x, end_y = coords
        self.result_image[start_y:end_y, start_x:end_x] = patch
    
    def update_result_display(self, current_row=None, current_col=None):
        """更新实时结果显示"""
        if self.result_image is None:
            return
        
        # 创建显示用的副本
        display_image = self.result_image.copy()
        
        # 如果指定了当前patch位置，绘制边框指示
        if current_row is not None and current_col is not None:
            patch_w, patch_h = self.patch_size
            step_w = patch_w - self.overlap
            step_h = patch_h - self.overlap
            
            # 计算当前patch的位置
            start_x = current_col * step_w
            start_y = current_row * step_h
            end_x = min(start_x + patch_w, display_image.shape[1])
            end_y = min(start_y + patch_h, display_image.shape[0])
            
            # 绘制当前patch的边框（红色）
            cv2.rectangle(display_image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 3)
            
            # 计算完成进度
            total_patches = self.total_rows * self.total_cols
            current_patch_num = current_row * self.total_cols + current_col + 1
            completed_patches = 0
            if self.patches_grid is not None:
                completed_patches = sum(1 for r in range(self.total_rows) for c in range(self.total_cols) 
                                      if self.patches_grid[r, c] > 0)
            
            # 在图像上显示详细的进度信息
            info_lines = [
                f"当前patch: ({current_row + 1}, {current_col + 1})",
                f"总进度: {current_patch_num}/{total_patches} ({(current_patch_num/total_patches)*100:.1f}%)",
                f"已完成: {completed_patches} patches",
                f"patch大小: {patch_w}x{patch_h}"
            ]
            
            # 创建半透明背景
            overlay = display_image.copy()
            cv2.rectangle(overlay, (5, 5), (400, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, display_image, 0.3, 0, display_image)
            
            # 显示信息
            # for i, line in enumerate(info_lines):
            #     y_pos = 25 + i * 25
            #     cv2.putText(display_image, line, (10, y_pos), 
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # 计算合适的显示尺寸
        max_display_height = 700
        max_display_width = 900
        
        img_h, img_w = display_image.shape[:2]
        if img_h > max_display_height or img_w > max_display_width:
            # 按比例缩放
            scale_h = max_display_height / img_h
            scale_w = max_display_width / img_w
            scale = min(scale_h, scale_w)
            
            new_width = int(img_w * scale)
            new_height = int(img_h * scale)
            display_image = cv2.resize(display_image, (new_width, new_height))
        
        cv2.imshow(self.result_window_name, display_image)
    
    def init_result_display(self):
        """初始化结果显示窗口"""
        if self.result_image is not None:
            # 创建窗口并设置位置
            cv2.namedWindow(self.result_window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(self.result_window_name, 50, 50)
            self.update_result_display()
    
    def save_progress(self, filename="patch_selection_progress.txt"):
        """保存当前选择进度"""
        if self.patches_grid is None:
            return
        with open(filename, 'w') as f:
            f.write(f"当前进度: {self.current_patch_row}/{self.total_rows} 行, {self.current_patch_col}/{self.total_cols} 列\n")
            f.write("Patch选择记录:\n")
            for row in range(self.total_rows):
                for col in range(self.total_cols):
                    f.write(f"({row},{col}): 源图像{self.patches_grid[row,col]}\n")
    
    def load_progress(self, filename="patch_selection_progress.txt"):
        """加载之前的选择进度"""
        if not os.path.exists(filename):
            return False
        
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
                # 解析进度信息
                for line in lines:
                    if line.startswith("当前进度:"):
                        parts = line.split()
                        self.current_patch_row = int(parts[1].split('/')[0])
                        self.current_patch_col = int(parts[3].split('/')[0])
                    elif '(' in line and ')' in line and ':' in line:
                        # 解析patch选择记录
                        parts = line.strip().split(': 源图像')
                        if len(parts) == 2:
                            coord_str = parts[0]
                            source_idx = int(parts[1])
                            # 提取坐标
                            coord_str = coord_str.strip('()')
                            row, col = map(int, coord_str.split(','))
                            if self.patches_grid is not None:
                                self.patches_grid[row, col] = source_idx
            return True
        except Exception as e:
            print(f"加载进度文件时出错: {e}")
            return False
    
    def rebuild_result_from_progress(self):
        """根据保存的进度重建结果图像"""
        if self.patches_grid is None:
            return
        
        print("正在从保存的进度重建结果图像...")
        for row in range(self.total_rows):
            for col in range(self.total_cols):
                source_idx = self.patches_grid[row, col]
                if source_idx > 0:  # 如果该位置有选择记录
                    # 获取对应的patch和坐标
                    patch, coords = self.extract_patch(self.source_images[source_idx - 1], row, col)
                    self.place_patch(patch, coords)
        print("结果图像重建完成!")
    
    def run_interactive_selection(self):
        """运行交互式选择过程"""
        self.calculate_grid_layout()
        
        # 初始化结果显示窗口
        self.init_result_display()
        
        # 尝试加载之前的进度
        if self.load_progress():
            print("检测到之前的进度，是否继续？(y/n): ", end='')
            if input().lower() == 'y':
                print(f"从位置 ({self.current_patch_row+1}, {self.current_patch_col+1}) 继续...")
                # 如果有之前的进度，重新构建result_image
                self.rebuild_result_from_progress()
            else:
                self.current_patch_row = 0
                self.current_patch_col = 0
        
        print("\n=== 交互式Patch选择器 ===")
        print("操作说明:")
        print("- 数字键 1-{}: 选择对应的源图像patch".format(len(self.source_images)))
        print("- 'b': 返回上一个patch")
        print("- 's': 保存当前进度")
        print("- 'q': 退出并保存")
        print("- 'a': 自动选择(使用第一张图像)")
        print("- 'r': 刷新显示")
        print("- ESC键: 跳过当前patch")
        print("- 按键后，实时结果窗口将显示拼接进度")
        
        try:
            for row in range(self.current_patch_row, self.total_rows):
                start_col = self.current_patch_col if row == self.current_patch_row else 0
                
                for col in range(start_col, self.total_cols):
                    self.current_patch_row = row
                    self.current_patch_col = col
                    
                    # 显示patch选项
                    patches, coords, window_name = self.show_patch_options(row, col)
                    
                    while True:
                        print(f"选择patch源图像 (1-{len(self.source_images)}), 或输入命令: ", end='')
                        key = cv2.waitKey(0) & 0xFF
                        
                        if key == ord('q'):  # 退出
                            cv2.destroyAllWindows()
                            self.save_current_result()
                            return
                        elif key == ord('s'):  # 保存进度
                            self.save_progress()
                            print("进度已保存!")
                            continue
                        elif key == ord('r'):  # 刷新显示
                            self.update_result_display(row, col)
                            print("显示已刷新!")
                            continue
                        elif key == ord('b'):  # 返回上一个
                            cv2.destroyWindow(window_name)
                            if col > 0:
                                self.current_patch_col = col - 1
                                col = self.current_patch_col - 1  # 会被外层循环+1
                            elif row > 0:
                                self.current_patch_row = row - 1
                                self.current_patch_col = self.total_cols - 1
                                row = self.current_patch_row - 1  # 会被外层循环+1
                                col = self.total_cols  # 会被外层循环重置
                            break
                        elif key == ord('a'):  # 自动选择第一张
                            selected_idx = 0
                            self.place_patch(patches[selected_idx], coords[selected_idx])
                            if self.patches_grid is not None:
                                self.patches_grid[row, col] = selected_idx + 1
                            print(f"自动选择: {self.source_names[selected_idx]}")
                            self.update_result_display(row, col)
                            cv2.destroyWindow(window_name)
                            break
                        elif key == 27:  # ESC - 跳过
                            print("跳过当前patch")
                            cv2.destroyWindow(window_name)
                            break
                        elif ord('1') <= key <= ord('9'):  # 数字选择
                            selected_idx = key - ord('1')
                            if selected_idx < len(patches):
                                self.place_patch(patches[selected_idx], coords[selected_idx])
                                if self.patches_grid is not None:
                                    self.patches_grid[row, col] = selected_idx + 1
                                print(f"已选择: {self.source_names[selected_idx]}")
                                self.update_result_display(row, col)
                                cv2.destroyWindow(window_name)
                                break
                            else:
                                print("无效选择!")
                    
                    if key == ord('b') and col == 0 and row > 0:
                        break  # 跳出列循环，回到上一行
            
            # 完成所有patch选择
            print("\n所有patch选择完成!")
            self.update_result_display()  # 最终显示，不显示边框
            print("按任意键关闭窗口...")
            cv2.waitKey(0)
        
        except KeyboardInterrupt:
            print("\n用户中断操作")
        
        finally:
            cv2.destroyAllWindows()
            self.save_current_result()
    
    def save_current_result(self, filename="interactive_result.png"):
        """保存当前拼接结果"""
        if self.result_image is not None:
            cv2.imwrite(filename, self.result_image)
            print(f"拼接结果已保存为: {filename}")
            
            # 也保存最终的选择记录
            self.save_progress("final_patch_selections.txt")
        else:
            print("没有结果可保存")


def main():
    """主函数"""
    print("=== 交互式图像拼接器 ===")
    
    # 默认源图像文件
    default_images = [
        'avg_background_vfov12hfov60_1.png',
        'avg_background_vfov12hfov60_2.png',
        'avg_background_vfov12hfov60_3.png'
    ]
    
    # 检查文件是否存在
    available_images = []
    for img_path in default_images:
        if os.path.exists(img_path):
            available_images.append(img_path)
        else:
            print(f"警告: 找不到文件 {img_path}")
    
    if len(available_images) < 2:
        print("错误: 至少需要2张源图像!")
        return
    
    print(f"找到 {len(available_images)} 张源图像:")
    for i, img in enumerate(available_images):
        print(f"  {i+1}. {img}")
    
    # 设置patch大小
    print(f"\n请输入patch大小 (默认: 64x64): ", end='')
    size_input = input().strip()
    if size_input:
        try:
            if 'x' in size_input:
                w, h = map(int, size_input.split('x'))
                patch_size = (w, h)
            else:
                size = int(size_input)
                patch_size = (size, size)
        except:
            print("输入格式错误，使用默认大小 64x64")
            patch_size = (64, 64)
    else:
        patch_size = (64, 64)
    
    print(f"Patch大小设置为: {patch_size[0]}x{patch_size[1]}")
    
    try:
        # 创建交互式选择器
        selector = InteractivePatchSelector(available_images, patch_size=patch_size)
        
        # 运行交互式选择
        selector.run_interactive_selection()
        
        print("拼接完成!")
        
    except Exception as e:
        print(f"运行时错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


# 你能否写一个脚本，就是对于我们最终想生成的目标图片，把他分成一个个小patch，从左到右，从上到下，我去从三张source image去选择，最终拼成一张好的图像，写一个可以交互的脚本