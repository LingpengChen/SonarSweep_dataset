import torch
import numpy as np
# import pickle # REMOVED:不再需要 pickle
import cv2
import os
import glob

from .network_scunet import SCUNet as net

class SonarDenoiser:
    """
    高效处理单张声纳图像，将其转换为回波概率（EP）图像。
    - 初始化时加载模型并处理背景图。
    - process() 方法用于处理单张目标图。
    """
    def __init__(self, model_path: str, background_image: np.ndarray, mu: float = 15.0, epsilon: float = 0.01):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mu = mu
        self.epsilon = epsilon
        
        # 1. 加载和准备模型
        self.model = net(in_nc=1, config=[4,4,4,4,4,4,4], dim=64)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        
        # 2. 处理并存储去噪后的背景图 (Tensor形式)
        # background_image 是一个 0-255 范围的 numpy 数组
        with torch.no_grad():
            # _numpy_to_tensor 方法会处理归一化
            background_tensor = self._numpy_to_tensor(background_image)
            self.denoised_background_tensor = self.model(background_tensor)
        
        print(f"SonarDenoiser initialized on {self.device}. Background processed.")

    def _numpy_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """
        MODIFIED: 将 HxW NumPy 数组（0-255, uint8）转换为 1x1xHxW PyTorch Tensor（0-1, float32）。
        这是最核心的修改，确保输入模型的数据范围正确。
        """
        # 1. 确保数据类型为 float32
        img = img.astype(np.float32)
        # 2. 归一化：将 0-255 范围映射到 0-1 范围
        img /= 255.0
        # 3. 转换为 Tensor 并增加维度 (H, W) -> (1, 1, H, W)
        return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def process(self, target_image: np.ndarray) -> np.ndarray:
        """
        处理单张目标图像（0-255, uint8），返回最终的 EP 图像（0-1, float32）。
        """
        # --- 步骤 1: 去噪目标图 ---
        # _numpy_to_tensor 方法会自动处理归一化
        target_tensor = self._numpy_to_tensor(target_image)
        denoised_target_tensor = self.model(target_tensor)

        # --- 以下部分被注释掉了，如果需要背景扣除和概率转换，请取消注释 ---
        # --- 并确保 self.epsilon 和 self.mu 的值对于 0-1 范围是合适的 ---
        subtracted_tensor = (denoised_target_tensor - self.denoised_background_tensor).clamp_(min=0.0)
        # ep_tensor = torch.zeros_like(subtracted_tensor)
        # mask = subtracted_tensor >= self.epsilon
        # ep_tensor[mask] = 1.0 / (1.0 + torch.exp(-self.mu * subtracted_tensor[mask]))
        # return ep_tensor.squeeze().cpu().numpy()
        
        # --- 返回结果 ---
        # 模型输出已经是 0-1 范围的 float tensor
        float_tensor = subtracted_tensor.squeeze().cpu().numpy()
        final_denoised_image = (float_tensor.clip(0, 1) * 255).astype(np.uint8)
        
        return final_denoised_image

def compute_average_background(background_dir: str = "./background", img_format: str = "png") -> np.ndarray:
    """
    MODIFIED: 从文件夹中读取所有图像文件（如.png），并计算平均背景图。
    """
    search_pattern = os.path.join(background_dir, f'*.{img_format}') # MODIFIED: 查找指定格式的图像文件
    background_paths = glob.glob(search_pattern)

    if not background_paths:
        raise FileNotFoundError(f"No background images found with format '{img_format}' in directory '{background_dir}'")

    background_images = []
    print(f"Found {len(background_paths)} background files. Loading to compute average...")

    for path in background_paths:
        try:
            # MODIFIED: 使用 cv2.imread 读取8位灰度图
            # 得到的是一个 HxW 的 numpy 数组，数据类型为 uint8，范围 0-255
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Warning: Could not read image at '{path}'. Skipping.")
                continue
            
            # 加载图像并确保是 float32 类型以进行平均值计算
            background_images.append(img.astype(np.float32))

        except Exception as e:
            print(f"Warning: Could not process file '{path}': {e}. Skipping.")

    if not background_images:
        raise ValueError("No valid background images could be loaded from the provided paths.")

    # 使用 np.mean 沿着第一个轴（批次轴）计算平均值
    # np.stack 将列表中的2D图像堆叠成一个3D数组 (N, H, W)
    # 平均后的图像范围仍然是 0-255 的 float 类型
    average_background = np.mean(np.stack(background_images, axis=0), axis=0)
    
    print("Average background computed successfully.")
    # 返回的数组是 float32 类型，范围 0-255
    return average_background 

# --- 主程序入口 ---
if __name__ == '__main__':
    # --- MODIFIED: 配置路径（请确保路径指向图像文件而不是.pkl文件） ---
    MODEL_PATH = './model/scunet_gray_25.pth'
    
    # NEW: 假设你的目标图和背景图现在是.png格式
    TARGET_IMAGE_PATH = './test_image/1_img/1752220845.3731437.png' # 例如
    BACKGROUND_DATA_DIR = './test_image/background/'
    IMAGE_FORMAT = 'png' # 背景图的格式

    # MODIFIED: 计算平均背景图，现在从图像文件读取
    avg_background_np_float = compute_average_background(BACKGROUND_DATA_DIR, img_format=IMAGE_FORMAT)
    avg_background_np = np.clip(avg_background_np_float, 0, 255).astype(np.uint8)

    cv2.imshow('avg_background_np', avg_background_np)
   
    # --- 1. 初始化转换器 ---
    # avg_background_np 是一个范围0-255的float数组
    converter = SonarDenoiser(MODEL_PATH, avg_background_np)
    
    # --- 2. 在循环中处理目标图像 ---
    print("\n--- Simulating real-time processing loop ---")
    
    # MODIFIED: 使用cv2.imread加载目标图像
    # target_img_np 是一个范围0-255的uint8数组
    target_img_np = cv2.imread(TARGET_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    if target_img_np is None:
        raise FileNotFoundError(f"Target image not found at {TARGET_IMAGE_PATH}")

    # 调用 process 方法处理单张目标图
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    start_time.record()

    # converter.process 会接收 0-255 的图像，并在内部处理归一化
    final_denoised_image = converter.process(target_img_np)
    
    end_time.record()
    torch.cuda.synchronize()
    processing_time = start_time.elapsed_time(end_time)
    
    print(f"Single image processed in: {processing_time:.4f} ms")
    
    # --- 结果可视化 ---
    # MODIFIED: 调整可视化逻辑
    
    # 输入的原始图像是 uint8 (0-255)，可以直接显示
    cv2.imshow('Original Target Image', target_img_np)

    # 模型输出的图像是 float32 (0-1)，需要转换才能正确显示
    # (img * 255).astype(np.uint8) 是标准做法
    cv2.imshow('Final Denoised Image', final_denoised_image) # 标题也更新一下
    cv2.imwrite("denoised.png", final_denoised_image)
    print("\nDisplaying result. Press any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()