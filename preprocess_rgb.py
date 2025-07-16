import cv2
import numpy as np
from matplotlib import pyplot as plt

def show_resized_image(title, image, scale=3):
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    cv2.imshow(title, resized)
    
# 假设 'test_image.jpg' 是您的那张深色图像
try:
    image_underwater = cv2.imread('./cropped_cam_left.png')
    if image_underwater is None:
        raise FileNotFoundError("图片未找到，请检查路径。")

    
    b, g, r = cv2.split(image_underwater)
    # 自定义权重，注意数据类型转换以避免溢出
    gray_custom = 0.4 * b + 0.6 * g + 0.0 * r 
    gray_custom = gray_custom.astype(np.uint8)
    show_resized_image("Custom Gray Image", gray_custom)

    gray_standard = cv2.cvtColor(image_underwater, cv2.COLOR_BGR2GRAY)
    show_resized_image("Standard Gray Image", gray_standard)

    # 2. **关键步骤：使用 CLAHE 增强对比度**
    # 创建 CLAHE 对象 (clipLimit 控制对比度放大的限制，tileGridSize 定义了网格的大小)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray_custom = clahe.apply(gray_custom)
    enhanced_gray_standard = clahe.apply(gray_standard)
    # 3. (可选) 此时可以尝试轻微模糊，以去除 CLAHE 可能引入的噪声
    # blurred_image = cv2.GaussianBlur(enhanced_gray, (3, 3), 0)
    # 对于此特定图像，我们先跳过模糊，直接在增强后的图像上操作

    # 5. 显示对比结果

    
    show_resized_image("enhanced_gray_custom.png", enhanced_gray_custom)
    show_resized_image("enhanced_gray_standard.png", enhanced_gray_standard)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except Exception as e:
    print(f"发生错误: {e}")