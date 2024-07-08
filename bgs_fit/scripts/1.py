import cv2

image_path = "../../../data/0003/image.png"  # 替换为你的图片路径

image = cv2.imread(image_path)
print(f"image size: {image.shape}")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 缩小图像
scale_factor = 0.5  # 缩小比例
small_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

# 使用 selectROI 在缩小后的图像上选择 ROI
roi = cv2.selectROI('Select ROI', small_image)
cv2.destroyWindow('Select ROI')