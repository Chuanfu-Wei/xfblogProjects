import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def imshow(img, winname='test', delay=0):
    """使用OpenCV显示图片"""
    cv2.imshow(winname, img)
    cv2.waitKey(delay)  # 等待按键
    cv2.destroyAllWindows()

def pil_to_cv2(img):
    """PIL图像转为OpenCV格式"""
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    return img

def bytes_to_cv2(img):
    """将二进制图片转换为OpenCV格式"""
    img_buffer_np = np.frombuffer(img, dtype=np.uint8)
    img_np = cv2.imdecode(img_buffer_np, 1)
    return img_np

def cv2_open(img, flag=None):
    """将不同类型的图片统一转换为OpenCV格式"""
    if isinstance(img, bytes):
        img = bytes_to_cv2(img)
    elif isinstance(img, (str, Path)):
        img = cv2.imread(str(img))
    elif isinstance(img, np.ndarray):
        img = img
    elif isinstance(img, Image.Image):
        img = pil_to_cv2(img)
    else:
        raise ValueError(f'无法解析的图片类型: {type(img)}')
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    return img

# 第一步：读取图片并展示原图
with open("./DXCaptcha_20241028/bg_img.png", "rb") as f:
    bg_img = f.read()
with open("./DXCaptcha_20241028/slider_img.webp", "rb") as f:
    slice_img = f.read()

bg = cv2_open(bg_img)
tp = cv2_open(slice_img)

# 展示背景图和滑块图
print("展示原始背景图：")
imshow(bg, 'Background')
print("展示原始滑块图：")
imshow(tp, 'Slider')

# 第二步：将两张图片都转为灰度图
tp_gray = cv2.cvtColor(tp, cv2.COLOR_BGR2GRAY)
bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
# 展示灰度化后的图片
print("展示灰度滑块图：")
imshow(tp_gray, 'Gray Slider')  # 展示灰度滑块
imshow(bg_gray, 'Gray Background')  # 展示灰度背景

# 第三步：对背景灰度图进行非局部均值去噪，能够同时保留边缘和细节
bg_shift = cv2.fastNlMeansDenoising(bg_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
# 对背景图进行金字塔均值漂移滤波处理
# bg_shift = cv2.pyrMeanShiftFiltering(bg_img, 5, 50)
# 对背景图进行高斯滤波处理
# bg_shift = cv2.GaussianBlur(bg_img, (5, 5), 0)
print("展示滤波后的背景图：")
imshow(bg_shift, 'Filtered Background')

def trackbar(x):
    """使用 OpenCV 创建一个窗口，并在该窗口中添加两个滑动条（trackbars），用于调整 Canny 边缘检测算法的阈值"""
    test1 = cv2.getTrackbarPos("test1", "cannyTest")
    test2 = cv2.getTrackbarPos("test2", "cannyTest")
    canny_img1 = cv2.Canny(tp_gray, test1, test2)
    canny_img2 = cv2.Canny(bg_shift, test1, test2)
    cv2.imshow("canny_img1", canny_img1)
    cv2.imshow("canny_img2", canny_img2)

cv2.namedWindow('cannyTest')
cv2.createTrackbar("test1", "cannyTest", 0, 255, trackbar)
cv2.createTrackbar("test2", "cannyTest", 0, 255, trackbar)

# 第四步：对滤波后的背景图和滑块进行边缘检测并展示
tp_edge = cv2.Canny(tp_gray, 255, 255)
bg_edge = cv2.Canny(bg_shift, 255, 255)
print("展示滑块边缘检测结果：")
imshow(tp_edge, 'Slider Edge')
print("展示背景图边缘检测结果：")
imshow(bg_edge, 'Background Edge')

# 第五步：使用模板匹配，并展示匹配结果
result = cv2.matchTemplate(bg_edge, tp_edge, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# 标出匹配位置
x, y = max_loc
tp_height, tp_width = tp_gray.shape[:2]
cv2.rectangle(bg, (x, y), (x + tp_width, y + tp_height), (0, 0, 255), 2)

print(f"匹配到的位置：x={x}, y={y}")
print("展示匹配后的背景图：")
imshow(bg, 'Matched Background')