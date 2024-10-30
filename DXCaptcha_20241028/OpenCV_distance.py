import cv2
import numpy as np
from PIL import Image
from pathlib import Path

def imshow(img, winname='test', delay=0):
    """使用OpenCV显示图片"""
    cv2.imshow(winname, img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def cv2_open(img, flag=None):
    """将不同类型的图片统一转换为OpenCV格式"""
    if isinstance(img, bytes):
        img = cv2.imdecode(np.frombuffer(img, dtype=np.uint8), 1)
    elif isinstance(img, (str, Path)):
        img = cv2.imread(str(img))
    elif isinstance(img, np.ndarray):
        img = img
    elif isinstance(img, Image.Image):
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    else:
        raise ValueError(f'无法解析的图片类型: {type(img)}')
    if flag is not None:
        img = cv2.cvtColor(img, flag)
    return img

def get_distance(bg_img, slice_img):
    """核心逻辑，完成背景图与滑块之间的匹配计算"""
    # 读取两张图片，在读取图像的同时进行灰度转换
    tp_gray = cv2_open(slice_img, flag=cv2.COLOR_BGR2GRAY)
    bg_gray = cv2_open(bg_img, flag=cv2.COLOR_BGR2GRAY)

    # 对背景图进行非局部均值去噪，能够同时保留边缘和细节
    bg_shift = cv2.fastNlMeansDenoising(bg_gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    # 边缘检测
    tp_gray = cv2.Canny(tp_gray, 255, 255)
    bg_gray = cv2.Canny(bg_shift, 255, 255)

    # 模板匹配
    result = cv2.matchTemplate(bg_gray, tp_gray, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)  # 解析匹配结果
    # if save_path or im_show:
    #     # 需要绘制的方框高度和宽度
    #     tp_height, tp_width = tp_gray.shape[:2]
    #     # 矩形左上角点位置
    #     x, y = max_loc
    #     # 矩形右下角点位置
    #     _x, _y = x + tp_width, y + tp_height
    #     # 绘制矩形
    #     bg_img = cv2_open(bg_img)
    #     cv2.rectangle(bg_img, (x, y), (_x, _y), (0, 0, 255), 2)
    #     # 保存缺口识别结果到背景图
    #     if save_path:
    #         save_path = Path(save_path).resolve()
    #         save_path = save_path.parent / f"{save_path.stem}{save_path.suffix}"
    #         save_path = save_path.__str__()
    #         cv2.imwrite(save_path, bg_img)
    #     # 显示缺口识别结果
    #     if im_show:
    #         imshow(bg_img)
    return max_loc[0]

with open("./bg_img.png", "rb") as f:
    bg_img = f.read()
with open("./slider_img.webp", "rb") as f:
    slice_img = f.read()
distance = get_distance(bg_img, slice_img)
print(distance)
