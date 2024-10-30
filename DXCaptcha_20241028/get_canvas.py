"""
@Date    : 2024/10/28 下午10:35
@Author  : https://xfblog.cn
@Project : get_canvas.py
@IDE     : PyCharm --> xfblogProjects
@Spider  : 
"""
import execjs
import math
from PIL import Image

def draw_image(canvas, source_img, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight):
    """模拟JS中的 c.drawImage(image, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
    :param canvas: 在循环开始之前创建一个空的 canvas，以便所有块都在同一个画布上绘制
    :param source_img: 传入源图片对象，避免在每次循环中重新加载图片，提升效率
    :param sx, sy: 源图像的裁剪起始位置（左上角）
    :param sWidth, sHeight: 裁剪区域的宽度和高度
    :param dx, dy: 目标画布上的绘制位置（左上角）
    :param dWidth, dHeight: 绘制到目标画布上的尺寸
    :return: None
    """
    # 裁剪出指定区域 (sx, sy, sx + sWidth, sy + sHeight)
    cropped_img = source_img.crop((sx, sy, sx + sWidth, sy + sHeight))
    # 缩放裁剪后的图像到指定尺寸
    resized_img = cropped_img.resize((dWidth, dHeight))
    # 在目标画布上粘贴缩放后的图像
    canvas.paste(resized_img, (dx, dy))

def restore_image(bg_img_path, save_path):
    # 加载原始图片
    image = Image.open(bg_img_path)
    img_width, img_height = image.size  # 获取图片的宽度和高度

    # 创建目标画布，大小与原始图片相同
    canvas = Image.new('RGBA', (img_width, img_height))

    # ---------------- 逆向逻辑：模拟 t 数组和乱序逻辑 ----------------
    arr = execjs.compile(open('./get_canvas.js').read()).call('f1', bg_img_path.split('/')[-1].split('.')[0])
    s = math.floor(400 / len(arr))

    def drow_image(n, r):
        t = n * s
        i = s
        draw_image(canvas, image, t, 0, i, img_height, r * s, 0, i, img_height)

    # 遍历 arr 数组，并在同一个画布上绘制所有块
    for index, item in enumerate(arr):
        drow_image(item, index)

    # 最终保存结果
    canvas.save(save_path)
