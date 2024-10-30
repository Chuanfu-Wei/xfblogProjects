"""
@Date    : 2024/10/28 下午10:38
@Author  : https://xfblog.cn
@Project : main.py
@IDE     : PyCharm --> xfblogProjects
@Spider  : https://www.dingxiang-inc.com/business/captcha
"""
import cytls3
import time
from loguru import logger
from faker import Faker
import random
import string
import requests
from get_canvas import restore_image
import os

class DXCaptcha:
    def __init__(self):
        self.session = cytls3.Session()
        self.user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36'
        self.faker = Faker()

    @staticmethod
    def random_str_num(length=6):
        """随机生成数字字符串"""
        return ''.join([str(random.randint(0, 9)) for _ in range(length)])

    @staticmethod
    def random_str_num_en(length=40):
        """随机生成小写英文和数字"""
        characters = string.ascii_lowercase + string.digits
        random_string = ''.join(random.choices(characters, k=length))
        return random_string

    def udid_c1(self):
        """第一步 --> 校验请求时间，获取校验参数"""
        try:
            response = self.session.get(
                'https://constid.dingxiang-inc.com/udid/c1',
                headers={
                    'Accept': 'application/json, text/plain, */*',
                    'Accept-Language': 'zh-CN,zh;q=0.9',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Origin': 'https://www.dingxiang-inc.com',
                    'Param': 'xxx',
                    'Pragma': 'no-cache',
                    'Referer': 'https://www.dingxiang-inc.com/',
                    'User-Agent': self.user_agent,
                },
                params={
                    '_t': str(int(time.time() * 1000))[-6:-1],
                }
            )
            if response.status_code == 200:
                result = response.json()
                params_c = result['data']
                logger.info(f"获取params_c：{params_c}")
                return params_c
            else:
                raise Exception(f"响应 {response.status_code} 状态码")
        except Exception as e:
            error_info = f"获取 params_c 失败......{e}"
            logger.exception(error_info)

    def get_img_info(self, params_c):
        """第二步 --> 获取缺口图片与背景图片，和滑动相关参数"""
        try:
            response = self.session.get(
                'https://cap.dingxiang-inc.com/api/a',
                headers={
                    'Accept': '*/*',
                    'Accept-Language': 'zh-CN,zh;q=0.9',
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'Origin': 'https://www.dingxiang-inc.com',
                    'Pragma': 'no-cache',
                    'Referer': 'https://www.dingxiang-inc.com/',
                    'User-Agent': self.user_agent,
                },
                params={
                    'w': '380',
                    'h': '165',
                    's': '50',
                    'ak': '99de95ad1f23597c23b3558d932ded3c',
                    'c': params_c,
                    'jsv': '5.1.49',
                    'aid': f'dx-{time.time() * 1000}-{self.random_str_num(8)}-3',
                    'wp': '1',
                    'de': '0',
                    'uid': '',
                    'lf': '0',
                    'tpc': '',
                    't': '4005F2EEAE84258CA8345CAE40A500DA475782FBC83422D6CE9AC3B6C2FFBEFF4A74739161BEFFDD2AE22B5475F4B6F7F6F02DCDA1D403250FC31F7307B4D49483EDCBBC96A7969BF4F2B528C378C8D5',
                    'cid': '64817763',
                    '_r': random.random(),
                },
            )
            if response.status_code == 200:
                result = response.json()
                bg_img = result['p1']
                slider_img = result['p2']
                logger.info(f"获取图片：{bg_img} {slider_img}")
                with open(bg_img.split('/')[-1], "wb") as f:
                    f.write(requests.get("https://static4.dingxiang-inc.com/picture" + bg_img).content)
                with open("./slider_img.webp", "wb") as f:
                    f.write(requests.get("https://static4.dingxiang-inc.com/picture" + slider_img).content)
                # 调用还原乱序图片 api 还原背景图片
                restore_image(bg_img.split('/')[-1],"bg_img.png")
                os.remove(bg_img.split('/')[-1])
            else:
                raise Exception(f"响应 {response.status_code} 状态码")
        except Exception as e:
            error_info = f"获取图片失败......{e}"
            logger.exception(error_info)

    def main(self):
        params_c = self.udid_c1()
        self.get_img_info(params_c)

if __name__ == '__main__':
    dx = DXCaptcha()
    dx.main()
