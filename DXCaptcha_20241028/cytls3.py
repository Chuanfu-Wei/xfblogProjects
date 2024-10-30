import json
import requests
import json as jn
from urllib.parse import urlencode
import http.cookies
import urllib
from urllib.parse import urlparse

ud = 'http://8.211.131.96:8888/tls'

def put(url, headers=None, proxies=None, data=None, json=None, cookies=None, timeout=20, http2=False,
        allow_redirects=True):

    if headers is None:
        headers = {}
    if cookies is not None:
        headers["Cookie"] = '; '.join([f'{k}={v}' for k, v in cookies.items()])
    for k, v in headers.items():
        if v is None or v == '':
            headers[k] = ' '

    if proxies is None:
        dt = {
            "appid": "kq0x5hb2tcn6mlhyn76hb0oc2uopa25g",
            "method": "PUT",
            "url": url,
            "headers": headers,
            "userAgent": headers.get('User-Agent') or headers.get('user-agent', ''),
            "body": json if json is not None else data,
            "redirect": allow_redirects
        }
    else:
        parsed = urlparse(proxies['http'])
        # 获取用户名、密码、IP 和端口
        username = parsed.username
        password = parsed.password
        ip = parsed.hostname
        port = parsed.port
        dt = {
            "appid": "kq0x5hb2tcn6mlhyn76hb0oc2uopa25g",
            "method": "PUT",
            "url": url,
            "headers": headers,
            "userAgent": headers.get('User-Agent') or headers.get('user-agent', ''),
            "body": json if json is not None else data,
            "proxyIp": ip,
            "proxyPort": port,
            "proxyAuth": f'{username}:{password}',
            "redirect": allow_redirects
        }

    response = requests.post(ud, json=dt, timeout=timeout, allow_redirects=True)
    set_cookies = response.headers.get('Set-Cookie')
    if set_cookies:
        parsed_cookies = http.cookies.SimpleCookie()
        parsed_cookies.load(set_cookies)
        response.cookies = parsed_cookies

    return response

def get(url, headers=None, proxies=None, params=None, cookies=None, timeout=20, http2=False, allow_redirects=True):
    if params is not None:
        url = url.split("?")[0]
        url = f'{url}?{urlencode(params)}'

    if headers is None:
        headers = {}
    if cookies is not None:
        headers["Cookie"] = '; '.join([f'{k}={v}' for k, v in cookies.items()])
    for k, v in headers.items():
        if v is None or v == '':
            headers[k] = ' '

    if proxies is None:
        dt = {
            "appid": "kq0x5hb2tcn6mlhyn76hb0oc2uopa25g",
            "method": "GET",
            "url": url,
            "headers": headers,
            "userAgent": headers.get('User-Agent') or headers.get('user-agent', ''),
            "body": '',
            "redirect": allow_redirects
        }
    else:
        parsed = urlparse(proxies['http'])
        # 获取用户名、密码、IP 和端口
        username = parsed.username
        password = parsed.password
        ip = parsed.hostname
        port = parsed.port
        dt = {
            "appid": "kq0x5hb2tcn6mlhyn76hb0oc2uopa25g",
            "method": "GET",
            "url": url,
            "headers": headers,
            "userAgent": headers.get('User-Agent') or headers.get('user-agent', ''),
            "body": '',
            "proxyIp": ip,
            "proxyPort": port,
            "proxyAuth": f'{username}:{password}',
            "redirect": allow_redirects
        }

    response = requests.post(ud, json=dt, timeout=timeout, allow_redirects=True)
    set_cookies = response.headers.get('Set-Cookie')
    if set_cookies:
        parsed_cookies = http.cookies.SimpleCookie()
        parsed_cookies.load(set_cookies)
        response.cookies = parsed_cookies

    return response

def post(url, headers=None, proxies=None, params=None, cookies=None, json=None, data=None, timeout=40, http2=False,
         allow_redirects=True):
    if params is not None:
        url = url.split("?")[0]
        url = f'{url}?{urlencode(params)}'
    lower_header = {}
    if headers is None:
        headers = {}
    if cookies is not None:
        lower_header["cookie"] = '; '.join([f'{k}={v}' for k, v in cookies.items()])
    for k, v in headers.items():
        if v is None or v == '':
            lower_header[k.lower()] = ' '
        else:
            lower_header[k.lower()] = v

    if isinstance(data, dict):
        data = urllib.parse.urlencode(data)

    postData = ""
    if data is not None:
        postData = data
    if json is not None:
        postData = jn.dumps(json)
        lower_header['content-type'] = 'application/json'

    if proxies is None:
        dt = {
            "appid": "kq0x5hb2tcn6mlhyn76hb0oc2uopa25g",
            "method": "POST",
            "url": url,
            "headers": lower_header,
            "userAgent": headers.get('User-Agent') or headers.get('user-agent', ''),
            "body": postData,
        }
    else:
        parsed = urlparse(proxies['http'])
        # 获取用户名、密码、IP 和端口
        username = parsed.username
        password = parsed.password
        ip = parsed.hostname
        port = parsed.port

        dt = {
            "appid": "kq0x5hb2tcn6mlhyn76hb0oc2uopa25g",
            "method": "POST",
            "url": url,
            "headers": lower_header,
            "userAgent": headers.get('User-Agent') or headers.get('user-agent', ''),
            "body": postData,
            "proxyIp": ip,
            "proxyPort": port,
            "proxyAuth": f'{username}:{password}',
            "redirect": allow_redirects
        }

    response = requests.post(ud, json=dt, timeout=timeout, allow_redirects=True)
    set_cookies = response.headers.get('Set-Cookie')
    if set_cookies:
        parsed_cookies = http.cookies.SimpleCookie()
        parsed_cookies.load(set_cookies)

        response.cookies = parsed_cookies
    return response

class Response:
    def __init__(self, httpCode, result, headers, cookies):
        self.status_code = httpCode
        self.text = result
        self.headers = headers
        self.cookies = cookies
        self._json_cache = None  # 用于缓存解析后的 JSON

    def json(self):
        # 如果已经解析过 JSON，则直接返回缓存结果
        if self._json_cache is not None:
            return self._json_cache
        self._json_cache = json.loads(self.text)  # 将文本解析为 JSON
        return self._json_cache

class Session:
    def __init__(self):
        self.headers = {}
        self.cookies = {}
        self.proxies = None

    def get(self, url, headers=None, proxies=None, params=None, cookies=None, timeout=20, allow_redirects=True):
        # 合并 session 中的 cookies 和传入的 cookies
        merged_cookies = self.cookies.copy()
        if cookies is not None:
            merged_cookies.update(cookies)

        # 调用封装的 get 方法
        response = get(url, headers=headers or self.headers, proxies=proxies or self.proxies, params=params,
                       cookies=merged_cookies, timeout=timeout, allow_redirects=allow_redirects)

        # 更新 session cookies
        if response.json()['cookies']:
            self.update_cookies(response.json()['cookies'])

        # 返回自定义的 Response 对象
        return Response(
            response.json()['httpCode'],
            response.json()['result'],
            response.json()['headers'],
            response.json()['cookies'],
        )

    def post(self, url, headers=None, proxies=None, params=None, cookies=None, json=None, data=None, timeout=40,
             allow_redirects=True):
        # 合并 session 中的 cookies 和传入的 cookies
        merged_cookies = self.cookies.copy()
        if cookies is not None:
            merged_cookies.update(cookies)

        # 调用封装的 post 方法
        response = post(url, headers=headers or self.headers, proxies=proxies or self.proxies, params=params,
                        cookies=merged_cookies, json=json, data=data, timeout=timeout, allow_redirects=allow_redirects)

        # 更新 session cookies
        if response.json()['cookies']:
            self.update_cookies(response.json()['cookies'])

        # 返回自定义的 Response 对象
        return Response(
            response.json()['httpCode'],
            response.json()['result'],
            response.json()['headers'],
            response.json()['cookies'],
        )

    def put(self, url, headers=None, proxies=None, data=None, json=None, cookies=None, timeout=20,
            allow_redirects=True):
        # 合并 session 中的 cookies 和传入的 cookies
        merged_cookies = self.cookies.copy()
        if cookies is not None:
            merged_cookies.update(cookies)

        # 调用封装的 put 方法
        response = put(url, headers=headers or self.headers, proxies=proxies or self.proxies,
                       cookies=merged_cookies, data=data, json=json, timeout=timeout, allow_redirects=allow_redirects)

        # 更新 session cookies
        if response.json()['cookies']:
            self.update_cookies(response.json()['cookies'])

        # 返回自定义的 Response 对象
        return Response(
            response.json()['httpCode'],
            response.json()['result'],
            response.json()['headers'],
            response.json()['cookies'],
        )

    def update_cookies(self, new_cookies):
        """将新收到的 cookies 合并到 session 中"""
        for cookie in new_cookies:
            if '=' in cookie:  # 防止无效的cookie格式
                key, value = cookie.split("=", 1)
                self.cookies[key] = value.rstrip(';')

    def set_headers(self, headers):
        """设置全局 headers"""
        self.headers.update(headers)

    def set_proxies(self, proxies):
        """设置代理"""
        self.proxies = proxies

    def clear_cookies(self):
        """清除所有 session cookies"""
        self.cookies.clear()

    def close(self):
        """关闭 session（目前无具体资源需要关闭）"""
        pass
