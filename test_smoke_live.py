#!/usr/bin/env python3
"""在线服务集成冒烟测试（2026-09-04 新增）。

打真实的 127.0.0.1:8788 服务，覆盖关键链路：
静态页 / 路径穿越 404 / 分类 API / 预览图 / 原片 Range / /view 页面。
服务不在线时整组 skip，不会误报失败。

运行：python3 -m unittest test_smoke_live -v
"""
import http.client
import json
import unittest
import urllib.request

BASE = "127.0.0.1:8788"


def _server_up():
    try:
        urllib.request.urlopen(f"http://{BASE}/", timeout=3).read(64)
        return True
    except Exception:
        return False


@unittest.skipUnless(_server_up(), "8788 服务不在线，跳过集成冒烟")
class LiveSmokeTest(unittest.TestCase):
    def _get(self, path, headers=None):
        req = urllib.request.Request(f"http://{BASE}{path}", headers=headers or {})
        return urllib.request.urlopen(req, timeout=15)

    def _post_json(self, path, payload):
        req = urllib.request.Request(
            f"http://{BASE}{path}", method="POST",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    def _any_asset_id(self, media_type=None):
        d = self._post_json("/api/category", {"cat": "latest", "value": "", "order": "desc"})
        assets = d.get("assets") or []
        if media_type:
            assets = [a for a in assets if a.get("type") == media_type]
        self.assertTrue(assets, f"总库里取不到 type={media_type or '任意'} 的资产")
        return assets[0].get("id") or assets[0].get("asset_id")

    def test_homepage_200(self):
        r = self._get("/")
        self.assertEqual(r.status, 200)
        self.assertIn("text/html", r.headers.get("Content-Type", ""))

    def test_library_page_200(self):
        r = self._get("/library.html")
        self.assertEqual(r.status, 200)
        self.assertIn("家庭影像总库", r.read().decode("utf-8", "ignore"))

    def test_path_traversal_raw_request_404(self):
        """http.client 发原始路径（不做客户端归一化），穿越必须 404。"""
        for evil in ("/../server.py", "/../../etc/passwd", "/..%2f..%2fserver.py"):
            con = http.client.HTTPConnection(BASE, timeout=10)
            con.putrequest("GET", evil, skip_accept_encoding=True)
            con.endheaders()
            resp = con.getresponse()
            resp.read()
            self.assertEqual(resp.status, 404, f"{evil} 应 404，实际 {resp.status}")
            con.close()

    def test_category_api_returns_assets(self):
        d = self._post_json("/api/category", {"cat": "latest", "value": "", "order": "desc"})
        self.assertIn("assets", d)
        self.assertGreater(len(d["assets"]), 0)

    def test_preview_image_200(self):
        # /preview 只服务照片（视频走 /orig + 封面管线），必须取 photo 类型取样
        aid = self._any_asset_id("photo")
        r = self._get(f"/preview?asset={aid}")
        self.assertEqual(r.status, 200)
        ctype = r.headers.get("Content-Type", "")
        self.assertTrue(ctype.startswith("image/") or "video" in ctype or "octet-stream" in ctype,
                        f"意外的 Content-Type: {ctype}")
        self.assertGreater(int(r.headers.get("Content-Length", "1") or 1), 0)

    def test_orig_range_request_206(self):
        aid = self._any_asset_id()
        req = urllib.request.Request(f"http://{BASE}/orig?asset={aid}",
                                     headers={"Range": "bytes=0-1023"})
        try:
            r = urllib.request.urlopen(req, timeout=30)
        except urllib.error.HTTPError as e:
            self.fail(f"Range 请求失败: {e.code}")
        self.assertEqual(r.status, 206)
        cr = r.headers.get("Content-Range", "")
        self.assertTrue(cr.startswith("bytes 0-1023/"), f"Content-Range 异常: {cr}")
        self.assertEqual(len(r.read()), 1024)

    def test_view_page_renders_and_quotes_aid(self):
        aid = self._any_asset_id()
        body = self._get(f"/view?asset={aid}").read().decode("utf-8", "ignore")
        # aid 本身是安全字符时 quote 是恒等；关键是页面正常渲染且 URL 拼接存在
        self.assertTrue(f"asset={aid}" in body, "/view 页面未包含媒体 URL")
        # 未知 aid 必须 404（不渲染页面）
        try:
            self._get("/view?asset=no-such-asset-xyz")
            self.fail("未知 asset 应 404")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main()
