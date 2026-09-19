"""服务端分页（/api/category limit/offset）回归测试。

背景（2026-09-06）：`/api/category latest` 一次返回全部 16408 张（4.2MB JSON），
前端渲染虽分页但数据层全量拉——首屏慢，且 agent-browser 内核对 >4MiB 响应会
网络层挂死。改造：search_by_category 支持 limit/offset 切片（hidden 在切片前
剔除保证 total/offset 口径一致），前端首屏一页 + 后台静默补页。

锁定行为：
  1. limit 切片正确、has_more 边界正确、跨页无重叠无遗漏；
  2. hidden（过滤表/回收站）在切片前剔除，total=可见数，且不出现在任何页；
  3. 不传 limit 的旧调用方（首页 slim、回收站）响应结构完全不变。

隔离方式：monkeypatch svr.DB 指向临时库，不碰真实数据库。
"""
import os
import sqlite3
import tempfile
import unittest
import importlib.util

SVR_PATH = os.environ.get("FF_SERVER_PATH") or str(Path(__file__).resolve().parent / "server.py")
_spec = importlib.util.spec_from_file_location("family_memory_svr", SVR_PATH)
svr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svr)


def _make_db(path, n=25):
    """建最小 schema + n 张 2026-01 照片（a001..aNNN，时间递增），其中 a001 进过滤表、
    a002 进回收站，二者都应从 latest 可见集剔除。"""
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE media_asset(asset_id TEXT PRIMARY KEY, capture_time TEXT,
                                 media_type TEXT DEFAULT 'photo', width INTEGER, height INTEGER);
        CREATE TABLE asset_geo_v0(asset_id TEXT PRIMARY KEY, region TEXT, province TEXT);
        CREATE TABLE media_file(asset_id TEXT, source_id TEXT, availability TEXT);
        CREATE TABLE source(source_id TEXT PRIMARY KEY, owner_label TEXT);
        CREATE TABLE face_instance_v0(asset_id TEXT, bbox_json TEXT);
        CREATE TABLE asset_filter_v0(asset_id TEXT, reason TEXT);
        CREATE TABLE recycle_v0(asset_id TEXT, created_at TEXT);
        CREATE TABLE user_category_v0(category_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE user_category_member_v0(category_id TEXT, asset_id TEXT);
    """)
    for i in range(1, n + 1):
        aid = f"a{i:03d}"
        con.execute("INSERT INTO media_asset(asset_id,capture_time) VALUES(?,?)",
                    (aid, f"2026-01-{i:02d}T10:00:00"))
        con.execute("INSERT INTO media_file(asset_id,source_id,availability) VALUES(?,?,?)",
                    (aid, "s1", "original"))
    con.execute("INSERT INTO source VALUES('s1','手机A')")
    con.execute("INSERT INTO asset_filter_v0 VALUES('a001','SCREENSHOT_FILENAME')")
    con.execute("INSERT INTO recycle_v0 VALUES('a002','2026-09-06T00:00:00')")
    con.commit()
    con.close()


class CategoryPaginationTests(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.tmp = path
        self._old_db = svr.DB
        svr.DB = self.tmp
        _make_db(self.tmp)

    def tearDown(self):
        svr.DB = self._old_db
        try:
            os.unlink(self.tmp)
        except OSError:
            pass

    def _latest(self, **kw):
        return svr.search_by_category("latest", "", kw.get("order"), offset=kw.get("offset"),
                                      limit=kw.get("limit"))

    def test_paged_shape_and_total(self):
        """limit 切片：total=可见数（hidden 剔除后 23），首页 10 条，has_more=True。"""
        d = self._latest(limit=10)
        self.assertEqual(d["total"], 23, "25 张中 a001(过滤)/a002(回收站) 应剔除")
        self.assertEqual(len(d["assets"]), 10)
        self.assertEqual(d["offset"], 0)
        self.assertTrue(d["has_more"])
        self.assertTrue(all(not a["hidden"] for a in d["assets"]),
                        "分页路径 hidden 条目已在服务端剔除，返回项不应再有 hidden=True")

    def test_pages_no_overlap_no_gap(self):
        """三页拼起来 == 全量可见集（顺序一致，不重不漏）。"""
        full = self._latest(order="desc", limit=4000)["assets"]
        p1 = self._latest(order="desc", limit=10)["assets"]
        p2 = self._latest(order="desc", offset=10, limit=10)["assets"]
        p3 = self._latest(order="desc", offset=20, limit=10)["assets"]
        self.assertFalse(self._latest(order="desc", offset=20, limit=10)["has_more"],
                         "末页 has_more 必须为 False")
        self.assertEqual([a["id"] for a in p1 + p2 + p3], [a["id"] for a in full])

    def test_hidden_absent_from_every_page(self):
        """被过滤(a001)与回收站(a002)的照片不出现在任何分页窗口。"""
        for off in range(0, 23, 10):
            ids = {a["id"] for a in self._latest(offset=off, limit=10)["assets"]}
            self.assertNotIn("a001", ids)
            self.assertNotIn("a002", ids)

    def test_legacy_call_unchanged(self):
        """不传 limit：响应无 total/offset/has_more 字段；回收站在服务端排除（旧行为），
        过滤表 hidden 行仍随行返回（由前端剔除）。"""
        d = self._latest()
        self.assertNotIn("total", d)
        self.assertNotIn("has_more", d)
        self.assertEqual(len(d["assets"]), 24, "25 - 1 回收站(a002) 服务端排除；过滤表 hidden 保留")
        ids = {a["id"] for a in d["assets"]}
        self.assertNotIn("a002", ids)
        self.assertIn("a001", ids)
        self.assertTrue(any(a.get("hidden") for a in d["assets"]))

    def test_limit_clamped(self):
        """limit 上限 4000（防止恶意超大值）；0/负值视为未分页回落旧行为。
        本库仅 23 张可见，clamp 后一页取完 has_more=False 属预期。"""
        d = self._latest(limit=99999)
        self.assertEqual(d["limit"], 4000)
        self.assertEqual(len(d["assets"]), 23)
        self.assertFalse(d["has_more"])
        d0 = self._latest(limit=0)
        self.assertNotIn("total", d0, "limit<=0 应回落为旧行为")

    def test_offset_beyond_total(self):
        d = self._latest(offset=100, limit=10)
        self.assertEqual(d["assets"], [])
        self.assertFalse(d["has_more"])
        self.assertEqual(d["total"], 23)


if __name__ == "__main__":
    unittest.main()
