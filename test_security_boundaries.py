#!/usr/bin/env python3
"""安全边界与 DB 层单元测试（2026-09-04 新增）。

不依赖在线服务、不碰生产数据库：
- get_orig_path 用临时 SQLite + 临时文件测（monkeypatch server.DB）
- safe_static_path 纯函数测路径穿越防护
"""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

import server


class GetOrigPathTest(unittest.TestCase):
    """get_orig_path：只经 asset_id 查库取原片路径，是文件服务的核心安全闸门。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.db_path = str(self.dir / "test.db")
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE media_asset (asset_id TEXT PRIMARY KEY)")
        con.execute("CREATE TABLE media_file (asset_id TEXT, absolute_path TEXT, byte_size INTEGER)")
        con.commit()
        con.close()
        self._orig_db = server.DB
        server.DB = self.db_path

    def tearDown(self):
        server.DB = self._orig_db
        self._tmp.cleanup()

    def _add_asset(self, asset_id, files):
        """files: [(磁盘上真实创建的文件名, byte_size), ...]，返回 {文件名: 绝对路径}"""
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO media_asset VALUES (?)", (asset_id,))
        paths = {}
        for name, size in files:
            p = self.dir / name
            p.write_bytes(b"x" * min(size, 16))
            paths[name] = str(p)
            con.execute("INSERT INTO media_file VALUES (?,?,?)", (asset_id, str(p), size))
        con.commit()
        con.close()
        return paths

    def test_returns_largest_file_when_multiple(self):
        paths = self._add_asset("a1", [("small.jpg", 100), ("large.jpg", 9000)])
        self.assertEqual(server.get_orig_path("a1"), paths["large.jpg"])

    def test_unknown_asset_returns_none(self):
        self.assertIsNone(server.get_orig_path("no-such-asset"))

    def test_file_missing_on_disk_returns_none(self):
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO media_asset VALUES ('a2')")
        con.execute("INSERT INTO media_file VALUES ('a2','/nonexistent/ghost.jpg',100)")
        con.commit()
        con.close()
        self.assertIsNone(server.get_orig_path("a2"))

    def test_sql_injection_style_id_returns_none(self):
        """参数化查询兜底：注入形态 asset_id 必须查不到、不报错、不泄露任何行。"""
        self._add_asset("a3", [("real.jpg", 100)])
        for evil in ("' OR 1=1 --",
                     "a3' UNION SELECT absolute_path FROM media_file --",
                     "\" OR \"\"=\""):
            self.assertIsNone(server.get_orig_path(evil), evil)

    def test_largest_missing_falls_to_none_currently(self):
        """现状记录：最大文件已丢失时返回 None（不回退到较小副本）。
        若未来改为回退，此测试需同步更新。"""
        paths = self._add_asset("a4", [("small.jpg", 100)])
        con = sqlite3.connect(self.db_path)
        con.execute("INSERT INTO media_file VALUES ('a4','/nonexistent/big.jpg',9000)")
        con.commit()
        con.close()
        self.assertIsNone(server.get_orig_path("a4"))
        self.assertTrue(os.path.exists(paths["small.jpg"]))  # 小副本其实在，但当前不回退


class SafeStaticPathTest(unittest.TestCase):
    """safe_static_path：静态文件服务的路径穿越防护。"""

    STATIC_REAL = server.STATIC.resolve()

    def test_root_maps_to_index(self):
        self.assertEqual(server.safe_static_path("/"), self.STATIC_REAL / "index.html")

    def test_normal_file_inside_static(self):
        f = server.safe_static_path("/library.html")
        self.assertIsNotNone(f)
        self.assertTrue(str(f).startswith(str(self.STATIC_REAL)))
        self.assertTrue(f.exists())

    def test_dotdot_traversal_blocked(self):
        for evil in ("/../server.py",
                     "/../../etc/passwd",
                     "/../../../../../../etc/passwd",
                     "/sub/../../server.py"):
            result = server.safe_static_path(evil)
            # 安全不变量：要么 None，要么仍在 STATIC 内（不存在则由 404 兜底）
            if result is not None:
                self.assertTrue(str(result).startswith(str(self.STATIC_REAL)),
                                f"{evil} 越出 STATIC: {result}")
        # 经典穿越必须直接 None
        self.assertIsNone(server.safe_static_path("/../server.py"))
        self.assertIsNone(server.safe_static_path("/../../etc/passwd"))

    def test_encoded_traversal_not_decoded(self):
        """%2e%2e 不被 urlparse 解码，按字面文件名处理，落在 STATIC 内即安全。"""
        result = server.safe_static_path("/%2e%2e/%2e%2e/server.py")
        if result is not None:
            self.assertTrue(str(result).startswith(str(self.STATIC_REAL)))

    def test_double_slash_stays_inside(self):
        result = server.safe_static_path("//etc/passwd")
        if result is not None:
            self.assertTrue(str(result).startswith(str(self.STATIC_REAL)))


if __name__ == "__main__":
    unittest.main()
