"""批量移动场景/物品标签（/api/asset action=move_scene）回归测试。

场景（2026-09-05 用户需求）：自动分类（VLM 场景 / 物品标签）错误很多，
用户要在多选模式下把选中的照片批量移到正确的分类。

实现语义（scene_tag_v0，场景与物品共用此表）：
  from_tag 非空 → 先 DELETE 该批 asset 在 from_tag 下的行，再 INSERT to_tag
  （source='manual', confidence=1.0，与自动打的区分，INSERT OR IGNORE 防重复）；
  from_tag 空 → 只加不移（多对多）。
  缺 asset_ids / 缺 to_tag / from==to → 拒绝。

隔离方式：monkeypatch svr.DB 指向临时库，不碰真实数据库。
"""
import os
import tempfile
import sqlite3
import unittest
import importlib.util

SVR_PATH = os.environ.get("FF_SERVER_PATH") or str(Path(__file__).resolve().parent / "server.py")
_spec = importlib.util.spec_from_file_location("family_memory_svr", SVR_PATH)
svr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(svr)


def _setup_db(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE scene_tag_v0 (
        asset_id TEXT, tag TEXT, source TEXT,
        confidence REAL, created_at TEXT,
        PRIMARY KEY(asset_id, tag))""")
    # a1/a2 在「儿童」下（source=vlm），a3 在「乐器」下；a1 同时挂在「户外」
    con.executemany("INSERT INTO scene_tag_v0 VALUES (?,?,?,?,?)", [
        ("a1", "儿童", "vlm", 0.8, "2026-01-01"),
        ("a2", "儿童", "vlm", 0.9, "2026-01-01"),
        ("a3", "乐器", "vlm", 0.7, "2026-01-01"),
        ("a1", "户外", "vlm", 0.6, "2026-01-01"),
    ])
    con.commit()
    con.close()


class MoveSceneTests(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.tmp = path
        _setup_db(path)
        self._old_db = svr.DB
        svr.DB = path

    def tearDown(self):
        svr.DB = self._old_db
        os.unlink(self.tmp)

    def _tags(self, asset_id):
        con = sqlite3.connect(self.tmp)
        rows = con.execute(
            "SELECT tag, source, confidence FROM scene_tag_v0 WHERE asset_id=?",
            (asset_id,)).fetchall()
        con.close()
        return sorted(rows)  # 集合比较，不依赖返回顺序（中文 tag 按字节序不可读）

    def test_move_removes_old_and_adds_new_manual(self):
        """from_tag 非空：旧标签删、新标签 source=manual confidence=1.0；其他标签不动。"""
        d = svr.move_scene_tags({"asset_ids": ["a1", "a2"],
                                 "from_tag": "儿童", "to_tag": "乐器"})
        self.assertEqual(d["removed"], 2)
        self.assertEqual(d["added"], 2)
        self.assertEqual(self._tags("a1"), [("乐器", "manual", 1.0), ("户外", "vlm", 0.6)])
        self.assertEqual(self._tags("a2"), [("乐器", "manual", 1.0)])

    def test_add_only_keeps_existing(self):
        """from_tag 空：只加不移，照片原有的其他标签全部保留。"""
        d = svr.move_scene_tags({"asset_ids": ["a3"], "to_tag": "儿童"})
        self.assertEqual(d["removed"], 0)
        self.assertEqual(self._tags("a3"), [("乐器", "vlm", 0.7), ("儿童", "manual", 1.0)])

    def test_insert_ignore_no_duplicate(self):
        """重复移到同一标签不产生重复行（主键 asset_id+tag + INSERT OR IGNORE）。"""
        svr.move_scene_tags({"asset_ids": ["a3"], "to_tag": "儿童"})
        d = svr.move_scene_tags({"asset_ids": ["a3"], "to_tag": "儿童"})
        self.assertEqual(d["added"], 0)
        con = sqlite3.connect(self.tmp)
        n = con.execute(
            "SELECT COUNT(*) FROM scene_tag_v0 WHERE asset_id='a3' AND tag='儿童'").fetchone()[0]
        con.close()
        self.assertEqual(n, 1)

    def test_missing_to_tag_rejected(self):
        with self.assertRaises(ValueError):
            svr.move_scene_tags({"asset_ids": ["a1"], "from_tag": "儿童", "to_tag": ""})

    def test_missing_asset_ids_rejected(self):
        with self.assertRaises(ValueError):
            svr.move_scene_tags({"from_tag": "儿童", "to_tag": "乐器"})

    def test_same_tag_rejected(self):
        with self.assertRaises(ValueError):
            svr.move_scene_tags({"asset_ids": ["a1"], "from_tag": "儿童", "to_tag": "儿童"})


if __name__ == "__main__":
    unittest.main()
