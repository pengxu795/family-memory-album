"""分类排序（/api/catman action=sort）回归测试。

历史 bug（2026-09-05 实测发现）：
  sort 原先只对 category_layout_v0 做 UPDATE。而自动创建的分类（人脸识别出的人物、
  VLM 打的场景标签）在这张表里根本没有行 —— 纯 UPDATE 影响 0 行，排序**静默失效**。
  实测覆盖率：person 16 人只有 8 人有 layout 行；scene 12139 个标签只有 32 个有。
  表现就是用户侧「有的分类拖得动，有的死活拖不动」。

修复：sort 改为先 INSERT OR IGNORE 补行（主键 kind,target），再 UPDATE 排序。
本测试锁定该行为，防止退化回纯 UPDATE。

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


class CatmanSortUpsertTests(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.tmp = path
        self._old_db = svr.DB
        svr.DB = self.tmp

    def tearDown(self):
        svr.DB = self._old_db
        try:
            os.unlink(self.tmp)
        except OSError:
            pass

    def _layout(self):
        con = sqlite3.connect(self.tmp)
        con.row_factory = sqlite3.Row
        try:
            return {r["target"]: dict(r) for r in con.execute(
                "SELECT * FROM category_layout_v0 WHERE kind='person'")}
        finally:
            con.close()

    def test_sort_creates_row_for_never_touched_target(self):
        """核心回归：从未进过 layout 的分类，sort 必须补行，不能静默丢弃。"""
        res = svr.catman_action({"action": "sort", "kind": "person",
                                 "parent": "person", "ordered": ["p_new"]})
        self.assertTrue(res.get("ok"))
        rows = self._layout()
        self.assertIn("p_new", rows,
                      "sort 必须为无记录的分类补行，否则排序永远写不进去")
        self.assertEqual(rows["p_new"]["sort_order"], 0)
        self.assertEqual(rows["p_new"]["parent"], "person")
        self.assertEqual(rows["p_new"]["show_on_home"], 1)

    def test_sort_persists_full_order(self):
        svr.catman_action({"action": "sort", "kind": "person",
                           "parent": "person", "ordered": ["p_a", "p_b", "p_c"]})
        rows = self._layout()
        self.assertEqual({k: v["sort_order"] for k, v in rows.items()},
                         {"p_a": 0, "p_b": 1, "p_c": 2})

    def test_sort_reorder_keeps_single_row_per_target(self):
        """重复排序不能因 UPSERT 写出重复行（主键 kind,target 唯一）。"""
        svr.catman_action({"action": "sort", "kind": "person",
                           "parent": "person", "ordered": ["p_x", "p_y"]})
        svr.catman_action({"action": "sort", "kind": "person",
                           "parent": "person", "ordered": ["p_y", "p_x"]})
        rows = self._layout()
        self.assertEqual({k: v["sort_order"] for k, v in rows.items()},
                         {"p_y": 0, "p_x": 1})
        self.assertEqual(len(rows), 2, "重复排序不应产生重复行")

    def test_sort_with_empty_parent_means_pinned_flat(self):
        """parent='' 表示一级平铺置顶，补行时 parent 应为空串而非 NULL。"""
        svr.catman_action({"action": "sort", "kind": "person",
                           "parent": "", "ordered": ["p_pin"]})
        rows = self._layout()
        self.assertEqual(rows["p_pin"]["parent"], "")

    def test_sort_rejects_empty_ordered(self):
        with self.assertRaises(ValueError):
            svr.catman_action({"action": "sort", "kind": "person",
                               "parent": "person", "ordered": []})


if __name__ == "__main__":
    unittest.main()
