"""搜索打分单测：MMR 相似度纯函数 + mmr_rerank_topk 多样性行为。

- _mmr_sim 是纯函数，可脱离 DB 直接断言打分规则；
- mmr_rerank_topk 用临时 SQLite 库验证「日期扎堆被拆散、多样性提升」，
  不碰真实库（服务不在线也能跑，与 test_smoke_live 互补）。

相似度规则（server.py _mmr_sim 真实行为，本测试即文档）：
  同 capture_date(+0.5)  +  0.5 * Jaccard(scene_tag_a, scene_tag_b)  (max 0.5)
  注：双方都无 capture_time 时 date_map 取值均为 None，None==None 命中 +0.5，
      即「无日期照片之间视为同日」——这是现状语义，已用 test 锁定，勿误改。
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


class MMRSimTests(unittest.TestCase):
    def test_same_date_adds_half(self):
        dm = {"a": "2026-01-01", "b": "2026-01-01"}
        self.assertAlmostEqual(svr._mmr_sim("a", "b", dm, {}), 0.5)

    def test_diff_date_no_tags_is_zero(self):
        dm = {"a": "2026-01-01", "b": "2026-02-02"}
        self.assertEqual(svr._mmr_sim("a", "b", dm, {}), 0.0)

    def test_same_date_full_tag_overlap_is_one(self):
        dm = {"a": "2026-01-01", "b": "2026-01-01"}
        tm = {"a": {"海边", "雪山"}, "b": {"海边", "雪山"}}
        self.assertAlmostEqual(svr._mmr_sim("a", "b", dm, tm), 1.0)

    def test_partial_tag_jaccard(self):
        # 隔离日期维度（不同日期），纯测 tag：交集 1 / 并集 3 → 0.5 * (1/3) = 1/6
        dm = {"a": "2026-01-01", "b": "2026-02-02"}
        tm = {"a": {"海边", "雪山"}, "b": {"海边", "城市"}}
        self.assertAlmostEqual(svr._mmr_sim("a", "b", dm, tm), 1 / 6)

    def test_no_tag_intersection_is_zero(self):
        dm = {"a": "2026-01-01", "b": "2026-02-02"}
        tm = {"a": {"海边"}, "b": {"雪山"}}
        self.assertEqual(svr._mmr_sim("a", "b", dm, tm), 0.0)

    def test_empty_tag_side_is_zero(self):
        dm = {"a": "2026-01-01", "b": "2026-02-02"}
        tm = {"a": set(), "b": {"雪山"}}
        self.assertEqual(svr._mmr_sim("a", "b", dm, tm), 0.0)

    def test_same_date_partial_tag(self):
        dm = {"a": "2026-01-01", "b": "2026-01-01"}
        tm = {"a": {"海边", "雪山"}, "b": {"海边", "城市"}}
        # 0.5 + 0.5 * (1/3) = 2/3
        self.assertAlmostEqual(svr._mmr_sim("a", "b", dm, tm), 2 / 3)

    def test_both_no_date_treated_as_same(self):
        # 现状语义锁定：双方都无 capture_time → None==None 命中 +0.5。
        # 即使 tag 完全不交，仍因「无日期」得 0.5；勿误改成「无日期=不相似」。
        tm = {"a": {"海边"}, "b": {"雪山"}}
        self.assertAlmostEqual(svr._mmr_sim("a", "b", {}, tm), 0.5)


class MMRRerankTests(unittest.TestCase):
    def _tmpdb(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row  # mmr_rerank_topk 内部用 r["capture_time"]
        con.execute("CREATE TABLE media_asset(asset_id TEXT, capture_time TEXT)")
        con.execute("CREATE TABLE scene_tag_v0(asset_id TEXT, tag TEXT)")
        return con, path

    def test_short_circuit_returns_unchanged(self):
        # len <= k+5 不做重排，原样返回
        con, path = self._tmpdb()
        scored = [(f"a{i}", 1.0, f"2026-01-0{i % 9 + 1}T00:00:00") for i in range(10)]
        out = svr.mmr_rerank_topk(con, scored, k=10)
        con.close()
        os.unlink(path)
        self.assertEqual([a[0] for a in out], [a[0] for a in scored])

    def test_diversity_mixes_dates(self):
        # 16 张：12 张同日期 A(晚) + 4 张日期 B(早)，全同分。
        # 原按 time desc 取前 5 = 5 张 A（单一日期扎堆）；
        # MMR 应把日期 B 提到前面，top5 含两类日期（多样性生效）。
        con, path = self._tmpdb()
        scored = []
        for i in range(12):
            aid = f"A{i}"
            con.execute("INSERT INTO media_asset VALUES(?,?)", (aid, "2026-03-03T10:00:00"))
            scored.append((aid, 1.0, "2026-03-03T10:00:00"))
        for i in range(4):
            aid = f"B{i}"
            con.execute("INSERT INTO media_asset VALUES(?,?)", (aid, "2026-01-01T10:00:00"))
            scored.append((aid, 1.0, "2026-01-01T10:00:00"))
        out = svr.mmr_rerank_topk(con, scored, k=5)
        con.close()
        os.unlink(path)
        top5 = out[:5]
        dates = {t[:10] for _, _, t in top5}
        self.assertEqual(len(dates), 2)  # 两类日期都在前 5，多样性生效


if __name__ == "__main__":
    unittest.main()
