#!/usr/bin/env python3
import datetime
import sqlite3
import unittest
from collections import Counter

import server


class FamilyMemoryCoreTest(unittest.TestCase):
    def setUp(self):
        self.original_parser = server.parse_intent
        server.parse_intent = server.fallback_parse
        # #16 隐私清洗：人物动态取自 DB（测试代码不硬编码家庭人名）
        con = sqlite3.connect(server.DB)
        con.row_factory = sqlite3.Row
        self.kids = con.execute(
            """SELECT DISTINCT child.display_name, child.birth_date
               FROM person_relation_v0 pr
               JOIN person child ON child.person_id=pr.to_person_id
               WHERE pr.relation_type IN ('father_of','mother_of')
                 AND child.birth_date IS NOT NULL
               ORDER BY child.birth_date"""
        ).fetchall()
        con.close()
        self.older_kid = self.kids[0]["display_name"] if self.kids else None
        self.younger_kid = self.kids[-1]["display_name"] if self.kids else None

    def tearDown(self):
        server.parse_intent = self.original_parser

    def test_memory_graph_composition(self):
        result = server.search("我们第一次带孩子去赛里木湖的时候，他几岁？")
        self.assertTrue(result["intent"].get("memory_graph_composition"))
        self.assertIn("2026-07-24", result["answer"])
        for kid in self.kids:  # 每个孩子的年龄文本动态复算（与 server 同一算法）
            years, months = server._age_on_date(kid["birth_date"], datetime.date(2026, 7, 24))
            self.assertIn(f"{kid['display_name']} {years}岁{months}个月", result["answer"])
        types = Counter(a["type"] for a in result["assets"] if not a["hidden"])
        self.assertGreater(types["photo"], 0)
        self.assertGreater(types["video"], 0)

    def test_cross_source_travel_returns_photos_and_videos(self):
        result = server.search("找同一次旅行里两部手机拍的照片和视频")
        self.assertTrue(result["intent"].get("same_event_cross_source"))
        types = Counter(a["type"] for a in result["assets"] if not a["hidden"])
        self.assertGreater(types["photo"], 0)
        self.assertGreater(types["video"], 0)

    def test_exclusion_does_not_leak_obvious_paths(self):
        result = server.search("排除截图、聊天记录和下载图片")
        markers = ("/screenshots/", "/screenshot/", "/download/", "/downloads/", "/weixin/", "/wechat/")
        con = sqlite3.connect(server.DB)
        for asset in result["assets"]:
            paths = [r[0].lower() for r in con.execute(
                "SELECT absolute_path FROM media_file WHERE asset_id=?", (asset["id"],))]
            self.assertFalse(any(any(marker in p for marker in markers) for p in paths), asset["id"])
        con.close()

    def test_deterministic_filter_covers_all_poc_negatives(self):
        con = sqlite3.connect(server.DB)
        leaked = con.execute(
            """SELECT count(*) FROM cohort_asset ca
               WHERE ca.cohort_id='cohort_poc_v01' AND ca.sample_role='negative'
                 AND NOT EXISTS (SELECT 1 FROM asset_filter_v0 af
                                 WHERE af.asset_id=ca.asset_id AND af.confidence>=0.99)"""
        ).fetchone()[0]
        self.assertEqual(leaked, 0)
        mmexport = con.execute(
            """SELECT count(*) FROM asset_filter_v0 af JOIN media_file mf USING(asset_id)
               WHERE lower(mf.filename) LIKE 'mmexport%' AND af.filter_reason='CHAT_EXPORT_FILENAME'"""
        ).fetchone()[0]
        self.assertGreater(mmexport, 0)
        con.close()

    def test_emotion_uses_validated_visible_expression_only(self):
        if not self.older_kid:
            self.skipTest("库中无亲子关系孩子数据")
        result = server.search(f"找{self.older_kid}特别开心的照片")
        self.assertGreater(len(result["assets"]), 0)
        self.assertIn("明显笑容", result["answer"])
        self.assertIn("不代表人物真实内心状态", result["answer"])
        con = sqlite3.connect(server.DB)
        for asset in result["assets"]:
            score = con.execute(
                """SELECT max(fx.happiness_score) FROM face_expression_v0 fx
                   JOIN face_instance_v0 fi USING(face_instance_id)
                   JOIN person p USING(person_id)
                   WHERE fi.asset_id=? AND p.display_name=?""",
                (asset["id"], self.older_kid),
            ).fetchone()[0]
            self.assertGreaterEqual(score, 0.90)
        con.close()

    def test_embedding_blob_contract_and_event_integrity(self):
        con = sqlite3.connect(server.DB)
        invalid = con.execute(
            "SELECT count(*) FROM embedding WHERE length(vector) != dimension*4"
        ).fetchone()[0]
        self.assertEqual(invalid, 0)
        self.assertEqual(con.execute("PRAGMA foreign_key_check").fetchall(), [])
        bad_sequences = con.execute(
            """SELECT count(*) FROM (
                 SELECT event_id,count(*) n,count(DISTINCT sequence_no) d
                 FROM event_asset GROUP BY event_id HAVING n!=d)"""
        ).fetchone()[0]
        self.assertEqual(bad_sequences, 0)
        total_assets = con.execute("SELECT count(*) FROM media_asset").fetchone()[0]
        # 生产检索模型 = SIGLIP（local_siglip_scores 唯一来源），必须全库覆盖。
        # 2026-09-02 更新: open_clip(A/B 实验时代遗留)已无代码引用, 从"必须全覆盖"降级为"存在即可",
        #   否则新增资产后测试永远红。
        siglip_count = con.execute(
            "SELECT count(*) FROM embedding WHERE subject_type='asset' AND model_name=?", (server.SIGLIP_MODEL_NAME,)
        ).fetchone()[0]
        self.assertEqual(siglip_count, total_assets, "siglip 嵌入必须全库覆盖")
        legacy = con.execute(
            "SELECT count(*) FROM embedding WHERE subject_type='asset' AND model_name='open_clip:ViT-B-32:laion2b_s34b_b79k'"
        ).fetchone()[0]
        self.assertGreater(legacy, 0, "open_clip 嵌入(历史)应仍存在于库中")
        con.close()

    def test_low_reliability_scene_is_disclosed(self):
        result = server.search("找下雨的照片")
        self.assertEqual(result["intent"].get("model"), server.SIGLIP_MODEL_NAME)
        self.assertIn("低可靠", result["answer"])

    def test_unconfirmed_grandfather_query_is_blocked(self):
        if not self.younger_kid:
            self.skipTest("库中无亲子关系孩子数据")
        result = server.search(f"找{self.younger_kid}5岁以前和爷爷一起出去玩的照片")
        # 爷爷身份未确认时必须阻断，不能退化为只查孩子；已确认时，
        # 返回的每一张照片都必须真实包含爷爷出镜，不能拿只有孩子的照片凑数。
        if result["intent"].get("blocked"):
            self.assertEqual(result["assets"], [])
            self.assertIn("没有用“老人”外观猜身份", result["answer"])
        else:
            con = sqlite3.connect(server.DB)
            try:
                for asset in result["assets"]:
                    hit = con.execute(
                        """SELECT COUNT(*) FROM face_instance_v0 fi
                           JOIN person p USING(person_id)
                           WHERE fi.asset_id=? AND p.identity_status='confirmed'
                             AND (p.display_name='爷爷' OR p.relationship_label='爷爷')""",
                        (asset["id"],)).fetchone()[0]
                    self.assertGreater(hit, 0, f"照片 {asset['id']} 无爷爷出镜却作为“和爷爷一起”返回")
            finally:
                con.close()

    def test_empty_video_query_uses_video_wording(self):
        result = server.search("找一家人在海边的视频")
        self.assertEqual(result["assets"], [])
        self.assertIn("没有找到相关视频", result["answer"])


if __name__ == "__main__":
    unittest.main()
