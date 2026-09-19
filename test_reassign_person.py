"""改人物归属（/api/asset action=reassign_person）回归测试。

场景（2026-09-05 用户需求）：人脸识别把照片归错了人（如「刘建强」页里出现
不是他的照片），用户要把这张照片改归到正确的人（如某个家庭成员）。

实现语义：
  把该 asset 上所有 person_id=from 的 face_instance_v0 行改为 to_person；
  同一张照片里其他人的脸实例不动；原片文件不动。
  to_person_id 为空 → 取消归属（person_id=NULL，脸回到待认领池）。
  from==to、缺参数、目标人物不存在、照片上没有该人物的脸 → 均拒绝。

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
    con.execute("""CREATE TABLE person (
        person_id TEXT PRIMARY KEY, display_name TEXT)""")
    con.execute("""CREATE TABLE face_instance_v0 (
        face_instance_id TEXT PRIMARY KEY,
        asset_id TEXT, person_id TEXT,
        detection_score REAL, sample_role TEXT)""")
    con.executemany("INSERT INTO person VALUES (?,?)",
                    [("p_a", "成员甲"), ("p_b", "成员乙")])
    # asset_1：2 张脸归 p_a（含同照片多脸场景）+ 1 张脸归 p_c（不该被动）
    con.executemany("INSERT INTO face_instance_v0 VALUES (?,?,?,?,?)", [
        ("f1", "asset_1", "p_a", 0.9, "auto"),
        ("f2", "asset_1", "p_a", 0.8, "auto"),
        ("f3", "asset_1", "p_c", 0.7, "auto"),
    ])
    con.commit()
    con.close()


class ReassignPersonTests(unittest.TestCase):
    def setUp(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.tmp = path
        _setup_db(path)
        self._old_db = svr.DB
        svr.DB = path

    def tearDown(self):
        svr.DB = self._old_db
        try:
            os.unlink(self.tmp)
        except OSError:
            pass

    def _faces(self):
        con = sqlite3.connect(self.tmp)
        rows = con.execute(
            "SELECT face_instance_id, person_id FROM face_instance_v0 ORDER BY face_instance_id"
        ).fetchall()
        con.close()
        return {r[0]: r[1] for r in rows}

    def test_reassign_moves_only_from_person_faces(self):
        """from 人物的 2 张脸全移到目标，其他人的脸不动。"""
        d = svr.reassign_person({"asset_id": "asset_1",
                                 "from_person_id": "p_a", "to_person_id": "p_b"})
        self.assertTrue(d["ok"])
        self.assertEqual(d["updated"], 2)
        self.assertEqual(d["from"], "刘建强")
        self.assertEqual(d["to"], "成员乙")
        f = self._faces()
        self.assertEqual(f["f1"], "p_b")
        self.assertEqual(f["f2"], "p_b")
        self.assertEqual(f["f3"], "p_c")  # 其他人的脸不受影响

    def test_same_person_rejected(self):
        """from == to 应拒绝，不产生任何写入。"""
        with self.assertRaises(ValueError):
            svr.reassign_person({"asset_id": "asset_1",
                                 "from_person_id": "p_a", "to_person_id": "p_a"})
        self.assertEqual(self._faces()["f1"], "p_a")

    def test_missing_args_rejected(self):
        """缺 asset_id / from_person_id 应拒绝。"""
        for body in ({"from_person_id": "p_a", "to_person_id": "p_b"},
                     {"asset_id": "asset_1", "to_person_id": "p_b"}):
            with self.assertRaises(ValueError):
                svr.reassign_person(body)

    def test_unknown_target_person_rejected(self):
        """目标人物不存在应拒绝。"""
        with self.assertRaises(ValueError):
            svr.reassign_person({"asset_id": "asset_1",
                                 "from_person_id": "p_a", "to_person_id": "p_ghost"})
        self.assertEqual(self._faces()["f1"], "p_a")

    def test_no_face_of_from_person_rejected(self):
        """照片上没有该人物的脸应拒绝（防止重复操作时静默 0 行）。"""
        with self.assertRaises(ValueError):
            svr.reassign_person({"asset_id": "asset_none",
                                 "from_person_id": "p_a", "to_person_id": "p_b"})

    def test_unassign_sets_null(self):
        """to_person_id 空 → 取消归属（person_id=NULL，回待认领池）。"""
        d = svr.reassign_person({"asset_id": "asset_1",
                                 "from_person_id": "p_a", "to_person_id": ""})
        self.assertEqual(d["to"], "（未归属）")
        f = self._faces()
        self.assertIsNone(f["f1"])
        self.assertIsNone(f["f2"])
        self.assertEqual(f["f3"], "p_c")

    def test_reassign_sets_person_lock(self):
        """人工纠正过的脸必须打 person_locked=1，未动的脸不打。"""
        svr.reassign_person({"asset_id": "asset_1",
                             "from_person_id": "p_a", "to_person_id": "p_b"})
        con = sqlite3.connect(self.tmp)
        rows = dict(con.execute(
            "SELECT face_instance_id, person_locked FROM face_instance_v0").fetchall())
        con.close()
        self.assertEqual(rows["f1"], 1)
        self.assertEqual(rows["f2"], 1)
        self.assertEqual(rows["f3"], 0)  # 其他人的脸不受影响

    def test_locked_faces_survive_redetect(self):
        """重检测的 DELETE（backfill_faces_full.detect_asset 同款 SQL）必须保留锁定脸。

        SQL 从 backfill_faces_full.py detect_asset 拷贝；两处改动要同步。
        """
        svr.reassign_person({"asset_id": "asset_1",
                             "from_person_id": "p_a", "to_person_id": "p_b"})
        # 再造一张人工框选脸（sample_role='manual'，原有保护）混在一起验证
        con = sqlite3.connect(self.tmp)
        # 测试库最小 schema 没建这两张表，detect_asset 的 DELETE 会联动它们，补上
        con.execute("CREATE TABLE face_embedding_v0 (face_instance_id TEXT, status TEXT)")
        con.execute("CREATE TABLE face_expression_v0 (face_instance_id TEXT)")
        con.execute("INSERT INTO face_instance_v0 (face_instance_id,asset_id,person_id,detection_score,sample_role,person_locked) VALUES ('f4','asset_1','p_b',0.9,'manual',0)")
        con.execute("DELETE FROM face_embedding_v0 WHERE face_instance_id IN (SELECT face_instance_id FROM face_instance_v0 WHERE asset_id=? AND sample_role != 'manual' AND COALESCE(person_locked,0)=0)", ("asset_1",))
        con.execute("DELETE FROM face_instance_v0 WHERE asset_id=? AND sample_role != 'manual' AND COALESCE(person_locked,0)=0", ("asset_1",))
        left = dict(con.execute(
            "SELECT face_instance_id, person_id FROM face_instance_v0").fetchall())
        con.close()
        # 锁定脸(f1,f2)与人工框选脸(f4)存活，未锁定的自动脸(f3)按原逻辑被删
        self.assertIn("f1", left)
        self.assertIn("f2", left)
        self.assertIn("f4", left)
        self.assertNotIn("f3", left)
        self.assertEqual(left["f1"], "p_b")  # 归属保持纠正后的值


if __name__ == "__main__":
    unittest.main()
