#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A4 联调回填脚本（⑨ 设计产物 —— 尚未执行，A4 全量重算完成后运行）

职责：把新功能①-⑥的 JSONL 产出回写 DB（迁移 004 之后的表结构）。
幂等：全部 INSERT OR REPLACE，重跑无损；先跑 --dry-run 看计划。

前置条件（按序）：
  1. A4 全量重算完成（backfill_faces_full.py + SigLIP2 全向量，WbsX3L 任务）
  2. sqlite3 < migrations/004_quality_attrs.sql
  3. 各模块全量 JSONL 就位：
     python3 blur_detector.py    --all --out data/blur_scores_full.jsonl
     python3 aesthetic_scorer.py --all --out data/aesthetic_full.jsonl
     python3 genderage.py        --all --out data/genderage_full.jsonl
     python3 smart_crop.py（--all 模式待补，见 A4_INTEGRATION.md 待办）
     python3 junk_scanner.py     --all --out data/junk_scan_full.jsonl   # QR 检出即过滤，无 --deep（2026-09-09 产品决策）
     python3 similar_dedup.py    --all --out data/dedup_groups_full.jsonl

用法：
  python3 backfill_quality.py --dry-run     # 打印将写入的行数，不碰库
  python3 backfill_quality.py               # 执行（自动备份 DB 到 data/*.bak）
"""
import argparse, json, shutil, sqlite3, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"

MODEL_VERSION = "quality-v1-20260909"
CROP_METHODS = {"face", "saliency", "horizon", "center"}
JUNK_REASON = {                      # junk_scanner kind → filter_reason
    "payment_qrcode_wechat": "JUNK_QR_PAYMENT_WECHAT",
    "payment_qrcode_alipay": "JUNK_QR_PAYMENT_ALIPAY",
    "screenshot": "JUNK_SCREENSHOT_VISUAL",
    # 2026-09-16 修复：qrcode_undecoded 不再过滤。cv2.QRCodeDetector 对海浪/砂石/
    # 树叶等纹理确定性误检「检出但解不出」，曾误藏 1130 张（45% 有人脸），抽样重检
    # 60/60 依旧全部「解不出」——该信号无判别力。解码成功的 qrcode 类才有资格过滤。
}


def load_jsonl(p):
    f = ROOT / "data" / p
    if not f.exists():
        print(f"  [缺] {p} —— 跳过（先跑对应模块 --all）")
        return []
    return [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]


def combined_quality(aesthetic, blur_label):
    """综合展示分：美学为主，模糊惩罚。公式唯一出处，前端只读不算。"""
    if aesthetic is None:
        return None
    pen = {"sharp": 1.0, "soft": 0.85, "blurry": 0.5}.get(blur_label, 1.0)
    return round(aesthetic * pen, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", default=str(DB))
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=120)
    con.row_factory = sqlite3.Row
    stats = {}

    blur = {r["asset_id"]: r for r in load_jsonl("blur_scores_full.jsonl")}
    aeth = {r["asset_id"]: r["aesthetic"] for r in load_jsonl("aesthetic_full.jsonl")}
    ga = load_jsonl("genderage_full.jsonl")
    crops = load_jsonl("crop_params_full.jsonl")
    junk = load_jsonl("junk_scan_full.jsonl")
    dedup = load_jsonl("dedup_groups_full.jsonl")
    ocr = load_jsonl("ocr_text_full.jsonl")
    expr = load_jsonl("face_expr_full.jsonl")

    # ① 画质：blur + aesthetic → asset_quality_v0 + media_asset.quality_score
    stats["quality"] = len(set(blur) | set(aeth))
    # ⑥ 性别年龄：face_instance_id → face_instance_v0 三列
    stats["genderage"] = len(ga)
    # ② 裁切：{x,y,w,h,method} → crop_v0（不动 manual 行）
    stats["crop_auto"] = sum(1 for c in crops if c.get("method") in CROP_METHODS)
    # ④ junk：kind → asset_filter_v0（QR 检出即隐藏，含 undecoded——2026-09-09 产品决策）
    stats["junk_filter"] = sum(1 for r in junk if r.get("kind") in JUNK_REASON)
    # ③ dedup：engine='dual-v1' 新组 → asset_similar_group_v0 + member
    stats["dedup_groups"] = len(dedup)
    # ⑩ OCR：全量写 asset_ocr_v0（含空文本=已扫无字标记），非空才进 FTS
    stats["ocr"] = len(ocr)
    # ⑪ 表情：face_instance_id → face_instance_v0 四列
    stats["expr"] = len(expr)

    print("回填计划:", stats, "（合计", sum(stats.values()), "行）")
    if args.dry_run:
        con.close()
        return

    bak = args.db + f".bak.{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.copy2(args.db, bak)
    print(f"DB 已备份 → {bak}")

    with con:
        for aid, r in blur.items():
            a = aeth.get(aid)
            con.execute("""INSERT OR REPLACE INTO asset_quality_v0
                (asset_id, blur_label, sharp_score, aesthetic, model_version)
                VALUES (?,?,?,?,?)""",
                (aid, r.get("label"), r.get("sharp_score"), a, MODEL_VERSION))
            if a is not None:
                con.execute("UPDATE media_asset SET quality_score=? WHERE asset_id=?",
                            (combined_quality(a, r.get("label")), aid))
        for r in ga:
            con.execute("""UPDATE face_instance_v0 SET gender=?, age_est=?, age_label=?
                WHERE face_instance_id=?""",
                (r.get("pred_gender"), r.get("pred_age"), r.get("pred_label"),
                 r["face_instance_id"]))
        for c in crops:
            if c.get("method") not in CROP_METHODS:
                continue
            con.execute("""INSERT OR REPLACE INTO crop_v0 (asset_id, x, y, w, h, updated_at, method)
                VALUES (?,?,?,?,?,datetime('now'),?)""",
                (c["asset_id"], c["x"], c["y"], c["w"], c["h"], c["method"]))
        for r in junk:
            reason = JUNK_REASON.get(r.get("kind"))
            if not reason:
                continue
            con.execute("""INSERT OR REPLACE INTO asset_filter_v0
                (asset_id, filter_reason, evidence_kind, evidence_value, confidence, rule_version, created_at)
                VALUES (?,?,?,?,?,?,datetime('now'))""",
                (r["asset_id"], reason,
                 "visual" if reason == "JUNK_SCREENSHOT_VISUAL" else "ocr",
                 r.get("content") or r.get("note") or "", r.get("confidence", 0.9), "junk-v1"))
        # ③ dedup：先清旧 dual-v1 组（幂等重跑），组号接续 sim_XXXXXXX 序列
        con.execute("""DELETE FROM asset_similar_member_v0 WHERE group_id IN
            (SELECT group_id FROM asset_similar_group_v0 WHERE engine='dual-v1')""")
        con.execute("DELETE FROM asset_similar_group_v0 WHERE engine='dual-v1'")
        r0 = con.execute(
            "SELECT MAX(CAST(substr(group_id,5) AS INTEGER)) FROM asset_similar_group_v0").fetchone()
        seq = (r0[0] if r0 and r0[0] is not None else -1) + 1
        for g in dedup:
            gid = f"sim_{seq:07d}"
            seq += 1
            con.execute("""INSERT INTO asset_similar_group_v0
                (group_id, best_asset_id, asset_count, created_at, updated_at, engine)
                VALUES (?,?,?,?,datetime('now'),'dual-v1')""",
                (gid, g["keep"], len(g["duplicates"]) + 1,
                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
            con.execute("""INSERT OR IGNORE INTO asset_similar_member_v0 (group_id, asset_id, is_best)
                VALUES (?,?,1)""", (gid, g["keep"]))
            for d in g["duplicates"]:
                con.execute("""INSERT OR IGNORE INTO asset_similar_member_v0 (group_id, asset_id, is_best)
                    VALUES (?,?,0)""", (gid, d))
        for r in ocr:
            con.execute("""INSERT OR REPLACE INTO asset_ocr_v0
                (asset_id, n_regions, mean_conf, text, model_version)
                VALUES (?,?,?,?,?)""",
                (r["asset_id"], r.get("n_regions", 0), r.get("mean_conf") or 0,
                 r.get("text") or "", MODEL_VERSION))
            if r.get("text"):
                con.execute("INSERT OR REPLACE INTO asset_ocr_fts (asset_id, text) VALUES (?,?)",
                            (r["asset_id"], r["text"]))
        for r in expr:
            con.execute("""UPDATE face_instance_v0 SET eyes_open=?, eye_open_score=?,
                smile_ratio=?, is_smiling=? WHERE face_instance_id=?""",
                (1 if r.get("eyes_open") else 0, r.get("eye_open_score"),
                 r.get("smile_ratio"), 1 if r.get("is_smiling") else 0,
                 r["face_instance_id"]))
    print("回填完成:", stats)
    con.close()


if __name__ == "__main__":
    main()
