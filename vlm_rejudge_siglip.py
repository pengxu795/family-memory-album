#!/usr/bin/env python3
"""VLM 重判 SIGLIP 打上的场景/物品标签（qwen2.5vl:7b 本地优先，DeepSeek 回退）。

背景：SIGLIP 零样本文本-图像相似度打分在物品/场景上误判严重
（江河 1008 / 山景 873 / 儿童 505 / 夜景 368 等数量虚高）。
本脚本对「带 SIGLIP 标签」的资产逐张让 VLM 独立看图重判：
  - 删除该资产上所有 source='SIGLIP' 标签（保留 FACE/GEO/VLM 等其他来源）
  - VLM 输出 scene/objects，命中词表的写 scene_tag_v0(source='VLM', confidence=0.75)
  - 描述写 asset_description_v0

断点续跑：已处理资产 SIGLIP 标签已删、VLM 标签已写，重跑时 WHERE source='SIGLIP'
自然排除，可安全中断后继续。

用法：
  python3 vlm_rejudge_siglip.py --limit 20    # 冒烟
  python3 vlm_rejudge_siglip.py               # 全量（后台，Ollama 单 GPU 顺序跑）
"""
import argparse
import datetime
import json
import os
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "data", "family_memory.db")

# 复用 vlm_describe_assets 的缩略图/编码/VLM 调用（同一套口径，避免复制漂移）
sys.path.insert(0, HERE)
from vlm_describe_assets import (  # noqa: E402
    ensure_thumb, img_b64, llm_vlm_describe, init_tables,
)

ERR_REASON = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--day", help="只跑某天")
    args = ap.parse_args()

    con = sqlite3.connect(DB, timeout=60, isolation_level=None)  # 自动提交防写锁占死
    con.row_factory = sqlite3.Row
    init_tables(con)

    sql = """SELECT ma.asset_id, ma.media_type, substr(ma.capture_time,1,10) day
             FROM media_asset ma
             WHERE ma.asset_id IN (SELECT DISTINCT asset_id FROM scene_tag_v0 WHERE source='SIGLIP')"""
    params = []
    if args.day:
        sql += " AND substr(ma.capture_time,1,10)=?"
        params.append(args.day)
    sql += " ORDER BY (ma.capture_time IS NULL), ma.capture_time"
    rows = con.execute(sql, params).fetchall()
    if args.limit:
        rows = rows[:args.limit]
    print(f"[vlm-rejudge] 待重判 {len(rows)} 张（SIGLIP 标签资产）", flush=True)

    done = errors = tagged = 0
    now0 = datetime.datetime.now()
    for r in rows:
        tp = ensure_thumb(r["asset_id"])
        if not tp:
            errors += 1
            ERR_REASON["缩略图/原片不可得"] = ERR_REASON.get("缩略图/原片不可得", 0) + 1
            continue
        b64 = img_b64(tp)
        if not b64:
            errors += 1
            ERR_REASON["图片编码失败"] = ERR_REASON.get("图片编码失败", 0) + 1
            continue
        parsed, err = llm_vlm_describe(b64)
        if not parsed:
            errors += 1
            if err and err in ("no_balance", "no_key"):
                llm_down = getattr(main, "_llm_down", 0)
                if llm_down < 5:
                    setattr(main, "_llm_down", llm_down + 1)
                    print(f"[vlm-rejudge] LLM 不可用({err}), 30s 后重试 ({llm_down + 1}/5)", flush=True)
                    time.sleep(30)
                    continue
                print(f"[vlm-rejudge] LLM 连续不可用({err})，中止", flush=True)
                sys.exit(2)
            continue
        now = datetime.datetime.now().isoformat(timespec="seconds")
        main._llm_down = 0
        for attempt in range(60):
            try:
                # 关键：先删该资产全部 SIGLIP 标签（保留 FACE/GEO/VLM 等），再用 VLM 结果覆盖
                con.execute("DELETE FROM scene_tag_v0 WHERE asset_id=? AND source='SIGLIP'",
                            (r["asset_id"],))
                con.execute("INSERT OR REPLACE INTO asset_description_v0 VALUES (?,?,?,?)",
                            (r["asset_id"], parsed["desc"], "qwen2.5vl:7b", now))
                for tag in parsed["scene"] + parsed["objects"]:
                    con.execute("INSERT OR REPLACE INTO scene_tag_v0 VALUES (?,?,?,?,?)",
                                (r["asset_id"], tag, "VLM", 0.75, now))
                tagged += 1
                break
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 59:
                    raise
                if attempt % 6 == 0:
                    print(f"[vlm-rejudge] 写库被锁({e}), 第{attempt+1}/60次等待重试...", flush=True)
                time.sleep(10)
        done += 1
        if done % 50 == 0:
            dt = (datetime.datetime.now() - now0).total_seconds()
            eta = (len(rows) - done - errors) * (dt / max(done, 1)) / 60
            print(f"[vlm-rejudge] done={done} errors={errors} tags={tagged} "
                  f"速度={dt/max(done,1):.1f}s/张 预计剩余{eta:.0f}分钟", flush=True)

    dt = (datetime.datetime.now() - now0).total_seconds()
    print(f"[vlm-rejudge] 完成: done={done} errors={errors} tags={tagged} "
          f"耗时{dt/60:.1f}分钟", flush=True)
    if ERR_REASON:
        print("[vlm-rejudge] 失败原因: " +
              " | ".join(f"{k}={v}" for k, v in sorted(ERR_REASON.items(), key=lambda x: -x[1])),
              flush=True)
    left = con.execute(
        "SELECT COUNT(DISTINCT asset_id) FROM scene_tag_v0 WHERE source='SIGLIP'"
    ).fetchone()[0]
    print(f"[vlm-rejudge] 剩余 SIGLIP 标签资产: {left}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
