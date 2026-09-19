#!/usr/bin/env python3
"""按天生成内容描述（本地结构化综合 + VLM 叙事增强）。

内容理解层的地基：库内 6,148 天影像 → 625 个「天」单元，每天一条中文描述。
- 本地层（零成本，全量）：日期/星期/地区/在场人物/场景标签/数量/相机 → 自然中文描述
- VLM 层（需 DeepSeek 余额）：每天抽最多 3 张代表帧 → 视觉模型生成「谁在做什么」叙事
- 产物：day_caption_v0 表 + memory_search FTS（trigram 分词，中文子串可检索）

用法:
  python3 caption_days.py                    # 本地综合全量（秒级）
  python3 caption_days.py --vlm              # VLM 叙事补跑（余额不足自动整轮跳过）
  python3 caption_days.py --day 2026-07-30   # 只跑一天（调试）
  python3 caption_days.py --limit 20 --vlm   # 冒烟
"""
import argparse
import base64
import datetime
import json
import os
import re
import sqlite3
import subprocess as sp
import sys
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("FF_DB_PATH") or os.path.join(os.environ.get("FF_DATA_DIR") or os.path.join(ROOT, "data"), "family_memory.db")

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

CRED_FILE = os.path.expanduser("~/.dsh/.credentials.yaml")

# 2026-09-19 开源发行版：ffmpeg 不硬编码 macOS homebrew；sips 仅 macOS 有
import shutil
FFMPEG_BIN = (os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
              or ("/usr/bin/ffmpeg" if os.path.exists("/usr/bin/ffmpeg") else "")
              or ("/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else "ffmpeg"))


def load_api_key():
    try:
        s = open(CRED_FILE).read()
    except Exception:
        return ""
    try:
        import yaml
        key = yaml.safe_load(s).get("DEEPSEEK_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    m = re.search(r"DEEPSEEK_API_KEY:\s*['\"]?([A-Za-z0-9_\-]+)", s)
    return m.group(1) if m else ""


def _img_b64(path, px=400):
    t = f"/tmp/cap_{os.getpid()}.jpg"
    try:
        if path.lower().endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")):
            sp.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
                    "-ss", "0.5", "-i", path, "-frames:v", "1", "-vf", f"scale={px}:-2",
                    "-f", "image2", t], capture_output=True, timeout=20, check=True)
        elif shutil.which("sips"):
            sp.run(["sips", "-Z", str(px), "-s", "format", "jpeg", path, "--out", t],
                   capture_output=True, timeout=20, check=True)
        else:
            # 容器/Linux：ffmpeg 转码缩图（与视频同路，无 seek）
            sp.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", path, "-frames:v", "1", "-vf", f"scale={px}:-2", t],
                   capture_output=True, timeout=20, check=True)
        with open(t, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return None


def init_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS day_caption_v0 (
        day TEXT PRIMARY KEY,
        family_id TEXT NOT NULL DEFAULT 'fam_default',
        caption_local TEXT,
        caption_vlm TEXT,
        vlm_model TEXT,
        rep_asset_ids TEXT,
        stats_json TEXT,
        updated_at TEXT NOT NULL
    )""")
    # FTS 重建为 trigram（unicode61 中文无法子串检索）。只在分词器不对时重建一次。
    row = con.execute(
        "SELECT sql FROM sqlite_master WHERE name='memory_search'").fetchone()
    if row and "trigram" not in (row[0] or ""):
        keep = con.execute(
            "SELECT subject_type, subject_id, text FROM memory_search").fetchall()
        con.execute("DROP TABLE memory_search")
        con.execute("""CREATE VIRTUAL TABLE memory_search USING fts5(
            subject_type UNINDEXED, subject_id UNINDEXED, text,
            tokenize='trigram')""")
        for r in keep:
            con.execute("INSERT INTO memory_search VALUES (?,?,?)", r)
    elif not row:
        con.execute("""CREATE VIRTUAL TABLE memory_search USING fts5(
            subject_type UNINDEXED, subject_id UNINDEXED, text,
            tokenize='trigram')""")


def asset_path(con, asset_id):
    row = con.execute(
        "SELECT absolute_path FROM media_file WHERE asset_id=? "
        "ORDER BY variant_kind='original' DESC LIMIT 1", (asset_id,)).fetchone()
    return row[0] if row else None


def day_weekday(day):
    try:
        return WEEKDAYS[datetime.date.fromisoformat(day).weekday()]
    except Exception:
        return ""


def load_days(con):
    """按天聚合：数量/地区/人物/场景/相机/代表帧。"""
    days = {}
    for r in con.execute(
            """SELECT substr(ma.capture_time,1,10) day, COUNT(*) n,
                      SUM(ma.media_type='photo') photos, SUM(ma.media_type='video') videos
               FROM media_asset ma
               WHERE ma.capture_time IS NOT NULL GROUP BY 1 ORDER BY 1"""):
        days[r[0]] = {"day": r[0], "n": r[1], "photos": r[2] or 0,
                      "videos": r[3] or 0}
    # 地区（当天多数）
    for r in con.execute(
            """SELECT substr(ma.capture_time,1,10) day, ag.region, COUNT(*) c
               FROM media_asset ma JOIN asset_geo_v0 ag USING(asset_id)
               WHERE ma.capture_time IS NOT NULL AND ag.region IS NOT NULL AND ag.region != ''
               GROUP BY 1,2"""):
        d = days.get(r[0])
        if d:
            d.setdefault("_regions", []).append((r[2], r[1]))
    # 人物（当天有脸归属的人）
    for r in con.execute(
            """SELECT substr(ma.capture_time,1,10) day, p.display_name, COUNT(DISTINCT fi.asset_id) c
               FROM media_asset ma
               JOIN face_instance_v0 fi ON fi.asset_id = ma.asset_id AND fi.person_id IS NOT NULL
               JOIN person p ON p.person_id = fi.person_id
               WHERE ma.capture_time IS NOT NULL GROUP BY 1,2"""):
        d = days.get(r[0])
        if d:
            d.setdefault("_people", []).append((r[2], r[1]))
    # 场景标签
    for r in con.execute(
            """SELECT substr(ma.capture_time,1,10) day, st.tag, COUNT(DISTINCT st.asset_id) c
               FROM media_asset ma JOIN scene_tag_v0 st ON st.asset_id = ma.asset_id
               WHERE ma.capture_time IS NOT NULL GROUP BY 1,2"""):
        d = days.get(r[0])
        if d:
            d.setdefault("_scenes", []).append((r[2], r[1]))
    # 相机
    for r in con.execute(
            """SELECT substr(ma.capture_time,1,10) day, ma.camera_model, COUNT(*) c
               FROM media_asset ma
               WHERE ma.capture_time IS NOT NULL AND ma.camera_model IS NOT NULL AND ma.camera_model != ''
               GROUP BY 1,2"""):
        d = days.get(r[0])
        if d:
            d.setdefault("_cameras", []).append((r[2], r[1]))
    # 代表帧：当天按时间排序的资产里，优先照片、有脸，均匀抽 3 张
    for r in con.execute(
            """SELECT substr(ma.capture_time,1,10) day, ma.asset_id, ma.media_type,
                      (SELECT COUNT(*) FROM face_instance_v0 fi WHERE fi.asset_id=ma.asset_id) faces
               FROM media_asset ma WHERE ma.capture_time IS NOT NULL ORDER BY day, ma.capture_time"""):
        d = days.get(r[0])
        if d:
            d.setdefault("_assets", []).append((r[1], r[2], r[3]))
    out = []
    # 物品标签(与 server.py OBJECT_TEXTS 同步): 描述里场景取前4、物品另取前2, 避免物品被大数量场景挤出
    OBJECT_TAGS = frozenset(["书包", "行李箱", "自行车", "火车", "飞机", "游船", "婴儿车",
                             "玩具", "气球", "风筝", "帐篷", "烧烤", "火锅", "灯笼",
                             "滑雪", "乐器"])
    for day, d in sorted(days.items()):
        regions = [x[1] for x in sorted(d.get("_regions", []), reverse=True)]
        people = [x[1] for x in sorted(d.get("_people", []), reverse=True) if x[0] >= 2][:8]
        scene_items = sorted(d.get("_scenes", []), reverse=True)
        scenes = [x[1] for x in scene_items if x[1] not in OBJECT_TAGS][:4]
        objects = [x[1] for x in scene_items if x[1] in OBJECT_TAGS][:2]
        scenes = scenes + objects
        camera = sorted(d.get("_cameras", []), reverse=True)[0][1] if d.get("_cameras") else ""
        assets = d.pop("_assets", [])
        photos = [a for a in assets if a[1] == "photo"]
        pool = photos if len(photos) >= 3 else assets
        # 均匀抽 3 张（有脸的优先排前面）
        pool = sorted(pool, key=lambda a: (-a[2],))
        if len(pool) <= 3:
            reps = [a[0] for a in pool]
        else:
            step = len(pool) / 3
            reps = [pool[int(i * step)][0] for i in range(3)]
        out.append({
            "day": day, "n": d["n"], "photos": d["photos"], "videos": d["videos"],
            "region": regions[0] if regions else "", "regions": regions,
            "people": people, "scenes": scenes, "camera": camera,
            "rep_asset_ids": reps,
        })
    return out


def synth_local_caption(d):
    """本地结构化综合 → 一句自然中文描述。"""
    parts = []
    md = re.match(r"(\d{4})-(\d{2})-(\d{2})", d["day"])
    if md:
        date_txt = f"{int(md.group(2))}月{int(md.group(3))}日"
    else:
        date_txt = d["day"]
    wd = day_weekday(d["day"])
    head = f"{date_txt}（{wd}）" if wd else date_txt
    media = []
    if d["photos"]:
        media.append(f"{d['photos']} 张照片")
    if d["videos"]:
        media.append(f"{d['videos']} 段视频")
    where = f"在{d['region']}，" if d["region"] else ""
    parts.append(f"{head}，{where}拍摄了 {'、'.join(media)}。" if media else f"{head}。")
    if d["people"]:
        parts.append(f"画面中出现{('、'.join(d['people']))}。")
    if d["scenes"]:
        parts.append(f"场景包含{('、'.join(d['scenes']))}。")
    if d["camera"]:
        parts.append(f"相机：{d['camera']}。")
    return "".join(parts)


def fts_upsert(con, subject_type, subject_id, text):
    if not text:
        return
    con.execute("DELETE FROM memory_search WHERE subject_type=? AND subject_id=?",
                (subject_type, subject_id))
    con.execute("INSERT INTO memory_search (subject_type, subject_id, text) VALUES (?,?,?)",
                (subject_type, subject_id, text))


def vlm_caption(rep_paths, context):
    """视觉模型叙事描述：最多 3 帧画面 + 结构化上下文。返回 (text, error)。
    统一走 server.llm_chat：本地 Ollama (qwen2.5vl) 优先，DeepSeek 回退。"""
    content = [{"type": "text", "text":
        f"这是同一天的家庭影像（{context}）中的几张代表照片。"
        f"请用中文写一段 80-120 字的描述：谁在做什么、什么活动、可见的物品或地标。"
        f"只输出描述本身，不要开头结尾客套。"}]
    for p in rep_paths:
        b64 = _img_b64(p)
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    if len(content) == 1:
        return None, "no_frames"
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from server import llm_chat, LLM_LOCAL_VISION
        text, err = llm_chat([{"role": "user", "content": content}], vision=True)
        return text, (f"no_balance" if err == "no_balance" else err)
    except Exception as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--vlm", action="store_true", help="跑 VLM 叙事层")
    ap.add_argument("--local", action="store_true", help="强制重跑本地综合层")
    args = ap.parse_args()

    con = sqlite3.connect(DB, timeout=30, isolation_level=None)  # 自动提交防写锁占死
    con.row_factory = sqlite3.Row
    init_tables(con)

    days = load_days(con)
    if args.day:
        days = [d for d in days if d["day"] == args.day]
    done = vlm_done = skipped_vlm = errors = 0
    no_balance = False

    for d in days:
        if args.limit and (done + vlm_done) >= args.limit:
            break
        local = synth_local_caption(d)
        stats = {"n": d["n"], "photos": d["photos"], "videos": d["videos"],
                 "regions": d["regions"], "people": d["people"], "scenes": d["scenes"],
                 "camera": d["camera"]}
        existing = con.execute(
            "SELECT caption_local, caption_vlm FROM day_caption_v0 WHERE day=?",
            (d["day"],)).fetchone()
        caption_vlm = existing["caption_vlm"] if existing else None
        if args.vlm:
            if no_balance:
                skipped_vlm += 1
            elif caption_vlm:
                vlm_done += 1  # 已有，跳过
            else:
                paths = [p for p in (asset_path(con, a) for a in d["rep_asset_ids"]) if p]
                text, err = vlm_caption(paths, local)
                if text:
                    caption_vlm = text
                    vlm_done += 1
                elif err and err.startswith("no_balance"):
                    no_balance = True
                    skipped_vlm += 1
                else:
                    errors += 1
        now = datetime.datetime.now().isoformat(timespec="seconds")
        con.execute("""INSERT INTO day_caption_v0 (day, caption_local, caption_vlm, vlm_model,
                        rep_asset_ids, stats_json, updated_at)
                       VALUES (?,?,?,?,?,?,?)
                       ON CONFLICT(day) DO UPDATE SET caption_local=excluded.caption_local,
                         caption_vlm=COALESCE(excluded.caption_vlm, caption_vlm),
                         rep_asset_ids=excluded.rep_asset_ids,
                         stats_json=excluded.stats_json, updated_at=excluded.updated_at""",
                    (d["day"], local, caption_vlm,
                     "qwen2.5vl:7b" if caption_vlm and args.vlm else None,
                     json.dumps(d["rep_asset_ids"]), json.dumps(stats, ensure_ascii=False), now))
        fts_upsert(con, "day", d["day"], f"{local} {caption_vlm or ''}".strip())
        done += 1
        if done % 100 == 0:
            print(f"[captions] local={done} vlm={vlm_done}", flush=True)

    print(f"[captions] 完成: local={done} vlm={vlm_done} vlm_skip={skipped_vlm} "
          f"errors={errors} no_balance={no_balance}", flush=True)


if __name__ == "__main__":
    main()
