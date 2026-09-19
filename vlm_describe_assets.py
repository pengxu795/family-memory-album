#!/usr/bin/env python3
"""无标签资产 VLM 描述兜底（本地 qwen2.5vl 优先，DeepSeek 回退）。

对 scene_tag_v0 里没有任何标签的资产（全库约 5,104 张，多为日常室内照，
SIGLIP 置信度不过阈值）逐张让 VLM 看图输出：
  {"desc": "一句话中文描述", "scene": [...], "objects": [...]}

- 描述写 asset_description_v0（新表，可断点续跑）
- scene/objects 命中词表的写 scene_tag_v0（source='VLM', confidence=0.75），
  INSERT OR IGNORE 不覆盖已有标签 → 问答/物品检索/天描述直接可用
- 视频用 0.5s 帧缩略图

用法：
  python3 vlm_describe_assets.py --limit 20   # 冒烟
  python3 vlm_describe_assets.py              # 全量（后台）
"""
import argparse
import base64
import datetime
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("FF_DB_PATH") or os.path.join(os.environ.get("FF_DATA_DIR") or os.path.join(HERE, "data"), "family_memory.db")
# 2026-09-19：认 FF_DATA_DIR（容器内 /data），别写死脚本目录
THUMBS = os.path.join(os.environ.get("FF_DATA_DIR") or os.path.join(HERE, "data"), "thumbs_mvp")
# 必须与 server.py 保持一致：server 生成的缩略图文件名带档位后缀 _t{edge}
# 2026-09-01 墙面改 480 档，1600 档已废弃删除；VLM 看图 480 足够，避免回填 1600
THUMB_EDGE = 480
# 2026-09-19：容器/Mac 通吃——env > PATH 探测 > 常见位置；不再写死 homebrew 路径
def _find_ffmpeg():
    import shutil
    return (os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg")
            or (p if os.path.exists(p := "/usr/bin/ffmpeg") else None)
            or (p if os.path.exists(p := "/opt/homebrew/bin/ffmpeg") else ""))
FFMPEG_BIN = _find_ffmpeg()
ERR_REASON = {}   # 失败原因计数，便于定位"全量报错"

# 词表（与 server.py 单一事实源保持一致；避免 import server 全量副作用，这里按值拷贝并校验）
SCENE_VOCAB = {"山景", "海景", "美食", "花草", "夜景", "雪景", "建筑", "江河", "宠物",
               "儿童", "生日", "车辆", "室内", "合影", "截图"}
OBJECT_VOCAB = {"书包", "行李箱", "自行车", "火车", "飞机", "游船", "婴儿车", "玩具",
                "气球", "风筝", "帐篷", "烧烤", "火锅", "灯笼", "滑雪", "乐器"}


def init_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS asset_description_v0 (
        asset_id TEXT PRIMARY KEY,
        description TEXT NOT NULL,
        model TEXT,
        created_at TEXT NOT NULL)""")


def thumb_candidates(asset_id):
    """server.py 当前按 {id}_t{edge}.jpg 存；历史缓存是 {id}.jpg。两种都认。"""
    stem = asset_id[6:]
    return [os.path.join(THUMBS, f"{stem}_t{THUMB_EDGE}.jpg"),
            os.path.join(THUMBS, f"{stem}.jpg")]


def source_row(asset_id):
    """取该资产最大的一份原片绝对路径 + 媒体类型。"""
    try:
        con = sqlite3.connect(DB, timeout=10)
        row = con.execute("""SELECT mf.absolute_path, ma.media_type FROM media_file mf
            JOIN media_asset ma USING(asset_id)
            WHERE mf.asset_id=? ORDER BY mf.byte_size DESC LIMIT 1""", (asset_id,)).fetchone()
        con.close()
    except Exception:
        return None, None
    return (row[0], row[1]) if row else (None, None)


def ensure_thumb(asset_id):
    """确保缩略图存在并返回路径；缺失时按 server.get_thumb 相同方式现生成。

    背景：缩略图原本只在浏览器请求 /thumb 时懒生成，VLM 脚本直接读会 100% miss；
    且 server 早已改用 _t{档位} 命名，脚本仍按无后缀名查找，导致"点了没反应"。
    这里复用同一套文件名和生成逻辑，VLM 与网页共用一份缓存，互不重复占盘。
    """
    cands = thumb_candidates(asset_id)
    for p in cands:
        if os.path.exists(p):
            return p
    path, media_type = source_row(asset_id)
    if not path or not os.path.exists(path):
        return None                      # 原片不可读（NAS 掉线/已删除）
    out = cands[0]
    try:
        os.makedirs(THUMBS, exist_ok=True)
        if media_type == "photo":
            # 2026-09-19：sips 是 macOS 专属，容器内没有——统一用 ffmpeg 缩图（与视频同路）
            cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                   "-i", path, "-vf", f"scale='min(1,{THUMB_EDGE}/iw)':-2",
                   "-q:v", "3", "-f", "image2", out]
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        else:
            # 与 server 一致：HEVC non-full-range 需 -strict unofficial；超短视频回退第 0 帧
            cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
                   "-ss", "0.5", "-i", path, "-frames:v", "1",
                   "-vf", f"scale={THUMB_EDGE}:-2", "-f", "image2", out]
            try:
                subprocess.run(cmd, capture_output=True, timeout=60, check=True)
            except Exception:
                cmd[cmd.index("0.5")] = "0"
                subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        return out if os.path.exists(out) else None
    except Exception:
        return None


def img_b64(path, px=768):
    """缩略图读入 → 再压到 px 边长（VLM 输入不必太大，7B 量化速度优先）。"""
    import subprocess as sp
    import tempfile
    try:
        with open(path, "rb") as f:
            data = f.read()
        if len(data) <= 400_000:
            return base64.b64encode(data).decode()
        t = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        t.close()
        if shutil.which("sips"):   # macOS 快路径；容器/Linux 走 ffmpeg（2026-09-19）
            sp.run(["sips", "-Z", str(px), "-s", "format", "jpeg", path, "--out", t.name],
                   capture_output=True, timeout=20, check=True)
        else:
            sp.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
                    "-i", path, "-frames:v", "1", "-vf", f"scale={px}:-2", t.name],
                   capture_output=True, timeout=20, check=True)
        with open(t.name, "rb") as f:
            out = base64.b64encode(f.read()).decode()
        os.unlink(t.name)
        return out
    except Exception:
        return None


def llm_vlm_describe(b64):
    sys.path.insert(0, HERE)
    from server import llm_chat
    scene_list = "/".join(sorted(SCENE_VOCAB))
    obj_list = "/".join(sorted(OBJECT_VOCAB))
    prompt = (
        "请看这张家庭照片（视频则是其封面帧）。只输出一个 JSON 对象，不要任何其他文字：\n"
        '{"desc": "一句话中文描述（15-40字，写清楚谁在做什么、什么场景、可见物品）",\n'
        ' "scene": ["从这些里选0-2个: ' + scene_list + '"],\n'
        ' "objects": ["从这些里选0-3个: ' + obj_list + '"]}\n'
        "规则：室内生活照选「室内」；多人合影选「合影」；截图/文档/聊天记录选「截图」；"
        "都不符合就给空数组。desc 必须写。")
    messages = [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}]
    text, err = llm_chat(messages, vision=True, temperature=0.1, max_tokens=300)
    if not text:
        return None, err
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no_json"
    try:
        d = json.loads(m.group(0))
    except Exception:
        cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
        try:
            d = json.loads(cleaned)
        except Exception:
            return None, "bad_json"
    desc = str(d.get("desc") or "").strip()
    if not desc:
        return None, "no_desc"
    scene = [s for s in (d.get("scene") or []) if s in SCENE_VOCAB][:2]
    objects = [o for o in (d.get("objects") or []) if o in OBJECT_VOCAB][:3]
    return {"desc": desc, "scene": scene, "objects": objects}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--day", help="只跑某天")
    ap.add_argument("--shard", help="分片 i/n, 如 0/3; 多进程并行用")
    args = ap.parse_args()

    con = sqlite3.connect(DB, timeout=60, isolation_level=None)  # 自动提交防写锁占死
    con.row_factory = sqlite3.Row
    init_tables(con)

    sql = """SELECT ma.asset_id, ma.media_type, substr(ma.capture_time,1,10) day
             FROM media_asset ma
             WHERE ma.asset_id NOT IN (SELECT DISTINCT asset_id FROM scene_tag_v0)
               AND ma.asset_id NOT IN (SELECT asset_id FROM asset_description_v0)"""
    params = []
    if args.day:
        sql += " AND substr(ma.capture_time,1,10)=?"
        params.append(args.day)
    sql += " ORDER BY (ma.capture_time IS NULL), ma.capture_time"
    rows = con.execute(sql, params).fetchall()
    if args.shard:
        # 分片: i/n 取 rows[i::n], 各分片互不重叠, 可多进程并行跑
        try:
            i, n = (int(x) for x in args.shard.split("/"))
            assert 0 <= i < n
        except Exception:
            print(f"[vlm-describe] 非法 --shard {args.shard}, 应形如 0/3", flush=True)
            con.close()
            return
        rows = rows[i::n]
        print(f"[vlm-describe] 分片 {args.shard}: 本片 {len(rows)} 张", flush=True)
    if args.limit:
        rows = rows[:args.limit]
    print(f"[vlm-describe] 待处理 {len(rows)} 张", flush=True)

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
                # ollama 瞬时抖动(重启/卡顿)会让请求落到 DeepSeek 回退上报 no_balance;
                # 等 30s 重试 5 次, 全失败才中止(退出码 2 供上层识别)
                llm_down = getattr(main, "_llm_down", 0)
                if llm_down < 5:
                    setattr(main, "_llm_down", llm_down + 1)
                    print(f"[vlm-describe] LLM 不可用({err}), 30s 后重试 "
                          f"({llm_down + 1}/5)", flush=True)
                    time.sleep(30)
                    continue
                print(f"[vlm-describe] LLM 连续不可用({err})，中止", flush=True)
                sys.exit(2)
            continue
        now = datetime.datetime.now().isoformat(timespec="seconds")
        main._llm_down = 0  # 成功即重置连续失败计数
        # 写库重试: 人脸检测等其他长任务会把写锁反复占住(25资产/事务),
        # 单次 INSERT 可能要等好几分钟, 重试上限放宽到 10 分钟
        for attempt in range(60):
            try:
                con.execute("INSERT OR REPLACE INTO asset_description_v0 VALUES (?,?,?,?)",
                            (r["asset_id"], parsed["desc"], "qwen2.5vl:7b", now))
                for tag in parsed["scene"] + parsed["objects"]:
                    con.execute("INSERT OR IGNORE INTO scene_tag_v0 VALUES (?,?,?,?,?)",
                                (r["asset_id"], tag, "VLM", 0.75, now))
                tagged += 1
                break
            except sqlite3.OperationalError as e:
                if "locked" not in str(e) or attempt == 59:
                    raise
                if attempt % 6 == 0:
                    print(f"[vlm-describe] 写库被锁({e}), 第{attempt+1}/60次等待重试...", flush=True)
                time.sleep(10)
        done += 1
        if done % 50 == 0:
            dt = (datetime.datetime.now() - now0).total_seconds()
            eta = (len(rows) - done - errors) * (dt / max(done, 1)) / 60
            print(f"[vlm-describe] done={done} errors={errors} tags={tagged} "
                  f"速度={dt/max(done,1):.1f}s/张 预计剩余{eta:.0f}分钟", flush=True)

    dt = (datetime.datetime.now() - now0).total_seconds()
    print(f"[vlm-describe] 完成: done={done} errors={errors} tags={tagged} "
          f"耗时{dt/60:.1f}分钟", flush=True)
    if ERR_REASON:
        print("[vlm-describe] 失败原因: " +
              " | ".join(f"{k}={v}" for k, v in sorted(ERR_REASON.items(), key=lambda x: -x[1])),
              flush=True)
    # 收尾统计
    left = con.execute(
        "SELECT COUNT(*) FROM media_asset WHERE asset_id NOT IN "
        "(SELECT DISTINCT asset_id FROM scene_tag_v0) "
        "AND asset_id NOT IN (SELECT asset_id FROM asset_description_v0)"
    ).fetchone()[0]
    print(f"[vlm-describe] 剩余无描述无标签资产: {left}", flush=True)
    con.close()


if __name__ == "__main__":
    main()
