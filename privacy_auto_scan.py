#!/usr/bin/env python3
"""隐私内容自动检测（两级级联算法，2026-09-07；v2 视频多帧增强）。

目标：把「个人暴露 / 私密」的照片和视频自动移入隐私相册（privacy_v0），
不碰原片、软隐藏、随时可移出。

算法（优化点：级联 = 快筛 + 精判，避免全库跑视觉模型）：
  Level 1  文本信号（全量、毫秒级/张）
    对 VLM 画面描述（asset_description_v0）+ 原始文件名（media_file.filename）
    做加权关键词评分：
      - 强规则（如 裸体/全裸/性行为/露点…）≥10 分 → 直接移入，无需看图
      - 中规则（内衣/泳装/浴缸/睡衣/试衣…）5~9 分 → 进 Level 2
      - <5 分 → 跳过（不写库，下次增量再评，文本重评成本可忽略）
  Level 2  VLM 看图复核（只对 Level 1 候选，本地 qwen2.5vl 优先）
    让视觉模型判隐私等级 0-3：
      0=普通（含海滩泳池普通泳装照） 1=日常无隐私问题
      2=居家私密状态（睡衣/内衣/浴室/镜前自拍） 3=身体明显暴露
    level>=2 才移入。家庭海滩泳装合影（level 0/1）不会被误藏。

v2 优化（2026-09-07 上午，应「视频也要覆盖 + 算法优化」需求）：
  1. 视频多帧采样：封面帧 0.5s 之外再抽 25%/50%/75% 时长处共 4 帧，
     一次 VLM 调用多图判定（多图失败自动回退逐帧取最高等级）——
     私密内容出现在视频中段的不再漏检。
  2. 无描述资产兜底：pending 改 LEFT JOIN——
     - 无描述的视频：文本层无从筛，直接进 Level 2 多帧抽检（约几百个，成本可控）
     - 无描述的照片：仅文件名命中才进 Level 2，其余等 VLM 描述补齐后增量覆盖
  3. 文本规则补词：浴巾/浴袍/性感/情趣/吊带/真空/只穿/贴身（中规则），
     文件名加 sex|porn|nsfw|擦边 等英文信号。
  4. 决策表加 media_type 列，方便统计和 UI 展示。

决策可追溯：每条结果写 privacy_auto_v0（decision=auto_added/skipped）。
- 已有决策的资产不重扫（断点续跑）
- 用户在隐私相册里「移出」→ server 把 decision 改为 manual_removed → 不复藏

用法：
  python3 privacy_auto_scan.py --dry-run --limit 200   # 试跑看命中
  python3 privacy_auto_scan.py                          # 正式全量
  python3 privacy_auto_scan.py --rescan                 # 无视历史决策重扫
"""
import argparse
import datetime
import json
import os
import re
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.environ.get("FF_DB_PATH") or os.path.join(os.environ.get("FF_DATA_DIR") or os.path.join(HERE, "data"), "family_memory.db")

# ============ Level 1：文本加权规则（单一事实源，server.py 引用同名常量） ============
# 强规则：单条命中即直接移入（分数≥10）。措辞从严：宁缺勿滥，误藏靠 VLM 兜底。
# 注意语境约束：「电线裸露/墙壁裸露/室内衣帽间」这类误报必须挡在文本层。
TEXT_STRONG_RULES = [
    (r"裸体|全裸|赤裸|一丝不挂|裸睡|赤身", 10, "裸体"),
    (r"(?:身体|上身|胸部|肌肤|肩膀|背部|双腿).{0,4}(?:裸露|袒露)|"
     r"(?:裸露|袒露)(?:的)?(?:上身|胸部|身体)|光着上身|上身.{0,4}裸|裸.{0,2}上身|"
     r"没穿衣服|未穿衣服|脱光", 10, "身体裸露"),
    (r"性行为|性爱|做爱|交合|缠绵|亲热|激情", 10, "亲密行为"),
    (r"露点|走光|透视装", 10, "暴露"),
]
# 中规则：累计 5~9 分进 VLM 复核。这些词语境双关（海滩泳装 vs 私密自拍），交给视觉判。
TEXT_MID_RULES = [
    (r"(?<!室)内衣(?!帽)|内裤|文胸|胸罩|bra", 6, "内衣"),
    (r"比基尼|泳装|泳衣|泳裤|游泳衣", 5, "泳装"),
    (r"浴缸|泡澡|洗澡|沐浴|淋浴|浴室|澡堂", 5, "浴室"),
    (r"浴巾|浴袍|裹着毛巾|围着毛巾|毛巾裹身", 5, "浴巾"),
    (r"睡衣|睡裙|睡袍|居家服", 3, "睡衣"),
    (r"换衣|试衣|更衣|穿衣镜", 4, "换衣"),
    (r"激吻|舌吻", 6, "亲吻"),
    (r"镜子自拍|全身镜|镜前", 3, "镜前自拍"),
    (r"性感|情趣|吊带(?:衫|裙|背心)|真空|贴身(?:衣物|自拍)|只穿着", 4, "着装暴露"),
]
# 文件名信号：相册翻拍/导出文件常带原始命名，命中加分（只作锦上添花，不单独立案）
FILENAME_RULES = [
    (r"private|私密|sexy|bath|洗澡|浴室", 4, "文件名"),
    (r"porn|nsfw|xxx|sex|nude|naked|擦边|18禁|horny|camgirl|onlyfans", 6, "文件名敏感词"),
]
STRONG_THRESHOLD = 10   # ≥ 此分直接移入
REVIEW_THRESHOLD = 5    # ≥ 此分（且 <STRONG）进 VLM 复核

_RULES_COMPILED = None


def _rules():
    global _RULES_COMPILED
    if _RULES_COMPILED is None:
        _RULES_COMPILED = ([(re.compile(p), w, t) for p, w, t in TEXT_STRONG_RULES],
                           [(re.compile(p), w, t) for p, w, t in TEXT_MID_RULES],
                           [(re.compile(p), w, t) for p, w, t in FILENAME_RULES])
    return _RULES_COMPILED


def score_text(desc, filename=None):
    """Level 1 评分。返回 (score, 命中理由列表)。"""
    strong, mid, fname = _rules()
    text = (desc or "").strip()
    hits, score = [], 0
    for rx, w, tag in strong:
        if rx.search(text):
            score += w
            hits.append(tag)
    for rx, w, tag in mid:
        if rx.search(text):
            score += w
            hits.append(tag)
    if filename:
        for rx, w, tag in fname:
            if rx.search(filename):
                # 文件名误报率高（bath=bathroom door?），普通信号只加 2 分；
                # 但 sex/porn/nsfw 等敏感词命名本身就是强信号，加 5 分够进 VLM 复核
                score += 5 if tag == "文件名敏感词" else 2
                hits.append(tag)
    return score, hits


# ============ Level 2：VLM 隐私等级复核（照片单帧 / 视频多帧） ============
def video_frame_b64s(asset_id, k=4):
    """视频抽 k 帧（0.5s 封面 + 均分 25%/50%/75% 时长处），返回 [b64,...]。

    v2 核心优化：私密内容常出现在视频中段，只看封面帧会漏检。
    ffmpeg 直接输出到管道，不留临时文件；抽不到的帧跳过（HEVC/超短视频回退 0.5s/1.0s）。
    库里 duration_seconds 覆盖率低（~12%），缺失时用 ffprobe 现场探测。
    """
    import base64
    import subprocess
    from vlm_describe_assets import FFMPEG_BIN, source_row
    path, mtype = source_row(asset_id)
    if not path or mtype != "video" or not os.path.exists(path):
        return []
    dur = None
    try:
        con = sqlite3.connect(DB, timeout=10)
        row = con.execute("SELECT duration_seconds FROM media_asset WHERE asset_id=?",
                          (asset_id,)).fetchone()
        con.close()
        dur = row[0] if row else None
    except Exception:
        pass
    if not dur or dur <= 3:  # 库里没有就用 ffprobe 现场探
        try:
            probe = [FFMPEG_BIN.replace("ffmpeg", "ffprobe"), "-v", "error",
                     "-show_entries", "format=duration", "-of", "csv=p=0", path]
            r = subprocess.run(probe, capture_output=True, timeout=30, text=True)
            dur = float(r.stdout.strip() or 0) or None
        except Exception:
            dur = None
    if dur and dur > 3:
        pts = [0.5] + [dur * p for p in (0.25, 0.5, 0.75)][: max(1, k - 1)]
    else:  # 时长探不到或超短视频：退化为 0.5s / 1.0s 两帧
        pts = [0.5, 1.0][: max(1, k)]
    out = []
    for ts in pts:
        try:
            cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
                   "-ss", f"{ts:.2f}", "-i", path, "-frames:v", "1",
                   "-vf", "scale=768:-2", "-f", "image2", "pipe:1"]
            r = subprocess.run(cmd, capture_output=True, timeout=60)
            if r.stdout and len(r.stdout) > 2000:  # 太小多半是黑帧/损坏帧
                out.append(base64.b64encode(r.stdout).decode())
        except Exception:
            continue
    return out


def _vlm_judge_once(b64s, is_video):
    """一次 VLM 调用：单图（照片）或多图（视频采样帧）。返回 (level, reason) 或 (None, err)。"""
    from server import llm_chat
    frame_note = ("（下面按时间顺序给出这段视频的多个采样帧，"
                  "请以其中最敏感的一帧为准）" if len(b64s) > 1 else "（照片则为其本身）")
    prompt = (
        "你是家庭相册的隐私审核员。看这张照片或视频采样帧，判断隐私敏感等级。"
        f"{frame_note}"
        "只输出一个 JSON 对象，不要任何其他文字：\n"
        '{"level": 0到3的整数, "reason": "10字内理由"}\n'
        "等级定义：\n"
        "0 = 普通家庭/风景/合影（含海滩、泳池里的普通泳装合影）\n"
        "1 = 日常内容，无隐私问题\n"
        "2 = 居家私密状态：睡衣、内衣、浴室浴缸、镜前自拍、试衣\n"
        "3 = 身体明显暴露或高度私密内容")
    messages = [{"role": "user", "content":
                 [{"type": "text", "text": prompt}] +
                 [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
                  for b in b64s]}]
    text, err = llm_chat(messages, vision=True, temperature=0.0, max_tokens=120)
    if not text:
        return None, err or "no_reply"
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no_json"
    try:
        d = json.loads(m.group(0))
        level = int(d.get("level", -1))
    except Exception:
        return None, "bad_json"
    if not 0 <= level <= 3:
        return None, "bad_level"
    return level, str(d.get("reason") or "")[:30]


def vlm_privacy_judge(b64s, is_video=False):
    """看图/看帧判隐私等级。b64s：照片 1 帧，视频多帧。
    多图调用失败时回退逐帧判定取最高等级（qwen2.5vl 多图偶发截断的兜底）。"""
    if not b64s:
        return None, "no_frames"
    level, jerr = _vlm_judge_once(b64s, is_video)
    if level is not None:
        return level, jerr
    if len(b64s) == 1:
        return None, jerr
    # 回退：逐帧单图判定，取 max（漏检代价 > 误检代价，误检可移出且不复发）
    best, best_reason, last_err = None, "", jerr
    for b in b64s:
        lv, rsn = _vlm_judge_once([b], False)
        if lv is None:
            last_err = rsn
            continue
        if best is None or lv > best:
            best, best_reason = lv, rsn
    if best is None:
        return None, last_err
    return best, best_reason + "(逐帧)"


# ============ 决策表 ============
def init_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS privacy_auto_v0 (
        asset_id TEXT PRIMARY KEY,
        score REAL,
        level INTEGER,
        reasons TEXT,
        decision TEXT NOT NULL,
        decided_by TEXT NOT NULL,
        created_at TEXT NOT NULL
    )""")
    # v2：加 media_type 列（老库平滑升级；server.py 建表语句已同步含此列）
    cols = {r[1] for r in con.execute("PRAGMA table_info(privacy_auto_v0)")}
    if "media_type" not in cols:
        con.execute("ALTER TABLE privacy_auto_v0 ADD COLUMN media_type TEXT DEFAULT ''")


def pending_rows(con, rescan=False, limit=None):
    """待检测资产：不在隐私相册、无历史决策（rescan 时忽略这两条）。

    v2：LEFT JOIN 描述表——无描述资产也进候选：
    - 无描述的视频 → scan() 里直接进 Level 2 多帧抽检（文本层无从筛）
    - 无描述的照片 → 仅文件名敏感词命中才值得花 VLM，其余等描述补齐后增量覆盖
    """
    sql = """SELECT ma.asset_id, ma.media_type, d.description,
                    (SELECT mf.filename FROM media_file mf
                     WHERE mf.asset_id=ma.asset_id
                     ORDER BY mf.byte_size DESC LIMIT 1) AS filename
             FROM media_asset ma
             LEFT JOIN asset_description_v0 d USING(asset_id)"""
    if not rescan:
        sql += """ WHERE ma.asset_id NOT IN (SELECT asset_id FROM privacy_v0)
                     AND ma.asset_id NOT IN (SELECT asset_id FROM privacy_auto_v0)"""
    sql += " ORDER BY (ma.capture_time IS NULL), ma.capture_time"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return con.execute(sql).fetchall()


def scan(limit=None, dry_run=False, rescan=False, full_vlm=False, log=print):
    """主流程。返回摘要 dict（脚本模式最后一行打印成 JSON 供 server 解析）。

    full_vlm=True：无视文本评分，全部候选过 VLM（约 1-2s/张，全库需数小时，
    供夜间兜底——描述写得含蓄的私密照只有视觉判定能抓住）。"""
    from vlm_describe_assets import ensure_thumb, img_b64
    con = sqlite3.connect(DB, timeout=60, isolation_level=None)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    init_tables(con)

    rows = pending_rows(con, rescan=rescan, limit=limit)
    now = datetime.datetime.now().isoformat(timespec="seconds")
    added = skipped = errors = 0
    vlmd = vlm_videos = vlm_photos = 0
    detail = []

    for r in rows:
        asset_id = r["asset_id"]
        mtype = r["media_type"] or "photo"
        desc = r["description"] or ""
        filename = r["filename"] or ""
        has_desc = bool(desc.strip())
        score, hits = score_text(desc, filename)
        reasons = "+".join(hits) if hits else ""

        # 无描述资产的处理（v2）：
        #   视频 → 文本层无从筛，直接进 Level 2 多帧抽检（约几百个，成本可控）
        #   照片 → 分数不够（含无描述且文件名没敏感词）自然不进下面的分支，
        #          等 VLM 描述补齐后会被增量扫描覆盖
        no_desc_video = (not has_desc) and mtype == "video"

        if not full_vlm and score >= STRONG_THRESHOLD:
            # Level 1 强规则：直接移入（描述已经写明裸露/亲密，无需再花 VLM）
            if dry_run:
                detail.append({"id": asset_id, "type": mtype, "score": score,
                               "by": "text", "reasons": reasons})
                added += 1
                continue
            _record(con, asset_id, mtype, score, None, reasons, "auto_added", "text_rule", now)
            added += 1
        elif full_vlm or no_desc_video or score >= REVIEW_THRESHOLD:
            # Level 2：VLM 看图复核（视频多帧采样，照片单帧）
            vlmd += 1
            if mtype == "video":
                vlm_videos += 1
                b64s = video_frame_b64s(asset_id)
            else:
                vlm_photos += 1
                thumb = ensure_thumb(asset_id)
                b64s = [img_b64(thumb)] if thumb else []
            if not b64s:
                errors += 1
                log(f"[privacy-auto] 抽帧失败 {asset_id} ({mtype})")
                continue
            level, jerr = vlm_privacy_judge(b64s, is_video=(mtype == "video"))
            if level is None:
                errors += 1
                log(f"[privacy-auto] VLM 判定失败 {asset_id}: {jerr}")
                continue
            if level >= 2:
                if dry_run:
                    detail.append({"id": asset_id, "type": mtype, "score": score,
                                   "level": level, "by": "vlm", "reasons": reasons or jerr})
                    added += 1
                    continue
                _record(con, asset_id, mtype, score, level, reasons or jerr,
                        "auto_added", "vlm", now)
                added += 1
            else:
                if dry_run:
                    continue
                # 记录 skipped：避免下次全量重跑 VLM（文本候选集是稳定幂等的）
                _record(con, asset_id, mtype, score, level, reasons, "skipped", "vlm", now)
                skipped += 1
        # score < REVIEW_THRESHOLD：不写库，留待下次增量评（纯文本零成本）

    summary = {"added": added, "skipped": skipped, "vlm_checked": vlmd,
               "vlm_videos": vlm_videos, "vlm_photos": vlm_photos,
               "errors": errors, "scanned": len(rows), "dry_run": bool(dry_run)}
    con.close()
    return summary, detail


def _record(con, asset_id, media_type, score, level, reasons, decision, decided_by, now):
    """写决策表 + decision=auto_added 时同步移入隐私相册（互斥清回收站）。"""
    for attempt in range(30):
        try:
            con.execute("""INSERT INTO privacy_auto_v0(asset_id,score,level,reasons,decision,
                               decided_by,created_at,media_type)
                           VALUES(?,?,?,?,?,?,?,?)
                           ON CONFLICT(asset_id) DO UPDATE SET score=excluded.score,
                             level=excluded.level, reasons=excluded.reasons,
                             decision=excluded.decision, decided_by=excluded.decided_by,
                             created_at=excluded.created_at,
                             media_type=excluded.media_type""",
                        (asset_id, score, level, reasons, decision, decided_by, now, media_type))
            if decision == "auto_added":
                con.execute("DELETE FROM recycle_v0 WHERE asset_id=?", (asset_id,))
                con.execute("INSERT OR IGNORE INTO privacy_v0(asset_id,created_at) VALUES(?,?)",
                            (asset_id, now))
            break
        except sqlite3.OperationalError as e:
            if "locked" not in str(e) or attempt == 29:
                raise
            import time
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="只统计不写库")
    ap.add_argument("--limit", type=int, help="只处理前 N 张")
    ap.add_argument("--rescan", action="store_true", help="无视历史决策重扫（会重新 VLM）")
    ap.add_argument("--full-vlm", action="store_true",
                    help="全部候选过 VLM（不看文本评分，慢，供夜间兜底全量扫描）")
    ap.add_argument("--detail", action="store_true", help="dry-run 时打印命中明细")
    args = ap.parse_args()

    sys.path.insert(0, HERE)
    summary, detail = scan(limit=args.limit, dry_run=args.dry_run,
                           rescan=args.rescan, full_vlm=args.full_vlm)
    if args.detail and detail:
        for d in detail[:100]:
            print("  ", json.dumps(d, ensure_ascii=False))
        if len(detail) > 100:
            print(f"   ... 共 {len(detail)} 条，仅显示前 100")
    print("PRIVACY_AUTO_RESULT: " + json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
