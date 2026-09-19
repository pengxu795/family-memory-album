#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""增量富化流水线（2026-09-16 新增）

背景：新照片导入后，除了入库索引 / 人脸 / SigLIP 向量 / 4 条文件名过滤之外，
其余模块（精确去重、画质、相似分组、择优、语义过滤）**从来没有自动跑过**——
离线模块是一次性批处理，`backfill_quality.py` 还要人工触发。结果就是
「相似照片」「已过滤内容」两个模块只对 2026-09-09 那批历史数据有效，
之后每天备份进来的新照片全都停在门口。本脚本把缺口补齐。

设计原则：
1. **幂等 + 增量**：每个步骤只处理「缺这项数据」的资产，重跑无损。
2. **只增不改**：不重建、不打散已有的相似组（2026-09-16 修复过一轮，
   再全量重造风险大于收益）。新照片要么**追加**进已有组，要么自己成新组。
3. **白名单优先**：用户在界面上「恢复」过的资产永不二次过滤。
4. **成本可控**：sha256 只算「文件大小出现碰撞」的（大小唯一的文件不可能重复）；
   相似比对只在同来源 ± 时间窗内做；语义过滤无 provider 时优雅跳过。
5. **单一写者**：由 server.py 以互斥方式拉起，避免与 autoscan 抢 SQLite 写锁。
6. **判定留痕与过滤解耦**：视觉判定的结果写 `asset_vision_check_v0`，
   **绝不往 asset_filter_v0 写哨兵行** —— 那张表就是过滤事实表，
   写进去等于把照片藏了。

步骤（可单独跑，`--all` = 按依赖顺序全跑）：
  --hash     sha256 精确去重基础（大小分桶预筛，省 95% IO）
  --blur     模糊检测 → asset_quality_v0.blur_label/sharp_score
  --junk     确定性文件名/路径过滤（白名单免检）
  --similar  相似分组：sha256 精确 > dHash 近重复 > SigLIP 语义，星形聚类
  --pick     择优：把模糊/美学分接入代表张选择（只动 is_best，不动成员）
  --vision   VLM 语义过滤（截图/文档/拍屏/白底商品图），需配 provider

用法：
  python3 enrich.py --status                 # 只看欠账，不写库
  python3 enrich.py --all                    # 全跑
  python3 enrich.py --all --dry-run          # 演练
  python3 enrich.py --blur --limit 50        # 单步 + 限量
"""
import argparse, base64, bisect, hashlib, json, math, os, sqlite3, sys, time, urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FF_DATA_DIR") or (ROOT / "data"))
DB = Path(os.environ.get("FF_DB_PATH") or (DATA_DIR / "family_memory.db"))
THUMBS = Path(os.environ.get("FF_THUMBS_DIR") or (DATA_DIR / "thumbs_mvp"))

MODEL_VERSION = "quality-v1-20260909"   # 与 backfill_quality.py 一致，别造新版本号

# ── 模糊阈值（t480 缩略图口径，与 blur_detector.py 完全一致，勿各行其是）
T_BLURRY, T_SOFT = 300.0, 1200.0

# ── 相似判定阈值（与 similar_dedup.py 一致）
DHASH_MAX = 6            # dHash 汉明距离 ≤6 视为近重复
SIGLIP_COS_MIN = 0.96    # 嵌入余弦阈值
SIGLIP_WIN_SEC = 90      # 嵌入引擎时间邻域
PHASH_WIN_SEC = 48 * 3600    # dHash 引擎时间邻域（48h，同来源）
APPEND_DHASH_MAX = 4     # 追加进已有组时更严（改动既有组要保守）
APPEND_WIN_SEC = 6 * 3600
CLUSTER_MAX = 40         # 单簇硬上限，防病态大簇

# ── 语义过滤
VISION_MIN_CONF = 0.80
VISION_MIN_INTERVAL_SEC = 1.0   # 批量语义判定的请求间隔（免费档连发即 429）

STEPS = ["hash", "blur", "junk", "similar", "pick", "vision"]


def log(*a):
    print(*a, flush=True)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect():
    con = sqlite3.connect(str(DB), timeout=120)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=120000")
    return con


def thumb_path(asset_id):
    """资产 → t480 缩略图（优先 _t480.jpg，兜底同名 .jpg）。"""
    stem = asset_id[6:] if asset_id.startswith("asset_") else asset_id
    for cand in (f"{stem}_t480.jpg", f"{stem}.jpg"):
        p = THUMBS / cand
        if p.exists():
            return p
    return None


def as_ts(t):
    """capture_time → epoch 秒。时区不统一（+08:00 / +00:00 / naive），naive 按 +8 处理。"""
    if not t:
        return None
    try:
        dt = datetime.fromisoformat(str(t))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        return dt.timestamp()
    except Exception:
        return None


# ══════════════════════════════════════════════════ 表结构

def _ensure_tables(con):
    """补齐富化要用到的表/列（幂等）。"""
    # ⚠️ 与 schema_raw.json 里真库的定义**逐字对齐**（含 evidence_kind 的 CHECK
    # 与三列主键）。曾经这里是个宽松版本（无 CHECK、两列主键），于是
    # 「enrich 自建的新库」与「线上库」行为不一致：宽松库上写 evidence_kind='rule'
    # 不报错，线上库上违反 CHECK 被 INSERT OR IGNORE 静默吞掉（2026-09-16 踩坑）。
    # 表结构一旦两套，这类 bug 只会在其中一端出现，极难排查。
    con.execute("""CREATE TABLE IF NOT EXISTS asset_filter_v0 (
        asset_id TEXT NOT NULL REFERENCES media_asset(asset_id),
        filter_reason TEXT NOT NULL,
        evidence_kind TEXT NOT NULL CHECK(evidence_kind IN ('path','filename','visual','ocr','user')),
        evidence_value TEXT NOT NULL,
        confidence REAL NOT NULL,
        rule_version TEXT NOT NULL,
        created_at TEXT NOT NULL,
        PRIMARY KEY(asset_id,filter_reason,rule_version))""")
    con.execute("""CREATE TABLE IF NOT EXISTS asset_allowlist_v0 (
        asset_id TEXT PRIMARY KEY, note TEXT, created_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS asset_quality_v0 (
        asset_id TEXT PRIMARY KEY, blur_label TEXT, sharp_score REAL,
        aesthetic REAL, model_version TEXT,
        created_at TEXT DEFAULT (datetime('now')))""")
    con.execute("""CREATE TABLE IF NOT EXISTS asset_similar_group_v0 (
        group_id TEXT PRIMARY KEY, best_asset_id TEXT NOT NULL, source_id TEXT,
        start_time TEXT, end_time TEXT, asset_count INTEGER NOT NULL,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, engine TEXT)""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(asset_similar_group_v0)")}
    if "engine" not in cols:
        con.execute("ALTER TABLE asset_similar_group_v0 ADD COLUMN engine TEXT")
    con.execute("""CREATE TABLE IF NOT EXISTS asset_similar_member_v0 (
        group_id TEXT NOT NULL, asset_id TEXT NOT NULL,
        pick_score REAL NOT NULL DEFAULT 0, is_best INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(group_id, asset_id))""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_asm_asset ON asset_similar_member_v0(asset_id)")
    # 视觉判定留痕表 —— 与过滤事实表解耦，判定为「正常」的也记一行，避免重复送模型
    con.execute("""CREATE TABLE IF NOT EXISTS asset_vision_check_v0 (
        asset_id TEXT PRIMARY KEY, kind TEXT, junk INTEGER,
        confidence REAL, raw TEXT, provider TEXT, model TEXT, checked_at TEXT)""")
    # 已考察标记（2026-09-16）：让每一步真正做到「只处理一次」。
    # 没有它，autoscan 每轮都会把「没重复所以永远不入组」的照片重新算一遍，
    # 全量 dHash 每次 4 分钟 —— 那样就不是「每天自动」，是「每天白烧 CPU」。
    con.execute("""CREATE TABLE IF NOT EXISTS asset_enrich_v0 (
        asset_id TEXT PRIMARY KEY,
        similar_at TEXT, hash_at TEXT, junk_at TEXT, vision_at TEXT)""")
    # 老表补列（CREATE TABLE IF NOT EXISTS 不会给已存在的表加列）
    cols = {r[1] for r in con.execute("PRAGMA table_info(asset_enrich_v0)")}
    if "blur_at" not in cols:
        con.execute("ALTER TABLE asset_enrich_v0 ADD COLUMN blur_at TEXT")
    # 缺图留痕（2026-09-16）：缩略图/原图读不到的照片，各步骤都无法真正处理，
    # 于是「欠账」永远清不掉 —— autoscan 每轮看到 >0 就白拉起一次流水线。
    # 试过但读不到图的在这里留一条，欠账体检扣掉它；数据盘挂回来或路径修好后
    # 跑 `--retry-miss` 清空标记即可重试。
    con.execute("""CREATE TABLE IF NOT EXISTS asset_imgmiss_v0 (
        asset_id TEXT, stage TEXT, tried_at TEXT, PRIMARY KEY(asset_id, stage))""")
    # dHash 缓存：新照片要和「同源 ±48h 内所有照片」比对做近重复判定，
    # 老照片的 dHash 必须能直接查到，否则每次都要重算全库缩略图。
    con.execute("""CREATE TABLE IF NOT EXISTS asset_dhash_v0 (
        asset_id TEXT PRIMARY KEY, dhash INTEGER NOT NULL, computed_at TEXT)""")
    con.commit()


def _mark(con, asset_id, field):
    """标记某资产的某一步已处理过。"""
    assert field in ("similar_at", "hash_at", "junk_at", "vision_at", "blur_at")
    con.execute(
        f"""INSERT INTO asset_enrich_v0 (asset_id, {field}) VALUES (?,?)
            ON CONFLICT(asset_id) DO UPDATE SET {field}=excluded.{field}""",
        (asset_id, now_iso()))


def _miss(con, asset_id, stage):
    """留痕「这一步试过，但图读不到」—— 防止欠账永不清零。"""
    if not asset_id:
        return
    con.execute(
        """INSERT INTO asset_imgmiss_v0 (asset_id, stage, tried_at) VALUES (?,?,?)
           ON CONFLICT(asset_id, stage) DO UPDATE SET tried_at=excluded.tried_at""",
        (asset_id, stage, now_iso()))



# ══════════════════════════════════════════════════ 步骤 1：sha256

def step_hash(con, limit=0, dry=False, include_videos=False):
    """精确去重的基础指纹。

    两个关键取舍：

    ① **大小预筛**：大小唯一的文件不可能有精确重复，所以只对「byte_size 出现
       碰撞」的桶计算 sha256。幂等要点是每次重跑都重新分桶 —— 过去大小唯一的
       文件，一旦来了同大小的新文件，这一轮就会连同它一起补算，不会漏。

    ② **默认只算照片**（2026-09-16 实测修正）：全库视频 722GB，其中大小碰撞的
       345 个文件就占 **154GB**；而相似去重模块只处理照片，算视频哈希没有任何
       消费者 —— 等于每小时白搬 154GB。实测照片侧只需哈希 48 个文件 / 53MB。
       要单独给视频建指纹时显式加 `--hash-videos`。
    """
    type_filter = "" if include_videos else "AND ma.media_type='photo'"
    rows = con.execute(
        f"""SELECT mf.file_id, mf.asset_id, mf.absolute_path, mf.byte_size, mf.sha256
            FROM media_file mf JOIN media_asset ma USING(asset_id)
            WHERE mf.byte_size IS NOT NULL AND mf.byte_size > 0 {type_filter}""").fetchall()
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["byte_size"]].append(r)

    todo = []
    for size, rs in buckets.items():
        if len(rs) < 2:
            continue
        todo.extend([r for r in rs if not r["sha256"]])
    seen = set()
    todo = [r for r in todo if not (r["file_id"] in seen or seen.add(r["file_id"]))]

    gb = sum(r["byte_size"] or 0 for r in todo) / 1e9
    log(f"[hash] 范围={'含视频' if include_videos else '仅照片'} 文件 {len(rows)} / "
        f"大小桶 {len(buckets)} / 需补算 {len(todo)} 个（{gb:.3f} GB）")
    if limit:
        todo = todo[:limit]
    stats = {"computed": 0, "mb": 0.0, "failed": 0}
    if not todo:
        return stats

    t0 = time.time()
    for i, r in enumerate(todo, 1):
        try:
            h = hashlib.sha256()
            with open(r["absolute_path"], "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()
        except Exception as exc:
            stats["failed"] += 1
            if stats["failed"] <= 5:
                log(f"  [hash] 读失败 {Path(r['absolute_path']).name}: {type(exc).__name__}")
            if not dry:
                _miss(con, r["asset_id"], "hash")   # 留痕，否则这 48 个文件每轮都欠账
            continue
        stats["computed"] += 1
        stats["mb"] += (r["byte_size"] or 0) / 1e6
        if not dry:
            con.execute("UPDATE media_file SET sha256=? WHERE file_id=?", (digest, r["file_id"]))
        if i % 200 == 0:
            if not dry:
                con.commit()
            log(f"  [hash] {i}/{len(todo)} {i/(time.time()-t0):.0f}/s "
                f"{stats['mb']:.0f}MB")
    if not dry:
        con.commit()
    log(f"[hash] 补算 {stats['computed']} 个（{stats['mb']:.1f}MB）失败 {stats['failed']} "
        f"耗时 {time.time()-t0:.0f}s")

    if not dry:
        d = con.execute(
            """SELECT count(*) g, COALESCE(sum(n-1),0) e FROM (
                 SELECT count(DISTINCT asset_id) n FROM media_file
                 WHERE sha256 IS NOT NULL AND sha256<>''
                 GROUP BY sha256 HAVING n>1)""").fetchone()
        stats["dup_groups"], stats["dup_extra"] = d["g"], d["e"]
        log(f"[hash] 精确重复组 {d['g']} 个 / 冗余 {d['e']} 张")
    return stats


def build_hash_index(con):
    """sha256 → 同哈希资产集合；仅保留有重复的哈希。返回 (hash_of, asset_hashes)。"""
    hash_of = defaultdict(set)
    for r in con.execute(
            """SELECT mf.sha256 h, ma.asset_id a FROM media_file mf
               JOIN media_asset ma USING(asset_id)
               WHERE mf.sha256 IS NOT NULL AND mf.sha256<>'' AND ma.media_type='photo'"""):
        hash_of[r["h"]].add(r["a"])
    hash_of = {h: s for h, s in hash_of.items() if len(s) > 1}
    asset_hashes = defaultdict(set)
    for h, s in hash_of.items():
        for a in s:
            asset_hashes[a].add(h)
    return hash_of, asset_hashes


# ══════════════════════════════════════════════════ 步骤 2：模糊

def classify_blur(img_bgr, grid=4):
    """与 blur_detector.py 同口径：分块 Laplacian 方差取 top-25% 均值。
    整图方差会被大面积虚化背景拖低造成假阳，所以取「画面里最锐主体」的锐度。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bs = max(48, min(h, w) // grid)
    scores = []
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            scores.append(cv2.Laplacian(gray[y:y + bs, x:x + bs], cv2.CV_64F).var())
    if not scores:
        scores = [cv2.Laplacian(gray, cv2.CV_64F).var()]
    scores.sort()
    topk = max(1, len(scores) // 4)
    topq = float(np.mean(scores[-topk:]))
    label = "blurry" if topq < T_BLURRY else ("soft" if topq < T_SOFT else "sharp")
    return round(topq, 1), label


def combined_quality(aesthetic, blur_label):
    """综合展示分：美学为主 + 模糊惩罚。与 backfill_quality.py 同一公式。"""
    if aesthetic is None:
        return None
    pen = {"sharp": 1.0, "soft": 0.85, "blurry": 0.5}.get(blur_label, 1.0)
    return round(aesthetic * pen, 3)


def step_blur(con, limit=0, dry=False):
    """给缺 blur_label 的照片补模糊分。只读本地 t480 缩略图（<50ms/张）。"""
    ids = [r[0] for r in con.execute(
        """SELECT ma.asset_id FROM media_asset ma
           LEFT JOIN asset_quality_v0 q ON q.asset_id = ma.asset_id
           LEFT JOIN asset_enrich_v0 e ON e.asset_id = ma.asset_id
           WHERE ma.media_type='photo'
             AND e.blur_at IS NULL
             AND (q.asset_id IS NULL OR q.blur_label IS NULL OR q.blur_label='')
           ORDER BY ma.asset_id""")]
    log(f"[blur] 缺模糊分照片 {len(ids)} 张")
    if limit:
        ids = ids[:limit]
    stats = {"done": 0, "miss": 0, "dist": defaultdict(int)}
    if not ids:
        return stats
    t0 = time.time()
    for i, aid in enumerate(ids, 1):
        tp = thumb_path(aid)
        img = cv2.imread(str(tp)) if tp else None
        if img is None:
            stats["miss"] += 1
            if not dry:
                # 缺图也要标记「试过」，否则这 3 张永远挂在欠账里
                _mark(con, aid, "blur_at")
                _miss(con, aid, "blur")
            continue
        sharp, label = classify_blur(img)
        stats["done"] += 1
        stats["dist"][label] += 1
        if not dry:
            _mark(con, aid, "blur_at")
            con.execute(
                """INSERT INTO asset_quality_v0 (asset_id, blur_label, sharp_score, model_version)
                   VALUES (?,?,?,?)
                   ON CONFLICT(asset_id) DO UPDATE SET
                     blur_label=excluded.blur_label,
                     sharp_score=excluded.sharp_score""", (aid, label, sharp, MODEL_VERSION))
            row = con.execute("SELECT aesthetic FROM asset_quality_v0 WHERE asset_id=?",
                              (aid,)).fetchone()
            cq = combined_quality(row["aesthetic"] if row else None, label)
            if cq is not None:
                con.execute("UPDATE media_asset SET quality_score=? WHERE asset_id=?", (cq, aid))
        if i % 200 == 0:
            if not dry:
                con.commit()
            log(f"  [blur] {i}/{len(ids)} {i/(time.time()-t0):.0f}/s")
    if not dry:
        con.commit()
    log(f"[blur] 完成 {stats['done']} 张（缺图 {stats['miss']}）{dict(stats['dist'])} "
        f"耗时 {time.time()-t0:.0f}s")
    return stats


# ══════════════════════════════════════════════════ 步骤 3：确定性过滤

def evidence_kind_for(reason):
    """过滤原因 → `asset_filter_v0.evidence_kind`（**只能**取这几个值）。

    真库那张表的 evidence_kind 带 CHECK 约束：
        CHECK(evidence_kind IN ('path','filename','visual','ocr','user'))
    2026-09-16 实测踩坑：step_junk 原来写死 `"rule"`，违反约束，而语句是
    `INSERT OR IGNORE` → 违规被**静默吞掉**，日志照报「命中 N 条」。
    结果自动过滤链路一直是断的（全量那轮 1350 条命中，一条都没落库）。
    现在按后缀给合法值，并且写入侧不再用 OR IGNORE 掩盖错误。
    """
    for suffix, kind in (("_FILENAME", "filename"), ("_PATH", "path")):
        if reason.endswith(suffix):
            return kind
    return "path"


def classify_junk_rules(filename, relpath):
    """确定性垃圾判定：只看文件名/相对路径，零解码，低误杀。

    与 server.py 的 `_classify_import_junk` 同源，但两处收紧：
    - **聊天导出规则保持下线**（2026-09-16 决策：1208 张 mmexport 全是唯一副本，
      16% 有人脸，按路径藏掉代价不对称）。
    - **下载目录规则收敛**：只在路径**末段目录名**属于下载类时命中，
      不再对任意层级做子串匹配（避免 "老下载归档" 这类目录误伤）。
    """
    fn = (filename or "").lower()
    rp = (relpath or "").lower()
    reasons = []

    if fn.startswith("screenshot") or "截图" in fn or "屏幕截图" in fn \
       or "screencapture" in fn or fn.startswith("screen_"):
        reasons.append("SCREENSHOT_FILENAME")
    if "/screenshots/" in rp or rp.startswith("screenshots/") \
       or "截图" in rp or "截屏" in rp or "screenshot" in rp:
        reasons.append("SCREENSHOT_PATH")
    segs = [s for s in rp.split("/")[:-1] if s]
    if segs and segs[-1] in ("download", "downloads", "下载", "baidunetdisk"):
        reasons.append("DOWNLOAD_PATH")
    if "screenrecording" in fn or "录屏" in fn or "screen record" in fn:
        reasons.append("SCREEN_RECORDING_FILENAME")
    return list(dict.fromkeys(reasons))


def step_junk(con, limit=0, dry=False):
    """确定性过滤：只处理「从没判定过」的资产，且白名单免检。

    用 `asset_enrich_v0.junk_at` 标记已判定，所以判定为「正常」的不会每次重判。
    用户在白名单里的资产跳过 —— 他手动恢复过，不再接受规则判定。
    """
    ids = [r[0] for r in con.execute(
        """SELECT ma.asset_id FROM media_asset ma
           WHERE ma.asset_id NOT IN (SELECT asset_id FROM asset_allowlist_v0)
             AND ma.asset_id NOT IN (SELECT asset_id FROM asset_enrich_v0 WHERE junk_at IS NOT NULL)
           ORDER BY ma.asset_id""")]
    log(f"[junk] 待判定资产 {len(ids)} 个")
    if limit:
        ids = ids[:limit]
    stats = {"judged": 0, "hit": 0, "written": 0, "duplicate": 0, "errors": 0,
             "dist": defaultdict(int)}
    if not ids:
        return stats
    for aid in ids:
        row = con.execute(
            """SELECT filename, relative_path FROM media_file
               WHERE asset_id=? ORDER BY byte_size DESC LIMIT 1""", (aid,)).fetchone()
        if not row:
            continue
        stats["judged"] += 1
        for rsn in classify_junk_rules(row["filename"], row["relative_path"]):
            stats["hit"] += 1
            stats["dist"][rsn] += 1
            if not dry:
                try:
                    con.execute(
                        """INSERT INTO asset_filter_v0
                           (asset_id, filter_reason, evidence_kind, evidence_value,
                            confidence, rule_version, created_at) VALUES (?,?,?,?,?,?,?)""",
                        (aid, rsn, evidence_kind_for(rsn),
                         f"{row['filename']}|{row['relative_path']}",
                         1.0, "auto-junk-v1", now_iso()))
                    stats["written"] += 1
                except sqlite3.IntegrityError as exc:
                    # 只容忍「同一条已经存在」，其他约束错误必须炸出来
                    if "UNIQUE" in str(exc) or "PRIMARY KEY" in str(exc):
                        stats["duplicate"] += 1
                    else:
                        stats["errors"] += 1
                        log(f"  [junk] 写过滤失败 {aid} {rsn}: {exc}")
        if not dry:
            _mark(con, aid, "junk_at")
    if not dry:
        con.commit()
    log(f"[junk] 判定 {stats['judged']} / 命中 {stats['hit']} 条 / 落库 {stats['written']} "
        f"（已存在 {stats['duplicate']} / 失败 {stats['errors']}）{dict(stats['dist'])}")
    return stats


# ══════════════════════════════════════════════════ 步骤 4：相似分组

MASK64 = (1 << 64) - 1


def dhash64(img_bgr):
    """dHash 的**无符号** 64 位表示（0 ~ 2^64-1）。

    注意：这个值直接写 SQLite 的 INTEGER 会溢出 —— SQLite 的 INTEGER 是
    64 位**有符号**，>2^63-1 的值 sqlite3 驱动会抛
    `OverflowError: Python int too large to convert to SQLite INTEGER`。
    落库请一律走 `db_int()`，算汉明距离一律走 `ham64()`。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    return int.from_bytes(np.packbits(diff.flatten()).tobytes(), "big")


def db_int(h):
    """无符号 64 位 → SQLite 可存的有符号 INT64（负数走二进制补码语义）。

    先 `& MASK64` 截断，保证任何输入都不会喂给 sqlite3 一个溢出的整数。
    """
    h &= MASK64
    return h - (1 << 64) if h > MASK64 >> 1 else h


def ham64(a, b):
    """两个（可能带符号存储的）dHash 之间的汉明距离。

    读出后先 `& MASK64` 还原成无符号再看异或 —— Python 的 `^` 对负数是
    无限位补码语义，直接 `bin(-1 ^ 0)` 会得到 `-0b1`，`.count("1")` 数错。
    """
    return bin((a & MASK64) ^ (b & MASK64)).count("1")


def star_clusters(ids, pairs, max_size=CLUSTER_MAX):
    """星形聚类：每个成员必须与簇心（度数最高的未分配点）**直接相似**。
    替代并查集是为了切断传递闭包 —— 并查集下 A~B、B~C 会把 C 串进 A 的簇，
    实测滚出过 121 张横跨 10 年 4 来源的灾难组。"""
    adj = defaultdict(list)
    for a, b in pairs:
        if a != b:
            adj[a].append(b)
            adj[b].append(a)
    assigned, clusters = set(), []
    for seed in sorted(adj, key=lambda k: -len(adj[k])):
        if seed in assigned:
            continue
        members = [seed] + [j for j in adj[seed] if j not in assigned]
        members = members[:max_size]
        assigned.update(members)
        if len(members) > 1:
            clusters.append(members)
    return clusters


def _photo_ctx(con):
    """照片 → {capture_time, source_id}。相似判定需要同源 + 时间窗约束。"""
    out = {}
    for r in con.execute(
            """SELECT ma.asset_id a, ma.capture_time t,
                      (SELECT mf.source_id FROM media_file mf
                       WHERE mf.asset_id=ma.asset_id LIMIT 1) s
               FROM media_asset ma WHERE ma.media_type='photo'"""):
        out[r["a"]] = {"t": as_ts(r["t"]), "s": r["s"]}
    return out


def _siglip_vectors(con):
    out = {}
    for r in con.execute(
            """SELECT subject_id, vector FROM embedding
               WHERE model_name='google/siglip2-base-patch16-224'"""):
        try:
            v = np.frombuffer(r["vector"], dtype="<f4")
            n = np.linalg.norm(v)
            if n > 1e-6:
                out[r["subject_id"]] = v / n
        except Exception:
            continue
    return out


def _dhash_cached(con, want_ids=None, limit=0):
    """返回 {asset_id: dhash}，缺的直接算并写 `asset_dhash_v0` 缓存。

    dHash 只依赖缩略图，算一次就能永久复用。缓存是「每天自动跑」的前提：
    新照片要和**同源 ±48h 内所有照片**比对近重复，若老照片的 dHash 每次都
    现算，每轮都要重读全库缩略图（12k 张 ≈ 4 分钟），根本没法高频跑。
    """
    # 读出后统一还原为无符号，兼容早期以有符号补码写入的行
    cached = {r[0]: r[1] & MASK64 for r in con.execute(
        "SELECT asset_id, dhash FROM asset_dhash_v0")}
    if want_ids is None:
        want = [r[0] for r in con.execute(
            """SELECT ma.asset_id FROM media_asset ma
               WHERE ma.media_type='photo' AND ma.asset_id NOT IN
                 (SELECT asset_id FROM asset_dhash_v0)""")]
    else:
        want = [a for a in want_ids if a not in cached]
    if limit:
        want = want[:limit]
    todo_n = len(want)
    if todo_n:
        log(f"  [dhash] 需新算 {todo_n} 张")
    t0 = time.time()
    miss = 0
    for i, aid in enumerate(want, 1):
        tp = thumb_path(aid)
        img = cv2.imread(str(tp)) if tp else None
        if img is None:
            miss += 1
            _miss(con, aid, "similar")   # 没图的照片永远进不了相似判定，留痕免欠账
            continue
        h = dhash64(img)
        cached[aid] = h
        con.execute(
            """INSERT OR REPLACE INTO asset_dhash_v0 (asset_id, dhash, computed_at)
               VALUES (?,?,?)""", (aid, db_int(h), now_iso()))
        if i % 500 == 0:
            con.commit()
            log(f"  [dhash] {i}/{todo_n} {i/(time.time()-t0):.0f}/s")
    if todo_n:
        con.commit()
        log(f"  [dhash] 新增 {todo_n-miss} 张（缺图 {miss}）耗时 {time.time()-t0:.0f}s")
    return cached


CAND_MAX = 120   # 单张照片最多比对多少个时间邻域候选（按 |Δt| 取最近的）


def step_similar(con, limit=0, dry=False):
    """增量相似分组：只处理「已考察标记为空」的照片。

    为什么需要 `asset_enrich_v0.similar_at`：大多数照片本来就没有重复，永远不会
    入组。若用「不在相似组里」当待办条件，autoscan 每轮都会把 8000+ 张重算一遍
    —— 那不是自动化，是每天白烧 CPU。标记让每张照片**只被考察一次**，
    之后新照片进来时，它会作为「老邻居」被候选查询捞到（dHash 走缓存，不重算）。

    Phase A 追加：与已有组的**代表张**强相似 → 追加为该组成员（不动既有成员）
    Phase B 新建：把新照片与它强相似的**老照片**一起聚类成新组

    三档证据（精确 > 近重复 > 语义）：
      sha256 相同            → 一定是同一张（不同尺寸/格式/目录的副本），跨时间跨来源
      dHash ≤6 + 同源 ±48h   → 连拍微差 / 重复导入
      SigLIP ≥0.96 + 同源 ±90s → 同场景连拍变体
    """
    _ensure_tables(con)
    t0 = time.time()

    # ① dHash 缓存补齐（首次全库 ~4min，之后只补新照片）
    dhashes = _dhash_cached(con)

    # ② 待考察集合
    ctx = _photo_ctx(con)
    grouped = {r[0] for r in con.execute(
        "SELECT DISTINCT asset_id FROM asset_similar_member_v0")}
    done = {r[0] for r in con.execute(
        "SELECT asset_id FROM asset_enrich_v0 WHERE similar_at IS NOT NULL")}
    hidden = ({r[0] for r in con.execute("SELECT DISTINCT asset_id FROM asset_filter_v0")}
              - {r[0] for r in con.execute("SELECT asset_id FROM asset_allowlist_v0")})
    todo = sorted(a for a in ctx
                  if a not in grouped and a not in done and a not in hidden)
    n_groups0 = con.execute("SELECT count(*) FROM asset_similar_group_v0").fetchone()[0]
    log(f"[similar] 待考察 {len(todo)} 张（已入组 {len(grouped)} / 已考察 {len(done)} / "
        f"已有 {n_groups0} 组保持不动）")
    if limit:
        todo = todo[:limit]
    stats = {"appended": 0, "new_groups": 0, "new_members": 0, "pairs": 0}
    if not todo:
        return stats

    hash_of, asset_hashes = build_hash_index(con)
    log(f"  [similar] sha256 精确重复桶 {len(hash_of)} 个")

    # ③ 同源按时间排序的时间索引（候选查询用）
    by_src = defaultdict(list)
    for aid, c in ctx.items():
        if c["t"] is not None and aid in dhashes:
            by_src[c["s"]].append((c["t"], aid))
    for src in by_src:
        by_src[src].sort()
    src_times = {s: [x[0] for x in arr] for s, arr in by_src.items()}

    vecs = None
    need_vec = False

    def sha_eq(a, b):
        return bool(asset_hashes.get(a) and (asset_hashes[a] & asset_hashes.get(b, set())))

    def near_dup(a, b, strict):
        if a not in dhashes or b not in dhashes:
            return False
        ca, cb = ctx.get(a), ctx.get(b)
        if not ca or not cb or ca["s"] != cb["s"] or ca["t"] is None or cb["t"] is None:
            return False
        if abs(ca["t"] - cb["t"]) > (APPEND_WIN_SEC if strict else PHASH_WIN_SEC):
            return False
        lim = APPEND_DHASH_MAX if strict else DHASH_MAX
        return ham64(dhashes[a], dhashes[b]) <= lim

    def sem_sim(a, b):
        nonlocal vecs, need_vec
        if a == b:
            return False
        ca, cb = ctx.get(a), ctx.get(b)
        if not ca or not cb or ca["s"] != cb["s"] or ca["t"] is None or cb["t"] is None:
            return False
        if abs(ca["t"] - cb["t"]) > SIGLIP_WIN_SEC:
            return False
        if vecs is None:
            vecs = _siglip_vectors(con)
        if a not in vecs or b not in vecs:
            return False
        return float(np.dot(vecs[a], vecs[b])) >= SIGLIP_COS_MIN

    def strong(a, b, strict):
        return sha_eq(a, b) or near_dup(a, b, strict) or sem_sim(a, b)

    def candidates(aid, win):
        """同源 ±win 秒内、按 |Δt| 最近的至多 CAND_MAX 个邻居。"""
        c = ctx.get(aid)
        if not c or c["t"] is None:
            return []
        arr_t = src_times.get(c["s"])
        if not arr_t:
            return []
        lo = bisect.bisect_left(arr_t, c["t"] - win)
        hi = bisect.bisect_right(arr_t, c["t"] + win)
        block = by_src[c["s"]][lo:hi]
        if len(block) > CAND_MAX:
            mid = (lo + hi) // 2
            half = CAND_MAX // 2
            block = by_src[c["s"]][max(lo, mid - half):min(hi, mid + half)]
        return [x[1] for x in block]

    # ④ Phase A：追加进已有组
    existing = con.execute(
        """SELECT group_id, best_asset_id FROM asset_similar_group_v0
           WHERE COALESCE(engine,'dual-v1')='dual-v1'""").fetchall()
    assigned = set()
    for g in existing:
        best = g["best_asset_id"]
        if best in hidden or best not in ctx:
            continue
        for aid in todo:
            if aid in assigned or aid == best:
                continue
            if strong(aid, best, strict=True):
                assigned.add(aid)
                stats["appended"] += 1
                if dry:
                    continue
                con.execute(
                    """INSERT OR IGNORE INTO asset_similar_member_v0
                       (group_id, asset_id, pick_score, is_best) VALUES (?,?,0,0)""",
                    (g["group_id"], aid))
                con.execute(
                    """UPDATE asset_similar_group_v0 SET asset_count=
                       (SELECT count(*) FROM asset_similar_member_v0 WHERE group_id=?)
                       WHERE group_id=?""", (g["group_id"], g["group_id"]))
        if stats["appended"] and stats["appended"] % 50 == 0 and not dry:
            con.commit()
    if not dry:
        con.commit()
    log(f"  [similar] Phase A 追加进已有组 {stats['appended']} 张")

    # ⑤ Phase B：剩余新照片 ↔ 同源时间邻域（含已考察过的老照片）
    rest = [a for a in todo if a not in assigned]
    pairs = []
    if rest:
        rest_set = set(rest)
        # (5.1) sha256 精确：跨来源跨时间
        for h, s in hash_of.items():
            hit = sorted(s & rest_set)
            for i in range(len(hit)):
                for j in range(i + 1, len(hit)):
                    pairs.append((hit[i], hit[j]))
        # (5.2) 时间邻域内的近重复 + 语义
        for aid in rest:
            for b in candidates(aid, PHASH_WIN_SEC):
                if b == aid or b in assigned:
                    continue
                if b not in rest_set and b in grouped:
                    continue          # 已入别的组，不在本步动
                if (aid, b) in pairs or (b, aid) in pairs:
                    continue
                if near_dup(aid, b, False) or sem_sim(aid, b):
                    pairs.append((aid, b))
    stats["pairs"] = len(pairs)
    log(f"  [similar] 候选相似对 {stats['pairs']} 对")

    # ⑥ 聚类：节点 = 出现在相似对里的全部照片（含被新照片拽进来的老照片）
    nodes = set()
    for a, b in pairs:
        nodes.add(a)
        nodes.add(b)
    nodes -= assigned
    if len(nodes) >= 2:
        clusters = star_clusters(sorted(nodes), pairs)
        seq0 = con.execute(
            "SELECT MAX(CAST(substr(group_id,5) AS INTEGER)) FROM asset_similar_group_v0").fetchone()[0]
        seq = (seq0 if seq0 is not None else -1) + 1
        for members in clusters:
            gid = f"sim_{seq:07d}"
            seq += 1
            stats["new_groups"] += 1
            stats["new_members"] += len(members)
            if dry:
                log(f"  [similar] (dry) 新组 {gid}: {len(members)} 张")
                continue
            keep = pick_best(con, members)
            con.execute(
                """INSERT INTO asset_similar_group_v0
                   (group_id, best_asset_id, asset_count, created_at, updated_at, engine)
                   VALUES (?,?,?,?,?,'dual-v1')""",
                (gid, keep or members[0], len(members), now_iso(), now_iso()))
            for m in members:
                con.execute(
                    """INSERT OR IGNORE INTO asset_similar_member_v0
                       (group_id, asset_id, pick_score, is_best) VALUES (?,?,0,?)""",
                    (gid, m, 1 if m == (keep or members[0]) else 0))
        log(f"  [similar] Phase B 新建组 {stats['new_groups']} 个 / 成员 {stats['new_members']} 张")

    # ⑦ 打标记：本步考察过的都记上，下次不再重算
    if not dry:
        for aid in todo:
            _mark(con, aid, "similar_at")
        con.commit()
    log(f"[similar] 完成 追加 {stats['appended']} 新组 {stats['new_groups']} "
        f"标记 {len(todo)} 张 耗时 {time.time()-t0:.0f}s")
    return stats


# ══════════════════════════════════════════════════ 步骤 5：择优

def score_asset(con, aid):
    """单张的择优分。

    ⚠️ 2026-09-16 实测修正：`asset_quality_v0.aesthetic` 的**真实范围是
    -41.9 ~ +44.1**（均值 5.4，34% 为负），并不是 aesthetic_scorer.py 文档
    里写的 0-10 —— 那是 LAION 线性头的**原始输出**，没做 AVA 标度变换。
    排序仍单调有效（越大越好看），但**绝不能直接相加**：原始值 ±44 会瞬间
    压死模糊信号（±6），择优退化成「只看美学」。所以先 tanh 归一化到 ±1，
    再乘一个有界权重，让它只当「同组内的偏好微调」。

    另：不重复计入 media_asset.quality_score —— 它就是 aesthetic 乘模糊惩罚
    得出的，和 aesthetic 是同一个信号，加两次等于偷偷给美学加倍。

    权重取舍（本项目一贯的不对称原则：误选一张丑的代价 < 误藏一张唯一的）：
      模糊  sharp +2 / soft 0 / blurry -6   糊的绝不能当代表张，硬否决
      美学  tanh(z/2) × 2.0 → ±2.0          有界，只做微调
      分辨率 0.4·log(面积)                  同组连拍差异小，主要防小图占位
      人脸  +1.5                            家庭相册里「有人」＞ 纯风景
    """
    r = con.execute(
        """SELECT ma.width w, ma.height h,
                  q.aesthetic a, q.blur_label bl,
                  (SELECT count(*) FROM face_instance_v0 f WHERE f.asset_id=ma.asset_id) nf
           FROM media_asset ma LEFT JOIN asset_quality_v0 q ON q.asset_id=ma.asset_id
           WHERE ma.asset_id=?""", (aid,)).fetchone()
    if not r:
        return 0.0
    area = max((r["w"] or 0) * (r["h"] or 0), 1)
    s = {"sharp": 2.0, "soft": 0.0, "blurry": -6.0}.get(r["bl"] or "sharp", 0.0)
    if r["a"] is not None:
        m, sd = _aes_stats(con)
        if sd > 1e-6:
            s += 2.0 * math.tanh((float(r["a"]) - m) / (2.0 * sd))
    s += math.log(area + 1) * 0.4
    if r["nf"]:
        s += 1.5
    return round(s, 4)


_AES_CACHE = {}


def _aes_stats(con):
    """美学分的均值/标准差（从库里现算，避免硬编码；带进程内缓存）。"""
    if "v" in _AES_CACHE:
        return _AES_CACHE["v"]
    r = con.execute(
        """SELECT avg(aesthetic) m, avg(aesthetic*aesthetic) m2, count(*) n
           FROM asset_quality_v0 WHERE aesthetic IS NOT NULL""").fetchone()
    if not r or not r["n"] or r["m"] is None:
        v = (0.0, 0.0)
    else:
        var = max(float(r["m2"]) - float(r["m"]) ** 2, 0.0)
        v = (float(r["m"]), math.sqrt(var))
    _AES_CACHE["v"] = v
    return v



def pick_best(con, members):
    scored = [(score_asset(con, a), a) for a in members]
    scored = [x for x in scored if x[1]]
    if not scored:
        return None
    return max(scored)[1]


def step_pick(con, limit=0, repick_all=False, dry=False):
    """重算代表张。

    默认**只处理还没算过择优分的组**，不动用户手动选过代表的既有组 ——
    界面上的「设为代表」是用户意图，不该被算法覆盖。
    `--repick-all` 才全量重算，需显式指定。
    """
    if repick_all:
        groups = con.execute(
            """SELECT group_id, best_asset_id FROM asset_similar_group_v0
               WHERE COALESCE(engine,'dual-v1')='dual-v1'""").fetchall()
    else:
        groups = con.execute(
            """SELECT g.group_id, g.best_asset_id FROM asset_similar_group_v0 g
               WHERE COALESCE(g.engine,'dual-v1')='dual-v1'
                 AND NOT EXISTS (SELECT 1 FROM asset_similar_member_v0 m
                                 WHERE m.group_id=g.group_id AND m.pick_score > 0)
                 AND (SELECT count(*) FROM asset_similar_member_v0 m2
                      WHERE m2.group_id=g.group_id) > 1""").fetchall()
    log(f"[pick] 待重算代表张的组 {len(groups)} 个（repick_all={repick_all}）")
    if limit:
        groups = groups[:limit]
    stats = {"groups": 0, "changed": 0}

    # 可回退：把改动前的代表张逐行留档（一次运行只写一份，不覆盖）
    bak = DATA_DIR / f"enrich_pick_backup_{datetime.now().strftime('%Y%m%d')}.jsonl"
    if not dry and groups and not bak.exists():
        try:
            with open(bak, "w", encoding="utf-8") as f:
                for g in groups:
                    f.write(json.dumps({"group_id": g["group_id"],
                                        "old_best": g["best_asset_id"]}) + "\n")
            log(f"[pick] 改动前代表张已留档 → {bak}")
        except Exception as exc:
            log(f"[pick] 留档失败（不阻断）：{exc}")

    for g in groups:
        members = [r[0] for r in con.execute(
            "SELECT asset_id FROM asset_similar_member_v0 WHERE group_id=?", (g["group_id"],))]
        if len(members) < 2:
            continue
        scores = {a: score_asset(con, a) for a in members}
        best = max(scores, key=lambda a: scores[a])
        stats["groups"] += 1
        if best != g["best_asset_id"]:
            stats["changed"] += 1
        if dry:
            continue
        for aid, s in scores.items():
            con.execute(
                """UPDATE asset_similar_member_v0 SET pick_score=?
                   WHERE group_id=? AND asset_id=?""", (s, g["group_id"], aid))
        con.execute("UPDATE asset_similar_member_v0 SET is_best=0 WHERE group_id=?", (g["group_id"],))
        con.execute("""UPDATE asset_similar_member_v0 SET is_best=1
                       WHERE group_id=? AND asset_id=?""", (g["group_id"], best))
        con.execute("""UPDATE asset_similar_group_v0 SET best_asset_id=?, updated_at=?
                       WHERE group_id=?""", (best, now_iso(), g["group_id"]))
    if not dry:
        con.commit()
    log(f"[pick] 完成 {stats['groups']} 组，代表张变化 {stats['changed']} 组")
    return stats


# ══════════════════════════════════════════════════ 步骤 6：VLM 语义过滤

VISION_PROMPT = (
    "你是家庭相册的垃圾图片判别器。判断这张图**是不是生活照片**。\n"
    "属于垃圾（junk=true）：手机/电脑屏幕截图、聊天记录截图、文档/表格/PPT 翻拍、"
    "收款码/二维码、白底商品图、纯文字笔记。\n"
    "**翻拍屏幕（screen_photo）从严判定**：只有屏幕/文档内容占画面主体（约七成以上）、"
    "几乎看不到真实世界场景时才算；画面里出现真人、动物、风景、货架、建筑物等真实场景的，"
    "即使恰好拍到屏幕也不算（2026-09-17 实测教训：动物园老虎照、烟花投影合影曾被误判为拍屏）。\n"
    "不属于垃圾（junk=false）：真人照片、家人合影、风景、宠物、食物、"
    "有生活场景的照片；**哪怕拍得很糊、构图很差的照片也不算垃圾**。\n"
    "宁可漏判也不要误判：拿不准就 junk=false。\n"
    '只输出 JSON 不要解释：{"junk": true/false, '
    '"kind": "screenshot|chat|screen_photo|document|qr|ecommerce|none", '
    '"confidence": 0.0-1.0}'
)

VISION_KIND_REASON = {
    "screenshot": "VISION_SCREENSHOT",
    "chat": "VISION_CHAT",
    "screen_photo": "VISION_SCREEN_PHOTO",
    "document": "VISION_DOCUMENT",
    "qr": "VISION_QR",
    "ecommerce": "VISION_ECOMMERCE",
}


def _algo(con, key, default):
    """读「算法偏好」设置（server.py 的算法偏好面板写 algo_* 键）。
    产品化要求：阈值/提示词不写死，界面可调；读不到就用内置默认。"""
    try:
        r = con.execute("SELECT value FROM app_setting_v0 WHERE key=?",
                        ("algo_" + key,)).fetchone()
        return r[0] if r else default
    except Exception:
        return default


def load_vision_provider(con):
    """从 api_providers_v0 取一个可用视觉模型（vision=1、enabled、有 key）。"""
    try:
        rows = con.execute(
            """SELECT name, base_url, api_key, model FROM api_providers_v0
               WHERE COALESCE(vision,1)=1 AND COALESCE(enabled,1)=1
                 AND api_key IS NOT NULL AND api_key<>''
               ORDER BY COALESCE(sort_order,999) LIMIT 1""").fetchall()
    except Exception:
        return None
    return dict(rows[0]) if rows else None


def vision_classify(prov, img_path, retries=4, prompt=None):
    """调视觉模型判定。返回 dict（含 _error 表示失败）。prompt 可由设置覆盖。

    2026-09-16 实测：智谱 glm-4.6v-flash 免费档有频率限制，连续两张就 429
    （Too Many Requests）。这里对 429 做指数退避（5/10/20/40s），批量才跑得动；
    其他错误（401 key 无效 / 400 参数 / 超时）不重试，直接返回 _error。
    """
    b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
    url = prov["base_url"].rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = {
        "model": prov["model"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt or VISION_PROMPT},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        ]}],
        "max_tokens": 120, "temperature": 0,
    }
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer " + prov["api_key"]})
        try:
            resp = json.loads(urllib.request.urlopen(req, timeout=45).read().decode())
            text = resp["choices"][0]["message"]["content"].strip()
            break
        except Exception as exc:
            if getattr(exc, "code", None) == 429 and attempt < retries:
                wait = 5 * (2 ** attempt)          # 5 / 10 / 20 / 40 秒
                log(f"  [vision] 429 限流，退避 {wait}s（第 {attempt+1}/{retries} 次）")
                time.sleep(wait)
                continue
            return {"_error": f"{type(exc).__name__}: {exc}"}
    if "```" in text:                      # 模型可能包 ```json
        text = text.split("```")[1].replace("json", "", 1).strip()
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < 0:
        return {"_error": f"非 JSON 输出: {text[:80]}"}
    try:
        return json.loads(text[i:j + 1])
    except Exception as exc:
        return {"_error": f"JSON 解析失败: {exc}"}


def step_vision(con, limit=0, dry=False):
    """语义过滤：补文件名规则抓不到的截图/文档/拍屏。

    **无 provider 时优雅跳过**（不改任何数据），并把原因回传，
    让 server 能把这个状态显示出来，而不是静默什么都不做。

    判定留痕写 `asset_vision_check_v0`；**只有判定为垃圾才写 asset_filter_v0**。
    """
    _ensure_tables(con)
    prov = load_vision_provider(con)
    if not prov:
        msg = ("未配置视觉模型（api_providers_v0 没有可用项）—— "
               "到「设置 → 模型服务」加一个支持 vision 的服务即可启用本档")
        log(f"[vision] 跳过：{msg}")
        return {"skipped": "no_provider", "hint": msg}
    log(f"[vision] 使用 {prov['name']} / {prov['model']}")
    # 阈值/间隔/提示词全部可由「算法偏好」覆盖（2026-09-17 产品化）
    min_conf = float(_algo(con, "vision_min_conf", VISION_MIN_CONF) or VISION_MIN_CONF)
    _interval = float(_algo(con, "vision_min_interval_sec", VISION_MIN_INTERVAL_SEC) or 0)
    _prompt = _algo(con, "vision_prompt", "") or None
    log(f"[vision] 参数：conf>={min_conf} 间隔={_interval}s 提示词={'自定义' if _prompt else '内置默认'}")

    ids = [r[0] for r in con.execute(
        """SELECT ma.asset_id FROM media_asset ma
           WHERE ma.media_type='photo'
             AND ma.asset_id NOT IN (SELECT asset_id FROM asset_vision_check_v0)
             AND ma.asset_id NOT IN (SELECT asset_id FROM asset_allowlist_v0)
             AND ma.asset_id NOT IN
                 (SELECT asset_id FROM asset_filter_v0
                  WHERE asset_id IS NOT NULL
                    AND asset_id NOT IN (SELECT asset_id FROM asset_allowlist_v0
                                         WHERE asset_id IS NOT NULL))
           ORDER BY ma.asset_id""")]
    log(f"[vision] 待语义判定照片 {len(ids)} 张")
    if limit:
        ids = ids[:limit]
    stats = {"judged": 0, "hit": 0, "error": 0, "dist": defaultdict(int)}
    if not ids:
        return stats
    t0 = time.time()
    for i, aid in enumerate(ids, 1):
        tp = thumb_path(aid)
        if not tp:
            # 2026-09-18 修复：缺缩略图必须留痕，否则这批照片永远在 todo 里，
            # pending 永不清零 → autoscan 每 30 分钟白拉起一次流水线（实测卡 3 张）。
            stats["miss"] = stats.get("miss", 0) + 1
            if not dry:
                _miss(con, aid, "vision")
            continue
        r = vision_classify(prov, tp, prompt=_prompt)
        if "_error" in r:
            stats["error"] += 1
            if stats["error"] <= 3:
                log(f"  [vision] {aid} 失败: {r['_error']}")
            if stats["error"] >= 5 and stats["judged"] == 0:
                log("  [vision] 连续失败，提前退出（检查 key / base_url / 模型是否支持图片输入）")
                break
            continue
        stats["judged"] += 1
        conf = float(r.get("confidence") or 0)
        kind = (r.get("kind") or "none").lower()
        is_junk = bool(r.get("junk")) and kind in VISION_KIND_REASON \
            and conf >= min_conf
        reason = VISION_KIND_REASON.get(kind, "")
        if is_junk:
            stats["hit"] += 1
            stats["dist"][reason] += 1
        if not dry:
            con.execute(
                """INSERT OR REPLACE INTO asset_vision_check_v0
                   (asset_id, kind, junk, confidence, raw, provider, model, checked_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (aid, kind, 1 if is_junk else 0, conf,
                 json.dumps(r, ensure_ascii=False), prov["name"], prov["model"], now_iso()))
            if is_junk:
                # 2026-09-17 护栏：有人脸的照片不得因 document/ecommerce 自动隐藏
                # （08-29 macos-VN 曾把孩子写作业/弹琴照判成"文档"藏了 227 张）。
                # 判定照常留痕，只是不写 asset_filter_v0；截图类不含此护栏。
                _face_n = 0
                if kind in ("document", "ecommerce"):
                    _face_n = con.execute(
                        "SELECT count(*) FROM face_instance_v0 WHERE asset_id=?", (aid,)).fetchone()[0]
                if _face_n:
                    stats["face_guard"] = stats.get("face_guard", 0) + 1
                    if stats["face_guard"] <= 5:
                        log(f"  [vision] 护栏放行 {aid}（{kind} 但有 {_face_n} 张人脸，只记判定不隐藏）")
                else:
                    # 同 step_junk：不用 OR IGNORE，避免约束错误被静默吞掉
                    try:
                        con.execute(
                            """INSERT INTO asset_filter_v0
                               (asset_id, filter_reason, evidence_kind, evidence_value,
                                confidence, rule_version, created_at) VALUES (?,?,?,?,?,?,?)""",
                            (aid, reason, "visual", json.dumps(r, ensure_ascii=False),
                             conf, "vision-filter-v1", now_iso()))
                    except sqlite3.IntegrityError as exc:
                        if not ("UNIQUE" in str(exc) or "PRIMARY KEY" in str(exc)):
                            stats["error"] += 1
                            log(f"  [vision] 写过滤失败 {aid} {reason}: {exc}")
        if i % 20 == 0:
            if not dry:
                con.commit()
            log(f"  [vision] {i}/{len(ids)} 命中 {stats['hit']} 护栏放行 {stats.get('face_guard', 0)} 失败 {stats['error']} "
                f"{i/(time.time()-t0):.1f}/s")
        if not dry and i < len(ids):
            # 免费档限流（实测 glm-4.6v-flash 连发即 429）：批量保底 1s/张。
            # 有 asset_vision_check_v0 断点，中断重跑会自动续，不怕慢。
            time.sleep(_interval)
    if not dry:
        con.commit()
    log(f"[vision] 完成 判定 {stats['judged']} 命中 {stats['hit']} 失败 {stats['error']} "
        f"{dict(stats['dist'])} 耗时 {time.time()-t0:.0f}s")
    return stats


# ══════════════════════════════════════════════════ 状态体检

def pending_work(con):
    """还有多少活要干（**便宜**，给 server 每轮 autoscan 调）。

    只查标记表，不做任何重活。全为 0 就说明没有欠账，server 不必拉起流水线
    —— 这是「每天自动跑」能成立的前提：没新照片时开销接近 0。
    """
    _ensure_tables(con)
    # 2026-09-16 修口径：原写法 `e.similar_at IS NULL OR d.asset_id IS NULL` 会把
    # 两类已「处理完」的照片永远算成欠账 ——
    #   ① 已入组的照片：step_similar 只因「已入组」就跳过它们，从不写 similar_at，
    #      于是 3754 张老组成员每轮都欠账；
    #   ② 缺缩略图的照片：永远算不出 dHash；
    #   ③ 被过滤掉的照片（filter − allowlist）：step_similar 有意跳过它们
    #      （不给垃圾照片做相似分组），于是也永远没有 similar_at。
    # 实测口径修正前 n1 报 983，全部是 ③ —— autoscan 每 30 分钟都判定「还有活」，
    # 白拉起一次流水线。判据必须与 step_similar 的 todo 完全镜像。
    n1 = con.execute(
        """SELECT count(*) FROM media_asset ma
           LEFT JOIN asset_enrich_v0 e USING(asset_id)
           WHERE ma.media_type='photo'
             AND e.similar_at IS NULL
             AND ma.asset_id NOT IN (SELECT asset_id FROM asset_similar_member_v0)
             AND ma.asset_id NOT IN
                 (SELECT asset_id FROM asset_imgmiss_v0 WHERE stage='similar')
             AND ma.asset_id NOT IN
                 (SELECT asset_id FROM asset_filter_v0
                  WHERE asset_id IS NOT NULL
                    AND asset_id NOT IN (SELECT asset_id FROM asset_allowlist_v0
                                         WHERE asset_id IS NOT NULL))""").fetchone()[0]
    n2 = con.execute(
        """SELECT count(*) FROM media_asset ma
           LEFT JOIN asset_enrich_v0 e USING(asset_id)
           WHERE e.junk_at IS NULL
             AND ma.asset_id NOT IN (SELECT asset_id FROM asset_allowlist_v0)""").fetchone()[0]
    # 同理：以 blur_at「试过」为准，否则 3 张缺缩略图的照片永远欠账
    n3 = con.execute(
        """SELECT count(*) FROM media_asset ma
           LEFT JOIN asset_quality_v0 q USING(asset_id)
           LEFT JOIN asset_enrich_v0 e USING(asset_id)
           WHERE ma.media_type='photo' AND e.blur_at IS NULL
             AND (q.asset_id IS NULL OR q.blur_label IS NULL OR q.blur_label='')""").fetchone()[0]
    # 与 step_hash 默认口径一致：**只算照片**。否则视频侧 154GB 的待哈希量
    # 会让「有欠账」永远成立，autoscan 每轮都白拉起流水线。
    # 同理：读不到文件的 48 个已在 imgmiss 留痕，不再每轮重报
    n4 = con.execute(
        """SELECT count(*) FROM media_file mf JOIN media_asset ma USING(asset_id)
           WHERE ma.media_type='photo' AND mf.sha256 IS NULL
             AND mf.asset_id NOT IN
                 (SELECT asset_id FROM asset_imgmiss_v0 WHERE stage='hash')
             AND mf.byte_size IN
             (SELECT mf2.byte_size FROM media_file mf2 JOIN media_asset ma2 USING(asset_id)
              WHERE ma2.media_type='photo' AND mf2.byte_size IS NOT NULL
              GROUP BY mf2.byte_size HAVING count(*)>1)""").fetchone()[0]
    # 语义档：没配视觉模型时**不算欠账** —— 否则 autoscan 每轮都看到
    # 「还有 1 万张没做语义判定」，于是每 30 分钟白拉起一次流水线。
    # 档位没启用 ≠ 有欠账。
    if load_vision_provider(con):
        n5 = con.execute(
            """SELECT count(*) FROM media_asset ma
               WHERE ma.media_type='photo'
                 AND ma.asset_id NOT IN (SELECT asset_id FROM asset_vision_check_v0)
                 AND ma.asset_id NOT IN (SELECT asset_id FROM asset_allowlist_v0)
                 AND ma.asset_id NOT IN
                     (SELECT asset_id FROM asset_imgmiss_v0 WHERE stage='vision')
                 AND ma.asset_id NOT IN
                     (SELECT asset_id FROM asset_filter_v0
                      WHERE asset_id IS NOT NULL
                        AND asset_id NOT IN (SELECT asset_id FROM asset_allowlist_v0
                                             WHERE asset_id IS NOT NULL))""").fetchone()[0]
    else:
        n5 = 0
    return {"similar": n1, "junk": n2, "blur": n3, "sha256": n4, "vision": n5,
            "total": n1 + n2 + n3 + n4 + n5}


def status(con):
    """欠账体检：每项的待处理量。给界面和人工排查用。"""
    _ensure_tables(con)
    q = lambda s: con.execute(s).fetchone()[0]
    prow = load_vision_provider(con)
    st = {
        "assets_total": q("SELECT count(*) FROM media_asset"),
        "photos_total": q("SELECT count(*) FROM media_asset WHERE media_type='photo'"),
        "similar_groups": q("SELECT count(*) FROM asset_similar_group_v0"),
        "filter_reasons": q("SELECT count(*) FROM asset_filter_v0"),
        "filtered_assets": q("SELECT count(DISTINCT asset_id) FROM asset_filter_v0"),
        "allowlist": q("SELECT count(*) FROM asset_allowlist_v0"),
        "files_total": q("SELECT count(*) FROM media_file"),
        "sha256_done": q("SELECT count(*) FROM media_file WHERE sha256 IS NOT NULL AND sha256<>''"),
        "sha256_dup_extra": q("""SELECT COALESCE(sum(n-1),0) FROM (
            SELECT count(DISTINCT asset_id) n FROM media_file
            WHERE sha256 IS NOT NULL AND sha256<>''
            GROUP BY sha256 HAVING n>1)"""),
        "dhash_cached": q("SELECT count(*) FROM asset_dhash_v0"),
        "img_missing": q("SELECT count(*) FROM asset_imgmiss_v0"),
        "vision_provider": (prow or {}).get("name") or "(未配置 - 语义过滤档未启用)",
        "vision_checked": q("SELECT count(*) FROM asset_vision_check_v0"),
        "vision_hit": q("SELECT count(*) FROM asset_vision_check_v0 WHERE junk=1"),
    }
    st["pending"] = pending_work(con)
    return st



# ══════════════════════════════════════════════════ main

def main():
    global DB
    ap = argparse.ArgumentParser(description="增量富化流水线（幂等，只处理欠账）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--status", action="store_true", help="只看欠账，不写库")
    ap.add_argument("--pending", action="store_true",
                    help="只输出待办量 JSON（便宜，给 server 判断要不要拉起）")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--repick-all", action="store_true", help="pick 全量重算（默认只算新组）")
    ap.add_argument("--hash-videos", action="store_true",
                    help="hash 步骤把视频也算上（默认只算照片；全库视频 722GB，慎用）")
    ap.add_argument("--retry-miss", action="store_true",
                    help="清空「缺图留痕」后重试（数据盘曾未挂载/路径修好后用）")
    for s in STEPS:
        ap.add_argument(f"--{s}", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    DB = Path(args.db)

    con = connect()
    _ensure_tables(con)

    if args.pending:
        log(json.dumps(pending_work(con), ensure_ascii=False))
        con.close()
        return

    if args.status:
        log("── 富化欠账体检 ──")
        for k, v in status(con).items():
            log(f"  {k:20} {v}")
        con.close()
        return

    if args.retry_miss:
        n = con.execute("SELECT count(*) FROM asset_imgmiss_v0").fetchone()[0]
        d = con.execute("""SELECT stage, count(*) FROM asset_imgmiss_v0
                           GROUP BY stage""").fetchall()
        con.execute("DELETE FROM asset_imgmiss_v0")
        con.commit()
        log(f"已清空缺图留痕 {n} 条 {[tuple(r) for r in d]} —— 下轮会重新尝试读图")
        con.close()
        return

    chosen = [s for s in STEPS if getattr(args, s)]
    if args.all or not chosen:
        chosen = STEPS
    order = [s for s in STEPS if s in chosen]   # STEPS 本身已是依赖顺序
    log(f"══ 富化流水线启动：{' → '.join(order)}{'（dry-run）' if args.dry_run else ''} ══")
    t0 = time.time()
    result = {}
    for s in order:
        ts = time.time()
        try:
            fn = {"hash": lambda: step_hash(con, args.limit, args.dry_run, args.hash_videos),
                  "blur": lambda: step_blur(con, args.limit, args.dry_run),
                  "junk": lambda: step_junk(con, args.limit, args.dry_run),
                  "similar": lambda: step_similar(con, args.limit, args.dry_run),
                  "pick": lambda: step_pick(con, args.limit, args.repick_all, args.dry_run),
                  "vision": lambda: step_vision(con, args.limit, args.dry_run)}[s]
            r = fn() or {}
            result[s] = {k: (dict(v) if isinstance(v, defaultdict) else v)
                         for k, v in r.items() if not isinstance(v, set)}
        except Exception as exc:
            import traceback
            traceback.print_exc()
            result[s] = {"error": f"{type(exc).__name__}: {exc}"}
            log(f"!! 步骤 {s} 失败：{exc}")
        log(f"── {s} 耗时 {time.time()-ts:.0f}s")
    log(f"══ 全部完成 耗时 {time.time()-t0:.0f}s ══")
    log("RESULT_JSON " + json.dumps(result, ensure_ascii=False, default=str))
    con.close()


if __name__ == "__main__":
    main()
