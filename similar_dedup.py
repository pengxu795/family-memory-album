#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""相似照片去重双引擎（新功能②：感知哈希 + SigLIP2 嵌入相似度）

双引擎分工（2026-09-09 设计）：
- 引擎 1 pHash（64bit DCT 感知哈希，Hamming 距离）：
  抓「几乎同一张图」——重复导入、连拍微差、同图不同尺寸/格式。
  全库两两比对（位运算分块，~1 分钟），不依赖时间。
- 引擎 2 SigLIP2 嵌入 cosine（≥0.95）：
  抓「同场景连拍变体」——构图/内容几乎相同的不同帧。
  只比对**同 source ±90 秒**邻域（候选空间缩小 4 个数量级，
  避免全库 16k×16k 余弦；跨设备的"同刻多拍"属 #5 已有功能不在此做）。

保留推荐：每组重复簇推荐保留 1 张——文件最大者优先（原图质量代理），
模块只标注不删除，清理动作由用户在 UI 确认。

不写库：pHash 缓存写 --cache JSONL，重复簇输出 JSONL。
回写 DB（如 asset_relation/清理建议表）留 A4 联调后。

用法：
  python3 similar_dedup.py --build              # 全库算 pHash 缓存
  python3 similar_dedup.py --find               # 双引擎找重复 → JSONL + 摘要
  python3 similar_dedup.py --build --find       # 一步到位
"""
import argparse, json, sqlite3, time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
THUMBS = ROOT / "data" / "thumbs_mvp"
CACHE = ROOT / "data" / "phash_cache.jsonl"

SIGLIP_MODEL = "google/siglip2-base-patch16-224"
PHASH_HAMMING_MAX = 6     # pHash 引擎阈值：≤6/64 位视为近重复
SIGLIP_COS_MIN = 0.96     # 嵌入引擎阈值（2026-09-09 T5 抽检后由 0.95 上调：
                          # 0.95 时 17 个簇两端完全无关（并查集传递链误伤），
                          # 0.96 降至 2 个，候选照片 32.2%→24.5%；0.95~0.96 段
                          # 直接对虽多为真连拍，但漏合只是多显示一张，误合会
                          # 隐藏唯一照片，代价不对称，取保守侧）
TIME_WINDOW_SEC = 90      # 嵌入引擎时间邻域


def thumb_path(asset_id):
    stem = asset_id[6:]
    for cand in (f"{stem}_t480.jpg", f"{stem}.jpg"):
        p = THUMBS / cand
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- pHash

def dhash64(img_bgr):
    """64bit 差异哈希（dHash 9x8 灰度横向梯度）——比 DCT pHash 快 5 倍，
    对缩放/亮度/轻微压缩鲁棒性同量级。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]  # 8x8 bool
    bits = np.packbits(diff.flatten())
    return int.from_bytes(bits.tobytes(), "big")


def hamming_np(a_u64, b_u64):
    """两组 uint64 的逐对汉明距离（numpy popcount 查表）。

    防御性 asort uint64：Python int > 2^63 直接 np.array 会掉成 float/object，
    后续移位在 numpy 2.x 直接 TypeError。"""
    a = np.asarray(a_u64).astype(np.uint64)
    b = np.asarray(b_u64).astype(np.uint64)
    x = np.bitwise_xor(a, b)
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    out = np.zeros(x.shape, dtype=np.uint8)
    for shift in range(0, 64, 8):
        out += table[(x >> np.uint64(shift)) & np.uint64(0xFF)]
    return out


def _star_clusters(n, pairs, ids):
    """星形聚类（2026-09-16 替换并查集）。

    并查集是传递闭包：A~B、B~C 会把 C 串进 A 的簇，长链越滚越大——
    实测在全库 dHash≤6 下滚出一个 121 张、横跨 10 年 4 个来源的灾难组。
    星形聚类的硬约束：**每个成员必须与簇心（度数最高的未分配点）直接相似**，
    链式关系直接断掉。代价是簇间的真重复可能各留一份——按本项目一贯的
    不对称原则（漏合只是多显示一张，误合会藏掉唯一照片），取保守侧。
    """
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in pairs:
        adj[a].append(b)
        adj[b].append(a)
    assigned = [False] * n
    clusters = []
    for seed in sorted(range(n), key=lambda i: -len(adj[i])):
        if assigned[seed] or not adj[seed]:
            continue
        members = [seed] + [j for j in adj[seed] if not assigned[j]]
        for m in members:
            assigned[m] = True
        if len(members) > 1:
            clusters.append([ids[m] for m in members])
    return clusters


def build_cache(con, out_path=CACHE):
    """全库照片算 pHash，写 JSONL 缓存（幂等覆盖）。"""
    ids = [r[0] for r in con.execute(
        "SELECT asset_id FROM media_asset WHERE media_type='photo'")]
    t0 = time.time()
    done = miss = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, aid in enumerate(ids, 1):
            tp = thumb_path(aid)
            if not tp:
                miss += 1
                continue
            img = cv2.imread(str(tp))
            if img is None:
                miss += 1
                continue
            f.write(json.dumps({"asset_id": aid, "phash": dhash64(img)}) + "\n")
            done += 1
            if i % 2000 == 0:
                print(f"  pHash {i}/{len(ids)} speed={i/(time.time()-t0):.0f}/s", flush=True)
    print(f"pHash 缓存: {done} 条（缺图 {miss}）耗时 {time.time()-t0:.0f}s → {out_path}")
    return done


# ---------------------------------------------------------------- 引擎 1

def find_phash_dupes(rows, max_dist=PHASH_HAMMING_MAX):
    """全库 pHash 两两比对（分块位运算），返回近重复簇列表。
    2026-09-16：并查集 → 星形聚类（见 _star_clusters），杜绝传递链滚大簇。"""
    ids = np.array([r["asset_id"] for r in rows])
    h = np.array([r["phash"] for r in rows], dtype=np.uint64)
    n = len(h)
    t0 = time.time()
    pairs = []
    B = 1024
    for s in range(0, n, B):
        e = min(s + B, n)
        d = hamming_np(h[s:e, None], h[None, :])
        ii, jj = np.where((d <= max_dist) & (np.arange(s, e)[:, None] < np.arange(n)[None, :]))
        pairs.extend(zip((s + a for a in ii), (int(b) for b in jj)))
    groups = _star_clusters(n, pairs, ids)
    print(f"引擎1 pHash: {n} 张两两比对 {time.time()-t0:.0f}s，近重复对 {len(pairs)}，簇 {len(groups)}")
    return [{"engine": "phash", "assets": g} for g in groups]


# ---------------------------------------------------------------- 引擎 2

def find_siglip_dupes(con, max_dist=SIGLIP_COS_MIN, window=TIME_WINDOW_SEC):
    """同 source 时间邻域内的 SigLIP2 cosine 高相似对 → 星形聚类。
    2026-09-16：并查集 → 星形聚类。90s 窗口只能约束单对，挡不住链式
    传递（实测 30 张的簇横跨 4 天就是这么滚出来的）。"""
    rows = con.execute("""
        SELECT e.subject_id asset_id, e.vector, mf.source_id, ma.capture_time
        FROM embedding e
        JOIN media_asset ma ON ma.asset_id = e.subject_id
        JOIN media_file mf ON mf.asset_id = e.subject_id
        WHERE e.model_name=? AND ma.media_type='photo'
          AND ma.capture_time IS NOT NULL
        ORDER BY mf.source_id, ma.capture_time""", (SIGLIP_MODEL,)).fetchall()
    t0 = time.time()
    n = len(rows)

    from datetime import datetime
    def tsec(t):
        try:
            return datetime.fromisoformat(str(t)[:19].replace("T", " ")).timestamp()
        except ValueError:
            return None

    pairs = []
    i = 0
    while i < n:
        j = i + 1
        ti = tsec(rows[i]["capture_time"])
        si = rows[i]["source_id"]
        # 窗口边界推进：同 source 且时间差 < window
        while j < n and rows[j]["source_id"] == si:
            tj = tsec(rows[j]["capture_time"])
            if ti is None or tj is None or tj - tsec(rows[i]["capture_time"]) > window:
                break
            j += 1
        if j - i > 1:
            vecs = np.vstack([np.frombuffer(r["vector"], dtype="<f4") for r in rows[i:j]])
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / (norms + 1e-9)
            sim = vecs @ vecs.T
            for a in range(j - i):
                for b in range(a + 1, j - i):
                    if sim[a, b] >= max_dist:
                        pairs.append((i + a, i + b))
        i = j if j > i else i + 1
    groups = _star_clusters(n, pairs, [r["asset_id"] for r in rows])
    print(f"引擎2 SigLIP2: {n} 张邻域比对 {time.time()-t0:.0f}s，cos≥{max_dist} 对 {len(pairs)}，簇 {len(groups)}")
    return [{"engine": "siglip", "assets": g} for g in groups]


# ---------------------------------------------------------------- 汇总

def file_size(con, asset_id):
    r = con.execute("SELECT max(byte_size) FROM media_file WHERE asset_id=?", (asset_id,)).fetchone()
    return r[0] or 0


def summarize(con, groups):
    """合并双引擎结果 → 每簇标注推荐保留张（文件最大）。"""
    out = []
    for g in groups:
        sizes = {a: file_size(con, a) for a in g["assets"]}
        keep = max(g["assets"], key=lambda a: sizes[a])
        out.append({"engine": g["engine"], "keep": keep,
                    "duplicates": [a for a in g["assets"] if a != keep],
                    "sizes": sizes})
    return out


def main():
    ap = argparse.ArgumentParser(description="相似照片去重双引擎（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--build", action="store_true", help="重建 pHash 缓存")
    ap.add_argument("--find", action="store_true", help="双引擎找重复")
    ap.add_argument("--out", default=str(ROOT / "data" / "dedup_groups.jsonl"))
    args = ap.parse_args()
    if not args.build and not args.find:
        args.build = args.find = True

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    if args.build:
        build_cache(con)
    if not args.find:
        return
    rows = [json.loads(l) for l in open(CACHE)]
    groups = find_phash_dupes(rows)
    groups += find_siglip_dupes(con)
    result = summarize(con, groups)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_dup = sum(len(r["duplicates"]) for r in result)
    print(f"共 {len(result)} 簇 / {n_dup} 张可清理候选 → {args.out}")
    con.close()


if __name__ == "__main__":
    main()
