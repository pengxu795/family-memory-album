#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""模糊照片检测模块（新功能①：blur 质量分类）

选型结论（2026-09-09 实验，20 糊脸照片 vs 19 锐脸照片对照）：
- 采用**纯 CV 多指标**：分块 Laplacian 方差（主体锐度 topq + 全图中位 medq），
  可选 Tenengrad 交叉验证。糊/锐两组 topq 中位 133 vs 6445（48 倍差、分布无重叠），
  单指标已足够区分，无需学习型模型。
- 不选 BRISQUE（要 opencv-contrib，引入新依赖）/ NIMA（Torch 模型太重、NAS 读图慢）。
- FFT 高频比实测无增益（实现成本高、区分度不叠加），砍掉。

关键设计：
- **分块 top-25%**：整图 Laplacian 会被大面积虚化背景/纯色墙拖低造成假阳；
  取 4×4 分块中锐度最高的 25% 块均值 = 「画面里最锐利主体」的锐度——
  家庭照片主体（人/物）糊了才是真糊，背景虚化是景深不是废片。
- **t480 缩略图**：直接读 data/thumbs_mvp/{id}_t480.jpg（480px），
  不读 NAS 原片（IO 从 3.6s/张 降到 <50ms/张）；缩略图分辨率已足够区分。
- **不写库**：A4 全库重算进行中，本模块只输出 JSONL/统计，回写
  media_asset.quality_score 留 A4 联调后统一执行。

阈值（t480 口径，log 尺度分档，可在标定后调整）：
  blurry < 300 < soft < 1200 < sharp
用法：
  python3 blur_detector.py --limit 200          # 抽样统计
  python3 blur_detector.py --all --out /tmp/x   # 全库评估 → JSONL + 分布摘要
  python3 blur_detector.py --image path.jpg     # 单图
"""
import argparse, json, sqlite3, sys, time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DB = Path(__file__).resolve().parent / "data" / "family_memory.db"
THUMBS = ROOT / "data" / "thumbs_mvp"

# 分类阈值（t480 缩略图口径）
T_BLURRY, T_SOFT = 300.0, 1200.0


def lap_block_scores(gray, grid=4):
    """grid×grid 分块 Laplacian 方差，返回块分数列表（升序）。"""
    h, w = gray.shape
    bs = max(48, min(h, w) // grid)
    scores = []
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            blk = gray[y:y + bs, x:x + bs]
            scores.append(cv2.Laplacian(blk, cv2.CV_64F).var())
    if not scores:  # 图小于块
        scores = [cv2.Laplacian(gray, cv2.CV_64F).var()]
    scores.sort()
    return scores


def classify_image(img_bgr):
    """输入 BGR 图，返回 {sharp_score, med_score, label}。
    sharp_score = 分块 top-25% Laplacian 方差均值（画面最锐利主体的锐度）。"""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    scores = lap_block_scores(gray)
    topk = max(1, len(scores) // 4)
    topq = float(np.mean(scores[-topk:]))
    med = float(np.median(scores))
    label = "blurry" if topq < T_BLURRY else ("soft" if topq < T_SOFT else "sharp")
    return {"sharp_score": round(topq, 1), "med_score": round(med, 1), "label": label}


def thumb_path(asset_id):
    """资产 → 缩略图路径（t480 优先，jpg 落盘版）。"""
    stem = asset_id[6:]  # asset_ 前缀去掉
    for cand in (f"{stem}_t480.jpg", f"{stem}.jpg"):
        p = THUMBS / cand
        if p.exists():
            return p
    return None


def load_asset_image(con, asset_id):
    """缩略图优先，兜底原片缩到 960。"""
    tp = thumb_path(asset_id)
    if tp:
        img = cv2.imread(str(tp))
        if img is not None:
            return img
    row = con.execute(
        "SELECT absolute_path, extension FROM media_file WHERE asset_id=? "
        "ORDER BY byte_size DESC LIMIT 1", (asset_id,)).fetchone()
    if not row:
        return None
    img = cv2.imread(row["absolute_path"])
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = 960 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def main():
    ap = argparse.ArgumentParser(description="模糊照片检测（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--limit", type=int, default=0, help="随机抽样 N 张")
    ap.add_argument("--all", action="store_true", help="全库照片评估")
    ap.add_argument("--image", help="单图模式")
    ap.add_argument("--out", default="", help="JSONL 输出路径")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"读不了: {args.image}")
        print(json.dumps(classify_image(img), ensure_ascii=False))
        return

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    q = "SELECT asset_id FROM media_asset WHERE media_type='photo'"
    ids = [r["asset_id"] for r in con.execute(q)]
    if args.limit and not args.all:
        import random
        random.seed(42)
        ids = random.sample(ids, min(args.limit, len(ids)))
    elif not args.all and args.limit == 0:
        args.limit = 200
        import random
        random.seed(42)
        ids = random.sample(ids, min(args.limit, len(ids)))

    out = open(args.out, "w", encoding="utf-8") if args.out else None
    labels = {"sharp": 0, "soft": 0, "blurry": 0}
    t0, miss = time.time(), 0
    for i, aid in enumerate(ids, 1):
        img = load_asset_image(con, aid)
        if img is None:
            miss += 1
            continue
        r = classify_image(img)
        r["asset_id"] = aid
        labels[r["label"]] += 1
        if out:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        if i % 500 == 0:
            print(f"  进度 {i}/{len(ids)} speed={i/(time.time()-t0):.0f}/s", flush=True)
    if out:
        out.close()
    n = sum(labels.values())
    print(f"评估 {n} 张（缺图 {miss}）耗时 {time.time()-t0:.0f}s "
          f"({n/(time.time()-t0):.0f}/s)" if n else "无可评估照片")
    print(f"分类分布: sharp={labels['sharp']} ({labels['sharp']*100//max(n,1)}%) "
          f"soft={labels['soft']} ({labels['soft']*100//max(n,1)}%) "
          f"blurry={labels['blurry']} ({labels['blurry']*100//max(n,1)}%)")
    con.close()


if __name__ == "__main__":
    main()
