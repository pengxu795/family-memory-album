#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""五张选优美学评分（新功能③：LAION Aesthetic Predictor）

选型（2026-09-09）：LAION Aesthetic Predictor V2（sa_0_4 ViT-B/32 线性头）
- 头：models/aesthetic/sa_0_4_vit_b_32_linear.pth（1×512 线性，官方 GitHub 3KB）
- 底座：open_clip ViT-B-32 laion2b_s34b_b79k——本机 HF cache 已缓存（577M），零下载
- 输出 0-10 分（AVA 数据集美学口径），官方推荐配对 laion 权重（嵌入空间一致）
- 不选 L/14 版头（需再下 1.7GB CLIP）；不选 NIMA（重、慢）；纯 CV 无法表达"美"

用法：
  python3 aesthetic_scorer.py --limit 200          # 抽样分布
  python3 aesthetic_scorer.py --all --out x.jsonl  # 全库
  python3 aesthetic_scorer.py --top5-by-day        # 五张选优：每天 top5 → JSONL
  python3 aesthetic_scorer.py --image path.jpg     # 单图
"""
import argparse, json, sqlite3, time
from pathlib import Path

import numpy as np
import torch
import open_clip
import cv2

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
THUMBS = ROOT / "data" / "thumbs_mvp"
HEAD = ROOT.parent / "models" / "aesthetic" / "sa_0_4_vit_b_32_linear.pth"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

_model = None
_head = None
_preprocess = None


def _load():
    global _model, _head, _preprocess
    if _model is not None:
        return
    _model, _, _preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    _model.eval()
    sd = torch.load(str(HEAD), map_location="cpu", weights_only=True)
    w = sd["weight"].float().flatten()
    b = float(sd["bias"].flatten()[0])
    _head = (w, b)


def score_image(img_bgr):
    """BGR 图 → 美学分（原始连续值，可负可超 10——排序/选优只用相对值，
    clamp 到 0-10 会把 sa_0_4 头的宽分布压扁失去区分度）。"""
    _load()
    pil = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    from PIL import Image
    pil = Image.fromarray(pil)
    t = _preprocess(pil).unsqueeze(0)
    with torch.no_grad():
        emb = _model.encode_image(t).float().flatten()
    w, b = _head
    return float(w @ emb + b)


def thumb_path(asset_id):
    stem = asset_id[6:]
    for cand in (f"{stem}_t480.jpg", f"{stem}.jpg"):
        p = THUMBS / cand
        if p.exists():
            return p
    return None


def load_asset_image(con, asset_id):
    tp = thumb_path(asset_id)
    if tp:
        img = cv2.imread(str(tp))
        if img is not None:
            return img
    row = con.execute(
        "SELECT absolute_path FROM media_file WHERE asset_id=? ORDER BY byte_size DESC LIMIT 1",
        (asset_id,)).fetchone()
    if not row:
        return None
    img = cv2.imread(row["absolute_path"])
    if img is None:
        return None
    h, w2 = img.shape[:2]
    scale = 480 / max(h, w2)
    if scale < 1:
        img = cv2.resize(img, (int(w2 * scale), int(h * scale)))
    return img


def _score_ids(con, ids, out):
    t0, miss = time.time(), 0
    results = []
    for i, aid in enumerate(ids, 1):
        img = load_asset_image(con, aid)
        if img is None:
            miss += 1
            continue
        s = score_image(img)
        results.append({"asset_id": aid, "aesthetic": round(s, 3)})
        if out:
            out.write(json.dumps({"asset_id": aid, "aesthetic": round(s, 3)}) + "\n")
        if i % 100 == 0:
            if out:
                out.flush()  # 流式落盘：被杀不丢已完成部分
            print(f"  进度 {i}/{len(ids)} speed={i/(time.time()-t0):.1f}/s", flush=True)
    return results, miss


def main():
    ap = argparse.ArgumentParser(description="LAION 美学评分 / 五张选优（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="断点续跑：跳过前 N 个（配合 --out 追加）")
    ap.add_argument("--top5-by-day", action="store_true", help="每天选美学前 5 张")
    ap.add_argument("--image", help="单图模式")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys_exit = f"读不了: {args.image}"
            print(sys_exit)
            return
        print(json.dumps({"aesthetic": round(score_image(img), 3)}))
        return

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    out = open(args.out, "a" if args.start else "w", encoding="utf-8") if args.out else None

    if args.top5_by_day:
        days = [r[0] for r in con.execute(
            """SELECT DISTINCT substr(capture_time,1,10) d FROM media_asset
               WHERE media_type='photo' AND capture_time NOT LIKE '0000%'
               ORDER BY d DESC""")]
        picks = []
        t0 = time.time()
        for di, d in enumerate(days, 1):
            ids = [r[0] for r in con.execute(
                """SELECT asset_id FROM media_asset WHERE media_type='photo'
                   AND substr(capture_time,1,10)=?""", (d,))]
            res, _miss = _score_ids(con, ids, None)
            res.sort(key=lambda x: -x["aesthetic"])
            for r in res[:5]:
                r["day"] = d
            picks += res[:5]
            if di % 30 == 0:
                print(f"  天 {di}/{len(days)} speed={(di/(time.time()-t0)):.1f} 天/s", flush=True)
        if out:
            for p in picks:
                out.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"五张选优: {len(days)} 天 → {len(picks)} 张 → {args.out or 'stdout省略'}")
        con.close()
        return

    # ORDER BY 固定全序：断点续跑接缝不错位
    ids = [r[0] for r in con.execute(
        "SELECT asset_id FROM media_asset WHERE media_type='photo' ORDER BY asset_id")]
    if args.start:
        ids = ids[args.start:]
    elif args.limit and not args.all:
        import random
        random.seed(42)
        ids = random.sample(ids, min(args.limit, len(ids)))
    elif not args.all and args.limit == 0:
        import random
        random.seed(42)
        ids = random.sample(ids, 200)
    res, miss = _score_ids(con, ids, out)
    scores = np.array([r["aesthetic"] for r in res])
    if len(scores):
        print(f"评估 {len(scores)} 张（缺图 {miss}）| "
              f"分位: p10={np.percentile(scores,10):.1f} 中位={np.median(scores):.1f} "
              f"p90={np.percentile(scores,90):.1f} | ≥7 分 {np.mean(scores>=7)*100:.0f}%")
    con.close()


if __name__ == "__main__":
    main()
