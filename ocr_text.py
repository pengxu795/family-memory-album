#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""照片内文字 OCR（新功能⑩：ocr_text.py）

选型：RapidOCR onnxruntime 版（PP-OCRv4 det+cls+rec，~15MB，Apache-2.0，中文强）
- pip 包 rapidocr_onnxruntime 自带模型，零额外下载
- 输入：t480 缩略图（店招/菜单/贺卡大字够用；小字票据精度有限，
  详单级识别等 A4 联调后按需对命中资产做原片重扫）
- 返回行结构 [box, text, conf_str]——conf 是字符串，必须 float() 转换
- 输出 JSONL：asset_id / n_regions / mean_conf / text（拼接，≤600 字）/ has_text
  回写 memory_search（可搜索）留 A4 联调，本模块不写库
- 速度实测 ~0.6-2.9s/张（CPU）→ 全库 12491 张约 4-7h：
  **流式写盘（每条 flush）+ --start 断点**，配合后台 ~2h 墙钟上限分段跑
  （教训同 junk_scanner commit 5252ea5）

用法：
  python3 ocr_text.py --asset <asset_id>          # 单资产
  python3 ocr_text.py --limit 30                  # 抽样验证
  python3 ocr_text.py --all --out data/ocr_text_full.jsonl            # 全量（后台）
  python3 ocr_text.py --all --start 6000 --out ...（append 续跑）
"""
import argparse, json, sqlite3, time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
THUMBS = ROOT / "data" / "thumbs_mvp"
MAX_TEXT_LEN = 600          # 单照片入库文本上限（检索用，不需要全文）


def load_ocr():
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR()


def ocr_image(ocr, img_bgr, min_conf=0.55):
    """→ {n_regions, mean_conf, text}；无文字返回 n_regions=0。"""
    res, _ = ocr(img_bgr)
    if not res:
        return {"n_regions": 0, "mean_conf": 0.0, "text": ""}
    keep = []
    for row in res:
        try:
            conf = float(row[2])
        except (ValueError, TypeError):
            conf = 0.0
        txt = (row[1] or "").strip() if len(row) > 1 else ""
        if txt and conf >= min_conf:
            keep.append((txt, conf))
    if not keep:
        return {"n_regions": 0, "mean_conf": 0.0, "text": ""}
    text = " ".join(t for t, _ in keep)[:MAX_TEXT_LEN]
    return {"n_regions": len(keep),
            "mean_conf": round(float(np.mean([c for _, c in keep])), 3),
            "text": text}


def thumb_image(asset_id):
    # 2026-09-09：补扫发现 306 张缺 OCR 的资产里 299 张只有旧版 {stem}.jpg（400px）缓存，
    # 只认 _t480 会全部 miss → 加 fallback（与 similar_dedup.thumb_path 一致）
    stem = asset_id[6:]
    for name in (f"{stem}_t480.jpg", f"{stem}.jpg"):
        p = THUMBS / name
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def main():
    ap = argparse.ArgumentParser(description="照片内文字 OCR（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--asset", help="单资产")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="断点续跑：跳过前 N 个")
    ap.add_argument("--ids-file", default="", help="只扫清单内 asset_id（一行一个，补漏用）")
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr_text_full.jsonl"))
    ap.add_argument("--min-conf", type=float, default=0.55)
    args = ap.parse_args()

    ocr = load_ocr()

    if args.asset:
        img = thumb_image(args.asset)
        if img is None:
            print("无 t480 缩略图")
            return
        r = ocr_image(ocr, img, args.min_conf)
        print(json.dumps({"asset_id": args.asset, **r}, ensure_ascii=False, indent=1))
        return

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    # ORDER BY asset_id 保证断点续跑顺序稳定
    ids = [r[0] for r in con.execute(
        "SELECT asset_id FROM media_asset WHERE media_type='photo' ORDER BY asset_id")]
    if args.ids_file:
        want = {l.strip() for l in open(args.ids_file, encoding="utf-8")
                if l.strip()}
        ids = [a for a in ids if a in want]
    if args.start:
        ids = ids[args.start:]
    if args.limit and not args.all:
        import random
        random.seed(42)
        ids = random.sample(ids, min(args.limit, len(ids)))

    mode = "a" if args.start else "w"
    out = open(args.out, mode, encoding="utf-8")
    t0 = time.time()
    n_hit = n_done = 0
    for i, aid in enumerate(ids, 1):
        img = thumb_image(aid)
        if img is not None:
            r = ocr_image(ocr, img, args.min_conf)
            r["asset_id"] = aid
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()                     # 2h 墙钟被杀时进度不丢
            n_done += 1
            if r["n_regions"]:
                n_hit += 1
        if i % 200 == 0:
            el = time.time() - t0
            eta = el / i * (len(ids) - i) / 60
            print(f"  进度 {i}/{len(ids)} {i/el:.2f}/s 含字率 {n_hit}/{n_done} "
                  f"ETA {eta:.0f}min 断点={args.start + i}", flush=True)
    out.close()
    print(f"OCR 完成：扫描 {n_done} 张，含文字 {n_hit} 张（{n_hit/max(n_done,1):.0%}），"
          f"耗时 {(time.time()-t0)/60:.0f}min → {args.out}")
    con.close()


if __name__ == "__main__":
    main()
