#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""智能 4:3 自动裁切（新功能⑤：smart_crop.py，相册展示卡用）

四级策略（2026-09-09 设计）：
1. face      有人脸 → 复用 YuNet 检测框（独立加载 models/adaface/ 里的 onnx，
             不 import face_backends 避免耦合 A4），所有脸的最小包围盒 +
             边距（脸高 40%）对齐 4:3 窗中心；脸群装不下时窗取最大、中心=质心
2. saliency  无脸 → U²-Netp 显著性（models/u2netp.onnx，4.5MB，Apache-2.0，
             rembg 同源导出）热图阈值化 → 主体包围盒/重心定位裁切窗
3. horizon   无脸且显著性均匀（风景）→ 地平线检测（Sobel 垂直梯度行能量
             峰值），地平线放 1/3 线（上 1/3 天空多取 2/3，反之 1/3）
4. center    兜底中心裁切

输出：归一化裁切参数 {x, y, w, h, method}（0-1 浮点，相对原图）——
只存参数不生成新图，前端按参数裁。不写库（回写留 A4 联调）。

用法：
  python3 smart_crop.py --image path.jpg        # 单图 → JSON
  python3 smart_crop.py --limit 100             # 抽样统计
  python3 smart_crop.py --all --out x.jsonl     # 全库（后台跑）
"""
import argparse, json, sqlite3, time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
THUMBS = ROOT / "data" / "thumbs_mvp"
YUNET_ONNX = ROOT.parent / "models" / "adaface" / "face_detection_yunet_2023mar.onnx"
U2NETP_ONNX = ROOT.parent / "models" / "u2netp.onnx"

_face_det = None
_sal_sess = None


def _face_detector(w=480):
    global _face_det
    if _face_det is None:
        _face_det = cv2.FaceDetectorYN_create(str(YUNET_ONNX), "", (w, w),
                                              0.72, 0.3, 5000)
    return _face_det


def detect_faces(img_bgr):
    """YuNet 人脸框（像素坐标 [x,y,w,h]）。"""
    h, w = img_bgr.shape[:2]
    det = _face_detector()
    det.setInputSize((w, h))
    _, faces = det.detect(img_bgr)
    if faces is None:
        return []
    return [f[:4].tolist() for f in faces]


def saliency_map(img_bgr):
    """U²-Netp 显著性图（0-1 float，输入分辨率 320）。"""
    global _sal_sess
    if _sal_sess is None:
        import onnxruntime as ort
        _sal_sess = ort.InferenceSession(str(U2NETP_ONNX),
                                         providers=["CPUExecutionProvider"])
    h, w = img_bgr.shape[:2]
    # rembg 导出的 u2netp：输入 = RGB/255，无 mean/std
    blob = cv2.dnn.blobFromImage(img_bgr, scalefactor=1.0 / 255, size=(320, 320),
                                 swapRB=True)
    out = _sal_sess.run(None, {_sal_sess.get_inputs()[0].name: blob})[0][0, 0]
    out = (out - out.min()) / (out.max() - out.min() + 1e-9)
    return cv2.resize(out, (w, h))


def _place_window(W, H, cx, cy):
    """以 (cx,cy) 为中心的 4:3 窗（像素），clamp 在图内。"""
    win_w = min(float(W), H * 4.0 / 3.0)
    win_h = win_w * 3.0 / 4.0
    x = min(max(cx - win_w / 2, 0), W - win_w)
    y = min(max(cy - win_h / 2, 0), H - win_h)
    return x, y, win_w, win_h


def _norm(r, W, H, method):
    return {"method": method,
            "x": round(r[0] / W, 4), "y": round(r[1] / H, 4),
            "w": round(r[2] / W, 4), "h": round(r[3] / H, 4)}


def crop_params(img_bgr):
    """主入口：BGR 图 → 裁切参数 dict。"""
    H, W = img_bgr.shape[:2]
    faces = detect_faces(img_bgr)
    if faces:
        xs1 = [f[0] for f in faces]; ys1 = [f[1] for f in faces]
        xs2 = [f[0] + f[2] for f in faces]; ys2 = [f[1] + f[3] for f in faces]
        bx1, by1 = max(0, min(xs1)), max(0, min(ys1))
        bx2, by2 = min(W, max(xs2)), min(H, max(ys2))
        fw, fh = bx2 - bx1, by2 - by1
        margin = fh * 0.4  # 安全边距：脸高 40%
        tx = (bx1 + bx2) / 2
        top_need = by1 - margin * 0.5  # 窗顶须在此之上（最高脸+边距）
        if fh * 4.0 / 3.0 > W:
            # 竖图人像特写：4:3 横窗物理上装不下脸群（脸高×4/3 > 图宽）→
            # 窗转 3:4 竖版（4:3 的竖向形式），前端可按窗宽高比渲染；
            # 完全装不下（脸群高>H）时中心=脸群质心尽力。
            win_h = min(float(H), W * 4.0 / 3.0)
            win_w = win_h * 3.0 / 4.0
            if win_h >= fh:
                y = max(0, min(top_need, H - win_h))
                x = max(0, min(tx - win_w / 2, W - win_w))
            else:
                x = max(0, min(tx - win_w / 2, W - win_w))
                y = max(0, min((by1 + by2) / 2 - win_h / 2, H - win_h))
            return _norm((x, y, win_w, win_h), W, H, "face")
        x, y, ww, hh = _place_window(W, H, tx, (by1 + by2) / 2 - margin * 0.3)
        if y > top_need:  # 窗顶切到最高脸 → 上移（宁切脚下）
            y = max(0, min(top_need, H - hh))
        return _norm((x, y, ww, hh), W, H, "face")

    sal = saliency_map(img_bgr)
    if sal.std() > 0.12:  # 有显著主体
        m = sal > 0.5
        if m.sum() > 0.005 * m.size:
            ys, xs = np.where(m)
            cx, cy = float(xs.mean()), float(ys.mean())
            bw = xs.max() - xs.min()
            bh = ys.max() - ys.min()
            x, y, ww, hh = _place_window(W, H, cx, cy)
            # 窗太小装不下主体 → 扩窗（4:3 约束下取最大窗、中心=主体重心）
            if ww < bw * 1.1 or hh < bh * 1.1:
                ww = min(float(W), H * 4.0 / 3.0)
                hh = ww * 3.0 / 4.0
                x = min(max(cx - ww / 2, 0), W - ww)
                y = min(max(cy - hh / 2, 0), H - hh)
            return _norm((x, y, ww, hh), W, H, "saliency")

    # 风景：地平线检测（Sobel_y 垂直梯度行能量峰 = 水平分界线；
    # 注意 Sobel(1,0) 是竖直边缘检测子，对水平线响应为 0）
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)).mean(axis=1)
    if gy.max() > gy.mean() * 1.8:  # 存在明显水平分界
        hy = int(np.argmax(gy[max(1, H//10): H*9//10]) + H // 10)
        frac = hy / H
        if frac < 0.5:   # 地平线在上半 → 天空少，窗对齐下 2/3（地平线放窗上 1/3 线）
            cy = min(hy + (H - hy) * 0.5, H - H * 3 / 8)
        else:            # 地平线在下半 → 地平线放窗下 1/3 线
            cy = max(hy - H * 3 / 16, H * 3 / 8)
        x, y, ww, hh = _place_window(W, H, W / 2, cy)
        return _norm((x, y, ww, hh), W, H, "horizon")

    return _norm(_place_window(W, H, W / 2, H / 2), W, H, "center")


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
    h, w = img.shape[:2]
    s = 480 / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)))
    return img


def main():
    ap = argparse.ArgumentParser(description="智能 4:3 裁切（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="断点续跑：跳过前 N 个（配合 --out 追加）")
    ap.add_argument("--image", help="单图模式")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"读不了: {args.image}")
            return
        print(json.dumps(crop_params(img), ensure_ascii=False))
        return

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
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

    out = open(args.out, "a" if args.start else "w", encoding="utf-8") if args.out else None
    from collections import Counter
    methods = Counter()
    t0, miss = time.time(), 0
    for i, aid in enumerate(ids, 1):
        img = load_asset_image(con, aid)
        if img is None:
            miss += 1
            continue
        r = crop_params(img)
        r["asset_id"] = aid
        methods[r["method"]] += 1
        if out:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
        if i % 200 == 0:
            if out:
                out.flush()  # 流式落盘：会话重启/被杀不丢已完成部分
            print(f"  进度 {i}/{len(ids)} speed={i/(time.time()-t0):.1f}/s", flush=True)
    if out:
        out.close()
    print(f"评估 {sum(methods.values())} 张（缺图 {miss}）耗时 {time.time()-t0:.0f}s")
    for k, v in methods.most_common():
        print(f"  {k}: {v} ({v*100//max(sum(methods.values()),1)}%)")
    con.close()


if __name__ == "__main__":
    main()
