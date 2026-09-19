#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""闭眼/微笑检测（新功能⑪：face_expr.py）——「五张选优」内容质量信号

选型：insightface 2d106det.onnx（4.8MB，buffalo_l 同源，非商业许可，仅本地自用）
- 输入 1×3×192×192；**RGB 原始 0~255**（图内烘焙 (x-127.5)*1/128，同 genderage 实证）
- 输出 212 = 106 点×2，**归一化坐标，解码 (pred+1)*96**（不是 ×192，官方
  landmark.py: pred+=1; pred*=input_size//2 —— 实证踩坑）
- 对齐：**bbox 中心 + 1.5 倍边距**（官方 face_align.transform 同款，不是 kps 紧对齐；
  kps 对齐点云只落在下半脸，实证踩坑）

106 点索引（视觉+数值双实证，40 张对照）：
  0-32 脸轮廓 | 33-42 右眼（图像左） | 88-95 左眼（图像右）
  43-51/96-105 眉 | 52-71 嘴（52/58 嘴角） | 72-87 鼻

指标（60 张标定）：
  眼开合比 = 眼点 y 跨度 / 眼宽    闭眼 ≤0.25 / 睁眼 ≥0.30 → 阈值 0.28
  微笑比   = 嘴宽(52-58) / 瞳距    无表情 ≤0.60 / 明显笑 ≥0.75 → 阈值 0.72
  输出为软信号：选优时 smiling>0 加权、eyes_open<0 惩罚，不建议硬过滤

输出 JSONL（每脸一行）：asset_id / face_instance_id / eye_ratio / eyes_open /
  smile_ratio / is_smiling。回写留 A4 联调。流式 flush + --start 断点（2h 墙钟教训）。

用法：
  python3 face_expr.py --asset <asset_id>   # 单资产逐脸
  python3 face_expr.py --limit 30           # 抽样验证
  python3 face_expr.py --all --out data/face_expr_full.jsonl   # 全量（后台）
"""
import argparse, json, sqlite3, time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
LM_ONNX = ROOT.parent / "models" / "2d106det.onnx"
THUMBS = ROOT / "data" / "thumbs_mvp"

EYE_OPEN_TH = 0.28      # 眼开合比阈值（闭≤0.25 / 睁≥0.30）
SMILE_TH = 0.72         # 微笑比阈值（平≤0.60 / 笑≥0.75）

_sess = None


def _session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        _sess = ort.InferenceSession(str(LM_ONNX),
                                     providers=["CPUExecutionProvider"])
    return _sess


def align_bbox15(img_bgr, bbox, size=192):
    """bbox 中心 + 1.5 倍边距对齐（官方 landmark 预处理同款）。"""
    H, W = img_bgr.shape[:2]
    cx = bbox["x"] * W + bbox["w"] * W / 2
    cy = bbox["y"] * H + bbox["h"] * H / 2
    side = max(bbox["w"] * W, bbox["h"] * H) * 1.5
    M = cv2.getRotationMatrix2D((cx, cy), 0, float(size) / side)
    M[0, 2] += size / 2 - cx
    M[1, 2] += size / 2 - cy
    return M


def landmarks_106(img_bgr, bbox):
    """→ 106×2 点（192 帧坐标）。"""
    M = align_bbox15(img_bgr, bbox)
    face = cv2.warpAffine(img_bgr, M, (192, 192))
    t = face[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)   # RGB 0-255
    pred = _session().run(None, {"data": t[None]})[0][0].reshape(106, 2)
    return (pred + 1.0) * 96.0


def metrics_from_pred(pred):
    """106×2 点（192 帧）→ 指标 dict。纯函数，便于测试。"""
    def eye_ratio(pts):
        eye_w = float(np.ptp(pts[:, 0]))
        if eye_w < 1:
            return None
        return round(float(np.ptp(pts[:, 1])) / eye_w, 3)
    rr, rl = eye_ratio(pred[33:43]), eye_ratio(pred[88:96])
    if rr is None or rl is None:
        return None
    if rr > 1.0 or rl > 1.0:
        return None                     # 坏 bbox / 非脸：landmark 不可靠（实测玩偶误检 2.58）
    eye_dist = float(np.linalg.norm(pred[33:43].mean(0) - pred[88:96].mean(0))) + 1e-6
    mouth_w = float(np.linalg.norm(pred[52] - pred[58]))
    smile_ratio = round(mouth_w / eye_dist, 3)
    if smile_ratio > 1.1:
        return None                     # 嘴宽超物理范围 → bbox 不可信
    eye_avg = (rr + rl) / 2
    return {"eye_ratio_r": rr, "eye_ratio_l": rl,
            "eyes_open": eye_avg > EYE_OPEN_TH,
            "eye_open_score": round(eye_avg, 3),
            "smile_ratio": smile_ratio,
            "is_smiling": smile_ratio > SMILE_TH}


def face_metrics(img_bgr, bbox):
    """→ 指标 dict 或 None（ landmark 不可靠时）。"""
    pred = landmarks_106(img_bgr, bbox)
    return metrics_from_pred(pred)


def thumb_image(asset_id):
    p = THUMBS / f"{asset_id[6:]}_t480.jpg"
    return cv2.imread(str(p)) if p.exists() else None


def main():
    ap = argparse.ArgumentParser(description="闭眼/微笑检测（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--asset", help="单资产逐脸")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="断点续跑：跳过前 N 个")
    ap.add_argument("--ids-file", default="",
                    help="只算清单里的 face_instance_id（补扫 A4 重检测新增脸）")
    ap.add_argument("--out", default=str(ROOT / "data" / "face_expr_full.jsonl"))
    args = ap.parse_args()

    if args.asset:
        img = thumb_image(args.asset)
        if img is None:
            print("无 t480 缩略图")
            return
        con = sqlite3.connect(args.db, timeout=60)
        con.row_factory = sqlite3.Row
        out = []
        for r in con.execute("SELECT face_instance_id, bbox_json FROM face_instance_v0 "
                             "WHERE asset_id=? AND quality_class='usable'", (args.asset,)):
            m = face_metrics(img, json.loads(r["bbox_json"]))
            if m:
                m["face_instance_id"] = r["face_instance_id"]
                out.append(m)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        return

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    # ORDER BY asset_id 保证断点顺序稳定
    ids = [r[0] for r in con.execute(
        "SELECT asset_id FROM media_asset WHERE media_type='photo' ORDER BY asset_id")]
    face_want = None
    if args.ids_file:
        # 补扫口径：只处理清单里的脸（按脸清单反查资产，避免全库重扫）
        want = {l.strip() for l in open(args.ids_file, encoding="utf-8") if l.strip()}
        pairs = con.execute("SELECT face_instance_id, asset_id FROM face_instance_v0 "
                            "WHERE quality_class='usable'").fetchall()
        face_want = {fid for fid, _ in pairs if fid in want}
        ids = sorted({aid for _, aid in pairs if aid in
                      {a for f, a in pairs if f in face_want}})
        print(f"ids-file: {len(want)} 目标，命中 {len(face_want)} 张脸 / {len(ids)} 资产")
    if args.start:
        ids = ids[args.start:]
    if args.limit and not args.all:
        import random
        random.seed(42)
        ids = random.sample(ids, min(args.limit, len(ids)))

    out = open(args.out, "a" if args.start else "w", encoding="utf-8")
    t0 = time.time()
    n_face = n_open = n_smile = 0
    for i, aid in enumerate(ids, 1):
        img = thumb_image(aid)
        if img is not None:
            H, W = img.shape[:2]
            for r in con.execute("SELECT face_instance_id, bbox_json FROM face_instance_v0 "
                                 "WHERE asset_id=? AND quality_class='usable'", (aid,)):
                if face_want is not None and r["face_instance_id"] not in face_want:
                    continue
                m = face_metrics(img, json.loads(r["bbox_json"]))
                if m:
                    m["asset_id"] = aid
                    m["face_instance_id"] = r["face_instance_id"]
                    out.write(json.dumps(m, ensure_ascii=False) + "\n")
                    n_face += 1
                    n_open += m["eyes_open"]
                    n_smile += m["is_smiling"]
            out.flush()                      # 2h 墙钟被杀时进度不丢
        if i % 500 == 0:
            el = time.time() - t0
            eta = el / i * (len(ids) - i) / 60
            print(f"  进度 {i}/{len(ids)} {i/el:.1f}/s 脸{n_face} 睁眼{n_open} "
                  f"微笑{n_smile} ETA {eta:.0f}min 断点={args.start + i}", flush=True)
    out.close()
    print(f"完成：{n_face} 脸，睁眼 {n_open}，微笑 {n_smile}，"
          f"耗时 {(time.time()-t0)/60:.0f}min → {args.out}")
    con.close()


if __name__ == "__main__":
    main()
