#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
手动框选建脸 v0.1
==================
照片标注页拖框 → 建 face_instance_v0 + SFace 嵌入(sample_role='manual')。
复用 backfill_faces_full 的加载/质量/嵌入代码, 由 server.py 以 _FACES_PY 拉起。

策略(逐级降级, 嵌入质量优先):
  ① 与该照片已有脸 IoU>0.6 → 直接复用已有脸(防重复建脸)
  ② 全图 YuNet 检测, 中心落框内 → 用检测结果(自带五点, 嵌入最准)
  ③ 框外扩 1.6x 裁剪放大再检测 → 命中则坐标映射回全图
  ④ 兜底: 人工框 + 估计五点(人脸比例先验)

用法: manual_face_create.py --asset-id AID --bbox x,y,w,h   (归一化 0~1)
输出: 单行 JSON {ok, face_instance_id, source, existing}
"""
import argparse, hashlib, json, sqlite3, subprocess, sys
from datetime import datetime, timezone

import cv2
import numpy as np

import backfill_faces_full as bf

ROOT = bf.ROOT
DB = bf.DB


def stamp():
    return datetime.now(timezone.utc).isoformat()


def iou(a, b):
    ax1, ay1, ax2, ay2 = a['x'], a['y'], a['x'] + a['w'], a['y'] + a['h']
    bx1, by1, bx2, by2 = b['x'], b['y'], b['x'] + b['w'], b['y'] + b['h']
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / (a['w'] * a['h'] + b['w'] * b['h'] - inter)


def estimate_landmarks(x, y, w, h):
    """人脸比例先验五点: 左眼/右眼/鼻/左嘴角/右嘴角 (SFace alignCrop 需要)"""
    le = (x + 0.31 * w, y + 0.42 * h)
    re = (x + 0.69 * w, y + 0.42 * h)
    nose = (x + 0.50 * w, y + 0.58 * h)
    lm = (x + 0.36 * w, y + 0.76 * h)
    rm = (x + 0.64 * w, y + 0.76 * h)
    return le, re, nose, lm, rm


def build_face_array(x, y, w, h, points=None, score=0.99):
    if points is None:
        points = estimate_landmarks(x, y, w, h)
    flat = [x, y, w, h]
    for px, py in points:
        flat.extend([px, py])
    flat.append(score)
    return np.array(flat, dtype=np.float32)


def detect_full(detector, image, box_px):
    """全图检测, 返回中心落在人工框内的最优检测(None 表示没命中)"""
    h, w = image.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(image)
    if faces is None:
        return None
    x, y, bw, bh = box_px
    cx, cy = x + bw / 2, y + bh / 2
    best, best_iou = None, 0.0
    for face in faces:
        fx, fy, fw, fh = map(float, face[:4])
        fx2, fy2, fw2, fh2 = fx / w, fy / h, fw / w, fh / h
        i = iou({'x': x / w, 'y': y / h, 'w': bw / w, 'h': bh / h}, {'x': fx2, 'y': fy2, 'w': fw2, 'h': fh2})
        if i > best_iou and x <= cx <= x + bw and y <= cy <= y + bh:
            best, best_iou = face, i
    return best


def detect_crop(detector, image, box_px):
    """外扩 1.6x 裁剪后放大检测, 命中则坐标映射回全图"""
    h, w = image.shape[:2]
    x, y, bw, bh = map(int, box_px)
    ex = int(bw * 0.8)
    ey = int(bh * 0.8)
    x0, y0 = max(0, x - ex), max(0, y - ey)
    x1, y1 = min(w, x + bw + ex), min(h, y + bh + ey)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return None
    crop = image[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    # 小脸放大到 ~600px 再检
    scale = max(1.0, 600.0 / max(cw, ch))
    if scale > 1.01:
        crop = cv2.resize(crop, (round(cw * scale), round(ch * scale)), interpolation=cv2.INTER_CUBIC)
    detector.setInputSize((crop.shape[1], crop.shape[0]))
    _, faces = detector.detect(crop)
    if faces is None:
        return None
    ux0, uy0, ux1, uy1 = x0 + (x - x0) * scale, y0 + (y - y0) * scale, \
        x0 + (x + bw - x0) * scale, y0 + (y + bh - y0) * scale
    cx, cy = (ux0 + ux1) / 2, (uy0 + uy1) / 2
    best = None
    for face in faces:
        fx, fy, fw, fh = map(float, face[:4])
        if fx <= cx <= fx + fw and fy <= cy <= fy + fh:
            best = face
            break
    if best is None:
        return None
    out = best.astype(np.float64).copy()
    out[0] = (out[0] / scale) + x0
    out[1] = (out[1] / scale) + y0
    out[2] = out[2] / scale
    out[3] = out[3] / scale
    for i in range(4, 14):
        out[i] = out[i] / scale + (x0 if i % 2 == 0 else y0)
    return out.astype(np.float32)


def load_asset_image(con, asset_id):
    row = con.execute('''WITH ranked AS (SELECT mf.*,ROW_NUMBER() OVER(PARTITION BY asset_id ORDER BY byte_size DESC,absolute_path) rn FROM media_file mf)
        SELECT r.absolute_path, r.extension, ma.media_type, ma.duration_seconds
        FROM ranked r JOIN media_asset ma USING(asset_id) WHERE r.asset_id=? AND r.rn=1''',
        (asset_id,)).fetchone()
    if not row:
        return None, None
    if row['media_type'] == 'photo':
        img = bf.load_photo(row['absolute_path'], row['extension'])
    else:
        # 与 /thumb 一致: 优先 0.5s 帧, 失败回退第 0 帧
        try:
            img = bf.load_video_frame(row['absolute_path'], 0.5)
        except Exception:
            img = None
        if img is None:
            try:
                img = bf.load_video_frame(row['absolute_path'], 0.0)
            except Exception:
                img = None
    return img, row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--asset-id', required=True)
    ap.add_argument('--bbox', required=True, help='归一化 x,y,w,h 逗号分隔')
    args = ap.parse_args()

    try:
        bx, by, bw, bh = (float(v) for v in args.bbox.split(','))
    except ValueError:
        print(json.dumps({'ok': False, 'error': f'bbox 格式错误: {args.bbox}'}))
        sys.exit(0)
    if not (0 <= bx <= 1 and 0 <= by <= 1 and 0.01 <= bw <= 1 and 0.01 <= bh <= 1):
        print(json.dumps({'ok': False, 'error': f'bbox 数值越界: {args.bbox}'}))
        sys.exit(0)
    if bx + bw > 1.02 or by + bh > 1.02:
        bx, by = min(bx, 1 - bw), min(by, 1 - bh)

    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA journal_mode=WAL')
    con.isolation_level = None

    # ① 已有脸高度重叠 → 复用
    user_box = {'x': bx, 'y': by, 'w': bw, 'h': bh}
    for r in con.execute('SELECT face_instance_id, bbox_json FROM face_instance_v0 WHERE asset_id=?',
                         (args.asset_id,)).fetchall():
        old = json.loads(r['bbox_json'])
        if iou(user_box, {k: old[k] for k in ('x', 'y', 'w', 'h')}) > 0.6:
            print(json.dumps({'ok': True, 'existing': True, 'source': 'reuse',
                              'face_instance_id': r['face_instance_id']}))
            con.close()
            return

    image, row = load_asset_image(con, args.asset_id)
    if image is None:
        print(json.dumps({'ok': False, 'error': '原图读取失败(NAS 掉线/视频抽帧失败)'}))
        con.close()
        return
    h, w = image.shape[:2]
    box_px = (bx * w, by * h, bw * w, bh * h)

    detector = cv2.FaceDetectorYN_create(str(bf.YUNET), '', (320, 320), 0.6, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF_create(str(bf.SFACE), '')

    face_arr, source = None, None
    face = detect_full(detector, image, box_px)
    if face is not None:
        face_arr, source = face[0], 'yunet-full'
    else:
        face = detect_crop(detector, image, box_px)
        if face is not None:
            face_arr, source = face, 'yunet-crop'
    if face_arr is None:
        face_arr = build_face_array(*box_px)
        source = 'manual-estimate'

    x, y, fw, fh = map(float, face_arr[:4])
    blur, pose, qclass = bf.quality(face_arr, image)
    # 人工确认的脸一律 usable: 检测质量只做参考
    norm_box = {'x': x / w, 'y': y / h, 'w': fw / w, 'h': fh / h, 'image_width': w, 'image_height': h}
    landmarks = [{'x': float(face_arr[i]) / w, 'y': float(face_arr[i + 1]) / h} for i in range(4, 14, 2)]
    fid = 'face_' + hashlib.sha256(f"{args.asset_id}:manual:{stamp()}:{bx:.4f},{by:.4f}".encode()).hexdigest()[:24]

    con.execute('''INSERT INTO face_instance_v0
      (face_instance_id,asset_id,frame_time_seconds,frame_key,bbox_json,landmarks_json,
       detection_score,face_width,face_height,blur_score,pose_score,quality_class,sample_role,detection_model,created_at)
      VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
        fid, args.asset_id, None if row['media_type'] == 'photo' else 0.5,
        f"{args.asset_id}:manual", json.dumps(norm_box), json.dumps(landmarks),
        0.99, round(fw), round(fh), blur, pose, 'usable', 'manual', f'manual-box+{source}', stamp()))
    try:
        aligned = recognizer.alignCrop(image, face_arr)
        emb = recognizer.feature(aligned).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        emb = emb / norm if norm else emb
        con.execute('INSERT INTO face_embedding_v0 VALUES(?,?,?,?,?,?,?)',
                    (fid, 'SFace-2021dec', len(emb), emb.astype('<f4').tobytes(), norm, 'success', stamp()))
    except Exception as exc:
        con.execute('INSERT INTO face_embedding_v0 VALUES(?,?,?,?,?,?,?)',
                    (fid, 'SFace-2021dec', 0, None, None, 'failed', stamp()))
        print(json.dumps({'ok': True, 'face_instance_id': fid, 'source': source,
                          'embed': 'failed', 'warn': str(exc)}))
        con.close()
        return
    con.close()
    print(json.dumps({'ok': True, 'face_instance_id': fid, 'source': source}))


if __name__ == '__main__':
    main()
