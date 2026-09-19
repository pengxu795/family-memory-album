#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人脸性别年龄识别（新功能⑥：genderage.py）

选型：insightface genderage.onnx（buffalo_l 官方包内，1.3MB，models/genderage.onnx）
- 输入 1×3×96×96 NCHW；模型图开头烘焙了 (data-127.5)*1/128，
  因此喂 **RGB 原始像素 0~255**（不要 /255，不要 ±127.5）
- 输出 [1,3]：[:2] 性别两值（**实证映射 argmax==1 → male**，
  与 insightface 官方 0=male 相反，本库 300 张对照实测），[2] age×100
- 对齐：复用 face_instance_v0 的 5 点 landmarks + ArcFace 模板（96 口径），
  与人脸识别管线同一套对齐；A/B 实测 kps 对齐分箱 63%/儿童 45%/性别 86%，
  bbox1.5 对齐 61%/36%/93% —— 取 kps（儿童与分箱优先）
- 已知短板（300 张生日真值实测，真值=拍摄时年龄）：
  年龄误差中位 ~9 岁、分箱命中 63%、儿童(<16)仅 45% —— 模型对儿童系统性
  高估（训练集以成人为主）。**已知家庭成员不要用模型猜年龄**，
  用 person.birth_date + capture_time 精确算；本模块只给未知人脸打粗标签
- 许可：genderage 属 insightface 官方 zoo（非商业研究许可）；仅本地自用，
  不随产品出厂分发（同 insightface 后端红线）

不写库：结果 JSONL；回写 face_instance_v0（加列）留 A4 联调。

用法：
  python3 genderage.py --asset <asset_id>     # 单资产逐脸
  python3 genderage.py --limit 200            # 抽样统计（含生日对照）
  python3 genderage.py --all --out x.jsonl    # 全库（后台）
"""
import argparse, json, sqlite3, time
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
GA_ONNX = ROOT.parent / "models" / "genderage.onnx"
THUMBS = ROOT / "data" / "thumbs_mvp"

# ArcFace 标准 5 点模板（112×112 口径，缩放到 96）
ARC_SRC = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32) * (96.0 / 112.0)

_sess = None


def _session():
    global _sess
    if _sess is None:
        import onnxruntime as ort
        _sess = ort.InferenceSession(str(GA_ONNX),
                                     providers=["CPUExecutionProvider"])
    return _sess


def umeyama_norm(src5, dst5):
    """相似变换（旋转+缩放+平移）把 5 点对齐到模板。"""
    M, _ = cv2.estimateAffinePartial2D(
        np.asarray(src5, dtype=np.float32),
        np.asarray(dst5, dtype=np.float32))
    assert M is not None, "相似变换求解失败"
    return M


def align_face(img_bgr, bbox, kps_norm):
    """bbox/kps（归一化）→ 96×96 对齐人脸图。kps 缺失时退化为 bbox 裁剪缩放。"""
    h, w = img_bgr.shape[:2]
    if kps_norm:
        kps = np.array([[p["x"] * w, p["y"] * h] for p in kps_norm], dtype=np.float32)
        if kps.shape == (5, 2):
            M = umeyama_norm(kps, ARC_SRC)
            return cv2.warpAffine(img_bgr, M, (96, 96))
    x, y, bw, bh = bbox["x"] * w, bbox["y"] * h, bbox["w"] * w, bbox["h"] * h
    crop = img_bgr[max(0, int(y)):int(y + bh), max(0, int(x)):int(x + bw)]
    return cv2.resize(crop, (96, 96)) if crop.size else None


def predict(img_bgr, bbox, kps_norm):
    """→ {gender: 'male'|'female', age: int}

    预处理：RGB 原始 0~255 NCHW（模型内部自带 (x-127.5)/128）。
    性别映射：argmax(out[:2])==1 → male（实证，与官方文档相反）。
    """
    face = align_face(img_bgr, bbox, kps_norm)
    if face is None:
        return None
    t = face[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)  # RGB 0-255, CHW
    out = _session().run(None, {"data": t[None]})[0][0]
    return {"gender": "male" if int(np.argmax(out[:2])) == 1 else "female",
            "age": int(round(float(out[2]) * 100))}


def age_label(age):
    if age < 12: return "child"
    if age < 18: return "teen"
    if age < 60: return "adult"
    return "elder"


def load_asset_image(con, asset_id):
    """优先原图（小脸放大后糊，原图显著更准），失败退 t480 缩略图。"""
    row = con.execute(
        "SELECT absolute_path FROM media_file WHERE asset_id=? ORDER BY byte_size DESC LIMIT 1",
        (asset_id,)).fetchone()
    if row:
        img = cv2.imread(row["absolute_path"])
        if img is not None:
            return img
    stem = asset_id[6:]
    for cand in (f"{stem}_t480.jpg", f"{stem}.jpg"):
        p = THUMBS / cand
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def _parse_capture(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def main():
    ap = argparse.ArgumentParser(description="人脸性别年龄识别（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--asset", help="单资产逐脸")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--unknown", action="store_true",
                    help="只跑未知脸（person_id IS NULL，全量覆盖用）")
    ap.add_argument("--start", type=int, default=0, help="断点续跑：跳过前 N 个")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row

    if args.asset:
        img = load_asset_image(con, args.asset)
        if img is None:
            print("读不了图")
            return
        out = []
        for r in con.execute(
            "SELECT face_instance_id, bbox_json, landmarks_json FROM face_instance_v0 WHERE asset_id=?",
                (args.asset,)):
            bbox = json.loads(r["bbox_json"])
            kps = json.loads(r["landmarks_json"]) if r["landmarks_json"] else None
            p = predict(img, bbox, kps)
            if p:
                p["face_instance_id"] = r["face_instance_id"]
                p["label"] = f"{age_label(p['age'])}_{p['gender']}"
                out.append(p)
        print(json.dumps(out, ensure_ascii=False, indent=1))
        con.close()
        return

    where = "fi.quality_class='usable'"
    if args.unknown:
        where += " AND fi.person_id IS NULL"
    else:
        where += " AND fi.person_id IS NOT NULL"
    rows = con.execute(f"""
        SELECT fi.face_instance_id, fi.asset_id, fi.bbox_json, fi.landmarks_json,
               p.display_name, p.birth_date, ma.capture_time
        FROM face_instance_v0 fi
        LEFT JOIN person p USING(person_id)
        LEFT JOIN media_asset ma USING(asset_id)
        WHERE {where}
        ORDER BY fi.asset_id""").fetchall()
    if args.start:
        rows = rows[args.start:]
    if args.limit and not args.all:
        import random
        random.seed(42)
        rows = random.sample(rows, min(args.limit, len(rows)))

    out = open(args.out, "a" if args.start else "w", encoding="utf-8") if args.out else None
    cache_img, cache_id = None, None
    n = 0
    age_errs, bin_ok, bin_tot = [], 0, 0
    child_ok, child_tot = 0, 0
    t0 = time.time()
    for i, r in enumerate(rows, 1):
        if r["asset_id"] != cache_id:
            cache_img = load_asset_image(con, r["asset_id"])
            cache_id = r["asset_id"]
        if cache_img is None:
            continue
        bbox = json.loads(r["bbox_json"])
        kps = json.loads(r["landmarks_json"]) if r["landmarks_json"] else None
        p = predict(cache_img, bbox, kps)
        if not p:
            continue
        n += 1
        rec = {"face_instance_id": r["face_instance_id"],
               "display_name": r["display_name"],
               "pred_gender": p["gender"], "pred_age": p["age"],
               "pred_label": age_label(p["age"])}
        # 生日真值对照：实龄按拍摄时刻算（相册照片横跨多年，不能拿今天实龄当真值）
        if r["birth_date"] and r["capture_time"]:
            cap = _parse_capture(r["capture_time"])
            try:
                b = datetime.strptime(r["birth_date"][:10], "%Y-%m-%d")
                real = (cap - b).days / 365.25 if cap else -1
            except ValueError:
                real = -1
            if 0 <= real < 120:
                rec["real_age_at_capture"] = round(real, 1)
                age_errs.append(abs(p["age"] - real))
                pl, rl = age_label(p["age"]), age_label(real)
                bin_tot += 1
                bin_ok += pl == rl
                if real < 16:
                    child_tot += 1
                    child_ok += p["age"] < 16
        if out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        if i % 500 == 0:
            if out:
                out.flush()  # 流式落盘：被杀不丢已完成部分
            print(f"  进度 {i}/{len(rows)} speed={i/(time.time()-t0):.1f}/s", flush=True)
    if out:
        out.close()
    print(f"评估 {n} 张脸耗时 {time.time()-t0:.0f}s")
    if age_errs:
        ae = np.array(age_errs)
        print(f"拍摄时年龄误差（{len(ae)} 张）: 中位 {np.median(ae):.1f} 岁, "
              f"p90 {np.percentile(ae, 90):.1f} 岁")
    if bin_tot:
        print(f"分箱命中(child/teen/adult/elder): {bin_ok}/{bin_tot} = {bin_ok/bin_tot:.1%}")
    if child_tot:
        print(f"儿童判定(真值<16 → 预测<16): {child_ok}/{child_tot} = {child_ok/child_tot:.1%}"
              "  ← 模型已知短板，家庭成员请用 birth_date 精确值")
    con.close()


if __name__ == "__main__":
    main()
