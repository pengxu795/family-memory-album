#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""截图/二维码/收款码自动识别清理（新功能④：junk_scanner.py）

分工（2026-09-09 设计，不重复已有基建）：
- asset_filter_v0 已覆盖截图清理（SCREENSHOT_FILENAME/PATH、VISION_SCREENSHOT
  等 10 类 ~3400 项），本模块**不重复**文件名/路径规则。
- 新增能力 1 **二维码/收款码识别**（现有过滤空白）：
  cv2.QRCodeDetector 多尺度检出（缩略图 + 2x 上采样，抓拍屏小码），
  解码内容分类：wxp:// → 微信收款码；alipay/QR.ALIPAY → 支付宝收款码；
  其他可解码 → qrcode；检出但解不出 → qrcode_undecoded。
  产品决策（2026-09-09）：检出即过滤——qrcode / qrcode_undecoded 一律写
  过滤表隐藏。
  **2026-09-16 推翻（见 backfill_quality.JUNK_REASON）**：实测 undecoded 分支
  无判别力——cv2 对海浪/砂石/树叶等纹理确定性误检（抽 60 张被隐藏照片重检，
  60/60 依旧「解不出」；45% 有人脸）。现仅**解码成功**的 QR 才有过滤资格
  （qrcode / payment_*），undecoded 只作信息输出，不再进入 JUNK_REASON 映射。
- 新增能力 2 **截图视觉补捞**（文件名线索缺失的漏网截图）：
  纵横比（≥1.9 或 ≤0.55 的手机全面屏比）+ 四边 3px 纯色边 + 顶部 6% 行
  低方差（状态栏）三条件同中 → screenshot_suspect（低置信，供人工复核）。

不写库：结果 JSONL；写 asset_filter_v0 / 清理动作留 A4 联调后由用户确认。
用法：
  python3 junk_scanner.py --all --out /tmp/junk.jsonl
  python3 junk_scanner.py --limit 300
  python3 junk_scanner.py --image path.jpg
"""
import argparse, json, sqlite3, time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
THUMBS = ROOT / "data" / "thumbs_mvp"

# 已有过滤覆盖的资产不再报 screenshot_suspect（QR 检测仍全量做——过滤表不管 QR）
EXISTING_FILTER_REASONS = None  # 懒加载


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
    scale = 480 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def classify_qr_content(data):
    """解码内容 → 类别。微信收款码 wxp://f2f…；支付宝 https://qr.alipay.com/…"""
    d = data.strip()
    dl = d.lower()
    if dl.startswith("wxp://") or "wx.tenpay.com" in dl:
        return "payment_qrcode_wechat"
    if "alipay" in dl or dl.startswith("https://qr.alipay"):
        return "payment_qrcode_alipay"
    return "qrcode"


def detect_qr(img_bgr):
    """多尺度 QR 检出。返回 (kind, content, pts)；无 QR 返回 (None, None, None)。"""
    det = cv2.QRCodeDetector()
    for scale_img in (img_bgr, cv2.resize(img_bgr, None, fx=2, fy=2,
                                          interpolation=cv2.INTER_CUBIC)):
        try:
            data, pts, _ = det.detectAndDecode(scale_img)
        except cv2.error:
            continue
        if data:
            return classify_qr_content(data), data, pts
        if pts is not None:  # 检出码格但解不出（小/糊）
            return "qrcode_undecoded", None, pts
    return None, None, None


def detect_qr_hi(resolver, img_bgr):
    """两阶段精扫：缩略图 undecoded 时，用 pts 区域映射回原片 crop 4x 重试。
    resolver(asset_id) → 原片图（调用方提供，避免不必要的原片 IO）。"""
    kind, content, pts = detect_qr(img_bgr)
    if kind not in ("qrcode_undecoded",) or pts is None:
        return kind, content
    big = resolver()
    if big is None:
        return kind, content
    h0, w0 = img_bgr.shape[:2]
    h1, w1 = big.shape[:2]
    sx, sy = w1 / w0, h1 / h0
    pts_arr = np.array(pts).reshape(-1, 2)
    x0 = max(0, int(pts_arr[:, 0].min() * sx) - 40)
    x1 = min(w1, int(pts_arr[:, 0].max() * sx) + 40)
    y0 = max(0, int(pts_arr[:, 1].min() * sy) - 40)
    y1 = min(h1, int(pts_arr[:, 1].max() * sy) + 40)
    crop = big[y0:y1, x0:x1]
    if crop.size == 0:
        return kind, content
    crop = cv2.resize(crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    return detect_qr(crop)[:2]


def screenshot_suspect(img_bgr):
    """视觉截图嫌疑：全面屏纵横比 + 四边纯色 + 顶部低方差（状态栏）。"""
    h, w = img_bgr.shape[:2]
    ar = max(h, w) / min(h, w)
    if not (ar >= 1.9 or ar <= 0.55):
        return False
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # 四边 3px 均值近纯色且标准差小（截图边缘干净；照片边缘有内容纹理）
    edges = [g[:3, :], g[-3:, :], g[:, :3], g[:, -3:]]
    for e in edges:
        if e.std() > 12:  # 边缘有纹理 → 大概率不是截图
            return False
    # 顶部 6% 行方差低（状态栏/标题栏均匀）
    top = g[: max(8, int(h * 0.06)), :]
    return bool(top.std() < 18)


def main():
    ap = argparse.ArgumentParser(description="截图/二维码/收款码识别（独立模块，不写库）")
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--image", help="单图模式")
    ap.add_argument("--out", default=str(ROOT / "data" / "junk_scan.jsonl"))
    ap.add_argument("--deep", action="store_true",
                    help="[非默认，仅排障] 对 undecoded 读原片 crop 4x 重试解码（NAS IO ~3s/码）。"
                         "2026-09-09 产品决策：QR 检出即过滤（undecoded 直接隐藏），"
                         "解码分类不影响任何显示结果，已从默认流程移除；保留 flag 仅供人工排查")
    ap.add_argument("--start", type=int, default=0,
                    help="断点续跑：跳过前 N 个资产（配合 append 模式）")
    ap.add_argument("--read-timeout", type=int, default=20,
                    help="deep 模式单张原片读图超时秒数（NAS 卡死护栏）")
    ap.add_argument("--ids-file", default="",
                    help="只扫清单里的 asset_id（一行一个，补扫用）")
    args = ap.parse_args()

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            print(f"读不了: {args.image}")
            return
        kind, content, _pts = detect_qr(img)
        r = {"kind": kind or ("screenshot_suspect" if screenshot_suspect(img) else "normal"),
             "content": content}
        print(json.dumps(r, ensure_ascii=False))
        return

    con = sqlite3.connect(args.db, timeout=60)
    con.row_factory = sqlite3.Row
    global EXISTING_FILTER_REASONS
    EXISTING_FILTER_REASONS = {r[0] for r in con.execute(
        "SELECT DISTINCT asset_id FROM asset_filter_v0")}
    ids = [r[0] for r in con.execute(
        "SELECT asset_id FROM media_asset WHERE media_type='photo' ORDER BY asset_id")]
    if args.ids_file:
        want = {l.strip() for l in open(args.ids_file, encoding="utf-8") if l.strip()}
        ids = [a for a in ids if a in want]
    if args.limit:
        import random
        random.seed(42)
        ids = random.sample(ids, min(args.limit, len(ids)))

    if args.start:
        ids = ids[args.start:]
    # 断点续跑用追加模式；每条 hit 立即 flush——后台 2h 墙钟上限被杀时进度不丢
    out = open(args.out, "a" if args.start else "w", encoding="utf-8")
    stats = {}
    t0 = time.time()
    scanned = 0

    def _resolver_for(aid):
        def resolver():
            row = con.execute(
                "SELECT absolute_path FROM media_file WHERE asset_id=? "
                "ORDER BY byte_size DESC LIMIT 1", (aid,)).fetchone()
            if not row:
                return None
            # NAS 读图护栏：SMB 卡死文件 20s 强断跳过（否则整个扫描挂死）
            import signal
            def _hang(signum, frame):
                raise TimeoutError(f"read hang: {row['absolute_path']}")
            signal.signal(signal.SIGALRM, _hang)
            signal.alarm(args.read_timeout)
            try:
                return cv2.imread(row["absolute_path"])
            finally:
                signal.alarm(0)
        return resolver

    for i, aid in enumerate(ids, 1):
        img = load_asset_image(con, aid)
        if img is None:
            continue
        scanned += 1
        kind, content = detect_qr(img)[:2]
        if args.deep and kind == "qrcode_undecoded":
            try:
                kind, content = detect_qr_hi(_resolver_for(aid), img)
            except TimeoutError as e:
                print(f"  [跳过卡死] {aid}: {e}", flush=True)
                continue
            except cv2.error:
                continue
        if kind is None and aid not in EXISTING_FILTER_REASONS \
                and screenshot_suspect(img):
            kind = "screenshot_suspect"
        if kind:
            stats[kind] = stats.get(kind, 0) + 1
            out.write(json.dumps({"asset_id": aid, "kind": kind,
                                  "content": content}, ensure_ascii=False) + "\n")
            out.flush()
        if i % 500 == 0:
            print(f"  进度 {i}/{len(ids)} speed={i/(time.time()-t0):.1f}/s "
                  f"断点={args.start + i}", flush=True)
    out.close()
    print(f"扫描 {scanned} 张耗时 {time.time()-t0:.0f}s → {args.out}")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    con.close()


if __name__ == "__main__":
    main()
