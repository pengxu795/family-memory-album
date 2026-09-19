#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""新功能①-⑤ 单元测试（合成图，不依赖 DB / 不写库）

覆盖：
  ① blur_detector   —— 清晰/模糊/软聚焦三分类 + top-25% 分块逻辑
  ② similar_dedup   —— dHash 稳定性 + 汉明距离正确性
  ③ aesthetic_scorer —— 打分确定性 + 数值有限（模型加载慢，仅 2 例）
  ④ junk_scanner    —— QR 内容分类 / 真实 QR 检出 / 截图视觉嫌疑
  ⑤ smart_crop      —— 四级策略路径 + 窗口约束（比例/越界）
  (附) genderage    —— 对齐输出形状 + 预测接口 smoke

跑法：python3 tests/test_new_features.py   （无 pytest 依赖，直接断言计数）
"""
import sys
import numpy as np
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL = 0, 0
SKIP = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        FAILURES.append((name, detail))
        print(f"  ✗ {name}  {detail}")


def skip(msg):
    global SKIP
    SKIP += 1
    print(f"  ⊘ SKIP: {msg}")


def sharp_image(w=480, h=360, seed=0):
    """高频棋盘+噪声 = 清晰图。"""
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    bs = 12
    for y in range(0, h, bs):
        for x in range(0, w, bs):
            c = 255 if (x // bs + y // bs) % 2 else 30
            img[y:y+bs, x:x+bs] = c
    img = img + rng.integers(-20, 20, img.shape).astype(np.int16)
    return np.clip(img, 0, 255).astype(np.uint8)


def blurry_image(w=480, h=360, seed=0):
    """清晰图重度高斯模糊 = 模糊图。"""
    return cv2.GaussianBlur(sharp_image(w, h, seed), (0, 0), 9)


# ---------------------------------------------------------------- ① 模糊检测
def test_blur():
    import blur_detector as bd
    print("[① blur_detector]")
    sharp = sharp_image()
    blurry = blurry_image()
    soft = cv2.GaussianBlur(sharp_image(seed=1), (0, 0), 2.2)

    rs, rb = bd.classify_image(sharp), bd.classify_image(blurry)
    check("清晰图判 sharp", rs["label"] == "sharp", str(rs))
    check("模糊图判 blurry", rb["label"] == "blurry", str(rb))
    check("清晰分数显著高于模糊", rs["sharp_score"] > rb["sharp_score"] * 5,
          f"{rs['sharp_score']} vs {rb['sharp_score']}")
    rsoft = bd.classify_image(soft)
    check("轻模糊分数落在 sharp 与 blurry 之间",
          rsoft["sharp_score"] < rs["sharp_score"] and rsoft["sharp_score"] > rb["sharp_score"],
          str(rsoft))
    # 纯色图（零方差）→ blurry
    flat = bd.classify_image(np.full((360, 480, 3), 128, np.uint8))
    check("纯色图判 blurry", flat["label"] == "blurry" and flat["sharp_score"] == 0.0, str(flat))
    # lap_block_scores 升序 + 数量 = 分块数
    scores = bd.lap_block_scores(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY))
    check("分块分数升序", scores == sorted(scores))
    check("分块数量合理(4x4 网格)", 9 <= len(scores) <= 25, str(len(scores)))


# ---------------------------------------------------------------- ② 相似去重
def test_dedup():
    import similar_dedup as sd
    print("[② similar_dedup]")
    img = smooth_image()
    h1 = sd.dhash64(img)
    h2 = sd.dhash64(img.copy())
    check("同图 dHash 完全相同", h1 == h2)
    check("dHash 是 64bit", 0 <= h1 < 2**64)

    bright = np.clip(img.astype(np.int16) + 60, 0, 255).astype(np.uint8)
    d_bright = int(sd.hamming_np(np.array([h1], dtype=np.uint64),
                                 np.array([sd.dhash64(bright)], dtype=np.uint64))[0])
    check("亮度平移汉明距离小(≤8)", d_bright <= 8, str(d_bright))

    other = 180 - img  # 反相：图案完全不同
    d_other = int(sd.hamming_np(np.array([h1], dtype=np.uint64),
                                np.array([sd.dhash64(other)], dtype=np.uint64))[0])
    check("不同图案汉明距离大(>16)", d_other > 16, str(d_other))

    # hamming_np 批量正确性：与逐位 bin count 一致
    a = np.array([h1, sd.dhash64(other)], dtype=np.uint64)
    b = np.array([h2, sd.dhash64(bright)], dtype=np.uint64)
    got = sd.hamming_np(a, b)
    exp = [bin(int(x ^ y)).count("1") for x, y in zip(a, b)]
    check("批量汉明距离与逐位一致", got.tolist() == exp, f"{got.tolist()} vs {exp}")


# ---------------------------------------------------------------- ③ 美学评分
def test_aesthetic():
    print("[③ aesthetic_scorer]（模型加载 ~350MB，稍等）")
    try:
        import torch  # noqa: F401
        import aesthetic_scorer as ae
    except ImportError as e:
        skip(f"本解释器无 torch（用 /usr/bin/python3 跑 aesthetic 段）: {e}")
        return
    img = sharp_image(seed=3)
    s1 = ae.score_image(img)
    s2 = ae.score_image(img)
    check("返回有限浮点", isinstance(s1, float) and np.isfinite(s1), str(s1))
    check("同图打分确定", abs(s1 - s2) < 1e-4, f"{s1} vs {s2}")
    noise = (np.random.default_rng(9).integers(0, 256, (360, 480, 3))).astype(np.uint8)
    sn = ae.score_image(noise)
    check("噪声图也能打出有限分", np.isfinite(sn), str(sn))


# ---------------------------------------------------------------- ② 相似去重（平滑图版）
def smooth_image(w=480, h=360):
    """平滑渐变+圆形（dHash 的设计场景：平滑自然图，非高频棋盘）。"""
    img = np.zeros((h, w, 3), np.uint8)
    for x in range(w):
        img[:, x] = int(x / w * 180)
    cv2.circle(img, (300, 180), 90, (235, 235, 235), -1)
    return cv2.GaussianBlur(img, (0, 0), 5)


# ---------------------------------------------------------------- ④ 截图/二维码
def _make_qr(text):
    import qrcode
    qr = qrcode.QRCode(border=2, box_size=6)
    qr.add_data(text)
    qr.make()
    return np.array(qr.get_matrix(), dtype=np.uint8)


def test_junk():
    import junk_scanner as js
    print("[④ junk_scanner]")
    check("微信收款码分类", js.classify_qr_content("wxp://f2f0909Gjf2") == "payment_qrcode_wechat")
    check("支付宝码分类", js.classify_qr_content("https://qr.alipay.com/bax123") == "payment_qrcode_alipay")
    check("普通 URL 归 qrcode", js.classify_qr_content("https://example.com/x") == "qrcode")

    # 真实 QR 生成 → 白底摆放 → 检出并分类
    mat = _make_qr("wxp://f2f0909Gjf2")
    qr_img = (1 - mat) * 255
    canvas = np.full((480, 480, 3), 255, np.uint8)
    q160 = cv2.resize(qr_img.astype(np.uint8), (160, 160), interpolation=cv2.INTER_NEAREST)
    canvas[100:260, 160:320] = cv2.cvtColor(q160, cv2.COLOR_GRAY2BGR)
    kind, data, pts = js.detect_qr(canvas)
    check("真实微信码检出", kind == "payment_qrcode_wechat" and data == "wxp://f2f0909Gjf2",
          f"kind={kind} data={data}")
    no_qr = np.full((480, 480, 3), 200, np.uint8)
    kind2, _, _ = js.detect_qr(no_qr)
    check("无 QR 返回 None", kind2 is None, str(kind2))

    # 截图视觉嫌疑
    shot = np.full((800, 400, 3), 245, np.uint8)          # 全面屏纯色"截图"
    shot[:30] = 230                                        # 顶部状态栏
    cv2.rectangle(shot, (20, 60), (380, 200), (180, 180, 180), -1)
    check("截图嫌疑判定 True", js.screenshot_suspect(shot) is True)
    photo = np.zeros((800, 400, 3), np.uint8)              # 自然照片：渐变+噪声
    for y in range(800):
        photo[y] = y % 256
    noise = np.random.default_rng(5).integers(0, 40, photo.shape)
    photo = np.clip(photo.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    check("自然照片判 False", js.screenshot_suspect(photo) is False)
    wide = np.zeros((400, 800, 3), np.uint8)               # 横图非全面屏比例 + 边缘有纹理
    noise = np.random.default_rng(6).integers(0, 130, wide.shape)
    wide = np.clip(wide.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    check("非全面屏纵横比判 False", js.screenshot_suspect(wide) is False)


# ---------------------------------------------------------------- ⑤ 智能裁切
def test_smart_crop():
    import smart_crop as sc
    print("[⑤ smart_crop]（U²-Net/YuNet 模型加载，稍等）")

    def assert_window(name, r, W, H):
        ok = (0 <= r["x"] <= 1 and 0 <= r["y"] <= 1
              and r["w"] > 0 and r["h"] > 0
              and r["x"] + r["w"] <= 1.001 and r["y"] + r["h"] <= 1.001)
        ar = (r["w"] * W) / (r["h"] * H)
        ok = ok and abs(ar - 4/3) < 0.02
        check(name, ok, str(r))

    # 1) 纯平图 → center
    flat = np.full((480, 640, 3), 100, np.uint8)
    r = sc.crop_params(flat)
    check("平图走 center", r["method"] == "center", str(r))
    assert_window("center 窗 4:3 且在界内", r, 640, 480)

    # 2) 显著亮斑 → saliency（断言"窗包含主体"，全宽窗时窗心必然回中）
    img = np.full((480, 640, 3), 60, np.uint8)
    cv2.ellipse(img, (420, 240), (70, 90), 0, 0, 360, (230, 230, 230), -1)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    r = sc.crop_params(img)
    check("亮斑走 saliency", r["method"] == "saliency", str(r))
    blob_in = (r["x"] * 640 <= 420 <= (r["x"] + r["w"]) * 640
               and r["y"] * 480 <= 240 <= (r["y"] + r["h"]) * 480)
    check("saliency 窗包含主体", blob_in,
          f"窗 x∈[{r['x']*640:.0f},{(r['x']+r['w'])*640:.0f}] y∈[{r['y']*480:.0f},{(r['y']+r['h'])*480:.0f}]")

    # 3) 地平线 → horizon
    sky = np.full((480, 640, 3), 210, np.uint8)
    sky[300:] = 80                                       # 上天下海分界在 y=300（下半 62.5%）
    r = sc.crop_params(sky)
    check("水平分界走 horizon", r["method"] == "horizon", str(r))
    assert_window("horizon 窗约束", r, 640, 480)

    # 4) _place_window 纯函数：W=800,H=400 → win=min(800, 400*4/3)=533×400
    x, y, ww, hh = sc._place_window(800, 400, 700, 50)
    check("_place_window 窗口尺寸=4:3 最大内接", abs(ww - 400*4/3) < 1 and abs(hh - 400) < 1,
          f"{ww}x{hh}")
    check("_place_window 越界被 clamp", x >= 0 and y >= 0 and x + ww <= 800 and y + hh <= 400,
          f"x={x} y={y}")

    # 5) 竖图特写兜底：极窄竖图平图 → 窗仍合法
    r = sc.crop_params(np.full((900, 300, 3), 120, np.uint8))
    assert_window("竖图平图窗合法(4:3)", r, 300, 900)


# ---------------------------------------------------------------- 附 genderage smoke
def test_genderage_smoke():
    import genderage as ga
    print("[附 genderage smoke]")
    img = sharp_image(seed=11)
    bbox = {"x": 0.2, "y": 0.2, "w": 0.6, "h": 0.6}
    face = ga.align_face(img, bbox, None)
    check("对齐输出 96×96", face is not None and face.shape[:2] == (96, 96),
          str(None if face is None else face.shape))
    kps = [{"x": 0.4, "y": 0.4}, {"x": 0.6, "y": 0.4}, {"x": 0.5, "y": 0.55},
           {"x": 0.42, "y": 0.65}, {"x": 0.58, "y": 0.65}]
    face2 = ga.align_face(img, bbox, kps)
    check("kps 对齐输出 96×96", face2 is not None and face2.shape[:2] == (96, 96),
          str(None if face2 is None else face2.shape))
    p = ga.predict(img, bbox, None)
    check("predict 返回 gender/age", p is not None and p["gender"] in ("male", "female")
          and isinstance(p["age"], int), str(p))
    check("age_label 分箱", ga.age_label(5) == "child" and ga.age_label(15) == "teen"
          and ga.age_label(40) == "adult" and ga.age_label(70) == "elder")


# ---------------------------------------------------------------- ⑩ OCR
def test_ocr():
    print("[⑩ ocr_text]")
    try:
        from rapidocr_onnxruntime import RapidOCR  # noqa: F401
    except ImportError as e:
        skip(f"未装 rapidocr_onnxruntime: {e}")
        return
    import ocr_text as ot
    ocr = ot.load_ocr()
    # 英文大字（cv2 putText 无中文字体，英文即可验证管线）
    img = np.full((360, 480, 3), 30, np.uint8)
    cv2.putText(img, "HELLO 2026", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 8)
    r = ot.ocr_image(ocr, img)
    check("大字图识别出文字", r["n_regions"] >= 1 and "2026" in r["text"].replace(" ", ""),
          str(r))
    check("mean_conf 有限", isinstance(r["mean_conf"], float) and 0 <= r["mean_conf"] <= 1,
          str(r))
    flat = ot.ocr_image(ocr, np.full((360, 480, 3), 128, np.uint8))
    check("纯色图无文字", flat["n_regions"] == 0 and flat["text"] == "", str(flat))
    # 截断保护
    check("text 截断 ≤600", len(r["text"]) <= 600)


# ---------------------------------------------------------------- ⑪ 表情
def test_expr():
    print("[⑪ face_expr]（纯函数段）")
    import face_expr as fe
    # 106×2 合成关键点：按实证索引布局构造睁眼/闭眼/微笑
    def make(eye_gap, smile):
        p = np.zeros((106, 2), np.float32)
        p[:, 0] = np.linspace(30, 160, 106)          # x 递增
        p[:, 1] = 60.0                                # 默认一条线
        # 右眼 33-42: x 固定跨度 28（60-88），上下睑分开 eye_gap
        p[33:38, 0] = np.linspace(60, 88, 5)
        p[38:43, 0] = np.linspace(60, 88, 5)
        p[33:38, 1] = 90 - eye_gap / 2                # 上睑 5 点
        p[38:43, 1] = 90 + eye_gap / 2                # 下睑 5 点
        # 左眼 88-96: x 跨度 28（110-138）
        p[88:92, 0] = np.linspace(110, 138, 4)
        p[92:96, 0] = np.linspace(110, 138, 4)
        p[88:92, 1] = 90 - eye_gap / 2
        p[92:96, 1] = 90 + eye_gap / 2
        # 嘴角 52/58：无表情窄嘴(宽16)，微笑宽嘴(宽40)；下唇中心 55-57
        if smile:
            p[52] = [70, 126]; p[58] = [110, 126]
        else:
            p[52] = [82, 130]; p[58] = [98, 130]
        p[55:58, 1] = 132
        return p
    m_open = fe.metrics_from_pred(make(10, False))
    check("睁眼判定", m_open is not None and m_open["eyes_open"] is True, str(m_open))
    m_closed = fe.metrics_from_pred(make(1, False))
    check("闭眼判定", m_closed is not None and m_closed["eyes_open"] is False, str(m_closed))
    m_smile = fe.metrics_from_pred(make(10, True))
    check("微笑判定", m_smile is not None and m_smile["is_smiling"] is True, str(m_smile))
    m_flat = fe.metrics_from_pred(make(10, False))
    check("无表情判定", m_flat is not None and m_flat["is_smiling"] is False, str(m_flat))
    # 坏 bbox 护栏：眼比 >1.0 → None
    bad = make(60, False)                             # 眼 gap 60 → 比值 >1
    check("眼比>1 丢弃", fe.metrics_from_pred(bad) is None)
    # align_bbox15：bbox 中心（x+w/2）映射到帧中心；side=max(w,h)*1.5
    img = np.zeros((960, 720, 3), np.uint8)           # H=960 W=720
    bbox = {"x": 0.5, "y": 0.5, "w": 0.2, "h": 0.2}
    M = fe.align_bbox15(img, bbox)
    cx, cy = 0.6 * 720, 0.6 * 960                     # bbox 中心 (432, 576)
    c = M @ np.array([cx, cy, 1.0])
    check("bbox15 中心映射到(96,96)", abs(c[0]-96) < 1 and abs(c[1]-96) < 1, str(c))
    scale = np.sqrt(M[0, 0]**2 + M[0, 1]**2)
    check("bbox15 缩放=192/(边*1.5)", abs(scale - 192/(288)) < 0.01, str(scale))


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    sections = {"blur": test_blur, "dedup": test_dedup, "aesthetic": test_aesthetic,
                "junk": test_junk, "crop": test_smart_crop, "genderage": test_genderage_smoke,
                "ocr": test_ocr, "expr": test_expr}
    for k, fn in sections.items():
        if only and k != only:
            continue
        fn()
    print(f"\n===== 结果: {PASS} 通过 / {FAIL} 失败 / {SKIP} 跳过 =====")
    if FAILURES:
        for name, detail in FAILURES:
            print(f"  失败: {name} {detail}")
        sys.exit(1)
