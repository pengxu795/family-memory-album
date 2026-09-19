#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A3 人脸后端抽象（施工图·算法线 A3；2026-09-09 深夜终裁：出厂默认 AdaFace）

统一 API 的人脸 检测+识别 后端抽象，三条实现：
  adaface     YuNet 检测 + AdaFace IR-18 识别（512 维）——产品出厂默认
              （2026-09-09 深夜用户拍板：跳过全量标定，学术基准+探针 0.746 > SFace 0.542
              已足够支撑；代码 MIT，本机库已全量 AdaFace 向量。注意：WebFace4M
              权重许可仅限非商业——本地自用无风险，若未来公开分发需复核权重授权。
              详见 eval/threshold_addendum_20260909.md）
  opencv      YuNet 检测 + SFace 识别（128 维）——Apache 2.0，可经 FF_FACE_BACKEND=opencv 启用
              （曾是出厂默认：pair-F1 标定 0.542 > AdaFace 0.511，但新探针口径 0.746 反超，
              用户决策以质量优先；全量标定降级为可选优化）
  insightface SCRFD-10G 检测 + ArcFace-R50 识别（512 维）——精度高但官方预训练
              权重仅限非商业研究，只做本地可选后端：权重不进镜像、不随产品分发

许可红线（2026-09-08 评审裁定 + 2026-09-09 A4 修订）：公开产品出厂带
adaface（MIT）+ opencv（Apache 2.0）后端；insightface 后端要求用户自行下载
权重并通过 FF_FACE_BACKEND=insightface 启用。

设计约束：
- 不引入 insightface pip 包（避免 Cython 编译 + 模型 zoo 下载器），
  直接用 onnxruntime 跑本地 onnx，SCRFD 解码/ArcFace 对齐自实现
- 两后端输出结构一致：[{bbox, score, kps(5x2), _raw}]，embed() 输入同一结构
- 换后端只影响 embedding_model / detection_model 字段取值，表结构零变化

用法：
  backend = create_backend('insightface', model_dir=Path('/path/to/insightface'))
  faces = backend.detect(image)          # image: BGR ndarray
  vec = backend.embed(image, faces[0])   # 归一化向量
"""
import os
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- 通用工具

# ArcFace 112x112 标准 5 点模板（insightface 官方值）
_ARCFACE_DST = np.float32([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]])


def _similarity_transform(src5, dst5):
    """Umeyama 相似变换（2D，无缩放翻转），等价 skimage SimilarityTransform"""
    src = np.asarray(src5, dtype=np.float64)
    dst = np.asarray(dst5, dtype=np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    var_s = (sc ** 2).sum() / len(src)
    scale = np.trace(np.diag(S)) / var_s if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    M = np.zeros((2, 3), dtype=np.float64)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M


def _nms(boxes, scores, iou_thresh):
    """纯 numpy NMS，boxes=(N,4) xyxy"""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


# ---------------------------------------------------------------- OpenCV 后端

class OpenCVFaceBackend:
    """YuNet 检测 + SFace 识别。出厂默认，权重随镜像分发（OpenCV Zoo, Apache 2.0）"""
    DETECTION_MODEL = 'YuNet-2023mar'
    EMBEDDING_MODEL = 'SFace-2021dec'
    DIM = 128

    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        yunet = model_dir / 'face_detection_yunet_2023mar.onnx'
        sface = model_dir / 'face_recognition_sface_2021dec.onnx'
        if not yunet.exists() or not sface.exists():
            raise FileNotFoundError(f'OpenCV 人脸模型缺失: {model_dir}')
        self._detector = cv2.FaceDetectorYN_create(str(yunet), '', (480, 480), 0.72, 0.3, 5000)
        self._recognizer = cv2.FaceRecognizerSF_create(str(sface), '')

    def detect(self, image, det_thresh=None):
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, rows = self._detector.detect(image)
        out = []
        if rows is None:
            return out
        for r in rows:
            out.append({
                'bbox': [float(r[0]), float(r[1]), float(r[2]), float(r[3])],
                'score': float(r[14]),
                'kps': np.array([[r[4 + 2 * i], r[5 + 2 * i]] for i in range(5)], dtype=np.float32),
                '_raw': r,
            })
        return out

    def embed(self, image, det):
        aligned = self._recognizer.alignCrop(image, det['_raw'])
        emb = self._recognizer.feature(aligned).astype(np.float32).reshape(-1)
        n = float(np.linalg.norm(emb))
        return emb / n if n else emb


# ------------------------------------------------------- InsightFace 后端

class InsightFaceBackend:
    """SCRFD-10G 检测 + ArcFace-R50 识别（onnxruntime 直跑，无 insightface 包依赖）。
    许可：官方预训练权重仅限非商业学术研究——只允许本地自用，禁止随产品分发。"""
    DETECTION_MODEL = 'SCRFD-10G'
    EMBEDDING_MODEL = 'ArcFace-w600k_r50'
    DIM = 512
    INPUT_SIZE = 640

    # det_10g.onnx 输出布局：3 个 stride(8,16,32) × [score, bbox, kps]，每位置 2 anchor
    _STRIDES = (8, 16, 32)
    _NUM_ANCHORS = 2

    def __init__(self, model_dir):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        det_p = model_dir / 'det_10g.onnx'
        rec_p = model_dir / 'w600k_r50.onnx'
        if not det_p.exists() or not rec_p.exists():
            raise FileNotFoundError(
                f'InsightFace 模型缺失（需 det_10g.onnx + w600k_r50.onnx）: {model_dir}')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        self._sess_det = ort.InferenceSession(str(det_p), opts, providers=['CPUExecutionProvider'])
        self._sess_rec = ort.InferenceSession(str(rec_p), opts, providers=['CPUExecutionProvider'])
        self._det_in = self._sess_det.get_inputs()[0].name
        self._rec_in = self._sess_rec.get_inputs()[0].name

    def detect(self, image, det_thresh=0.5):
        h0, w0 = image.shape[:2]
        # letterbox 到 640×640
        scale = min(self.INPUT_SIZE / w0, self.INPUT_SIZE / h0)
        nw, nh = max(1, round(w0 * scale)), max(1, round(h0 * scale))
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (nw, nh) != (w0, h0) else image
        canvas = np.zeros((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=np.uint8)
        canvas[:nh, :nw] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, None, (127.5, 127.5, 127.5), swapRB=True)
        outputs = self._sess_det.run(None, {self._det_in: blob})
        # det_10g 输出按【类型】分组：scores×3(stride 8/16/32) → bboxes×3 → kps×3
        # 每 stride 展平为 (fm*fm*2,)，H-major → W → anchor
        scores_all, boxes_all, kps_all = outputs[0:3], outputs[3:6], outputs[6:9]

        all_boxes, all_scores, all_kps = [], [], []
        for si, stride in enumerate(self._STRIDES):
            fm = self.INPUT_SIZE // stride
            a = self._NUM_ANCHORS
            scores_flat = scores_all[si].reshape(-1)
            pos = np.where(scores_flat >= det_thresh)[0]
            if len(pos) == 0:
                continue
            # 展平序 idx = (y*fm + x)*a + anchor → 反解 (x, y)
            ys = pos // (fm * a)
            xs = (pos // a) % fm
            centers = np.stack([xs, ys], axis=1).astype(np.float32) * stride          # (M,2)
            s = scores_flat[pos]                                                      # (M,)
            # delta 为无符号距离 (l,t,r,b)：x1=acx-l, y1=acy-t, x2=acx+r, y2=acy+b
            d = boxes_all[si][pos] * np.float32(stride)
            b = np.stack([centers[:, 0] - d[:, 0], centers[:, 1] - d[:, 1],
                          centers[:, 0] + d[:, 2], centers[:, 1] + d[:, 3]], axis=1)
            k = kps_all[si][pos].reshape(-1, 5, 2).astype(np.float32) * np.float32(stride) + centers[:, None, :]
            all_boxes.append(b)
            all_scores.append(s)
            all_kps.append(k)
        if not all_boxes:
            return []
        boxes = np.concatenate(all_boxes)
        scores = np.concatenate(all_scores)
        kps = np.concatenate(all_kps)
        # 映射回原图坐标
        boxes = boxes / scale
        kps = kps / scale
        keep = _nms(boxes, scores, 0.4)
        out = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            out.append({
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': float(scores[i]),
                'kps': kps[i].astype(np.float32),
                '_raw': None,
            })
        return out

    def embed(self, image, det):
        M = _similarity_transform(det['kps'], _ARCFACE_DST)
        aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, None, (127.5, 127.5, 127.5), swapRB=True)
        emb = self._sess_rec.run(None, {self._rec_in: blob})[0].reshape(-1)
        n = float(np.linalg.norm(emb))
        return emb / n if n else emb


# ------------------------------------------------------- AdaFace 后端

class AdaFaceBackend:
    """YuNet 检测（复用 opencv 后端的检测器）+ AdaFace IR-18 识别（512 维）。
    MIT 许可（权重与代码均可商用）——产品出厂可安全分发，低质量/小脸场景
    论文指标优于 ArcFace。权重：models/adaface/adaface_ir_18.onnx（本地目录）。

    预处理对齐 AdaFace 官方仓库（mk-minchul/AdaFace）推理配置：
    112×112 对齐（ArcFace 标准 5 点模板）→ RGB → (x-127.5)/127.5。"""
    DETECTION_MODEL = 'YuNet-2023mar'
    EMBEDDING_MODEL = 'AdaFace-IR18-WebFace4M'
    DIM = 512

    def __init__(self, model_dir):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        yunet = model_dir / 'face_detection_yunet_2023mar.onnx'
        rec_p = model_dir / 'adaface_ir_18.onnx'
        if not yunet.exists() or not rec_p.exists():
            raise FileNotFoundError(
                f'AdaFace 后端模型缺失（需 yunet onnx + adaface_ir_18.onnx）: {model_dir}')
        self._detector = cv2.FaceDetectorYN_create(str(yunet), '', (480, 480), 0.72, 0.3, 5000)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
        self._sess_rec = ort.InferenceSession(str(rec_p), opts, providers=['CPUExecutionProvider'])
        self._rec_in = self._sess_rec.get_inputs()[0].name

    def detect(self, image, det_thresh=None):
        h, w = image.shape[:2]
        self._detector.setInputSize((w, h))
        _, rows = self._detector.detect(image)
        out = []
        if rows is None:
            return out
        for r in rows:
            out.append({
                'bbox': [float(r[0]), float(r[1]), float(r[2]), float(r[3])],
                'score': float(r[14]),
                'kps': np.array([[r[4 + 2 * i], r[5 + 2 * i]] for i in range(5)], dtype=np.float32),
                '_raw': r,
            })
        return out

    def embed(self, image, det):
        M = _similarity_transform(det['kps'], _ARCFACE_DST)
        aligned = cv2.warpAffine(image, M, (112, 112), borderValue=0)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, None, (127.5, 127.5, 127.5), swapRB=True)
        emb = self._sess_rec.run(None, {self._rec_in: blob})[0].reshape(-1)
        n = float(np.linalg.norm(emb))
        return emb / n if n else emb


# ---------------------------------------------------------------- 工厂

_BACKENDS = {'opencv': OpenCVFaceBackend, 'insightface': InsightFaceBackend,
             'adaface': AdaFaceBackend}


def create_backend(name=None, model_dir=None):
    """按名字建后端。出厂默认 adaface（2026-09-09 深夜终裁，见模块头注释）；
    FF_FACE_BACKEND 环境变量可覆盖（如 opencv/insightface）。
    insightface 的 model_dir 必须含 det_10g.onnx + w600k_r50.onnx。"""
    name = (name or os.environ.get('FF_FACE_BACKEND') or 'adaface').strip().lower()
    if name not in _BACKENDS:
        raise ValueError(f'未知人脸后端: {name}（可选: {", ".join(_BACKENDS)}）')
    return _BACKENDS[name](model_dir)


def embedding_model_of(name):
    """后端名 → 嵌入模型名（不实例化，读类属性）。供查询按 model_name 过滤。"""
    return _BACKENDS[(name or '').strip().lower()].EMBEDDING_MODEL
