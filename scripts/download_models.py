#!/usr/bin/env python3
"""下载人脸/抠图模型权重（Apache-2.0，可随开源产品分发）。

用法：
    python3 scripts/download_models.py            # 下载合规三件套（YuNet/SFace/U2Netp）
    python3 scripts/download_models.py --adaface  # 额外下载 AdaFace 权重（⚠ 非商业研究许可，
                                                  #   仅限本地自用，禁止再分发/商用，见 docs/MODEL_LICENSES.md）

Docker 构建前先跑本脚本（Dockerfile 会 COPY baked_models/ 进镜像）。
FF_FACE_BACKEND=opencv 用 YuNet+SFace（默认合规组合）；
FF_FACE_BACKEND=adaface 需要 --adaface 下载的权重（128 维→512 维向量不通用，换后端需重建人脸库）。
"""
import os
import sys
import urllib.request
from pathlib import Path

MODELS = {
    # OpenCV Zoo（Apache-2.0）：Git LFS 文件，raw 链接会 302 到 media.githubusercontent.com
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "人脸检测（YuNet）"),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "人脸识别（SFace，出厂默认后端）"),
    # U²-Netp（Apache-2.0，rembg 同源导出）
    "u2netp.onnx": (
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2netp.onnx",
        "显著性裁切（U²-Netp）"),
}

ADAFACE = {
    "adaface_ir_18.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_adaface/adaface_ir_18.onnx",
        "人脸识别（AdaFace）"),
}


def fetch(url, dest):
    print(f"下载 {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "family-memory-album/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    if dest.exists():
        dest.unlink()
    tmp.rename(dest)


def main():
    root = Path(__file__).resolve().parent.parent / "baked_models"
    root.mkdir(parents=True, exist_ok=True)
    want = dict(MODELS)
    if "--adaface" in sys.argv:
        print("⚠ AdaFace 权重基于 WebFace4M（非商业研究许可）：仅限本地自用，禁止再分发/商用。\n")
        want.update(ADAFACE)
    for name, (url, desc) in want.items():
        dest = root / name
        if dest.exists() and dest.stat().st_size > 1024:
            print(f"已存在，跳过：{name}（{desc}）")
            continue
        fetch(url, dest)
        print(f"完成：{name} ← {desc}（{dest.stat().st_size // 1024} KB）")
    print("\n全部就绪。Docker 构建示例：\n"
          "  docker build -t family-memory-album .\n"
          "  docker run -d -p 8788:8788 -v /your/photos:/data family-memory-album\n"
          "人脸后端选择（compose 环境变量 FF_FACE_BACKEND）：opencv（默认，合规）/ adaface（需 --adaface）")


if __name__ == "__main__":
    main()
