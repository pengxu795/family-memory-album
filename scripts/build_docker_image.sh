#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 家庭回忆相册 — NAS 预构建镜像打包（开发机执行，产物给 NAS 用户）
# 产出:
#   releases/family-memory-album-<ver>-nas-amd64.tar.gz   镜像离线包（docker load）
#   releases/family-memory-album-nas-deploy-<ver>.tar.gz  一键部署包（脚本+镜像，小白全离线）
# 用法:
#   scripts/build_docker_image.sh            # 默认 0.5
#   scripts/build_docker_image.sh 0.5.1
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"     # mvp/
VER="${1:-0.5}"
IMAGE="family-memory-album:${VER}"
MODELS_SRC="$HERE/../models"                  # 仓库根 models/

echo "==> [1/4] 准备 baked_models（核心模型内置：adaface + 检测/属性 onnx，≈103M）"
rm -rf "$HERE/baked_models" && mkdir -p "$HERE/baked_models/adaface"
cp -R "$MODELS_SRC/adaface/." "$HERE/baked_models/adaface/"
for f in genderage.onnx 2d106det.onnx u2netp.onnx; do
  [ -f "$MODELS_SRC/$f" ] && cp "$MODELS_SRC/$f" "$HERE/baked_models/"
done
du -sh "$HERE/baked_models"

echo "==> [2/4] 构建 linux/amd64 镜像 $IMAGE"
docker build --platform linux/amd64 -t "$IMAGE" "$HERE"

echo "==> [3/4] 导出离线镜像包"
mkdir -p "$HERE/releases"
docker save "$IMAGE" | gzip > "$HERE/releases/family-memory-album-${VER}-nas-amd64.tar.gz"

echo "==> [4/4] 打一键部署包（脚本 + compose 模板 + 镜像）"
STAGE=$(mktemp -d)
cp "$HERE/scripts/install_nas.sh" "$STAGE/"
chmod +x "$STAGE/install_nas.sh"
cp "$HERE/releases/family-memory-album-${VER}-nas-amd64.tar.gz" "$STAGE/"
tar -czf "$HERE/releases/family-memory-album-nas-deploy-${VER}.tar.gz" -C "$STAGE" .
rm -rf "$STAGE"

ls -lh "$HERE/releases/" | grep -v "\.dmg"
echo "==> 完成。镜像 $IMAGE；离线包与一键部署包在 releases/"
