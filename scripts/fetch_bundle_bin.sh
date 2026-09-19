#!/usr/bin/env bash
# ============================================================
# 下载 bundle_bin/ 内置二进制（构建安装包前的准备步骤，幂等）
#   - ffmpeg 9.0.1  evermeet.cx 静态构建（tessus，macOS x86_64，Rosetta 兼容）
#   - exiftool 13.55 GitHub 官方可移植发行版（脚本 + lib/，依赖系统 /usr/bin/perl）
# ffmpeg 25M 二进制不入 git（.gitignore），exiftool 已入库；本脚本用于重取/更新。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bundle_bin

FFMPEG_VER="${FFMPEG_VER:-9.0.1}"
EXIFTOOL_VER="${EXIFTOOL_VER:-13.55}"

# ---- ffmpeg（evermeet.cx）----
if [[ -x bundle_bin/ffmpeg ]] && bundle_bin/ffmpeg -version 2>/dev/null | grep -q "ffmpeg version $FFMPEG_VER"; then
  echo "ffmpeg $FFMPEG_VER 已就绪"
else
  echo "下载 ffmpeg $FFMPEG_VER（evermeet.cx）..."
  curl -sL --max-time 300 -o /tmp/ffmpeg-dl.zip "https://evermeet.cx/ffmpeg/ffmpeg-$FFMPEG_VER.zip"
  unzip -o -j /tmp/ffmpeg-dl.zip ffmpeg -d bundle_bin/ >/dev/null
  chmod +x bundle_bin/ffmpeg
  bundle_bin/ffmpeg -version 2>/dev/null | head -1
  rm -f /tmp/ffmpeg-dl.zip
fi

# ---- exiftool（GitHub 官方 release）----
if [[ -x bundle_bin/exiftool ]] && bundle_bin/exiftool -ver 2>/dev/null | grep -q "$EXIFTOOL_VER"; then
  echo "exiftool $EXIFTOOL_VER 已就绪"
else
  echo "下载 exiftool $EXIFTOOL_VER（github.com/exiftool）..."
  curl -sL --max-time 300 -o /tmp/exiftool-dl.tar.gz \
    "https://github.com/exiftool/exiftool/archive/refs/tags/$EXIFTOOL_VER.tar.gz"
  rm -rf /tmp/exiftool-src && mkdir -p /tmp/exiftool-src
  tar -xzf /tmp/exiftool-dl.tar.gz -C /tmp/exiftool-src
  cp "/tmp/exiftool-src/exiftool-$EXIFTOOL_VER/exiftool" bundle_bin/exiftool
  rm -rf bundle_bin/lib
  cp -R "/tmp/exiftool-src/exiftool-$EXIFTOOL_VER/lib" bundle_bin/lib
  chmod +x bundle_bin/exiftool
  bundle_bin/exiftool -ver
  rm -rf /tmp/exiftool-src /tmp/exiftool-dl.tar.gz
fi

echo "bundle_bin 就绪："
ls -la bundle_bin/ | grep -v "^total\|^d"
