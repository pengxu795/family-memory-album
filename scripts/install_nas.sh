#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# ============================================================
# 家庭回忆相册 — NAS 一键部署脚本（面向所有 NAS 用户）
#
# 用法（NAS 的 SSH 终端里执行）：
#   ./install_nas.sh                              # 默认：脚本所在目录安装，在线拉镜像
#   ./install_nas.sh --image family-memory-album-0.5-nas-amd64.tar.gz
#                                                 # 离线安装（一键部署包自带镜像）
#   ./install_nas.sh --dir /volume1/docker/family-memory-album --port 8788
#   REGISTRY=registry.cn-hangzhou.aliyuncs.com/你的命名空间 ./install_nas.sh
#
# 做什么：检测 Docker → 建数据目录 → 获取镜像（在线/离线）→ 生成
#         docker-compose.yml → 启动 → 输出访问地址和首次使用引导。
# ============================================================
set -u
IMAGE_DEFAULT="family-memory-album:1.0.0"
DEFAULT_REGISTRY=""                        # 例：registry.cn-hangzhou.aliyuncs.com/你的命名空间
IMAGE="${IMAGE:-$IMAGE_DEFAULT}"
IMAGE_TAR=""
INSTALL_DIR="$(cd "$(dirname "$0")" && pwd)"
PORT="${PORT:-8788}"
CNAME="${CNAME:-family-memory-album}"      # 容器名（多实例测试可改）
PHOTOS_SRC=""                              # 可选：已有照片目录

# ---------- 参数解析 ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --image)  IMAGE_TAR="$2"; shift 2;;
    --dir)    INSTALL_DIR="$2"; shift 2;;
    --port)   PORT="$2"; shift 2;;
    --photos) PHOTOS_SRC="$2"; shift 2;;
    --name)   CNAME="$2"; shift 2;;
    --image-name) IMAGE="$2"; shift 2;;
    -h|--help) sed -n '3,16p' "$0"; exit 0;;
    *) echo "未知参数: $1"; exit 1;;
  esac
done

say()  { printf '\n\033[1;32m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[!]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[x]\033[0m %s\n' "$*"; exit 1; }

# 2026-09-19：一键部署包场景——脚本同目录自带离线镜像包则自动走离线安装
if [ -z "$IMAGE_TAR" ]; then
  _auto="$(cd "$(dirname "$0")" 2>/dev/null && ls family-memory-album-*-nas-amd64.tar.gz 2>/dev/null | head -1)"
  if [ -n "$_auto" ]; then
    IMAGE_TAR="$(cd "$(dirname "$0")" && pwd)/$_auto"
    say "发现同目录离线镜像包：$_auto（自动离线安装）"
  fi
fi

# ---------- 1. 检测 Docker（兼容群晖路径） ----------
say "检测 Docker 环境…"
DOCKER=""
for c in docker /usr/local/bin/docker /usr/local/bin/dockerx /volume1/@appstore/ContainerManager/usr/bin/docker; do
  if command -v "$c" >/dev/null 2>&1; then DOCKER="$c"; break; fi
done
[ -z "$DOCKER" ] && die "没找到 Docker。
  · 群晖 DSM：套件中心安装「Container Manager」（7.2+）或「Docker」（7.0-7.1）后重试
  · 威联通 QTS：安装 Container Station 后重试
  · 其他 Linux：先安装 Docker（curl -fsSL https://get.docker.com | sh）"
echo "  docker: $DOCKER"

if ! "$DOCKER" ps >/dev/null 2>&1; then
  warn "当前用户无 docker 权限，尝试 sudo…"
  [ "$(id -u)" = "0" ] || exec sudo -E "$0" "$@"   # 以 root 重跑自身
  "$DOCKER" ps >/dev/null 2>&1 || die "docker 不可用（权限/服务未启动）"
fi

ARCH=$("$DOCKER" version --format '{{.Server.Arch}}' 2>/dev/null || uname -m)
say "CPU 架构: $ARCH"
[ "$ARCH" = "x86_64" ] || [ "$ARCH" = "amd64" ] || warn "本镜像是 linux/amd64 构建；ARM 架构 NAS（如部分 Realtek/ARM 群晖）暂不保证可用，J4125/N100/N5105 等 x86 机型无问题。"

COMPOSE=""
if "$DOCKER" compose version >/dev/null 2>&1; then COMPOSE="$DOCKER compose"
elif command -v docker-compose >/dev/null 2>&1; then COMPOSE="docker-compose"
elif [ -x /usr/local/bin/docker-compose ]; then COMPOSE="/usr/local/bin/docker-compose"
fi

# ---------- 2. 建目录 ----------
say "创建数据目录：$INSTALL_DIR"
mkdir -p "$INSTALL_DIR/data" "$INSTALL_DIR/photos"
# models 目录：0.5 镜像已内置核心模型，留空目录供可选挂载（语义搜索模型等）
mkdir -p "$INSTALL_DIR/models"
if [ -n "$PHOTOS_SRC" ]; then
  [ -d "$PHOTOS_SRC" ] || die "照片目录不存在：$PHOTOS_SRC"
  PHOTOS_MOUNT="$PHOTOS_SRC"
else
  PHOTOS_MOUNT="$INSTALL_DIR/photos"
  warn "示例照片目录 $INSTALL_DIR/photos（把照片/视频放进来，或用 --photos 指定已有目录）"
fi

# ---------- 3. 获取镜像 ----------
if [ -n "$IMAGE_TAR" ]; then
  [ -f "$IMAGE_TAR" ] || [ -f "$INSTALL_DIR/$IMAGE_TAR" ] || die "镜像包不存在：$IMAGE_TAR"
  TAR_PATH="${IMAGE_TAR:-$INSTALL_DIR/$IMAGE_TAR}"; [ -f "$TAR_PATH" ] || TAR_PATH="$INSTALL_DIR/$IMAGE_TAR"
  say "离线导入镜像：$TAR_PATH（约 1-2 分钟）"
  # 兼容两种包：镜像包（docker save|gzip）或一键部署包（内含 install_nas.sh + 镜像包）
  if tar -tzf "$TAR_PATH" 2>/dev/null | grep -q "family-memory-album-.*-nas-amd64.tar.gz"; then
    warn "检测到一键部署包（内含镜像），先解出镜像…"
    STAGE=$(mktemp -d)
    tar -xzf "$TAR_PATH" -C "$STAGE"
    TAR_PATH="$STAGE/$(tar -tzf "$TAR_PATH" | grep -o 'family-memory-album-.*-nas-amd64.tar.gz' | head -1)"
    trap 'rm -rf "$STAGE"' EXIT
  fi
  gzip -dc "$TAR_PATH" | "$DOCKER" load || die "镜像导入失败（文件损坏？重新下载部署包）"
  IMAGE=$(gzip -dc "$TAR_PATH" 2>/dev/null | "$DOCKER" load 2>/dev/null | grep -o 'family-memory-album:[0-9.]*' | head -1)
  IMAGE="${IMAGE:-$IMAGE_DEFAULT}"
else
  say "拉取镜像：$IMAGE"
  PULL_TARGET="$IMAGE"
  if [ -n "${REGISTRY:-}" ]; then PULL_TARGET="${REGISTRY%/}/$(echo "$IMAGE" | sed 's|^docker.io/||')"; fi
  "$DOCKER" pull "$PULL_TARGET" || {
    echo
    die "拉取失败。两个办法：
  1) 离线安装：把一键部署包里的 family-memory-album-*-nas-amd64.tar.gz 传到 NAS，执行
       ./install_nas.sh --image <该文件>
  2) 或配置 REGISTRY 后重试（如阿里云 ACR 个人实例）：
       REGISTRY=registry.cn-hangzhou.aliyuncs.com/你的命名空间 ./install_nas.sh"
  }
  IMAGE="$PULL_TARGET"
fi
say "使用镜像：$IMAGE"

# ---------- 4. 生成 docker-compose.yml ----------
say "生成 $INSTALL_DIR/docker-compose.yml"
cat > "$INSTALL_DIR/docker-compose.yml" <<EOF
# 家庭回忆相册 — 由 install_nas.sh 自动生成
services:
  album:
    image: $IMAGE
    container_name: $CNAME
    restart: unless-stopped
    ports:
      - "$PORT:8788"
    environment:
      - PORT=8788
      - FF_DATA_DIR=/data
      - FF_FACE_BACKEND=opencv
      - FF_OPENCV_MODEL_DIR=/models
      - FF_SIGLIP2_DIR=/models/siglip2-base-patch16-224
      - FF_BIND=0.0.0.0
      - MEDIA_ROOTS=/photos
    volumes:
      - $INSTALL_DIR/data:/data
      - $PHOTOS_MOUNT:/photos
    healthcheck:
      test: ["CMD-SHELL", "python -c \"import http.client; c=http.client.HTTPConnection('127.0.0.1', 8788, timeout=12); c.request('GET','/'); r=c.getresponse(); r.read(); exit(0 if r.status<500 else 1)\""]
      interval: 60s
      timeout: 15s
      retries: 3
EOF

# ---------- 5. 启动 ----------
say "启动服务…"
if [ -n "$COMPOSE" ]; then
  (cd "$INSTALL_DIR" && $COMPOSE up -d) || die "启动失败，查看日志：$DOCKER logs family-memory-album"
else
  # 无 compose 的老环境：等价 docker run
  "$DOCKER" rm -f "$CNAME" >/dev/null 2>&1 || true
  "$DOCKER" run -d --name "$CNAME" --restart unless-stopped \
    -p "$PORT:8788" -e PORT=8788 -e FF_DATA_DIR=/data -e FF_FACE_BACKEND=opencv \
    -e FF_OPENCV_MODEL_DIR=/models \
    -e FF_SIGLIP2_DIR=/models/siglip2-base-patch16-224 \
    -e FF_BIND=0.0.0.0 -e MEDIA_ROOTS=/photos \
    -v "$INSTALL_DIR/data:/data" -v "$PHOTOS_MOUNT:/photos" \
    "$IMAGE" || die "docker run 失败"
fi

# ---------- 6. 等待就绪 ----------
say "等待服务就绪（首次启动要加载人脸模型，约 10-30 秒）…"
for i in $(seq 1 30); do
  if curl -s --noproxy '*' -o /dev/null --max-time 3 "http://127.0.0.1:$PORT/" 2>/dev/null; then break; fi
  # 群晖可能没有 curl
  if "$DOCKER" exec "$CNAME" python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8788/', timeout=3)" >/dev/null 2>&1; then break; fi
  sleep 2
done

# NAS IP 提示（Linux NAS 优先；macOS/其它环境兜底）
IP=""
if command -v ip >/dev/null 2>&1; then IP=$(ip route get 1 2>/dev/null | awk '{print $NF; exit}'); fi
if [ -z "$IP" ] && command -v hostname >/dev/null 2>&1; then IP=$(hostname -I 2>/dev/null | awk '{print $1}'); fi
if [ -z "$IP" ] && command -v ipconfig >/dev/null 2>&1; then IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null); fi
[ -z "$IP" ] && IP="<NAS-IP>"

cat <<EOF

============================================================
  ✅ 家庭回忆相册部署完成！
============================================================

  访问地址：  http://$IP:$PORT
  （在浏览器打开；同一局域网内任何设备都能访问）

  首次使用（跟着浏览器向导走，共 3 步）：
    1. 创建管理员账号（记好密码）
    2. 选择照片目录 → 填 /photos（照片已在 $PHOTOS_MOUNT）
    3. 开始扫描，等进度条跑完就能看到照片墙

  常用命令：
    看日志    $DOCKER logs -f $CNAME
    重启      $DOCKER restart $CNAME
    升级      下载新版部署包后重跑 install_nas.sh --image <新镜像包>
    卸载      cd $INSTALL_DIR && $DOCKER rm -f $CNAME
              （照片和数据库在 $INSTALL_DIR，删容器不会丢）

  数据位置：  $INSTALL_DIR/data（数据库与缩略图）
              $PHOTOS_MOUNT（照片原片，始终不动）

  内存建议：  4G 可用（人脸识别可用）；8G 及以上体验最佳
  语义搜索：  可选增强，见 docs/INSTALL_NAS.md「开启语义搜索」
============================================================
EOF
