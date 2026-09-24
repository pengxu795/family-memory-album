# ============================================================
# 家庭回忆相册 — 服务端镜像（施工图#7）
# 构建：docker build -t family-memory-album:0.1 .
# 运行：见 docker-compose.yml
# ============================================================

# 锁 Python 3.12：server.py 顶层 import 的 cgi 模块在 3.13 已移除，升 3.13 会启动即崩
FROM python:3.12-slim

# 国内网络直连 PyPI 极慢（NAS 实测 17 分钟下不完轮子），默认走清华源；
# 海外机器构建可 --build-arg 覆盖或改回官方源
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PORT=8788 \
    FF_DATA_DIR=/data \
    FF_FACE_BACKEND=opencv \
    FF_OPENCV_MODEL_DIR=/models \
    FF_SIGLIP2_DIR=/models/siglip2-base-patch16-224 \
    FF_BIND=0.0.0.0

# 系统依赖（Debian 包名注意：exiftool 的包是 libimage-exiftool-perl）
#   ffmpeg                    视频转码 / 抽帧 / 封面生成
#   libimage-exiftool-perl    EXIF 读取（拍摄时间、GPS）
#   libheif1                  HEIC 运行时解码库
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libimage-exiftool-perl \
        libheif1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖（与 requirements-docker.txt 一一对应，无编译型依赖）
COPY requirements-docker.txt .
RUN pip install -r requirements-docker.txt

# ---- 代码与运行时资源 ----
# import 清单核对（AST 扫描 server.py 全文件，2026-09-08）：
#   顶层 import 本地模块 : geo_cities_cn.py（中国地图城市数据，唯一顶层非标准库 import）
#   懒加载本地模块       : schema.py（启动建库）、backfill_faces_full.py（人脸管线子进程）
#   子进程脚本           : backfill_siglip_embedding.py（SigLIP 向量回填）
#   子进程脚本           : enrich.py（富化流水线，server.py 以 ROOT/enrich.py 拉起）
#   建库依赖             : schema_raw.json（71 表基线）、migrations/*.sql（增量迁移）
#   前端资源             : static/
# 2026-09-18 修复：此前漏了 enrich.py，导致重新构建的镜像里没有富化流水线
#   （server.py 靠 subprocess 拉起 /app/enrich.py，缺文件则富化静默失效）
COPY server.py schema.py geo_cities_cn.py \
     backfill_faces_full.py face_backends.py backfill_siglip_embedding.py \
     enrich.py vlm_describe_assets.py caption_days.py privacy_auto_scan.py \
     manual_face_create.py transcode_log_videos.py \
     migrate_paths.py schema_raw.json /app/
# ---- 核心模型内置（开源发行版，全部 Apache-2.0 可分发）----
# YuNet 人脸检测 + SFace 人脸识别（后端默认 opencv）+ U²-Netp 抠图，合计 ≈43M。
# siglip2（1.4G 向量模型）不内置：4G 内存 NAS 带不动且拖慢首启；
# 开启语义搜索的方法见 docs/INSTALL_NAS.md「可选增强」（hf-mirror 手动下载挂载）。
# AdaFace 等非商用权重（WebFace4M 许可）不随镜像分发；如需更高精度，
# 自行下载权重挂载并设 FF_FACE_BACKEND=adaface（见 docs/MODEL_LICENSES.md）。
# 注意：容器内 /models 已内置核心模型，compose 不应再挂载 /models 卷覆盖。
COPY baked_models/ /models/
COPY migrations/ /app/migrations/
COPY static/ /app/static/

# 数据目录由数据卷提供（compose 挂 ./data:/data，FF_DATA_DIR 已指向 /data）
# 日志目录 /app/logs 由服务启动时自动创建（server.py:7895 mkdir）
# 可选模型（如 siglip2 语义向量、AdaFace 高精度识别）不进镜像：
#   下载后挂载到对应目录并按 docs/INSTALL_NAS.md「可选增强」配置环境变量

EXPOSE 8788

# 健康检查：任意 HTTP 响应（含 302 初始化门禁）都算活，5xx 才算死
# 2026-09-18 放宽 timeout 5s → 15s：NAS 总内存仅 3.9GB 且 swap 常用 ~1.5GB，
#   负载上来时 GET / 偶发超过 5s，会让容器无故翻成 unhealthy（空闲实测仅 0.26–0.46s）。
#   15s 仍能在真正挂死时（连接超时/无响应）及时判死，只是不再被抖动误伤。
HEALTHCHECK --interval=60s --timeout=15s --start-period=30s --retries=3 \
    CMD python -c "import http.client,os; c=http.client.HTTPConnection('127.0.0.1', int(os.environ.get('PORT','8788')), timeout=12); c.request('GET','/'); r=c.getresponse(); r.read(); raise SystemExit(0 if r.status < 500 else 1)"

CMD ["python", "-u", "server.py"]
