# 家庭回忆相册 · Family Memory Album

自托管的家庭照片库：把散落在 NAS / 网盘 / 本机目录里的照片和视频聚到一个入口，
自动整理、识别人脸与场景，并支持「宝宝2岁时的照片」这类**自然语言直接检索**。

单文件 Python 服务（`server.py`）+ SQLite，资源占用极低，一台 4GB 内存的 NAS 即可常年运行。

## 功能特性

- **一键导入多来源**：本机目录 / NAS 挂载 / 网盘，增量和自动扫描，新照片自动入库
- **照片墙与时间轴**：按天/月聚合的平铺照片墙，月份条按照片流位置对齐
- **人脸识别与人物管理**：内置 YuNet/SFace（Apache-2.0），可选 AdaFace 后端；按人浏览、手动归属
- **相似照片整理**：自动分组、AI 择优、一键重选"最佳"
- **自然语言问答检索**：「去年夏天在海边的照片」——本地 SIGLIP 语义检索 + 可接入大模型增强
- **地点与地图**：GPS 聚合、旅行标记、中国地图视图（可缩放到市级）
- **AI 语义筛选**：接入任意 OpenAI 兼容视觉服务，自动过滤截图/文档/垃圾照片（误杀有护栏）
- **隐私相册**：两级级联自动检测（文本快筛 + 视觉精判），私密内容自动移入、随时移出
- **多成员账号**：家人/访客角色分级，访客只读
- **在线升级**：配置更新源后，公网推送升级包

## 快速开始（Docker）

```bash
# 1. 下载模型权重（Apache-2.0，约 40MB；首次构建必须）
python3 scripts/download_models.py

# 2. 构建镜像
docker build -t family-memory-album .

# 3. 运行（/data 存数据库与缓存，照片目录按需挂载）
docker run -d --name family-memory-album \
  -p 8788:8788 \
  -v /your/data/dir:/data \
  -v /your/photos:/photos:ro \
  -e FF_DATA_DIR=/data \
  family-memory-album
```

浏览器打开 `http://localhost:8788`，首次进入会引导创建管理员账号并选择照片目录。

> 本地直接运行也行：`pip install -r requirements-docker.txt && python server.py`（需 Python 3.11+，opencv/onnxruntime 等见 requirements）。

## 接入 AI 服务（可选）

设置 → AI 与任务 → 添加 AI 服务：粘贴任意 **OpenAI 兼容 API 地址**（自动识别服务商）+ Key 即可，
用于视觉语义筛选、自然语言问答等增强能力。**不接入也完全可用**（人脸、地图、去重、整理均为本地能力）。

常用免费/低价视觉服务：SiliconFlow（Qwen3-VL 系列）、智谱 GLM-4V 等。

## 环境变量

| 变量 | 说明 | 默认 |
|---|---|---|
| `FF_DATA_DIR` | 数据目录（SQLite、缩略图、日志） | `./data` |
| `FF_FACE_BACKEND` | 人脸后端：`opencv`（SFace，合规默认）/ `adaface`（需自担权重许可） | `opencv` |
| `FF_BIND` / `PORT` | 监听地址 / 端口 | `127.0.0.1` / `8788` |
| `FF_ADAFACE_DIR` / `FF_SIGLIP2_DIR` | 模型目录覆盖 | 内置 `baked_models/` |

## 隐私与数据

- 所有照片、数据库、人脸向量都在你自己的磁盘上，**没有内置任何云依赖或上报**
- AI 增强能力只有在你主动配置服务商 Key 后才会把缩略图发给该服务商
- 数据库单文件 SQLite，备份 = 复制 `family_memory.db`

## 模型许可

代码 MIT。内置权重均为可分发许可（YuNet/SFace/U²-Netp 等 Apache-2.0），
AdaFace 等非商业权重**不随产品分发**，仅供本地自用——详见 [docs/MODEL_LICENSES.md](docs/MODEL_LICENSES.md)。

## License

[MIT](LICENSE)
