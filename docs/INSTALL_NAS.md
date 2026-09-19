# NAS 安装指南（小白版）

把「家庭回忆相册」装到你的 NAS 上，只需要 10 分钟。
全程只需要两样东西：**部署包**（我们提供）和**你的照片目录**。

---

## 一、你的 NAS 符合要求吗？

| 项目 | 最低要求 | 推荐 |
|---|---|---|
| 机型 | x86 架构 NAS（Intel/AMD CPU，如群晖 J4125、N100、N5105） | 同左 |
| 内存 | **4G 可用**（实测容器常态占用约 0.2G） | **8G 及以上** |
| 磁盘 | 镜像 1.7G + 数据（缩略图约照片数的 1/20 大小） | 留 10G 余量 |
| 系统 | 群晖 DSM 7.0+（装 Container Manager）/ 威联通 / 其他能跑 Docker 的 NAS | DSM 7.2+ |
| 照片 | 目前支持 x86 NAS 常见照片/视频（JPG/HEIC/MP4/MOV…） | — |

> ARM 架构 NAS（CPU 非 Intel/AMD）暂不保证可用。

---

## 二、安装（三选一，推荐方式 A）

### 方式 A：一键部署包（推荐，全程复制粘贴 2 条命令）

1. **下载部署包**：`family-memory-album-nas-deploy-0.5.tar.gz`（约 440M，内含程序镜像 + 安装脚本）
2. **传到 NAS**：群晖用 File Station 把它上传到任意目录，例如 `/homes/你的用户名/`
3. **开 SSH 并登录**（群晖：控制面板 → 终端机和 SNMP → 勾选「启用 SSH 功能」；然后用终端软件连 `ssh 你的NAS用户名@NAS的IP`）
4. **执行两条命令**：

```bash
cd /volume1/homes/你的用户名        # 换成你上传部署包的目录
./install_nas.sh --image family-memory-album-nas-deploy-0.5.tar.gz \
                 --dir /volume1/docker/family-memory-album
```

> 脚本会自动：检测 Docker → 建数据目录 → 导入镜像 → 生成配置 → 启动 →
> 打印访问地址。如果提示没有权限，在命令前加 `sudo `。

### 方式 B：纯图形界面（不碰命令行，用群晖 Container Manager）

1. 套件中心安装 **Container Manager**
2. 解压部署包，得到 `install_nas.sh` 和 `family-memory-album-0.5-nas-amd64.tar.gz`
3. Container Manager → **映像** → **导入** → 选择该镜像文件
4. Container Manager → **项目** → **创建**：
   - 项目名：`family-memory-album`
   - 路径：新建文件夹 `/docker/family-memory-album`
   - 来源：创建 docker-compose.yml，粘贴下面内容（改第 1 行镜像名和你自己的路径）：

```yaml
services:
  album:
    image: family-memory-album:0.5
    container_name: family-memory-album
    restart: unless-stopped
    ports:
      - "8788:8788"
    environment:
      - PORT=8788
      - FF_DATA_DIR=/data
      - FF_FACE_BACKEND=adaface
      - FF_ADAFACE_DIR=/models/adaface
      - FF_SIGLIP2_DIR=/models/siglip2-base-patch16-224
      - FF_BIND=0.0.0.0
      - MEDIA_ROOTS=/photos
    volumes:
      - /volume1/docker/family-memory-album/data:/data   # 数据库/缩略图
      - /volume1/photos:/photos                          # ← 改成你的照片目录
```

5. 点「下一步 → 完成」，容器状态变绿色即成功

### 方式 C：已有 Docker 的命令行用户（在线拉取）

```bash
# 从镜像仓库拉取（镜像发布地址见发布页；也可用阿里云 ACR 加速）
docker pull <仓库地址>/family-memory-album:0.5
REGISTRY=<仓库地址> ./install_nas.sh --dir /volume1/docker/family-memory-album
```

---

## 三、首次使用（3 步向导）

1. 浏览器打开 **`http://NAS的IP:8788`**（安装完成时脚本会打印这个地址）
2. **创建管理员账号** —— 这是全家共用的管理账号，密码记好
3. **选择照片目录** —— 向导的目录浏览器里选 `/photos`
   （方式 A/B 里 `/photos` 就是你在配置里填的那个照片文件夹）
4. 点「开始扫描」→ 等进度条跑完（1 万张约 30-60 分钟，后台进行，关掉浏览器也没关系）
5. 回到照片墙，按 **时间线 / 地点 / 人物 / 场景** 浏览；家人可以用你创建的「家人账号」看照片

> 手机/平板/电视：同一 WiFi 下浏览器打开同一地址即可，电视端有专门的 `/tv.html` 大屏模式。

---

## 四、性能预期（低配 NAS 实测参考）

以下为容器内实测（含人脸模型加载完成的常态）：

| 操作 | 耗时 | 说明 |
|---|---|---|
| 服务启动（首启） | 10-30 秒 | 加载内置人脸模型 |
| 照片缩略图（首次生成） | < 0.5 秒/张 | 后台自动批量生成，翻页无感 |
| 缩略图（已缓存） | < 10 毫秒 | 秒开 |
| 视频封面抽取 | 0.1-1 秒/个 | ffmpeg 抽帧 |
| 人脸识别（首扫批处理） | 每千张 5-15 分钟 | 后台慢慢跑，不卡浏览 |
| 全库分类接口（1.6 万张） | < 0.5 秒 | 时间线/地点/人物页 |
| 内存占用（常态） | 约 0.2G | 4G 机型无压力 |

> 首扫期间 NAS CPU 会比较高（风扇可能转起来），扫完恢复安静。
> 识别、缩略图、OCR 全部在你的 NAS 本地完成，**照片不会上传到任何云端**。

---

## 五、可选增强：开启语义搜索（AI 找照片）

默认安装已支持**按文字搜索照片**（基于已识别的场景/人物/地点标签）。
若想开启更强的 AI 向量搜索（如「夕阳下的合影」），需下载 1.4G 的 SigLIP 模型：

1. 确认 NAS 内存 ≥ 8G，然后在**电脑**上从国内镜像源下载模型（7 个文件，约 1.4G）：

   ```bash
   # macOS/Linux 电脑上执行，下载到当前目录 siglip2 文件夹
   BASE="https://hf-mirror.com/google/siglip2-base-patch16-224/resolve/main"
   mkdir -p siglip2-base-patch16-224 && cd siglip2-base-patch16-224
   for f in config.json model.safetensors preprocessor_config.json \
            special_tokens_map.json tokenizer.json tokenizer.model tokenizer_config.json; do
     curl -L -O "$BASE/$f"
   done
   ```

2. 把整个 `siglip2-base-patch16-224` 文件夹上传到 NAS 的
   `/volume1/docker/family-memory-album/models/` 下（File Station 拖拽即可）
3. 在 compose 的 volumes 里加一行：
   `- /volume1/docker/family-memory-album/models:/models-extra`
   并把环境变量改为 `- FF_SIGLIP2_DIR=/models-extra/siglip2-base-patch16-224`，重启容器

---

## 六、升级与卸载

```bash
# 升级：下载新版部署包，重跑安装脚本（数据全在 data/ 目录，不会丢）
./install_nas.sh --image family-memory-album-nas-deploy-<新版>.tar.gz --dir /volume1/docker/family-memory-album

# 看日志 / 重启
docker logs -f family-memory-album
docker restart family-memory-album

# 卸载（照片与数据保留）
docker rm -f family-memory-album
# 彻底清除：删除安装目录（内含数据库）——删除前请确认照片原片不在这里
```

---

## 七、常见问题

**Q：浏览器打不开 http://NAS的IP:8788？**
看容器状态：Container Manager 里是否「运行中」；或 SSH 执行 `docker logs family-memory-album`。防火墙放行 8788 端口。

**Q：照片扫描完是空的？**
确认 compose 里 `/photos` 挂载的宿主路径就是你放照片的目录；容器内执行 `ls /photos` 能看到照片。

**Q：人脸识别不准/认错人？**
设置 → 模型与任务里可调「人脸归属严格度」；也可以在人物页手动合并/拆分，系统会记住你的修正。

**Q：想换端口？**
改 compose 里 `"8788:8788"` 左边的数字，重启容器。

**Q：外网能访问吗？**
默认仅局域网。外网访问建议走群晖自带 VPN/快车套件或 Tailscale，不建议直接把 8788 暴露公网（未登录看不到照片，但仍建议加一层访问控制）。
