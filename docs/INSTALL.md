# 家庭回忆相册 — macOS 安装说明

> 安装包：`releases/FamilyMemoryAlbum-<版本>.dmg`（约 1.5G，模型已内置，装完即用）。
> 服务器/Docker 部署见 `docs/DEPLOY_DOCKER.md`；升级见 `docs/UPGRADE.md`。

## 一、安装步骤

1. 双击打开 `FamilyMemoryAlbum-<版本>.dmg`。
2. 把 **家庭回忆相册.app** 拖入右边的 **Applications** 文件夹。
3. 从「启动台 / 应用程序」启动。

**⚠️ 未签名应用（决策点③）**：安装包是 ad-hoc 签名（无开发者账号），
拷贝到**其他 Mac** 首次打开会被 Gatekeeper 拦——**右键点 app → 打开 → 再点打开**，
只需一次。本机构建本机用不受影响。

## 二、首次使用（三步向导）

1. **创建管理员**：设管理员用户名和密码（家人/访客账号以后在设置里加）。
2. **选择照片目录**：点「📁 浏览目录」从常用位置（照片图库、/Volumes 下的
   NAS 挂载盘等）逐级选择，或直接粘贴完整路径；点「校验目录」确认。
3. **首次扫描**：自动遍历目录入库（进度按入库真实百分比显示）。

数据全部存本机：`~/Library/Application Support/FamilyMemoryAlbum/`
（DB、缩略图、人脸数据；原片只读，相册不修改任何原文件）。
模型权重已内置 app 内，无需额外下载。

## 三、三个出厂决策点（用户可改）

| 决策点 | 现状 | 影响 | 怎么改 |
|---|---|---|---|
| ① ffmpeg/exiftool **已内置**（0.4 起） | 静态 ffmpeg 9.0.1（evermeet.cx）+ exiftool 13.55 打进 app（Contents/Resources/bin/），无需 brew | 视频缩略图/转码、EXIF 时间/GPS 提取开箱即用 | 仍可用 `FFMPEG_BIN`/`EXIFTOOL_BIN` 环境变量指向其他版本 |
| ② console 模式 | .app 启动时保留终端窗口显示服务日志 | 便于看启动/错误日志；启动后会多一个终端窗口 | 不想要窗口：spec 里 `console=True` 改 `False` 重打包 |
| ③ 未签名（ad-hoc） | 无 Apple 开发者账号签名/公证 | 其他 Mac 首开被 Gatekeeper 拦（右键→打开绕过） | 有开发者账号后 `codesign --deep --sign "Developer ID" + notarytool 公证` |

### 许可说明（内置二进制）

- **ffmpeg 7.1.1（静态构建）**：ffmpeg 基于 LGPL/GPL 各组件构建；静态全功能构建
  按 **GPL v2+** 分发。本安装包仅为本地个人使用分发，不对外销售；若未来公开分发
  安装包，需遵守 GPL（提供对应源码链接，或改用 LGPL 构建 + 动态链接）。
- **exiftool 13.55**：Perl Artistic License（2.0 / GPL 1+ 双许可），可自由再分发。
  内置为官方可移植发行版（exiftool 脚本 + lib/ 模块目录），运行依赖 macOS 自带
  `/usr/bin/perl`。许可原文见 app 内 `Contents/Resources/bin/lib/../` 上游
  `LICENSE`（bundle_bin/lib 同源）。
- 其余开源组件许可见 `docs/MODEL_LICENSES.md`（AdaFace MIT / SFace Apache 2.0 等）。

## 四、日常使用提示

- 启动 app 即自动起服务并打开浏览器 `http://127.0.0.1:8788`；
  家里其他设备访问 `http://<这台Mac的IP>:8788`。
- 关掉终端窗口 = 服务停止；要常驻可后续做 launchd 开机自启。
- **语义检索（SigLIP）出厂离线**：该功能需要 Python torch 环境（体积 +1.5G），
  默认未带；人脸/地图/过滤/去重/搜索全部功能不受影响。
- NAS 上也有同款服务时，两边数据独立——标注工作在 Mac 做完用
  `scripts/sync_db_to_nas.sh` 一键推给 NAS（见 docs/UPGRADE.md 第四节）。
