# -*- mode: python ; coding: utf-8 -*-
# 家庭回忆相册 — macOS 安装包 spec（pyinstaller onedir → .app → DMG）
# 构建：pyinstaller FamilyMemoryAlbum.spec
# 产物：dist/FamilyMemoryAlbum.app（models 内置 Contents/Resources/models）

import os
from pathlib import Path

HERE = Path(SPECPATH)
MODELS_DIR = HERE.parent / "models"   # 仓库上一级 models/（adaface + siglip2）
BUNDLE_BIN = HERE / "bundle_bin"      # 内置 exiftool（官方可移植发行版：脚本+lib/）

block_cipher = None

datas = [
    ("static", "static"),
    ("migrations", "migrations"),
    ("schema_raw.json", "."),
    # runpy 动态执行的脚本：pyinstaller 分析不到，必须显式带上
    ("server.py", "."),
    ("schema.py", "."),
    ("geo_cities_cn.py", "."),
    ("face_backends.py", "."),
    ("backfill_faces_full.py", "."),
    ("backfill_siglip_embedding.py", "."),
    ("caption_days.py", "."),
    ("manual_face_create.py", "."),
    ("migrate_paths.py", "."),
]
if MODELS_DIR.exists():
    datas.append((str(MODELS_DIR), "models"))

# 内置 exiftool（脚本 + perl 模块目录）：exiftool 用 $0 相对 ../lib 找模块，
# 所以布局必须是 bin/exiftool + bin/lib/（datas 目标 "bin" + "bin/lib"）
if BUNDLE_BIN.exists():
    datas.append((str(BUNDLE_BIN / "exiftool"), "bin"))
    datas.append((str(BUNDLE_BIN / "lib"), "bin/lib"))

binaries = []

# 内置静态 ffmpeg 9.0.1（决策点①，evermeet.cx 静态构建，scripts/fetch_bundle_bin.sh 获取）：
# pyinstaller binaries 保留执行权限；先 bundle_bin 后 ~/.local/bin 兜底
import glob as _glob, os as _os
_FFMPEG_SRC = str(HERE / "bundle_bin" / "ffmpeg")
if not _os.path.exists(_FFMPEG_SRC):
    _FFMPEG_SRC = _os.path.expanduser("~/.local/bin/ffmpeg")
if _os.path.exists(_FFMPEG_SRC):
    binaries.append((_FFMPEG_SRC, "bin"))

hiddenimports = [
    # server.py 经 runpy 执行，pyinstaller 只分析 launcher 的 import 链，
    # server.py 顶层标准库也要显式列出（uuid 等实测会被漏收）
    "uuid", "secrets", "hashlib", "cgi", "http.cookies", "http.server",
    "socketserver", "webbrowser", "shutil", "tempfile", "urllib.parse",
    "sqlite3", "json", "base64", "gzip", "mimetypes", "pathlib",
    # server.py / worker 脚本经 runpy 执行，三方依赖必须显式收集
    "PIL", "PIL.Image", "PIL.ImageOps",
    "pillow_heif",
    "cv2",
    "numpy",
    "yaml",
    "onnxruntime",
    # 本地模块（server 懒加载 / worker 脚本 import）
    "geo_cities_cn",
    "schema",
    "face_backends",
    "backfill_faces_full",
]

a = Analysis(
    ["launcher.py"],
    pathex=[str(HERE)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["torch", "transformers", "tkinter", "matplotlib", "scipy"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="FamilyMemoryAlbum",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,   # 第一版保留控制台便于看启动日志（决策点：后续可切 windowed+文件日志）
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="FamilyMemoryAlbum",
)

app = BUNDLE(
    coll,
    name="FamilyMemoryAlbum.app",
    icon=None,
    bundle_identifier="io.github.family-memory-album",
)
