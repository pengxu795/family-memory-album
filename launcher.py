#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""家庭回忆相册 — macOS 安装包启动器（pyinstaller 唯一入口）

职责（server.py 保持零打包感知，所有路径/环境适配集中在这里）：
1. 装配运行环境：DATA_DIR、models、ffmpeg 等 env（frozen 时默认
   ~/Library/Application Support/FamilyMemoryAlbum，非 frozen 直跑等价于原 server.py）
2. --ff-worker <key> 子进程分发：pyinstaller 打包后 sys.executable 不再是
   python 解释器，"python xxx.py" 拉起方式失效，统一走本启动器分发
3. --serve 主服务：runpy 执行 _MEIPASS/server.py（run_name=__main__）

用法（打包后）：
  FamilyMemoryAlbum            → 起服务 + 自动开浏览器（FF_OPEN_BROWSER=0 关闭）
  FamilyMemoryAlbum --ff-worker faces   → 人脸管线子进程（由主服务拉起，勿手动）
"""
import os
import sys
from pathlib import Path

IS_FROZEN = getattr(sys, "frozen", False)
if IS_FROZEN:
    BUNDLE_ROOT = Path(sys._MEIPASS).resolve()          # server.py / static / migrations 所在
    try:
        APP_DIR = Path(sys.executable).resolve().parents[3]  # .../FamilyMemoryAlbum.app 上一级
    except IndexError:
        APP_DIR = Path.home()
else:
    BUNDLE_ROOT = Path(__file__).resolve().parent
    APP_DIR = BUNDLE_ROOT

# ---- 模型目录：app 内 Resources/models 优先，其次 app 同级 models/，env 最高 ----
def _model_dir(name):
    for base in (BUNDLE_ROOT / "models", APP_DIR / "models"):
        p = base / name
        if p.exists():
            return str(p)
    return None

if IS_FROZEN:
    data_default = Path.home() / "Library" / "Application Support" / "FamilyMemoryAlbum"
    os.environ.setdefault("FF_DATA_DIR", str(data_default))
    adaf = _model_dir("adaface")
    if adaf:
        os.environ.setdefault("FF_ADAFACE_DIR", adaf)
    sig = _model_dir("siglip2-base-patch16-224")
    if sig:
        os.environ.setdefault("FF_SIGLIP2_DIR", sig)
    os.environ.setdefault("FF_FACE_BACKEND", "adaface")
    os.environ.setdefault("FF_BIND", "127.0.0.1")
    # 打包版无 torch/transformers：SigLIP 探针自动判离线（server.py 优雅降级）
    os.environ.setdefault("SIGLIP_PYTHON", "/usr/bin/python3-nonexistent-frozen")

# ---- 子进程分发（必须赶在 import server 之前，因为主服务 Popen 拉的就是本二进制）----
if "--ff-worker" in sys.argv:
    i = sys.argv.index("--ff-worker")
    key = sys.argv[i + 1]
    rest = sys.argv[i + 2:]
    scripts = {
        "faces": "backfill_faces_full.py",
        "siglip": "backfill_siglip_embedding.py",
        "captions": "caption_days.py",
        "manual_face": "manual_face_create.py",
        # 2026-09-10 修 .app 启动锁死：vlm 描述/隐私检测原来用
        # `sys.executable xxx.py` 拉起——frozen 下 executable 是本二进制，
        # 不匹配 --ff-worker → 走主服务分支再起一个实例（端口冲突 + 多实例
        # 抢 SQLite 写锁）。现在统一经此分发。
        "vlm": "vlm_describe_assets.py",
        "privacy": "privacy_auto_scan.py",
        # 2026-09-16：导入后增量富化（精确去重/画质/相似分组/择优/语义过滤）
        "enrich": "enrich.py",
    }
    script = BUNDLE_ROOT / scripts[key]
    sys.argv = [sys.argv[0]] + rest
    import runpy
    runpy.run_path(str(script), run_name="__main__")
    raise SystemExit(0)

# ---- 主服务 ----
if "--serve" in sys.argv or True:
    if IS_FROZEN:
        # 2026-09-10 日志落盘：open/Finder 启动时 stdout 被 macOS 吞掉，
        # sqlite locked 之类的问题完全无从排查。主服务 stdout/stderr 同步
        # 写入 logs/server-<时间>.log，保留最近 5 份，启动即打印路径。
        LOG_DIR = (Path(os.environ.get("FF_DATA_DIR", "")) if os.environ.get("FF_DATA_DIR")
                   else Path.home() / "Library" / "Application Support" / "FamilyMemoryAlbum")
        LOG_DIR = LOG_DIR / "logs"
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            old_logs = sorted(LOG_DIR.glob("server-*.log"))
            for p in old_logs[:-4]:
                p.unlink(missing_ok=True)
            _logf = open(LOG_DIR / f"server-{__import__('datetime').datetime.now():%Y%m%d-%H%M%S}.log",
                         "a", buffering=1, encoding="utf-8")

            class _Tee:
                """同时写原 stdout/stderr 与日志文件（行缓冲）。"""
                def __init__(self, *streams):
                    self._streams = streams
                def write(self, b):
                    n = 0
                    for s in self._streams:
                        try:
                            n = s.write(b)
                        except Exception:
                            pass
                    return n
                def flush(self):
                    for s in self._streams:
                        try:
                            s.flush()
                        except Exception:
                            pass

            sys.stdout = _Tee(sys.stdout, _logf)
            sys.stderr = _Tee(sys.stderr, _logf)
            print(f"[launcher] 日志落盘: {_logf.name}", flush=True)
        except Exception as exc:
            print(f"[launcher] 日志落盘失败(不影响运行): {exc}", flush=True)
    if IS_FROZEN and os.environ.get("FF_OPEN_BROWSER", "1") != "0":
        import threading
        import webbrowser
        port = os.environ.get("PORT", "8788")
        threading.Timer(2.0, lambda: webbrowser.open(f"http://127.0.0.1:{port}/")).start()
    import runpy
    runpy.run_path(str(BUNDLE_ROOT / "server.py"), run_name="__main__")
