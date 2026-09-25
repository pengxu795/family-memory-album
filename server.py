#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
家庭记忆 MVP 后端
================
单页应用服务：自然语言查询 → 意图解析 → Memory Graph 检索 → 照片 + 回答

架构：
- /api/ask : POST 自然语言查询 → { answer, assets[], events[], persons[] }
- /thumb?asset=xxx : 按需生成并缓存缩略图（照片 sips / 视频 ffmpeg）
- / : 静态前端页面

意图解析：DeepSeek API（LLM 把自然语言转结构化 JSON）
检索：SQLite 组合（Person / Event / Memory / Time / Location / Age）
"""
import json, math, os, re, secrets, shutil, sqlite3, subprocess, tempfile, time, urllib.request, hashlib, uuid
import gzip
import html, mimetypes
import cgi, base64
import sys
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# 2026-09-08 NAS Docker 化：backfill_faces_full 依赖 cv2，改懒加载——
# 浏览/搜索/缩略图等核心服务不再要求 opencv，仅人脸任务首次调用时导入。
_bff = None


def _get_bff():
    global _bff
    if _bff is None:
        import backfill_faces_full
        _bff = backfill_faces_full
    return _bff

ROOT = Path(__file__).resolve().parent
# 数据目录外置（施工图#7）：Docker 里用 FF_DATA_DIR=/data 挂数据卷，换镜像升级不丢库。
# 不设置时默认 ROOT/data，本地 Mac 行为零变化。
DATA_DIR = Path(os.environ.get("FF_DATA_DIR") or (ROOT / "data"))
APP_VERSION = "1.0.9"   # 在线升级版本号（发布新包时同步改这里，见 make_update.py）
DB = DATA_DIR / "family_memory.db"
STATIC = ROOT / "static"
THUMB_DIR = DATA_DIR / "thumbs_mvp"
PREVIEW_DIR = DATA_DIR / "previews_mvp"
VIDEO_LC_DIR = DATA_DIR / "videos_lc"   # 2026-09-23 Log 视频还原转码缓存（可选，转好一条生效一条）
FACE_CROP_DIR = DATA_DIR / "face_crops"
UPLOAD_DIR = DATA_DIR / "face_uploads"
LOGS_DIR = ROOT / "logs"
# 2026-09-16 修：此前 siglip_backfill_run / faces_backfill_run 直接
# open(ROOT/"logs"/"...","a")，而容器镜像里没有 logs/ 目录 → FileNotFoundError。
# 该异常发生在后台线程体内，外层 try 只包住 Thread.start()，接不住 → 线程静默死亡，
# 日志里连一行错误都没有。后果：**手机每天备份进来的新照片，人脸归属和 SigLIP 向量
# 回填从来没在 NAS 上跑起来过**（未归属人脸 58% 的原因）。这里启动即建目录。
try:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass
PORT = int(os.environ.get("PORT", 8788))

# ============ 配置外置（施工图#6）：外部工具与模型路径全部可用环境变量覆盖 ============
# 默认值保持 Mac 本机行为不变；Docker 镜像里 ffmpeg/exiftool 在 PATH 上由 which 解析。
# 2026-09-10 决策点①：安装包内置二进制（Contents/.../bin/，frozen 时 ROOT 即包内 _MEIPASS），
#   解析顺序：env > 包内内置 > PATH > homebrew 兜底；Docker 镜像不内置（保持现状）。
# 通用性硬约束：server.py 内禁止出现机器专属绝对路径（2026-09-08 起，#16 清洗项）。
FFMPEG_BIN = (os.environ.get("FFMPEG_BIN")
              or ((ROOT / "bin" / "ffmpeg").exists() and str(ROOT / "bin" / "ffmpeg"))
              or shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg")
EXIFTOOL_BIN = os.environ.get("EXIFTOOL_BIN") or shutil.which("exiftool") or ""
FACE_MODEL_DIR = Path(os.environ.get(
    "FACE_MODEL_DIR", str(ROOT / "baked_models")))
# 人脸/SigLIP 子进程解释器：默认用当前进程解释器（容器内即容器 Python；Mac 上即服务 venv）
FACES_PYTHON = os.environ.get("FACES_PYTHON") or sys.executable
SIGLIP_PYTHON = os.environ.get("SIGLIP_PYTHON") or sys.executable


def _py_cmd(worker_key, *args):
    """后台任务子进程统一拉起入口（2026-09-10 修 .app 启动锁死配套）。
    frozen（Mac .app）下 sys.executable 是 FamilyMemoryAlbum 二进制而非 python，
    `python xxx.py` 式拉起会经 launcher 主服务分支再起一个完整服务实例
    （端口冲突崩溃 + 多实例互抢 SQLite 写锁）。统一改走 launcher 的
    --ff-worker runpy 分发；非 frozen 直跑等价 `python xxx.py args`。"""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--ff-worker", worker_key, *args]
    return [sys.executable, str(WORKER_SCRIPTS[worker_key]), *args]


# launcher --ff-worker 分发表（与 launcher.py 的 scripts 字典保持一致）
WORKER_SCRIPTS = {k: str(ROOT / v) for k, v in {
    "faces": "backfill_faces_full.py",
    "siglip": "backfill_siglip_embedding.py",
    "captions": "caption_days.py",
    "manual_face": "manual_face_create.py",
    "vlm": "vlm_describe_assets.py",
    "privacy": "privacy_auto_scan.py",
    # 2026-09-16 新增：导入后增量富化（精确去重/画质/相似分组/择优/语义过滤）
    "enrich": "enrich.py",
    # 2026-09-24 新增：Log 视频播放转码（D-Log 4K HEVC 10bit → H.264 8bit）
    "videolc": "transcode_log_videos.py",
}.items()}

# 全局后台重任务互斥锁：autoscan / vlm-describe / privacy-scan 任意时刻只允许
# 一个在跑（2026-09-10：三路并发各自长持 SQLite 写锁 → 主线程查询全部超时）
_BG_TASK_LOCK = threading.Lock()
# 历史遗留的默认源候选（已无引用，产品扫描走 source 表）；env 冒号分隔可选注入
MEDIA_ROOTS = [p for p in os.environ.get("MEDIA_ROOTS", "").split(os.pathsep) if p.strip()]

# ---- SigLIP 语义检索模型（算法线 A2/A4：SigLIP v1 → SigLIP2，中文检索 A/B 胜出）----
# A/B 实测 Recall@10 0.733→0.792（eval/ab_siglip2_*.md）。权重在本地 models/ 目录，
# 不依赖 hub 缓存布局；env 可覆盖。历史 v1 向量按 model_name 留库不删，可回滚。
SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"
SIGLIP_MODEL_DIR = os.environ.get("FF_SIGLIP2_DIR") or str(
    Path(__file__).resolve().parent.parent / "models" / "siglip2-base-patch16-224")


# ---- 资产缓存（缩略图/预览）清理：一律移入 ~/.Trash，可恢复，原片不受影响 ----
def _trash_cache_dir():
    d = Path.home() / ".Trash" / "family-memory-cache"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _purge_asset_cache(asset_ids):
    """把指定资产的缩略图/预览缓存移入废纸篓。返回移除的文件数。"""
    moved = 0
    if not asset_ids:
        return moved
    trash = _trash_cache_dir()
    for aid in asset_ids:
        token = aid[6:] if aid.startswith("asset_") else aid
        for base_dir in (THUMB_DIR, PREVIEW_DIR):
            if not base_dir.is_dir():
                continue
            for p in base_dir.glob(f"{token}*.jpg"):
                try:
                    shutil.move(str(p), str(trash / p.name))
                    moved += 1
                except Exception:
                    pass
    return moved


def _cleanup_orphan_cache():
    """启动后台清理：库里已不存在的资产残留的缓存文件（如移除来源后遗留的缩略图）。"""
    try:
        con = sqlite3.connect(DB, timeout=30)
        assets = {r[0] for r in con.execute("SELECT asset_id FROM media_asset")}
        con.close()
    except Exception as exc:
        print(f"[cache-cleanup] 读取数据库失败，跳过: {exc}", flush=True)
        return
    orphan_ids = set()
    for base_dir in (THUMB_DIR, PREVIEW_DIR):
        if not base_dir.is_dir():
            continue
        for p in base_dir.iterdir():
            name = p.name
            if not name.endswith(".jpg"):
                continue
            token = name[:-4]
            # 2026-09-24 修：Log 还原版缓存名带 _lc<tag> 后缀（如 xxx_t480_lcv1s100.jpg、
            # xxx_lcv1s100.jpg）。旧解析只认 _t<档位>，遇到 _lc 后缀就还原不出 asset_id
            # → 整批被当成「库里已不存在的孤儿」清掉（实测每次重启清掉 69 张还原缩略图，
            # 用户每重启一次就要重抽一遍 4K 帧）。先剥 _lc 后缀再剥档位后缀。
            token = re.sub(r"_lc[0-9A-Za-z]*$", "", token)
            m = re.match(r"^(.*?)_t\d+$", token)
            if m:
                token = m.group(1)
            if f"asset_{token}" not in assets:
                orphan_ids.add(f"asset_{token}")
    moved = _purge_asset_cache(orphan_ids)
    if moved:
        print(f"[cache-cleanup] 清理孤儿缓存 {moved} 个文件 → ~/.Trash/family-memory-cache/", flush=True)


# API 凭据（服务进程的 python 可能没有 yaml 模块，做正则兜底解析，避免 API_KEY 静默为空）
CRED_FILE = Path.home() / ".dsh" / ".credentials.yaml"
API_KEY = ""
if CRED_FILE.exists():
    try:
        import yaml
        API_KEY = yaml.safe_load(CRED_FILE.read_text()).get("DEEPSEEK_API_KEY", "")
    except Exception:
        pass
    if not API_KEY:
        m = re.search(r"DEEPSEEK_API_KEY:\s*['\"]?([A-Za-z0-9_\-]+)", CRED_FILE.read_text())
        API_KEY = m.group(1) if m else ""

# LLM 端点：本地 Ollama (qwen2.5-vl) 优先，DeepSeek 云端回退
# 2026-08-30 DeepSeek 余额 -0.91 欠费停服，切换本地免费模型；充值后无需改代码，
# Ollama 不在时自动回退 DeepSeek。
OLLAMA_BASE = "http://127.0.0.1:11434/v1"
LLM_LOCAL_VISION = "qwen2.5vl:7b"   # 视觉/图文任务
LLM_LOCAL_TEXT = "qwen2.5vl:7b"     # 纯文本任务（7B 本地文本质量足够意图解析/综合回答）
DS_VISION_MODEL = "deepseek-v4-flash-vision-exp"
DS_TEXT_MODEL = "deepseek-v4-flash"


# 直连 opener：绕过系统代理（shell/launchd 环境可能带 HTTP_PROXY，
# 本地 127.0.0.1 请求走代理会被 502）
_DIRECT_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _ollama_up():
    """本地 Ollama 是否在跑（localhost 直连，失败是即时的，不会拖慢调用）。"""
    try:
        with _DIRECT_OPENER.open(f"{OLLAMA_BASE}/models", timeout=3):
            return True
    except Exception:
        return False


# ============ 自定义 API 服务注册表（2026-09-07 models.html 重设计） ============
# 用户可在「模型与任务」页自选添加免费/付费 OpenAI 兼容 API，与本地 Ollama、
# 内置 DeepSeek 组成调用链；优先级（本地优先/云端优先）存 settings。

_OLLAMA_PULLS = {}  # model -> {"running": bool, "started_at": str, ...}


def _ensure_api_provider_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS api_providers_v0 (
        provider_id TEXT PRIMARY KEY, name TEXT NOT NULL, base_url TEXT NOT NULL,
        api_key TEXT DEFAULT '', model TEXT NOT NULL, vision INTEGER DEFAULT 1,
        free INTEGER DEFAULT 0, enabled INTEGER DEFAULT 1, status TEXT DEFAULT '',
        created_at TEXT NOT NULL)""")
    try:
        con.execute("ALTER TABLE api_providers_v0 ADD COLUMN sort_order INTEGER DEFAULT 0")
    except Exception:
        pass  # 列已存在
    con.commit()


def _provider_list(mask=True):
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    con.row_factory = sqlite3.Row
    _ensure_api_provider_table(con)
    rows = con.execute("SELECT * FROM api_providers_v0 ORDER BY sort_order, rowid").fetchall()
    con.close()
    out = []
    for r in rows:
        d = dict(r)
        k = d.get("api_key") or ""
        d["api_key_masked"] = (k[:4] + "****" + k[-4:]) if k else ""
        d.pop("api_key", None)
        d["enabled"] = bool(d["enabled"])
        d["free"] = bool(d["free"])
        d["vision"] = bool(d["vision"])
        out.append(d)
    return out


def _enabled_providers(vision=None):
    """启用的自定义 API（按添加顺序）。vision=True 时只要支持视觉的。"""
    try:
        con = sqlite3.connect(DB, timeout=10)
        con.execute("PRAGMA busy_timeout=10000")
        con.row_factory = sqlite3.Row
        _ensure_api_provider_table(con)
        q = "SELECT * FROM api_providers_v0 WHERE enabled=1"
        if vision:
            q += " AND vision=1"
        rows = con.execute(q + " ORDER BY sort_order, rowid").fetchall()
        con.close()
        out = []
        for r in rows:
            d = dict(r)
            # 默认种子的 DeepSeek 服务 Key 为空时回退内置 Key（本机/内置配置兜底）
            if not (d.get("api_key") or "").strip() and "deepseek" in (d.get("base_url") or ""):
                d["api_key"] = get_setting("deepseek_api_key", "") or API_KEY
            out.append(d)
        return out
    except Exception:
        return []


def _call_openai_compat(base_url, api_key, model, messages, max_tokens=64, timeout=30):
    """调 OpenAI 兼容服务。base_url 容错：…/v1 与 …/v1/chat/completions 都接受。"""
    url = base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        if not url.endswith("/v1"):
            url += "/v1"
        url += "/chat/completions"
    body = {"model": model, "messages": messages, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    msg = data["choices"][0]["message"]
    # 思考型模型(如 qwen3/deepseek-r1)会把 token 花在 reasoning 上, content 可能为空
    text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
    return text, round(time.time() - t0, 2)


def llm_chat(messages, vision=False, temperature=None, max_tokens=1024):
    """统一 LLM 调用入口。返回 (text, error)。
    error ∈ {None, 'no_balance', 'no_key', 'error'}；视觉消息用 OpenAI content 数组。

    2026-09-07 v2：新增自定义 API 服务注册表（models.html 可增删启停，免费/付费自选）。
    优先级由 settings llm_priority 控制：
      local_first（默认）: 本地 Ollama → 启用的自定义 API → DeepSeek(内置)
      api_first:          启用的自定义 API → DeepSeek(内置) → 本地 Ollama"""
    priority = get_setting("llm_priority", "local_first")
    customs = _enabled_providers(vision=True) if vision else _enabled_providers()

    def try_ollama():
        if not _ollama_up():
            return None, "skip"
        try:
            o_model = get_setting("ollama_vision_model" if vision else "ollama_text_model", "")
            body = {
                "model": o_model or (LLM_LOCAL_VISION if vision else LLM_LOCAL_TEXT),
                "messages": messages,
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                body["temperature"] = temperature
            req = urllib.request.Request(
                f"{OLLAMA_BASE}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with _DIRECT_OPENER.open(req, timeout=180) as resp:
                data = json.loads(resp.read())
            text = (data["choices"][0]["message"].get("content") or "").strip()
            return (text, None) if text else (None, "error")
        except Exception:
            return None, "error"

    def try_custom(p):
        try:
            text, _ = _call_openai_compat(p["base_url"], p["api_key"], p["model"],
                                          messages, max_tokens=max_tokens, timeout=120)
            return (text, None) if text else (None, "error")
        except urllib.error.HTTPError as e:
            return None, ("no_balance" if e.code in (401, 402, 403) else "error")
        except Exception:
            return None, "error"

    def try_deepseek():
        key = get_setting("deepseek_api_key", "") or API_KEY
        if not key:
            return None, "no_key"
        ds_model = get_setting("ds_vision_model" if vision else "ds_text_model", "")
        body = {
            "model": ds_model or (DS_VISION_MODEL if vision else DS_TEXT_MODEL),
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            msg = data["choices"][0]["message"]
            text = (msg.get("content") or msg.get("reasoning_content") or "").strip()
            return (text or None), None
        except urllib.error.HTTPError as e:
            return None, ("no_balance" if e.code in (401, 402, 403) else "error")
        except Exception:
            return None, "error"

    custom_steps = [("custom:" + p["name"],
                     (lambda pp: lambda: try_custom(pp))(p)) for p in customs]
    if priority == "api_first":
        chain = custom_steps + [try_deepseek, try_ollama]
    else:
        chain = [try_ollama] + custom_steps + [try_deepseek]
    last_err = "error"
    for item in chain:
        # 2026-09-19 修复：custom_steps 是 (标签, 可调用) 元组，与裸函数混在一条链里，
        # 直接 fn() 会 'tuple' object is not callable → 配置了自定义 API 后 llm_chat 必崩
        fn = item[1] if isinstance(item, tuple) else item
        text, err = fn()
        if text:
            return text, None
        if err and err != "skip":
            last_err = err
    return None, last_err


# 地区词表缓存：由 asset_geo_v0 的实际地区动态生成（fallback_parse 首次调用时填充）
_GEO_WORDS = None
# 人名词典（#16 隐私清洗）：家庭人名/别名/称呼全部数据化，代码零人名。
# 数据源 = person 表（confirmed 成员 + relationship_label 称呼）
#        + person_alias 表（ASR/拼音/口语变体，管理员可增删）。
# 进程内缓存 60s；update_people 改动人物后立即失效。
_PERSON_LEXICON = {"ts": 0.0, "family": [], "name_aliases": {}, "roles": {}, "children": []}


def _invalidate_person_lexicon():
    _PERSON_LEXICON["ts"] = 0.0


def _person_lexicon():
    now = time.time()
    if now - _PERSON_LEXICON["ts"] < 60 and _PERSON_LEXICON["family"] is not None:
        return _PERSON_LEXICON
    family, roles, children = [], {}, []
    name_aliases, source_aliases = {}, {}
    try:
        con = sqlite3.connect(DB, timeout=10)
        con.row_factory = sqlite3.Row
        for r in con.execute(
            """SELECT display_name, relationship_label, birth_date FROM person
               WHERE identity_status='confirmed'"""
        ):
            if r["display_name"]:
                family.append(r["display_name"])
                if r["relationship_label"]:
                    roles[r["relationship_label"]] = r["display_name"]
        # 孩子 = 出生日期在 12 岁以内的 confirmed 人物（按出生升序，大在前）
        for r in con.execute(
            """SELECT display_name FROM person
               WHERE identity_status='confirmed' AND birth_date IS NOT NULL
                 AND (julianday('now') - julianday(birth_date)) / 365.25 < 12
               ORDER BY birth_date"""
        ):
            children.append(r["display_name"])
        for r in con.execute("SELECT alias, canonical, kind FROM person_alias"):
            if r["kind"] == "source":
                source_aliases[r["alias"]] = r["canonical"]
            else:
                name_aliases.setdefault(r["canonical"], []).append(r["alias"])
        con.close()
    except sqlite3.Error:
        pass  # 空库/新装：词典为空，问答人物增强静默降级
    _PERSON_LEXICON.update(ts=now, family=family, name_aliases=name_aliases,
                           roles=roles, children=children, source_aliases=source_aliases)
    return _PERSON_LEXICON

MEDIA_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".heic", ".heif", ".avif", ".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
SCAN_LOCK = threading.Lock()

# 后台扫描任务状态（内存），供前端轮询进度
SCAN_JOBS = {}

# ------------------------------------------------------------------
# 新文件核心元数据提取：拍摄时间 + GPS（数值）
# 导入硬约束：任何新入库媒体必须尽力保留 EXIF 的 DateTimeOriginal 与
# GPS 坐标；文件名正则只做兜底，位置不允许靠猜。
# ------------------------------------------------------------------
_EXIFTOOL_BIN = None

def _resolve_exiftool():
    # 2026-09-10 决策点①：env > 安装包内置（ROOT/bin/exiftool，官方可移植发行版
    # exiftool 脚本+lib/，依赖系统 /usr/bin/perl）> PATH > homebrew
    for cand in (os.environ.get("EXIFTOOL_BIN"),
                 (ROOT / "bin" / "exiftool").exists() and str(ROOT / "bin" / "exiftool"),
                 shutil.which("exiftool"),
                 "/opt/homebrew/bin/exiftool",
                 "/usr/local/bin/exiftool"):
        if cand and os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None

def _normalize_exif_time(value):
    """exiftool -n 时间 '2026:08:27 12:15:13[+08:00]' → ISO 8601。失败返回 None。"""
    if not value or not isinstance(value, str):
        return None
    m = re.match(
        r"(\d{4}):(\d{2}):(\d{2}) (\d{2}):(\d{2}):(\d{2})(?:\.\d+)?([+-]\d{2}:?\d{2})?$",
        value.strip())
    if not m:
        return None
    y, mo, d, h, mi, s, tz = m.groups()
    if tz:
        tz = tz[:3] + ":" + tz[3:].replace(":", "").ljust(2, "0")
    else:
        tz = "+08:00"  # 家人手机均在国内拍摄，无时区标记时按东八区
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}{tz}"

def _extract_core_meta(paths):
    """批量提取拍摄时间与 GPS。返回 {path: {dt, lat, lon}}。
    exiftool 批内有坏文件时 returncode != 0，但 stdout 仍含有效 JSON，不能整批丢弃。"""
    global _EXIFTOOL_BIN
    result = {p: {"dt": None, "lat": None, "lon": None} for p in paths}
    if _EXIFTOOL_BIN is None:
        _EXIFTOOL_BIN = _resolve_exiftool()
    if not _EXIFTOOL_BIN or not paths:
        return result
    BATCH = 400
    for i in range(0, len(paths), BATCH):
        batch = paths[i:i + BATCH]
        fd, listfile = tempfile.mkstemp(suffix=".txt", prefix="fm_exif_")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(batch) + "\n")
            proc = subprocess.run(
                [_EXIFTOOL_BIN, "-json", "-n", "-fast", "-api", "QuickTimeUTC=1",
                 "-DateTimeOriginal", "-CreateDate",
                 "-GPSLatitude", "-GPSLongitude", "-@", listfile],
                capture_output=True, text=True, timeout=300)
            if not proc.stdout.strip():
                continue
            try:
                items = json.loads(proc.stdout)
            except ValueError:
                continue
            for item in items:
                src = item.get("SourceFile")
                if src not in result:
                    continue
                dt = item.get("DateTimeOriginal") or item.get("CreateDate")
                result[src]["dt"] = _normalize_exif_time(dt)
                for key, field in (("GPSLatitude", "lat"), ("GPSLongitude", "lon")):
                    try:
                        result[src][field] = float(item.get(key))
                    except (TypeError, ValueError):
                        result[src][field] = None
        except Exception:
            pass
        finally:
            try:
                os.unlink(listfile)
            except OSError:
                pass
    return result


def _check_new_files(roots_override=None, progress=None):
    """扫描并增量写入数据库索引；NAS 原片始终只读。roots_override 可只扫指定来源根目录。
    progress 为可选 dict，用于前端轮询（会被就地更新）。"""
    con = sqlite3.connect(DB, timeout=60)
    con.execute("PRAGMA busy_timeout=60000")
    con.row_factory = sqlite3.Row
    _ensure_source_schema(con)
    if roots_override is not None:
        roots = list(roots_override)
    else:
        roots = [r[0] for r in con.execute(
            "SELECT root_path FROM source WHERE enabled=1 ORDER BY root_path")]
    indexed = {r[0] for r in con.execute("SELECT absolute_path FROM media_file")}
    scanned = 0
    new_files = 0
    indexed_new = 0
    failed = []
    new_paths = []
    new_meta = []
    by_root = {}
    if progress is not None:
        progress["status"] = "scanning"
        progress["message"] = "正在遍历文件..."
    for root in roots:
        root_count = root_new = 0
        if not os.path.isdir(root):
            by_root[root] = {"scanned": 0, "new_files": 0, "available": False}
            if progress is not None:
                progress["message"] = f"目录不可访问：{root}"
            continue
        for base, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in {"@eaDir", "#recycle"} and not d.startswith(".")]
            for name in files:
                if Path(name).suffix.lower() not in MEDIA_EXTS:
                    continue
                path = os.path.join(base, name)
                scanned += 1
                root_count += 1
                if path not in indexed:
                    new_files += 1
                    root_new += 1
                    new_paths.append(path)
                if progress is not None and scanned % 50 == 0:
                    progress["scanned"] = scanned
                    progress["new_files"] = new_files
                    progress["message"] = f"已扫描 {scanned} 个文件，发现 {new_files} 个新文件"
        by_root[root] = {"scanned": root_count, "new_files": root_new, "available": True}
    sources = [dict(r) for r in con.execute("SELECT source_id,root_path FROM source ORDER BY length(root_path) DESC")]
    stamp = now_iso()
    total_new = len(new_paths)
    failures_by_root = {}
    # 导入硬约束：新文件先批量提取 EXIF 拍摄时间与 GPS，再入库
    meta_by_path = _extract_core_meta(new_paths) if total_new else {}
    geo_new = 0
    if progress is not None:
        progress["scanned"] = scanned
        progress["total_files"] = scanned
        progress["new_files"] = new_files
        progress["total_new"] = total_new
        progress["message"] = f"扫描完成，正在入库 {total_new} 个新文件..."
    for idx, path in enumerate(new_paths):
        try:
            source = next((s for s in sources if path == s["root_path"] or path.startswith(s["root_path"] + os.sep)), None)
            if not source:
                failed.append({"path": path, "reason": "NO_SOURCE"})
                if progress is not None:
                    progress["failed"] = len(failed)
                src_root = next((s for s in sources if path == s["root_path"] or path.startswith(s["root_path"] + os.sep)), None)
                if src_root:
                    failures_by_root[src_root["root_path"]] = failures_by_root.get(src_root["root_path"], 0) + 1
                continue
            st = os.stat(path)
            token = hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
            asset_id, file_id = "asset_" + token, "file_" + token
            ext = Path(path).suffix.lower()
            media_type = "video" if ext in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"} else "photo"
            # 拍摄时间：EXIF DateTimeOriginal 优先，文件名正则兜底
            em = meta_by_path.get(path) or {}
            capture_time = em.get("dt")
            time_conf = 0.98 if capture_time else 0.0
            if not capture_time:
                m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[^0-9]?(\d{2})[-_:]?(\d{2})[-_:]?(\d{2})", Path(path).stem)
                if m:
                    capture_time = f"{m.group(1)}-{m.group(2)}-{m.group(3)}T{m.group(4)}:{m.group(5)}:{m.group(6)}+08:00"
                    time_conf = 0.75
            # GPS：数值坐标，无效则置空
            lat, lon = em.get("lat"), em.get("lon")
            if lat is None or lon is None or (lat == 0 and lon == 0):
                lat = lon = None
            else:
                geo_new += 1
            con.execute("""INSERT OR IGNORE INTO media_asset
                (asset_id,family_id,media_type,capture_time,time_precision,time_confidence,
                 latitude,longitude,location_confidence,is_original,privacy_level,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (asset_id,"family_default",media_type,capture_time,"second" if capture_time else "unknown",time_conf,
                 lat,lon,1.0 if lat is not None else None,1,"private",stamp,stamp))
            con.execute("""INSERT OR IGNORE INTO media_file
                (file_id,asset_id,source_id,absolute_path,relative_path,filename,extension,byte_size,filesystem_mtime,mime_type,variant_kind,availability,indexed_at)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (file_id,asset_id,source["source_id"],path,os.path.relpath(path,source["root_path"]),Path(path).name,ext,st.st_size,datetime.fromtimestamp(st.st_mtime,timezone.utc).isoformat(),mimetypes.guess_type(path)[0],"original","online",stamp))
            new_meta.append((asset_id, Path(path).name, os.path.relpath(path, source["root_path"])))
            indexed_new += 1
            # 2026-09-10 修 .app 启动锁死：原来整个循环 INSERT 完才 commit 一次，
            # 1.3 万文件经 SMB 逐个入库时 SQLite 写锁被持有几十分钟，主线程
            # 全部查询 busy 超时 → 服务瘫痪。改为每 200 个文件提交一批，
            # 写锁持有时间压到毫秒级；中途崩溃也不丢已入库部分。
            if indexed_new % 200 == 0:
                con.commit()
            if progress is not None and (indexed_new % 20 == 0 or idx == total_new - 1):
                progress["indexed"] = indexed_new
                progress["message"] = f"已入库 {indexed_new}/{total_new} 个新文件"
        except Exception as exc:
            failed.append({"path": path, "reason": str(exc)})
            if progress is not None:
                progress["failed"] = len(failed)
            src_root = next((s for s in sources if path == s["root_path"] or path.startswith(s["root_path"] + os.sep)), None)
            if src_root:
                failures_by_root[src_root["root_path"]] = failures_by_root.get(src_root["root_path"], 0) + 1
    for root in roots:
        try:
            info = by_root.get(root, {})
            total_files = info.get("scanned", 0)
            failed_count = failures_by_root.get(root, 0)
            row = con.execute("""SELECT source_id FROM source WHERE root_path=?""", (root,)).fetchone()
            if row:
                source_id = row[0]
                indexed_count = con.execute(
                    "SELECT COUNT(*) FROM media_file WHERE source_id=?", (source_id,)).fetchone()[0]
                con.execute("""UPDATE source SET last_scan_at=?, total_files=?, indexed_count=?, failed_count=?
                               WHERE root_path=?""",
                            (stamp, total_files, indexed_count, failed_count, root))
        except Exception:
            pass
    con.commit()
    total_assets = con.execute("SELECT COUNT(*) FROM media_asset").fetchone()[0]
    con.close()
    if geo_new > 0:
        # 有新 GPS 入库：自动重建地理标签，保证地点立即可查
        try:
            rebuild_geo()
        except Exception:
            pass
        # 同步刷新多标签（旅行/山景等）——SIGLIP 全库一次约 10s，后台跑不阻塞扫描返回
        try:
            threading.Thread(target=rebuild_scene_tags, daemon=True).start()
        except Exception:
            pass
    if indexed_new > 0:
        # 新照片入库：自动跑人脸检测+身份归属（幂等脚本只处理未处理资产）
        try:
            threading.Thread(target=faces_backfill_run, daemon=True).start()
        except Exception:
            pass
        # 同时自动补 SIGLIP 图像向量（缩略图就绪后才有意义；脚本自身幂等可重跑）
        try:
            threading.Thread(target=siglip_backfill_run, daemon=True).start()
        except Exception:
            pass
        # 导入即确定性垃圾过滤（文件名/路径规则，零图片解码，低误杀；不自动隐藏需 VLM 语义判断的"拍屏幕"照）
        if indexed_new > 0 and new_meta:
            try:
                _junk = _classify_import_junk(new_meta)
                if _junk:
                    filter_add(_junk)
            except Exception as _e:
                print(f"[import-junk] 导入过滤判定失败(忽略): {_e}", flush=True)
        # 2026-09-16 新增：导入后增量富化（精确去重 sha256 / 画质 / 相似分组 / 择优 / 语义过滤）。
        # 此前这些环节导入后从没跑过，导致「相似照片」「已过滤内容」对新照片完全失效。
        # 幂等且只处理欠账，重复触发无损；由 _BG_TASK_LOCK 与 autoscan 串行。
        if indexed_new > 0:
            try:
                threading.Thread(target=enrich_run, kwargs={"trigger": "import"}, daemon=True).start()
            except Exception as _e:
                print(f"[enrich] 导入后触发失败(忽略): {_e}", flush=True)
    result = {"scanned": scanned, "new_files": new_files, "indexed": indexed_new, "failed": failed, "total_assets": total_assets, "by_root": by_root, "read_only": True, "geo_new": geo_new}
    if progress is not None:
        progress.update({"status": "done", "done": True, "scanned": scanned, "new_files": new_files,
                         "indexed": indexed_new, "failed": len(failed), "result": result,
                         "message": f"完成：扫描 {scanned} 个文件，新增入库 {indexed_new} 项" + (f"，失败 {len(failed)} 项" if failed else "")})
    return result


def check_new_files(roots_override=None, progress=None):
    """同一时间只允许一个增量扫描，避免多窗口争用 SQLite。"""
    with SCAN_LOCK:
        return _check_new_files(roots_override, progress=progress)


# ============ 数据来源管理 ============
CLOUD_CANDIDATE_DIRS = [
    ("iCloud 云盘", os.path.expanduser("~/Library/Mobile Documents/com~apple~CloudDocs")),
    ("百度网盘下载", os.path.expanduser("~/BaiduNetdiskDownload")),
    ("百度网盘同步", os.path.expanduser("~/BaiduSync")),
    ("Dropbox", os.path.expanduser("~/Dropbox")),
    ("OneDrive", os.path.expanduser("~/OneDrive")),
    ("Google Drive", os.path.expanduser("~/Google Drive")),
    ("坚果云", os.path.expanduser("~/Nutstore")),
    ("本机图片", os.path.expanduser("~/Pictures")),
    ("本机下载", os.path.expanduser("~/Downloads")),
]
DISCOVER_SKIP_NAMES = {"Macintosh HD", "Recovery", "VMware Shared Folders", ".timemachine"}


def _ensure_source_schema(con):
    """source 表补充 enabled/total_files/indexed_count/failed_count 列 + 应用设置表。"""
    try:
        con.execute("ALTER TABLE source ADD COLUMN enabled INTEGER NOT NULL DEFAULT 1")
    except Exception:
        pass  # 列已存在
    for col in ("total_files", "indexed_count", "failed_count"):
        try:
            con.execute(f"ALTER TABLE source ADD COLUMN {col} INTEGER NOT NULL DEFAULT 0")
        except Exception:
            pass  # 列已存在
    try:
        # 用实际 media_file 数量回填 indexed_count，避免旧数据全部显示 0/0
        con.execute("""UPDATE source SET indexed_count = COALESCE((SELECT COUNT(*) FROM media_file mf WHERE mf.source_id = source.source_id), 0)
                       WHERE COALESCE(indexed_count, 0) = 0""")
        # 在拿到真实总数前，先把 total_files 回退为 indexed_count，避免 0/0
        con.execute("""UPDATE source SET total_files = indexed_count
                       WHERE COALESCE(total_files, 0) = 0 AND indexed_count > 0""")
    except Exception:
        pass
    con.execute("""CREATE TABLE IF NOT EXISTS app_setting_v0 (
        key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)""")
    con.commit()


def get_setting(key, default):
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    _ensure_source_schema(con)
    row = con.execute("SELECT value FROM app_setting_v0 WHERE key=?", (key,)).fetchone()
    con.close()
    return row[0] if row else default


def set_setting(key, value):
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    _ensure_source_schema(con)
    con.execute("INSERT OR REPLACE INTO app_setting_v0 (key, value, updated_at) VALUES (?,?,?)",
                (key, str(value), now_iso()))
    con.commit()
    con.close()


def autoscan_config(enabled=None, interval_min=None):
    """读取/更新自动扫描配置。"""
    if enabled is not None:
        set_setting("autoscan_enabled", "1" if enabled else "0")
    if interval_min is not None:
        set_setting("autoscan_interval_min", str(max(15, int(interval_min))))
    try:
        interval = int(get_setting("autoscan_interval_min", "120"))
    except Exception:
        interval = 120
    return {
        "enabled": get_setting("autoscan_enabled", "1") == "1",
        "interval_min": max(15, interval),
        "last_scan_at": get_setting("autoscan_last_scan_at", ""),
        "last_scan_indexed": get_setting("autoscan_last_scan_indexed", ""),
    }


def source_list():
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    con.row_factory = sqlite3.Row
    _ensure_source_schema(con)
    rows = con.execute("""
        SELECT s.source_id, s.owner_label, s.root_path, s.source_type, s.last_scan_at, s.enabled,
               s.total_files, s.indexed_count, s.failed_count,
               (SELECT COUNT(*) FROM media_file mf WHERE mf.source_id = s.source_id) AS file_count
        FROM source s ORDER BY indexed_count DESC""").fetchall()
    con.close()
    return {"sources": [{
        "source_id": r["source_id"],
        "label": r["owner_label"] or os.path.basename(r["root_path"]) or r["root_path"],
        "root_path": r["root_path"],
        "source_type": r["source_type"],
        "last_scan_at": r["last_scan_at"],
        "enabled": bool(r["enabled"]),
        "file_count": r["file_count"],
        "total_files": r["total_files"] or 0,
        "indexed_count": r["indexed_count"] or 0,
        "failed_count": r["failed_count"] or 0,
        "available": os.path.isdir(r["root_path"]),
        "healthy": bool((r["indexed_count"] or 0) > 0),
    } for r in rows]}


def _count_media_files(root, max_depth=3, cap=4000):
    """深度受限地统计目录里的影像文件数（发现候选用，不做全量 walk）。跳过 .app 应用包。"""
    count = 0
    stack = [(root, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    name = entry.name
                    if name.startswith(".") or name in {"@eaDir", "#recycle"}:
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if depth + 1 <= max_depth and not name.endswith(".app") and name != "Applications":
                            stack.append((entry.path, depth + 1))
                    elif Path(name).suffix.lower() in MEDIA_EXTS:
                        count += 1
                        if count >= cap:
                            return count
        except Exception:
            continue
    return count


def _is_app_volume(vp):
    """判断挂载卷是否为 DMG 安装卷（顶层只有 .app 应用包）。"""
    try:
        entries = [e for e in os.listdir(vp) if not e.startswith(".")]
    except Exception:
        return False
    if not entries:
        return False
    return all(e.endswith(".app") or e == "Applications" for e in entries)


def _start_scan_job(source_id, root_path):
    """启动后台扫描线程，返回 scan_id；前端用 /api/source/scan_status 轮询。"""
    scan_id = f"scan_{source_id}_{int(time.time() * 1000)}"
    progress = {
        "scan_id": scan_id, "source_id": source_id, "root_path": root_path,
        "status": "pending", "done": False,
        "scanned": 0, "new_files": 0, "indexed": 0, "failed": 0,
        "message": "等待扫描锁...", "result": None, "error": None,
    }
    SCAN_JOBS[scan_id] = progress
    def run():
        try:
            progress["status"] = "scanning"
            progress["message"] = "开始扫描..."
            check_new_files([root_path], progress=progress)
        except Exception as exc:
            progress["status"] = "error"
            progress["error"] = str(exc)
            progress["message"] = f"扫描异常：{exc}"
            progress["done"] = True
    threading.Thread(target=run, daemon=True).start()
    return scan_id


def source_add(path, label="", source_type="local_folder", auto_scan=True):
    """添加一个数据来源目录；校验通过后立即落库并启动后台扫描，接口快速返回 scan_id。"""
    raw_path = path.strip().rstrip("/")
    path = os.path.expanduser(raw_path)
    # macOS 文件系统不区分大小写：/volumes 与 /Volumes 是同一目录，但字符串不同会让库内路径分裂成两套
    if path.startswith("/volumes/"):
        path = "/Volumes/" + path[len("/volumes/"):]
    path = os.path.abspath(path)
    if not os.path.isdir(path):
        hint = ""
        if raw_path.lower().startswith("/volume1/"):
            hint = "（这是群晖内部路径，Mac 上请改用挂载路径 /Volumes/...）"
        return {"error": f"目录不存在或无法访问：{path}{hint}"}
    con = sqlite3.connect(DB, timeout=60)
    con.execute("PRAGMA busy_timeout=60000")
    con.row_factory = sqlite3.Row
    _ensure_source_schema(con)
    roots = [r[0] for r in con.execute("SELECT root_path FROM source")]
    con.close()
    pl = path.lower()
    for r in roots:
        rl = r.lower()
        if pl == rl:
            return {"error": "该目录已经是数据来源"}
        if pl.startswith(rl + os.sep):
            return {"error": f"目录已被现有来源覆盖：{r}"}
        if rl.startswith(pl + os.sep):
            return {"error": f"该目录是现有来源 {r} 的上级目录，请先在来源管理中移除旧来源再添加"}
    found = _count_media_files(path)
    if found == 0:
        return {"error": "该目录（3 层深度内）没有找到照片或视频文件"}
    source_id = "src_" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:24]
    con = sqlite3.connect(DB, timeout=60)
    con.execute("PRAGMA busy_timeout=60000")
    con.execute("""INSERT INTO source (source_id, family_id, owner_label, root_path, source_type, read_only, last_scan_at, enabled, total_files, indexed_count, failed_count)
        VALUES (?,?,?,?,?,1,?,1,?,?,?)""",
        (source_id, "family_default", (label or "").strip() or os.path.basename(path) or path,
         path, source_type, now_iso(), found, 0, 0))
    con.commit()
    con.close()
    scan_id = _start_scan_job(source_id, path) if auto_scan else None
    return {"source_id": source_id, "root_path": path, "label": (label or "").strip() or os.path.basename(path),
            "media_files_found": found, "scan_id": scan_id, "status": "scanning" if scan_id else "added"}


def source_scan(source_id):
    """只扫描某一个来源根目录，立即返回 scan_id。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT root_path FROM source WHERE source_id=?", (source_id,)).fetchone()
    con.close()
    if not row:
        return {"error": "来源不存在"}
    scan_id = _start_scan_job(source_id, row["root_path"])
    return {"scan_id": scan_id, "status": "scanning"}


def source_scan_status(scan_id):
    """返回扫描任务当前状态。"""
    job = SCAN_JOBS.get(scan_id)
    if not job:
        return {"error": "扫描任务不存在或已过期"}
    return dict(job)


def source_scanning():
    """返回当前进行中的扫描任务（按 source_id 索引），供来源卡片实时进度条使用。"""
    active = {}
    cutoff = time.time() - 3600
    # 2026-09-02: 清理已完成/失败超过 1 小时的 job, 防 SCAN_JOBS 无限增长
    for scan_id, job in list(SCAN_JOBS.items()):
        if not (job.get("done") or job.get("status") == "error"):
            continue
        ts = 0
        if "_" in scan_id:
            try:
                ts = int(scan_id.rsplit("_", 1)[-1]) / 1000
            except Exception:
                pass
        if not ts or ts < cutoff:
            SCAN_JOBS.pop(scan_id, None)
    for scan_id, job in list(SCAN_JOBS.items()):
        if job.get("done") or job.get("error") or job.get("status") == "error":
            continue
        ts = 0
        if "_" in scan_id:
            try:
                ts = int(scan_id.rsplit("_", 1)[-1]) / 1000
            except Exception:
                pass
        if ts and ts < cutoff:
            continue
        sid = job.get("source_id")
        if not sid:
            continue
        active[sid] = {
            "scanned": job.get("scanned", 0),
            "total_files": job.get("total_files", job.get("scanned", 0)),
            "indexed": job.get("indexed", 0),
            "failed": job.get("failed", 0),
            "message": job.get("message", "扫描中..."),
        }
    return {"active": active}


def source_set_enabled(source_id, enabled):
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    cur = con.execute("UPDATE source SET enabled=? WHERE source_id=?", (1 if enabled else 0, source_id))
    con.commit()
    con.close()
    if cur.rowcount == 0:
        return {"error": "来源不存在"}
    return {"source_id": source_id, "enabled": bool(enabled)}


def source_remove(source_id):
    """移除来源：删其 media_file，清理无引用的 asset 及派生表数据，解散过小分组。"""
    con = sqlite3.connect(DB, timeout=60)
    con.execute("PRAGMA busy_timeout=60000")
    con.row_factory = sqlite3.Row
    src = con.execute("SELECT source_id, root_path FROM source WHERE source_id=?", (source_id,)).fetchone()
    if not src:
        con.close()
        return {"error": "来源不存在"}
    affected = [r[0] for r in con.execute(
        "SELECT DISTINCT asset_id FROM media_file WHERE source_id=?", (source_id,))]
    con.execute("DELETE FROM media_file WHERE source_id=?", (source_id,))
    orphaned = []
    for aid in affected:
        n = con.execute("SELECT COUNT(*) FROM media_file WHERE asset_id=?", (aid,)).fetchone()[0]
        if n == 0:
            orphaned.append(aid)
            for tbl in ("media_asset", "asset_filter_v0", "asset_allowlist_v0", "asset_similar_member_v0"):
                try:
                    con.execute(f"DELETE FROM {tbl} WHERE asset_id=?", (aid,))
                except Exception:
                    pass
    # 分组失去成员：解散 <2 人的组，最佳丢失的组重选
    # 孤儿资产的缩略图/预览缓存一并移入废纸篓，避免残留（2026-08-30 修复：此前只清库不清缓存，积累了 444 个孤儿缩略图）
    cache_moved = _purge_asset_cache(orphaned)
    small = [r[0] for r in con.execute(
        "SELECT group_id FROM asset_similar_member_v0 GROUP BY group_id HAVING COUNT(*)<2")]
    for gid in small:
        con.execute("DELETE FROM asset_similar_member_v0 WHERE group_id=?", (gid,))
        con.execute("DELETE FROM asset_similar_group_v0 WHERE group_id=?", (gid,))
    stale_best = [r[0] for r in con.execute("""
        SELECT g.group_id FROM asset_similar_group_v0 g
        WHERE NOT EXISTS (SELECT 1 FROM asset_similar_member_v0 m
                          WHERE m.group_id=g.group_id AND m.asset_id=g.best_asset_id)""")]
    for gid in stale_best:
        try:
            similar_repick(gid)
        except Exception:
            pass
    con.execute("DELETE FROM source WHERE source_id=?", (source_id,))
    con.commit()
    total_assets = con.execute("SELECT COUNT(*) FROM media_asset").fetchone()[0]
    con.close()
    return {"removed": source_id, "root_path": src["root_path"], "assets_removed": len(orphaned),
            "cache_files_removed": cache_moved,
            "total_assets": total_assets}


def source_discover():
    """自动发现可添加的数据来源候选（网盘目录 / 其他挂载卷 / 本机常见目录）。"""
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    roots = [r[0] for r in con.execute("SELECT root_path FROM source")]
    con.close()

    def covered(p):
        for r in roots:
            if p == r or p.startswith(r + os.sep) or r.startswith(p + os.sep):
                return True
        return False

    candidates = []  # (path, kind)
    try:
        for vol in sorted(os.listdir("/Volumes")):
            if vol.startswith(".") or vol in DISCOVER_SKIP_NAMES:
                continue
            vp = os.path.join("/Volumes", vol)
            if not os.path.isdir(vp):
                continue
            if _is_app_volume(vp):
                continue  # DMG 安装卷，里面是应用包不是照片
            candidates.append((vp, "外部卷/网络存储"))
            try:
                for child in sorted(os.listdir(vp)):
                    cp = os.path.join(vp, child)
                    if child.startswith(".") or child in {"@eaDir", "#recycle", "Applications"} or child.endswith(".app"):
                        continue
                    if os.path.isdir(cp):
                        candidates.append((cp, f"{vol} 子目录"))
            except Exception:
                pass
    except Exception:
        pass
    for label, p in CLOUD_CANDIDATE_DIRS:
        if os.path.isdir(p):
            candidates.append((p, label))

    results = []
    for path, kind in candidates:
        if covered(path):
            continue
        n = _count_media_files(path)
        if n >= 3:
            results.append({"path": path, "kind": kind, "media_files": n})
    results.sort(key=lambda x: -x["media_files"])
    return {"candidates": results[:12]}


# ============ 初始化向导（施工图#5） ============
# 三步：建管理员（原子互斥）→ 选照片目录（校验+登记来源）→ 触发首次全量扫描（进度轮询）。
# 范围纪律：只建 user 记录和照片源，登录/会话是 #9 的事，这里不做。
# 门禁唯一依据是 is_initialized()：完成后全部 /api/init/* 永久禁用，向导页重定向回首页。
INIT_SCAN = {"progress": {"status": "idle"}}
_INIT_SCAN_LOCK = threading.Lock()


def is_initialized():
    """初始化状态唯一权威判定：admin 存在且已登记照片源 = 向导走完。
    仅 admin 存在（向导进行中）不算完成——否则会锁死向导自己的后续步骤。"""
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            has_admin = bool(con.execute("SELECT 1 FROM user WHERE role='admin' LIMIT 1").fetchone())
            has_source = bool(con.execute("SELECT 1 FROM source LIMIT 1").fetchone())
            return has_admin and has_source
        finally:
            con.close()
    except sqlite3.OperationalError:
        return False


def wizard_state():
    """向导细分状态：has_admin / has_source，前端据此恢复到对应步骤。"""
    try:
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        try:
            has_admin = bool(con.execute("SELECT 1 FROM user WHERE role='admin' LIMIT 1").fetchone())
            has_source = bool(con.execute("SELECT 1 FROM source LIMIT 1").fetchone())
            return has_admin, has_source
        finally:
            con.close()
    except sqlite3.OperationalError:
        return False, False


def _hash_password(pw, iterations=100_000):
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"), salt, iterations)
    return f"{iterations}${salt.hex()}${digest.hex()}"


def init_create_admin(username, password):
    """原子建唯一管理员。DB 层部分唯一索引 idx_user_single_admin 兜底，
    双浏览器竞态下只有一个 INSERT 成功，另一个回 'admin_exists'。"""
    if not username or len(username) < 2:
        return False, "用户名至少 2 个字符"
    if not password or len(password) < 6:
        return False, "密码至少 6 位"
    con = sqlite3.connect(DB, timeout=10)
    try:
        con.execute("PRAGMA busy_timeout=10000")
        cur = con.execute(
            "INSERT INTO user (user_id, username, password_hash, role, created_at) "
            "SELECT ?, ?, ?, 'admin', ? WHERE NOT EXISTS "
            "(SELECT 1 FROM user WHERE role='admin')",
            (f"user_{uuid.uuid4().hex[:12]}", username.strip(),
             _hash_password(password), now_iso()))
        con.commit()
        if cur.rowcount == 0:
            return False, "已有管理员"
        return True, "ok"
    except sqlite3.IntegrityError:
        return False, "用户名已被使用"
    finally:
        con.close()


# ==================== 施工图#9 两级鉴权 ====================
# 角色：admin=全权；family=家人(浏览+问答，只读)；guest=访客(只读，v0.1 同 family，
#       相册白名单 v0.2)。迁移 001 已建 user/session 表。
# 电视 Cookie 保持：登录支持"记住此设备"(365 天长效 session)；电视 webview 若
#       不持久化 Cookie，可用 ?token=<token> URL 兜底（登录接口原样返回 token）。

AUTH_COOKIE = "ff_session"
_AUTH_PUBLIC_PATHS = {
    "/login.html", "/tv.html", "/favicon.ico",   # tv.html 自带登录（401 时页面内处理）
    "/init_wizard.html",   # 空库首启向导（未初始化时公开直达；已初始化时门禁 302 回首页）
    "/api/auth/login", "/api/auth/logout", "/api/auth/me",
    "/api/init/status", "/api/init/admin", "/api/init/validate-dir", "/api/init/finish",
    "/api/init/browse-dir",
    "/api/brand",   # 品牌标题：读公开（首页/登录前即可拉取）；写入在 handler 内校验 admin
}
# 非 admin 用户（GET 也拒绝）：管理面 API
_AUTH_ADMIN_PREFIXES = (
    "/api/init", "/api/llm", "/api/settings", "/api/source", "/api/models",
    "/api/privacy", "/api/vlm", "/api/captions", "/api/catman", "/api/recycle",
    "/api/scene", "/api/similar/scan", "/api/similar/repick", "/api/similar/set-best",
    "/api/similar/ungroup", "/api/trip/rebuild", "/api/trip/delete", "/api/trip/save",
    "/api/faces/run", "/api/faces/status", "/api/geo/seed", "/api/geo/vision-verify",
    # 2026-09-24 后台重活：改配置 / 手动拉起预热与转码，仅管理员
    "/api/tasks/config", "/api/tasks/warm", "/api/tasks/videolc",
    "/api/filter/add", "/api/filter/remove", "/api/auth/users",
    # 2026-09-16 增量富化：状态查询与手动触发都属后台任务，admin 专属
    "/api/enrich",
)
# 非 admin 允许的写方法白名单：本服务路由几乎全是 POST（含只读查询），
# 故非 admin 采用「只读 POST 白名单 + 默认拒绝」，白名单只收纯读接口。
_AUTH_READ_POST_ALLOW = (
    "/api/auth/passwd",      # 2026-09-13 本人修改密码（handler 内拦 guest，访客由管理员重置）
    "/api/ask",              # 中文自然语言问答（招牌功能，纯查询）
    "/api/asset/search",     # 照片搜索
    "/api/check-new",        # 新照片轮询
    "/api/geo/regions", "/api/geo/map",   # 中国地图
    "/api/trip/list",        # 旅行列表
    "/api/filter/list",      # 筛选器
    "/api/filter/count",     # 侧栏过滤计数徽标（只读）
    "/api/tags",             # 标签
    "/api/categories", "/api/category",   # 分类浏览
    "/api/crop",             # 局部图预览
)
# family 角色可用的内容级写操作（2026-09-13）：与用户管理页角色描述"可看可管理"
# 一致——移出/恢复墙面、回收站、裁切、手动定位锚点。系统配置/后台任务仍 admin 专属。
_AUTH_FAMILY_POST_ALLOW = (
    "/api/filter/add", "/api/filter/remove",
    "/api/recycle",
    "/api/geo/seed",
)
# /api/people 特例：读分支(detail/无action)放行、写分支(update_*)在 handler 内校验 admin


def _verify_password(pw, stored):
    try:
        iterations, salt_hex, hash_hex = stored.split("$")
        digest = hashlib.pbkdf2_hmac("sha256", pw.encode("utf-8"),
                                     bytes.fromhex(salt_hex), int(iterations))
        return secrets.compare_digest(digest.hex(), hash_hex)
    except Exception:
        return False


def _auth_connect():
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    return con


# ---- 登录限速（2026-09-13）：滑动窗口失败锁定，防密码爆破 ----
# 双层计数：同 IP+用户名 15 分钟内失败 5 次锁；同 IP 全局 30 次锁（防换用户名扫射）。
# 进程内存态（重启即清），成功登录清该 IP+用户名计数；IP 级保留不奖赏。
_LOGIN_FAIL_LOCK = threading.Lock()
_LOGIN_FAILS = {}            # (ip, username) -> [失败时间戳列表]（滑动窗口内）
_LOGIN_IP_FAILS = {}         # ip -> [失败时间戳列表]
_LOGIN_WINDOW = 900.0        # 15 分钟滑动窗口
_LOGIN_MAX_USER = 5          # 同 IP+用户名窗口内允许失败次数
_LOGIN_MAX_IP = 30           # 同 IP（任意用户名）窗口内允许失败总次数


def _login_fail_purge(now):
    """清掉窗口外/空计数，防内存无限涨（每次检查顺带执行）。"""
    for d in (_LOGIN_FAILS, _LOGIN_IP_FAILS):
        for k in [k for k, v in d.items() if not v or now - v[-1] > _LOGIN_WINDOW]:
            d.pop(k, None)


def _login_rate_check(ip, username):
    """登录前检查。返回 None（放行）或需等待秒数（已被限速）。"""
    now = time.time()
    with _LOGIN_FAIL_LOCK:
        _login_fail_purge(now)
        fu = _LOGIN_FAILS.get((ip, username)) or []
        fi = _LOGIN_IP_FAILS.get(ip) or []
        if len(fu) >= _LOGIN_MAX_USER:
            return max(1, int(_LOGIN_WINDOW - (now - fu[0])))
        if len(fi) >= _LOGIN_MAX_IP:
            return max(1, int(_LOGIN_WINDOW - (now - fi[0])))
        return None


def _login_rate_record(ip, username, ok):
    """登录落账：成功清 (ip,username) 计数；失败追加两层计数。"""
    if ok:
        with _LOGIN_FAIL_LOCK:
            _LOGIN_FAILS.pop((ip, username), None)
        return
    now = time.time()
    with _LOGIN_FAIL_LOCK:
        _LOGIN_FAILS.setdefault((ip, username), []).append(now)
        _LOGIN_IP_FAILS.setdefault(ip, []).append(now)


def auth_login(username, password, remember):
    """校验并建 session。返回 (ok, info)；ok 时 info={token, max_age, role, username}"""
    if not username or not password:
        return False, "请输入用户名和密码"
    con = _auth_connect()
    try:
        user = con.execute("SELECT * FROM user WHERE username=?", (username.strip(),)).fetchone()
        if not user or not _verify_password(password, user["password_hash"]):
            return False, "用户名或密码错误"
        days = 365 if remember else 1
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=days)).isoformat()
        con.execute("DELETE FROM session WHERE expires_at < ?", (now_iso(),))  # 顺手清过期
        con.execute("INSERT INTO session(token,user_id,created_at,expires_at) VALUES(?,?,?,?)",
                    (token, user["user_id"], now_iso(), expires))
        con.commit()
        return True, {"token": token, "max_age": days * 86400,
                      "role": user["role"], "username": user["username"]}
    finally:
        con.close()


# ---- 在线升级（2026-09-14）：更新包放 /data/updates/*.zip，应用内检查+应用 ----
# 包结构：manifest.json {version, notes, files:[{path, md5}]} + server.py + static/...
# 只允许覆盖 server.py 和 static/ 下文件；逐文件 md5 校验；替换前备份到
# /data/update_backup/<旧版本>/；替换完成后延迟自杀，docker unless-stopped
# 自动拉起新代码完成升级。
import zipfile

UPDATE_DIR = DATA_DIR / "updates"
UPDATE_BACKUP_DIR = DATA_DIR / "update_backup"


def _ver_tuple(v):
    try:
        return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3]) or (0,)
    except Exception:
        return (0,)


def update_best_package():
    """扫描更新目录，返回版本最高的合法更新包 manifest；无包/坏包返回 None。"""
    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    for zp in UPDATE_DIR.glob("*.zip"):
        try:
            with zipfile.ZipFile(zp) as zf:
                man = json.loads(zf.read("manifest.json").decode("utf-8"))
            if not man.get("version") or not man.get("files"):
                continue
            if best is None or _ver_tuple(man["version"]) > _ver_tuple(best["version"]):
                man["_zip"] = str(zp)
                best = man
        except Exception:
            continue
    return best


# 2026-09-14 远程更新源（公网推送）：feed = 一个固定 URL 下的 manifest.json + 更新包 zip。
# manifest: {version, file, md5, notes, size}。渠道自定（GitHub Releases / 对象存储 / 自有域名），
# 客户端逻辑一致：检查远程 → 下载到 /data/updates/ → md5 校验 → 走本地 apply 流程。
def get_update_feed_url():
    try:
        return (get_setting("update_feed_url", "") or "").strip()
    except Exception:
        return ""


def update_check_remote():
    """拉远程 feed 的 manifest.json；未配置/网络失败/格式错返回 None。"""
    feed = get_update_feed_url()
    if not feed:
        return None
    try:
        req = urllib.request.Request(feed.rstrip("/") + "/manifest.json",
                                     headers={"User-Agent": "FamilyMemory/" + APP_VERSION})
        with urllib.request.urlopen(req, timeout=8) as r:
            man = json.loads(r.read().decode("utf-8"))
        if not man.get("version") or not man.get("file") or not man.get("md5"):
            return None
        return man
    except Exception:
        return None


def update_fetch_remote(man):
    """下载远程更新包到 /data/updates/（500MB 上限），校验 md5 与包内版本一致性。"""
    feed = get_update_feed_url()
    fname = str(man["file"])
    # 2026-09-14 审计修复：文件名白名单——远程 manifest 是外部可控数据，禁路径拼接
    if "/" in fname or "\\" in fname or ".." in fname or not fname.endswith(".zip"):
        return False, f"非法更新包文件名: {fname}"
    url = man.get("url") or (feed.rstrip("/") + "/" + fname)
    dst = UPDATE_DIR / fname
    UPDATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(f".{threading.get_ident()}.dl.tmp")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FamilyMemory/" + APP_VERSION})
        h = hashlib.md5()
        n = 0
        with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
            while True:
                chunk = r.read(1 << 16)
                if not chunk:
                    break
                n += len(chunk)
                if n > 500 * 1024 * 1024:
                    raise ValueError("更新包超过 500MB 上限")
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != man.get("md5"):
            tmp.unlink(missing_ok=True)
            return False, "下载包 md5 校验失败（源可能被篡改或下载不完整）"
        with zipfile.ZipFile(tmp) as zf:
            inner = json.loads(zf.read("manifest.json").decode("utf-8"))
        if inner.get("version") != man.get("version"):
            tmp.unlink(missing_ok=True)
            return False, f"包内版本 {inner.get('version')} 与源声明 {man.get('version')} 不一致"
        os.replace(tmp, dst)
        return True, f"下载完成 {n/1024/1024:.1f}MB"
    except Exception as e:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        return False, f"下载失败: {e}"


# 更新包允许落地的文件（白名单最小化，2026-09-24 补）。
# 曾经这里只放行 server.py + static/**，结果 worker 脚本（转码等脱离 web 进程的
# 后台程序）的修复**永远分发不到用户机器**，只能手工进容器拷文件——对一个要公开发布
# 的产品是致命缺口：别人下载了你的项目，修好的转码器他却拿不到。
# 名单外的路径一律拒绝：远程 manifest 属外部可控数据，绝不放开任意文件写。
UPDATE_ALLOWED_FILES = {"server.py", "transcode_log_videos.py"}


def _update_path_ok(p):
    """更新包中的相对路径是否允许落到 ROOT 下。"""
    if not p or ".." in p or p.startswith("/") or "\\" in p:
        return False
    return p.startswith("static/") or p in UPDATE_ALLOWED_FILES


def update_apply():
    """校验+备份+替换文件+延迟重启。返回 (ok, 消息)。"""
    man = update_best_package()
    if not man:
        return False, "更新目录里没有可用更新包"
    if _ver_tuple(man["version"]) <= _ver_tuple(APP_VERSION):
        return False, f"已是最新版本 {APP_VERSION}"
    try:
        zf = zipfile.ZipFile(man["_zip"])
    except Exception as e:
        return False, f"更新包打不开: {e}"
    # 路径安全 + 逐文件 md5 校验（全部通过才动手）
    data_map = {}
    for f in man["files"]:
        p = str(f.get("path") or "")
        if not _update_path_ok(p):
            return False, f"非法更新路径: {p}"
        data = zf.read(p)
        if hashlib.md5(data).hexdigest() != f.get("md5"):
            return False, f"md5 校验失败: {p}"
        data_map[p] = data
    if "server.py" not in data_map:
        return False, "更新包缺少 server.py"
    # 备份将被替换的现有文件（文件名带路径 md5 前缀，防不同子目录同名互撞）
    bak = UPDATE_BACKUP_DIR / f"v{APP_VERSION}"
    bak.mkdir(parents=True, exist_ok=True)
    for p, data in data_map.items():
        dst = ROOT / p
        if dst.exists():
            shutil.copy2(dst, bak / (p.replace("/", "_") + "." + hashlib.md5(p.encode()).hexdigest()[:8]))
    # 两阶段替换（2026-09-14 审计修复）：先全部写成 tmp，再统一 os.replace——
    # 单循环逐文件替换时中途死会把库切成两个版本的大杂烩
    try:
        tmp_map = {}
        for p, data in data_map.items():
            dst = ROOT / p
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(dst.suffix + f".{threading.get_ident()}.up.tmp")
            tmp.write_bytes(data)
            tmp_map[tmp] = dst
        for tmp, dst in tmp_map.items():
            os.replace(tmp, dst)
    except Exception as e:
        for tmp in tmp_map.values():
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
        return False, f"替换文件失败（未完成升级，可重试）: {e}"
    threading.Timer(2.0, lambda: os._exit(0)).start()  # docker 自动拉起新版本
    return True, f"已更新到 v{man['version']}，服务重启中（约 5-10 秒）"


def auth_logout(token):
    if not token:
        return
    with _SESSION_CACHE_LOCK:
        _SESSION_CACHE.pop(token, None)   # 立即失效缓存（否则 60s 内 token 仍可用）
    con = _auth_connect()
    try:
        con.execute("DELETE FROM session WHERE token=?", (token,))
        con.commit()
    finally:
        con.close()


# 2026-09-10 二次锁死修复（修改单#2）：session 校验进程内缓存。
# 首页一次加载 = 几百个请求，每个都查一次 session 表——请求洪水时 DB 查询
# 数倍增。60s TTL 缓存；登出/改角色最迟 60s 生效（token 本身仍须匹配，可接受）。
_SESSION_CACHE = {}          # token -> (user_dict 或 None, expire_ts)
_SESSION_CACHE_TTL = 60.0
_SESSION_CACHE_LOCK = threading.Lock()


def _current_session_user(handler):
    """从 Cookie 或 ?token= 取当前用户（未登录返回 None）。电视端 Cookie 失效时用 URL token 兜底。"""
    token = None
    for part in (handler.headers.get("Cookie") or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == AUTH_COOKIE and v:
            token = v
            break
    if not token:
        try:
            q = urllib.parse.urlparse(handler.path).query
            token = (urllib.parse.parse_qs(q).get("token") or [None])[0]
        except Exception:
            pass
    if not token:
        return None, None
    now = time.time()
    with _SESSION_CACHE_LOCK:
        hit = _SESSION_CACHE.get(token)
        if hit and now < hit[1]:
            return hit[0], token
    con = _auth_connect()
    try:
        row = con.execute("""SELECT s.token, s.expires_at, u.user_id, u.username, u.role
                             FROM session s JOIN user u USING(user_id) WHERE s.token=?""",
                          (token,)).fetchone()
        if not row:
            result = None
        elif row["expires_at"] < now_iso():
            con.execute("DELETE FROM session WHERE token=?", (token,))
            con.commit()
            result = None
        else:
            result = dict(row)
    finally:
        con.close()
    with _SESSION_CACHE_LOCK:
        _SESSION_CACHE[token] = (result, now + _SESSION_CACHE_TTL)
        if len(_SESSION_CACHE) > 500:
            for k in [k for k, v in _SESSION_CACHE.items() if v[1] < now]:
                _SESSION_CACHE.pop(k, None)
    return result, token


def _strip_token_param(handler):
    """把 ?token= 从 self.path 摘掉：token 只用于鉴权；路由是精确匹配，
    不摘会 404（电视端 URL token 兜底依赖此函数）。"""
    parsed = urllib.parse.urlparse(handler.path)
    if "token" not in urllib.parse.parse_qs(parsed.query):
        return
    qs = [(k, v) for k, v in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True) if k != "token"]
    handler.path = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(qs)))


def _auth_gate(handler, method):
    """鉴权门：返回 None=放行；否则返回 {"status":..,"json":..} 或 {"redirect":..}。"""
    path = urllib.parse.urlparse(handler.path).path
    # 2026-09-10 二次锁死修复（修改单#2）：静态资源零 DB。锁死期间静态文件
    # 必须照常返回——CSS/JS/字体/图标不含敏感数据，直接放行不查 session。
    if path.startswith(("/static/", "/fonts/", "/logo/")):
        return None
    if path.endswith((".css", ".js", ".map", ".png", ".svg", ".ico", ".woff", ".woff2", ".otf", ".ttf")):
        return None
    if path in _AUTH_PUBLIC_PATHS:
        return None
    user, token = _current_session_user(handler)
    if user is None:
        if path.startswith("/api/"):
            return {"status": 401, "json": {"error": "未登录"}}
        return {"redirect": "/login.html"}
    role = user["role"]
    if role == "admin":
        return None
    # 2026-09-13 family 内容级写操作白名单：与用户管理页的角色描述"可看可管理"一致。
    # 系统配置/后台任务仍 admin 专属；guest 保持全站只读。
    family_content_ok = (role == "family" and (
        path in _AUTH_FAMILY_POST_ALLOW or path.startswith("/api/crop")))
    if not family_content_ok and any(path.startswith(p) for p in _AUTH_ADMIN_PREFIXES):
        return {"status": 403, "json": {"error": "需要管理员权限"}}
    if method in ("GET", "HEAD"):
        return None
    if path in _AUTH_READ_POST_ALLOW or path == "/api/people":
        return None
    if family_content_ok:
        return None
    return {"status": 403, "json": {"error": "只读账户，无权执行此操作"}}


def auth_users_list():
    con = _auth_connect()
    try:
        rows = con.execute("SELECT user_id, username, role, created_at FROM user ORDER BY created_at").fetchall()
        return {"users": [dict(r) for r in rows]}
    finally:
        con.close()


def auth_user_create(username, password, role):
    """管理员建家人/访客账号（admin 角色由向导独占创建，此处拒绝）。"""
    if role not in ("family", "guest"):
        return False, "角色只能是 family 或 guest"
    if not username or len(username.strip()) < 2:
        return False, "用户名至少 2 个字符"
    if not password or len(password) < 6:
        return False, "密码至少 6 位"
    con = _auth_connect()
    try:
        con.execute("INSERT INTO user (user_id, username, password_hash, role, created_at) VALUES(?,?,?,?,?)",
                    (f"user_{uuid.uuid4().hex[:12]}", username.strip(),
                     _hash_password(password), role, now_iso()))
        con.commit()
        return True, "ok"
    except sqlite3.IntegrityError:
        return False, "用户名已被使用"
    finally:
        con.close()


def auth_user_delete(user_id):
    con = _auth_connect()
    try:
        row = con.execute("SELECT role FROM user WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return False, "用户不存在"
        if row["role"] == "admin":
            return False, "不能删除管理员"
        con.execute("DELETE FROM session WHERE user_id=?", (user_id,))
        con.execute("DELETE FROM user WHERE user_id=?", (user_id,))
        con.commit()
        _session_cache_drop_user(user_id)
        return True, "ok"
    finally:
        con.close()


def _session_cache_drop_user(user_id):
    """按 user_id 清掉会话缓存。2026-09-13 修复：重置密码/删除用户只删 DB 会话
    不清缓存，被踢用户在 60s TTL 内依然通过鉴权（实测 admin 重置后旧 token 仍 200）。"""
    with _SESSION_CACHE_LOCK:
        for tk, (u, _exp) in list(_SESSION_CACHE.items()):
            if u and u.get("user_id") == user_id:
                _SESSION_CACHE.pop(tk, None)


def auth_user_set_password(user_id, new_password):
    """管理员重置任意用户密码；成功后踢掉该用户全部会话，强制重新登录。
    2026-09-13 新增：成员账号管理页（users.html）。"""
    if not new_password or len(new_password) < 6:
        return False, "密码至少 6 位"
    con = _auth_connect()
    try:
        if not con.execute("SELECT 1 FROM user WHERE user_id=?", (user_id,)).fetchone():
            return False, "用户不存在"
        con.execute("UPDATE user SET password_hash=? WHERE user_id=?",
                    (_hash_password(new_password), user_id))
        con.execute("DELETE FROM session WHERE user_id=?", (user_id,))
        con.commit()
        _session_cache_drop_user(user_id)
        return True, "ok"
    finally:
        con.close()


def auth_change_own_password(user_id, old_password, new_password, current_token):
    """本人修改密码：需验证旧密码；成功后踢掉本人其它会话（当前会话保留）。
    2026-09-13 新增：成员账号管理页（users.html）。"""
    if not new_password or len(new_password) < 6:
        return False, "新密码至少 6 位"
    con = _auth_connect()
    try:
        row = con.execute("SELECT password_hash FROM user WHERE user_id=?", (user_id,)).fetchone()
        if not row:
            return False, "用户不存在"
        if not _verify_password(old_password or "", row["password_hash"]):
            return False, "旧密码不正确"
        con.execute("UPDATE user SET password_hash=? WHERE user_id=?",
                    (_hash_password(new_password), user_id))
        con.execute("DELETE FROM session WHERE user_id=? AND token<>?", (user_id, current_token))
        con.commit()
        return True, "ok"
    finally:
        con.close()


def init_validate_dir(path):
    """目录校验：存在/可读/含媒体文件。返回 (ok, info)。info 含媒体文件数预览（计数上限防超深目录）。"""
    p = Path(path)
    if not str(path).strip():
        return False, "请输入照片目录路径"
    if not p.is_dir():
        return False, f"目录不存在：{path}"
    if not os.access(str(p), os.R_OK):
        return False, f"目录不可读（检查挂载/权限）：{path}"
    n = 0
    for base, dirs, files in os.walk(str(p)):
        dirs[:] = [d for d in dirs if d not in {"@eaDir", "#recycle"} and not d.startswith(".")]
        n += sum(1 for f in files if Path(f).suffix.lower() in MEDIA_EXTS)
        if n >= 20000:
            break
    if n == 0:
        return False, "目录里没有找到照片/视频文件"
    return True, {"media_files": n, "truncated": n >= 20000}


def init_browse_dir(path):
    """向导目录浏览：path 为空返回常用候选根；为目录返回可下钻的子目录列表。
    只读、只列目录不列文件；每个子目录附媒体文件数预览（上限 500 防超深目录卡顿）。"""
    def _media_preview(p):
        n = 0
        try:
            for base, dirs, files in os.walk(str(p)):
                dirs[:] = [d for d in dirs if d not in {"@eaDir", "#recycle"} and not d.startswith(".")]
                n += sum(1 for f in files if Path(f).suffix.lower() in MEDIA_EXTS)
                if n >= 500:
                    return "500+"
            return str(n)
        except OSError:
            return "?"
    if not str(path).strip():
        home = Path.home()
        candidates = [home / "Pictures", home / "Desktop", home / "Documents"]
        vols = Path("/Volumes")
        if vols.is_dir():
            candidates += [v for v in vols.iterdir()
                           if v.is_dir() and v.name not in {"Macintosh HD", "PreBoot"}]
        return {"ok": True, "roots": [
            {"path": str(c), "media": _media_preview(c)}
            for c in candidates if c.is_dir()]}
    p = Path(str(path)).expanduser()
    if not p.is_dir():
        return {"ok": False, "msg": f"目录不存在：{path}"}
    try:
        dirs = sorted(
            (d for d in p.iterdir()
             if d.is_dir() and d.name not in {"@eaDir", "#recycle", "Library"}
             and not d.name.startswith(".")),
            key=lambda d: d.name.lower())
    except OSError as exc:
        return {"ok": False, "msg": f"目录不可读：{exc}"}
    parent = str(p.parent) if p.parent != p else ""
    return {"ok": True, "current": str(p), "parent": parent,
            "dirs": [{"name": d.name, "path": str(d)} for d in dirs[:200]]}


def init_add_source(path):
    """登记照片源（幂等：root_path UNIQUE + OR IGNORE）。返回是否新登记。"""
    con = sqlite3.connect(DB, timeout=10)
    try:
        con.execute("PRAGMA busy_timeout=10000")
        fam = con.execute("SELECT family_id FROM family LIMIT 1").fetchone()
        fam_id = fam[0] if fam else "family_default"
        con.execute(
            "INSERT OR IGNORE INTO family (family_id, name, created_at) VALUES (?, ?, ?)",
            (fam_id, "我的家庭", now_iso()))
        cur = con.execute(
            "INSERT OR IGNORE INTO source (source_id, family_id, owner_label, root_path, "
            "source_type, read_only, last_scan_at) VALUES (?, ?, ?, ?, 'local_folder', 1, ?)",
            (f"src_{uuid.uuid4().hex[:12]}", fam_id, "管理员", str(path), now_iso()))
        con.commit()
        return cur.rowcount > 0
    finally:
        con.close()


def init_start_scan(path):
    """触发首次全量扫描：后台线程 + progress dict 供前端轮询（复用 _check_new_files 机制）。"""

    def _run():
        with _INIT_SCAN_LOCK:
            try:
                INIT_SCAN["progress"] = {"status": "scanning", "scanned": 0, "new_files": 0,
                                         "message": "正在遍历文件..."}
                r = _check_new_files(roots_override=[str(path)], progress=INIT_SCAN["progress"])
                INIT_SCAN["progress"]["status"] = "done"
                INIT_SCAN["progress"]["indexed"] = r.get("indexed", 0)
                INIT_SCAN["progress"]["total_assets"] = r.get("total_assets", 0)
                INIT_SCAN["progress"]["message"] = (f"完成：入库 {r.get('indexed', 0)} 张，"
                                                    f"共 {r.get('total_assets', 0)} 个资产")
            except Exception as exc:
                INIT_SCAN["progress"] = {"status": "error", "message": str(exc)}

    threading.Thread(target=_run, daemon=True, name="init-scan").start()


# ============ Log 视频转码自动守护（2026-09-24）============
# D-Log 原片是 4K HEVC 10bit，Chrome 解不动，必须转成 H.264 8bit 缓存才能播。
# 全量转码要跑好几个小时，而容器重启 / 在线升级 / OOM 都会把它杀掉，
# 所以常驻一个看门狗：有活儿且没在跑就自动续上，不需要人工记着拉起。
_VIDEO_LC_SCRIPT = ROOT / "transcode_log_videos.py"


def videolc_pending_count():
    """待转码条数：is_log=1 的视频里还没有产物（或产物 0 字节）的。"""
    try:
        con = sqlite3.connect(DB, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=10000")
        rows = con.execute("""SELECT lc.asset_id FROM asset_log_color_v0 lc
                              JOIN media_asset ma ON ma.asset_id = lc.asset_id
                              WHERE lc.is_log = 1 AND ma.media_type = 'video'""").fetchall()
        con.close()
    except Exception:
        return 0            # 表还没迁 / 库忙 → 当没活儿，下轮再看
    n = 0
    for r in rows:
        f = VIDEO_LC_DIR / (str(r["asset_id"])[6:] + "_lc.mp4")
        try:
            if f.exists() and f.stat().st_size > 0:
                continue
        except OSError:
            pass
        n += 1
    return n


def videolc_run(trigger="manual"):
    """拉起转码（幂等：已在跑 / 无待办则不重复）。"""
    if trigger == "autorun" and get_setting("videolc_autorun_enabled", "1") != "1":
        return {"ok": True, "skipped": "videolc_autorun_enabled=0"}
    pend = videolc_pending_count()
    if pend <= 0:
        return {"ok": True, "skipped": "没有待转码的 Log 视频", "pending": 0}
    if not res_ok(for_task="转码"):
        return {"ok": True, "skipped": "整机内存/负载不满足护栏条件，转码暂不启动"}
    r = _spawn_worker("videolc", _VIDEO_LC_SCRIPT, "videolc.log",
                      r"transcode_log_videos\.py")
    r["pending"] = pend
    if r.get("started"):
        set_setting("videolc_last_run_at", now_iso())
        set_setting("videolc_last_trigger", trigger)
    return r


def videolc_status():
    """转码进度（界面 / 排障用）。"""
    return {"enabled": get_setting("videolc_autorun_enabled", "1") == "1",
            "pending": videolc_pending_count(),
            "running": bool(_pgrep(r"transcode_log_videos\.py")),
            "last_run_at": get_setting("videolc_last_run_at", ""),
            "last_trigger": get_setting("videolc_last_trigger", "")}


def tasks_config_action(body):
    """后台任务与内存护栏设置（2026-09-24 产品化）。

    护栏阈值本身按机器内存自适应（res_levels），不暴露给普通用户乱调——
    真要覆盖留给高级用户：res_warn_mb / res_danger_mb 为空即走自适应。
    """
    changed = {}
    for key in ("warm_autorun_enabled", "videolc_autorun_enabled",
                "res_guard_enabled"):
        v = body.get(key)
        if v is None:
            continue
        set_setting(key, "1" if str(v) in ("1", "true", "True", True) else "0")
        changed[key] = get_setting(key, "1")
    for key in ("res_warn_mb", "res_danger_mb"):
        v = body.get(key)
        if v is None:
            continue
        v = str(v).strip()
        if v == "":
            set_setting(key, "")
            changed[key] = ""
        else:
            try:
                iv = max(128, int(v))
            except ValueError:
                continue
            set_setting(key, str(iv))
            changed[key] = str(iv)
    return {"ok": True, "changed": changed}


def _videolc_watchdog():
    """看门狗：每 10 分钟看一次，没在跑且有活儿就自动续上。

    首轮延迟 3 分钟 —— 避开启动期的扫描/富化，别跟它们抢 CPU（这台 NAS
    只有 4 核，ffmpeg 全速会拖慢整站，见 09-24 的转码降核修复）。
    """
    time.sleep(180)
    while True:
        try:
            # 与预热互斥：两个都是 4K 解码，同时跑内存必然吃紧（09-24 风暴教训）。
            # 注意不能用 continue——会跳过末尾 sleep 变成忙循环，本轮不拉即可。
            if (not WARM_STATE.get("running")
                    and get_setting("videolc_autorun_enabled", "1") == "1"):
                r = videolc_run(trigger="autorun")
                if r.get("started"):
                    print(f"[videolc] 看门狗自动续跑转码（欠 {r.get('pending')} 条）", flush=True)
        except Exception as exc:
            print(f"[videolc] watchdog error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(600)


# ============ 缩略图预热自动守护（2026-09-24 产品化）============
# 视频缩略图要靠 ffmpeg 抽帧，一条 4K HEVC 就能吃 1~2GB。用户第一次打开视频墙
# 会一次性触发几十条抽帧 → 内存风暴（实测 load 76、整页灰块）。与其等用户撞上，
# 不如服务空闲时自己串行补齐。**串行是硬要求**：并发数按核数定就是事故，
# 必须按「单请求峰值内存」定。这里一次只解一条，且遇到转码或高负载就主动让位。
WARM_STATE = {"running": False, "done": 0, "total": 0, "started_at": "",
              "last_result": ""}


def _load1():
    """1 分钟负载；读不到返回 0（非 Linux 环境）。"""
    try:
        return float(open("/proc/loadavg").read().split()[0])
    except Exception:
        return 0.0


def warm_pending_ids():
    """待预热的视频（按时间倒序，最近的先补）——**只留缓存确实缺档的**。

    2026-09-25 改。原实现返回全部视频（3925 条），预热一轮要刷近 2 小时；而预热与
    转码是互斥的（同时跑 4K 解码就是 09-24 那场 swap 风暴的根因），于是预热那一轮里
    转码几乎被饿死——实测预热跑到 98/3925 时，转码 143 条一条都没动，用户看到的就是
    「转码又不动了」。
    这里整目录扫一次建好文件名索引，只把 t480 / t1600 两档不齐全的视频留下来；
    缓存其实齐全的视频不再占用那把串行锁。
    """
    try:
        con = sqlite3.connect(DB, timeout=10)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=10000")
        rows = con.execute("SELECT asset_id FROM media_asset WHERE media_type='video' "
                           "ORDER BY capture_time DESC").fetchall()
        con.close()
    except Exception:
        return []
    if not rows:
        return []
    # 一次性把缩略图目录建成索引，避免 3925 次 glob（每次都要重扫目录）
    try:
        with os.scandir(THUMB_DIR) as it:
            have = [e.name for e in it]
    except Exception:
        have = []                       # 目录读不到（还很常见）：退回全量，宁可多跑
    have480, have1600 = set(), set()
    for name in have:
        if not name.endswith(".jpg"):
            continue
        base = name[:-4]
        k = base.rfind("_t")            # 形如 <asset_id后6位>_t480_lcv1.jpg
        if k <= 0:
            continue
        stem, edge = base[:k], base[k + 2:].split("_")[0]
        if edge == "480":
            have480.add(stem)
        elif edge == "1600":
            have1600.add(stem)
    return [str(r["asset_id"]) for r in rows
            if str(r["asset_id"])[6:] not in have480
            or str(r["asset_id"])[6:] not in have1600]


def warm_status():
    """预热状态（任务面板/排障用）。"""
    return {"enabled": get_setting("warm_autorun_enabled", "1") == "1",
            "running": WARM_STATE["running"], "done": WARM_STATE["done"],
            "total": WARM_STATE["total"], "started_at": WARM_STATE["started_at"],
            "last_at": get_setting("warm_last_at", ""),
            "last_result": WARM_STATE["last_result"]}


def _warm_worker(ids):
    WARM_STATE.update(running=True, total=len(ids), done=0, started_at=now_iso())
    done = fail = 0
    for aid in ids:
        # 让位一：转码在跑就不跟它抢内存（两个都是 4K 解码，同时跑必然内存吃紧）
        if _pgrep(r"transcode_log_videos\.py"):
            WARM_STATE["last_result"] = "让位转码，本轮结束"
            break
        # 让位二：整机水位不够就退避（内存才是真凶，负载只是表象）
        n = 0
        while not res_ok(for_task="预热") and n < 10:
            time.sleep(30)
            n += 1
        if n >= 10:
            WARM_STATE["last_result"] = "内存/负载不满足护栏条件，本轮暂停"
            break
        # 熔断：跌破底线就直接杀掉 ffmpeg 收手（宁可失败，不能拖垮整机）
        if res_emergency(reason="预热巡检"):
            WARM_STATE["last_result"] = "触发内存熔断，本轮结束"
            break
        try:
            get_thumb(aid)
            get_thumb(aid, 1600)
            done += 1
        except Exception:
            fail += 1
        WARM_STATE["done"] = done
        time.sleep(0.2)
    WARM_STATE["running"] = False
    WARM_STATE["last_result"] = f"本轮补齐 {done} 条，失败 {fail} 条"
    set_setting("warm_last_at", now_iso())
    print(f"[warm] {WARM_STATE['last_result']}", flush=True)


def warm_run(trigger="autorun"):
    """拉起预热（幂等：已在跑 / 转码占用 / 无视频则不重复）。"""
    if trigger == "autorun" and get_setting("warm_autorun_enabled", "1") != "1":
        return {"ok": True, "skipped": "warm_autorun_enabled=0"}
    if WARM_STATE["running"]:
        return {"ok": True, "already_running": True}
    if _pgrep(r"transcode_log_videos\.py"):
        return {"ok": True, "skipped": "转码运行中，预热让位"}
    if not res_ok(for_task="预热"):
        return {"ok": True, "skipped": "整机内存/负载不满足护栏条件，预热暂不启动"}
    ids = warm_pending_ids()
    if not ids:
        return {"ok": True, "skipped": "没有待预热视频（缓存已齐）"}
    threading.Thread(target=_warm_worker, args=(ids,), daemon=True,
                     name="thumb-warm").start()
    return {"ok": True, "started": True, "pending": len(ids)}


def _warm_watchdog():
    """看门狗：启动 5 分钟后首检，之后每 30 分钟。缓存齐全就几秒扫完退出。"""
    time.sleep(300)
    while True:
        try:
            r = warm_run(trigger="autorun")
            if r.get("started"):
                print(f"[warm] 看门狗拉起预热（{r.get('pending')} 条待检查）", flush=True)
        except Exception as exc:
            print(f"[warm] watchdog error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(1800)


def _autoscan_loop():
    """后台线程：定期增量扫描所有启用来源，实现自动更新。
    2026-09-10 修 .app 启动锁死：首轮不再 90 秒就开跑（启动即扫 NAS 挂载、
    与 vlm/privacy 三路并发抢写锁），首轮延迟 = 正常扫描间隔，让服务先稳定。"""
    try:
        interval = max(15, int(get_setting("autoscan_interval_min", "120")))
    except Exception:
        interval = 120
    time.sleep(interval * 60)
    while True:
        try:
            if get_setting("autoscan_enabled", "1") == "1" and _BG_TASK_LOCK.acquire(blocking=False):
                try:
                    with SCAN_LOCK:
                        r = _check_new_files()
                    set_setting("autoscan_last_scan_at", now_iso())
                    set_setting("autoscan_last_scan_indexed", str(r.get("indexed", 0)))
                    print(f"[autoscan] scanned={r.get('scanned')} indexed={r.get('indexed')} "
                          f"total={r.get('total_assets')}", flush=True)
                finally:
                    _BG_TASK_LOCK.release()
                # 2026-09-16 新增：富化欠账补跑。
                # 导入时已触发过一轮，但可能因限量/中断/停机没跑完；每轮扫描后
                # 补一次保证最终一致。**无欠账时开销接近 0**（只查标记表，不拉进程），
                # 所以可以放心每 30 分钟跑。放在锁外，避免把锁持有时间拖长。
                try:
                    if get_setting("enrich_enabled", "1") == "1":
                        _pw = _enrich_pending()
                        if _pw and _pw.get("total", 0) > 0:
                            print(f"[enrich] 检测到欠账 {_pw}，拉起富化流水线", flush=True)
                            enrich_run(trigger="autoscan")
                except Exception as exc:
                    print(f"[enrich] 补跑失败(忽略): {type(exc).__name__}: {exc}", flush=True)
            elif get_setting("autoscan_enabled", "1") == "1":
                print("[autoscan] 其他后台任务运行中，本轮跳过", flush=True)
        except Exception as exc:
            print(f"[autoscan] error: {exc}", flush=True)
        try:
            interval = max(15, int(get_setting("autoscan_interval_min", "120")))
        except Exception:
            interval = 120
        time.sleep(interval * 60)


# ============ VLM 自动描述（增量，常驻后台；2026-08-31） ============
# 机制同 autoscan：配置存库、网页可改、后台线程定时检查；新照片导入后自动补描述。
VLM_JOB = {"running": False, "started_at": "", "trigger": "",
           "last_result": "", "log_tail": ""}
_VLM_LOCK = threading.Lock()   # 2026-09-02: vlm_run 并发触发的 check-then-set 竞态锁


def vlm_pending_count():
    """待描述资产数：既无场景标签又无 VLM 描述。"""
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    n = con.execute(
        "SELECT COUNT(*) FROM media_asset WHERE asset_id NOT IN "
        "(SELECT DISTINCT asset_id FROM scene_tag_v0) "
        "AND asset_id NOT IN (SELECT asset_id FROM asset_description_v0)"
    ).fetchone()[0]
    con.close()
    return n


def vlm_autorun_config(enabled=None, interval_min=None):
    """读取/更新 VLM 自动描述配置。"""
    if enabled is not None:
        set_setting("vlm_autorun_enabled", "1" if enabled else "0")
    if interval_min is not None:
        set_setting("vlm_autorun_interval_min", str(max(5, int(interval_min))))
    try:
        interval = int(get_setting("vlm_autorun_interval_min", "180"))
    except Exception:
        interval = 180
    return {
        "enabled": get_setting("vlm_autorun_enabled", "1") == "1",
        "interval_min": max(5, interval),
        "last_run_at": get_setting("vlm_autorun_last_run_at", ""),
        "last_described": get_setting("vlm_autorun_last_described", ""),
        "pending": vlm_pending_count(),
    }


def vlm_status():
    return {"running": VLM_JOB["running"], "started_at": VLM_JOB["started_at"],
            "trigger": VLM_JOB["trigger"], "last_result": VLM_JOB["last_result"],
            "log_tail": VLM_JOB["log_tail"][-400:], **vlm_autorun_config()}


def _vlm_snapshot():
    """当前已有描述的 asset_id 集合，用于跑完后 diff 出新增。"""
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    ids = {r[0] for r in con.execute("SELECT asset_id FROM asset_description_v0")}
    con.close()
    return ids


def _vlm_worker(trigger, wait_lock=False):
    """增量描述 + 受影响日期的叙事重跑。全程后台，不阻塞服务。"""
    # 2026-09-10 全局后台互斥：拿不到锁 = 其他后台任务在跑。自动触发直接让路
    # （下轮再来），手动触发最多等 30s——绝不与 autoscan/privacy 并发抢写锁。
    if not _BG_TASK_LOCK.acquire(timeout=30 if wait_lock else 0):
        VLM_JOB["running"] = False
        VLM_JOB["trigger"] = trigger
        VLM_JOB["last_result"] = "其他后台任务运行中，本轮跳过（稍后自动重试或手动重试）"
        print(f"[vlm-autorun] 与其他后台任务冲突，跳过（trigger={trigger}）", flush=True)
        return
    VLM_JOB["running"] = True
    VLM_JOB["trigger"] = trigger
    VLM_JOB["started_at"] = now_iso()
    VLM_JOB["last_result"] = ""
    VLM_JOB["log_tail"] = ""
    try:
        before = _vlm_snapshot()
        pending = vlm_pending_count()
        if pending == 0:
            VLM_JOB["last_result"] = "无待描述照片"
            return
        VLM_JOB["log_tail"] = f"[{trigger}] 开始描述 {pending} 张\n"
        proc = subprocess.run(_py_cmd("vlm"), cwd=str(ROOT),
                              capture_output=True, text=True, timeout=21600)
        tail = (proc.stdout or "").strip().splitlines()[-3:]
        VLM_JOB["log_tail"] += "\n".join(tail) + "\n"
        new_ids = _vlm_snapshot() - before
        set_setting("vlm_autorun_last_described", str(len(new_ids)))
        VLM_JOB["last_result"] = f"新增描述 {len(new_ids)} 张"
        # 只重跑新增描述涉及的那几天叙事，避免全量 625 天
        if new_ids:
            con = sqlite3.connect(DB, timeout=10)
            con.execute("PRAGMA busy_timeout=10000")
            ids = list(new_ids)
            q = ",".join("?" * len(ids))
            days = [r[0] for r in con.execute(
                f"SELECT DISTINCT substr(capture_time,1,10) FROM media_asset "
                f"WHERE asset_id IN ({q}) AND capture_time IS NOT NULL", tuple(ids)) if r[0]]
            con.close()
            done_days = 0
            for d in sorted(days)[:60]:
                try:
                    subprocess.run(_py_cmd("captions", "--vlm", "--day", d),
                                   cwd=str(ROOT), capture_output=True, text=True, timeout=900)
                    done_days += 1
                except Exception:
                    pass
            VLM_JOB["last_result"] += f" · 重跑 {done_days}/{len(days)} 天叙事"
        print(f"[vlm-autorun] {VLM_JOB['last_result']}", flush=True)
        # 新描述就绪 → 隐私自动检测跟着补一轮（文本快筛，秒级）
        try:
            if get_setting("privacy_auto_enabled", "1") == "1" and not PRIVACY_AUTO_JOB["running"]:
                threading.Thread(target=_privacy_auto_worker, args=("vlm-hook",),
                                 daemon=True, name="privacy-auto").start()
        except Exception as exc:
            print(f"[privacy-auto] vlm-hook 触发失败: {exc}", flush=True)
    except Exception as exc:
        VLM_JOB["last_result"] = f"失败: {exc}"
        VLM_JOB["log_tail"] += f"\n{exc}"
        print(f"[vlm-autorun] error: {exc}", flush=True)
    finally:
        set_setting("vlm_autorun_last_run_at", now_iso())
        VLM_JOB["running"] = False
        _BG_TASK_LOCK.release()


def vlm_run(trigger="manual"):
    """触发一次增量描述；已有任务在跑则返回当前状态。"""
    with _VLM_LOCK:
        if VLM_JOB["running"]:
            return {"started": False, **vlm_status()}
        VLM_JOB["running"] = True  # 先占位, _vlm_worker 开头会再次置 True(幂等), 防并发双启动
    threading.Thread(target=_vlm_worker, args=(trigger, trigger.startswith("manual")),
                     daemon=True, name="vlm-describe").start()
    return {"started": True, **vlm_status()}


def _vlm_autorun_loop():
    """后台线程：定期检查有无新照片需要描述，有则补齐。
    2026-09-10 修 .app 启动锁死：①首轮延迟 = 正常间隔（原 180 秒）；
    ②自动触发只处理增量小批量（pending ≤ 300），全量欠账等手动触发——
    自动跑 7956 张会经子进程长期持写锁拖垮服务。"""
    try:
        interval = max(5, int(get_setting("vlm_autorun_interval_min", "180")))
    except Exception:
        interval = 180
    time.sleep(interval * 60)
    while True:
        try:
            if get_setting("vlm_autorun_enabled", "1") == "1" and not VLM_JOB["running"]:
                n = vlm_pending_count()
                if n > 300:
                    print(f"[vlm-autorun] pending={n} 超过自动阈值 300，等待手动触发"
                          "（设置页 → VLM 描述 → 立即描述）", flush=True)
                elif n > 0:
                    print(f"[vlm-autorun] 发现 {n} 张待描述，自动开始", flush=True)
                    _vlm_worker("auto")
        except Exception as exc:
            print(f"[vlm-autorun] error: {exc}", flush=True)
        try:
            interval = max(5, int(get_setting("vlm_autorun_interval_min", "180")))
        except Exception:
            interval = 180
        time.sleep(interval * 60)


# ============ 隐私内容自动检测（两级级联；2026-09-07） ============
# 算法在 privacy_auto_scan.py（文本加权评分快筛 + VLM 看图复核），此处只做
# 任务编排：常驻后台线程定时增量扫 + VLM 描述完成后自动补扫 + 网页手动触发。
PRIVACY_AUTO_JOB = {"running": False, "started_at": "", "trigger": "",
                    "last_result": "", "log_tail": ""}
_PRIVACY_AUTO_LOCK = threading.Lock()


def _init_privacy_auto_table(con):
    """隐私自动检测决策表：记录每条判定（含被用户撤销的），保证可追溯、撤销不复发。
    v2：media_type 列标记照片/视频（privacy_auto_scan.py 老表自动 ALTER 补列）。"""
    con.execute("""CREATE TABLE IF NOT EXISTS privacy_auto_v0 (
        asset_id TEXT PRIMARY KEY,
        score REAL,
        level INTEGER,
        reasons TEXT,
        decision TEXT NOT NULL,
        decided_by TEXT NOT NULL,
        created_at TEXT NOT NULL,
        media_type TEXT DEFAULT ''
    )""")
    cols = {r[1] for r in con.execute("PRAGMA table_info(privacy_auto_v0)")}
    if "media_type" not in cols:
        con.execute("ALTER TABLE privacy_auto_v0 ADD COLUMN media_type TEXT DEFAULT ''")


def privacy_auto_pending_count():
    """待检测资产数：不在隐私相册、无历史决策（含无描述资产，与脚本口径一致）。
    注：低分不写库的资产会一直计在 pending 里（文本重评零成本），仅作队列参考数。"""
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    _init_privacy_auto_table(con)
    n = con.execute(
        "SELECT COUNT(*) FROM media_asset ma LEFT JOIN asset_description_v0 d USING(asset_id) "
        "WHERE ma.asset_id NOT IN (SELECT asset_id FROM privacy_v0) "
        "AND ma.asset_id NOT IN (SELECT asset_id FROM privacy_auto_v0)").fetchone()[0]
    con.close()
    return n


def privacy_auto_config(enabled=None, interval_min=None):
    """读取/更新自动检测配置（存 app_setting，网页可改）。"""
    if enabled is not None:
        set_setting("privacy_auto_enabled", "1" if enabled else "0")
    if interval_min is not None:
        set_setting("privacy_auto_interval_min", str(max(30, int(interval_min))))
    try:
        interval = int(get_setting("privacy_auto_interval_min", "240"))
    except Exception:
        interval = 240
    return {
        "enabled": get_setting("privacy_auto_enabled", "1") == "1",
        "interval_min": max(30, interval),
        "last_run_at": get_setting("privacy_auto_last_run_at", ""),
        "last_added": get_setting("privacy_auto_last_added", ""),
    }


def privacy_auto_status():
    con = sqlite3.connect(DB, timeout=10)
    con.execute("PRAGMA busy_timeout=10000")
    _init_privacy_auto_table(con)
    auto_total = con.execute(
        "SELECT COUNT(*) FROM privacy_auto_v0 WHERE decision='auto_added'").fetchone()[0]
    con.close()
    return {"running": PRIVACY_AUTO_JOB["running"],
            "started_at": PRIVACY_AUTO_JOB["started_at"],
            "trigger": PRIVACY_AUTO_JOB["trigger"],
            "last_result": PRIVACY_AUTO_JOB["last_result"],
            "log_tail": PRIVACY_AUTO_JOB["log_tail"][-400:],
            "auto_total": auto_total,
            "pending": privacy_auto_pending_count(),
            **privacy_auto_config()}


def _privacy_auto_worker(trigger, full_vlm=False, wait_lock=False):
    """后台跑 privacy_auto_scan.py，解析结果摘要。全程不阻塞服务。"""
    # 2026-09-10 全局后台互斥（同 _vlm_worker）：不与其他后台任务并发抢写锁
    if not _BG_TASK_LOCK.acquire(timeout=30 if wait_lock else 0):
        PRIVACY_AUTO_JOB["running"] = False
        PRIVACY_AUTO_JOB["trigger"] = trigger
        PRIVACY_AUTO_JOB["last_result"] = "其他后台任务运行中，本轮跳过（稍后自动重试或手动重试）"
        print(f"[privacy-auto] 与其他后台任务冲突，跳过（trigger={trigger}）", flush=True)
        return
    PRIVACY_AUTO_JOB["running"] = True
    PRIVACY_AUTO_JOB["trigger"] = trigger
    PRIVACY_AUTO_JOB["started_at"] = now_iso()
    PRIVACY_AUTO_JOB["last_result"] = ""
    PRIVACY_AUTO_JOB["log_tail"] = ""
    try:
        cmd = _py_cmd("privacy")
        if full_vlm:
            cmd.append("--full-vlm")
        PRIVACY_AUTO_JOB["log_tail"] = f"[{trigger}] 开始检测{'（全量VLM）' if full_vlm else ''}\n"
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=43200)
        summary, added = {}, ""
        for line in (proc.stdout or "").splitlines():
            if line.startswith("PRIVACY_AUTO_RESULT: "):
                try:
                    summary = json.loads(line[len("PRIVACY_AUTO_RESULT: "):])
                except Exception:
                    pass
        if summary:
            added = str(summary.get("added", ""))
            vids = summary.get("vlm_videos", 0)
            vlm_part = (f"（视频 {vids}）" if vids else "") if summary.get("vlm_checked") else ""
            PRIVACY_AUTO_JOB["last_result"] = (
                f"新增移入 {summary.get('added', 0)} 项 · VLM 复核 {summary.get('vlm_checked', 0)} 项{vlm_part}"
                f" · 跳过 {summary.get('skipped', 0)} · 错误 {summary.get('errors', 0)}")
        else:
            tail = (proc.stdout or proc.stderr or "").strip().splitlines()[-3:]
            PRIVACY_AUTO_JOB["log_tail"] += "\n".join(tail) + "\n"
            PRIVACY_AUTO_JOB["last_result"] = f"完成（无摘要，exit={proc.returncode}）"
        set_setting("privacy_auto_last_added", added)
        print(f"[privacy-auto] {PRIVACY_AUTO_JOB['last_result']}", flush=True)
    except Exception as exc:
        PRIVACY_AUTO_JOB["last_result"] = f"失败: {exc}"
        PRIVACY_AUTO_JOB["log_tail"] += f"\n{exc}"
        print(f"[privacy-auto] error: {exc}", flush=True)
    finally:
        set_setting("privacy_auto_last_run_at", now_iso())
        PRIVACY_AUTO_JOB["running"] = False
        _BG_TASK_LOCK.release()


def privacy_auto_run(trigger="manual", full_vlm=False):
    """触发一次检测；已有任务在跑则返回当前状态。"""
    with _PRIVACY_AUTO_LOCK:
        if PRIVACY_AUTO_JOB["running"]:
            return {"started": False, **privacy_auto_status()}
        PRIVACY_AUTO_JOB["running"] = True  # 占位防并发双启动（worker 开头幂等置 True）
    threading.Thread(target=_privacy_auto_worker,
                     args=(trigger, full_vlm, trigger.startswith("manual")),
                     daemon=True, name="privacy-auto").start()
    return {"started": True, **privacy_auto_status()}


def _privacy_auto_loop():
    """后台线程：定时增量检测（新照片在 VLM 描述完成后即可被扫到）。
    2026-09-10 修 .app 启动锁死：①首轮延迟 = 正常间隔（原 300 秒）；
    ②自动触发只处理增量小批量（pending ≤ 300），全量欠账等手动触发。"""
    try:
        interval = max(30, int(get_setting("privacy_auto_interval_min", "240")))
    except Exception:
        interval = 240
    time.sleep(interval * 60)
    while True:
        try:
            if (get_setting("privacy_auto_enabled", "1") == "1"
                    and not PRIVACY_AUTO_JOB["running"]):
                n = privacy_auto_pending_count()
                if n > 300:
                    print(f"[privacy-auto] pending={n} 超过自动阈值 300，等待手动触发"
                          "（设置页 → 隐私检测 → 立即检测）", flush=True)
                elif n > 0:
                    print(f"[privacy-auto] 发现 {n} 张待检测，自动开始", flush=True)
                    _privacy_auto_worker("auto")
        except Exception as exc:
            print(f"[privacy-auto] error: {exc}", flush=True)
        try:
            interval = max(30, int(get_setting("privacy_auto_interval_min", "240")))
        except Exception:
            interval = 240
        time.sleep(interval * 60)


def privacy_auto_action(body):
    """隐私自动检测操作：status / scan / config。"""
    action = body.get("action")
    if action == "status":
        return privacy_auto_status()
    if action == "scan":
        return privacy_auto_run(trigger="manual-full" if body.get("full_vlm") else "manual",
                                full_vlm=bool(body.get("full_vlm")))
    if action == "config":
        return privacy_auto_config(enabled=body.get("enabled"),
                                   interval_min=body.get("interval_min"))
    raise ValueError("未知 action")


# ============ 本地物品语义检索 ============
def local_siglip_scores(texts, candidate_ids=None):
    """SIGLIP 纯本地文本-图像分数；返回 {asset_id: [score,...]}。"""
    import subprocess
    payload = json.dumps({"texts": texts, "candidate_ids": list(candidate_ids or [])}, ensure_ascii=False)
    script = r'''
import json,os,sqlite3,sys
import numpy as np,torch
os.environ['HF_HUB_OFFLINE']='1'
from transformers import AutoModel,AutoProcessor
db,payload=sys.argv[1],json.loads(sys.argv[2])
model_name=__SIGLIP_NAME__
model_dir=__SIGLIP_DIR__
con=sqlite3.connect(db)
rows=con.execute("SELECT subject_id,vector FROM embedding WHERE subject_type='asset' AND model_name=?",(model_name,)).fetchall()
wanted=set(payload.get('candidate_ids') or [])
if wanted: rows=[r for r in rows if r[0] in wanted]
if not rows: print('{}');raise SystemExit(0)
ids=[r[0] for r in rows]
X=np.stack([np.frombuffer(r[1],dtype='<f4') for r in rows])
X=X/(np.linalg.norm(X,axis=1,keepdims=True)+1e-9)
model=AutoModel.from_pretrained(model_dir,trust_remote_code=True,local_files_only=True)
proc=AutoProcessor.from_pretrained(model_dir,trust_remote_code=True,local_files_only=True)
model.eval()
with torch.no_grad():
 inp=proc(text=payload['texts'],padding='max_length',truncation=True,return_tensors='pt')
 T=model.get_text_features(**inp).cpu().numpy().astype('<f4')
T=T/(np.linalg.norm(T,axis=1,keepdims=True)+1e-9)
S=X@T.T
print(json.dumps({aid:[float(v) for v in S[i]] for i,aid in enumerate(ids)}))
'''
    # 注入 SigLIP2 模型名与本地权重目录（json.dumps 产出合法 Python 字面量）
    script = (script
              .replace("__SIGLIP_NAME__", json.dumps(SIGLIP_MODEL_NAME))
              .replace("__SIGLIP_DIR__", json.dumps(SIGLIP_MODEL_DIR)))
    try:
        # SigLIP 子进程解释器：默认用当前解释器（容器内 pip 装 torch+transformers 即可用，
        # 2026-09-17 修复：原默认 /usr/bin/python3 在容器里不存在 → 打分静默失败、标签全空）。
        # 也可用环境变量 SIGLIP_PYTHON 指向其它装有 torch+transformers 的解释器。
        p = subprocess.run([os.environ.get("SIGLIP_PYTHON") or sys.executable,
                            "-c", script, str(DB), payload],
                           capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            print(f"[siglip] 本地语义评分失败: {p.stderr[-300:]}")
            return {}
        return json.loads(p.stdout.strip() or "{}")
    except Exception as exc:
        print(f"[siglip] 本地语义评分异常: {exc}")
        return {}


def object_search(con, obj, question):
    """物品查询。两级：
    ① 预计算物品标签（object_tag_for_word 命中词表同义词）→ scene_tag_v0 精确过滤，
       置信度排序，可与其他维度组合；
    ② 词表外任意词 → 本地 SIGLIP 文本-图像相似度 Top-N 兜底（不上传家庭照片）。"""
    # 媒体类型
    want_video = "视频" in question or "录像" in question
    # 媒体类型过滤工具
    def _media_ok(aid):
        m = con.execute("SELECT media_type FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
        if not m:
            return False
        if want_video:
            return m["media_type"] == "video"
        return m["media_type"] == "photo"

    # 1) 预计算物品标签优先
    tag = object_tag_for_word(obj)
    if tag:
        rows = con.execute(
            "SELECT asset_id, confidence FROM scene_tag_v0 WHERE tag=? ORDER BY confidence DESC",
            (tag,)).fetchall()
        tagged = [r["asset_id"] for r in rows if _media_ok(r["asset_id"])]
        if tagged:
            out = []
            for r in rows:
                aid = r["asset_id"]
                if aid not in tagged:
                    continue
                ct = con.execute("SELECT capture_time FROM media_asset WHERE asset_id=?",
                                 (aid,)).fetchone()
                out.append({"id": aid, "time": ct["capture_time"] if ct else "",
                            "type": "video" if want_video else "photo", "hidden": False,
                            "semantic_score": r["confidence"]})
                if len(out) >= 30:
                    break
            return {"answer": f"找到 {len(out)} 张「{tag}」照片（预计算物品标签，按置信度排序）。",
                    "assets": out, "memory": [], "persons": [],
                    "intent": {"object": obj, "tag": tag, "object_tag": True}}

    # 2) 词表外：本地 SIGLIP 全库向量初筛。
    n_vectors = con.execute(
        "SELECT count(*) FROM embedding WHERE subject_type='asset' AND model_name=?",
        (SIGLIP_MODEL_NAME,)
    ).fetchone()[0]
    if not n_vectors:
        return {"answer": "没有可检索的照片。", "assets": [], "memory": [], "persons": [], "intent": {"object": obj}}

    score_rows = local_siglip_scores([obj, obj + " 物品", obj + " 特写"])
    if not score_rows:
        return {"answer": f"没有找到「{obj}」的照片。", "assets": [], "memory": [], "persons": [], "intent": {"object": obj}}
    candidates = dict(sorted(
        ((aid, max(scores)) for aid, scores in score_rows.items()),
        key=lambda item: -item[1]
    )[:40])

    # 2. Local First：只返回本地向量候选，默认禁止把家庭照片上传云端视觉模型。
    candidates_out = []
    for aid, sim in sorted(candidates.items(), key=lambda x: -x[1]):
        # 媒体类型过滤
        m = con.execute("SELECT media_type FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
        if m and want_video and m["media_type"] != "video":
            continue
        if m and not want_video and m["media_type"] != "photo":
            continue
        ct = con.execute("SELECT capture_time FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
        candidates_out.append({"id": aid, "time": ct["capture_time"] if ct else "",
                               "type": m["media_type"] if m else "photo", "hidden": False,
                               "semantic_score": sim})
        if len(candidates_out) >= 15:
            break

    if candidates_out:
        if obj in ("蛋糕", "下雨"):
            ans = (f"找到 {len(candidates_out)} 个本地语义候选。独立盲审显示「{obj}」"
                   "属于当前模型的低可靠类别，不能当作已确认标签。")
        else:
            ans = f"找到 {len(candidates_out)} 个本地语义候选，按与「{obj}」的相似度排序；尚未经人工逐张确认。"
        return {"answer": ans, "assets": candidates_out, "memory": [], "persons": [],
                "intent": {"object": obj, "local_only": True, "model": SIGLIP_MODEL_NAME}}
    return {"answer": f"没有找到「{obj}」的照片。", "assets": [], "memory": [], "persons": [], "intent": {"object": obj}}


# ============ 图片工具跨平台层（2026-09-08 NAS Docker 化）============
# macOS：优先 sips（零依赖，命令格式与历史完全一致，行为零变化）。
# Linux 容器：无 sips，走 Pillow（EXIF 转正 + HEIC 解码 pillow-heif）。
# Pillow 为懒加载可选依赖，不影响 server.py「零硬性第三方依赖」原则。
SIPS_BIN = shutil.which("sips")


def _pil_load(path):
    """Pillow 打开图片：注册 HEIC 解码，应用 EXIF orientation 转正。"""
    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    im = Image.open(str(path))
    im = ImageOps.exif_transpose(im)  # 转正（sips 老路径由 ffmpeg _upright_photo 上游处理）
    return im


def resize_to_jpeg(src, dst, max_edge, quality=None):
    """任意图片 → 最长边 max_edge 的 JPEG。
    sips 可用走原命令格式（quality=None 时不带 formatOptions，与历史一致）；
    否则 Pillow：thumbnail 只缩不放（小图不糊化），quality 默认 85。
    2026-09-12 缩略图"自动放大"修复：改为先写临时文件、写完 os.replace 原子替换。
    原实现直接写最终路径，首页并发洪水时另一个请求在文件写到一半时 exists()
    判定为命中，把半截 JPEG 发给浏览器（只能解码出顶部几行 → 铺满卡片看起来
    放大好多倍），且响应带 max-age=3600 被浏览器缓存 1 小时。"""
    import threading as _th
    tmp = Path(str(dst) + f".tmp{_th.get_ident()}")
    try:
        if SIPS_BIN:
            cmd = [SIPS_BIN, "-Z", str(max_edge), "-s", "format", "jpeg"]
            if quality is not None:
                cmd += ["-s", "formatOptions", str(quality)]
            cmd += [str(src), "--out", str(tmp)]
            subprocess.run(cmd, capture_output=True, timeout=60, check=True)
        else:
            from PIL import Image
            im = _pil_load(src)
            im.thumbnail((max_edge, max_edge), Image.LANCZOS)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            q = 85 if quality is None else int(quality)
            im.save(str(tmp), "JPEG", quality=q)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def image_dims(path):
    """读图片像素尺寸 (W, H)。sips 可用走原命令；否则 Pillow。"""
    if SIPS_BIN:
        r = subprocess.run([SIPS_BIN, "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
                           capture_output=True, text=True, timeout=10)
        w = h = 0
        for line in r.stdout.splitlines():
            m = re.match(r"\s*pixelWidth:\s*(\d+)", line)
            if m:
                w = int(m.group(1))
            m = re.match(r"\s*pixelHeight:\s*(\d+)", line)
            if m:
                h = int(m.group(1))
        return (w, h)
    try:
        im = _pil_load(path)
        return im.size
    except Exception:
        return (0, 0)


def _img_b64(path, px=400):
    """压缩成小图并返回 base64（视频取第 1 秒帧），避免大图 API 返回空。"""
    import base64
    import subprocess as sp
    import threading
    # 2026-09-02 修复: 原用 os.getpid() 命名, ThreadingHTTPServer 多线程并发时
    # 不同请求互相覆盖临时文件, 可能把 A 图发给 VLM 当 B 图看。加线程 id 隔离。
    t = f"/tmp/obj_check_{os.getpid()}_{threading.get_ident()}.jpg"
    try:
        if path.lower().endswith((".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm")):
            sp.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
                    "-ss", "1", "-i", path, "-frames:v", "1", "-vf", f"scale={px}:-2",
                    "-f", "image2", t], capture_output=True, timeout=20, check=True)
        else:
            resize_to_jpeg(path, t, px)
        tmp_path = t
    except Exception:
        tmp_path = path
    try:
        with open(tmp_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception:
        return None


def vision_region_verify(path, place_desc):
    """视觉模型验证照片场景与推断地区是否相符（辅助验证同日互证结果）。
    返回 'match' / 'mismatch' / 'uncertain' / None(调用失败)。"""
    b64 = _img_b64(path)
    if not b64:
        return None
    text, err = llm_chat([{"role": "user", "content": [
        {"type": "text", "text": f"请判断这张照片的拍摄场景（地貌、植被、建筑风格、可见地标）与「{place_desc}」是否相符。请只回答三个词之一：相符 / 不符 / 不确定。不要解释。"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}], vision=True)
    if err == "no_balance":
        return "no_balance"  # 账号问题（余额/鉴权）：调用方应降级本地 SIGLIP
    if not text:
        return None
    if "不符" in text:
        return "mismatch"
    if "相符" in text or "符合" in text:
        return "match"
    return "uncertain"


# ============ 分类检索 ============
def get_categories():
    """返回动态分类维度：总量 / 来源成员 / 人物 / 地点 / 场景标签 / 自定义分类。

    数量口径统一为「墙面可见」：排除过滤表/相似组非最佳，白名单可恢复，
    与照片墙 renderRibbon 实际展示一致。此前 scene 标签数直接
    GROUP BY scene_tag_v0，把被过滤资产也算进去，出现「标签写 76 张、
    点开只剩 7 张」的错位（截图 179 张里 172 张已隐藏）。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    # 与 get_hidden_ids 同口径：过滤表 + 相似组非最佳 - 白名单
    hidden = get_hidden_ids(con)
    con.execute("CREATE TEMP TABLE IF NOT EXISTS _hidden_ids_v0(asset_id TEXT PRIMARY KEY)")
    con.execute("DELETE FROM _hidden_ids_v0")
    con.executemany("INSERT OR IGNORE INTO _hidden_ids_v0 VALUES(?)", [(a,) for a in hidden])
    NH = "NOT EXISTS(SELECT 1 FROM _hidden_ids_v0 h WHERE h.asset_id=ma.asset_id)"
    years = [dict(r) for r in con.execute(
        f"""SELECT substr(ma.capture_time,1,4) as year, COUNT(*) as count FROM media_asset ma
           WHERE ma.capture_time IS NOT NULL AND {NH} GROUP BY year ORDER BY year DESC""")]
    locations = [dict(r) for r in con.execute(
        f"""SELECT g.region, COUNT(DISTINCT g.asset_id) as count FROM asset_geo_v0 g
           JOIN media_asset ma ON ma.asset_id=g.asset_id
           WHERE {NH} GROUP BY g.region ORDER BY count DESC""")]
    # 2026-09-07 15:30：地点壳补显。「选省/市添加场地」只写 category_layout_v0(kind='location') 空壳，
    # 没照片时上面按 asset_geo_v0 的查询不返回它 → 侧栏「地点」里看不见这个场地（用户以为没建成）。
    # 把 layout 里有、照片里没有的场地以 count=0 补进数组；target 以 cat_ 开头的是历史迁移残留
    # （target 误存了旧分类 id），跳过不显示。
    try:
        _have = {r["region"] for r in locations}
        for r in con.execute("SELECT target FROM category_layout_v0 WHERE kind='location'").fetchall():
            t = (r["target"] or "").strip()
            if t and not t.startswith("cat_") and t not in _have:
                locations.append({"region": t, "count": 0})
    except sqlite3.OperationalError:
        pass
    persons = [dict(r) for r in con.execute(
        f"""SELECT p.display_name, p.person_id, COUNT(DISTINCT fi.asset_id) as count
           FROM person p JOIN face_instance_v0 fi USING(person_id)
           JOIN media_asset ma ON ma.asset_id=fi.asset_id
           WHERE {NH} GROUP BY p.person_id ORDER BY count DESC""")]
    # 来源别名：owner_label ↔ 显示名映射在 person_alias(kind='source')，代码零人名
    source_alias = _person_lexicon().get("source_aliases", {})
    sources = []
    for r in con.execute(
        f"""SELECT s.owner_label,s.root_path,COUNT(DISTINCT mf.asset_id) count
           FROM source s LEFT JOIN media_file mf USING(source_id)
           LEFT JOIN media_asset ma ON ma.asset_id=mf.asset_id
           WHERE ma.asset_id IS NULL OR {NH}
           GROUP BY s.source_id ORDER BY count DESC"""):
        raw = r["owner_label"] or Path(r["root_path"]).parent.name
        sources.append({"display_name": source_alias.get(raw, raw), "source_value": raw, "count": r["count"]})
    total = con.execute(
        f"SELECT count(*) FROM media_asset ma WHERE {NH}").fetchone()[0]
    media_types = {r[0]: r[1] for r in con.execute(
        f"SELECT media_type,count(*) FROM media_asset ma WHERE {NH} GROUP BY media_type")}
    # 多标签：一张照片可同时出现在 合影/旅行/山景/海景 等多个标签下；
    # 只统计墙面可见资产，可见数为 0 的标签（如截图/滑雪）不再出现在墙上
    try:
        scenes = [dict(r) for r in con.execute(
            f"""SELECT st.tag, COUNT(DISTINCT st.asset_id) as count FROM scene_tag_v0 st
               JOIN media_asset ma ON ma.asset_id=st.asset_id
               WHERE {NH} GROUP BY st.tag HAVING count>0 ORDER BY count DESC""")]
    except sqlite3.OperationalError:
        scenes = []
    # 自定义分类：用户自建；成员数为该分类下全部成员（用户修正意图优先于可见集），
    # 但回收站/隐私相册成员不计（点开分类不会显示），与 /api/catman、/api/category 口径一致
    try:
        customs = [dict(r) for r in con.execute(
            """SELECT c.category_id, c.name, COUNT(m.asset_id) AS count
               FROM user_category_v0 c
               LEFT JOIN user_category_member_v0 m ON m.category_id=c.category_id
                 AND NOT EXISTS(SELECT 1 FROM recycle_v0 r WHERE r.asset_id=m.asset_id)
                 AND NOT EXISTS(SELECT 1 FROM privacy_v0 p WHERE p.asset_id=m.asset_id)
               GROUP BY c.category_id ORDER BY c.sort_order, c.created_at""")]
    except sqlite3.OperationalError:
        customs = []
    # 分类展示配置（层级 parent / 排序 sort / 首页显示 show），供首页与总库侧栏渲染
    layout = {}
    groups_view = []
    try:
        _init_cat_layout_table(con)
        groups_view = _cat_group_view(con)
        # source 必须含在内：否则来源项的 layout 读不回来，排序/改层级全部无效
        for kind in ("person", "location", "scene", "custom", "source"):
            layout[kind] = {r["target"]: {"parent": r["parent"], "sort": r["sort_order"],
                                          "show": r["show_on_home"]} for r in con.execute(
                "SELECT target,parent,sort_order,show_on_home FROM category_layout_v0 WHERE kind=?",
                (kind,)).fetchall()}
    except Exception:
        layout = {}
    con.execute("DROP TABLE IF EXISTS _hidden_ids_v0")
    con.close()
    return {"total": total, "years": years, "locations": locations, "persons": persons,
            "sources": sources, "media_types": media_types, "scenes": scenes,
            "customs": customs, "layout": layout, "groups": groups_view}


def get_people_payload():
    """Confirmed people plus still-unresolved Top-30 cluster candidates."""
    con = sqlite3.connect(DB, timeout=20)
    con.row_factory = sqlite3.Row
    people = []
    for p in con.execute(
        """SELECT p.*,count(DISTINCT fi.asset_id) asset_count,count(fi.face_instance_id) face_count
           FROM person p LEFT JOIN face_instance_v0 fi USING(person_id)
           GROUP BY p.person_id ORDER BY asset_count DESC,p.display_name"""
    ):
        item = dict(p)
        item["sample_assets"] = [r[0] for r in con.execute(
            "SELECT DISTINCT asset_id FROM face_instance_v0 WHERE person_id=? LIMIT 4", (p["person_id"],))]
        # 头像直接给人脸裁切(带 bbox)，前端缩放到脸部区域，避免整图看不出是谁
        item["sample_faces"] = []
        for r in con.execute(
            """SELECT fi.face_instance_id, fi.asset_id, fi.bbox_json
               FROM face_instance_v0 fi WHERE fi.person_id=?
               ORDER BY CASE fi.quality_class WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                        fi.face_width*fi.face_height DESC LIMIT 4""",
                (p["person_id"],)):
            sf = dict(r)
            sf["bbox"] = json.loads(sf.pop("bbox_json"))
            item["sample_faces"].append(sf)
        people.append(item)
    candidates = []
    for c in con.execute(
        """SELECT * FROM anonymous_person_cluster_v0
           WHERE hypothesis_status='candidate'
           ORDER BY asset_count DESC,member_count DESC LIMIT 30"""
    ):
        item = dict(c)
        item["sample_assets"] = [r[0] for r in con.execute(
            """SELECT DISTINCT fi.asset_id FROM anonymous_person_membership_v0 m
               JOIN face_instance_v0 fi USING(face_instance_id)
               WHERE m.cluster_id=? LIMIT 6""", (c["cluster_id"],))]
        # 待确认组样例改人脸裁切：整图里多脸/无人脸都会误导，只看「这组到底是哪张脸」
        item["samples"] = []
        for r in con.execute(
            """SELECT fi.face_instance_id, fi.asset_id, fi.bbox_json, ma.capture_time
               FROM anonymous_person_membership_v0 m
               JOIN face_instance_v0 fi USING(face_instance_id)
               JOIN media_asset ma ON ma.asset_id=fi.asset_id
               WHERE m.cluster_id=? AND fi.person_id IS NULL
               ORDER BY CASE fi.quality_class WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                        fi.face_width*fi.face_height DESC LIMIT 6""",
                (c["cluster_id"],)):
            s = dict(r)
            s["bbox"] = json.loads(s.pop("bbox_json"))
            item["samples"].append(s)
        candidates.append(item)
    processed = con.execute(
        "SELECT count(*) FROM person_asset_processing_v0 WHERE status='success'"
    ).fetchone()[0]
    total = con.execute("SELECT count(*) FROM media_asset").fetchone()[0]
    unresolved_faces = []
    for face in con.execute(
        """SELECT fi.face_instance_id,fi.asset_id,fi.bbox_json,fi.detection_score,
                  fi.quality_class,ma.capture_time,ma.media_type
           FROM face_instance_v0 fi JOIN media_asset ma USING(asset_id)
           WHERE fi.person_id IS NULL
           ORDER BY CASE fi.quality_class WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                    fi.face_width*fi.face_height DESC,fi.detection_score DESC LIMIT 5000"""
    ):
        item = dict(face)
        item["bbox"] = json.loads(item.pop("bbox_json"))
        unresolved_faces.append(item)
    unresolved_count = con.execute(
        "SELECT count(*) FROM face_instance_v0 WHERE person_id IS NULL"
    ).fetchone()[0]
    con.close()
    return {"people": people, "candidates": candidates, "processed_assets": processed,
            "total_assets": total, "max_people": 30, "unresolved_faces": unresolved_faces,
            "unresolved_face_count": unresolved_count}


def _unblob(blob, dim):
    """将 BLOB 里的 float32 向量解压为 list。不用 numpy，因为 server 的运行时没装它。"""
    import struct
    n = len(blob) // 4
    if n > dim:
        n = dim
    return list(struct.unpack(f'<{n}f', blob[:n*4]))

def _norm(v):
    m = math.sqrt(sum(x*x for x in v)) or 1
    return [x/m for x in v]

def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def get_person_detail(person_id, confirmed_limit=16):
    """人物详情页：人物信息 + 已确认人脸样例 + 未标注人脸推荐（向量质心 Top-K）。"""
    con = sqlite3.connect(DB, timeout=20)
    con.row_factory = sqlite3.Row
    person = con.execute("SELECT * FROM person WHERE person_id=?", (person_id,)).fetchone()
    if not person:
        con.close()
        raise ValueError("人物不存在")
    person = dict(person)
    counts = con.execute(
        "SELECT COUNT(DISTINCT asset_id) asset_count, COUNT(*) face_count FROM face_instance_v0 WHERE person_id=?",
        (person_id,)).fetchone()
    person.update(dict(counts))

    # 已确认人脸样例：质量优先，用于页面展示（人脸池页可传 confirmed_limit 拉全量做移除管理）
    confirmed = []
    for r in con.execute(
        """SELECT fi.face_instance_id, fi.asset_id, fi.bbox_json, fi.quality_class, ma.capture_time
           FROM face_instance_v0 fi JOIN media_asset ma USING(asset_id)
           WHERE fi.person_id=?
           ORDER BY CASE fi.quality_class WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                    fi.face_width*fi.face_height DESC LIMIT ?""",
        (person_id, min(max(int(confirmed_limit or 16), 1), 50000))):
        s = dict(r)
        s["bbox"] = json.loads(s.pop("bbox_json"))
        confirmed.append(s)

    # 未标注人脸推荐：基于该人物所有确认脸的向量质心，找最相似的未归属脸
    suggestions = []
    ref_rows = con.execute(
        """SELECT fe.embedding FROM face_instance_v0 fi
           JOIN face_embedding_v0 fe USING(face_instance_id)
           WHERE fi.person_id=? AND fe.status='success'""",
        (person_id,)).fetchall()
    if ref_rows:
        dim = 128  # SFace-2021dec
        ref_vecs = [_unblob(r["embedding"], dim) for r in ref_rows]
        if not ref_vecs:
            ref_vecs = [[0.0]*dim]
        # 质心
        centroid = [sum(v[i] for v in ref_vecs)/len(ref_vecs) for i in range(dim)]
        centroid = _norm(centroid)
        rows = con.execute(
            """SELECT fi.face_instance_id, fi.asset_id, fi.bbox_json, ma.capture_time,
                      fe.embedding
               FROM face_instance_v0 fi
               JOIN media_asset ma USING(asset_id)
               JOIN face_embedding_v0 fe USING(face_instance_id)
               WHERE fi.person_id IS NULL AND fe.status='success'
                 AND fi.quality_class IN ('high','medium','usable')
               ORDER BY fi.face_width*fi.face_height DESC, fi.detection_score DESC
               LIMIT 2000""").fetchall()
        scored = []
        for r in rows:
            v = _norm(_unblob(r["embedding"], dim))
            sim = _dot(centroid, v)
            if sim >= 0.55:
                item = dict(r)
                item["bbox"] = json.loads(item.pop("bbox_json"))
                item.pop("embedding", None)
                item["sim"] = round(sim, 4)
                scored.append(item)
        scored.sort(key=lambda x: x["sim"], reverse=True)
        suggestions = scored[:40]
    con.close()
    return {"person": person, "confirmed_faces": confirmed, "suggestions": suggestions}


def get_similar_unlabeled_faces(person_id, threshold=0.50, limit=1000):
    """为指定人物推荐最相似的未标注人脸（人脸池模式）。

    基于该人物所有确认脸的 SFace 向量质心，计算全部未标注人脸的 cosine 相似度，
    按相似度从高到低返回。阈值可调，用于「点谁进来看谁的池子」。
    2026-09-07 v2：默认阈值 0.40→0.50（0.40 会把成年男性推荐进儿童池）；
    新增出生日期硬门——拍摄时间早于出生日期的脸物理上不可能是本人，直接排除。"""
    con = sqlite3.connect(DB, timeout=20)
    con.row_factory = sqlite3.Row
    try:
        if not con.execute("SELECT 1 FROM person WHERE person_id=?", (person_id,)).fetchone():
            raise ValueError("人物不存在")
        ref_rows = con.execute(
            """SELECT fe.embedding FROM face_instance_v0 fi
               JOIN face_embedding_v0 fe USING(face_instance_id)
               WHERE fi.person_id=? AND fe.status='success'""",
            (person_id,)).fetchall()
        if not ref_rows:
            # 该人物从未添加过人脸：没有质心可算相似度，兜底返回全部未标注脸，
            # 按质量/人脸大小排序，让用户手动点选建立第一批样本
            rows = con.execute(
                """SELECT fi.face_instance_id, fi.asset_id, fi.bbox_json, fi.quality_class,
                          fi.detection_score, ma.capture_time
                   FROM face_instance_v0 fi
                   JOIN media_asset ma USING(asset_id)
                   JOIN face_embedding_v0 fe USING(face_instance_id)
                   WHERE fi.person_id IS NULL AND fe.status='success'
                   ORDER BY CASE fi.quality_class WHEN 'high' THEN 1 WHEN 'medium' THEN 2
                            WHEN 'usable' THEN 3 ELSE 4 END,
                            fi.face_width*fi.face_height DESC
                   LIMIT ?""", (limit,)).fetchall()
            faces = []
            for r in rows:
                item = dict(r)
                item["bbox"] = json.loads(item.pop("bbox_json"))
                item["sim"] = None
                faces.append(item)
            return {"person_id": person_id, "threshold": threshold, "faces": faces,
                    "total": len(faces), "sorted_by": "quality"}

        dim = 128  # SFace-2021dec
        # 出生日期硬门：拍摄时间早于出生日期 → 不可能是本人
        birth = (con.execute("SELECT birth_date FROM person WHERE person_id=?",
                             (person_id,)).fetchone()[0] or "")[:10]
        ref_vecs = [_unblob(r["embedding"], dim) for r in ref_rows]
        centroid = [sum(v[i] for v in ref_vecs)/len(ref_vecs) for i in range(dim)]
        centroid = _norm(centroid)
        total = con.execute(
            """SELECT COUNT(*) FROM face_instance_v0 fi
               JOIN face_embedding_v0 fe USING(face_instance_id)
               WHERE fi.person_id IS NULL AND fe.status='success'""").fetchone()[0]

        rows = con.execute(
            """SELECT fi.face_instance_id, fi.asset_id, fi.bbox_json, fi.quality_class,
                      fi.detection_score, ma.capture_time, fe.embedding
               FROM face_instance_v0 fi
               JOIN media_asset ma USING(asset_id)
               JOIN face_embedding_v0 fe USING(face_instance_id)
               WHERE fi.person_id IS NULL AND fe.status='success'
               ORDER BY fi.face_width*fi.face_height DESC, fi.detection_score DESC
               LIMIT 9000""").fetchall()
        scored = []
        for r in rows:
            # 出生门：照片早于出生日期 → 物理不可能，跳过（无时间/异常时间的不管）
            ct = r["capture_time"] or ""
            if birth and ct >= "1000" and ct[:10] < birth:
                continue
            v = _norm(_unblob(r["embedding"], dim))
            sim = _dot(centroid, v)
            # threshold<=0 表示不过滤：加载全部未标注脸，仅按相似度降序排列
            if threshold > 0 and sim < threshold:
                continue
            item = dict(r)
            item["bbox"] = json.loads(item.pop("bbox_json"))
            item.pop("embedding", None)
            item["sim"] = round(sim, 4)
            scored.append(item)
        scored.sort(key=lambda x: x["sim"], reverse=True)
        return {"person_id": person_id, "threshold": threshold, "faces": scored[:limit], "total": total}
    finally:
        con.close()


def _merge_custom_into_location(con, cid, name, now):
    """重名自动合并（2026-09-07 15:30）：把自定义分类并入同名地点。

    场景：用户手动定位添加了场地「青岛」，又自建了自定义分类「青岛」→ 侧栏出现两个青岛。
    合并方向统一为「自定义并入地点」（与手动定位的产品方向一致）：
    - 成员照片 region 为空的补 region=name（USER_MOVE，用户意图优先）；
    - 已有真实地区的照片不动（不抢 GPS 归属，照片可同时留在多个地区语义下）；
    - 删除该自定义分类及其成员表、布局行。"""
    for (aid,) in con.execute("SELECT asset_id FROM user_category_member_v0 WHERE category_id=?",
                              (cid,)).fetchall():
        cur = con.execute("SELECT region FROM asset_geo_v0 WHERE asset_id=?", (aid,)).fetchone()
        if cur is None:
            con.execute("INSERT INTO asset_geo_v0(asset_id,region,location_source,created_at) "
                        "VALUES(?,?,'USER_MOVE',?)", (aid, name, now))
        elif not cur["region"]:
            con.execute("UPDATE asset_geo_v0 SET region=?, location_source='USER_MOVE' WHERE asset_id=?",
                        (name, aid))
    con.execute("DELETE FROM user_category_member_v0 WHERE category_id=?", (cid,))
    con.execute("DELETE FROM user_category_v0 WHERE category_id=?", (cid,))
    con.execute("DELETE FROM category_layout_v0 WHERE kind='custom' AND target=?", (cid,))


def _location_conflict(con, name):
    """名字是否已作为地点存在（真实地区或地点壳）。"""
    if con.execute("SELECT 1 FROM asset_geo_v0 WHERE region=? LIMIT 1", (name,)).fetchone():
        return True
    return bool(con.execute("SELECT 1 FROM category_layout_v0 WHERE kind='location' AND target=?",
                            (name,)).fetchone())


def _custom_cid_by_name(con, name, exclude_cid=None):
    """按名字查自定义分类 id（可选排除自身），用于 rename/create 撞名合并。"""
    row = con.execute("SELECT category_id FROM user_category_v0 WHERE name=? AND category_id<>? LIMIT 1",
                      (name, exclude_cid or "")).fetchone()
    return row["category_id"] if row else None


def custom_category_action(body):
    """自定义分类 CRUD：create / rename / delete / add_assets / remove_assets / set_parent / set_visible。

    层级通过 category_layout_v0.parent 字段表达：''=顶层（挂在「全部影像」下作二级），
    否则为父 category_id（作三级或更深）。show_on_home 控制该分类照片是否进入「全部影像」照片墙。
    所有写入动作同步刷新 get_categories() 数据（customs 维度）。"""
    action = body.get("action")
    if action not in ("create", "rename", "delete", "add_assets", "remove_assets", "move_to", "reorder", "set_parent", "set_visible"):
        raise ValueError("未知 action")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _init_user_category_table(con)
    _init_recycle_table(con)
    _init_cat_layout_table(con)
    family_id = con.execute("SELECT family_id FROM family LIMIT 1").fetchone()["family_id"]
    now = now_iso()
    if action == "create":
        name = (body.get("name") or "").strip()
        if not name:
            raise ValueError("分类名不能为空")
        existing = con.execute("SELECT category_id FROM user_category_v0 WHERE name=? AND family_id=?",
                                (name, family_id)).fetchone()
        if existing:
            return {"category_id": existing["category_id"], "name": name, "existed": True}
        # 重名自动合并：名字已作为地点存在（真实地区/地点壳）→ 不建重复的自定义分类，
        # 让前端直接落到对应地点（新建时还没有成员，无需搬迁照片）
        if _location_conflict(con, name):
            return {"merged": "location", "target": name, "name": name}
        cid = "cat_" + uuid.uuid4().hex[:10]
        max_order = con.execute("SELECT COALESCE(MAX(sort_order),0) FROM user_category_v0").fetchone()[0]
        con.execute("INSERT INTO user_category_v0(category_id,family_id,name,sort_order,created_at) "
                    "VALUES(?,?,?,?,?)", (cid, family_id, name, max_order + 1, now))
        # 新建分类默认挂「全部影像」下作二级（parent=''）；可传 parent 指定父分类作三级
        parent = (body.get("parent") or "").strip()
        con.execute("""INSERT INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at)
                       VALUES('custom',?,?,1,1,?)
                       ON CONFLICT(kind,target) DO UPDATE SET parent=excluded.parent,updated_at=excluded.updated_at""",
                    (cid, parent, now))
        con.commit()
        return {"category_id": cid, "name": name}
    if action == "rename":
        cid = body.get("category_id")
        name = (body.get("name") or "").strip()
        if not cid or not name:
            raise ValueError("缺少 category_id 或 name")
        # 重名自动合并（2026-09-07 15:30）：
        # ① 撞其他自定义分类 → 成员并入对方，删自身；
        # ② 撞同名地点 → 整个并入地点（成员照片 region 归位），删自身；
        # ③ 无冲突 → 普通改名。
        other_cid = _custom_cid_by_name(con, name, exclude_cid=cid)
        if other_cid:
            con.execute("""INSERT OR IGNORE INTO user_category_member_v0(category_id,asset_id,added_by,added_at)
                           SELECT ?, asset_id, added_by, added_at FROM user_category_member_v0
                           WHERE category_id=?""", (other_cid, cid))
            con.execute("DELETE FROM user_category_member_v0 WHERE category_id=?", (cid,))
            con.execute("DELETE FROM user_category_v0 WHERE category_id=?", (cid,))
            con.execute("DELETE FROM category_layout_v0 WHERE kind='custom' AND target=?", (cid,))
            con.commit()
            return {"category_id": other_cid, "name": name, "merged": "custom"}
        if _location_conflict(con, name):
            _merge_custom_into_location(con, cid, name, now)
            con.commit()
            return {"category_id": cid, "name": name, "merged": "location", "target": name}
        con.execute("UPDATE user_category_v0 SET name=? WHERE category_id=?", (name, cid))
        con.commit()
        return {"category_id": cid, "name": name}
    if action == "delete":
        cid = body.get("category_id")
        if not cid:
            raise ValueError("缺少 category_id")
        con.execute("DELETE FROM user_category_member_v0 WHERE category_id=?", (cid,))
        con.execute("DELETE FROM user_category_v0 WHERE category_id=?", (cid,))
        con.commit()
        return {"category_id": cid, "deleted": True}
    if action == "add_assets":
        cid = body.get("category_id")
        asset_ids = body.get("asset_ids") or []
        if not cid or not asset_ids:
            raise ValueError("缺少 category_id 或 asset_ids")
        if not con.execute("SELECT 1 FROM user_category_v0 WHERE category_id=?", (cid,)).fetchone():
            raise ValueError("分类不存在")
        rows = [(cid, a, "user", now) for a in asset_ids if a]
        con.executemany("INSERT OR IGNORE INTO user_category_member_v0(category_id,asset_id,added_by,added_at) "
                         "VALUES(?,?,?,?)", rows)
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM user_category_member_v0 WHERE category_id=?", (cid,)).fetchone()[0]
        return {"category_id": cid, "added": len(rows), "count": count}
    if action == "remove_assets":
        cid = body.get("category_id")
        asset_ids = body.get("asset_ids") or []
        if not cid or not asset_ids:
            raise ValueError("缺少 category_id 或 asset_ids")
        con.executemany("DELETE FROM user_category_member_v0 WHERE category_id=? AND asset_id=?",
                         [(cid, a) for a in asset_ids if a])
        con.commit()
        count = con.execute("SELECT COUNT(*) FROM user_category_member_v0 WHERE category_id=?", (cid,)).fetchone()[0]
        return {"category_id": cid, "removed": len(asset_ids), "count": count}
    if action == "reorder":
        # 自定义分类拖拽排序：order 数组给出新的 category_id 顺序。
        # parent 指定本批排序所属的父层级（''=顶层平铺，否则为父 category_id）。
        # 同一 parent 下的分类按 order 重排；不同 parent 的不动。
        order = body.get("order") or []
        parent = (body.get("parent") or "").strip()
        if not order:
            raise ValueError("缺少 order 数组")
        valid = {r["category_id"] for r in con.execute("SELECT category_id FROM user_category_v0")}
        seen = set()
        sanitized = []
        for cid in order:
            if cid in valid and cid not in seen:
                seen.add(cid); sanitized.append(cid)
        for cid in valid - seen:
            sanitized.append(cid)
        now2 = now_iso()
        for idx, cid in enumerate(sanitized, 1):
            con.execute("""INSERT INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at)
                           VALUES('custom',?,?,?,?,?)
                           ON CONFLICT(kind,target) DO UPDATE SET sort_order=excluded.sort_order,
                                                                  parent=excluded.parent,
                                                                  updated_at=excluded.updated_at""",
                        (cid, parent, idx, 1, now2))
        con.commit()
        con.close()
        return {"reordered": len(sanitized)}
    if action == "set_parent":
        # 改层级：把 custom 移到新父级下（''=提到顶层）。父级可以是另一个 custom 的
        # category_id（作它的子级），也可以是系统组 key（person/location/scene/…，挂进该组当子项）。
        cid = body.get("category_id")
        parent = (body.get("parent") or "").strip()
        if not cid:
            raise ValueError("缺少 category_id")
        if parent and parent == cid:
            raise ValueError("不能把分类设为自己的父级")
        parent_is_custom = False
        if parent:
            parent_is_custom = bool(con.execute(
                "SELECT 1 FROM user_category_v0 WHERE category_id=?", (parent,)).fetchone())
            if not parent_is_custom and parent not in {g["key"] for g in _CAT_GROUP_META}:
                raise ValueError("父级不存在")
            if parent_is_custom:
                # 防环：parent 不能是 cid 的后代
                seen = set()
                cur = parent
                while cur and cur not in seen:
                    seen.add(cur)
                    row = con.execute("SELECT parent FROM category_layout_v0 WHERE kind='custom' AND target=?", (cur,)).fetchone()
                    cur = row["parent"] if row else ''
                    if cur == cid:
                        raise ValueError("不能把分类移到自己的子级下（会成环）")
        con.execute("""INSERT INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at)
                       VALUES('custom',?,?,1,1,?)
                       ON CONFLICT(kind,target) DO UPDATE SET parent=excluded.parent,updated_at=excluded.updated_at""",
                    (cid, parent, now))
        con.commit()
        return {"category_id": cid, "parent": parent}
    if action == "set_visible":
        # 显隐开关：控制该分类照片是否进入「全部影像」照片墙（show_on_home）。
        cid = body.get("category_id")
        show = 1 if body.get("show") else 0
        if not cid:
            raise ValueError("缺少 category_id")
        con.execute("""INSERT INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at)
                       VALUES('custom',?,'',1,?,?)
                       ON CONFLICT(kind,target) DO UPDATE SET show_on_home=excluded.show_on_home,updated_at=excluded.updated_at""",
                    (cid, show, now))
        con.commit()
        return {"category_id": cid, "show": show}
    if action == "move_to":
        # 整个自定义分类移动到目标：照片从本分类移出，并写入目标
        # 目标类型：custom(另一自定义分类) / scene(场景标签) / location(地点 region)
        cid = body.get("category_id")
        target_type = body.get("target_type")
        target = (body.get("target") or "").strip()
        if not cid or target_type not in ("custom", "scene", "location") or not target:
            raise ValueError("缺少 category_id / target_type / target")
        if not con.execute("SELECT 1 FROM user_category_v0 WHERE category_id=?", (cid,)).fetchone():
            raise ValueError("分类不存在")
        member_ids = [r["asset_id"] for r in con.execute(
            "SELECT asset_id FROM user_category_member_v0 WHERE category_id=?", (cid,))]
        if not member_ids:
            return {"category_id": cid, "moved": 0, "count": 0}
        if target_type == "custom":
            if cid == target:
                raise ValueError("目标不能是当前分类本身")
            if not con.execute("SELECT 1 FROM user_category_v0 WHERE category_id=?", (target,)).fetchone():
                raise ValueError("目标分类不存在")
            con.executemany("INSERT OR IGNORE INTO user_category_member_v0(category_id,asset_id,added_by,added_at) "
                            "VALUES(?,?,?,?)", [(target, a, "user", now) for a in member_ids])
        elif target_type == "scene":
            con.executemany("INSERT OR IGNORE INTO scene_tag_v0(asset_id,tag,source,confidence,created_at) "
                            "VALUES(?,?,?,?,?)", [(a, target, "USER_MOVE", 1.0, now) for a in member_ids])
        elif target_type == "location":
            con.executemany(
                "INSERT INTO asset_geo_v0(asset_id,region,location_source,created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(asset_id) DO UPDATE SET region=excluded.region, location_source='USER_MOVE'",
                [(a, target, "USER_MOVE", now) for a in member_ids])
    con.executemany("DELETE FROM user_category_member_v0 WHERE category_id=? AND asset_id=?",
                     [(cid, a) for a in member_ids])
    con.commit()
    result = {"category_id": cid, "target_type": target_type, "target": target,
              "moved": len(member_ids), "count": 0}
    con.close()  # 2026-09-02: 原 con.close() 写在 return 之后不可达(各分支提前 return), 上移到这里
    return result


# ===== 单张照片改归属（瀑布流右键/拖动修分类） =====
def set_asset_source(body):
    """单张 asset 改 source 归属：UPDATE media_file SET source_id WHERE asset_id=? AND availability='original'。

    返回旧/新 owner_label，前端用于提示「已从旧来源移到新归属人」。
    改完后 next 扫这个 source 时 indexed_count 会自动更新；这里顺手回填一次。
    """
    asset_id = (body.get("asset_id") or "").strip()
    source_id = (body.get("source_id") or "").strip()
    if not asset_id or not source_id:
        raise ValueError("缺少 asset_id 或 source_id")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys=ON")
    target = con.execute("SELECT source_id, owner_label FROM source WHERE source_id=?", (source_id,)).fetchone()
    if not target:
        con.close()
        raise ValueError("目标来源不存在")
    old = con.execute(
        """SELECT mf.source_id, s.owner_label
           FROM media_file mf LEFT JOIN source s ON s.source_id=mf.source_id
           WHERE mf.asset_id=? AND mf.availability='original' LIMIT 1""",
        (asset_id,)).fetchone()
    if not old:
        con.close()
        raise ValueError("找不到该照片的原片记录")
    if old["source_id"] == source_id:
        con.close()
        return {"asset_id": asset_id, "source_id": source_id, "updated": 0,
                "old_source": old["owner_label"], "new_source": target["owner_label"], "noop": True}
    n = con.execute(
        """UPDATE media_file SET source_id=?
           WHERE asset_id=? AND availability='original'""",
        (source_id, asset_id)).rowcount
    if n:
        # 同步两端的 indexed_count（旧源会减、新源会加），避免侧栏计数滞后
        con.execute("""UPDATE source SET indexed_count =
            COALESCE((SELECT COUNT(*) FROM media_file mf WHERE mf.source_id=source.source_id), 0)""")
    con.commit()
    con.close()
    return {"asset_id": asset_id, "source_id": source_id, "updated": n,
            "old_source": old["owner_label"], "new_source": target["owner_label"]}


def _ensure_person_locked_column(con):
    """face_instance_v0.person_locked：人工归属锁（2026-09-05）。

    用户用「改人物归属」纠正过的脸打 person_locked=1：
    backfill_faces_full 重检测时保留这些行不删（同 manual），归属永不被自动覆盖；
    人工确认的脸同时作为 kNN 参照样本，纠正越多识别越准。
    """
    try:
        con.execute("ALTER TABLE face_instance_v0 ADD COLUMN person_locked INTEGER DEFAULT 0")
        con.commit()
    except sqlite3.OperationalError:
        pass  # 列已存在（幂等）


def reassign_person(body):
    """改单张照片的人物归属（2026-09-05 新增）。

    场景：人脸识别把照片归错了人 —— 用户在「刘建强」页里看到一张不是他的照片，
    要把它改归到正确的家庭成员。实现：把这张照片上所有归属 from_person 的
    face_instance_v0.person_id 改成 to_person；照片里其他人的脸实例不动，原片不动。

    to_person_id 传空 → 取消归属（person_id=NULL，脸回到 faces 页待认领池）。
    """
    asset_id = (body.get("asset_id") or "").strip()
    from_pid = (body.get("from_person_id") or "").strip()
    to_pid = (body.get("to_person_id") or "").strip()
    if not asset_id or not from_pid:
        raise ValueError("缺少 asset_id 或 from_person_id")
    if to_pid and to_pid == from_pid:
        raise ValueError("目标人物与当前人物相同，无需修改")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _ensure_person_locked_column(con)
    try:
        frm = con.execute("SELECT person_id, display_name FROM person WHERE person_id=?", (from_pid,)).fetchone()
        if not frm:
            raise ValueError("来源人物不存在")
        to_name = "（未归属）"
        if to_pid:
            to = con.execute("SELECT display_name FROM person WHERE person_id=?", (to_pid,)).fetchone()
            if not to:
                raise ValueError("目标人物不存在")
            to_name = to["display_name"]
        faces = con.execute(
            "SELECT face_instance_id FROM face_instance_v0 WHERE asset_id=? AND person_id=?",
            (asset_id, from_pid)).fetchall()
        if not faces:
            raise ValueError(f"这张照片上没有归属「{frm['display_name']}」的脸，可能已被移动过，刷新后再试")
        n = con.execute(
            "UPDATE face_instance_v0 SET person_id=?, person_locked=1 WHERE asset_id=? AND person_id=?",
            (to_pid or None, asset_id, from_pid)).rowcount
        con.commit()
        return {"ok": True, "asset_id": asset_id, "updated": n,
                "from": frm["display_name"], "to": to_name}
    finally:
        con.close()


def move_scene_tags(body):
    """批量把资产移动/加入场景（物品）标签（2026-09-05 新增，多选改分类用）。

    场景与物品共用 scene_tag_v0。from_tag 非空 = 先从旧标签移出（纠正自动分类），
    空 = 只加标签（多对多，一张照片可以同时挂在多个标签下）。
    source='manual' + confidence=1.0：人工纠正，与 VLM 自动打的（confidence<1）区分。
    """
    asset_ids = [a for a in (body.get("asset_ids") or []) if a]
    from_tag = (body.get("from_tag") or "").strip()
    to_tag = (body.get("to_tag") or "").strip()
    if not asset_ids:
        raise ValueError("缺少 asset_ids")
    if not to_tag:
        raise ValueError("缺少 to_tag")
    if from_tag and from_tag == to_tag:
        raise ValueError("目标标签与当前标签相同，无需修改")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        now = datetime.now(timezone.utc).isoformat()
        removed = 0
        if from_tag:
            ph = ",".join("?" * len(asset_ids))
            removed = con.execute(
                f"DELETE FROM scene_tag_v0 WHERE tag=? AND asset_id IN ({ph})",
                [from_tag, *asset_ids]).rowcount
        added = 0
        for aid in asset_ids:
            added += con.execute(
                "INSERT OR IGNORE INTO scene_tag_v0(asset_id,tag,source,confidence,created_at) "
                "VALUES (?,?,?,?,?)",
                (aid, to_tag, "manual", 1.0, now)).rowcount
        con.commit()
        return {"ok": True, "moved": len(asset_ids), "added": added,
                "removed": removed, "to_tag": to_tag}
    finally:
        con.close()


def add_to_category(body):
    """批量把照片「加到」任意二级分类（2026-09-06 新增，右键菜单「加到分类…」用）。

    kind ∈ custom | scene | object | activity | moment | other | location
    target：custom → category_id；其它 kind → tag 名（scene/object/activity/moment/other 共用 scene_tag_v0）；location → region 名。

    - custom / scene / object / activity / moment / other：多对多关系，加进去不覆盖旧的（多次加同一个 kind 也不会重复）。
    - location：asset_geo_v0.asset_id 是 PRIMARY KEY，一张照片只有一个 region，UPSERT 等于"覆盖原地点"。
    - person：语义不同（要指定 from_person_id 才能改脸实例归属），不在本接口范围；如需走 /api/asset reassign_person。
    - source：已下线（瀑布流右键菜单不再暴露"改归属"入口）。

    source 标记：scene/object/activity/moment/other 用 'manual'/1.0 与 VLM 自动（confidence<1）区分；
    location 用 'USER_MOVE' 标记（与 GPS / vision / trip 区分）；custom 用 added_by='user'。
    """
    asset_ids = [a for a in (body.get("asset_ids") or []) if a]
    kind = (body.get("kind") or "").strip()
    target = (body.get("target") or "").strip()
    if not asset_ids:
        raise ValueError("缺少 asset_ids")
    if kind not in ("custom", "scene", "object", "activity", "moment", "other", "location"):
        raise ValueError(f"kind 不支持: {kind}（需要 custom/scene/object/activity/moment/other/location）")
    if not target:
        raise ValueError("缺少 target")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    try:
        now = datetime.now(timezone.utc).isoformat()
        if kind == "custom":
            if not con.execute("SELECT 1 FROM user_category_v0 WHERE category_id=?", (target,)).fetchone():
                raise ValueError("目标 custom 分类不存在")
            rows = [(target, a, "user", now) for a in asset_ids]
            con.executemany(
                "INSERT OR IGNORE INTO user_category_member_v0(category_id,asset_id,added_by,added_at) "
                "VALUES(?,?,?,?)", rows)
            added = con.execute(
                "SELECT COUNT(*) FROM user_category_member_v0 WHERE category_id=?", (target,)).fetchone()[0]
            return {"kind": kind, "target": target, "added": len(rows), "count": added}
        if kind == "location":
            # asset_geo_v0.asset_id 是 PRIMARY KEY，UPSERT 即覆盖原 region（用户纠正自动识别的语义）
            rows = [(a, target, "USER_MOVE", now) for a in asset_ids]
            con.executemany(
                "INSERT INTO asset_geo_v0(asset_id,region,location_source,created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(asset_id) DO UPDATE SET region=excluded.region, location_source='USER_MOVE'",
                rows)
            con.commit()
            return {"kind": kind, "target": target, "updated": len(rows)}
        # scene/object/activity/moment/other：共用 scene_tag_v0（多对多，PRIMARY KEY 是 (asset_id, tag)）
        rows = [(a, target, "manual", 1.0, now) for a in asset_ids]
        con.executemany(
            "INSERT OR IGNORE INTO scene_tag_v0(asset_id,tag,source,confidence,created_at) "
            "VALUES(?,?,?,?,?)", rows)
        con.commit()
        return {"kind": kind, "target": target, "added": len(rows)}
    finally:
        con.close()


# ===== 分类管理（展示配置：层级/排序/首页显示开关） =====
# 一级分组（二级分类的归属容器），与前端 LIB_GROUP_META 一致
_CAT_GROUP_META = [
    {"key": "source", "label": "来源 · 谁拍的"},
    {"key": "person", "label": "人物"},
    {"key": "location", "label": "地点"},
    {"key": "scene", "label": "场景 · 风景"},
    {"key": "object", "label": "物品"},
    {"key": "activity", "label": "美食 · 活动"},
    {"key": "moment", "label": "人物时刻"},
    {"key": "other", "label": "其他"},
    {"key": "custom", "label": "我的分类"},
]
# 场景标签 → 自然分组（未映射的落入「其他」）
_CAT_SCENE_GROUP = {
    "旅行": "scene", "江河": "scene", "室内": "scene", "山景": "scene", "海景": "scene",
    "夜景": "scene", "雪景": "scene", "建筑": "scene", "花草": "scene",
    "书包": "object", "玩具": "object", "灯笼": "object", "婴儿车": "object",
    "行李箱": "object", "乐器": "object", "风筝": "object",
    "车辆": "object", "火车": "object", "飞机": "object", "自行车": "object", "游船": "object",
    "美食": "activity", "火锅": "activity", "烧烤": "activity", "帐篷": "activity",
    "滑雪": "activity", "气球": "activity", "生日": "activity", "宠物": "activity",
    "合影": "moment", "儿童": "moment", "截图": "other",
}
# 分类类型 → 自然分组
_CAT_NATURAL_GROUP = {"person": "person", "location": "location", "custom": "custom"}


def _init_cat_layout_table(con):
    """分类展示配置：parent 层级（''=一级平铺置顶，否则为分组 key）、sort_order 排序、show_on_home 首页显示开关。"""
    con.execute("""CREATE TABLE IF NOT EXISTS category_layout_v0 (
        kind TEXT NOT NULL,
        target TEXT NOT NULL,
        parent TEXT NOT NULL DEFAULT '',
        sort_order INTEGER NOT NULL DEFAULT 0,
        show_on_home INTEGER NOT NULL DEFAULT 1,
        updated_at TEXT NOT NULL,
        PRIMARY KEY(kind, target)
    )""")
    # 一级分组改名（2026-09-02：侧栏组名可编辑）
    con.execute("""CREATE TABLE IF NOT EXISTS group_label_v0 (
        group_key TEXT PRIMARY KEY,
        label TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""")
    # 一级分组拖拽顺序（2026-09-03：侧栏分组间自由排序；稀疏存储，未记录的分组回退默认顺序）
    con.execute("""CREATE TABLE IF NOT EXISTS group_order_v0 (
        group_key TEXT PRIMARY KEY,
        sort_order INTEGER NOT NULL DEFAULT 0,
        updated_at TEXT NOT NULL
    )""")
    con.commit()


def _cat_group_labels(con):
    """一级分组显示名覆盖表：{group_key: 用户自定义组名}。"""
    try:
        _init_cat_layout_table(con)
        return {r["group_key"]: r["label"] for r in con.execute(
            "SELECT group_key,label FROM group_label_v0")}
    except Exception:
        return {}


def _cat_group_order(con):
    """一级分组顺序覆盖表：{group_key: sort_order}（用户拖拽后持久化）。"""
    try:
        return {r["group_key"]: r["sort_order"] for r in con.execute(
            "SELECT group_key,sort_order FROM group_order_v0")}
    except Exception:
        return {}


def _cat_group_view(con):
    """侧栏/首页渲染用一级分组列表：默认名 + 用户改名覆盖；含 custom 顶层（key='custom:<id>'）。
    顺序 = 用户拖拽顺序（group_order_v0）优先，未调整过的分组回退 _CAT_GROUP_META 默认顺序。
    custom 顶层参与混排：此前被排除导致收藏永远钉在组头前，拖不动。"""
    labels = _cat_group_labels(con)
    order = _cat_group_order(con)
    out = []
    # 系统分组：按 _CAT_GROUP_META 默认序填占位 sort（用户拖过则用 group_order_v0 覆盖）
    base_idx = 0
    for g in _CAT_GROUP_META:
        if g["key"] == "custom":
            continue
        out.append({"key": g["key"], "label": labels.get(g["key"], g["label"]),
                    "sort": order.get(g["key"], base_idx), "_kind": g["key"]})
        base_idx += 1
    # custom 顶层（parent='' 且非子级）：追加进混排列表
    try:
        for r in con.execute(
            "SELECT c.category_id,c.name FROM user_category_v0 c "
            "LEFT JOIN category_layout_v0 l ON l.kind='custom' AND l.target=c.category_id "
            "WHERE (l.parent IS NULL OR l.parent='') ORDER BY c.sort_order, c.created_at"):
            out.append({"key": "custom:" + r["category_id"], "label": r["name"],
                        "sort": order.get("custom:" + r["category_id"], base_idx),
                        "_kind": "custom", "_target": r["category_id"]})
            base_idx += 1
    except sqlite3.OperationalError:
        pass
    out.sort(key=lambda x: x["sort"])
    return out


def _cat_natural_parent(kind, target):
    """分类的自然归属分组（未手动移动时的默认层级）。"""
    if kind == "scene":
        return _CAT_SCENE_GROUP.get(target, "other")
    return _CAT_NATURAL_GROUP.get(kind, "other")


def _cat_layout_map(con, kind):
    """读取某类分类的 layout 配置：{target: {parent, sort_order, show_on_home}}。"""
    rows = con.execute(
        "SELECT target,parent,sort_order,show_on_home FROM category_layout_v0 WHERE kind=?",
        (kind,)).fetchall()
    return {r["target"]: {"parent": r["parent"], "sort_order": r["sort_order"],
                          "show_on_home": r["show_on_home"]} for r in rows}


def catman_list(con):
    """分类管理视图：人物/地点/场景/自定义分类的完整清单（含 0 张的空壳）＋ layout 配置。

    数据源 = 源表 UNION layout 中已定义但源表尚无记录的空壳（手动新建的场景/地点）。"""
    _init_cat_layout_table(con)
    view = {}
    now = now_iso()

    # 2026-09-01：计数口径与侧栏 /api/categories、点开分类后的 answer 统一为「可见」
    # （过滤表 + 相似组非最佳 + 回收站 - 白名单，同 get_hidden_ids）。
    # 此前管理器数原始成员数，出现「管理器显示 3000、点开只剩 900」的错位。
    # 自定义分类例外：成员是用户人工归属（意图优先于可见集），仅排除回收站（点开不显示）。
    hidden = get_hidden_ids(con)
    con.execute("CREATE TEMP TABLE IF NOT EXISTS _hidden_ids_v0(asset_id TEXT PRIMARY KEY)")
    con.execute("DELETE FROM _hidden_ids_v0")
    con.executemany("INSERT OR IGNORE INTO _hidden_ids_v0 VALUES(?)", [(a,) for a in hidden])
    NH = "NOT EXISTS(SELECT 1 FROM _hidden_ids_v0 h WHERE h.asset_id={col})"

    def merge(kind, items, key_of, allow_shell=True):
        layout = _cat_layout_map(con, kind)
        out = []
        for it in items:
            k = key_of(it)
            lay = layout.get(k)
            if lay is None:
                # 首次出现：写默认 layout（自然组 + 追加排序）
                parent = _cat_natural_parent(kind, k)
                max_order = con.execute(
                    "SELECT COALESCE(MAX(sort_order),0) FROM category_layout_v0 WHERE kind=? AND parent=?",
                    (kind, parent)).fetchone()[0]
                con.execute(
                    "INSERT OR IGNORE INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at) "
                    "VALUES(?,?,?,?,1,?)", (kind, k, parent, max_order + 1, now))
                lay = {"parent": parent, "sort_order": max_order + 1, "show_on_home": 1}
            out.append({**it, "parent": lay["parent"], "sort_order": lay["sort_order"],
                        "show_on_home": lay["show_on_home"]})
        # 空壳：layout 里存在但源表没有（手动新增的场景/地点）
        # 2026-09-19：空壳只对 location/scene 合法（新建时只写 layout 等成员）。
        # person/custom 的新建都写真表，这类"源表没有"的行只可能是历史脏数据
        # （曾把地名写进 kind='person'/'custom'，导致人物 tab 混入一堆地点）→ 不渲染。
        exist = {key_of(it) for it in items}
        layout = _cat_layout_map(con, kind)
        for k, lay in layout.items():
            if k in exist:
                continue
            if not allow_shell:
                continue
            if k.startswith(("cat_", "person_")):
                # 2026-09-19：地点/场景的 target 是用户手输的地名，不可能是 ID；
                # 这类行来自历史合并/迁移 bug，不渲染
                continue
            out.append({"target": k, "name": k, "count": 0, "parent": lay["parent"],
                        "sort_order": lay["sort_order"], "show_on_home": lay["show_on_home"],
                        "shell": True})
        out.sort(key=lambda x: (x["parent"] or "\uffff", x["sort_order"]))
        return out

    # 人物
    view["persons"] = merge("person", [dict(r) for r in con.execute(
        f"""SELECT p.person_id AS target, p.display_name AS name,
                  COUNT(DISTINCT fi.asset_id) AS count
           FROM person p LEFT JOIN face_instance_v0 fi
             ON fi.person_id=p.person_id AND {NH.format(col='fi.asset_id')}
           GROUP BY p.person_id""")], lambda x: x["target"], allow_shell=False)
    # 地点（region 聚合；不含仅 GPS 无 region 的资产）
    view["locations"] = merge("location", [dict(r) for r in con.execute(
        f"""SELECT region AS target, region AS name, COUNT(DISTINCT asset_id) AS count
           FROM asset_geo_v0 WHERE region IS NOT NULL AND region<>''
             AND {NH.format(col='asset_geo_v0.asset_id')} GROUP BY region""")],
        lambda x: x["target"])
    # 场景标签
    try:
        view["scenes"] = merge("scene", [dict(r) for r in con.execute(
            f"""SELECT tag AS target, tag AS name, COUNT(DISTINCT asset_id) AS count
               FROM scene_tag_v0 WHERE {NH.format(col='scene_tag_v0.asset_id')} GROUP BY tag""")],
            lambda x: x["target"])
    except sqlite3.OperationalError:
        view["scenes"] = []
    # 自定义分类：用户人工归属，仅排除回收站/隐私相册成员
    try:
        recycled = get_recycled_ids(con)
        _rc_parts = []
        if recycled:
            _rc_parts.append("NOT EXISTS(SELECT 1 FROM recycle_v0 r WHERE r.asset_id=m.asset_id)")
        if _privacy_has_rows(con):
            _rc_parts.append("NOT EXISTS(SELECT 1 FROM privacy_v0 p WHERE p.asset_id=m.asset_id)")
        RC = (" AND ".join(_rc_parts)) if _rc_parts else ""
        view["customs"] = merge("custom", [dict(r) for r in con.execute(
            f"""SELECT c.category_id AS target, c.name AS name, COUNT(m.asset_id) AS count
               FROM user_category_v0 c LEFT JOIN user_category_member_v0 m
                 ON m.category_id=c.category_id{(' AND ' + RC) if RC else ''}
               GROUP BY c.category_id""")], lambda x: x["target"], allow_shell=False)
    except sqlite3.OperationalError:
        view["customs"] = []
    con.execute("DROP TABLE IF EXISTS _hidden_ids_v0")
    con.commit()
    view["groups"] = _cat_group_view(con)
    return view


def catman_action(body):
    """分类管理：list / rename / create / delete / move / sort / show。

    kind ∈ person|location|scene|custom；target = person_id / region / tag / category_id。
    move 的 parent='' 表示一级平铺置顶，否则为分组 key。"""
    action = body.get("action")
    if action not in ("list", "rename", "rename_group", "sort_groups", "create", "delete", "move", "sort", "show", "sort_province"):
        raise ValueError("未知 action")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _init_cat_layout_table(con)
    now = now_iso()
    try:
        if action == "list":
            return catman_list(con)

        # 一级分组改名（target=分组 key，如 person/location/scene/object/activity/moment/other/source）
        if action == "rename_group":
            key = (body.get("target") or "").strip()
            name = (body.get("name") or "").strip()
            valid = {g["key"] for g in _CAT_GROUP_META if g["key"] != "custom"}
            if key not in valid:
                raise ValueError(f"分组不存在: {key}")
            if not name:
                raise ValueError("分组名称不能为空")
            con.execute("INSERT OR REPLACE INTO group_label_v0(group_key,label,updated_at) VALUES(?,?,?)",
                        (key, name, now))
            con.commit()
            return {"ok": True, "group": key, "label": name}

        # 一级分组拖拽排序（ordered = 新分组 key 顺序，不含 custom）
        if action == "sort_groups":
            ordered = body.get("ordered") or []
            if not ordered:
                raise ValueError("缺少 ordered")
            valid = {g["key"] for g in _CAT_GROUP_META if g["key"] != "custom"}
            # 自定义分类也参与顶层排序：key 格式 'custom:<category_id>'，
            # 与系统分组共用 group_order_v0 的序号空间，才能混排（此前 custom 被排除 → 收藏永远钉在组头前，拖不动）
            try:
                valid_custom = {r["category_id"] for r in con.execute("SELECT category_id FROM user_category_v0")}
            except sqlite3.OperationalError:
                valid_custom = set()
            seen = set()
            for i, k in enumerate(ordered):
                if k in seen:
                    continue
                if k in valid:
                    ok = True
                elif k.startswith("custom:"):
                    ok = k[7:] in valid_custom
                else:
                    ok = False
                if not ok:
                    continue
                seen.add(k)
                # custom 同步更新 category_layout_v0.sort_order，保证两侧（分类管理/侧栏）口径一致
                if k.startswith("custom:"):
                    con.execute(
                        """INSERT INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at)
                           VALUES('custom',?,'',?,1,?)
                           ON CONFLICT(kind,target) DO UPDATE SET sort_order=excluded.sort_order,updated_at=excluded.updated_at""",
                        (k[7:], i, now))
                con.execute(
                    "INSERT OR REPLACE INTO group_order_v0(group_key,sort_order,updated_at) VALUES(?,?,?)",
                    (k, i, now))
            con.commit()
            return {"ok": True, "sorted_groups": len(seen)}

        kind = body.get("kind")
        if kind not in ("person", "location", "scene", "custom", "source"):
            raise ValueError("kind 必须是 person/location/scene/custom/source")

        if action == "rename":
            target = (body.get("target") or "").strip()
            name = (body.get("name") or "").strip()
            if not target or not name:
                raise ValueError("缺少 target 或 name")
            _merged = None
            if kind == "person":
                cur = con.execute("SELECT person_id FROM person WHERE person_id=?", (target,)).fetchone()
                if not cur:
                    raise ValueError("人物不存在")
                con.execute("UPDATE person SET display_name=?, updated_at=? WHERE person_id=?",
                            (name, now, target))
            elif kind == "scene":
                con.execute("UPDATE scene_tag_v0 SET tag=? WHERE tag=?", (name, target))
                con.execute("UPDATE category_layout_v0 SET target=? WHERE kind='scene' AND target=?",
                            (name, target))
            elif kind == "location":
                # 重名自动合并：新名下若有同名自定义分类，先整个并入该地点再改名
                ocid = _custom_cid_by_name(con, name)
                if ocid:
                    _merge_custom_into_location(con, ocid, name, now)
                    _merged = "location"
                # 新名与现有其他地点重名：UPDATE 后两批 region 行同名，所有 GROUP BY region
                # 查询天然合并为一条，无副作用
                con.execute("UPDATE asset_geo_v0 SET region=? WHERE region=?", (name, target))
                con.execute("UPDATE category_layout_v0 SET target=? WHERE kind='location' AND target=?",
                            (name, target))
            else:  # custom
                # 重名自动合并（2026-09-07 15:30）：
                # ① 撞其他自定义分类 → 成员并入对方，删自身；
                # ② 撞同名地点 → 整个并入地点（成员照片 region 归位），删自身；
                # ③ 无冲突 → 普通改名。
                ocid = _custom_cid_by_name(con, name, exclude_cid=target)
                if ocid:
                    con.execute("""INSERT OR IGNORE INTO user_category_member_v0(category_id,asset_id,added_by,added_at)
                                   SELECT ?, asset_id, added_by, added_at FROM user_category_member_v0
                                   WHERE category_id=?""", (ocid, target))
                    con.execute("DELETE FROM user_category_member_v0 WHERE category_id=?", (target,))
                    con.execute("DELETE FROM user_category_v0 WHERE category_id=?", (target,))
                    con.execute("DELETE FROM category_layout_v0 WHERE kind='custom' AND target=?", (target,))
                    _merged = "custom"
                elif _location_conflict(con, name):
                    _merge_custom_into_location(con, target, name, now)
                    _merged = "location"
                else:
                    con.execute("UPDATE user_category_v0 SET name=? WHERE category_id=?", (name, target))
            con.commit()
            r = {"ok": True, "kind": kind, "target": target, "name": name}
            if _merged:
                r["merged"] = _merged
            return r

        if action == "create":
            name = (body.get("name") or "").strip()
            if not name:
                raise ValueError("分类名不能为空")
            parent = body.get("parent") if body.get("parent") is not None else ""
            if kind == "person":
                count = con.execute("SELECT COUNT(*) FROM person").fetchone()[0]
                if count >= 30:
                    raise ValueError(f"人物已达上限 30 个（当前 {count}），请先合并或删除不常用人物")
                family_id = con.execute("SELECT family_id FROM family LIMIT 1").fetchone()[0]
                pid = "person_" + uuid.uuid4().hex[:12]
                con.execute(
                    """INSERT INTO person(person_id,family_id,display_name,relationship_label,birth_date,
                       identity_status,created_at,updated_at) VALUES(?,?,?,NULL,NULL,'confirmed',?,?)""",
                    (pid, family_id, name, now, now))
                target = pid
            elif kind == "scene":
                target = name
                # 空壳：写入 layout，源表等 move_to/打标时才有行
                con.execute("DELETE FROM category_layout_v0 WHERE kind='scene' AND target=?", (name,))
            elif kind == "location":
                target = name
                # 重名自动合并：同名自定义分类先整个并入该地点（成员照片 region 归位，删分类）
                ocid = _custom_cid_by_name(con, name)
                if ocid:
                    _merge_custom_into_location(con, ocid, name, now)
                con.execute("DELETE FROM category_layout_v0 WHERE kind='location' AND target=?", (name,))
            else:  # custom
                family_id = con.execute("SELECT family_id FROM family LIMIT 1").fetchone()[0]
                existing = con.execute("SELECT category_id FROM user_category_v0 WHERE name=? AND family_id=?",
                                       (name, family_id)).fetchone()
                if existing:
                    return {"ok": True, "kind": kind, "target": existing["category_id"], "name": name, "existed": True}
                # 重名自动合并：名字已作为地点存在 → 不建重复的自定义分类，返回地点目标
                if _location_conflict(con, name):
                    return {"ok": True, "kind": kind, "target": name, "name": name, "merged": "location"}
                cid = "cat_" + uuid.uuid4().hex[:10]
                max_order = con.execute("SELECT COALESCE(MAX(sort_order),0) FROM user_category_v0").fetchone()[0]
                con.execute("INSERT INTO user_category_v0(category_id,family_id,name,sort_order,created_at) "
                            "VALUES(?,?,?,?,?)", (cid, family_id, name, max_order + 1, now))
                target = cid
            # 写 layout：parent 缺省用自然组
            if not parent:
                parent = _cat_natural_parent(kind, target)
            con.execute(
                "INSERT OR REPLACE INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at) "
                "VALUES(?,?,?,?,1,?)",
                (kind, target, parent,
                 con.execute("SELECT COALESCE(MAX(sort_order),0) FROM category_layout_v0 WHERE kind=? AND parent=?",
                             (kind, parent)).fetchone()[0] + 1, now))
            con.commit()
            return {"ok": True, "kind": kind, "target": target, "name": name}

        if action == "delete":
            target = (body.get("target") or "").strip()
            if not target:
                raise ValueError("缺少 target")
            if kind == "person":
                if not con.execute("SELECT 1 FROM person WHERE person_id=?", (target,)).fetchone():
                    raise ValueError("人物不存在")
                con.execute("UPDATE face_instance_v0 SET person_id=NULL, sample_role='auto' WHERE person_id=?", (target,))
                con.execute("DELETE FROM person WHERE person_id=?", (target,))
            elif kind == "scene":
                con.execute("DELETE FROM scene_tag_v0 WHERE tag=?", (target,))
            elif kind == "location":
                con.execute("DELETE FROM asset_geo_v0 WHERE region=?", (target,))
            else:
                con.execute("DELETE FROM user_category_member_v0 WHERE category_id=?", (target,))
                con.execute("DELETE FROM user_category_v0 WHERE category_id=?", (target,))
            con.execute("DELETE FROM category_layout_v0 WHERE kind=? AND target=?", (kind, target))
            con.commit()
            return {"ok": True, "kind": kind, "target": target, "deleted": True}

        if action == "move":
            target = (body.get("target") or "").strip()
            parent = body.get("parent") if body.get("parent") is not None else ""
            if not target:
                raise ValueError("缺少 target")
            groups = {g["key"] for g in _CAT_GROUP_META}
            if parent and parent not in groups:
                raise ValueError(f"目标层级无效: {parent}")
            if not con.execute("SELECT 1 FROM category_layout_v0 WHERE kind=? AND target=?", (kind, target)).fetchone():
                con.execute(
                    "INSERT OR IGNORE INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at) "
                    "VALUES(?,?,?,?,1,?)",
                    (kind, target, parent, 0, now))
            con.execute("UPDATE category_layout_v0 SET parent=?, updated_at=? WHERE kind=? AND target=?",
                        (parent, now, kind, target))
            con.commit()
            return {"ok": True, "kind": kind, "target": target, "parent": parent}

        if action == "sort":
            ordered = body.get("ordered") or []
            parent = body.get("parent")
            if not ordered:
                raise ValueError("缺少 ordered")
            for i, t in enumerate(ordered):
                # 从未被移动/创建过的分类在 layout 里根本没有行，纯 UPDATE 影响 0 行 → 排序静默失效
                # （实测：person 16 人只有 8 人有 layout 行，scene 12139 个只有 32 个有）。
                # 先按主键 (kind,target) 补一行，再更新排序。
                con.execute(
                    "INSERT OR IGNORE INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at) "
                    "VALUES(?,?,?,?,1,?)",
                    (kind, t, parent if parent is not None else "", i, now))
                if parent is not None:
                    con.execute(
                        "UPDATE category_layout_v0 SET sort_order=?, updated_at=? WHERE kind=? AND parent=? AND target=?",
                        (i, now, kind, parent, t))
                else:
                    con.execute(
                        "UPDATE category_layout_v0 SET sort_order=?, updated_at=? WHERE kind=? AND target=?",
                        (i, now, kind, t))
            con.commit()
            return {"ok": True, "kind": kind, "sorted": len(ordered)}

        # 2026-09-07 按省份默认排序（仅 location）：地点归属省份取 user_trip（用户标注优先）
        # 或 asset_geo_v0 多数票；省份之间按该省照片总量降序、省内按地点照片量降序、无省份垫底。
        # 写回 category_layout_v0.sort_order，之后用户仍可手动拖拽微调。
        if action == "sort_province":
            if kind != "location":
                raise ValueError("sort_province 仅支持 kind=location")
            rows = con.execute("""
                SELECT region AS target, COUNT(DISTINCT g.asset_id) AS cnt,
                       COALESCE(
                         (SELECT t.province FROM user_trip t WHERE t.region=g.region
                           ORDER BY t.updated_at DESC LIMIT 1),
                         (SELECT g2.province FROM asset_geo_v0 g2
                           WHERE g2.region=g.region AND g2.province IS NOT NULL
                             AND g2.province<>'' AND g2.province<>'未知'
                           GROUP BY g2.province ORDER BY COUNT(*) DESC LIMIT 1)
                       ) AS province
                FROM asset_geo_v0 g
                WHERE g.region IS NOT NULL AND g.region<>''
                GROUP BY g.region""").fetchall()
            shells = [r["target"] for r in con.execute(
                "SELECT target FROM category_layout_v0 WHERE kind='location'").fetchall()]
            prov_of = {r["target"]: (r["province"] or "") for r in rows}
            cnt_of = {r["target"]: r["cnt"] for r in rows}
            targets = list({r["target"] for r in rows} | set(shells))
            prov_total = {}
            for t in targets:
                pr = prov_of.get(t) or ""
                prov_total[pr] = prov_total.get(pr, 0) + cnt_of.get(t, 0)
            ordered = sorted(targets, key=lambda t: (
                1 if not prov_of.get(t) else 0,          # 有省份的在前
                -prov_total.get(prov_of.get(t, ""), 0),  # 省份按照片总量
                -cnt_of.get(t, 0),                       # 省内按地点照片量
                t))                                      # 同量按名称稳定排序
            layout = _cat_layout_map(con, "location")
            for i, t in enumerate(ordered):
                parent = (layout.get(t) or {}).get("parent") or _cat_natural_parent("location", t)
                con.execute(
                    "INSERT OR IGNORE INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at) "
                    "VALUES('location',?,?,?,1,?)", (t, parent, i, now))
                con.execute(
                    "UPDATE category_layout_v0 SET sort_order=?, updated_at=? WHERE kind='location' AND target=?",
                    (i, now, t))
            con.commit()
            return {"ok": True, "ordered": ordered,
                    "provinces": {t: prov_of.get(t, "") for t in ordered}}

        if action == "show":
            target = (body.get("target") or "").strip()
            show = 1 if body.get("show") else 0
            if not target:
                raise ValueError("缺少 target")
            if not con.execute("SELECT 1 FROM category_layout_v0 WHERE kind=? AND target=?", (kind, target)).fetchone():
                con.execute(
                    "INSERT OR IGNORE INTO category_layout_v0(kind,target,parent,sort_order,show_on_home,updated_at) "
                    "VALUES(?,?,?,?,?,?)", (kind, target, _cat_natural_parent(kind, target), 0, show, now))
            con.execute("UPDATE category_layout_v0 SET show_on_home=?, updated_at=? WHERE kind=? AND target=?",
                        (show, now, kind, target))
            con.commit()
            return {"ok": True, "kind": kind, "target": target, "show_on_home": show}
    finally:
        con.close()
    raise ValueError("未处理的操作")


def update_people(body):
    """Explicit user actions only: create/update a Person, assign or ignore a cluster."""
    action = body.get("action")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    now = now_iso()
    result_extra = {}   # 各 action 可塞自定义回包字段（如 delete_faces 删了多少张）
    if action == "create":
        name = (body.get("display_name") or "").strip()
        if not name:
            raise ValueError("姓名不能为空")
        existing = con.execute("SELECT person_id FROM person WHERE display_name=?", (name,)).fetchone()
        if existing:
            pid = existing["person_id"]
        else:
            # 人物上限 30：新人物只能人工创建，满了必须先精简现有人物
            count = con.execute("SELECT COUNT(*) FROM person").fetchone()[0]
            if count >= 30:
                raise ValueError(f"人物已达上限 30 个（当前 {count}），请先合并或删除不常用人物")
            family_id = con.execute("SELECT family_id FROM family LIMIT 1").fetchone()[0]
            pid = "person_" + uuid.uuid4().hex[:12]
            con.execute(
                """INSERT INTO person(person_id,family_id,display_name,relationship_label,birth_date,
                   identity_status,created_at,updated_at) VALUES(?,?,?,?,?,'confirmed',?,?)""",
                (pid, family_id, name, (body.get("relationship_label") or "").strip() or None,
                 (body.get("birth_date") or "").strip() or None, now, now),
            )
    elif action == "update":
        pid = body.get("person_id")
        if not con.execute("SELECT 1 FROM person WHERE person_id=?", (pid,)).fetchone():
            raise ValueError("人物不存在")
        con.execute(
            """UPDATE person SET display_name=?,relationship_label=?,birth_date=?,updated_at=?
               WHERE person_id=?""",
            ((body.get("display_name") or "").strip(),
             (body.get("relationship_label") or "").strip() or None,
             (body.get("birth_date") or "").strip() or None, now, pid),
        )
    elif action == "assign_cluster":
        pid, cid = body.get("person_id"), body.get("cluster_id")
        if not con.execute("SELECT 1 FROM person WHERE person_id=?", (pid,)).fetchone():
            raise ValueError("人物不存在")
        cluster = con.execute(
            "SELECT hypothesis_status FROM anonymous_person_cluster_v0 WHERE cluster_id=?", (cid,)
        ).fetchone()
        if not cluster or cluster["hypothesis_status"] != "candidate":
            raise ValueError("该人脸组不是待确认状态")
        con.execute(
            """UPDATE face_instance_v0 SET person_id=? WHERE face_instance_id IN
               (SELECT face_instance_id FROM anonymous_person_membership_v0 WHERE cluster_id=?)""",
            (pid, cid),
        )
        con.execute("UPDATE anonymous_person_membership_v0 SET status='confirmed' WHERE cluster_id=?", (cid,))
        con.execute("UPDATE anonymous_person_cluster_v0 SET hypothesis_status='confirmed' WHERE cluster_id=?", (cid,))
    elif action == "ignore_cluster":
        cid = body.get("cluster_id")
        con.execute("UPDATE anonymous_person_cluster_v0 SET hypothesis_status='rejected' WHERE cluster_id=?", (cid,))
        con.execute("UPDATE anonymous_person_membership_v0 SET status='rejected' WHERE cluster_id=?", (cid,))
    elif action == "assign_faces":
        pid = body.get("person_id")
        face_ids = list(dict.fromkeys(body.get("face_instance_ids") or []))
        if not con.execute("SELECT 1 FROM person WHERE person_id=?", (pid,)).fetchone():
            raise ValueError("人物不存在")
        if not face_ids or len(face_ids) > 9000:
            raise ValueError("请选择 1–9000 张人脸")
        placeholders = ",".join("?" for _ in face_ids)
        # 人工认领的脸直接进参照库(sample_role='manual')，后续 kNN 自动归类以它为准
        con.execute(
            f"""UPDATE face_instance_v0 SET person_id=?, sample_role='manual'
                WHERE person_id IS NULL AND face_instance_id IN ({placeholders})""",
            (pid, *face_ids),
        )
    elif action == "unassign_faces":
        # 已确认影像里选错的人脸：解绑回未认领（不删脸/照片），sample_role 恢复 auto 供自动归类重新判定
        face_ids = list(dict.fromkeys(body.get("face_instance_ids") or []))
        if not face_ids or len(face_ids) > 9000:
            raise ValueError("请选择 1–9000 张人脸")
        ph = ",".join("?" for _ in face_ids)
        con.execute(
            f"""UPDATE face_instance_v0 SET person_id=NULL, sample_role='auto'
                WHERE face_instance_id IN ({ph})""", face_ids)
    elif action == "set_avatar":
        # 用户自选头像：从该人物的已认领脸里挑一张做头像（/face_crop 输出更清晰、裁切可控）。
        # face_instance_id 为空表示清除自选、回退到自动样例。
        pid = (body.get("person_id") or "").strip()
        fid = (body.get("face_instance_id") or "").strip() or None
        if not con.execute("SELECT 1 FROM person WHERE person_id=?", (pid,)).fetchone():
            raise ValueError("人物不存在")
        if fid and not con.execute(
            "SELECT 1 FROM face_instance_v0 WHERE face_instance_id=? AND person_id=?",
            (fid, pid)).fetchone():
            raise ValueError("该人脸不属于此人物")
        con.execute("UPDATE person SET avatar_face_id=?, updated_at=? WHERE person_id=?",
                    (fid, now, pid))
    elif action == "label_faces":
        # 照片级人工标注：在一张照片(如全家福)上逐脸指定人物，可内联新建人物(受 30 上限)。
        # 标注脸 sample_role='manual' 进参照库；保存后自动触发 kNN 归属，其余脸自动归类。
        # new_faces: 照片上人工拖框补的脸(检测漏掉的)，先建 face_instance 再随 labels 一起归属。
        labels = list(body.get("labels") or [])
        new_faces = body.get("new_faces") or []
        if not labels and not new_faces:
            raise ValueError("请标注 1–60 张人脸")
        if len(labels) + len(new_faces) > 60:
            raise ValueError("一次最多标注 60 张人脸")
        new_face_result = {"created": 0, "reused": 0, "sources": [], "errors": []}
        for nf in new_faces[:60]:
            bbox = nf.get("bbox") or {}
            try:
                bx = float(bbox.get("x")); by = float(bbox.get("y"))
                bw = float(bbox.get("w")); bh = float(bbox.get("h"))
            except (TypeError, ValueError):
                raise ValueError(f"新框选坐标无效: {bbox}")
            if not (0 <= bx <= 1 and 0 <= by <= 1 and 0.01 <= bw <= 1 and 0.01 <= bh <= 1):
                raise ValueError(f"新框选坐标越界: {bbox}")
            if not con.execute("SELECT 1 FROM media_asset WHERE asset_id=?", (nf.get("asset_id"),)).fetchone():
                raise ValueError("照片不存在")
            proc = subprocess.run(
                _py_cmd("manual_face", "--asset-id", str(nf.get("asset_id")),
                        "--bbox", f"{bx:.5f},{by:.5f},{bw:.5f},{bh:.5f}"),
                cwd=str(ROOT), capture_output=True, text=True, timeout=120)
            line = next((l for l in reversed(proc.stdout.strip().splitlines()) if l.startswith("{")), "")
            info = json.loads(line) if line else {}
            if proc.returncode != 0 or not info.get("ok"):
                err = info.get("error") or proc.stderr.strip()[-200:] or f"建脸失败(exit {proc.returncode})"
                new_face_result["errors"].append({"bbox": [bx, by, bw, bh], "error": err})
                continue
            fid = info["face_instance_id"]
            if info.get("existing"):
                new_face_result["reused"] += 1
            else:
                new_face_result["created"] += 1
                new_face_result["sources"].append(info.get("source"))
            labels.append({"face_instance_id": fid,
                           "person_id": nf.get("person_id"), "person": nf.get("person")})
        if new_face_result["errors"]:
            raise ValueError("部分新框选失败: " + "; ".join(e["error"] for e in new_face_result["errors"][:3]))
        if not labels:
            raise ValueError("没有可标注的人脸")
        assigned, created = [], []
        for lab in labels:
            fid = lab.get("face_instance_id")
            if not con.execute("SELECT 1 FROM face_instance_v0 WHERE face_instance_id=?", (fid,)).fetchone():
                raise ValueError(f"人脸不存在: {fid}")
            pid = lab.get("person_id")
            if not pid:
                np = lab.get("person") or {}
                name = (np.get("display_name") or "").strip()
                if not name:
                    raise ValueError("新建人物必须填写姓名")
                existing = con.execute("SELECT person_id FROM person WHERE display_name=?", (name,)).fetchone()
                if existing:
                    pid = existing["person_id"]
                else:
                    count = con.execute("SELECT COUNT(*) FROM person").fetchone()[0]
                    if count >= 30:
                        raise ValueError(f"人物已达上限 30 个（当前 {count}），无法新建「{name}」")
                    family_id = con.execute("SELECT family_id FROM family LIMIT 1").fetchone()[0]
                    pid = "person_" + uuid.uuid4().hex[:12]
                    con.execute(
                        """INSERT INTO person(person_id,family_id,display_name,relationship_label,birth_date,
                           identity_status,created_at,updated_at) VALUES(?,?,?,?,?,'confirmed',?,?)""",
                        (pid, family_id, name, (np.get("relationship_label") or "").strip() or None,
                         (np.get("birth_date") or "").strip() or None, now, now),
                    )
                    created.append(name)
            if not con.execute("SELECT 1 FROM person WHERE person_id=?", (pid,)).fetchone():
                raise ValueError("人物不存在")
            # 人工标注可覆盖之前的归属(含改判)，覆盖时清掉旧 sample_role 语义统一为 manual
            con.execute(
                """UPDATE face_instance_v0 SET person_id=?, sample_role='manual'
                   WHERE face_instance_id=?""",
                (pid, fid),
            )
            assigned.append(fid)
    elif action == "delete_faces":
        # 删除人脸实例（face_instance_v0 + 关联 embedding/cluster 成员/avatar 引用）。
        # 原片和 person 都不动；后续重新扫描或 kNN 都会跳过已删除的脸。
        face_ids = list(dict.fromkeys(body.get("face_instance_ids") or []))
        if not face_ids or len(face_ids) > 9000:
            raise ValueError("请选择 1–9000 张人脸")
        ph = ",".join("?" for _ in face_ids)
        # 1) 清掉引用该 face 的 avatar（避免 person.avatar_face_id 指向已删记录）
        con.execute(f"UPDATE person SET avatar_face_id=NULL WHERE avatar_face_id IN ({ph})", face_ids)
        # 2) 删 embedding / cluster 成员
        con.execute(f"DELETE FROM face_embedding_v0 WHERE face_instance_id IN ({ph})", face_ids)
        con.execute(f"DELETE FROM anonymous_person_membership_v0 WHERE face_instance_id IN ({ph})", face_ids)
        # 3) 删 face_instance 本身
        n = con.execute(f"DELETE FROM face_instance_v0 WHERE face_instance_id IN ({ph})", face_ids).rowcount
        # 4) 清掉这些脸的裁切缓存（2026-09-14 审计修复：fid 删后重建会复用 id，旧缓存会顶替新脸）
        for _fid in face_ids:
            for _p in FACE_CROP_DIR.glob(f"{_fid}_k*_s*_v*.jpg"):
                try:
                    _p.unlink()
                except OSError:
                    pass
        # 受影响的 cluster member_count / asset_count 让下次加载时自然从 get_people_payload 重新聚合，
        # 这里不主动 UPDATE 避免和后台检测任务竞争；下次拉取 people 时会重算。
        result_extra["deleted_faces"] = n
    elif action == "delete":
        # 删除人物：其人脸全部解绑回未认领（不删脸/照片数据），sample_role 恢复 auto 供自动归类重新判定
        pid = body.get("person_id")
        if not con.execute("SELECT 1 FROM person WHERE person_id=?", (pid,)).fetchone():
            raise ValueError("人物不存在")
        con.execute("UPDATE face_instance_v0 SET person_id=NULL, sample_role='auto' WHERE person_id=?", (pid,))
        con.execute("DELETE FROM person WHERE person_id=?", (pid,))
    else:
        raise ValueError("未知操作")
    con.commit()
    con.close()
    payload = get_people_payload()
    if action in ("assign_faces", "label_faces"):
        # 人工标注后自动归类：拉起 kNN 归属子进程(检测任务在跑时由其收尾统一归属)
        try:
            payload["auto_assign"] = faces_assign_run()
        except Exception as exc:
            payload["auto_assign"] = {"ok": False, "error": str(exc)}
    if action == "label_faces":
        payload["label_result"] = {"labeled": len(assigned), "created_persons": created}
        if new_faces:
            payload["label_result"]["new_faces"] = new_face_result
    if result_extra:
        payload.update(result_extra)
    # #16：人物增删改后失效人名词典缓存（family/roles 下次调用重读 DB）
    _invalidate_person_lexicon()
    return payload


def _display_ratio_for(w, h, mtype=None):
    """根据原图宽高选最近的瀑布比例：3:4 竖 / 4:3 横 / 1:1 方，最小化裁切损失。
    无尺寸时按 mtype 给一个合理默认（视频多竖拍 → 3:4，照片默认 4:3）。"""
    if not w or not h or w <= 0 or h <= 0:
        return '3x4' if mtype == 'video' else '4x3'
    r = w / h  # >1 横版，<1 竖版
    d43 = abs(r - 4/3); d11 = abs(r - 1); d34 = abs(r - 3/4)
    if d11 <= d43 and d11 <= d34:
        return '1x1'
    if d43 <= d34:
        return '4x3'
    return '3x4'


def _face_positions_for(asset_ids, con):
    """按 asset_id 批量取最大一张人脸的中心点（归一化坐标 x/y ∈ [0,1]，与 upright 空间
    一致；thumb 是 upright 等比缩放 → 前端 object-position 直接用百分比即可）。
    返回 {asset_id: {"x":..., "y":...}}，无人脸的 asset 不出现在结果里。"""
    if not asset_ids:
        return {}
    ph = ",".join("?" for _ in asset_ids)
    rows = con.execute(
        f"""SELECT asset_id, bbox_json FROM face_instance_v0
            WHERE asset_id IN ({ph}) AND bbox_json IS NOT NULL""",
        asset_ids).fetchall()
    best = {}
    for r in rows:
        try:
            b = json.loads(r["bbox_json"])
            x = float(b.get("x", 0)); y = float(b.get("y", 0))
            bw = float(b.get("w", 0)); bh = float(b.get("h", 0))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if bw <= 0 or bh <= 0:
            continue
        area = bw * bh
        cx = x + bw / 2; cy = y + bh / 2
        cur = best.get(r["asset_id"])
        if cur is None or area > cur[2]:
            best[r["asset_id"]] = (cx, cy, area)
    return {aid: {"x": cx, "y": cy} for aid, (cx, cy, _area) in best.items()}


def _quality_for(asset_ids, con):
    """T4 接线（2026-09-15）：按 asset_id 批量取画质（模糊档位 + 美学分）。
    asset_quality_v0 由离线模块回填（blur_detector / aesthetic_scorer，model quality-v1）。
    返回 {asset_id: {"blur": "sharp|soft|blurry", "aesthetic": float, "sharp": float}}。
    表不存在时静默返回空 dict —— 老库/未跑质量模块的实例不受影响。"""
    if not asset_ids:
        return {}
    out = {}
    ph = ",".join("?" for _ in asset_ids)
    try:
        rows = con.execute(
            f"""SELECT asset_id, blur_label, sharp_score, aesthetic
                FROM asset_quality_v0 WHERE asset_id IN ({ph})""",
            asset_ids).fetchall()
    except sqlite3.OperationalError:
        return {}
    for r in rows:
        out[r["asset_id"]] = {
            "blur": r["blur_label"],
            "aesthetic": (round(float(r["aesthetic"]), 1) if r["aesthetic"] is not None else None),
            "sharp": (round(float(r["sharp_score"])) if r["sharp_score"] is not None else None),
        }
    return out


def daily_best(day=None, limit=5, min_aesthetic=None):
    """T4 接线（2026-09-15）：每日精选 —— 按 LAION 美学分取当天前 N 张。
    day: 'YYYY-MM-DD'；省略时取「最新的**有美学分**的那天」。
    口径：只算墙面可见（排除过滤表 / 相似组非最佳 / 回收站 / 隐私），只要照片。
    返回 {day, available_days, items:[{id,time,aesthetic,blur,region}], total_day}

    注意：候选日必须从 asset_quality_v0 派生，不能从 media_asset 取「最新有照片的天」——
    新导入的照片还没被离线打分模块覆盖，那样会挑到空的一天（首版就是这么错的）。
    """
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    hidden = get_hidden_ids(con)
    con.execute("CREATE TEMP TABLE IF NOT EXISTS _hidden_ids_daily(asset_id TEXT PRIMARY KEY)")
    con.execute("DELETE FROM _hidden_ids_daily")
    con.executemany("INSERT OR IGNORE INTO _hidden_ids_daily VALUES(?)", [(a,) for a in hidden])
    # day_counts 必须是「墙面可见」口径：离线打分覆盖了但整批被过滤表/相似折叠吃掉的日子
    # （聊天导出、连拍折叠）不能进 available_days，否则日历上列出来、用户点进去一片空白。
    # 2026-09-15 上线实测命中：2026-08-26 有分但 0 张可见。
    aq_sql, aq_params = "", []
    if min_aesthetic is not None:
        aq_sql = " AND q.aesthetic >= ?"
        aq_params.append(min_aesthetic)
    try:
        day_rows = con.execute(
            f"""SELECT substr(ma.capture_time,1,10) d, count(*) n
               FROM asset_quality_v0 q
               JOIN media_asset ma ON ma.asset_id = q.asset_id
               WHERE ma.media_type='photo' AND q.aesthetic IS NOT NULL
                 AND ma.capture_time IS NOT NULL
                 AND ma.capture_time NOT LIKE '0000%'
                 AND length(ma.capture_time) >= 10
                 AND NOT EXISTS(SELECT 1 FROM _hidden_ids_daily h WHERE h.asset_id = ma.asset_id){aq_sql}
               GROUP BY d ORDER BY d DESC""", aq_params).fetchall()
    except sqlite3.OperationalError:
        day_rows = []
    if not day_rows:
        con.close()
        return {"day": None, "available_days": [], "items": [], "total_day": 0}
    day_counts = {r["d"]: r["n"] for r in day_rows}
    days = list(day_counts.keys())
    want = max(1, min(int(limit or 5), 50))
    floor = max(want * 2, 8)

    def items_for(d):
        rows = con.execute(
            """SELECT q.asset_id, q.aesthetic, q.blur_label, ma.capture_time,
                      g.region, g.province
               FROM asset_quality_v0 q
               JOIN media_asset ma ON ma.asset_id = q.asset_id
               LEFT JOIN asset_geo_v0 g ON g.asset_id = q.asset_id
               WHERE ma.media_type='photo' AND substr(ma.capture_time,1,10)=?
                 AND q.aesthetic IS NOT NULL
               ORDER BY q.aesthetic DESC LIMIT 400""", (d,)).fetchall()
        out = []
        for r in rows:
            if r["asset_id"] in hidden:
                continue
            if min_aesthetic is not None and (r["aesthetic"] or 0) < min_aesthetic:
                continue
            out.append({
                "id": r["asset_id"], "time": r["capture_time"],
                "aesthetic": round(float(r["aesthetic"]), 1),
                "blur": r["blur_label"],
                "region": r["region"] or r["province"] or "",
            })
            if len(out) >= want:
                break
        return out

    if day is not None:
        # 显式指定日期就照实返回：那天没有可见精选就给空列表（前端提示「这天暂无精选」），
        # 不能静默换成别的日子 —— 用户点的是哪一天，就得显示哪一天。
        items = items_for(day) if day in day_counts else []
    else:
        # 默认日：从「最近的、评分张数够」的日子往前找，优先选能凑满 want 张的那天。
        # 实测踩过两次坑：① 最近几天常只有 1~3 张新导入照片（离线模块还没覆盖）→ 选出寂寞精选；
        # ② 有些天整批是聊天导出/截图或连拍，几乎全被过滤表与相似折叠吃掉 → 精选开天窗。
        pool = [x for x in days if day_counts.get(x, 0) >= floor][:20] or days[:20]
        items, best_day = [], (pool[0] if pool else days[0])
        for cand in pool:
            got = items_for(cand)
            if len(got) >= want:
                items, best_day = got, cand
                break
            if len(got) > len(items):
                items, best_day = got, cand
        day = best_day
    con.close()
    pickable = [x for x in days if day_counts.get(x, 0) >= 5][:60]
    if day not in pickable:
        pickable = [day] + pickable[:59]
    return {"day": day, "available_days": pickable, "items": items,
            "total_day": day_counts.get(day, 0)}


def geo_map_data():
    """中国地图照片堆聚合：按 region 聚合 asset_geo_v0，返回每地平均经纬度/数量/代表缩略图/旅行标签数。
    口径与 /api/categories 的 locations 完全一致：墙面可见（排除过滤表/相似组非最佳，含白名单）。
    前端用 CHINA_MAP.project(lon,lat) 把经纬度投到地图 SVG 坐标，照片堆堆叠在地理位置上。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    hidden = get_hidden_ids(con)
    con.execute("CREATE TEMP TABLE IF NOT EXISTS _hidden_ids_v0(asset_id TEXT PRIMARY KEY)")
    con.execute("DELETE FROM _hidden_ids_v0")
    con.executemany("INSERT OR IGNORE INTO _hidden_ids_v0 VALUES(?)", [(a,) for a in hidden])
    NH = "NOT EXISTS(SELECT 1 FROM _hidden_ids_v0 h WHERE h.asset_id=ma.asset_id)"
    try:
        rows = con.execute(f"""
            WITH ranked AS (
              SELECT g.region, g.province, g.latitude, g.longitude, g.asset_id, ma.capture_time,
                     CASE WHEN st.asset_id IS NOT NULL THEN 1 ELSE 0 END is_travel,
                     ROW_NUMBER() OVER(PARTITION BY g.region ORDER BY
                       (ma.capture_time IS NULL OR ma.capture_time LIKE '0000%'),
                       substr(ma.capture_time,1,19) DESC) rn
              FROM asset_geo_v0 g
              JOIN media_asset ma ON ma.asset_id=g.asset_id
              LEFT JOIN scene_tag_v0 st ON st.asset_id=g.asset_id AND st.tag='旅行'
              WHERE g.region IS NOT NULL AND g.region<>'' AND {NH}
            )
            SELECT region, province, AVG(latitude) lat, AVG(longitude) lon,
                   COUNT(*) cnt, SUM(is_travel) travel_cnt,
                   GROUP_CONCAT(CASE WHEN rn<=5 THEN asset_id END) samples
            FROM ranked GROUP BY region ORDER BY cnt DESC
        """).fetchall()
    except sqlite3.OperationalError:
        rows = []
    regions = []
    for r in rows:
        samples = [s for s in (r["samples"] or "").split(",") if s]
        regions.append({
            "region": r["region"], "province": r["province"],
            "lat": round(r["lat"], 5) if r["lat"] is not None else None,
            "lon": round(r["lon"], 5) if r["lon"] is not None else None,
            "count": r["cnt"], "travel_count": r["travel_cnt"] or 0,
            "samples": samples,
        })
    # 省份聚合：省名标注 + 省级数量 + 代表照片；排除"未知"/"沿海"非标准省份（地图上无对应轮廓）
    prov_rows = con.execute(f"""
        WITH prank AS (
          SELECT g.province, g.latitude, g.longitude, g.asset_id, ma.capture_time,
                 CASE WHEN st.asset_id IS NOT NULL THEN 1 ELSE 0 END is_travel,
                 ROW_NUMBER() OVER(PARTITION BY g.province ORDER BY
                   (ma.capture_time IS NULL OR ma.capture_time LIKE '0000%'),
                   substr(ma.capture_time,1,19) DESC) rn
          FROM asset_geo_v0 g JOIN media_asset ma ON ma.asset_id=g.asset_id
          LEFT JOIN scene_tag_v0 st ON st.asset_id=g.asset_id AND st.tag='旅行'
          WHERE g.province IS NOT NULL AND g.province<>'' AND g.province NOT IN ('未知','沿海') AND {NH}
        )
        SELECT province, AVG(latitude) lat, AVG(longitude) lon,
               COUNT(*) cnt, SUM(is_travel) tc,
               GROUP_CONCAT(CASE WHEN rn<=5 THEN asset_id END) samples
        FROM prank GROUP BY province ORDER BY cnt DESC
    """).fetchall()
    provinces = [{"name": r["province"],
                  "lat": round(r["lat"], 5) if r["lat"] is not None else None,
                  "lon": round(r["lon"], 5) if r["lon"] is not None else None,
                  "count": r["cnt"], "travel_count": r["tc"] or 0,
                  "samples": [s for s in (r["samples"] or "").split(",") if s]} for r in prov_rows]
    # 自定义分类上图（2026-09-07）：用户人工建的分类（如「青岛」，照片不在 asset_geo_v0、无 GPS），
    # 按 _geo_region_coords()（GPS 均值 + GEO_CITY_PRESETS 预设）匹配到坐标后作为独立堆返回 customs；
    # 前端照片堆点击跳 /index.html?cat=custom&value=<category_id>（search_by_category 已有 custom 分支）。
    # 名字与已有 GPS 地区重名的跳过——该地点已有真实照片堆，避免同点双堆。
    customs = []
    try:
        _coord_map = {r["region"]: r for r in _geo_region_coords()}
        _region_names = {r["region"] for r in regions}
        _hidden_set = set(hidden)
        _recycle_ids = set()
        try:
            _recycle_ids = {r[0] for r in con.execute("SELECT asset_id FROM recycle_v0")}
        except sqlite3.OperationalError:
            pass
        _privacy_ids = set()
        try:
            _privacy_ids = {r[0] for r in con.execute("SELECT asset_id FROM privacy_v0")}
        except sqlite3.OperationalError:
            pass
        for _c in con.execute("SELECT category_id, name FROM user_category_v0").fetchall():
            _coord = _coord_map.get(_c["name"])
            if not _coord or _c["name"] in _region_names:
                continue
            _members = [r["asset_id"] for r in con.execute(
                "SELECT asset_id FROM user_category_member_v0 WHERE category_id=?", (_c["category_id"],))]
            _members = [a for a in _members
                        if a not in _hidden_set and a not in _recycle_ids and a not in _privacy_ids]
            if not _members:
                continue
            _q = ",".join("?" * min(len(_members), 200))
            _rows = con.execute(
                f"""SELECT asset_id FROM media_asset WHERE asset_id IN ({_q})
                    ORDER BY (capture_time IS NULL OR capture_time LIKE '0000%'),
                             substr(capture_time,1,19) DESC""",
                _members[:200]).fetchall()
            customs.append({
                "category_id": _c["category_id"], "name": _c["name"],
                "province": _coord.get("province", ""),
                "lat": _coord["lat"], "lon": _coord["lon"],
                "count": len(_members), "travel_count": 0,
                "samples": [r["asset_id"] for r in _rows[:5]],
            })
    except sqlite3.OperationalError:
        customs = []
    # 全量照片精确经纬度（轻量）：放大后散点显示，前端按缩放级别+视口过滤渲染
    pt_rows = con.execute(f"""
        SELECT g.asset_id, g.latitude, g.longitude, g.region,
               CASE WHEN st.asset_id IS NOT NULL THEN 1 ELSE 0 END t
        FROM asset_geo_v0 g JOIN media_asset ma ON ma.asset_id=g.asset_id
        LEFT JOIN scene_tag_v0 st ON st.asset_id=g.asset_id AND st.tag='旅行'
        WHERE g.latitude IS NOT NULL AND g.longitude IS NOT NULL AND {NH}
    """).fetchall()
    points = [{"id": r["asset_id"], "lat": round(r["latitude"], 5), "lon": round(r["longitude"], 5),
               "r": r["region"], "t": r["t"]} for r in pt_rows]
    total = con.execute(f"SELECT count(*) FROM media_asset ma WHERE {NH}").fetchone()[0]
    con.execute("DROP TABLE IF EXISTS _hidden_ids_v0")
    con.close()
    return {"regions": regions, "provinces": provinces, "customs": customs, "points": points, "total": total}


def search_by_category(cat, value, order=None, slim=False, offset=None, limit=None, quality=None):
    """按分类检索照片。cat: year/location/person/latest。order: asc/desc（默认按分类保持原有习惯）。
    2026-09-06 服务端分页：limit>0 时按 offset 窗口切片返回（附 total/has_more），
    避免 latest 一次返回 16408 张 4.2MB；不传 limit 的调用方（首页 slim、回收站等）行为不变。
    2026-09-15 T4：quality ∈ {sharp, soft, blurry} 时按 asset_quality_v0.blur_label 过滤
    （必须在分页切片之前过滤，否则 total/has_more 口径与前端展示对不上）。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    default_order = {"latest": "desc", "year": "asc", "location": "asc",
                     "person": "asc", "member": "desc", "type": "desc"}.get(cat, "desc")
    order = (order or default_order).lower()
    if order not in ("asc", "desc"):
        order = default_order
    if cat == "latest":
        rows = con.execute(
            f"""SELECT asset_id, capture_time FROM media_asset
               ORDER BY (capture_time IS NULL OR capture_time LIKE '0000%'),
                        substr(capture_time,1,19) {order.upper()}""").fetchall()
    elif cat == "year":
        rows = con.execute(
            f"""SELECT asset_id, capture_time FROM media_asset
               WHERE substr(capture_time,1,4)=? AND capture_time IS NOT NULL
               ORDER BY substr(capture_time,1,19) {order.upper()}""", (value,)).fetchall()
    elif cat == "location":
        # 2026-09-07：地图省级照片堆传的是省名（data-prov），老查询只匹配 g.region 省名查空 →
        # 点省级堆拿不到照片。region/province 不会重名（region 是市/景点名），OR 查询两种都命中。
        rows = con.execute(
            f"""SELECT g.asset_id, ma.capture_time FROM asset_geo_v0 g
               JOIN media_asset ma USING(asset_id)
               WHERE g.region=? OR g.province=?
               ORDER BY (ma.capture_time IS NULL OR ma.capture_time LIKE '0000%'),
                        substr(ma.capture_time,1,19) {order.upper()}""", (value, value)).fetchall()
    elif cat == "person":
        pid = con.execute("SELECT person_id FROM person WHERE display_name=?", (value,)).fetchone()
        if not pid:
            con.close()
            return {"answer": f"没有「{value}」的照片。", "assets": [], "memory": [], "persons": [], "intent": {"cat": cat}}
        rows = con.execute(
            f"""SELECT DISTINCT fi.asset_id, ma.capture_time FROM face_instance_v0 fi
               JOIN media_asset ma USING(asset_id)
               WHERE fi.person_id=?
               ORDER BY (ma.capture_time IS NULL OR ma.capture_time LIKE '0000%'),
                        substr(ma.capture_time,1,19) {order.upper()}""", (pid["person_id"],)).fetchall()
    elif cat == "member":
        # 反向映射：显示名→owner_label（person_alias kind='source' 反转）
        _rev = {}
        for _alias, _disp in _person_lexicon().get("source_aliases", {}).items():
            _rev.setdefault(_disp, _alias)
        owner = _rev.get(value, value)
        if not owner:
            con.close()
            return {"answer": f"没有「{value}」的来源。", "assets": [], "memory": [], "persons": [], "intent": {"cat": cat}}
        # member（来源）原来 ORDER BY 缺 IS NULL/LIKE '0000%'，asc 时 NULL/零时间照片堆最前破坏时间线；同时 substr(...,1,19) 忽略时区后缀与微秒，避免 +00:00 vs +08:00 的小时级错排
        rows = con.execute(
            f"""SELECT DISTINCT ma.asset_id, ma.capture_time FROM media_asset ma
               JOIN media_file mf USING(asset_id) JOIN source s USING(source_id)
               WHERE s.owner_label=?
               ORDER BY (ma.capture_time IS NULL OR ma.capture_time LIKE '0000%'),
                        substr(ma.capture_time,1,19) {order.upper()}""", (owner,)).fetchall()
    elif cat == "type":
        media_type = "video" if value == "video" else "photo"
        rows = con.execute(
            f"""SELECT asset_id, capture_time FROM media_asset
               WHERE media_type=?
               ORDER BY (capture_time IS NULL OR capture_time LIKE '0000%'),
                        substr(capture_time,1,19) {order.upper()}""", (media_type,)).fetchall()
    elif cat == "scene":
        # 多标签检索：scene_tag_v0 多对多，一张照片可在多个标签下出现
        rows = con.execute(
            f"""SELECT st.asset_id, ma.capture_time FROM scene_tag_v0 st
               JOIN media_asset ma USING(asset_id)
               WHERE st.tag=?
               ORDER BY (ma.capture_time IS NULL OR ma.capture_time LIKE '0000%'),
                        substr(ma.capture_time,1,19) {order.upper()}""",
            (value,)).fetchall()
    elif cat == "custom":
        # 自定义分类：用户人工归属，遍历全部成员（不过滤 hidden，用户修正意图优先）
        name = con.execute("SELECT name FROM user_category_v0 WHERE category_id=?", (value,)).fetchone()
        if not name:
            con.close()
            return {"answer": f"没有「{value}」的自定义分类。", "assets": [], "memory": [], "persons": [], "intent": {"cat": cat}}
        rows = con.execute(
            f"""SELECT m.asset_id, ma.capture_time FROM user_category_member_v0 m
               JOIN media_asset ma USING(asset_id)
               WHERE m.category_id=?
               ORDER BY (ma.capture_time IS NULL OR ma.capture_time LIKE '0000%'),
                        substr(ma.capture_time,1,19) {order.upper()}""",
            (value,)).fetchall()
    elif cat == "recycle":
        # 回收站：软删除的照片只在这里可见（点「恢复」回到正常分类）
        # 2026-09-06 默认按删除时间排序（r.created_at），order 参数控制方向
        _init_recycle_table(con)
        rows = con.execute(
            f"""SELECT ma.asset_id, ma.capture_time FROM recycle_v0 r
               JOIN media_asset ma USING(asset_id)
               ORDER BY r.created_at {order.upper()}, ma.capture_time IS NULL, ma.capture_time {order.upper()}""").fetchall()
    else:
        con.close()
        return {"answer": "未知分类。", "assets": [], "memory": [], "persons": [], "intent": {"cat": cat}}

    # 管理页需要遍历完整集合；前端仍按 90 项一批渲染和懒加载缩略图。
    media_info = {r["asset_id"]: (r["media_type"], r["width"], r["height"]) for r in con.execute("SELECT asset_id, media_type, width, height FROM media_asset")}
    # 位置信息：首页照片墙在预览下方展示拍摄地（region=具体地点，province=省份）
    geo_info = {r["asset_id"]: (r["region"], r["province"]) for r in con.execute("SELECT asset_id, region, province FROM asset_geo_v0")}
    # 2026-09-04：来源归属 owner（瀑布流右键/拖动改分类时显示「当前属于谁」用）
    owner_info = {r["asset_id"]: r["owner_label"] for r in con.execute(
        """SELECT mf.asset_id, s.owner_label
           FROM media_file mf JOIN source s ON s.source_id=mf.source_id
           WHERE mf.availability='original'""")}
    # 2026-08-29 接上 Codex 的垃圾过滤表：截图/聊天导出/录屏/文档照标记 hidden，墙上不再出现
    filtered_ids = get_hidden_ids(con)
    assets = [{"id": r["asset_id"], "time": r["capture_time"], "type": media_info.get(r["asset_id"], ("photo",None,None))[0], "width": media_info.get(r["asset_id"], (None,None,None))[1], "height": media_info.get(r["asset_id"], (None,None,None))[2], "region": (geo_info.get(r["asset_id"]) or (None, None))[0], "province": (geo_info.get(r["asset_id"]) or (None, None))[1], "owner": owner_info.get(r["asset_id"]), "hidden": r["asset_id"] in filtered_ids} for r in rows]
    # 回收站软删除：除 cat='recycle' 外，正常分类一律不返回回收站里的照片
    # 隐私相册同理：所有分类视图都不返回隐私照片（隐私内容只走 /api/privacy list）
    if cat != "recycle":
        try:
            recycle_ids = {r[0] for r in con.execute("SELECT asset_id FROM recycle_v0")}
        except sqlite3.OperationalError:
            recycle_ids = set()
        if recycle_ids:
            assets = [a for a in assets if a["id"] not in recycle_ids]
    if _privacy_has_rows(con):
        privacy_ids = {r[0] for r in con.execute("SELECT asset_id FROM privacy_v0")}
        if privacy_ids:
            assets = [a for a in assets if a["id"] not in privacy_ids]
    # 2026-09-15 T4：画质筛选（必须在 slim/分页之前，保证 total/has_more 口径一致）
    quality = (quality or "").strip().lower()
    if quality in ("sharp", "soft", "blurry"):
        try:
            keep = {r[0] for r in con.execute(
                "SELECT asset_id FROM asset_quality_v0 WHERE blur_label=?", (quality,))}
        except sqlite3.OperationalError:
            keep = set()
        assets = [a for a in assets if a["id"] in keep]
    if slim:
        # 首页照片墙 slim 模式：只要墙上渲染用得到的字段。
        # ① 丢掉 hidden 项（前端 renderRibbon 第一件事就是 filter 掉，白传 25%）
        # ② time 截到 YYYY-MM-DDTHH:MM:SS（聚焦卡片要显示真实拍摄时刻；
        #    纯日期的记录前端按「只有日期」展示，不再伪造 08:00:00）
        assets = [{"id": a["id"], "time": (a["time"] or "")[:19], "type": a["type"],
                   "width": a["width"], "height": a["height"], "region": a["region"],
                   "province": a["province"]}
                  for a in assets if not a.get("hidden")]
    # 2026-09-06 服务端分页：hidden（过滤表/相似折叠）原来由前端渲染前剔除，
    # 分页后必须在切片前剔除，否则 total/offset 与前端最终展示数量口径不一致。
    # custom/recycle 保持原口径（hidden 全保留：人工归属意图优先 / 回收站本来就全是 hidden）。
    paged = limit is not None and not isinstance(limit, bool) and isinstance(limit, (int, float)) and limit > 0
    if paged:
        if cat not in ("custom", "recycle"):
            assets = [a for a in assets if not a.get("hidden")]
        total = len(assets)
        off = max(0, int(offset or 0))
        lim = min(max(1, int(limit)), 4000)
        page_assets = assets[off:off + lim]
        has_more = off + lim < total
    else:
        total = None
        page_assets = assets
    label_map = {"year": f"{value}年", "location": value, "person": value, "member": value,
                 "type": "视频" if value == "video" else "照片", "latest": "全部成员", "scene": value,
                 "custom": value, "recycle": "回收站"}
    if cat == "custom":
        crow = con.execute("SELECT name FROM user_category_v0 WHERE category_id=?", (value,)).fetchone()
        if crow: label_map["custom"] = crow["name"]
    label = label_map[cat]
    # 2026-09-04 瀑布流：给每个 asset 附上「展示比例」(3:4/4:3/1:1，最贴近原图) 和
    # 「最大人脸中心」(object-position 用)。前端把 .thumb-wrap 的 aspect-ratio 设为
    # display_ratio，把 <img> 的 object-position 设为 face_pos（无人脸则默认居中），
    # 实现「按时间顺序排列 + 智能裁切到三种比例 + 人物居中」的瀑布流。
    # 2026-09-06：只对实际返回的窗口计算（分页时省去全量 16k 项的人脸查询）。
    face_pos_map = _face_positions_for([a["id"] for a in page_assets], con)
    # T4 接线（2026-09-15）：画质（模糊档位 + 美学分）随列表下发，前端出「清晰/偏软/模糊」徽标 +
    # 「只看清晰」开关。只算当前窗口，代价与 face_pos_map 同级。
    quality_map = _quality_for([a["id"] for a in page_assets], con)
    # 2026-09-23 Log 还原标记：让查看器把最高清层换成已还原的 /preview
    # （/orig 永远是未还原的原片，不给它挂滤镜）。同窗口一次查完。
    logc_map = _logcolor_for_assets([a["id"] for a in page_assets], con)
    for a in page_assets:
        a["display_ratio"] = _display_ratio_for(a.get("width"), a.get("height"), a.get("type"))
        a["face_pos"] = face_pos_map.get(a["id"])
        if a["id"] in logc_map:
            a["logcolor"] = 1
        q = quality_map.get(a["id"])
        if q and a.get("type") == "photo":
            a["blur"] = q["blur"]
            a["aesthetic"] = q["aesthetic"]
            a["sharp"] = q["sharp"]
    con.close()
    # 数量口径与照片墙一致：只数可见资产（rows 含 hidden 供管理页遍历）。
    # 自定义分类例外：成员为用户人工归属（意图优先于可见集），计数=全部成员（回收站已在上面排除），
    # 与侧栏 /api/categories、分类管理器 /api/catman 的 customs 口径一致。
    # 分页时 hidden 已在切片前剔除，total 即可见数。
    if total is None:
        visible_n = len(assets) if cat == "custom" else sum(1 for a in assets if not a.get("hidden"))
    else:
        visible_n = total
    result = {
        "answer": f"「{label}」共 {visible_n} 项，当前可分批查看全部影像。",
        "assets": page_assets, "memory": [], "persons": [], "intent": {"cat": cat, "value": value},
    }
    if total is not None:
        result.update({"total": total, "offset": off, "limit": lim, "has_more": has_more})
    if cat == "recycle":
        # 回收站里全是 hidden（被软删除），真实数量按行数算，供侧栏「回收站 · N」展示
        result["total"] = len(assets)
    return result


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def get_ab_review_payload():
    """返回模型盲审池；不暴露模型名、分数或原始排名。"""
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=60000")
    con.row_factory = sqlite3.Row
    exists = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='ab_blind_review_v1'"
    ).fetchone()
    if not exists:
        con.close()
        return {"status": "not_ready", "groups": [], "reviewed": 0, "total": 0}
    rows = con.execute(
        """SELECT b.query_id,b.variant,b.query_text,b.asset_id,b.blind_order,b.label,ma.media_type
           FROM ab_blind_review_v1 b JOIN media_asset ma USING(asset_id)
           ORDER BY b.query_id,b.variant,b.blind_order"""
    ).fetchall()
    groups = []
    current = None
    for row in rows:
        key = (row["query_id"], row["variant"])
        if current is None or current["key"] != key:
            current = {"key": key, "query_id": row["query_id"], "variant": row["variant"],
                       "query_text": row["query_text"], "items": []}
            groups.append(current)
        current["items"].append({"asset_id": row["asset_id"], "order": row["blind_order"],
                                 "label": row["label"], "type": row["media_type"]})
    total = len(rows)
    reviewed = sum(r["label"] is not None for r in rows)
    con.close()
    for group in groups:
        group.pop("key", None)
    return {"status": "ready", "groups": groups, "reviewed": reviewed, "total": total}


def save_ab_review_label(body):
    label = body.get("label")
    if label not in ("relevant", "not_relevant", "unsure"):
        raise ValueError("invalid label")
    con = sqlite3.connect(DB, timeout=30)
    con.execute("PRAGMA busy_timeout=60000")
    # zh / bilingual 只是同一语义的文本变体；同一图片的人工相关性不应重复判两次。
    # 因此一次盲审同步到该 query_id 的所有文本变体，但绝不跨查询传播。
    cur = con.execute(
        """UPDATE ab_blind_review_v1 SET label=?,reviewer=?,reviewed_at=?,note=?
           WHERE query_id=? AND asset_id=?""",
        (label, "local_blind_review", now_iso(), body.get("note"),
         body.get("query_id"), body.get("asset_id")),
    )
    con.commit(); con.close()
    if cur.rowcount < 1:
        raise ValueError("review item not found")
    return {"ok": True, "updated_variants": cur.rowcount}


# ============ 连拍去重 ============
def dedup_burst(con, assets):
    """只折叠已确认的完全重复/衍生副本；绝不按时间或视觉相似度合并。"""
    if not assets:
        return []

    by_id = {aid: (aid, score, t) for aid, score, t in assets}
    parent = {aid: aid for aid in by_id}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    ids = list(by_id)
    placeholders = ",".join("?" * len(ids))
    rows = con.execute(
        f"""SELECT left_asset_id,right_asset_id FROM asset_relationship_v01
            WHERE status='confirmed' AND relationship_type IN ('EXACT_DUPLICATE','DERIVED_COPY')
              AND left_asset_id IN ({placeholders}) AND right_asset_id IN ({placeholders})""",
        (*ids, *ids),
    ).fetchall()
    for row in rows:
        union(row["left_asset_id"], row["right_asset_id"])
    grouped = {}
    for aid in ids:
        grouped.setdefault(find(aid), []).append(by_id[aid])

    result = []
    for group in grouped.values():
        if len(group) == 1:
            aid, _, t = group[0]
            result.append({"id": aid, "time": t, "hidden": False})
            continue
        # 同一原始影像的衍生副本中选择检索分、质量、分辨率和文件体积最佳者。
        best = None
        best_key = None
        for aid, score, t in group:
            q = con.execute("SELECT quality_score, width, height FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
            qs = q["quality_score"] if q and q["quality_score"] else 0
            res = (q["width"] or 0) * (q["height"] or 0) if q else 0
            f = con.execute("SELECT byte_size FROM media_file WHERE asset_id=? ORDER BY byte_size DESC LIMIT 1", (aid,)).fetchone()
            size = f["byte_size"] if f else 0
            key = (score or 0, qs, res, size)
            if best is None or key > best_key:
                best = (aid, t)
                best_key = key
        for aid, _, t in group:
            result.append({"id": aid, "time": t, "hidden": aid != best[0]})

    result.sort(key=lambda x: x["time"] or "", reverse=True)
    return result


# ============ Diversity Reranker (MMR, 2026-09-04 #5) ============
def _mmr_sim(a, b, date_map, tag_map):
    """MMR 重排的两两相似度（纯函数，便于单测）：同 capture_date +0.5，
    scene_tag Jaccard(ta,tb) 加权 0.5（max 0.5）。mmr_rerank_topk 内部调用。"""
    s = 0.0
    if date_map.get(a) == date_map.get(b):
        s += 0.5
    ta, tb = tag_map.get(a, set()), tag_map.get(b, set())
    if ta and tb:
        s += 0.5 * len(ta & tb) / len(ta | tb)
    return s


def mmr_rerank_topk(con, scored, k=10, lam=0.7):
    """Maximal Marginal Relevance 重排前 k 张，让结果的多样性可见。
    相似度 = 同 capture_date (0.5) + 同 scene_tag Jaccard (max 0.5)。
    剩余 11+ 名不动，避免破坏尾部时间倒序。
    仅当 ≥ k+5 张才启用（小于等于 k 没必要重排）。"""
    if len(scored) <= k + 5:
        return scored
    head = scored[:50]
    tail = scored[50:]   # 2026-09-06：保留 scored[50:] 全部尾部不动。原实现
                         # 直接 return selected + pool 会把 50 之外的 ~2150 张
                         # 全部丢掉，用户感知"搜出几十张"。
    aid_list = [a[0] for a in head]
    ph = ",".join("?" * len(aid_list))
    # 一次预拉：日期 + scene_tag set
    date_map = {}
    for r in con.execute(
            f"SELECT asset_id, capture_time FROM media_asset WHERE asset_id IN ({ph})",
            aid_list):
        if r["capture_time"]:
            date_map[r["asset_id"]] = r["capture_time"][:10]
    tag_map = {}
    for r in con.execute(
            f"SELECT asset_id, tag FROM scene_tag_v0 WHERE asset_id IN ({ph})",
            aid_list):
        tag_map.setdefault(r["asset_id"], set()).add(r["tag"])

    pool = list(head)
    # 当前排序：score desc + 同分时间 desc（参见 search 函数末尾）；
    # 重排前先按 (score desc, time desc) 整体排序 pool 作为基线。
    pool.sort(key=lambda x: (x[1], x[2] or ""), reverse=True)
    selected = [pool.pop(0)]  # 最高分作为锚
    while len(selected) < k and pool:
        # 归一化 score 到 [0,1]
        max_s = max((s[1] for s in selected + pool), default=1.0) or 1.0
        min_s = min((s[1] for s in selected + pool), default=0.0)
        denom = (max_s - min_s) or 1.0
        best_i, best_score = 0, -1e9
        for i, cand in enumerate(pool):
            rel = (cand[1] - min_s) / denom
            max_sim = max(_mmr_sim(cand[0], sel[0], date_map, tag_map) for sel in selected) if selected else 0
            mmr_s = lam * rel - (1 - lam) * max_sim
            if mmr_s > best_score:
                best_score = mmr_s
                best_i = i
        selected.append(pool.pop(best_i))
    return selected + pool + tail


# ============ 场景过滤（CLIP） ============
_SCENE_CACHE = {}  # {scene_keyword: {asset_id: is_scene}}
_SCENE_SCORE_CACHE = {}  # {scene_keyword: {asset_id: positive_margin}}
_EMOTION_SCORE_CACHE = {}  # {asset_id: validated facial-expression probability}


def scene_filter(con, candidate, scene_kw, protect=None):
    """用独立盲审胜出的中文 SIGLIP 匹配山景/海边，排除无场景近景。

    protect：GPS 已确认山区/沿海且含已确认家人人脸的资产。这些是强证据，
    语义场景模型（负类含"人物近景"）不得否决——家人与山景的合影正是
    "一家人在山上"这类查询要找的目标，不能因人物占画面大而被删。
    """
    protect = protect or set()
    if scene_kw in _SCENE_CACHE:
        cache = _SCENE_CACHE[scene_kw]
        return {a for a in candidate if cache.get(a, True) or a in protect}

    if scene_kw == "山":
        texts = ["山峦 山峰 群山 山景 户外", "室内 商场 家里", "人物近景 自拍 无山景"]
    else:
        texts = ["大海 海浪 海滩 海面 海岸线", "室内 商场 家里", "人物近景 自拍 无海景"]
    scores = local_siglip_scores(texts, candidate)
    if not scores:
        return candidate
    result = {aid: values[0] > max(values[1:]) for aid, values in scores.items()}
    _SCENE_SCORE_CACHE[scene_kw] = {
        aid: values[0] - max(values[1:]) for aid, values in scores.items()
    }
    _SCENE_CACHE[scene_kw] = result
    filtered = {aid for aid in candidate if result.get(aid, True) or aid in protect}
    # GPS 山区/沿海 + 已确认人物是更强证据；语义模型只能辅助，不能把唯一真实候选删掉。
    return filtered or set(candidate)


def emotion_filter(con, candidate, person_names, threshold=0.90):
    """Keep only high-precision visible-happiness evidence for requested people.

    The threshold was selected on a stratified, model-blind local audit.  This is
    facial-expression evidence only and is never described as a person's true
    internal emotional state.
    """
    global _EMOTION_SCORE_CACHE
    if not candidate or not person_names:
        return set()
    try:
        names = sorted(set(person_names))
        name_ph = ",".join("?" * len(names))
        asset_ph = ",".join("?" * len(candidate))
        rows = con.execute(
            f"""SELECT fi.asset_id,max(fx.happiness_score) score
                FROM face_expression_v0 fx
                JOIN face_instance_v0 fi USING(face_instance_id)
                JOIN person p USING(person_id)
                WHERE fx.inference_status='success'
                  AND p.display_name IN ({name_ph})
                  AND fi.asset_id IN ({asset_ph})
                GROUP BY fi.asset_id""",
            (*names, *candidate),
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    _EMOTION_SCORE_CACHE = {r["asset_id"]: float(r["score"]) for r in rows}
    return {aid for aid, score in _EMOTION_SCORE_CACHE.items() if score >= threshold}


# ============ 意图解析 ============
def parse_intent(question):
    """用 LLM 把自然语言查询转成结构化意图。"""
    # 空查询不进 LLM（实测 LLM 对空问题会虚构意图，如"孩子+海边"），
    # 返回空意图 → 检索无约束 → 按时间排最近的影像，即"看看有什么"
    if not (question or "").strip():
        return {"parser": "local_rules"}
    # Local First：已支持的家庭查询不发送到第三方。只有本地规则完全无法理解时才云端兜底。
    local = fallback_parse(question)
    understood = any(local.get(k) for k in (
        "persons", "events", "memory_keywords", "relationship", "age_constraint",
        "object", "exclude", "exclude_scene", "same_event_cross_source", "emotion",
    )) or local.get("media_type") not in (None, "any") \
        or bool((local.get("time") or {}).get("start"))
    # 2026-09-04 (#4) 负面信号：fallback 和 LLM 路径都要叠加 exclude_scene。
    _neg_scene = _extract_neg_scene(question)
    if _neg_scene:
        local["exclude_scene"] = _neg_scene
    if understood:
        local["parser"] = "local_rules"
        return local
    # #16 隐私清洗：已知成员与示例人名运行时从 DB 构造，出厂 prompt 零人名
    _lx = _person_lexicon()
    _fam_rows = []
    try:
        _c0 = sqlite3.connect(DB, timeout=10)
        for _n, _rl, _bd in _c0.execute(
            """SELECT display_name, relationship_label, birth_date FROM person
               WHERE identity_status='confirmed' ORDER BY birth_date"""
        ):
            _fam_rows.append(f"{_n}({_rl or '家人'},{(_bd or '')[:10]})")
        _c0.close()
    except sqlite3.Error:
        pass
    _fam_line = "、".join(_fam_rows) if _fam_rows else "（从 person 表读取）"
    _dad = _lx["roles"].get("爸爸") or "爸爸"
    _mom = _lx["roles"].get("妈妈") or "妈妈"
    _kid1 = _lx["children"][0] if _lx["children"] else "老大"
    _kid2 = _lx["children"][1] if len(_lx["children"]) > 1 else "老二"
    _kids_all = "、".join(_lx["children"]) if _lx["children"] else "孩子们"
    system = f"""你是家庭记忆检索系统的意图解析器。把用户的中文查询解析成结构化 JSON。
已知家庭成员：{_fam_line}。
返回 JSON 格式：
{{
  "persons": ["要查找的人物姓名，不含关系词"],
  "relationship": {{"from": "家长姓名", "type": "father_of/mother_of"}},
  "events": ["地点/事件关键词，如新疆/赛里木湖/山"],
  "memory_keywords": ["口述记忆关键词"],
  "time": {{"start": "YYYY-MM-DD或空", "end": "YYYY-MM-DD或空", "year": 数字或空}},
  "age_constraint": {{"person": "孩子姓名或'孩子'", "max_age": 数字}},
  "media_type": "photo/video/any",
  "count": 数字或空,
  "object": "物品词或空，如 杯子/礼盒/装修/玩具/车/花",
  "exclude": ["screenshot/download/forwarded"],
  "same_event_cross_source": true或false,
  "emotion": "happy或空"
}}
示例：
- "找{_dad}的孩子5岁以前的照片" -> {{"persons":[],"relationship":{{"from":"{_dad}","type":"father_of"}},"age_constraint":{{"person":"孩子","max_age":5}},"media_type":"photo"}}
- "找一家人在山上的照片" -> {{"persons":[{_kids_all}],"events":["山"],"media_type":"photo"}}
- "看看{_kid1}和爸爸一起的照片" -> {{"persons":["{_kid1}","{_dad}"],"media_type":"photo"}}
- "{_kid2}当时几岁？" -> {{"persons":["{_kid2}"],"age_constraint":{{"person":"{_kid2}"}}}}
- "我们第一次去赛里木湖是什么时候？" -> {{"memory_keywords":["赛里木湖"],"time":{{"year":null}}}}
- "杯子的照片" -> {{"object":"杯子","media_type":"photo"}}
- "礼盒" -> {{"object":"礼盒"}}
- "家里装修的照片" -> {{"object":"装修"}}
- "同一次旅行里两部手机拍的照片和视频" -> {{"events":["旅行"],"same_event_cross_source":true,"media_type":"any"}}
- "排除截图、聊天记录和下载图片" -> {{"exclude":["screenshot","download","forwarded"],"media_type":"any"}}
- "{_kid1}特别开心的照片" -> {{"persons":["{_kid1}"],"emotion":"happy","media_type":"photo"}}
- "海边的照片" -> {{"events":["海边"],"media_type":"photo"}}
- "下雪的照片" -> {{"events":["下雪"],"media_type":"photo"}}
规则：
- "孩子"在 persons 里展开为{_kids_all}，但 age_constraint.person 可保留"孩子"
- 人物只写真实姓名（已知家庭成员名单里的名字）
- 物品类查询（无人物无地点）：填 object 字段
- **场景/景物/氛围词（海边/沙滩/雪景/美食/夜景/花草/生日/宠物/建筑/合影/室内等）一律填入 events，严禁虚构 persons/relationship/age_constraint**——用户没提人就不要猜人
- 只输出 JSON，不要其他文字"""

    text, _err = llm_chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
        temperature=0.1, max_tokens=500)
    if text:
        # 提取 JSON
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                _llm_intent = json.loads(m.group(0))
            except Exception:
                # 容忍尾逗号
                cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
                _llm_intent = json.loads(cleaned)
            # 2026-09-04 (#4) 负面信号：LLM 路径也补 exclude_scene
            _neg_scene = _extract_neg_scene(question)
            if _neg_scene:
                _llm_intent["exclude_scene"] = _neg_scene
            return _llm_intent
    # 兜底：规则解析
    _fb = fallback_parse(question)
    _neg_scene = _extract_neg_scene(question)
    if _neg_scene:
        _fb["exclude_scene"] = _neg_scene
    return _fb


# ============ 场景词表（模块级常量，_extract_neg_scene 和 fallback_parse 共用） ============
_SCENE_ALIAS = {
    # 海/雪/夜景/建筑 等基础场景
    "海边": "海景", "沙滩": "海景", "大海": "海景", "海": "海景",
    "下雪": "雪景", "堆雪人": "雪景", "雪景": "雪景",
    "吃饭": "美食", "好吃的": "美食", "大餐": "美食",
    "生日": "生日", "蛋糕": "生日",
    "夜景": "夜景", "烟花": "夜景", "灯光": "夜景",
    "花": "花草", "花草": "花草", "公园": "花草",
    "宠物": "宠物", "猫": "宠物", "狗": "宠物",
    "建筑": "建筑", "城市": "建筑",
    "合影": "合影", "合照": "合影", "全家福": "合影",
    "室内": "室内",
    "儿童": "儿童", "小朋友": "儿童",
    "火车": "车辆", "汽车": "车辆", "车": "车辆",
}
_PHRASE_LEXICON = {
    # 美食细分（不含已进 OBJECT_TEXTS 的：火锅/烧烤）
    "家常菜": "美食", "寿司": "美食", "面条": "美食", "饺子": "美食",
    "点心": "美食", "早餐": "美食", "夜宵": "美食",
    "甜品": "美食", "饮料": "美食", "奶茶": "美食",
    # 场景细分
    "商场": "建筑", "超市": "建筑", "餐厅": "美食",
    "酒店": "建筑", "医院": "建筑", "学校": "建筑", "游乐场": "儿童",
    "花园": "花草", "泳池": "海景", "海里": "海景",
    "雪景": "雪景", "雪地": "雪景",
    # 节日
    "春节": "节日", "过年": "节日", "中秋": "节日", "圣诞": "节日",
    "元宵": "节日", "端午": "节日", "元宵节": "节日",
    # 室内场景
    "客厅": "室内", "卧室": "室内", "厨房": "室内", "阳台": "室内",
    "卫生间": "室内", "浴室": "室内", "书房": "室内",
    # 状态/活动
    "旅游": "旅行", "徒步": "旅行", "爬山": "山景",
}


def fallback_parse(question):
    """LLM 不可用时的简单规则解析。"""
    intent = {"persons": [], "events": [], "memory_keywords": [], "media_type": "any"}
    # 2026-09-06：人名别名预处理——Web Speech / 拼音输入法 / 用户口音常把
    # 标准姓名识别成错字。变体数据在 person_alias 表（#16 隐私清洗后代码零人名）。
    # 在所有规则匹配前先把变体替换回标准名，否则下面人物匹配完全无法命中。
    # 长别名优先匹配避免短别名"小"抢先误配等问题；按字符长度降序。
    _lex = _person_lexicon()
    _aliases_by_len = sorted(
        ((alias, canonical) for canonical, aliases in _lex["name_aliases"].items() for alias in aliases if alias and alias != canonical),
        key=lambda x: -len(x[0]))
    _normalized = question
    for alias, canonical in _aliases_by_len:
        if alias in _normalized:
            _normalized = _normalized.replace(alias, canonical)
    question = _normalized
    # 地点/事件：静态基础词典 + 从 asset_geo_v0/user_trip 动态加载全部地区名（含斜杠拆词）
    # 2026-09-06：补静态地名词典。库里没有大理/丽江等云南时期元数据时，光从
    # 数据库 dynamic load 出来的 _GEO_WORDS 全无这些地名，会让 search 直接
    # 走 blocked；补静态词典后至少能把"大理/丽江/苍山/洱海..."识别为 events，
    # search 进入 geo 路径走到"四张表全 0"分支，前端 hint 才能告诉用户
    # "库里没有这些照片记录"。如果库里真有这些照片的数据，search 同样会
    # 经 geo 路径精确命中（asset_geo_v0.region LIKE '%大理%'），跟之前行为一致。
    _STATIC_GEO = {
        # 云南
        "大理", "丽江", "云南", "苍山", "洱海", "玉龙雪山", "玉龙", "束河", "拉市海",
        "喜洲", "双廊", "西双版纳", "版纳", "香格里拉", "抚仙湖", "昆明", "滇池",
        # 其他省常见别名（库里有 region 就直接命中；库里没有也至少让 hint 知道是在问地点）
        "贵阳", "成都", "广州", "深圳", "杭州", "苏州", "青岛", "大连", "三亚",
        "哈尔滨", "长春", "沈阳", "兰州", "西宁", "银川", "呼和浩特", "乌鲁木齐",
        "拉萨", "桂林", "阳朔", "北海", "漠河", "长白山", "天涯", "海角",
        # 已有 region 不一一列出，这里只补"库里没有但是高频被问到"的
    }
    global _GEO_WORDS
    if _GEO_WORDS is None:
        words = {"赛里木湖", "新疆", "旅行", "山", "海边"}
        words |= _STATIC_GEO
        try:
            con = sqlite3.connect(DB, timeout=10)
            for row in con.execute(
                    "SELECT region FROM asset_geo_v0 UNION SELECT province FROM asset_geo_v0 "
                    "UNION SELECT name FROM user_trip"):
                for part in str(row[0] or "").split("/"):
                    part = part.strip()
                    if len(part) >= 2:
                        words.add(part)
            con.close()
        except Exception:
            pass
        _GEO_WORDS = words
    matched = []
    for kw in sorted(_GEO_WORDS, key=len, reverse=True):
        if kw in question:
            # 长词已命中时跳过它包含的短词（如"武夷山"命中后不再单加"山"），
            # 避免"山"把全部山区照片并进结果造成假命中
            if any(kw in m for m in matched):
                continue
            matched.append(kw)
    intent["events"] = intent["events"] + matched
    # 场景词（_SCENE_ALIAS / _PHRASE_LEXICON 是模块级常量，2026-09-04 提升以便复用）。
    # 实测本地 7B 模型意图解析质量差（"雪景"→虚构孩子年龄约束、"美食"→海边），
    # 场景类查询规则即可准确解析，LLM 留给真正复杂的自然语言问题。
    scene_hits = []
    # 2026-09-17: 长词守卫 —— 问句含更长物品词时，短别名不得再命中场景
    # （「婴儿车」里的「车」曾把 events 塞进"车辆"，与 object=婴儿车 相交只剩 2 张）。
    _obj_word_set = ("杯子", "礼盒", "蛋糕", "汽车", "室内装修", "装修", "下雨", "雨伞", "玩具",
                     "书包", "背包", "行李箱", "自行车", "单车", "高铁", "火车", "动车", "飞机", "机场",
                     "轮船", "游船", "婴儿车", "推车", "气球", "风筝", "帐篷", "露营", "烧烤",
                     "火锅", "灯笼", "春联", "红包", "滑雪", "雪橇", "乐器", "吉他", "钢琴",
                     "船", "车", "花")

    def _shadowed(w):
        # 等值也算被遮蔽：alias「火车→车辆」让位给 object=火车 的专属标签映射
        return any(o in question and (w in o or o in w) for o in _obj_word_set)
    for w, tag in _SCENE_ALIAS.items():
        if w in question and tag not in scene_hits and not _shadowed(w):
            scene_hits.append(tag)
    for w, tag in _PHRASE_LEXICON.items():
        if w in question and tag not in scene_hits and not _shadowed(w):
            scene_hits.append(tag)
    for tag in ("山景", "海景", "美食", "花草", "夜景", "雪景", "建筑",
                "江河", "宠物", "儿童", "生日", "车辆", "室内", "合影", "截图"):
        if tag in question and tag not in scene_hits:
            scene_hits.append(tag)
    if scene_hits:
        intent["events"] = intent["events"] + scene_hits

    # 人物
    if "孩子" in question and "5岁" in question and "孩子" not in intent["persons"]:
        # "孩子5岁前" → relationship + age
        pass
    for name in _lex["family"]:
        if name in question:
            intent["persons"].append(name)
    for role, name in _lex["roles"].items():
        if role in question and name:
            if name not in intent["persons"]:
                intent["persons"].append(name)
    # 一家/全家
    if "一家人" in question or "全家" in question:
        intent["persons"] = [n for n in _lex["family"]
                             if n not in intent["persons"]] or intent["persons"]
    # 孩子
    if "孩子" in question and "一家人" not in question and "5岁" in question:
        # 找XX的孩子5岁前
        for role, rtype in (("爸爸", "father_of"), ("妈妈", "mother_of")):
            name = _lex["roles"].get(role)
            if name and name in question:
                intent["relationship"] = {"from": name, "type": rtype}
                intent["age_constraint"] = {"person": "孩子", "max_age": 5}
        if "relationship" not in intent and "5岁" in question:
            intent["age_constraint"] = {"person": "孩子", "max_age": 5}
    elif "孩子" in question and "孩子" not in intent["persons"]:
        intent["persons"].append("孩子")
    # 几岁
    if "几岁" in question:
        for name in _lex["family"]:
            if name in question:
                intent["age_constraint"] = {"person": name}
    # 年龄约束泛化（2026-09-17）：「某人2岁(时/的时候)」「2岁前/以前」「2岁以后」。
    # 此前只有硬编码「孩子5岁」，「N岁时」完全解析不出 → 返回全部年龄的照片。
    # 语义：N岁时 → [N, N+1) 周岁；N岁前 → <N；N岁后 → >=N；N岁半 → ±半年窗口。
    if "age_constraint" not in intent and re.search(r"(\d+|[一二两三四五六七八九十])岁", question):
        _cn = {"一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
               "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
        _ma = re.search(r"(\d+|[一二两三四五六七八九十])(岁半|岁)", question)
        _age_val = _ma.group(1)
        _age = float(_cn.get(_age_val, _age_val))
        _half = _ma.group(2) == "岁半"
        _tail = question[_ma.end():_ma.end() + 4]
        _min_age = _max_age = None
        if _tail.startswith(("以前", "之前", "前")):
            _max_age = _age
        elif _tail.startswith(("以后", "之后", "后")):
            _min_age = _age
        elif _half:
            _min_age, _max_age = _age - 0.5, _age + 0.5
        else:
            _min_age, _max_age = _age, _age + 1.0
        # 人物定位：问句里出现的已知人物 > 孩子 > 年龄词前 2-4 字猜名
        _age_person = None
        for _p in intent.get("persons") or []:
            if _p in question:
                _age_person = _p
                break
        if not _age_person and "孩子" in question:
            _age_person = "孩子"
        if not _age_person:
            _pre = question[:_ma.start()]
            _m_pre = re.search(r"[\u4e00-\u9fa5]{2,4}$", _pre)
            if _m_pre and _m_pre.group(0) in _lex.get("family", set()):
                _age_person = _m_pre.group(0)
        if _age_person:
            intent["age_constraint"] = {"person": _age_person,
                                        "min_age": _min_age, "max_age": _max_age}
    # 时间表达（2026-09-04）：去年/今年/前年/2024年/最近N天——
    # 此前完全不解析，"去年拍的视频"会无视时间约束返回最旧的视频。
    _now = date.today()
    if "前年" in question:
        _y = _now.year - 2
        intent["time"] = {"start": f"{_y}-01-01", "end": f"{_y}-12-31", "year": _y}
    elif "去年" in question:
        _y = _now.year - 1
        intent["time"] = {"start": f"{_y}-01-01", "end": f"{_y}-12-31", "year": _y}
    elif "今年" in question:
        _y = _now.year
        intent["time"] = {"start": f"{_y}-01-01", "end": f"{_y}-12-31", "year": _y}
    else:
        _m = re.search(r"(20\d{2})\s*年", question)
        if _m:
            _y = int(_m.group(1))
            intent["time"] = {"start": f"{_y}-01-01", "end": f"{_y}-12-31", "year": _y}
    if "最近" in question:
        _m2 = re.search(r"最近\s*(\d+)\s*(天|日|周|星期|个月)", question)
        _days = 90
        if _m2:
            _n, _unit = int(_m2.group(1)), _m2.group(2)
            _days = _n * (30 if _unit == "个月" else 7 if _unit in ("周", "星期") else 1)
        _t0 = intent.get("time") or {}
        intent["time"] = {"start": (_now - timedelta(days=_days)).isoformat(),
                          "end": _now.isoformat(), "year": _t0.get("year")}
    # 记忆关键词
    if "第一次" in question:
        intent["memory_keywords"] = [kw for kw in ["赛里木湖", "新疆"] if kw in question]
    has_video_word = "视频" in question or "录像" in question
    has_photo_word = "照片" in question or "图片" in question
    if has_video_word and has_photo_word:
        intent["media_type"] = "any"
    elif has_video_word:
        intent["media_type"] = "video"
    elif has_photo_word:
        intent["media_type"] = "photo"
    if any(k in question for k in ("截图", "聊天记录", "下载图片")) and "排除" in question:
        intent["exclude"] = ["screenshot", "download", "forwarded"]
        intent["media_type"] = "any"
    if "旅行" in question and any(k in question for k in ("两部手机", "不同手机", "两个设备", "跨设备")):
        intent["same_event_cross_source"] = True
    # 多设备互证：问的是「多个设备/多个人同时拍下的时刻」
    # 注意: 「同一次旅行里两部手机」已走事件级 same_event_cross_source(照片+视频都召回),
    # 不再叠加同刻互证过滤(互证关系几乎全是照片, 会把视频全滤掉)
    if not intent.get("same_event_cross_source") and any(
            k in question for k in ("多设备", "互证", "同拍", "同时拍", "两台设备", "多台设备",
                                    "不同设备", "两部手机", "不同手机", "两个设备", "跨设备")):
        intent["cross_device"] = True
    if any(k in question for k in ("开心", "大笑", "笑得很开心")):
        intent["emotion"] = "happy"
    # 物品词：最长匹配优先（2026-09-17 修复：「婴儿车」先被「车」截胡 → object=车 误召回）。
    # 含子串关系的词（车/婴儿车/推车、船/游船）必须长的先比。
    _obj_words = ("杯子", "礼盒", "蛋糕", "汽车", "室内装修", "装修", "下雨", "雨伞", "玩具",
                  "书包", "背包", "行李箱", "自行车", "单车", "高铁", "火车", "动车", "飞机", "机场",
                  "轮船", "游船", "婴儿车", "推车", "气球", "风筝", "帐篷", "露营", "烧烤",
                  "火锅", "灯笼", "春联", "红包", "滑雪", "雪橇", "乐器", "吉他", "钢琴",
                  "船", "车", "花")
    for obj in sorted(_obj_words, key=len, reverse=True):
        if obj in question:
            intent["object"] = obj
            break
    return intent


# ============ 负面信号解析（独立函数，#4 2026-09-04） ============
_NEG_WORDS = ("没有", "不是", "不要", "不包含", "排除", "别", "无")
_NEG_SCENE_TAGS = {"山景", "海景", "美食", "花草", "夜景", "雪景", "建筑",
                   "江河", "宠物", "儿童", "生日", "车辆", "室内", "合影", "截图", "节日"}
# 中文口语单字 → 标准 scene_tag（如"没有雪" → 排除"雪景"）
_NEG_TAG_ALIASES = {"雪": "雪景", "山": "山景", "海": "海景", "花": "花草",
                    "海景": "海景", "夜": "夜景"}


def _extract_neg_scene(question):
    """从用户查询中解析否定场景词，返回 list[tag]（如 ["雪景"]）。

    "没有雪"/"不是海边"/"不包含截图" → 排除对应 tag。
    解析策略：找否定词"没有/不是/不要/不包含/排除/无"后接 1~3 个字，
    用 _SCENE_ALIAS / _PHRASE_LEXICON / _NEG_TAG_ALIASES 试探回退到合理 tag。
    """
    import re as _re
    q = question.replace("的照片", "").replace("的图片", "").replace("的影像", "")
    neg_scene = []
    for neg in _NEG_WORDS:
        for m in _re.finditer(_re.escape(neg), q):
            tail = q[m.end():m.end() + 3]
            for L in (3, 2, 1):
                if len(tail) < L:
                    continue
                w = tail[:L]
                tag = (_SCENE_ALIAS.get(w) or _PHRASE_LEXICON.get(w) or
                       _NEG_TAG_ALIASES.get(w) or
                       (w if w in _NEG_SCENE_TAGS else None))
                if tag and tag not in neg_scene:
                    neg_scene.append(tag)
                    break
    return neg_scene


# ============ 检索 ============
def get_person_id(con, name):
    if name == "爷爷":
        r = con.execute(
            """SELECT person_id FROM person
               WHERE identity_status='confirmed'
                 AND (display_name='爷爷' OR relationship_label='爷爷') LIMIT 1"""
        ).fetchone()
    else:
        r = con.execute("SELECT person_id FROM person WHERE display_name=?", (name,)).fetchone()
    return r["person_id"] if r else None


def assets_with_person(con, person_id):
    return set(r["asset_id"] for r in con.execute(
        "SELECT DISTINCT asset_id FROM face_instance_v0 WHERE person_id=?", (person_id,)))


def _age_on_date(birth_text, shot):
    """返回拍摄日的完整岁数和月数；不使用约30天换算。"""
    birth = date.fromisoformat(birth_text)
    months = (shot.year - birth.year) * 12 + shot.month - birth.month
    if shot.day < birth.day:
        months -= 1
    months = max(months, 0)
    return divmod(months, 12)


def memory_graph_composition(con, question):
    """确定性完成 Memory -> Event -> Relationship -> Dynamic Age 组合查询。

    只在问题同时询问某条已存 Memory 的时间/年龄时触发。人物是否参与 Event
    以 event_asset 上的已确认 face_instance 为证据；关系图本身不冒充参与证据。
    """
    if not any(k in question for k in ("几岁", "什么时候", "哪年")):
        return None

    memories = [dict(r) for r in con.execute(
        "SELECT * FROM memory_note WHERE event_id IS NOT NULL ORDER BY created_at"
    )]
    matched = None
    for memory in memories:
        raw_tokens = re.split(r"[，,、；;\s]+", memory.get("location_text") or "")
        tokens = [t.strip() for t in raw_tokens if len(t.strip()) >= 2]
        if any(t in question for t in tokens):
            matched = memory
            break
    if not matched:
        return None

    event = con.execute(
        "SELECT * FROM event WHERE event_id=?", (matched["event_id"],)
    ).fetchone()
    if not event or not event["start_time"]:
        return None
    shot = date.fromisoformat(event["start_time"][:10])

    rows = con.execute(
        """SELECT ma.asset_id, ma.capture_time, ma.media_type
           FROM event_asset ea JOIN media_asset ma USING(asset_id)
           WHERE ea.event_id=? ORDER BY ea.sequence_no, ma.capture_time""",
        (event["event_id"],),
    ).fetchall()
    # 结果中同时保留照片和视频，避免视频因 sequence_no 靠后而永远进不了 Top 50。
    photos = [r for r in rows if r["media_type"] == "photo"][:40]
    videos = [r for r in rows if r["media_type"] == "video"][:10]
    selected_rows = sorted(photos + videos, key=lambda r: r["capture_time"] or event["start_time"])
    raw_assets = [(r["asset_id"], 1.0, r["capture_time"] or event["start_time"]) for r in selected_rows]
    assets = dedup_burst(con, raw_assets)[:50]
    media_types = {r["asset_id"]: r["media_type"] for r in rows}
    for asset in assets:
        asset["type"] = media_types.get(asset["id"], "photo")

    # 关系图找“孩子”，再用 Event 内已确认人脸记录参与证据。
    children = con.execute(
        """SELECT DISTINCT child.person_id, child.display_name, child.birth_date
           FROM person_relation_v0 pr
           JOIN person child ON child.person_id=pr.to_person_id
           WHERE pr.relation_type IN ('father_of','mother_of')
             AND child.birth_date IS NOT NULL
           ORDER BY child.birth_date"""
    ).fetchall()
    age_parts, evidenced, not_evidenced = [], [], []
    for child in children:
        years, months = _age_on_date(child["birth_date"], shot)
        age_parts.append(f"{child['display_name']} {years}岁{months}个月")
        face_count = con.execute(
            """SELECT count(DISTINCT fi.asset_id)
               FROM event_asset ea JOIN face_instance_v0 fi USING(asset_id)
               WHERE ea.event_id=? AND fi.person_id=?""",
            (event["event_id"], child["person_id"]),
        ).fetchone()[0]
        (evidenced if face_count else not_evidenced).append(child["display_name"])

    answer = f"这次记忆发生在 {shot.isoformat()}。"
    if "几岁" in question and age_parts:
        answer += " 按生日计算，当时" + "，".join(age_parts) + "。"
        if evidenced:
            answer += " 当前影像中已确认出现：" + "、".join(evidenced) + "。"
        if not_evidenced:
            answer += " 尚未在该事件的人脸结果中确认：" + "、".join(not_evidenced) + "。"
    if matched.get("text"):
        answer += f" 家庭口述记忆：「{matched['text']}」"
    return {
        "answer": answer,
        "assets": assets,
        "memory": [matched],
        "persons": [r["display_name"] for r in children],
        "intent": {"memory_graph_composition": True, "event_id": event["event_id"]},
    }


def fts_day_recall(con, keywords):
    """按天内容描述召回：关键词命中天描述 → 返回该天全部资产。
    ≥3 字走 trigram FTS；<3 字（火锅/灯笼等双字词构不成 trigram）直接
    LIKE day_caption_v0（仅 625 行, 毫秒级）。"""
    matched_days = set()
    for kw in keywords or []:
        kw = (kw or "").strip()
        if not kw:
            continue
        if len(kw) >= 3:
            try:
                for r in con.execute(
                    """SELECT subject_id FROM memory_search
                       WHERE memory_search MATCH ? AND subject_type='day'""", (kw,)):
                    matched_days.add(r["subject_id"])
            except Exception:
                pass
        else:
            for r in con.execute(
                "SELECT day FROM day_caption_v0 WHERE caption_local LIKE ?", (f"%{kw}%",)):
                matched_days.add(r["day"])
    if not matched_days:
        return set()
    ph = ",".join("?" * len(matched_days))
    return {r["asset_id"] for r in con.execute(
        f"""SELECT asset_id FROM media_asset
            WHERE substr(capture_time,1,10) IN ({ph})""", tuple(matched_days))}


def llm_synthesize_answer(question, result):
    """LLM 综合检索证据 → 自然语言回答（问答的 LLM 层，本地 Ollama 优先）。
    只允许基于给出的证据回答，证据不足要明说；失败返回 None，保留模板回答。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    evidence = top_day_captions(con, result.get("assets") or [], limit=8)
    con.close()
    if not evidence:
        return None
    # 多设备互证证据：哪些天有多个来源/设备同时记录
    try:
        con2 = sqlite3.connect(DB, timeout=10)
        con2.row_factory = sqlite3.Row
        evidence += cross_device_evidence(con2, result.get("assets") or [], limit=3)
        con2.close()
    except Exception:
        pass
    n_assets = len(result.get("assets") or [])
    persons = "、".join(result.get("persons") or []) or "无"
    mem = [m.get("text") for m in (result.get("memory") or []) if m.get("text")][:3]
    # 2026-09-04：把命中的时间跨度也作为证据给模型——原来只给 5 条天描述，
    # qwen2.5vl 文本能力弱，描述里没提到的人物它就敢说"无法确定"，
    # 与检索到 50 张的事实自相矛盾。
    _times = [a.get("time") for a in (result.get("assets") or [])
              if a.get("time") and a.get("time")[:4] not in ("", "0000")]
    _span = f"，时间跨度 {_times[-1][:10]} ~ {_times[0][:10]}" if _times else ""
    prompt = (
        "你在回答关于一个家庭相册的问题。检索系统已从数据库确认了 "
        f"{n_assets} 项匹配的影像（人脸识别/场景/时间条件均已过滤验证），"
        f"涉及人物：{persons}{_span}。\n\n"
        "下面是其中部分照片所在天的内容描述（证据）：\n"
        + "\n".join(f"- {e}" for e in evidence)
        + (f"\n记忆备注：{'; '.join(mem)}" if mem else "")
        + f"\n\n问题：{question}\n"
        + "请用中文回答，60-150 字，概括找到了什么（时间、人物、场景）。"
          "注意：检索结果已经过数据库验证，严禁回答\"无法确定是否存在\"或否定检索结果；"
          "证据描述未提及的细节可以略过，但不得据此否认照片存在。直接给出回答内容。"
    )
    text, _err = llm_chat(
        [{"role": "user", "content": prompt}],
        max_tokens=1024)
    return text or None


def search(question):
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    # 排除集合 = 回收站 + 隐私相册（搜索结果一律不可见）
    recycled_ids = get_recycled_ids(con)
    try:
        recycled_ids |= {r[0] for r in con.execute("SELECT asset_id FROM privacy_v0")}
    except sqlite3.OperationalError:
        pass
    # “爷爷”尚未绑定到任何已确认 Person。关系事实缺失时必须阻断，
    # 不能退化为只查孩子，否则会制造看似有答案的错误记忆。
    if "爷爷" in question:
        grandfather = con.execute(
            """SELECT p.person_id FROM person p
               WHERE identity_status='confirmed'
                 AND (display_name='爷爷' OR relationship_label='爷爷')
                 AND EXISTS (SELECT 1 FROM face_instance_v0 fi WHERE fi.person_id=p.person_id)
               LIMIT 1"""
        ).fetchone()
        if grandfather is None:
            con.close()
            return {
                "answer": "这条查询暂时不能可靠执行：家庭人物库里还没有确认“爷爷”对应哪位 Person。系统没有用“老人”外观猜身份，也没有返回只有孩子的错误结果。",
                "assets": [], "memory": [], "persons": (_person_lexicon()["children"][-1:] or []) + ["爷爷"],
                "intent": {"blocked": True, "blocked_reason": "grandfather_identity_unconfirmed"},
            }
    composition = memory_graph_composition(con, question)
    if composition:
        composition["assets"] = exclude_recycled(con, composition.get("assets") or [])
        con.close()
        return composition
    intent = parse_intent(question)

    # ===== 空意图守卫 =====
    # 本地规则和云端 LLM 都没能理解查询时，intent 不含任何约束。
    # 此时绝不能退回"返回全库前 50 项"——那会制造"找到 50 项"的假命中，
    # 用户看到的会是完全不相关的照片。诚实返回未理解。
    # 2026-09-06：移除 `media_type in (None, "any")` 这条——"的照片"被解析成
    # media_type="photo" + 空 events/p/time 会绕开 blocked 直接走全库前 50，
    # 制造"找到 50 项"的虚假命中（用户问大理却看到新疆照片回 50 张）。
    # media_type 不是搜索意图，只是结果过滤维度，不应替代事件/人物/地点约束。
    t = intent.get("time") or {}
    intent_is_empty = (
        not intent.get("persons")
        and not intent.get("events")
        and not intent.get("memory_keywords")
        and not intent.get("object")
        and not intent.get("exclude")
        and not intent.get("exclude_scene")
        and not intent.get("emotion")
        and not intent.get("relationship")
        and not intent.get("age_constraint")
        and not intent.get("same_event_cross_source")
        and not intent.get("cross_device")
        and not (t.get("start") or t.get("end") or t.get("year"))
    )
    if intent_is_empty:
        con.close()
        return {
            "answer": "没有理解这条查询，没有返回任何照片。换个说法试试，例如：人物（家人姓名）、地点（山/海边）、事件（赛里木湖/旅行）、物品（杯子/礼盒）或状态（开心）。",
            "assets": [], "memory": [], "persons": [],
            "intent": {"blocked": True, "blocked_reason": "unparsed_query"},
        }

    # ===== 物品查询分支（杯子/礼盒/装修等无人物无地点的查询） =====
    obj = intent.get("object") or ""
    has_person = bool(intent.get("persons"))
    has_event = bool(intent.get("events")) or bool(intent.get("memory_keywords"))
    if obj and not has_person and not has_event:
        try:
            result = object_search(con, obj, question)
            result["assets"] = exclude_recycled(con, result.get("assets") or [])
            con.close()
            return result
        except Exception as e:
            print(f"[object] 检索失败: {e}")

    persons = list(intent.get("persons", []))
    if "爷爷" in question and "爷爷" not in persons:
        persons.append("爷爷")
    # "我们/大家" 语境下的全家不应做多人交集 → 仅当查询明确"一家人/全家"才展开
    _lex_now = _person_lexicon()
    _kids = _lex_now["children"]
    _kid_set = set(_kids)
    if "一家人" in question or "全家" in question:
        persons = list(_lex_now["family"]) or persons
    # 展开"孩子"：孩子们是"二选一"语义（union），不是同框交集。
    # 2026-09-04 修复："孩子的照片"原实现走多人物交集，只返回娃们同框
    # 的照片，单独一个孩子的照片全部被错误排除。
    kids_union = False
    expanded = []
    for p in persons:
        if p == "孩子":
            kids_union = True
            for kid in _kids:
                if kid not in expanded:
                    expanded.append(kid)
        else:
            expanded.append(p)
    persons = expanded

    # Person 过滤
    asset_sets = []
    if kids_union:
        # "孩子" = 任一孩子出现即可；显式点名的人物（如"老大和孩子"）
        # 再单独成集合求交集。
        kid_set = set()
        for kid in _kids:
            pid = get_person_id(con, kid)
            if pid:
                kid_set |= assets_with_person(con, pid)
        if kid_set:
            asset_sets.append(kid_set)
        for name in persons:
            if name in _kids:
                continue
            pid = get_person_id(con, name)
            if pid:
                asset_sets.append(assets_with_person(con, pid))
    else:
        for name in persons:
            pid = get_person_id(con, name)
            if pid:
                asset_sets.append(assets_with_person(con, pid))
    # "一家人/全家" = 至少 2 个家庭成员出现（不需要全部同框）
    is_family_query = any(k in question for k in ["一家人", "全家"])
    if asset_sets:
        if is_family_query and len(asset_sets) >= 2:
            # 至少2个家庭成员的资产
            from collections import Counter
            cnt = Counter()
            for s in asset_sets:
                for aid in s:
                    cnt[aid] += 1
            candidate = {aid for aid, c in cnt.items() if c >= 2}
        elif len(asset_sets) >= 2:
            # 多个人名 = 同框要求（交集）
            candidate = set.intersection(*asset_sets)
        else:
            # 单个人名 = 出现即可
            candidate = asset_sets[0]
    else:
        # 全库（不限于 cohort）
        candidate = set(r["asset_id"] for r in con.execute(
            "SELECT asset_id FROM media_asset"))

    # 关系：父亲/母亲的孩子
    rel = intent.get("relationship")
    if rel and rel.get("type") in ("father_of", "mother_of"):
        fname = rel.get("from", "")
        fpid = get_person_id(con, fname)
        if fpid:
            children = [r["display_name"] for r in con.execute(
                """SELECT t.display_name FROM person_relation_v0 pr
                   JOIN person t ON t.person_id=pr.to_person_id
                   WHERE pr.from_person_id=? AND pr.relation_type=?""", (fpid, rel["type"]))]
            child_assets = set()
            for c in children:
                cpid = get_person_id(con, c)
                if cpid:
                    child_assets |= assets_with_person(con, cpid)
            candidate &= child_assets

    # Event / Memory 关键词
    memory_keywords = intent.get("memory_keywords", []) + intent.get("events", [])
    memory_hits = []
    if memory_keywords:
        for kw in memory_keywords:
            for r in con.execute("SELECT * FROM memory_note WHERE text LIKE ?", (f"%{kw}%",)):
                memory_hits.append(dict(r))
        # Memory → Event 资产
        if memory_hits:
            event_ids = [m["event_id"] for m in memory_hits if m.get("event_id")]
            if event_ids:
                placeholders = ",".join("?" * len(event_ids))
                ev_assets = set(r["asset_id"] for r in con.execute(
                    f"SELECT DISTINCT asset_id FROM event_asset WHERE event_id IN ({placeholders})", event_ids))
                if memory_keywords and not persons:
                    candidate = ev_assets
                else:
                    candidate &= ev_assets

    # 同一旅行、跨来源设备：只召回至少有两个来源成员共同记录的 travel Event。
    if intent.get("same_event_cross_source"):
        cross_assets = set(r["asset_id"] for r in con.execute(
            """SELECT DISTINCT ea.asset_id FROM event_asset ea JOIN event e USING(event_id)
               WHERE e.event_type='travel' AND e.status IN ('candidate','confirmed')
                 AND e.event_id IN (
                   SELECT e2.event_id FROM event e2 JOIN event_asset ea2 USING(event_id)
                   JOIN media_file mf2 USING(asset_id) JOIN source s2 USING(source_id)
                   WHERE e2.event_type='travel' AND e2.status IN ('candidate','confirmed')
                   GROUP BY e2.event_id HAVING count(DISTINCT s2.owner_label)>=2)"""
        ))
        candidate &= cross_assets

    # 多设备互证：只保留参与跨设备关系的资产。
    # same_moment = 两台设备各自按了快门（真互证）；
    # asset_id 下挂多个来源的物理文件 = 跨设备互传的同一照片；
    # 跨资产 exact_duplicate（索引器未合并的）作为兜底。
    if intent.get("cross_device"):
        target = set(r["a"] for r in con.execute(
            """SELECT left_asset_id AS a FROM asset_relation WHERE relation_type='same_moment'
                  OR (relation_type='exact_duplicate' AND evidence_json LIKE '%"cross_source":true%')
               UNION
               SELECT right_asset_id AS a FROM asset_relation WHERE relation_type='same_moment'
                  OR (relation_type='exact_duplicate' AND evidence_json LIKE '%"cross_source":true%')
               UNION
               SELECT asset_id AS a FROM media_file GROUP BY asset_id HAVING COUNT(DISTINCT source_id) > 1"""))
        # 2026-09-02 修复: 原逻辑 `candidate = (candidate & target) if (candidate - target) else target`
        # 在 candidate ⊆ target 时返回整个 target, 会把被排除条件(如排除截图)挡掉的资产拉回来;
        # 且候选为空时也错误放宽成全量 target。统一按注释语义收紧为交集。
        candidate &= target

    # 物品标签过滤（组合查询: 如「新疆的火锅照片」「孩子的玩具照片」——
    # 物品条件不再因带人物/地点而被丢弃, 与其他维度取交集）
    if obj:
        otag = object_tag_for_word(obj)
        if otag:
            tagged = set(r["asset_id"] for r in con.execute(
                "SELECT asset_id FROM scene_tag_v0 WHERE tag=?", (otag,)))
            candidate &= tagged

    # 时间过滤
    t = intent.get("time") or {}
    # 2026-09-04：LLM 解析常只回 year（start/end 为空），而下方过滤只认
    # start/end——year 单独出现时展开为整年区间，否则时间条件被静默丢弃。
    if t.get("year") and not (t.get("start") or t.get("end")):
        try:
            _y = int(t["year"])
            t = {"start": f"{_y}-01-01", "end": f"{_y}-12-31", "year": _y}
            intent["time"] = t
        except Exception:
            pass
    if t.get("start") or t.get("end"):
        # 用 media_asset.capture_time
        pass  # 简化：时间约束在排序时处理

    # 媒体类型
    mtype = intent.get("media_type")
    if mtype and mtype != "any":
        keep = set(r["asset_id"] for r in con.execute(
            "SELECT asset_id FROM media_asset WHERE media_type=?", (mtype,)))
        candidate &= keep

    # “特别开心”采用通过本地分层盲检的高精度阈值。只判断可见表情，
    # 不把模型输出表述为人物的真实内心情绪。
    if intent.get("emotion") == "happy":
        candidate = emotion_filter(con, candidate, persons, threshold=0.90)

    # 负样本排除使用独立事实层，不读取评测 cohort 的 positive/negative 标签。
    if intent.get("exclude"):
        try:
            excluded = get_hidden_ids(con)
        except sqlite3.OperationalError:
            excluded = set()
        candidate -= excluded

    # ===== 负面信号（#4, 2026-09-04）=====
    # 用户说"没有雪"/"不是海边"等：硬减掉命中对应 tag 的资产，
    # 让 evidence_weight 排序不会把它们送进结果。无法 100% 准确
    # （VLM 漏标的雪景会漏过），但对已正确打 tag 的资产是有效信号。
    neg_tags = intent.get("exclude_scene") or []
    if neg_tags:
        ph = ",".join("?" * len(neg_tags))
        neg_assets = set(r["asset_id"] for r in con.execute(
            f"SELECT DISTINCT asset_id FROM scene_tag_v0 WHERE tag IN ({ph})", neg_tags))
        if neg_assets:
            candidate -= neg_assets

    # 地点过滤：事件关键词匹配地理特征（新疆/赛里木湖/山等）
    geo_keywords = [k for k in (intent.get("events") or []) if k not in ("旅行",)]
    geo_assets = set()
    if geo_keywords:
        for kw in geo_keywords:
            # 特殊语义：山 → 山区(is_mountainous)，海边 → 沿海(is_coastal)
            if kw == "山":
                for r in con.execute("SELECT asset_id FROM asset_geo_v0 WHERE is_mountainous=1"):
                    geo_assets.add(r["asset_id"])
                continue
            if kw in ("海边", "海"):
                for r in con.execute("SELECT asset_id FROM asset_geo_v0 WHERE is_coastal=1"):
                    geo_assets.add(r["asset_id"])
                continue
            # 普通地点：匹配 region/province
            for r in con.execute("SELECT asset_id FROM asset_geo_v0 WHERE region LIKE ? OR province LIKE ?",
                                 (f"%{kw}%", f"%{kw}%")):
                geo_assets.add(r["asset_id"])
        if geo_assets:
            candidate &= geo_assets
        else:
            # 地理表没匹配到，尝试 Event title 匹配
            ev_ids = [r["event_id"] for r in con.execute(
                "SELECT event_id FROM event WHERE title LIKE ?", (f"%{geo_keywords[0]}%",))]
            if ev_ids:
                ph = ",".join("?" * len(ev_ids))
                ev_assets = set(r["asset_id"] for r in con.execute(
                    f"SELECT DISTINCT asset_id FROM event_asset WHERE event_id IN ({ph})", ev_ids))
                candidate &= ev_assets
            else:
                # 第三兜底：按天内容描述 FTS 召回（trigram 中文子串可检索）。
                # 地理表/事件标题都没命中的地点词（赛里木湖、具体景点名等），
                # 天描述（本地综合含地区，VLM 叙事含地标细节）里可能提到。
                day_assets = fts_day_recall(con, geo_keywords)
                if day_assets:
                    candidate &= day_assets
                else:
                    # 场景标签兜底（2026-09-01）：VLM/SIGLIP 写入 scene_tag_v0 的
                    # 场景词（雪景/美食/生日/海景/夜景…）不在地理表里，此前直接
                    # 返回空。这里用标签子串 + 口语同义词命中场景标签。
                    tag_assets = set()
                    SYNONYM = {"海边": "海景", "沙滩": "海景", "大海": "海景", "海": "海景",
                               "下雪": "雪景", "吃饭": "美食", "蛋糕": "美食",
                               "花": "花草", "城市": "建筑", "房子": "建筑", "猫狗": "宠物"}
                    for kw in geo_keywords:
                        for r in con.execute(
                                "SELECT DISTINCT asset_id FROM scene_tag_v0 WHERE tag LIKE ?",
                                (f"%{kw}%",)):
                            tag_assets.add(r["asset_id"])
                        # 2026-09-06 改名：原 `t = SYNONYM.get(kw)` 会覆盖上面
                        # `# 时间过滤` 块里的 t（intent.get("time") or {}），
                        # 后续 _TIME_FALLBACK 段 `if t.get("start") or t.get("end")`
                        # 在大理这种 events=["大理"] 的空时间查询上直接 AttributeError 崩。
                        # 真实踩中：用户问"大理的照片" → search 抛 'NoneType' has no 'get'。
                        syn_tag = SYNONYM.get(kw)
                        if syn_tag:
                            for r in con.execute(
                                    "SELECT DISTINCT asset_id FROM scene_tag_v0 WHERE tag=?", (syn_tag,)):
                                tag_assets.add(r["asset_id"])
                    if tag_assets:
                        candidate &= tag_assets
                    else:
                        # 真的全没命中：诚实返回空，而不是返回全库
                        candidate = set()

    # 年龄约束
    age_c = intent.get("age_constraint") or {}
    _has_age_bound = age_c.get("max_age") is not None or age_c.get("min_age") is not None
    if age_c.get("person") and _has_age_bound:
        # "孩子"展开为词典中的孩子们（#16：动态取自 person 表）
        age_persons = _kids if age_c["person"] == "孩子" else [age_c["person"]]
        max_age = age_c.get("max_age")
        min_age = age_c.get("min_age")
        keep = set()
        for ap in age_persons:
            pid = get_person_id(con, ap)
            if not pid:
                continue
            bd = con.execute("SELECT birth_date FROM person WHERE person_id=?", (pid,)).fetchone()
            if not bd or not bd["birth_date"]:
                continue
            by, bm, bd_ = map(int, bd["birth_date"].split("-"))
            birth = date(by, bm, bd_)
            # 周岁窗口按天数折算（365.25/年，兼容 2/29 生日，2026-09-17）
            cut_max = date.toordinal(birth) + int(365.25 * max_age) if max_age is not None else None
            cut_min = date.toordinal(birth) + int(365.25 * min_age) if min_age is not None else None
            for r in con.execute(
                """SELECT DISTINCT fi.asset_id, ma.capture_time FROM face_instance_v0 fi
                   JOIN media_asset ma USING(asset_id)
                   WHERE fi.person_id=? AND ma.capture_time IS NOT NULL""", (pid,)):
                ct = r["capture_time"][:10]
                try:
                    y, m, d = map(int, ct.split("-"))
                    ord_ct = date(y, m, d).toordinal()
                    if (cut_max is None or ord_ct < cut_max) and (cut_min is None or ord_ct >= cut_min):
                        keep.add(r["asset_id"])
                except Exception:
                    pass
        candidate &= keep

    # 场景过滤：只有泛化场景词（山/海边）才走 CLIP 视觉过滤。
    # 具体地名（武夷山/庐山/三清山等）已由 asset_geo_v0 精确命中（GPS/同日互证），
    # 是强证据；SIGLIP 视觉判断不得否决地名结果（实测 127 张武夷山照片被误杀 126 张）。
    scene_kw = None
    for k in geo_keywords:
        if k == "山":
            scene_kw = "山"
            break
        if k in ("海边", "海"):
            scene_kw = "海边"
            break
    if scene_kw and candidate:
        # protect：GPS 场景区 + 已确认家人人脸同框 = 强证据，语义场景模型不得否决。
        # 修复：家人山地/海边合影会被 SIGLIP 负类"人物近景"整类误杀（实测 80/80 全灭）。
        protect = set()
        if geo_assets:
            for r in con.execute(
                """SELECT DISTINCT fi.asset_id FROM face_instance_v0 fi
                   JOIN person p ON p.person_id = fi.person_id
                   WHERE p.identity_status='confirmed'"""):
                if r["asset_id"] in geo_assets:
                    protect.add(r["asset_id"])
        try:
            candidate = scene_filter(con, candidate, scene_kw, protect=protect)
        except Exception as e:
            print(f"[scene] 过滤失败: {e}")

    # ===== 排序：维度加权打分 (2026-09-04 evidence_weight 改造) =====
    if recycled_ids:
        candidate -= recycled_ids

    # 一次拉所有 candidate 的多源信号，避免 N 次单点 SQL
    _s_aids = list(candidate)
    W_PERSON, W_GEO, W_SCENE, W_MEMORY, W_OBJECT = 1.0, 0.7, 0.3, 1.0, 0.4

    # (A) 人脸: 候选集涉及的 person display_name
    _asset_persons = {}
    if _s_aids:
        _ph_a = ",".join("?" * len(_s_aids))
        for _r in con.execute(
                f"SELECT fi.asset_id, p.display_name FROM face_instance_v0 fi "
                f"JOIN person p ON p.person_id=fi.person_id "
                f"WHERE fi.asset_id IN ({_ph_a})", _s_aids):
            _asset_persons.setdefault(_r["asset_id"], set()).add(_r["display_name"])

    # (B) scene_tag 命中：events + object 映射到 tag
    # 2026-09-17: 有专属物品标签的词一律映射到同名标签（原来到处兜到「车辆/日常物品」，
    # 「婴儿车」搜出 430 张汽车照）。映射目标必须真实存在于 scene_tag_v0 词表。
    _OBJ2TAG = {"杯子":"日常物品","礼盒":"礼盒","蛋糕":"美食","汽车":"车辆","车":"车辆",
                "室内装修":"室内","装修":"室内","下雨":"日常物品","雨伞":"日常物品",
                "玩具":"玩具","书包":"书包","背包":"书包","行李箱":"行李箱",
                "自行车":"自行车","单车":"自行车","高铁":"火车","火车":"火车","动车":"火车",
                "飞机":"飞机","机场":"飞机","船":"游船","轮船":"游船","游船":"游船",
                "婴儿车":"婴儿车","推车":"婴儿车","气球":"气球","风筝":"风筝",
                "帐篷":"帐篷","露营":"帐篷","烧烤":"烧烤","火锅":"火锅",
                "灯笼":"灯笼","春联":"灯笼","红包":"灯笼","滑雪":"滑雪","雪橇":"滑雪",
                "吉他":"乐器","钢琴":"乐器","乐器":"乐器","花":"花草"}
    _want_tags = set(intent.get("events") or [])
    _obj = intent.get("object")
    if _obj:
        _want_tags.add(_OBJ2TAG.get(_obj, _obj))
    _asset_scene_match = {}
    if _s_aids and _want_tags:
        _ph_a = ",".join("?" * len(_s_aids))
        _ph_t = ",".join("?" * len(_want_tags))
        for _r in con.execute(
                f"SELECT st.asset_id, st.tag FROM scene_tag_v0 st "
                f"WHERE st.asset_id IN ({_ph_a}) AND st.tag IN ({_ph_t})",
                list(_s_aids) + list(_want_tags)):
            _asset_scene_match.setdefault(_r["asset_id"], set()).add(_r["tag"])

    # (C) memory event 命中
    _mem_ev_ids = [m["event_id"] for m in memory_hits if m.get("event_id")]
    _asset_in_mem = set()
    if _s_aids and _mem_ev_ids:
        _ph_a = ",".join("?" * len(_s_aids))
        _ph_e = ",".join("?" * len(_mem_ev_ids))
        for _r in con.execute(
                f"SELECT DISTINCT ea.asset_id FROM event_asset ea "
                f"WHERE ea.asset_id IN ({_ph_a}) AND ea.event_id IN ({_ph_e})",
                list(_s_aids) + list(_mem_ev_ids)):
            _asset_in_mem.add(_r["asset_id"])

    _persons_set = set(persons)
    _is_family = any(k in question for k in ("一家人", "全家"))
    _want_tags_count = len(_want_tags) or 1

    scored = []
    for aid in _s_aids:
        s = 0.0
        # 维度1: 人脸命中
        #   - 一家人/全家：≥2 人同框即满分（用户只关心"家庭成员有露面"，不强求全部）
        #   - 单点人名：期望人出现在图片里给 1.0（少量同框会按比例加少量）
        _ps = _asset_persons.get(aid, set())
        if _persons_set:
            _matched = _ps & _persons_set
            if _matched:
                if _is_family:
                    # 至少 2 人才算"一家人同框"，≥2 给满分
                    s += W_PERSON if len(_matched) >= 2 else 0.3
                else:
                    # 单点人名：全命中得满分，部分命中按人数比例
                    s += W_PERSON * (len(_matched) / len(_persons_set))
        # 维度2: 地理命中 (geo_assets 已在前面算过)
        if aid in geo_assets:
            s += W_GEO
        # 维度3: scene_tag 命中（多命中归一）
        _sh = _asset_scene_match.get(aid, set())
        if _sh:
            s += W_SCENE * (len(_sh & _want_tags) / _want_tags_count)
        # 维度4: memory event 命中（用户的口述记忆）
        if aid in _asset_in_mem:
            s += W_MEMORY
        # 维度5: object 命中（显式让搜索 "杯子" 时杯子分明显）
        if _obj and _obj in _sh:
            s += W_OBJECT * 0.5
        # 兼容 SIGLIP scene score / emotion（弱权，避免它们压倒主观维度）
        if scene_kw:
            s += _SCENE_SCORE_CACHE.get(scene_kw, {}).get(aid, 0.0) * 0.3
        if intent.get("emotion") == "happy":
            s += _EMOTION_SCORE_CACHE.get(aid, 0.0) * 0.3
        ct = con.execute("SELECT capture_time FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
        scored.append((aid, s, ct["capture_time"] if ct else ""))

    # 时间范围约束（#2, 2026-09-04 智能回退）
    # 严格区间无结果时渐进放宽：±1月 → ±1季 → ±1年 → 全部放开。
    # 旧逻辑硬过滤会把"去年 3 月拍的雪景"丢成空（去年同期 31 天里
    # 实际只有 1 张），现在 5 张阈值后回退到 ±1 月即可命中。
    _TIME_FALLBACK_LEVELS = [
        (0, 0),       # 严格区间
        (1, 1),       # ±1 月
        (3, 3),       # ±1 季
        (12, 12),     # ±1 年
        (10**4, 10**4),  # 完全放开
    ]
    if t.get("start") or t.get("end"):
        start = t.get("start") or "0000-01-01"
        end = t.get("end") or "9999-12-31"
        _base = scored
        _fallback_used = 0
        for _lv, (m_lo, m_hi) in enumerate(_TIME_FALLBACK_LEVELS):
            if _lv == 0:
                _s_lo, _s_hi = start, end
            elif _lv == len(_TIME_FALLBACK_LEVELS) - 1:
                _s_lo, _s_hi = "0000-01-01", "9999-12-31"
            else:
                try:
                    import calendar as _cal
                    _sy, _sm, _sd = map(int, start.split("-"))
                    _ey, _em, _ed = map(int, end.split("-"))
                    _lo_dt = date(_sy, _sm, _sd)
                    _hi_dt = date(_ey, _em, _ed)

                    def _add_months(d, months):
                        m = d.month - 1 + months
                        y = d.year + m // 12
                        m = m % 12 + 1
                        day = min(d.day, _cal.monthrange(y, m)[1])
                        return date(y, m, day)
                    _s_lo = _add_months(_lo_dt, -m_lo).isoformat()
                    _s_hi = _add_months(_hi_dt, m_hi).isoformat()
                except Exception:
                    _s_lo, _s_hi = start, end
            _filtered = [s for s in _base if _s_lo <= (s[2] or "")[:10] <= _s_hi]
            if len(_filtered) >= 5:
                scored = _filtered
                _fallback_used = _lv
                break
        else:
            # 极端情况下也取个非空
            scored = _filtered if _filtered else _base
            _fallback_used = len(_TIME_FALLBACK_LEVELS) - 1
        # 答案提示用：把实际生效区间回填到 intent，便于 build_answer/llm_synthesize
        if _fallback_used > 0:
            intent["time_fallback_level"] = _fallback_used
            try:
                import calendar as _cal2
                _sy, _sm, _sd = map(int, start.split("-"))
                _ey, _em, _ed = map(int, end.split("-"))
                _lo_dt = date(_sy, _sm, _sd)
                _hi_dt = date(_ey, _em, _ed)

                def _add_months2(d, months):
                    m = d.month - 1 + months
                    y = d.year + m // 12
                    m = m % 12 + 1
                    day = min(d.day, _cal2.monthrange(y, m)[1])
                    return date(y, m, day)
                m_lo, m_hi = _TIME_FALLBACK_LEVELS[_fallback_used][:2]
                intent["time_effective_start"] = _add_months2(_lo_dt, -m_lo).isoformat()
                intent["time_effective_end"] = _add_months2(_hi_dt, m_hi).isoformat()
            except Exception:
                pass

    # 2026-09-04 修复：「孩子」并集后，各家孩子独家池在 50 张上限里严重失衡，
    # 按时间倒序稳定排序会把"独家+近期"照片多的孩子挤满前 50，导致用户
    # 感觉"只搜到其中一个孩子的照片"。处理：对 children 名单里的每个
    # 人各先取等额上限（取 min(8, 该人专属数) 张，独占照片），再补同框
    # 候选，确保每个孩子在结果中同时露脸。最后再走原有时间倒序截断。
    # #16 隐私清洗：孩子名单动态取自 person 表（_kids），代码零人名。
    if kids_union and candidate:
        from collections import defaultdict as _dd
        # 1) 拆分独家/同框/外人人脸
        asset_to_kids = _dd(set)
        for r in con.execute(
                "SELECT fi.asset_id, p.display_name FROM face_instance_v0 fi "
                "JOIN person p ON p.person_id=fi.person_id"):
            if r["display_name"] in _kid_set:
                asset_to_kids[r["asset_id"]].add(r["display_name"])
        only_pools = {kid: [] for kid in _kids}  # 各孩子独家池（保序）
        both, others = [], []
        kid_owner = {}
        for aid in candidate:
            kids_here = asset_to_kids.get(aid, set())
            kc = kids_here & _kid_set
            if len(kc) == 1:
                kid = next(iter(kc))
                only_pools[kid].append(aid)
                kid_owner[aid] = ("only", kid)
            elif len(kc) >= 2:
                both.append(aid)
                kid_owner[aid] = ("both",)
            else:
                others.append(aid)
        # 2) 每类按 capture_time 倒序
        _ct_q = ("SELECT capture_time FROM media_asset WHERE asset_id=?")
        def _sort_by_time_desc(aids):
            rows = [(aid, (con.execute(_ct_q, (aid,)).fetchone() or {"capture_time": ""})["capture_time"] or "")
                    for aid in aids]
            rows.sort(key=lambda x: x[1], reverse=True)
            return rows
        only_r = {kid: _sort_by_time_desc(aids) for kid, aids in only_pools.items()}
        both_r = _sort_by_time_desc(both)
        # 3) 各取至少 8 张（让每个孩子都露脸），不足再给齐
        PER_KID = 8
        picked = []
        seen = set()
        def _take(rows, n):
            out = []
            for aid, ct in rows:
                if aid in seen:
                    continue
                out.append((aid, 0.5, ct))  # 人脸命中给基础分 0.5
                seen.add(aid)
                if len(out) >= n:
                    break
            return out
        # 同框优先（热门组合），但同框少的时候让独家先露脸
        # 平均从每个孩子各拿一半名额，再补同框
        picked += _take(both_r, PER_KID * 2)
        for kid in _kids:
            picked += _take(only_r[kid], PER_KID)
        # 再补 any-候选（时间倒序排进来）直到 800（2026-09-06：原 50 截断
        # 让"孩子的照片"只显示某一孩子独占 + 同框为主，其他孩子独占被吃掉一大半）。
        _other_r = _sort_by_time_desc(others)
        for aid, ct in _other_r:
            if aid in seen:
                continue
            picked.append((aid, 0.0, ct))
            seen.add(aid)
            if len(picked) >= 800:
                break
        scored = picked[:800]
        # 跳过后续默认排序逻辑
        _skip_default_sort = True
    else:
        _skip_default_sort = False

    if intent.get("exclude"):
        # 管理过滤默认展示最近的正常影像，而不是最早的 50 项。
        scored.sort(key=lambda x: x[2] or "", reverse=True)
    elif "第一次" in question:
        # "第一次去…"类查询：同分内时间正序（最早的那次排最前）
        scored.sort(key=lambda x: (-x[1], x[2] or ""))
    elif _skip_default_sort:
        pass  # kids_union 已构造好 picked
    else:
        # 2026-09-04 修复：默认最新优先。原实现时间升序导致
        # "爸爸的照片"返回 2016 年旧照排在最前，用户感知"搜索不准"。
        # 先按时间倒序，再按分数稳定排序（同分时时间新的在前）。
        scored.sort(key=lambda x: x[2] or "", reverse=True)
        scored.sort(key=lambda x: -x[1])
    # ===== 多样性重排（#5, 2026-09-04 MMR）=====
    # 前 k 张里同一天、同 scene_tag 扎堆时，贪心重排避免用户感觉
    # "照片墙都是同一时刻"。只动头部，尾部时间倒序保持。
    scored = mmr_rerank_topk(con, scored, k=10, lam=0.7)
    # 2026-09-06：用户反馈"搜某孩子只出几十张，但可能有几百张"。
    # 50 上限截断了照片量大（3000+）的人脸信号强的查询；
    # 上限放宽到 800 让对角线墙可以滚到"近期 + 一些老照片"的视野。
    # 性能验证：3266 张 evidence_weight 评分实测 37ms，
    # 前端只渲染 wallOffset 周围 ~50 个 DOM 元素，数据长度不影响渲染成本。
    # 800 是平衡"用户能滚到的视野"和"异常查询爆量保护"的安全帽。
    result_assets = scored[:800]

    # ===== 结果折叠：仅已确认的完全重复/衍生副本 =====
    result_assets = dedup_burst(con, result_assets)
    if intent.get("exclude"):
        result_assets.sort(key=lambda a: a.get("time") or "", reverse=True)
    if result_assets:
        type_map = {r["asset_id"]: r["media_type"] for r in con.execute(
            "SELECT asset_id,media_type FROM media_asset WHERE asset_id IN ({})".format(
                ",".join("?" * len(result_assets))), [a["id"] for a in result_assets])}
        for asset in result_assets:
            asset["type"] = type_map.get(asset["id"], "photo")
            dims = con.execute("SELECT width,height FROM media_asset WHERE asset_id=?", (asset["id"],)).fetchone()
            if dims:
                asset["width"], asset["height"] = dims["width"], dims["height"]

    # 生成回答
    answer = build_answer(question, intent, memory_hits, result_assets, con)
    result = {
        "answer": answer,
        "assets": result_assets,
        "memory": memory_hits,
        "persons": persons,
        "intent": intent,
    }
    # 2026-09-06：无结果时给结构化 hint，区分"地点/人物/物品/事件"四种空原因，
    # 让用户知道为什么没找到（只输出"没找到"答不出"为什么没有"，用户会重复问也答不出来）。
    # re-open DB 因为前面 close 在 return 之后做——先借用 con 余下的连接做查询
    if not result_assets and not intent.get("blocked"):
        try:
            result["hint"] = _build_empty_hint(question, intent, con)
        except Exception as _e:
            print(f"[hint] 生成失败: {_e}")
    con.close()
    return result


def _build_empty_hint(question, intent, con):
    """2026-09-06：根据意图分类给出诚实的"为什么没找到"提示。
    地点/人物/物品/事件 四种空原因单独说明，让用户知道下一步该做什么。"""
    # ===== 地点类：geo 表 / event 标题 / scene_tag 三张表全 0 =====
    # description 表不算地点信号——VLM 把材质"大理石桌面"也会写进描述里，
    # 用 LIKE '%大理%' 会误命中 64 条"大理石"分母，反而掩盖了"真无云南数据"的真相。
    geo = [k for k in (intent.get("events") or []) if k not in ("旅行",)]
    has_person = bool(intent.get("persons"))
    has_object = bool(intent.get("object"))
    has_event_other = bool(intent.get("memory_keywords"))
    if geo and not has_person and not has_object and not has_event_other:
        for kw in geo:
            n_geo = con.execute(
                "SELECT COUNT(*) FROM asset_geo_v0 WHERE region LIKE ? OR province LIKE ?",
                (f"%{kw}%", f"%{kw}%")).fetchone()[0]
            n_ev = con.execute(
                "SELECT COUNT(*) FROM event WHERE title LIKE ?",
                (f"%{kw}%",)).fetchone()[0]
            n_tag = con.execute(
                "SELECT COUNT(*) FROM scene_tag_v0 WHERE tag LIKE ?",
                (f"%{kw}%",)).fetchone()[0]
            n_geo_ev_tag = n_geo + n_ev + n_tag
            if n_geo_ev_tag == 0:
                return (
                    f"库里没找到「{kw}」相关的照片。"
                    f"地理表 / 事件标题 / 场景标签 三张表全 0 命中；"
                    f"很可能是那批照片还没接进 source，或者 VLM 描述里没把地名写出来。"
                )
        # 有命中但被多条件交集筛空（其它维度把人/事件跟地点交得没结果）
        return (
            f"「{'、'.join(geo)}」的记录存在，但被其它条件（时间 / 人物 / 排除项等）筛光了。"
            f"试着把时间/人物这些条件去掉再问一次。"
        )

    # ===== 人物类：face_instance_v0 没人脸 =====
    persons = intent.get("persons") or []
    if persons and not has_object and not geo:
        missing = []
        for p in persons:
            n = con.execute(
                "SELECT COUNT(*) FROM face_instance_v0 fi "
                "JOIN person pe ON pe.person_id=fi.person_id "
                "WHERE pe.display_name=? AND pe.identity_status='confirmed'",
                (p,)).fetchone()[0]
            if n == 0:
                missing.append(p)
        if missing:
            return (
                f"库里没有「{'、'.join(missing)}」的人脸标签记录。"
                f"先到「人物」页确认几张疑似脸是否为本人，再回来问问。"
            )
        return "该人物有人脸数据，但其它条件（时间 / 地点 / 排除等）把结果筛光了，把其它条件去掉再问问。"

    # ===== 物品类：scene_tag 不命中 =====
    obj = intent.get("object")
    if obj and not has_person and not geo:
        otag = object_tag_for_word(obj)
        n_tag = 0
        if otag:
            n_tag = con.execute(
                "SELECT COUNT(*) FROM scene_tag_v0 WHERE tag=?",
                (otag,)).fetchone()[0]
        if n_tag == 0:
            return (
                f"库里没有打「{obj}」相关场景标签的记录。"
                f"VLM 还没给照片打这类标签，或者这批照片还没扫进 source。"
            )

    # ===== 事件/记忆类：memory_note 未命中 =====
    mem_kws = intent.get("memory_keywords") or []
    if mem_kws and not has_person and not has_object and not geo:
        for kw in mem_kws:
            n = con.execute(
                "SELECT COUNT(*) FROM memory_note WHERE text LIKE ?",
                (f"%{kw}%",)).fetchone()[0]
            if n == 0:
                return (
                    f"库里「{kw}」没有相关记忆备注。"
                    f"打开「人物详情 → 记忆」页给这张照片加一句口述记忆，下次就能搜到了。"
                )

    # ===== 兜底 =====
    return (
        "没找到符合条件的照片。"
        "如果刚才能查到、现在查不到，也许是这批照片被加入了「排除项」或在隐私相册里。"
    )


def build_answer(question, intent, memory_hits, assets, con):
    """根据意图和检索结果生成自然语言回答。"""
    parts = []
    if intent.get("exclude"):
        excluded_count = len(get_hidden_ids(con))
        total_count = con.execute("SELECT count(*) FROM media_asset").fetchone()[0]
        parts.append(
            f"已排除 {excluded_count} 项确定的截图、录屏、聊天导出或下载影像；"
            f"保留 {total_count-excluded_count} 项正常影像，下面显示最近的 {len([a for a in assets if not a.get('hidden')])} 项。"
        )
    if intent.get("emotion"):
        visible = [a for a in assets if not a.get("hidden")]
        if visible:
            parts.append(f"找到 {len(visible)} 个高置信度的明显笑容候选。这是本地模型对可见表情的判断，不代表人物真实内心状态。")
        else:
            parts.append("没有找到达到高精度阈值的明显笑容候选；系统没有用低置信度结果凑数。")
    # 时间问题（什么时候/哪年）
    if "什么时候" in question or "哪年" in question or "几岁" in question:
        if memory_hits:
            m = memory_hits[0]
            ev = con.execute("SELECT start_time FROM event WHERE event_id=?", (m.get("event_id"),)).fetchone()
            if ev and ev["start_time"]:
                parts.append(f"是 {ev['start_time'][:10]}。")
            if m.get("text"):
                parts.append(f"当时的记忆：「{m['text']}」")
        # 年龄问题
        if "几岁" in question:
            # 找第一个资产的时间，判断哪个孩子
            for a in assets[:1]:
                aid = a["id"]
                ct = con.execute("SELECT capture_time FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
                if ct and ct["capture_time"]:
                    y, m, d = map(int, ct["capture_time"][:10].split("-"))
                    shot = date(y, m, d)
                    for name in _kids:  # #16：孩子动态取自 person 表
                        bd = con.execute("SELECT birth_date FROM person WHERE display_name=?", (name,)).fetchone()
                        if bd and bd["birth_date"]:
                            by, bm, bd_ = map(int, bd["birth_date"].split("-"))
                            birth = date(by, bm, bd_)
                            age_y = shot.year - birth.year - ((shot.month, shot.day) < (birth.month, birth.day))
                            age_m = (shot - date(birth.year + age_y, birth.month, birth.day)).days // 30
                            # 检查该资产是否含此人
                            pid = get_person_id(con, name)
                            has = con.execute(
                                "SELECT 1 FROM face_instance_v0 WHERE asset_id=? AND person_id=?", (aid, pid)).fetchone()
                            if has:
                                parts.append(f"{name}当时 {age_y} 岁 {age_m} 个月左右。")
                                break
    if not parts:
        if assets:
            visible = [a for a in assets if not a.get("hidden")]
            times = [a["time"][:10] for a in visible if a.get("time")]
            n_hidden = len(assets) - len(visible)
            if times:
                base = f"找到 {len(visible)} 项，时间范围 {min(times)} ~ {max(times)}。"
            else:
                base = f"找到 {len(visible)} 项。"
            if n_hidden > 0:
                base += f"（另有 {n_hidden} 个已确认重复/衍生副本折叠）"
            parts.append(base)
            # 内容理解层：结果覆盖的「天」有 AI 描述时，附上最相关几天的描述摘录
            day_summaries = top_day_captions(con, visible or assets, limit=2)
            for ds in day_summaries:
                parts.append(ds)
            # 多设备互证层：该天有多个来源/设备同时记录时，给出互证证据
            for ce in cross_device_evidence(con, visible or assets, intent):
                parts.append(ce)
        else:
            media_word = "视频" if intent.get("media_type") == "video" else "照片"
            parts.append(f"没有找到相关{media_word}。可能需要导入更多影像，或换个说法试试。")
    return " ".join(parts)


def cross_device_evidence(con, assets, intent=None, limit=2):
    """多设备互证证据：结果覆盖的天里，有 ≥2 个来源/设备同时记录时返回证据句。
    冲突项（同刻拍摄但 GPS 对不上 / 同一文件但时间对不上）一并提示，不掩饰。"""
    days = []
    for a in (assets or []):
        t = a.get("time") or ""
        if t and t[:10] not in days:
            days.append(t[:10])
    if not days:
        return []
    ph = ",".join("?" * len(days))
    out = []
    for r in con.execute(
        f"""SELECT day, sources_json, device_count, same_moment_count,
                   exact_cross_count, geo_conflict_count FROM day_corroboration_v0
            WHERE day IN ({ph}) ORDER BY same_moment_count DESC, exact_cross_count DESC""", days):
        try:
            sources = json.loads(r["sources_json"] or "{}")
        except Exception:
            sources = {}
        names = [k for k in sources if k]
        bits = []
        if r["same_moment_count"] > 0:
            bits.append(f"{r['same_moment_count']} 组不同设备同刻拍摄")
        if r["exact_cross_count"] > 0:
            bits.append(f"{r['exact_cross_count']} 组跨设备互传的同一照片")
        if not bits and r["device_count"] >= 2:
            bits.append(f"{r['device_count']} 台设备各自记录")
        if bits:
            line = f"互证：{r['day']} 有 {len(names)} 个相册来源（{'、'.join(names[:4])}），" + "，".join(bits) + "。"
        else:
            continue
        if r["geo_conflict_count"] > 0:
            line += f"（有 {r['geo_conflict_count']} 组 GPS 位置对不上，已降置信）"
        out.append(line)
        if len(out) >= limit:
            break
    return out


def top_day_captions(con, assets, limit=2):
    """取结果资产覆盖的天里，有 VLM 叙事描述优先、否则本地综合描述的摘录（每天一句）。"""
    days = []
    for a in (assets or []):
        t = a.get("time") or ""
        if t and t[:10] not in days:
            days.append(t[:10])
    if not days:
        return []
    ph = ",".join("?" * len(days))
    out = []
    for r in con.execute(
        f"""SELECT day, caption_local, caption_vlm FROM day_caption_v0
            WHERE day IN ({ph}) ORDER BY day DESC""", days):
        text = (r["caption_vlm"] or r["caption_local"] or "").strip()
        if text:
            out.append(f"「{r['day']}」{text}")
        if len(out) >= limit:
            break
    return out


# ============ 缩略图 ============
def get_orig_path(asset_id):
    """返回原片绝对路径（不存在返回 None）。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    row = con.execute("""SELECT mf.absolute_path FROM media_file mf
        JOIN media_asset ma USING(asset_id)
        WHERE mf.asset_id=? ORDER BY mf.byte_size DESC LIMIT 1""", (asset_id,)).fetchone()
    con.close()
    if not row:
        return None
    path = row["absolute_path"]
    return path if os.path.exists(path) else None


THUMB_EDGE = 480  # 2026-09-01 全站统一 480 档（1600 档已废弃删除，见 get_thumb 注释）

# 2026-09-04 原图存在性进程级缓存：缩略图瀑布流每张图都要判断原图是否还在，
# 避免每次缓存命中都重复查 SQLite + stat NAS。
# 存在(True)缓存 60s 兼顾性能；不存在(False)只缓存 5s，能较快反映"原图被还原/重挂"。
_ORIG_EXISTS_CACHE = {}      # asset_id -> (ts, bool)
_ORIG_EXISTS_TTL = 60.0
_ORIG_EXISTS_MISS_TTL = 5.0

def _orig_exists(asset_id):
    now = time.time()
    hit = _ORIG_EXISTS_CACHE.get(asset_id)
    if hit:
        ttl = _ORIG_EXISTS_TTL if hit[1] else _ORIG_EXISTS_MISS_TTL
        if now - hit[0] < ttl:
            return hit[1]
    ok = get_orig_path(asset_id) is not None
    _ORIG_EXISTS_CACHE[asset_id] = (now, ok)
    return ok

# ── Log 原片色彩还原（2026-09-23）────────────────────────────────────────
# 大疆 D-Log / 影石 Flat 灰片的通病：缩略图上又灰又平，用户以为拍坏了。
# 判定在 enrich.py 第 7 步（logcolor）落 asset_log_color_v0，这里只管出图挂滤镜。
# **原片文件永不修改**——还原只作用于缩略图缓存与预览图，关掉开关即恢复原样。
# 参数取 4 档实测里最自然的一档（/tmp/logcolor_probe/compare.jpg）。
LOGCOLOR_FULL = {"rimin": 0.09, "rimax": 0.92, "contrast": 1.10,
                 "saturation": 1.55, "gamma": 1.03}

_lc_cfg_cache = [0.0, True, 1.0]    # [读取时刻, 是否开, 全局强度] —— 10s，改设置不必重启
_LC_ENABLED_TTL = 10.0
_LC_ROW_CACHE = {}                  # asset_id -> (读取时刻, 当时的全局强度, (ver,params,tag)|None)
_LC_ROW_TTL = 300.0


def _logcolor_cfg():
    """(是否启用, 全局强度)。走算法偏好面板（algo_logcolor_*），与其它阈值同约定。"""
    now = time.time()
    if now - _lc_cfg_cache[0] < _LC_ENABLED_TTL:
        return _lc_cfg_cache[1], _lc_cfg_cache[2]
    try:
        on = get_algo("logcolor_enabled", "1") == "1"
        gs = max(0.0, min(1.5, float(get_algo("logcolor_strength", "1") or 1)))
    except Exception:
        on, gs = True, 1.0
    _lc_cfg_cache[:] = [now, on, gs]
    return on, gs


def _logcolor_params(strength):
    """strength 0~1.5 线性插值到「无效果↔B 档」，0 = 原样输出。"""
    s = max(0.0, min(1.5, float(strength if strength is not None else 1.0)))
    f = LOGCOLOR_FULL
    return {"rimin": f["rimin"] * s,
            "rimax": 1.0 - (1.0 - f["rimax"]) * s,
            "contrast": 1.0 + (f["contrast"] - 1.0) * s,
            "saturation": 1.0 + (f["saturation"] - 1.0) * s,
            "gamma": 1.0 + (f["gamma"] - 1.0) * s}


def _logcolor_vf(p):
    """ffmpeg 滤镜片段（不含 format/scale，由调用方拼在链首）。
    注意：eq/colorlevels 遇 10bit 管线会输出**全黑帧**，所以整条 vf 必须以
    format=yuv420p 开头先把 HEVC 10bit 降到 8bit——这是本功能的必修坑。"""
    return (f"colorlevels=rimin={p['rimin']:.4f}:gimin={p['rimin']:.4f}:bimin={p['rimin']:.4f}"
            f":rimax={p['rimax']:.4f}:gimax={p['rimax']:.4f}:bimax={p['rimax']:.4f},"
            f"eq=contrast={p['contrast']:.4f}:saturation={p['saturation']:.4f}:gamma={p['gamma']:.4f}")


_LOGCOLOR_PIL_SAT_FIX = 0.903   # 同数值下 PIL Color 比 ffmpeg eq 饱和高约 11%（实测 1.40 才等值于 1.55）


# ============ 全局资源护栏（2026-09-24 产品化：保证不把用户的 NAS 拖崩）============
# 单条 ffmpeg 加 RLIMIT 只是第一层。真正的杀手是**多条叠加 + 整机本来就不宽裕**：
# 实测过 load 76、swap 风暴把 ssh 和 dockerd 一起压死。所以还必须有整机水位闸门：
#   · 起任何重活之前先看水位，不够就不起（退避，而不是硬上）
#   · 运行中持续巡检，跌破危险线直接熔断（宁可这次转码/抽帧失败，脚本幂等会续跑）
# 所有阈值按机器内存自适应——4GB 的 NAS 和 32GB 的服务器不能共用一套数字。
def _meminfo():
    """(总内存MB, 可用MB)；非 Linux / 读不到返回 (0,0)，此时护栏放行。"""
    try:
        d = {}
        for line in open("/proc/meminfo"):
            k, _, v = line.partition(":")
            d[k.strip()] = int(v.split()[0]) // 1024
        return d.get("MemTotal", 0), d.get("MemAvailable", 0)
    except Exception:
        return 0, 0


def res_levels():
    """(总内存MB, 警戒线MB, 熔断线MB)：起活的门槛 与 必须停手的底线。

    默认按机器内存自适应；高级用户可用 res_warn_mb / res_danger_mb 覆盖
    （设置面板里可填，留空 = 自适应）。
    """
    p = machine_profile()
    total = p["mem_total_mb"]
    if not total:
        return 0, 0, 0
    ow = (get_setting("res_warn_mb", "") or "").strip()
    od = (get_setting("res_danger_mb", "") or "").strip()
    if ow or od:
        try:
            warn_mb = int(ow) if ow else p["warn_mb"]
            danger_mb = int(od) if od else max(128, warn_mb // 2)
            if danger_mb >= warn_mb:
                danger_mb = max(128, warn_mb // 2)
            return total, warn_mb, danger_mb
        except ValueError:
            pass
    return total, p["warn_mb"], p["danger_mb"]


def res_ffmpeg_rlimit():
    """单条 ffmpeg 的虚拟内存上限（自适应）。转码脚本也调这个，与抽帧同源。"""
    return machine_profile()["ffmpeg_rlimit_mb"] * 1024 * 1024


def res_snapshot():
    """(总内存MB, 可用MB, 1 分钟负载)"""
    try:
        load1 = float(open("/proc/loadavg").read().split()[0])
    except Exception:
        load1 = 0.0
    total, avail = _meminfo()
    return total, avail, load1


def res_ok(for_task=""):
    """能不能起一个重活：可用内存够 + 负载不高，两者都满足才放行。"""
    if get_setting("res_guard_enabled", "1") != "1":
        return True
    total, avail, load1 = res_snapshot()
    if not total:
        return True                       # 读不到内存信息（非 Linux）→ 放行
    _, warn_mb, _ = res_levels()
    load_cap = machine_profile()["load_budget"]   # 按机器内存档给，不按核数：4 核 NAS 也得压
    if avail < warn_mb:
        print(f"[res-guard] 可用内存 {avail}MB < {warn_mb}MB，{for_task}暂不启动", flush=True)
        return False
    if load1 > load_cap:
        print(f"[res-guard] 负载 {load1} > {load_cap}，{for_task}暂不启动", flush=True)
        return False
    return True


def res_wait_ok(timeout=20.0, for_task=""):
    """等水位恢复到安全线；超时返回 False（调用方自行决定放行还是拒绝）。"""
    end = time.time() + max(0.0, timeout)
    while True:
        if res_ok(for_task=for_task):
            return True
        if time.time() >= end:
            return False
        time.sleep(2)


def res_admit(for_task="", wait=20.0):
    """请求级准入（用户刷视频墙这种突发量才需要）。

    跟后台任务的 res_ok 不同：用户在前台等着看图，不能一句「内存不够」就甩
    灰块。所以先等一会儿（默认 20s），等到就正常生成；等不到再分情况：
      · 只是警戒线以下 → 放行（单条 ffmpeg 有 RLIMIT 兜底，串行锁也只有 1 路）
      · 已跌破熔断线   → 拒绝，退回旧缓存/占位，绝不再往火上浇油
    """
    if get_setting("res_guard_enabled", "1") != "1":
        return True
    if res_wait_ok(wait, for_task=for_task):
        return True
    total, avail, _ = res_snapshot()
    if not total:
        return True
    _, _, danger_mb = res_levels()
    if avail < danger_mb:
        print(f"[res-guard] 拒绝{for_task}：可用内存 {avail}MB 已跌破熔断线 "
              f"{danger_mb}MB（退回占位，不新增解码）", flush=True)
        return False
    return True


def _kill_ffmpeg():
    """终止正在跑的 ffmpeg（内存大头）。pattern 用变量拼接——直接写字符串会
    匹配到本进程命令行自身，把自己也杀掉（本项目踩过两次）。"""
    n = 0
    pat = "ffm" + "peg"
    try:
        for d in os.listdir("/proc"):
            if not d.isdigit():
                continue
            try:
                cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode("utf8", "replace")
            except Exception:
                continue
            if pat not in cmd:
                continue
            try:
                os.kill(int(d), 9)
                n += 1
            except Exception:
                pass
    except Exception:
        pass
    return n


def res_emergency(reason=""):
    """熔断：可用内存跌破底线 → 杀掉 ffmpeg。宁可这次失败（幂等会续跑），
    也不能让整机滑进 swap 风暴（那会连 SSH 和 docker 一起拖死）。"""
    total, avail, _ = res_snapshot()
    if not total:
        return False
    _, _, danger_mb = res_levels()
    if avail >= danger_mb:
        return False
    n = _kill_ffmpeg()
    if n:
        print(f"[res-guard] 熔断：可用内存仅 {avail}MB（<{danger_mb}MB），"
              f"已终止 {n} 个 ffmpeg（{reason}）", flush=True)
    return bool(n)


def _res_guard_watchdog():
    """每 60 秒巡检一次整机水位，跌破底线就熔断。"""
    time.sleep(60)
    while True:
        try:
            if get_setting("res_guard_enabled", "1") == "1":
                res_emergency(reason="定时巡检")
        except Exception as exc:
            print(f"[res-guard] watchdog error: {type(exc).__name__}: {exc}", flush=True)
        time.sleep(60)


# ============ 资源护栏：按机器内存自适应（2026-09-24 产品化）============
# 4GB 的 NAS 和 32GB 的台式不能共用一套数字。4K HEVC 10bit 解码是本项目最吃内存的
# 操作，一条就能把小机器推进 swap（内存交换）死亡螺旋（09-24 实锤：load 76、
# dockerd 被压死、SSH 命令被 SIGTERM（终止信号））。
# **所有重活开工前必须先问 machine_profile()**，并发数 / 内存上限 / 线程数一律从
# 这里取，调用处不许再写死。
# 刻意不读数据库：模块导入阶段就要用这里的并发数建信号量，那时 DB 还没建好；
# 用户手填的水位覆盖另外在 res_levels() 里叠加。

# 档位表：每档数字要么来自实锤故障，要么来自容器里实测通过的值，改之前先想清楚。
#   warn_mb / danger_mb        整机水位线：低于 warn 不起新活，低于 danger 直接熔断
#   ffmpeg_rlimit_mb           单条抽帧 ffmpeg 的虚拟内存上限（RLIMIT_AS 地址空间上限）
#   transcode_rlimit_mb        单条整片转码的上限（比抽帧宽：要带 x264 的多帧缓存）
#   video_concurrency          同时解几条 4K（按**内存**定，不按 CPU 核数定）
#   load_budget                1 分钟负载预算：超了就退避，把机器让给正在刷图的人
#   day_threads / night_threads 重活线程数（白天压着别卡站，夜里没人用才提速）
_MACHINE_TIERS = {
    "small":  dict(warn_mb=400,  danger_mb=220,  ffmpeg_rlimit_mb=800,
                   transcode_rlimit_mb=1000, video_concurrency=1,
                   load_budget=2.0, day_threads=1, night_threads=1),
    "mid":    dict(warn_mb=700,  danger_mb=400,  ffmpeg_rlimit_mb=1500,
                   transcode_rlimit_mb=2200, video_concurrency=1,
                   load_budget=3.0, day_threads=1, night_threads=2),
    "large":  dict(warn_mb=900,  danger_mb=500,  ffmpeg_rlimit_mb=2400,
                   transcode_rlimit_mb=2600, video_concurrency=2,
                   load_budget=6.0, day_threads=2, night_threads=2),
    "xlarge": dict(warn_mb=1200, danger_mb=700,  ffmpeg_rlimit_mb=3000,
                   transcode_rlimit_mb=3200, video_concurrency=2,
                   load_budget=8.0, day_threads=2, night_threads=4),
}


def machine_profile():
    """整机资源档位：本项目「这件事最多能吃多少」的唯一答案。

    为什么要自适应：同一份代码要跑在 3.9GB 的群晖、16GB 的台式、还有 Mac 直跑。
    写死一套数字，要么小机器被拖死，要么大机器慢得没道理。
    """
    total, avail = _meminfo()
    if not total:                      # 读不到（Mac / 非 Linux）：按中等档，本机内存通常够
        tier = "mid"
    elif total <= 3072:
        tier = "small"
    elif total <= 8192:                # 4~8GB：典型家用 NAS，本项目主力目标机
        tier = "mid"
    elif total <= 16384:
        tier = "large"
    else:
        tier = "xlarge"
    cfg = dict(_MACHINE_TIERS[tier])
    cfg.update(tier=tier, mem_total_mb=total, mem_avail_mb=avail)
    return cfg


def ffmpeg_budget_ok(need_mb=None):
    """起 ffmpeg 之前的两道闸：(ok, 原因)。

    ① 可用内存够不够这条命令本身（不够的话起了也是 malloc 失败退出，白烧 CPU 还
       把机器往 swap 里推）；② 整机负载是否超档位预算——有人正在刷图就让出去。
    后台任务和前台都走同一个预算，不存在「前台偷偷开口子把机器压垮」的情况。
    """
    p = machine_profile()
    need = int(need_mb or p["ffmpeg_rlimit_mb"])
    floor = max(p["warn_mb"], int(need * 0.8))
    if p["mem_avail_mb"] and p["mem_avail_mb"] < floor:
        return False, f"可用内存 {p['mem_avail_mb']}MB < 安全线 {floor}MB"
    if _load1() > p["load_budget"]:
        return False, f"负载 {_load1():.1f} > 预算 {p['load_budget']}"
    return True, ""


def bj_hour():
    """当前北京时间（0-23）。Docker 镜像默认 UTC（协调世界时），直接 localtime
    会错 8 小时（09-24 实锤：北京 15 点被判成凌晨 7 点，白天走了夜间提速档）。"""
    return (time.gmtime().tm_hour + 8) % 24



def _ffmpeg_run_guarded(cmd, **kw):
    """ffmpeg 子进程内存护栏（2026-09-24 风暴实锤后加）。

    NAS 3.9GB 内存上，4K HEVC D-Log 的 ffmpeg 抽帧曾把整机推进 swap 死亡螺旋
    （I/O 风暴压死 dockerd，见 state-20260923 快照的故障记录）。护栏：
      · RLIMIT_AS 按机器内存自适应（res_ffmpeg_rlimit()，4GB 机器 1.2GB 起）——
        超限 ffmpeg 自己报错退出，调用方已有「第 0 帧重试 / 未还原回退 /
        _thumb_fallback」三级兜底，宁可灰片不能压死机器；
      · nice 19 —— 最低优先级，风暴期不与关键服务抢 CPU；
      · 水位闸门不在本函数内部（照片缩略图也走它，一律拦截会让正常浏览也变灰），
        由调用方按场景选择：后台批量用 res_ok()，前台刷图用 res_admit()；
        运行中由 _res_guard_watchdog 每 60 秒巡检，跌破熔断线直接杀 ffmpeg。
    """
    import resource as _res

    def _limit():
        _rl = res_ffmpeg_rlimit()
        try:
            _res.setrlimit(_res.RLIMIT_AS, (_rl, _rl))
        except Exception:
            pass
        try:
            os.nice(19)
        except Exception:
            pass

    kw.setdefault("preexec_fn", _limit)
    return subprocess.run(cmd, **kw)


def _logcolor_pil(im, p):
    """Pillow 版还原 —— 作为 ffmpeg 那条链的等价物（视频走 ffmpeg，照片走这里）。
    三处必须按 ffmpeg 语义来，不能图省事直接用 ImageEnhance：
      · 对比度以 **0.5 为轴**（ffmpeg eq 的 pivot）；ImageEnhance.Contrast 是绕
        「图像均值」转的，暗片（正是灰片）上结果完全不一样；
      · 饱和度在色度域等价缩放，PIL Color 同值偏高 ~11% → 乘补偿系数；
      · gamma 与色阶、对比度一起压进同一条逐通道 LUT。
    实测（平滑色度测试图）：与 ffmpeg 输出平均绝对差 0.009（满量程 1.0）。"""
    from PIL import ImageEnhance
    if im.mode != "RGB":
        im = im.convert("RGB")
    span = max(1e-6, p["rimax"] - p["rimin"])
    lo = p["rimin"] * 255.0
    gam = 1.0 / max(1e-6, p["gamma"])
    lut = []
    for v in range(256):
        x = min(1.0, max(0.0, (v - lo) / span / 255.0))            # colorlevels
        x = min(1.0, max(0.0, (x - 0.5) * p["contrast"] + 0.5))    # 对比度，轴 0.5
        lut.append(min(255, max(0, int(round(x ** gam * 255.0)))))  # gamma
    im = im.point(lut * 3)
    # 注意判断用**原始** saturation：拿补偿后的值判断，strength=0（sat=1.0）
    # 会被误判成需要降饱和（1.0*0.903 < 1）→ 把图洗淡。
    if abs(p["saturation"] - 1.0) > 1e-3:
        im = ImageEnhance.Color(im).enhance(p["saturation"] * _LOGCOLOR_PIL_SAT_FIX)
    return im


def logcolor_for(asset_id):
    """该资产要不要还原。返回 (thumb_ver, params, tag) 或 None。
    生效强度 = 全局 algo_logcolor_strength × 该资产自己的 strength（默认 1.0）；
    二者都编进 tag，所以调参数后旧缩略图缓存自动失效、不用手工清目录。"""
    on, gs = _logcolor_cfg()
    if not on:
        return None
    now = time.time()
    hit = _LC_ROW_CACHE.get(asset_id)
    if hit and now - hit[0] < _LC_ROW_TTL and abs(hit[1] - gs) < 1e-6:
        return hit[2]
    val = None
    try:
        con = sqlite3.connect(DB, timeout=10)
        con.row_factory = sqlite3.Row
        row = con.execute("""SELECT is_log, strength, thumb_ver FROM asset_log_color_v0
                             WHERE asset_id=?""", (asset_id,)).fetchone()
        con.close()
        if row and int(row["is_log"] or 0) == 1:
            per = float(row["strength"] if row["strength"] is not None else 1.0)
            s = max(0.0, min(1.5, gs * per))
            ver = int(row["thumb_ver"] or 1)
            val = (ver, _logcolor_params(s), f"v{ver}s{int(round(s * 100))}")
    except Exception:
        val = None      # 表还没迁（老库）→ 静默不还原，不影响出图
    if len(_LC_ROW_CACHE) > 5000:
        _LC_ROW_CACHE.clear()
    _LC_ROW_CACHE[asset_id] = (now, gs, val)
    return val


def _logcolor_for_assets(ids, con):
    """批量查「哪些资产要还原」，供列表接口一次性打标（照 quality_map / face_pos_map
    的写法）。前端拿到 logcolor=1 就把查看器最高清层换成 /preview（已还原），
    免得点开大图看到的是灰片原图。"""
    if not ids or not _logcolor_cfg()[0]:
        return {}
    out = {}
    try:
        q = ",".join("?" * len(ids))
        for r in con.execute("SELECT asset_id FROM asset_log_color_v0 "
                             f"WHERE is_log=1 AND asset_id IN ({q})", list(ids)):
            out[r[0]] = 1
    except Exception:
        return {}
    return out


_THUMB_SEM = threading.BoundedSemaphore(8)   # 2026-09-10：回源 NAS 生成缩略图全局限流
# 2026-09-24 视频抽帧专用串行锁：一条 4K HEVC 10bit 解码就能吃 1~2GB，
# _THUMB_SEM 的 8 路并发 × 4K = 内存耗尽 → swap 风暴（实测 load 76、ssh 都被
# SIGTERM，墙面 217 张缩略图全部超时灰块）。视频缩略图串行生成，慢但稳。
_VIDEO_THUMB_SEM = threading.BoundedSemaphore(machine_profile()["video_concurrency"])


def get_thumb(asset_id, edge=THUMB_EDGE):
    """生成指定边长档位的缩略图，按 edge 分档缓存。

    2026-09-01 分档：墙上卡片只渲染 ~233px，却要下载 1600px 档（470KB/张），
    首页 34 张就 15.6MB。小档优先从已缓存的更大档本地缩小生成（不重读 NAS），
    只有连更大档都没有时才回源原图。

    2026-09-04：原图被外部删除后，缩略图缓存即使还在也返回 404（前端 <img onerror>
    据此静默删除卡片），不再出现"幽灵缩略图"——缩略图缓存会欺骗用户以为照片还在。
    """
    # 先确认原图仍存在，否则直接 404（无论缩略图缓存是否命中）
    if not _orig_exists(asset_id):
        return None
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    # 2026-09-23 Log 还原：缓存名多带 _lc 后缀，与未还原的旧缓存彻底隔离。
    # 命中还原时**不复用更大档**——那些档可能是加滤镜前生成的，缩出来就是灰片。
    lc = logcolor_for(asset_id)
    stem = asset_id[6:]
    out = (THUMB_DIR / f"{stem}_t{edge}_lc{lc[2]}.jpg" if lc
           else THUMB_DIR / f"{stem}_t{edge}.jpg")   # 文件名带档位，旧缓存不冲突
    if out.exists():
        return out
    if not lc:
        # 优先从已缓存的更大档本地缩小（秒出，不碰 NAS）
        for bigger in (1600, 1200, 800, 600, 480, 400):
            if bigger <= edge:
                continue
            src = THUMB_DIR / f"{asset_id[6:]}_t{bigger}.jpg"
            if src.exists():
                try:
                    resize_to_jpeg(src, out, edge, quality=85)
                    if out.exists():
                        return out
                except Exception:
                    pass
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    row = con.execute("""SELECT mf.absolute_path, ma.media_type FROM media_file mf
        JOIN media_asset ma USING(asset_id)
        WHERE mf.asset_id=? ORDER BY mf.byte_size DESC LIMIT 1""", (asset_id,)).fetchone()
    con.close()
    if not row:
        return None
    path = row["absolute_path"]
    if not os.path.exists(path):
        # 2026-09-04 原图已被外部删除 → 缩略图也直接 404，让前端 <img onerror> 把卡片静默移除。
        # 旧 _thumb_fallback 会退回旧 400 缓存（磁盘残留），造成"幽灵缩略图"——欺骗用户以为照片还在，废弃。
        return None
    # 2026-09-10 二次锁死修复：回源 NAS（SMB 慢 IO）全局限流 4 并发。原实现
    # 首页洪水时几十线程同时跨 SMB 读原图，每个读事务都持 SQLite SHARED 锁，
    # delete 模式下写者被饿死 → 全站 503。
    with _THUMB_SEM:
        try:
            if row["media_type"] == "photo":
                if lc:
                    # 2026-09-23 Log 还原：必须走 Pillow——sips 没有调色能力，
                    # 且要与 ffmpeg 那条链同参数。_pil_load 自带 EXIF/HEIC 转正。
                    from PIL import Image
                    im = _logcolor_pil(_pil_load(path), lc[1])
                    im.thumbnail((edge, edge), Image.LANCZOS)
                    if im.mode not in ("RGB", "L"):
                        im = im.convert("RGB")
                    ptmp = out.with_name(out.name + f".tmp{threading.get_ident()}")
                    im.save(str(ptmp), "JPEG", quality=90)
                    os.replace(ptmp, out)
                elif SIPS_BIN:
                    # Mac 原路径：sips 不应用 EXIF orientation / HEIC irot 转正，
                    # 先走 _upright_photo（ffmpeg 全图解码转正 + 磁盘缓存）再 sips 缩放。
                    up, _uw, _uh = _upright_photo(path)
                    src = up if up else path
                    resize_to_jpeg(src, out, edge, quality=90)
                else:
                    # 2026-09-13 缩略图"横躺+发暗"修复（Linux 容器）：ffmpeg 解码部分
                    # EXIF-orientation 照片会多转 90°（人脸横躺）且色彩发暗；容器内
                    # pillow_heif 可用，_pil_load（pillow_heif + exif_transpose）方向
                    # 与色彩均正确，改为 Pillow 直读原图；失败回退 ffmpeg 转正旧路径。
                    try:
                        resize_to_jpeg(path, out, edge, quality=90)
                    except Exception:
                        up, _uw, _uh = _upright_photo(path)
                        src = up if up else path
                        resize_to_jpeg(src, out, edge, quality=90)
            else:
                # 2026-08-29 修复：大疆 HEVC non-full-range YUV 需 -strict unofficial；超短视频(<0.5s)回退第 0 帧
                # 2026-09-12 缩略图"自动放大"修复：ffmpeg 同样改为临时文件 + os.replace 原子替换
                # 2026-09-23 Log 还原：vf 链首必须 format=yuv420p（10bit 直进 eq → 全黑帧）
                import threading as _th
                if not res_admit(for_task="视频抽帧"):
                    return _thumb_fallback(asset_id)
                with _VIDEO_THUMB_SEM:      # 4K HEVC 解码串行化（见信号量定义处的血案注释）
                    vtmp = out.with_name(out.name + f".tmp{_th.get_ident()}")
                    vf = f"scale={edge}:-2"
                    if lc:
                        vf = f"format=yuv420p,{vf},{_logcolor_vf(lc[1])}"
                    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
                           "-threads", "2", "-ss", "0.5", "-i", path, "-frames:v", "1", "-vf", vf,
                           "-f", "image2", str(vtmp)]
                    try:
                        # timeout 90s：负载高时 4K 解码要 30s+，30s 会把能成功的也判死
                        _ffmpeg_run_guarded(cmd, capture_output=True, timeout=90, check=True)
                    except subprocess.CalledProcessError:
                        cmd[cmd.index("0.5")] = "0"
                        _ffmpeg_run_guarded(cmd, capture_output=True, timeout=90, check=True)
                    os.replace(vtmp, out)
            return out
        except Exception:
            return _thumb_fallback(asset_id)


def _thumb_fallback(asset_id):
    """高清档生成失败（NAS 掉线/格式问题）时，退回旧 400px 缓存。"""
    old = THUMB_DIR / f"{asset_id[6:]}.jpg"
    return old if old.exists() else None


YUNET_MODEL = FACE_MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = FACE_MODEL_DIR / "face_recognition_sface_2021dec.onnx"


_DIMS_CACHE = {}  # path -> (W,H) 进程级缓存，避免每次 face_crop 都 ffprobe


def _probe_dims_video(path):
    """探测视频「显示」像素尺寸（应用 rotation 转正后），结果缓存。

    检测器 load_video_frame 用 ffmpeg 解码时会 autorotate（Display Matrix），
    bbox 归一化坐标落在转正帧空间；而 ffmpeg -i 的 WxH 是存储尺寸，竖拍视频
    （90/270 度旋转）横竖颠倒会导致 face_crop 错位。这里同时解析 stderr 里的
    "Display Matrix: rotation of X degrees"，±90/270 时交换宽高。"""
    if not path or not os.path.exists(path):
        return (0, 0)
    key = str(path)
    if key in _DIMS_CACHE:
        return _DIMS_CACHE[key]
    w = h = 0
    rot = 0.0
    try:
        r = subprocess.run([FFMPEG_BIN, "-hide_banner", "-i", str(path)],
                           capture_output=True, text=True, timeout=10)
        for line in r.stderr.splitlines():
            if "Video:" in line:
                m = re.search(r",\s*(\d+)x(\d+)", line)
                if m:
                    w, h = int(m.group(1)), int(m.group(2))
                    break
        m = re.search(r"rotation of\s*(-?\d+(?:\.\d+)?)\s*degrees", r.stderr)
        if m:
            rot = float(m.group(1))
    except Exception:
        pass
    if w and h and int(abs(rot)) % 180 == 90:
        w, h = h, w
    _DIMS_CACHE[key] = (w, h)
    return (w, h)


_UPRIGHT_DIR = None  # 懒初始化: FACE_CROP_DIR/_upright


def _upright_photo(path):
    """照片全图转正 jpg：ffmpeg 解码整图（自动应用 EXIF orientation / HEIC irot
    转正、HEIC tile 拼接），磁盘缓存。返回 (转正jpg路径, W, H)；失败 (None,0,0)。

    2026-09-01 face_crop 两阶段修复：
    1. bbox image_width 是检测器输入尺寸（常缩到 960），与原图实际尺寸差数倍，
       直接反算像素坐标会让 crop 偏到左上角。
    2. HEIC：ffmpeg -vf 直接裁会因 tile 拼接的 complex filtergraph 冲突而失败
       （exit 234）；sips 转换又不应用转正（横竖颠倒）。改两步法：先 ffmpeg
       全图转 jpg（自动转正），再在转正图上裁。坐标空间统一为「ffmpeg 解码
       转正后」= 人脸检测器坐标空间（检测器输入即转正图，如 960x1280 竖）。"""
    global _UPRIGHT_DIR
    if not path or not os.path.exists(path):
        return (None, 0, 0)
    if _UPRIGHT_DIR is None:
        _UPRIGHT_DIR = FACE_CROP_DIR / "_upright"
        _UPRIGHT_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(str(path).encode()).hexdigest() + ".jpg"
    out = _UPRIGHT_DIR / key
    if not (out.exists() and out.stat().st_size > 0):
        # tmp 带 thread id：ThreadingHTTPServer 并发请求同一张图不互踩
        tmp = out.with_suffix(f".{threading.get_ident()}.tmp.jpg")
        ok = False
        try:
            subprocess.run([FFMPEG_BIN, "-hide_banner", "-loglevel", "error",
                            "-i", str(path), "-frames:v", "1",
                            "-f", "image2", "-y", str(tmp)],
                           capture_output=True, timeout=60, check=True)
            ok = tmp.exists() and tmp.stat().st_size > 0
        except Exception:
            ok = False
        if not ok:
            # ffmpeg 不认的格式（RAW 等）→ 兜底转换（Mac sips / 容器 Pillow+HEIC）
            try:
                resize_to_jpeg(path, tmp, 2400, quality=90)
                ok = tmp.exists() and tmp.stat().st_size > 0
            except Exception:
                ok = False
        if not ok:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return (None, 0, 0)
        shutil.move(str(tmp), str(out))
    w = h = 0
    try:
        w, h = image_dims(out)
    except Exception:
        pass
    if not (w and h):
        return (None, 0, 0)
    return (str(out), w, h)


def get_face_crop(face_instance_id, k=3.6, size=460):
    """人脸高清裁切：从原图按 bbox 中心扩展 k 倍（含整个头 + 一点身体），
    缩放到 size 输出方形 JPEG 并磁盘缓存。

    - 照片优先 ffmpeg 精确按像素 crop；ffmpeg 不认的格式（RAW 等）回退
      sips 转高分辨率 jpg 后再裁。
    - 视频抽 0.5s 帧（失败回退第 0 帧）后裁切。
    返回缓存文件路径，失败返回 None。"""
    FACE_CROP_DIR.mkdir(parents=True, exist_ok=True)
    # v5：2026-09-18 小脸铺满修复（thumbnail 永不放大导致黑边，见下），v4 缓存作废
    key = f"{face_instance_id}_k{int(round(k * 10))}_s{int(size)}_v5.jpg"
    out = FACE_CROP_DIR / key
    if out.exists():
        return out
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT asset_id, bbox_json, frame_time_seconds FROM face_instance_v0 WHERE face_instance_id=?",
                      (face_instance_id,)).fetchone()
    if not row or not row["bbox_json"]:
        con.close()
        return None
    asset_id = row["asset_id"]
    mtype = con.execute("SELECT media_type FROM media_asset WHERE asset_id=?", (asset_id,)).fetchone()
    con.close()
    b = json.loads(row["bbox_json"])
    x = float(b.get("x") or 0)
    y = float(b.get("y") or 0)
    w = float(b.get("w") or 0)
    h = float(b.get("h") or 0)
    if w <= 0 or h <= 0:
        return None
    path = get_orig_path(asset_id)
    if not path:
        return None
    im = None  # 照片分支的 Pillow 解码图（视频为 None）
    is_video = mtype and mtype["media_type"] == "video"
    if is_video:
        base = path
        W, H = _probe_dims_video(path)
    else:
        # 2026-09-14 色彩修复：照片改 Pillow 直读（pillow_heif + exif_transpose）。
        # 原 ffmpeg/_upright 路径解码部分 HEIC 会发暗/发灰/近全黑（与 t1600 缩略图
        # 同根因，抽样 11 张照片 4 张异常），不再经 ffmpeg，直接在 Pillow 解码图上裁。
        # 2026-09-14 坐标空间对齐：检测器（backfill_faces_full.load_photo）对 HEIC 走
        # sips 转正后检测，对其它格式直接 cv2.imread（不应用 EXIF orientation）——
        # 裁切空间必须与检测器一致：HEIC 转正（pillow_heif 已应用 irot，exif_transpose
        # 兜底 EXIF），非 HEIC 不转正，否则 EXIF 旋转 JPEG 的 bbox 会裁到错误区域。
        try:
            from PIL import Image as _PILImage, ImageOps as _PILImageOps
            try:
                import pillow_heif as _pheif
                _pheif.register_heif_opener()
            except ImportError:
                pass
            im = _PILImage.open(str(path))
            if os.path.splitext(path)[1].lower() in (".heic", ".heif"):
                im = _PILImageOps.exif_transpose(im)
        except Exception:
            im = None
        if im is None:
            return None
        W, H = im.size
    if not (W and H):
        # 2026-09-14 审计修复：探测失败不再回退 bbox 记录尺寸——视频抽帧坐标在
        # 探测失败的原始尺寸上必然错位，宁可裁不出（可重试）也不给错图
        return None
    if W <= 0 or H <= 0:
        return None
    cx = (x + w / 2) * W
    cy = (y + h / 2) * H
    side = max(w * W, h * H) * k
    # 2026-09-10 质控修复：统一取 side_px 方框（中心不变），输出恒为 size×size。
    # 2026-09-17 比例修复：旧逻辑用 max(0,..)/min(W,..) 把取景框 clip 到图片边缘，
    # 贴边人脸会裁出 cw≠ch 的非方形区域，只能靠 pad 黑边补方形 →
    # 脸贴边 / 脸占画面比例大时，裁切图左右或上下出现大黑条（用户报「图像比例有问题」）。
    # 现在：取景框最大收到图片短边（任何图片内部必然存在这么大的正方形），
    # 再整体平移进图片内 —— 输出恒为真方形、零黑边；脸未必正居中，但绝不裁丢、绝无黑边。
    side_px = max(8, int(round(side)))
    side_px = min(side_px, min(W, H))
    x0 = int(round(cx - side_px / 2))
    y0 = int(round(cy - side_px / 2))
    x0 = max(0, min(x0, W - side_px))
    y0 = max(0, min(y0, H - side_px))
    x1 = x0 + side_px
    y1 = y0 + side_px
    cw = ch = side_px
    # 输出恒为方形：保比缩放 + 居中 pad（cw==ch 时 pad 无操作）
    # cw==ch 恒为正方形 → scale 精确到 size×size（原 decrease+pad 对小于 size 的
    # 抽帧区域同样不放大、留黑边，与照片分支同病，一并修掉）
    VF_SQUARE = (
        f"crop={cw}:{ch}:{x0}:{y0},"
        f"scale={size}:{size}"
    )
    # tmp 带 thread id：ThreadingHTTPServer 并发生成不互踩
    tmp = out.with_suffix(f".{threading.get_ident()}.raw.jpg")
    try:
        if is_video:
            # 抽检测帧的真实时间（检测器在 10%/50%/90% 采样，而非固定 0.5s）；
            # frame_time_seconds 缺失/非法时回退 0.5s，抽帧失败再回退第 0 帧。
            try:
                seek = max(0.0, float(row["frame_time_seconds"]))
            except (TypeError, ValueError):
                seek = 0.5
            seek_s = f"{seek:.3f}"
            cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
                   "-ss", seek_s, "-i", str(base), "-vf", VF_SQUARE,
                   "-frames:v", "1", "-f", "image2", str(tmp)]
            try:
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)
            except subprocess.CalledProcessError:
                cmd[cmd.index(seek_s)] = "0"
                subprocess.run(cmd, capture_output=True, timeout=30, check=True)
        else:
            # 2026-09-14：照片 Pillow 裁切 + 居中 pad 黑边（与视频分支 pad=black 一致）
            from PIL import Image
            region = im.crop((x0, y0, x1, y1)).convert("RGB")
            # 2026-09-18 小脸铺满修复：region 恒为正方形（cw==ch），但 thumbnail **永不放大** ——
            # 小于 size 的小脸裁切（远景/合影里几十像素的脸）会原尺寸贴到黑画布中央，
            # 四周一大圈黑边（用户报「裁切还是什么问题」的直接根因）。
            # 改 resize 强制铺满 size×size：小脸会糊，但铺满无黑边；region 是正方形，无变形。
            region = region.resize((size, size), Image.LANCZOS)
            canvas = Image.new("RGB", (size, size), (0, 0, 0))
            canvas.paste(region, (0, 0))
            canvas.save(str(tmp), "JPEG", quality=90)
    except Exception:
        pass
    if tmp.exists() and tmp.stat().st_size > 0:
        try:
            shutil.move(str(tmp), str(out))
            return out
        except Exception:
            return None
    return None


def get_preview(asset_id):
    """生成浏览器兼容的大图预览；只写本地缓存，不修改 NAS 原片。
    2026-09-23：Log 原片在这里同样过一遍还原，缓存名带 _lc 后缀隔离。"""
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    lc = logcolor_for(asset_id)
    stem = asset_id[6:]
    out = PREVIEW_DIR / (f"{stem}_lc{lc[2]}.jpg" if lc else f"{stem}.jpg")
    if out.exists():
        return out
    path = get_orig_path(asset_id)
    if not path:
        return None
    if lc:
        # 2026-09-23 修复：视频走 ffmpeg 抽帧（PIL 打不开 HEVC/MOV，之前落到
        # 未还原回退路径 → 视频永远 404）。与 get_thumb 视频分支同构：
        # 链首必须 format=yuv420p（10bit 直进 eq → 全黑帧），失败回退第 0 帧。
        try:
            _con = sqlite3.connect(DB, timeout=10)
            _mt = _con.execute("SELECT media_type FROM media_asset WHERE asset_id=?",
                               (asset_id,)).fetchone()
            _con.close()
            if _mt and _mt[0] == "video":
                if not res_admit(for_task="视频预览抽帧"):
                    return None
                with _VIDEO_THUMB_SEM:   # 与 get_thumb 视频分支同一把串行锁
                    vtmp = out.with_name(out.name + f".tmp{threading.get_ident()}")
                    vf = f"format=yuv420p,scale=2200:-2,{_logcolor_vf(lc[1])}"
                    cmd = [FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
                           "-threads", "2", "-ss", "0.5", "-i", path, "-frames:v", "1", "-vf", vf,
                           "-q:v", "2", "-f", "image2", str(vtmp)]
                    try:
                        _ffmpeg_run_guarded(cmd, capture_output=True, timeout=120, check=True)
                    except subprocess.CalledProcessError:
                        cmd[cmd.index("0.5")] = "0"
                        _ffmpeg_run_guarded(cmd, capture_output=True, timeout=120, check=True)
                    os.replace(vtmp, out)
                return out
        except Exception:
            pass        # 视频抽帧失败：退到下面的未还原回退，宁可灰片也不能 404
        try:
            from PIL import Image
            im = _logcolor_pil(_pil_load(path), lc[1])
            im.thumbnail((2200, 2200), Image.LANCZOS)
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            ptmp = out.with_name(out.name + f".tmp{threading.get_ident()}")
            im.save(str(ptmp), "JPEG", quality=90)
            os.replace(ptmp, out)
            return out
        except Exception:
            pass        # 还原失败退回未还原预览，宁可灰片也不能 404
        out = PREVIEW_DIR / f"{stem}.jpg"
        if out.exists():
            return out
    try:
        resize_to_jpeg(path, out, 2200, quality=90)
        return out if out.exists() else None
    except Exception:
        return None


# ============ 从上传照片/视频新建人物 ============
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _ensure_upload_source(con):
    """返回用于人脸上传的 source_id，不存在则创建。"""
    sid = "src_face_uploads"
    root = str(UPLOAD_DIR)
    row = con.execute("SELECT source_id FROM source WHERE source_id=?", (sid,)).fetchone()
    if not row:
        now = datetime.now(timezone.utc).isoformat()
        con.execute("""INSERT INTO source(source_id,family_id,owner_label,root_path,source_type,read_only,last_scan_at,enabled,total_files,indexed_count,failed_count)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
            (sid, "family_default", "人脸上传", root, "local_folder", 0, now, 1, 0, 0, 0))
    return sid


def _load_upload_frame(path, media_type):
    """读取上传文件：照片返回图像；视频抽 0.5s 帧（失败回退第 0 帧）。"""
    if media_type == "photo":
        return _get_bff().load_photo(str(path), Path(path).suffix)
    try:
        return _get_bff().load_video_frame(str(path), 0.5)
    except Exception:
        try:
            return _get_bff().load_video_frame(str(path), 0.0)
        except Exception:
            return None


def _detect_faces_upload(image):
    """对上传图像做 YuNet 检测，返回 [(face_array, score), ...]。"""
    import cv2
    import numpy as np
    if image is None:
        return []
    h, w = image.shape[:2]
    detector = cv2.FaceDetectorYN_create(str(YUNET_MODEL), "", (w, h), 0.6, 0.3, 5000)
    _, faces = detector.detect(image)
    if faces is None:
        return []
    out = []
    for face in faces:
        arr = face.astype(np.float32)
        score = float(arr[-1])
        out.append((arr, score))
    return out


def _norm_bbox(face_arr, w, h):
    """把 YuNet 输出转成人脸表存储的归一化 bbox/landmarks。"""
    x, y, fw, fh = map(float, face_arr[:4])
    box = {"x": x / w, "y": y / h, "w": fw / w, "h": fh / h, "image_width": w, "image_height": h}
    landmarks = [{"x": float(face_arr[i]) / w, "y": float(face_arr[i + 1]) / h} for i in range(4, 14, 2)]
    return box, landmarks


def _face_preview_base64(image, face_arr, size=240):
    """生成单个人脸的方形预览图 base64。"""
    import cv2
    h, w = image.shape[:2]
    x, y, fw, fh = map(int, face_arr[:4])
    cx, cy = x + fw / 2, y + fh / 2
    side = max(fw, fh) * 1.6
    half = side / 2
    x0, y0 = max(0, int(cx - half)), max(0, int(cy - half))
    x1, y1 = min(w, int(cx + half)), min(h, int(cy + half))
    crop = image[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    scale = size / max(cw, ch)
    if scale < 1:
        crop = cv2.resize(crop, (int(cw * scale), int(ch * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _extract_embedding_upload(image, face_arr):
    """对上传图像中的人脸提取 SFace embedding；失败返回 None。"""
    import cv2
    import numpy as np
    try:
        recognizer = cv2.FaceRecognizerSF_create(str(SFACE_MODEL), "")
        aligned = recognizer.alignCrop(image, face_arr)
        emb = recognizer.feature(aligned).astype(np.float32).reshape(-1)
        norm = float(np.linalg.norm(emb))
        emb = emb / norm if norm else emb
        return emb, norm
    except Exception as exc:
        print(f"[upload-face] embedding failed: {exc}", flush=True)
        return None, None


def _save_uploaded_file(filename, body):
    """保存上传文件到 UPLOAD_DIR，返回 (upload_id, path, media_type)。"""
    # 2026-09-14 审计修复：客户端文件名只取 basename，防 ../ 路径穿越
    filename = os.path.basename(str(filename or "upload"))
    ext = Path(filename).suffix.lower()
    media_type = "video" if ext in {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"} else "photo"
    upload_id = "upload_" + hashlib.sha256(f"{filename}:{time.time()}:{uuid.uuid4()}".encode()).hexdigest()[:24]
    save_dir = UPLOAD_DIR / upload_id
    save_dir.mkdir(parents=True, exist_ok=True)
    saved_path = save_dir / filename
    with open(saved_path, "wb") as f:
        f.write(body)
    return upload_id, saved_path, media_type


def _persist_upload_asset(con, source_id, saved_path, media_type, w, h, duration=None):
    """把上传文件登记为 media_asset + media_file，返回 asset_id。"""
    token = hashlib.sha256(str(saved_path).encode("utf-8")).hexdigest()[:24]
    asset_id, file_id = "asset_" + token, "file_" + token
    now = datetime.now(timezone.utc).isoformat()
    ext = saved_path.suffix.lower()
    st = saved_path.stat()
    con.execute("""INSERT OR IGNORE INTO media_asset
        (asset_id,family_id,media_type,capture_time,time_precision,time_confidence,
         latitude,longitude,location_confidence,width,height,duration_seconds,is_original,privacy_level,created_at,updated_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (asset_id, "family_default", media_type, now, "second", 0.5,
         None, None, None, w, h, duration, 1, "private", now, now))
    con.execute("""INSERT OR IGNORE INTO media_file
        (file_id,asset_id,source_id,absolute_path,relative_path,filename,extension,byte_size,filesystem_mtime,mime_type,variant_kind,availability,indexed_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (file_id, asset_id, source_id, str(saved_path), saved_path.name, saved_path.name, ext, st.st_size,
         datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
         mimetypes.guess_type(str(saved_path))[0], "original", "online", now))
    return asset_id


# 上传会话缓存：upload_id -> {path, media_type, image, faces, w, h}
_UPLOAD_SESSIONS = {}


def handle_face_upload_preview(handler):
    """处理 POST /api/face/upload-preview：保存文件、检测人脸、返回预览。"""
    ctype = handler.headers.get("Content-Type", "")
    if not ctype.startswith("multipart/form-data"):
        return {"error": "请使用 multipart/form-data 上传文件"}, 400
    form = cgi.FieldStorage(fp=handler.rfile, headers=handler.headers, environ={"REQUEST_METHOD": "POST"})
    if "file" not in form:
        return {"error": "缺少文件字段 file"}, 400
    field = form["file"]
    filename = field.filename or "upload.jpg"
    body = field.file.read()
    if not body:
        return {"error": "文件为空"}, 400
    upload_id, saved_path, media_type = _save_uploaded_file(filename, body)
    image = _load_upload_frame(saved_path, media_type)
    if image is None:
        return {"error": "无法读取该文件（照片损坏/视频抽帧失败）"}, 400
    h, w = image.shape[:2]
    faces = _detect_faces_upload(image)
    _UPLOAD_SESSIONS[upload_id] = {
        "path": str(saved_path), "media_type": media_type,
        "image": image, "faces": faces, "w": w, "h": h, "duration": None
    }
    previews = []
    for idx, (face_arr, score) in enumerate(faces):
        box, _ = _norm_bbox(face_arr, w, h)
        preview = _face_preview_base64(image, face_arr)
        previews.append({"index": idx, "score": round(score, 3), "bbox": box, "preview": preview})
    return {
        "upload_id": upload_id,
        "media_type": media_type,
        "width": w, "height": h,
        "faces": previews
    }, 200


def handle_face_create_from_upload(body):
    """处理 POST /api/face/create-from-upload：根据上传人脸创建/更新人物。"""
    upload_id = (body.get("upload_id") or "").strip()
    face_index = int(body.get("face_index", -1))
    display_name = (body.get("display_name") or "").strip()
    person_id = (body.get("person_id") or "").strip()
    if not upload_id or upload_id not in _UPLOAD_SESSIONS:
        return {"error": "上传会话已过期，请重新上传"}, 400
    if face_index < 0 or face_index >= len(_UPLOAD_SESSIONS[upload_id]["faces"]):
        return {"error": "未选择有效人脸"}, 400
    if not person_id and not display_name:
        return {"error": "请输入新人物姓名或选择现有人物"}, 400
    sess = _UPLOAD_SESSIONS[upload_id]
    image = sess["image"]
    face_arr, score = sess["faces"][face_index]
    h, w = sess["h"], sess["w"]
    box, landmarks = _norm_bbox(face_arr, w, h)
    emb, norm = _extract_embedding_upload(image, face_arr)
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.isolation_level = None
    try:
        source_id = _ensure_upload_source(con)
        asset_id = _persist_upload_asset(con, source_id, Path(sess["path"]), sess["media_type"], w, h, sess.get("duration"))
        now = datetime.now(timezone.utc).isoformat()
        if person_id:
            existing = con.execute("SELECT 1 FROM person WHERE person_id=?", (person_id,)).fetchone()
            if not existing:
                return {"error": "选择的人物不存在"}, 400
        else:
            person_id = "person_" + hashlib.sha256(f"{display_name}:{now}".encode()).hexdigest()[:24]
            con.execute("""INSERT INTO person(person_id,family_id,display_name,relationship_label,birth_date,identity_status,created_at,updated_at)
                VALUES(?,?,?,?,?,?,?,?)""",
                (person_id, "family_default", display_name, None, None, "confirmed", now, now))
        x, y, fw, fh = map(float, face_arr[:4])
        blur, pose, qclass = _get_bff().quality(face_arr, image)
        fid = "face_" + hashlib.sha256(f"{asset_id}:upload:{now}:{x:.4f}:{y:.4f}".encode()).hexdigest()[:24]
        frame_time = None if sess["media_type"] == "photo" else 0.5
        con.execute("""INSERT INTO face_instance_v0
            (face_instance_id,asset_id,frame_time_seconds,frame_key,bbox_json,landmarks_json,
             detection_score,face_width,face_height,blur_score,pose_score,quality_class,sample_role,detection_model,created_at,person_id,person_locked)
            VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (fid, asset_id, frame_time, f"{asset_id}:upload", json.dumps(box), json.dumps(landmarks),
             score, round(fw), round(fh), blur, pose, qclass, "manual", "YuNet-2023mar", now, person_id, 1))
        if emb is not None:
            con.execute("INSERT INTO face_embedding_v0 VALUES(?,?,?,?,?,?,?)",
                        (fid, "SFace-2021dec", len(emb), emb.astype("<f4").tobytes(), norm, "success", now))
        # 把新建/更新的人物头像指向这张脸
        con.execute("UPDATE person SET avatar_face_id=?, updated_at=? WHERE person_id=?", (fid, now, person_id))
        con.execute("UPDATE source SET indexed_count=indexed_count+1, total_files=total_files+1 WHERE source_id=?", (source_id,))
        return {"ok": True, "person_id": person_id, "face_instance_id": fid, "asset_id": asset_id}, 200
    except Exception as exc:
        print(f"[api-error] /api/face/create-from-upload: {exc}", flush=True)
        return {"error": str(exc)}, 400
    finally:
        con.close()
        _UPLOAD_SESSIONS.pop(upload_id, None)


def get_face_positions():
    """人脸优先裁切锚点：按资产聚合人脸框中心（检测分加权），归一化 0-1。
    前端竖版照片裁 16:9 时用它定位 object-position，保证人脸不被裁掉。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT asset_id, bbox_json, detection_score FROM face_instance_v0
           WHERE bbox_json IS NOT NULL""").fetchall()
    con.close()
    # 2026-09-14 隐私/回收站门：隐藏资产的锚点不下发（墙面上本来就不显示）
    privacy, recycle = _hidden_media_ids()
    hidden = privacy | recycle
    acc = {}
    for r in rows:
        if r["asset_id"] in hidden:
            continue
        try:
            b = json.loads(r["bbox_json"])
            cx = b["x"] + b["w"] / 2
            cy = b["y"] + b["h"] / 2
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                continue
            w = max(0.05, r["detection_score"] or 0.5)
            a = acc.setdefault(r["asset_id"], [0.0, 0.0, 0.0])
            a[0] += cx * w; a[1] += cy * w; a[2] += w
        except Exception:
            continue
    out = {}
    for aid, (sx, sy, sw) in acc.items():
        if sw > 0:
            # 收敛到 [0.12, 0.88]，避免锚点贴边导致裁切过偏
            out[aid] = [min(0.88, max(0.12, sx / sw)), min(0.88, max(0.12, sy / sw))]
    return {"count": len(out), "anchor": out}


# ============ 过滤表管理（墙面垃圾内容：截图/快递单/收据/证件等） ============
# 设计：asset_filter_v0 为唯一事实表；asset_allowlist_v0 为用户白名单——
# 从管理页"恢复"的资产永久豁免，即使重跑确定性过滤脚本也不会再被误杀。
FILTER_REASON_LABELS = {
    "CHAT_EXPORT_FILENAME": "聊天导出", "CHAT_EXPORT_PATH": "聊天导出",
    "SCREENSHOT_FILENAME": "手机截图", "SCREENSHOT_PATH": "手机截图",
    "SCREEN_RECORDING_FILENAME": "录屏", "DOWNLOAD_PATH": "下载目录",
    "VISION_SCREENSHOT": "视觉识别·截图", "VISION_DOCUMENT": "视觉识别·文档",
    "MANUAL": "手动移出",
    # 2026-09-15 T4 接线：补齐四个离线模块回填的 reason（原来直接漏出英文枚举）
    "JUNK_QR_UNDECODED": "二维码", "JUNK_QR_PAYMENT_WECHAT": "微信收款码",
    "JUNK_QR_PAYMENT_ALIPAY": "支付宝收款码",
    "SHORT_VIDEO_SAVED": "短视频保存件", "ECOMMERCE_SHARE_FILENAME": "电商分享图",
    "WATERMARK_CAMERA_FILENAME": "相机水印图",
}

def get_recycled_ids(con):
    """回收站集合：用户「软删除」的照片，从所有正常检索路径排除。"""
    try:
        return {r[0] for r in con.execute("SELECT asset_id FROM recycle_v0")}
    except sqlite3.OperationalError:
        return set()


def _privacy_has_rows(con):
    """隐私相册是否有内容（决定 SQL 里要不要拼 privacy 排除条件）。"""
    try:
        return con.execute("SELECT COUNT(*) FROM privacy_v0").fetchone()[0] > 0
    except sqlite3.OperationalError:
        return False


def exclude_recycled(con, assets):
    """从 assets 列表中剔除回收站 + 隐私相册照片（兼容 id / asset_id 两种字段）。"""
    rid = get_recycled_ids(con)
    try:
        rid |= {r[0] for r in con.execute("SELECT asset_id FROM privacy_v0")}
    except sqlite3.OperationalError:
        pass
    if not rid:
        return assets
    out = []
    for a in assets:
        if not isinstance(a, dict):
            out.append(a)
            continue
        aid = a.get("id") or a.get("asset_id")
        if aid not in rid:
            out.append(a)
    return out


# ---- 媒体端点隐私/回收站门（2026-09-14 审计 P0 修复）----
# /thumb /orig /preview /face_crop /api/facepos 此前不查 privacy_v0/recycle_v0：
# 任何登录用户拿到 asset_id 即可绕过 PIN 直读隐私照片或已过滤照片的原图。
# 策略：隐私资产 = PIN 验证通过后 30 分钟内的会话才可读（对 admin 同样要求 PIN）；
#       回收站/已过滤资产 = admin/family 可读（过滤管理页需要），guest 拒绝。
_MEDIA_HIDDEN_CACHE = {"t": 0.0, "privacy": set(), "recycle": set()}
_MEDIA_HIDDEN_LOCK = threading.Lock()
_PRIVACY_UNLOCK = {}          # session token -> 解锁过期时间戳
_PRIVACY_UNLOCK_TTL = 1800.0


def _hidden_media_ids():
    now = time.time()
    with _MEDIA_HIDDEN_LOCK:
        if now - _MEDIA_HIDDEN_CACHE["t"] > 3.0:
            c = sqlite3.connect(DB, timeout=10)
            try:
                try:
                    _MEDIA_HIDDEN_CACHE["privacy"] = {r[0] for r in c.execute("SELECT asset_id FROM privacy_v0")}
                except sqlite3.OperationalError:
                    _MEDIA_HIDDEN_CACHE["privacy"] = set()
                try:
                    _MEDIA_HIDDEN_CACHE["recycle"] = {r[0] for r in c.execute("SELECT asset_id FROM recycle_v0")}
                except sqlite3.OperationalError:
                    _MEDIA_HIDDEN_CACHE["recycle"] = set()
                _MEDIA_HIDDEN_CACHE["t"] = now
            finally:
                c.close()
        return _MEDIA_HIDDEN_CACHE["privacy"], _MEDIA_HIDDEN_CACHE["recycle"]


def _current_session_token(handler):
    try:
        _, token = _current_session_user(handler)
        return token
    except Exception:
        return None


def _privacy_session_unlocked(handler):
    tok = _current_session_token(handler)
    if not tok:
        return False
    exp = _PRIVACY_UNLOCK.get(tok, 0)
    if exp >= time.time():
        return True
    _PRIVACY_UNLOCK.pop(tok, None)
    return False


def _privacy_mark_unlocked(handler):
    tok = _current_session_token(handler)
    if not tok:
        return
    now = time.time()
    for k in [k for k, v in _PRIVACY_UNLOCK.items() if v < now]:
        _PRIVACY_UNLOCK.pop(k, None)
    _PRIVACY_UNLOCK[tok] = now + _PRIVACY_UNLOCK_TTL


def media_gate_allows(handler, asset_id):
    """媒体端点统一门：True=放行。不抛异常，失败默认按隐藏处理需显式放行。"""
    if not asset_id:
        return True
    privacy, recycle = _hidden_media_ids()
    if asset_id in privacy:
        return _privacy_session_unlocked(handler)
    if asset_id in recycle:
        user, _ = _current_session_user(handler)
        return (user or {}).get("role") in ("admin", "family")
    return True

def get_filter_list(order='desc'):
    """过滤表完整清单（管理页用）：每资产聚合全部命中原因。
    order: 'desc' 最新在前（默认），'asc' 最旧在前。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    rows = con.execute("""SELECT f.asset_id, f.filter_reason, f.evidence_kind, f.confidence,
        ma.capture_time, ma.media_type FROM asset_filter_v0 f
        LEFT JOIN media_asset ma USING(asset_id)""").fetchall()
    con.close()
    assets = {}
    for r in rows:
        a = assets.setdefault(r["asset_id"], {
            "id": r["asset_id"], "time": r["capture_time"],
            "type": r["media_type"] or "photo", "reasons": [], "conf": 0.0})
        a["reasons"].append({
            "reason": r["filter_reason"],
            "label": FILTER_REASON_LABELS.get(r["filter_reason"], r["filter_reason"]),
            "kind": r["evidence_kind"], "confidence": r["confidence"]})
        a["conf"] = max(a["conf"], r["confidence"] or 0.0)
    reverse = str(order).lower() != 'asc'
    out = sorted(assets.values(), key=lambda x: x["time"] or "", reverse=reverse)
    return {"assets": out, "total": len(out)}

def filter_add(items):
    """加入过滤表（手动移出墙面 / Vision 批量导入）。
    items: [{asset_id, reason?, confidence?, rule_version?}]；同时清白名单。"""
    con = sqlite3.connect(DB, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS asset_allowlist_v0 (
        asset_id TEXT PRIMARY KEY, note TEXT, created_at TEXT)""")
    now = now_iso()
    n = 0
    for it in items:
        aid = (it.get("asset_id") or "").strip()
        if not aid:
            continue
        con.execute("DELETE FROM asset_allowlist_v0 WHERE asset_id=?", (aid,))
        con.execute("""INSERT OR IGNORE INTO asset_filter_v0
            (asset_id, filter_reason, evidence_kind, evidence_value, confidence, rule_version, created_at)
            VALUES (?,?,?,?,?,?,?)""",
            (aid, it.get("reason", "MANUAL"), it.get("evidence_kind", "user"),
             it.get("evidence_value", "user-action"), float(it.get("confidence", 1.0)),
             it.get("rule_version", "manual-v1"), now))
        n += 1
    con.commit()
    total = con.execute("SELECT count(DISTINCT asset_id) FROM asset_filter_v0").fetchone()[0]
    con.close()
    return {"added": n, "total_filtered": total}

def filter_remove(asset_ids):
    """恢复资产到墙面：删过滤行 + 写白名单防复杀。"""
    con = sqlite3.connect(DB, timeout=10)
    con.execute("""CREATE TABLE IF NOT EXISTS asset_allowlist_v0 (
        asset_id TEXT PRIMARY KEY, note TEXT, created_at TEXT)""")
    now = now_iso()
    for aid in asset_ids:
        con.execute("DELETE FROM asset_filter_v0 WHERE asset_id=?", (aid,))
        con.execute("INSERT OR REPLACE INTO asset_allowlist_v0 (asset_id, note, created_at) VALUES (?,?,?)",
                    (aid, "user-restore", now))
    con.commit()
    total = con.execute("SELECT count(DISTINCT asset_id) FROM asset_filter_v0").fetchone()[0]
    con.close()
    return {"removed": len(asset_ids), "total_filtered": total}


def get_filter_count():
    """轻量过滤计数（侧栏徽标用，不取明细）。auto=规则自动判定，manual=手动移出。"""
    con = sqlite3.connect(DB, timeout=10)
    try:
        total = con.execute("SELECT count(DISTINCT asset_id) FROM asset_filter_v0").fetchone()[0]
        auto = con.execute("SELECT count(DISTINCT asset_id) FROM asset_filter_v0 WHERE rule_version LIKE 'auto-%'").fetchone()[0]
    except sqlite3.OperationalError:
        total = auto = 0
    con.close()
    return {"total": total, "auto": auto, "manual": max(0, total - auto)}


def _classify_import_junk(new_meta):
    """导入时确定性垃圾判定：仅文件名/路径规则，零图片解码，低误杀。
    与 asset_filter_v0 现有 reason 语义一致。new_meta: [(asset_id, filename, relative_path), ...]
    返回 filter_add 兼容的 items 列表（每个命中 reason 一行）。"""
    items = []
    for aid, fn, rp in new_meta:
        fn_l = (fn or "").lower()
        rp_l = (rp or "").lower()
        reasons = []
        # 截图：文件名 Screenshot_ / 截图 / 屏幕截图 / screencapture / screen_
        if fn_l.startswith("screenshot") or "截图" in fn_l or "屏幕截图" in fn_l \
           or "screencapture" in fn_l or fn_l.startswith("screen_"):
            reasons.append("SCREENSHOT_FILENAME")
        # 截图：路径在 Screenshots / 截图 / 截屏 / screenshot 目录
        if "/screenshots/" in rp_l or "截图" in rp_l or "截屏" in rp_l or "screenshot" in rp_l:
            reasons.append("SCREENSHOT_PATH")
        # 2026-09-16 修复：聊天导出规则整体下线。实测 1208 张 mmexport/wx_camera 照片
        # sha256 与墙面可见照片重复数为 0（全是唯一副本），16% 有人脸——按路径/文件名
        # 隐藏会藏掉家庭照片的唯一一份，代价不对称。聊天来源的照片照常上墙。
        # if "mmexport" in fn_l or "wx_camera" in fn_l:
        #     reasons.append("CHAT_EXPORT_FILENAME")
        # if "micromsg" in rp_l or "wechat" in rp_l:
        #     reasons.append("CHAT_EXPORT_PATH")
        # 下载目录
        if "/download" in rp_l or "下载" in rp_l or "baidunetdisk" in rp_l \
           or rp_l.endswith("/downloads") or rp_l.endswith("/download"):
            reasons.append("DOWNLOAD_PATH")
        # 录屏
        if "screenrecording" in fn_l or "录屏" in fn_l or "screen record" in fn_l:
            reasons.append("SCREEN_RECORDING_FILENAME")
        for rsn in dict.fromkeys(reasons):
            items.append({"asset_id": aid, "reason": rsn, "evidence_kind": "rule",
                          "evidence_value": f"{fn}|{rp}", "confidence": 1.0,
                          "rule_version": "auto-junk-v1"})
    return items


# ============ 相似照片择优（连拍折叠） ============
# 设计：同一人/同设备短时间（默认 5 秒）内连续拍摄的照片视为一组，
# 按人脸质量、笑容、人脸占比、分辨率自动选最佳展示；其余折叠。
# 表 asset_similar_group_v0 / asset_similar_member_v0 记录分组结果；
# 用户可在 library.html 的「相似照片」专区手动切换最佳或解散分组。
SIMILAR_TIME_WINDOW_SECONDS = 5
QUALITY_CLASS_RANK = {'usable': 3, 'nonfrontal_candidate': 2, 'small': 1, 'blurry': 0}


def _ensure_similar_tables(con):
    con.execute("""CREATE TABLE IF NOT EXISTS asset_similar_group_v0 (
        group_id TEXT PRIMARY KEY,
        best_asset_id TEXT NOT NULL,
        source_id TEXT,
        start_time TEXT,
        end_time TEXT,
        asset_count INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        engine TEXT
    )""")
    # 2026-09-15 T4：老库补 engine 列（migrations/004 加过，但新建库要跟上；
    # engine 用于区分算法来源：dual-v1=SigLIP2+phash 离线去重，timewin-v1=时间窗连拍分组）
    cols = {r[1] for r in con.execute("PRAGMA table_info(asset_similar_group_v0)")}
    if "engine" not in cols:
        con.execute("ALTER TABLE asset_similar_group_v0 ADD COLUMN engine TEXT")
    con.execute("""CREATE TABLE IF NOT EXISTS asset_similar_member_v0 (
        group_id TEXT NOT NULL REFERENCES asset_similar_group_v0(group_id),
        asset_id TEXT NOT NULL REFERENCES media_asset(asset_id),
        pick_score REAL NOT NULL DEFAULT 0,
        is_best INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(group_id, asset_id)
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_asm_asset ON asset_similar_member_v0(asset_id)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_asg_best ON asset_similar_group_v0(best_asset_id)")


def _parse_capture_time(t):
    if not t:
        return None
    try:
        dt = datetime.fromisoformat(t)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
        return dt
    except Exception:
        return None


def _score_photo(asset_id, row, faces, happiness):
    """单张连拍照片的综合质量分；越高越值得展示。"""
    w, h = (row["width"] or 0), (row["height"] or 0)
    area = max(w * h, 1)
    score = 0.0

    face_area = 0
    n_usable = n_blurry = n_frontal = 0
    for f in faces:
        qc = f["quality_class"]
        rank = QUALITY_CLASS_RANK.get(qc, 0)
        score += rank * 1.2
        if qc == 'usable':
            n_usable += 1
        if qc == 'blurry':
            n_blurry += 1
        if qc in ('usable', 'nonfrontal_candidate'):
            n_frontal += 1
        face_area += (f["face_width"] or 0) * (f["face_height"] or 0)

    # 有人脸且清晰的照片优先；全糊的扣分
    if faces:
        score += n_usable * 2.5
        score -= n_blurry * 2.0
        score += (face_area / area) * 6.0
        # 笑容是强信号（0~1 区间，权重 4 足够把笑脸拉到前列）
        score += (happiness or 0.0) * 4.0
    else:
        # 无人脸的风景/静物：看全局质量与分辨率
        score += 1.0

    # 整体分辨率与质量
    score += math.log(area + 1) * 0.4
    if row.get("quality_score"):
        score += row["quality_score"] * 0.5

    return round(score, 4)


def _photo_rows_for_grouping(con):
    """返回可参与分组的影像：有 capture_time 的照片，按实际时间排序（处理混用时区）。"""
    rows = con.execute("""
        SELECT ma.asset_id, ma.capture_time, ma.width, ma.height, ma.quality_score, mf.source_id
        FROM media_asset ma
        JOIN media_file mf ON mf.asset_id = ma.asset_id
        WHERE ma.media_type = 'photo'
          AND ma.capture_time IS NOT NULL
          AND ma.time_precision = 'second'
        GROUP BY ma.asset_id
    """).fetchall()
    # 数据库里 capture_time 时区不统一（+08:00 / +00:00 / naive），按解析后的 aware datetime 排序
    rows = sorted(rows, key=lambda r: (
        r["source_id"] or "",
        _parse_capture_time(r["capture_time"]) or datetime.min.replace(tzinfo=timezone.utc)
    ))
    return rows


def _load_faces_and_happiness(con, asset_ids):
    faces = {}
    for r in con.execute(
        """SELECT asset_id, quality_class, face_width, face_height
           FROM face_instance_v0 WHERE asset_id IN ({})""".format(",".join("?" * len(asset_ids))),
        tuple(asset_ids)
    ):
        faces.setdefault(r["asset_id"], []).append({
            "quality_class": r["quality_class"],
            "face_width": r["face_width"],
            "face_height": r["face_height"],
        })
    happiness = {}
    if asset_ids:
        for r in con.execute(
            """SELECT fi.asset_id, MAX(fx.happiness_score) h
               FROM face_expression_v0 fx
               JOIN face_instance_v0 fi USING(face_instance_id)
               WHERE fx.inference_status='success'
                 AND fi.asset_id IN ({})
               GROUP BY fi.asset_id""".format(",".join("?" * len(asset_ids))),
            tuple(asset_ids)
        ):
            happiness[r["asset_id"]] = float(r["h"])
    return faces, happiness


def compute_similar_groups(time_window=SIMILAR_TIME_WINDOW_SECONDS, min_group_size=2,
                           engine="timewin-v1"):
    """扫描全库照片，按同来源短时间窗口分组并自动选优。

    2026-09-15 T4：**删除范围按 engine 收敛**。旧实现无条件 `DELETE FROM asset_similar_group_v0`
    全表清空 —— 会把离线去重模块（similar_dedup.py）回填的 1,959 组 dual-v1 结果一起抹掉，
    且新写入的行没有 engine 标记，看起来仍像 dual-v1。现在只重算自己这一档。
    """
    con = sqlite3.connect(DB, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=60000")
    _ensure_similar_tables(con)

    from collections import defaultdict
    rows = _photo_rows_for_grouping(con)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source_id"] or ""].append(r)

    groups = []
    for src_rows in by_source.values():
        current = []
        current_start = None
        prev_time = None
        for r in src_rows:
            t = _parse_capture_time(r["capture_time"])
            if not t:
                continue
            # 新开组条件：来源变化、间隔超过窗口、或连拍总跨度超过 60 秒
            span = (t - current_start).total_seconds() if current_start else 0
            if (current and prev_time is not None and
                    ((t - prev_time).total_seconds() <= time_window) and span <= 60):
                current.append(r)
            else:
                if len(current) >= min_group_size:
                    groups.append(current[:])
                current = [r]
                current_start = t
            prev_time = t
        if len(current) >= min_group_size:
            groups.append(current[:])

    # 清除旧分组，重新计算（只清本 engine 档，保住另一套算法的结果）
    con.execute(
        """DELETE FROM asset_similar_member_v0 WHERE group_id IN
           (SELECT group_id FROM asset_similar_group_v0 WHERE COALESCE(engine,?) = ?)""",
        (engine, engine))
    con.execute("DELETE FROM asset_similar_group_v0 WHERE COALESCE(engine,?) = ?",
                (engine, engine))

    now = now_iso()
    stats = {"groups": 0, "assets": 0, "best_changed": 0}
    for idx, g in enumerate(groups):
        asset_ids = [r["asset_id"] for r in g]
        faces, happiness = _load_faces_and_happiness(con, asset_ids)
        scored = []
        for r in g:
            aid = r["asset_id"]
            s = _score_photo(aid, dict(r), faces.get(aid, []), happiness.get(aid, 0.0))
            scored.append((aid, s, r["capture_time"]))
        scored.sort(key=lambda x: -x[1])
        best_aid = scored[0][0]
        group_id = f"sim_{idx:07d}"
        start_t = min(r["capture_time"] for r in g)
        end_t = max(r["capture_time"] for r in g)
        con.execute(
            """INSERT INTO asset_similar_group_v0
               (group_id, best_asset_id, source_id, start_time, end_time, asset_count, created_at, updated_at, engine)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (group_id, best_aid, g[0]["source_id"], start_t, end_t, len(g), now, now, engine)
        )
        for aid, s, _ in scored:
            con.execute(
                """INSERT INTO asset_similar_member_v0
                   (group_id, asset_id, pick_score, is_best) VALUES (?,?,?,?)""",
                (group_id, aid, s, 1 if aid == best_aid else 0)
            )
        stats["groups"] += 1
        stats["assets"] += len(g)

    con.commit()
    con.close()
    stats["engine"] = engine
    return stats


def get_hidden_ids(con):
    """墙面隐藏集合 = 过滤表 + 相似组非最佳 + 回收站 - 白名单。不再按 confidence 阈值过滤。"""
    try:
        allowed = {r[0] for r in con.execute("SELECT asset_id FROM asset_allowlist_v0")}
    except sqlite3.OperationalError:
        allowed = set()
    try:
        filtered = {r[0] for r in con.execute("SELECT DISTINCT asset_id FROM asset_filter_v0")}
    except sqlite3.OperationalError:
        filtered = set()
    try:
        similar_hidden = {r[0] for r in con.execute(
            """SELECT asset_id FROM asset_similar_member_v0 m
               JOIN asset_similar_group_v0 g USING(group_id)
               WHERE m.is_best=0""")}
    except sqlite3.OperationalError:
        similar_hidden = set()
    try:
        recycled = {r[0] for r in con.execute("SELECT asset_id FROM recycle_v0")}
    except sqlite3.OperationalError:
        recycled = set()
    return (filtered | similar_hidden | recycled) - allowed


def get_similar_groups(engine=None):
    """返回所有相似组（管理页用），含成员与评分。
    engine: 可选，按算法来源过滤（'dual-v1' 离线去重 / 'timewin-v1' 时间窗连拍）；
            省略=全部。dual-v1 组的 start_time 为空（离线模块只写相似关系），
            所以排序补 created_at + group_id 兜底，避免 NULL 段顺序随机。
    2026-09-15 T4：新增 engine 字段随组返回，前端据此标注来源。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    _ensure_similar_tables(con)
    where, args = "", []
    if engine:
        where, args = "WHERE COALESCE(engine,'timewin-v1')=?", [engine]
    groups = []
    for g in con.execute(
        f"""SELECT group_id, best_asset_id, source_id, start_time, end_time, asset_count, engine
           FROM asset_similar_group_v0 {where}
           ORDER BY start_time DESC, created_at DESC, group_id""", args
    ):
        members = [dict(r) for r in con.execute(
            """SELECT m.asset_id, m.pick_score, m.is_best, ma.capture_time, ma.media_type
               FROM asset_similar_member_v0 m
               LEFT JOIN media_asset ma USING(asset_id)
               WHERE m.group_id=?
               ORDER BY m.is_best DESC, m.pick_score DESC""",
            (g["group_id"],)
        )]
        groups.append({
            "id": g["group_id"],
            "best_asset_id": g["best_asset_id"],
            "source_id": g["source_id"],
            "start_time": g["start_time"],
            "end_time": g["end_time"],
            "count": g["asset_count"],
            "engine": g["engine"] or "timewin-v1",
            "members": members,
        })
    total_hidden = con.execute(
        """SELECT count(*) FROM asset_similar_member_v0 m
           JOIN asset_similar_group_v0 g USING(group_id) WHERE m.is_best=0"""
    ).fetchone()[0]
    # 各 engine 档的组数（前端做引擎切换用）
    by_engine = {r[0] or "timewin-v1": r[1] for r in con.execute(
        "SELECT engine, count(*) FROM asset_similar_group_v0 GROUP BY engine")}
    con.close()
    return {"groups": groups, "total": len(groups), "total_hidden": total_hidden,
            "engines": by_engine}


def similar_set_best(group_id, asset_id):
    """手动指定某组的最佳展示照片。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    _ensure_similar_tables(con)
    g = con.execute("SELECT group_id FROM asset_similar_group_v0 WHERE group_id=?", (group_id,)).fetchone()
    if not g:
        con.close()
        return {"error": "分组不存在"}
    m = con.execute("SELECT 1 FROM asset_similar_member_v0 WHERE group_id=? AND asset_id=?",
                    (group_id, asset_id)).fetchone()
    if not m:
        con.close()
        return {"error": "该资产不在分组内"}
    con.execute("UPDATE asset_similar_member_v0 SET is_best=0 WHERE group_id=?", (group_id,))
    con.execute("UPDATE asset_similar_member_v0 SET is_best=1 WHERE group_id=? AND asset_id=?",
                (group_id, asset_id))
    con.execute("UPDATE asset_similar_group_v0 SET best_asset_id=? WHERE group_id=?",
                (asset_id, group_id))
    con.commit()
    con.close()
    return {"ok": True, "group_id": group_id, "best_asset_id": asset_id}


def similar_ungroup(group_id):
    """解散分组：所有成员恢复墙面展示。"""
    con = sqlite3.connect(DB, timeout=10)
    _ensure_similar_tables(con)
    cur = con.execute("DELETE FROM asset_similar_member_v0 WHERE group_id=?", (group_id,))
    n = cur.rowcount
    con.execute("DELETE FROM asset_similar_group_v0 WHERE group_id=?", (group_id,))
    con.commit()
    con.close()
    return {"ok": True, "removed": n}


def similar_repick(group_id):
    """重新按算法为该组选优。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    _ensure_similar_tables(con)
    g = con.execute("SELECT group_id, source_id FROM asset_similar_group_v0 WHERE group_id=?",
                    (group_id,)).fetchone()
    if not g:
        con.close()
        return {"error": "分组不存在"}
    rows = con.execute(
        """SELECT ma.asset_id, ma.capture_time, ma.width, ma.height, ma.quality_score
           FROM asset_similar_member_v0 m
           JOIN media_asset ma USING(asset_id)
           WHERE m.group_id=?""",
        (group_id,)
    ).fetchall()
    asset_ids = [r["asset_id"] for r in rows]
    faces, happiness = _load_faces_and_happiness(con, asset_ids)
    scored = []
    for r in rows:
        aid = r["asset_id"]
        s = _score_photo(aid, dict(r), faces.get(aid, []), happiness.get(aid, 0.0))
        scored.append((aid, s))
    scored.sort(key=lambda x: -x[1])
    best = scored[0][0]
    con.execute("UPDATE asset_similar_member_v0 SET is_best=0 WHERE group_id=?", (group_id,))
    con.execute("UPDATE asset_similar_member_v0 SET is_best=1, pick_score=? WHERE group_id=? AND asset_id=?",
                (scored[0][1], group_id, best))
    for aid, s in scored[1:]:
        con.execute("UPDATE asset_similar_member_v0 SET pick_score=? WHERE group_id=? AND asset_id=?",
                    (s, group_id, aid))
    con.execute("UPDATE asset_similar_group_v0 SET best_asset_id=?, updated_at=? WHERE group_id=?",
                (best, now_iso(), group_id))
    con.commit()
    con.close()
    return {"ok": True, "group_id": group_id, "best_asset_id": best}


# ============ 旅行地点 / 地理反编码 V1（修正 crude 矩形，支持人工标记地点） ============
from geo_cities_cn import CITY_COORDS   # 全国省级→地级市坐标表（2026-09-07 手动定位）

GEO_CITY_PRESETS = {
    "黄山": {"lat": 30.1333, "lon": 118.1667, "province": "安徽", "mountainous": 1},    "西安": {"lat": 34.2667, "lon": 108.9500, "province": "陕西", "mountainous": 0},
    "婺源": {"lat": 29.2500, "lon": 117.8667, "province": "江西", "mountainous": 0},
    "三清山": {"lat": 28.9000, "lon": 118.0667, "province": "江西", "mountainous": 1},
    "北京": {"lat": 39.9042, "lon": 116.4074, "province": "北京", "mountainous": 0},
    "上海": {"lat": 31.2304, "lon": 121.4737, "province": "上海", "mountainous": 0},
    "杭州": {"lat": 30.2741, "lon": 120.1551, "province": "浙江", "mountainous": 0},
    "成都": {"lat": 30.5728, "lon": 104.0668, "province": "四川", "mountainous": 0},
}
# 常用城市坐标（2026-09-07 手动定位扩充）：老照片没 GPS、库里有该地照片才能进下拉，
# 这份表让「青岛」这类从未有过 GPS 的目的地也能直接新建标记
GEO_CITY_PRESETS.update({
    "青岛": {"lat": 36.0671, "lon": 120.3826, "province": "山东", "mountainous": 0},
    "济南": {"lat": 36.6512, "lon": 117.1201, "province": "山东", "mountainous": 0},
    "烟台": {"lat": 37.4638, "lon": 121.4479, "province": "山东", "mountainous": 0},
    "威海": {"lat": 37.5128, "lon": 122.1206, "province": "山东", "mountainous": 0},
    "泰安": {"lat": 36.2000, "lon": 117.0880, "province": "山东", "mountainous": 1},
    "曲阜": {"lat": 35.5855, "lon": 116.9879, "province": "山东", "mountainous": 0},
    "大连": {"lat": 38.9140, "lon": 121.6147, "province": "辽宁", "mountainous": 0},
    "沈阳": {"lat": 41.8057, "lon": 123.4315, "province": "辽宁", "mountainous": 0},
    "长春": {"lat": 43.8171, "lon": 125.3235, "province": "吉林", "mountainous": 0},
    "哈尔滨": {"lat": 45.8038, "lon": 126.5350, "province": "黑龙江", "mountainous": 0},
    "天津": {"lat": 39.3434, "lon": 117.3616, "province": "天津", "mountainous": 0},
    "秦皇岛": {"lat": 39.9354, "lon": 119.5977, "province": "河北", "mountainous": 0},
    "承德": {"lat": 40.9762, "lon": 117.9633, "province": "河北", "mountainous": 1},
    "石家庄": {"lat": 38.0428, "lon": 114.5149, "province": "河北", "mountainous": 0},
    "太原": {"lat": 37.8706, "lon": 112.5489, "province": "山西", "mountainous": 0},
    "大同": {"lat": 40.0768, "lon": 113.3001, "province": "山西", "mountainous": 0},
    "平遥": {"lat": 37.1892, "lon": 112.1763, "province": "山西", "mountainous": 0},
    "呼和浩特": {"lat": 40.8424, "lon": 111.7490, "province": "内蒙古", "mountainous": 0},
    "满洲里": {"lat": 49.5985, "lon": 117.3797, "province": "内蒙古", "mountainous": 0},
    "南京": {"lat": 32.0603, "lon": 118.7969, "province": "江苏", "mountainous": 0},
    "苏州": {"lat": 31.2989, "lon": 120.5853, "province": "江苏", "mountainous": 0},
    "无锡": {"lat": 31.4912, "lon": 120.3119, "province": "江苏", "mountainous": 0},
    "扬州": {"lat": 32.3947, "lon": 119.4126, "province": "江苏", "mountainous": 0},
    "常州": {"lat": 31.8107, "lon": 119.9740, "province": "江苏", "mountainous": 0},
    "南通": {"lat": 31.9802, "lon": 120.8943, "province": "江苏", "mountainous": 0},
    "合肥": {"lat": 31.8206, "lon": 117.2272, "province": "安徽", "mountainous": 0},
    "九华山": {"lat": 30.4854, "lon": 117.7996, "province": "安徽", "mountainous": 1},
    "厦门": {"lat": 24.4798, "lon": 118.0894, "province": "福建", "mountainous": 0},
    "福州": {"lat": 26.0745, "lon": 119.2965, "province": "福建", "mountainous": 0},
    "泉州": {"lat": 24.8741, "lon": 118.6757, "province": "福建", "mountainous": 0},
    "宁波": {"lat": 29.8683, "lon": 121.5440, "province": "浙江", "mountainous": 0},
    "温州": {"lat": 27.9938, "lon": 120.6994, "province": "浙江", "mountainous": 0},
    "绍兴": {"lat": 30.0300, "lon": 120.5804, "province": "浙江", "mountainous": 0},
    "普陀山": {"lat": 29.9734, "lon": 122.3825, "province": "浙江", "mountainous": 1},
    "千岛湖": {"lat": 29.6087, "lon": 119.0370, "province": "浙江", "mountainous": 1},
    "莫干山": {"lat": 30.6017, "lon": 119.8627, "province": "浙江", "mountainous": 1},
    "南昌县": {"lat": 28.5475, "lon": 115.9263, "province": "江西", "mountainous": 0},
    "井冈山": {"lat": 26.5800, "lon": 114.1650, "province": "江西", "mountainous": 1},
    "龙虎山": {"lat": 28.1156, "lon": 117.0300, "province": "江西", "mountainous": 1},
    "郑州": {"lat": 34.7466, "lon": 113.6254, "province": "河南", "mountainous": 0},
    "洛阳": {"lat": 34.6197, "lon": 112.4540, "province": "河南", "mountainous": 0},
    "开封": {"lat": 34.7971, "lon": 114.3074, "province": "河南", "mountainous": 0},
    "嵩山": {"lat": 34.4866, "lon": 113.0343, "province": "河南", "mountainous": 1},
    "武汉": {"lat": 30.5928, "lon": 114.3055, "province": "湖北", "mountainous": 0},
    "宜昌": {"lat": 30.6920, "lon": 111.2865, "province": "湖北", "mountainous": 1},
    "神农架": {"lat": 31.7445, "lon": 110.6759, "province": "湖北", "mountainous": 1},
    "长沙": {"lat": 28.2282, "lon": 112.9388, "province": "湖南", "mountainous": 0},
    "张家界": {"lat": 29.1173, "lon": 110.4793, "province": "湖南", "mountainous": 1},
    "凤凰": {"lat": 27.9494, "lon": 109.5995, "province": "湖南", "mountainous": 1},
    "衡山": {"lat": 27.2469, "lon": 112.7342, "province": "湖南", "mountainous": 1},
    "广州": {"lat": 23.1291, "lon": 113.2644, "province": "广东", "mountainous": 0},
    "深圳": {"lat": 22.5431, "lon": 114.0579, "province": "广东", "mountainous": 0},
    "珠海": {"lat": 22.2707, "lon": 113.5767, "province": "广东", "mountainous": 0},
    "汕头": {"lat": 23.3540, "lon": 116.6820, "province": "广东", "mountainous": 0},
    "桂林": {"lat": 25.2742, "lon": 110.2902, "province": "广西", "mountainous": 1},
    "南宁": {"lat": 22.8170, "lon": 108.3665, "province": "广西", "mountainous": 0},
    "北海": {"lat": 21.4811, "lon": 109.1199, "province": "广西", "mountainous": 0},
    "海口": {"lat": 20.0442, "lon": 110.1999, "province": "海南", "mountainous": 0},
    "三亚": {"lat": 18.2528, "lon": 109.5119, "province": "海南", "mountainous": 0},
    "重庆": {"lat": 29.5630, "lon": 106.5516, "province": "重庆", "mountainous": 1},
    "昆明": {"lat": 24.8801, "lon": 102.8329, "province": "云南", "mountainous": 1},
    "大理": {"lat": 25.6065, "lon": 100.2676, "province": "云南", "mountainous": 1},
    "丽江": {"lat": 26.8721, "lon": 100.2303, "province": "云南", "mountainous": 1},
    "西双版纳": {"lat": 22.0017, "lon": 100.7979, "province": "云南", "mountainous": 1},
    "香格里拉": {"lat": 27.8258, "lon": 99.7069, "province": "云南", "mountainous": 1},
    "腾冲": {"lat": 25.0303, "lon": 98.4941, "province": "云南", "mountainous": 1},
    "贵阳": {"lat": 26.6470, "lon": 106.6302, "province": "贵州", "mountainous": 1},
    "遵义": {"lat": 27.7255, "lon": 106.9273, "province": "贵州", "mountainous": 1},
    "黄果树": {"lat": 25.9350, "lon": 105.6744, "province": "贵州", "mountainous": 1},
    "拉萨": {"lat": 29.6520, "lon": 91.1721, "province": "西藏", "mountainous": 1},
    "林芝": {"lat": 29.6486, "lon": 94.3624, "province": "西藏", "mountainous": 1},
    "日喀则": {"lat": 29.2690, "lon": 88.8802, "province": "西藏", "mountainous": 1},
    "兰州": {"lat": 36.0611, "lon": 103.8343, "province": "甘肃", "mountainous": 0},
    "敦煌": {"lat": 40.1421, "lon": 94.6619, "province": "甘肃", "mountainous": 0},
    "张掖": {"lat": 38.9259, "lon": 100.4496, "province": "甘肃", "mountainous": 1},
    "西宁": {"lat": 36.6171, "lon": 101.7782, "province": "青海", "mountainous": 1},
    "青海湖": {"lat": 36.8814, "lon": 100.1218, "province": "青海", "mountainous": 1},
    "银川": {"lat": 38.4872, "lon": 106.2309, "province": "宁夏", "mountainous": 0},
    "中卫": {"lat": 37.5149, "lon": 105.1896, "province": "宁夏", "mountainous": 0},
    "乌鲁木齐": {"lat": 43.8256, "lon": 87.6168, "province": "新疆", "mountainous": 1},
    "喀什": {"lat": 39.4704, "lon": 75.9898, "province": "新疆", "mountainous": 1},
    "香港": {"lat": 22.3193, "lon": 114.1694, "province": "香港", "mountainous": 0},
    "澳门": {"lat": 22.1987, "lon": 113.5439, "province": "澳门", "mountainous": 0},
    "台北": {"lat": 25.0330, "lon": 121.5654, "province": "中国台湾", "mountainous": 0},
})

def classify_geo(lat, lon):
    """经纬度 → 地区/省份/沿海/山区。保留 V0.3 规则并修复明显误标。"""
    # 新疆：东经 73-96, 北纬 34-49
    if 73 <= lon <= 96 and 34 <= lat <= 49:
        if 88 <= lon <= 94 and 41.5 <= lat <= 44:
            return "吐鲁番/哈密盆地", "新疆", 0, 0
        # 2026-09-04 实测簇细分：全库 966 张新疆 EXACT_GPS 按 0.01° 网格聚类，
        # 每簇抽 VLM 描述核实（赛里木湖=海边+山脉、乌鲁木齐=夜晚塔楼市场/大巴扎、
        # 裕民=小白杨哨所柱子、喀纳斯=森林湖泊、那拉提=草原山景、昭苏=草原牌坊）
        if 47.0 <= lat <= 49.0 and 86.4 <= lon <= 88.6:
            return "喀纳斯/禾木", "新疆", 0, 1
        if 44.3 <= lat <= 45.0 and 80.7 <= lon <= 81.7:
            return "赛里木湖", "新疆", 0, 1
        if 43.5 <= lat <= 44.3 and 80.8 <= lon <= 81.7:
            return "伊宁/伊犁河谷", "新疆", 0, 0
        if 43.5 <= lat <= 44.3 and 83.3 <= lon <= 85.0:
            return "那拉提/巩乃斯", "新疆", 0, 1
        if 43.6 <= lat <= 44.2 and 87.2 <= lon <= 88.0:
            return "乌鲁木齐", "新疆", 0, 0
        if 42.8 <= lat <= 43.5 and 80.7 <= lon <= 82.0:
            return "昭苏/特克斯", "新疆", 0, 1
        if 45.4 <= lat <= 46.0 and 82.3 <= lon <= 82.8:
            return "小白杨哨所/裕民", "新疆", 0, 0
        return "新疆", "新疆", 0, 1

    # 安徽：黄山（东经 117.8-118.5，北纬 29.8-30.5）
    # 必须放在江西大框之前，否则黄山坐标先被江西（113-118.5E）吞掉
    if 117.8 <= lon <= 118.5 and 29.8 <= lat <= 30.5:
        return "黄山", "安徽", 0, 1

    # 长沙（实测簇 28.2N / 113.1E）——同样必须在江西大框之前（113E 重叠）
    if 112.70 <= lon <= 113.30 and 28.00 <= lat <= 28.45:
        return "长沙", "湖南", 0, 0

    # 江西：东经 113-118.5, 北纬 24-31
    if 113 <= lon <= 118.5 and 24 <= lat <= 31:
        if 115.7 <= lon <= 116.2 and 28.4 <= lat <= 28.9:
            return "南昌", "江西", 0, 0
        # 乐平（家所在城市，实测坐标簇 29.0-29.15N / 117.1E，208+ 天）
        if 117.05 <= lon <= 117.35 and 28.90 <= lat <= 29.19:
            return "乐平", "江西", 0, 0
        # 景德镇市区（实测簇 29.25-29.35N / 117.15-117.25E）
        if 117.10 <= lon <= 117.35 and 29.20 <= lat <= 29.45:
            return "景德镇", "江西", 0, 0
        # 萍乡（实测簇 27.6-27.8N / 113.6-113.9E）
        if 113.40 <= lon <= 114.00 and 27.40 <= lat <= 27.90:
            return "萍乡", "江西", 0, 1
        # 庐山/九江（含庐山西海）：东经 115.2-116.1，北纬 29.0-29.6
        if 115.2 <= lon <= 116.1 and 29.0 <= lat <= 29.6:
            return "庐山/九江", "江西", 0, 1
        if 116.0 <= lon <= 116.6 and 28.9 <= lat <= 29.4:
            return "鄱阳湖平原", "江西", 0, 0
        # 武夷山（赣闽交界）：东经 117.5-118.1，北纬 27.4-27.9
        if 117.5 <= lon <= 118.1 and 27.4 <= lat <= 27.9:
            return "武夷山", "福建", 0, 1
        if 117.5 <= lon <= 118.5 and 28.5 <= lat <= 29.5:
            return "三清山/上饶", "江西", 0, 1
        # 婺源（东经 117.6-117.9，北纬 29.1-29.4）
        if 117.6 <= lon <= 117.9 and 29.1 <= lat <= 29.4:
            return "婺源", "江西", 0, 0
        return "江西其他", "江西", 0, 1

    # 陕西：西安（东经 108.5-109.5，北纬 33.9-34.6）
    if 108.5 <= lon <= 109.5 and 33.9 <= lat <= 34.6:
        return "西安", "陕西", 0, 0

    # 重庆（实测簇 29.7N / 106.5E）
    if 106.20 <= lon <= 107.00 and 29.40 <= lat <= 29.95:
        return "重庆", "重庆", 0, 0

    # 恩施（实测簇 30.0-30.3N / 109.2-109.5E，恩施大峡谷线）
    if 108.90 <= lon <= 110.00 and 29.80 <= lat <= 30.60:
        return "恩施", "湖北", 0, 1

    # 川西北（汶川-理县-松潘-九寨沟一线，实测簇 31.5-33.4N / 103.6-104.1E）
    # lat 下限 31.3 避免吞掉成都（30.6N/104.07E）
    if 103.40 <= lon <= 104.30 and 31.30 <= lat <= 33.60:
        return "川西北/九寨沟", "四川", 0, 1

    # 陇南/青川（川甘陕交界，实测簇 33.3N / 105.0E）
    if 104.60 <= lon <= 105.30 and 33.10 <= lat <= 33.55:
        return "陇南/川甘界", "甘肃", 0, 1

    # 东南沿海省份（浙江/福建/广东/广西/海南真实沿海区域），
    # 注意江西/西安已被上方处理，这里只留东部/南部沿海
    # 2026-09-07：上海市框从「东南沿海」大筐里拆出来——用户上海工作时期 504 张
    # GPS 照片全落在 30.9-31.9N/120.9-122.1E，之前全被塞进东南沿海（含苏州/嘉兴方向）
    if 30.7 <= lat <= 31.9 and 120.9 <= lon <= 122.1:
        return "上海", "上海", 1, 0
    # 苏州/无锡方向（上海框西侧，长三角平原）
    if 30.7 <= lat <= 32.0 and 119.5 <= lon < 120.9:
        return "苏南", "江苏", 0, 0
    if 118 <= lon <= 122 and 20 <= lat <= 32:
        return "东南沿海", "沿海", 1, 0
    if 110 <= lon < 118 and 20 <= lat <= 25:
        return "东南沿海", "沿海", 1, 0

    # 华北山区（承德/张家口）
    if 114 <= lon <= 120 and 39 <= lat <= 42.5:
        return "华北", "河北", 0, 1

    return "其他", "未知", 0, 0


def _init_user_trip_table(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS user_trip (
        trip_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        region TEXT NOT NULL,
        province TEXT NOT NULL,
        latitude REAL,
        longitude REAL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """)
    # 2026-09-09 v2：区县第三级（产品决策：旅行标记表单含 地点/district + 起止日期）
    cols = {r[1] for r in con.execute("PRAGMA table_info(user_trip)")}
    if "district" not in cols:
        con.execute("ALTER TABLE user_trip ADD COLUMN district TEXT DEFAULT ''")
        con.commit()


def list_trips():
    con = sqlite3.connect(DB, timeout=20)
    con.row_factory = sqlite3.Row
    _init_user_trip_table(con)
    rows = [dict(r) for r in con.execute(
        "SELECT trip_id,name,region,province,district,latitude,longitude,start_date,end_date,created_at,updated_at "
        "FROM user_trip ORDER BY start_date"
    )]
    con.close()
    return {"trips": rows}


# ---------- 防抖合并（2026-09-09：高频标注时 30s 窗口内多次触发只重建一次） ----------
_GEO_RB_LOCK = threading.Lock()
_GEO_RB_STATE = {"running": False, "dirty": False, "last_done": 0.0}
GEO_REBUILD_DEBOUNCE_SEC = 30.0


def _geo_rebuild_drain():
    """后台排水：冷却结束后补跑一次；跑完若又有 dirty 再循环，直到干净。"""
    import time as _t
    while True:
        with _GEO_RB_LOCK:
            wait = GEO_REBUILD_DEBOUNCE_SEC - (_t.time() - _GEO_RB_STATE["last_done"])
            _GEO_RB_STATE["dirty"] = False
        if wait > 0:
            _t.sleep(wait)
        try:
            rebuild_geo()
        except Exception as exc:
            print(f"[rebuild-geo] 防抖补跑异常: {exc}", flush=True)
        with _GEO_RB_LOCK:
            _GEO_RB_STATE["last_done"] = _t.time()
            if not _GEO_RB_STATE["dirty"]:
                _GEO_RB_STATE["running"] = False
                return


def request_rebuild_geo(reason=""):
    """标注写库后的重建入口：首次同步执行（调用方拿到 assigned），
    重建中/30s 冷却期内的后续触发只置 dirty，由后台线程合并补跑一次。
    返回 {'assigned': N} 或 {'queued': True, 'assigned': None}。"""
    import time as _t
    with _GEO_RB_LOCK:
        if _GEO_RB_STATE["running"] or \
                (_t.time() - _GEO_RB_STATE["last_done"] < GEO_REBUILD_DEBOUNCE_SEC):
            _GEO_RB_STATE["dirty"] = True
            if not _GEO_RB_STATE["running"]:     # 冷却期内无人在跑 → 起排水线程
                _GEO_RB_STATE["running"] = True
                threading.Thread(target=_geo_rebuild_drain, daemon=True,
                                 name="geo-rebuild-drain").start()
            print(f"[rebuild-geo] 防抖合并（{reason}）：排队待补跑", flush=True)
            return {"queued": True, "assigned": None, "debounced": True}
        _GEO_RB_STATE["running"] = True
    try:
        r = rebuild_geo()
    except Exception:
        with _GEO_RB_LOCK:
            _GEO_RB_STATE["running"] = False
        raise
    with _GEO_RB_LOCK:
        _GEO_RB_STATE["last_done"] = _t.time()
        pending = _GEO_RB_STATE["dirty"]
        _GEO_RB_STATE["dirty"] = False
        if pending:                              # 重建期间有人触发过 → 排水补跑
            threading.Thread(target=_geo_rebuild_drain, daemon=True,
                             name="geo-rebuild-drain").start()
        else:
            _GEO_RB_STATE["running"] = False
    return r


def _trip_match_dates(con, trips):
    """返回 {asset_id: trip} 每个资产按拍摄日期命中的旅行。"""
    if not trips:
        return {}
    assets = con.execute(
        "SELECT asset_id, capture_time FROM media_asset WHERE capture_time IS NOT NULL"
    ).fetchall()
    mapping = {}
    for asset_id, capture_time in assets:
        d = capture_time[:10]
        for t in trips:
            if t["start_date"] <= d <= t["end_date"]:
                mapping[asset_id] = t
                break
    return mapping


def rebuild_geo():
    """重建 asset_geo_v0：优先 EXACT_GPS → 用户旅行标记 → 同日互证 TIME_NEIGHBOR → EVENT_INFERRED。"""
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _init_user_trip_table(con)
    con.executescript("""
    CREATE TABLE IF NOT EXISTS asset_geo_v0 (
      asset_id TEXT PRIMARY KEY REFERENCES media_asset(asset_id),
      latitude REAL, longitude REAL,
      region TEXT, province TEXT,
      is_coastal INTEGER NOT NULL DEFAULT 0,
      is_mountainous INTEGER NOT NULL DEFAULT 0,
      location_source TEXT NOT NULL,
      created_at TEXT NOT NULL
    );
    """)
    con.execute("DELETE FROM asset_geo_v0")
    con.execute("DELETE FROM location_inference_v0 WHERE location_source='USER_TRIP_REBUILD'")
    con.execute("DELETE FROM location_inference_v0 WHERE evidence_json LIKE '%same_day_majority%'")
    con.execute("DELETE FROM location_inference_v0 WHERE evidence_json LIKE '%user_trip%' AND location_source='USER_CONFIRMED'")
    # 视觉模型辅助验证缓存：(day, region) -> verdict（mismatch 时撤销天赋值，match 时提置信度）
    _init_vision_geo_cache(con)
    _vision_geo_cache = {(r["day"], r["region"]): r["verdict"]
                         for r in con.execute("SELECT day, region, verdict FROM vision_geo_check_v0")}

    trips = [dict(r) for r in con.execute("SELECT * FROM user_trip")]
    trip_map = _trip_match_dates(con, trips)

    # EXACT_GPS
    exact_rows = []
    for r in con.execute("SELECT asset_id, latitude, longitude FROM media_asset WHERE latitude IS NOT NULL"):
        exact_rows.append((r["asset_id"], r["latitude"], r["longitude"]))

    # EVENT_INFERRED
    inferred_rows = []
    for r in con.execute(
        "SELECT asset_id, latitude, longitude FROM location_inference_v0 WHERE location_source='EVENT_INFERRED'"
    ):
        inferred_rows.append((r["asset_id"], r["latitude"], r["longitude"]))

    now = now_iso()
    inserted = 0
    handled = set()

    # 1) EXACT_GPS
    for asset_id, lat, lon in exact_rows:
        region, province, coastal, mtn = classify_geo(lat, lon)
        con.execute(
            """INSERT INTO asset_geo_v0
               (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, region, province, coastal, mtn, "EXACT_GPS", now),
        )
        handled.add(asset_id)
        inserted += 1

    # 2) 用户旅行（未命中 EXACT_GPS 的资产）
    for asset_id, trip in trip_map.items():
        if asset_id in handled:
            continue
        lat = trip.get("latitude")
        lon = trip.get("longitude")
        if lat is None or lon is None:
            preset = GEO_CITY_PRESETS.get(trip["region"])
            if preset:
                lat, lon = preset["lat"], preset["lon"]
        region = trip["region"]
        province = trip["province"]
        mountainous = 1 if trip.get("region") in ("黄山", "三清山", "新疆") else 0
        coastal = 1 if trip.get("region") == "东南沿海" else 0
        if lat is None or lon is None:
            lat = lon = None
        con.execute(
            """INSERT INTO asset_geo_v0
               (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, region, province, coastal, mountainous, "USER_TRIP", now),
        )
        # 同时写回 location_inference_v0，方便与 Codex 管道对齐
        if lat is not None and lon is not None:
            con.execute(
            """INSERT OR REPLACE INTO location_inference_v0
               (asset_id, latitude, longitude, location_source, confidence, evidence_json, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, "USER_CONFIRMED", 0.95,
             json.dumps({"method": "user_trip", "trip": trip.get("name")}, ensure_ascii=False), now),
            )
        handled.add(asset_id)
        inserted += 1

    # 3) TIME_NEIGHBOR 同日互证：当天 >=2 个 GPS 点且 >=70% 落同一地区，
    #    才把该地区（及坐标中值）赋给当天无 GPS 的照片/视频
    day_points = {}
    for r in con.execute(
        "SELECT substr(capture_time,1,10) d, latitude, longitude FROM media_asset "
        "WHERE latitude IS NOT NULL AND latitude!=0 AND capture_time IS NOT NULL"
    ):
        day_points.setdefault(r["d"], []).append((r["latitude"], r["longitude"]))
    day_region = {}  # d -> (region, province, coastal, mtn, lat, lon)
    for d, pts in day_points.items():
        if len(pts) < 2:
            continue
        groups = {}
        for lat, lon in pts:
            key = classify_geo(lat, lon)[:2]
            groups.setdefault(key, []).append((lat, lon))
        best_key, best_pts = None, None
        for key, gpts in groups.items():
            if best_pts is None or len(gpts) > len(best_pts):
                best_key, best_pts = key, gpts
        if best_pts is None or len(best_pts) < 2 or len(best_pts) < 0.7 * len(pts):
            continue  # 当天坐标分歧大（跨地区赶路等），不猜
        region, province = best_key
        slat = sorted(p[0] for p in best_pts)
        slon = sorted(p[1] for p in best_pts)
        lat = slat[len(slat) // 2]
        lon = slon[len(slon) // 2]
        _, _, coastal, mtn = classify_geo(lat, lon)
        day_region[d] = (region, province, coastal, mtn, lat, lon)

    # 手动定位种子（2026-09-07 v5）：用户指定的地区优先级最高，覆盖同日 GPS 多数，
    # 让「当天无 GPS / GPS 分歧大」的天也能整组互证赋值（evidence 记 manual_seed_day 供视觉复核）
    # v5：起止日期——一次旅行标一个区间，区间内每天都生成手动天（上限 90 天）
    _init_geo_manual_table(con)
    manual_rows = con.execute("""SELECT gm.asset_id, gm.region, gm.province,
        gm.latitude, gm.longitude, gm.start_date, gm.end_date,
        substr(ma.capture_time,1,10) d
        FROM geo_manual_v0 gm JOIN media_asset ma ON ma.asset_id=gm.asset_id""").fetchall()
    manual_days = {}
    import datetime as _dt
    for mr in manual_rows:
        coastal = mtn = 0
        if mr["latitude"] and mr["longitude"]:
            _, _, coastal, mtn = classify_geo(mr["latitude"], mr["longitude"])
        vals = (mr["region"], mr["province"] or "", coastal, mtn, mr["latitude"], mr["longitude"])
        span_days = [mr["d"]]
        try:
            d0 = mr["start_date"] or mr["d"]
            d1 = mr["end_date"] or d0
            if d0 and d1:
                sd, ed = _dt.date.fromisoformat(d0), _dt.date.fromisoformat(d1)
                span_days = [(sd + _dt.timedelta(days=i)).isoformat()
                             for i in range((ed - sd).days + 1)][:91]
        except ValueError:
            pass
        for d in span_days:
            if d:
                manual_days[d] = vals
    for md, mvals in manual_days.items():
        # 手动种子补位规则（2026-09-07 v2）：当天没有 GPS 多数、或 GPS 多数是
        # 「其他」（坐标落在分类框外的低质量归类）时，用户标注生效；
        # GPS 已能定出真实地区的天不覆盖（用户标在有 GPS 的照片上，锚点保留 GPS）
        cur_region = day_region.get(md, (None,))[0]
        if cur_region is None or cur_region == "其他":
            day_region[md] = mvals

    time_neighbors = []
    if day_region:
        for r in con.execute(
            "SELECT asset_id, substr(capture_time,1,10) d FROM media_asset "
            "WHERE (latitude IS NULL OR latitude=0) AND capture_time IS NOT NULL"
        ):
            hit = day_region.get(r["d"])
            if hit:
                time_neighbors.append((r["asset_id"], r["d"], hit))
    for asset_id, day, (region, province, coastal, mtn, lat, lon) in time_neighbors:
        if asset_id in handled:
            continue
        is_manual_day = day in manual_days
        # 视觉模型辅助验证：若当天已缓存「多数不符」判定，撤销该天赋值
        # （手动定位的天除外：用户明确指定优先，复核结论只降置信度不撤销）
        vkey = _vision_geo_cache.get((day, region))
        if vkey == "mismatch" and not is_manual_day:
            continue
        confidence = 0.8 if vkey == "match" else 0.75
        if is_manual_day:
            evidence = {"method": "manual_seed_day", "day": day,
                        "gps_points": len(day_points.get(day, []))}
        else:
            evidence = {"method": "same_day_majority", "day": day,
                        "gps_points": len(day_points.get(day, []))}
        if vkey:
            evidence["vision"] = vkey
        con.execute(
            """INSERT INTO asset_geo_v0
               (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, region, province, coastal, mtn, "TIME_NEIGHBOR", now),
        )
        con.execute(
            """INSERT OR REPLACE INTO location_inference_v0
               (asset_id, latitude, longitude, location_source, confidence, evidence_json, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, "EVENT_INFERRED", confidence,
             json.dumps(evidence, ensure_ascii=False), now),
        )
        handled.add(asset_id)
        inserted += 1

    # 3.1) SOURCE 内同日跟随（A6 地理修复 2026-09-09）：同日互证（3) 覆盖不到两类天——
    #      ① 当天只有 1 个 GPS 点（<2 不互证）② 多点但 ≥70% 同区不成立（跨区赶路）。
    #      这两类天实测漏掉 ~1600 张。规则：同一 source（同一部手机）当天的 GPS 点
    #      合成参照——多点时该 source 内仍须 ≥70% 同区；单点时额外要求资产与该点
    #      时间差 <6h——然后只赋给「同 source 同天」的无 GPS 资产。同一部手机当天
    #      的移动轨迹是强证据而非猜测；跨 source 不跟（另一部手机可能人在异地）。
    #      赋值仍标 TIME_NEIGHBOR（享受视觉复核管线），evidence 记 same_source_day。
    _gps_src_day = {}  # (day, source_id) -> [(lat, lon, capture_time)]
    for r in con.execute(
        """SELECT substr(ma.capture_time,1,10) d, mf.source_id, ma.latitude, ma.longitude, ma.capture_time
           FROM media_asset ma JOIN media_file mf USING(asset_id)
           WHERE ma.latitude IS NOT NULL AND ma.latitude!=0 AND ma.capture_time IS NOT NULL"""
    ):
        _gps_src_day.setdefault((r["d"], r["source_id"]), []).append(
            (r["latitude"], r["longitude"], r["capture_time"]))
    _src_day_region = {}  # (day, source_id) -> (region, province, coastal, mtn, lat, lon) | None
    for key, pts in _gps_src_day.items():
        groups = {}
        for lat, lon, _t in pts:
            k2 = classify_geo(lat, lon)[:2]
            groups.setdefault(k2, []).append((lat, lon, _t))
        best_key, best_pts = None, None
        for k2, gpts in groups.items():
            if best_pts is None or len(gpts) > len(best_pts):
                best_key, best_pts = k2, gpts
        if best_pts is None or (len(pts) >= 2 and len(best_pts) < 0.7 * len(pts)):
            _src_day_region[key] = None  # 该 source 当天也分歧大，不猜
            continue
        region, province = best_key
        slat = sorted(p[0] for p in best_pts)
        slon = sorted(p[1] for p in best_pts)
        lat = slat[len(slat) // 2]
        lon = slon[len(slon) // 2]
        _, _, coastal, mtn = classify_geo(lat, lon)
        _src_day_region[key] = (region, province, coastal, mtn, lat, lon,
                                best_pts[0][2] if len(best_pts) == 1 else None)
    src_neighbors = []
    if _src_day_region:
        def _parse_t(v):
            s = str(v)[:19].replace("T", " ")
            return _dt.datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        for r in con.execute(
            """SELECT ma.asset_id, substr(ma.capture_time,1,10) d, mf.source_id, ma.capture_time
               FROM media_asset ma JOIN media_file mf USING(asset_id)
               WHERE (ma.latitude IS NULL OR ma.latitude=0)
                 AND ma.capture_time IS NOT NULL AND ma.capture_time NOT LIKE '0000%'"""
        ):
            key2 = (r["d"], r["source_id"])
            ref = _src_day_region.get(key2)
            if ref is None:
                continue
            region, province, coastal, mtn, lat, lon, single_t = ref
            # 单点天：时间差 <6h 才跟随（该手机当天只留了一个位置证据，离得太远不可信）
            if single_t is not None:
                try:
                    if abs((_parse_t(r["capture_time"]) - _parse_t(single_t)).total_seconds()) > 6 * 3600:
                        continue
                except ValueError:
                    continue
            src_neighbors.append((r["asset_id"], r["d"], r["source_id"],
                                  (region, province, coastal, mtn, lat, lon)))
    for asset_id, day, src_id, (region, province, coastal, mtn, lat, lon) in src_neighbors:
        if asset_id in handled:
            continue
        vkey = _vision_geo_cache.get((day, region))
        if vkey == "mismatch":
            continue
        confidence = 0.7 if vkey == "match" else 0.6  # 比全日互证低一档
        evidence = {"method": "same_source_day", "day": day,
                    "gps_points": len(_gps_src_day.get((day, src_id), []))}
        con.execute(
            """INSERT INTO asset_geo_v0
               (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, region, province, coastal, mtn, "TIME_NEIGHBOR", now),
        )
        con.execute(
            """INSERT OR REPLACE INTO location_inference_v0
               (asset_id, latitude, longitude, location_source, confidence, evidence_json, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, "EVENT_INFERRED", confidence,
             json.dumps(evidence, ensure_ascii=False), now),
        )
        handled.add(asset_id)
        inserted += 1

    # 3.2) 手动天上的「其他」GPS 归并（2026-09-07 v2）：用户标注的天里，坐标落在
    #      分类框外（region='其他'）的 GPS 照片一并归入手动地区——坐标保留（它们
    #      本来就在该地拍的），只修正无意义的地区名
    for md, mvals in manual_days.items():
        con.execute(
            """UPDATE asset_geo_v0 SET region=?, province=?
               WHERE region='其他' AND asset_id IN
               (SELECT asset_id FROM media_asset WHERE substr(capture_time,1,10)=?)""",
            (mvals[0], mvals[1] or "", md),
        )

    # 3.5) 手动定位锚点本体（2026-09-07）：单独标 MANUAL_SEED，视觉复核撤销时不被清掉；
    #      仅当该张本身已有真实地区的 EXACT_GPS（更精确）时保留 GPS
    for mr in manual_rows:
        cur = con.execute("SELECT location_source, region FROM asset_geo_v0 WHERE asset_id=?",
                          (mr["asset_id"],)).fetchone()
        if cur and cur[0] == "EXACT_GPS" and cur[1] not in ("其他", ""):
            continue
        vals = manual_days.get(mr["d"])
        con.execute(
            """INSERT OR REPLACE INTO asset_geo_v0
               (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (mr["asset_id"], mr["latitude"], mr["longitude"], mr["region"], mr["province"] or "",
             vals[2] if vals else 0, vals[3] if vals else 0, "MANUAL_SEED", now),
        )
        con.execute(
            """INSERT OR REPLACE INTO location_inference_v0
               (asset_id, latitude, longitude, location_source, confidence, evidence_json, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (mr["asset_id"], mr["latitude"], mr["longitude"], "USER_CONFIRMED", 1.0,
             json.dumps({"method": "manual_seed", "region": mr["region"]}, ensure_ascii=False), now),
        )
        if not cur:
            inserted += 1
        handled.add(mr["asset_id"])

    # 4) EVENT_INFERRED（剩余）
    for asset_id, lat, lon in inferred_rows:
        if asset_id in handled:
            continue
        region, province, coastal, mtn = classify_geo(lat, lon)
        con.execute(
            """INSERT INTO asset_geo_v0
               (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
               VALUES(?,?,?,?,?,?,?,?,?)""",
            (asset_id, lat, lon, region, province, coastal, mtn, "EVENT_INFERRED", now),
        )
        handled.add(asset_id)
        inserted += 1

    # 5) SIGLIP 相似度传播（2026-09-07 v6）：长期居住地（如上海工作时期）照片散布
    #    多年，同日/区间规则覆盖不了——无位置且有 SIGLIP 向量的照片，向高置信地区
    #    （EXACT_GPS + MANUAL_SEED）的嵌入质心算余弦相似度，最优 ≥0.60 且领先第二名
    #    ≥0.03 才赋值（source=SIMILAR，坐标用地区均值便于地图显示；不参与视觉复核撤销）
    #    2026-09-08：可在「算法偏好」开关（algo_sim_propagate_enabled）
    try:
        import numpy as _np
        sim_on = get_algo("sim_propagate_enabled", "1") == "1"
        coop_on = get_algo("person_coop_enabled", "1") == "1"
        sim_min_score = float(get_algo("sim_min_score", "0.60"))
        coop_ratio = float(get_algo("coop_ratio", "0.50"))
        reg_meta = {}
        for r in con.execute("""SELECT region, province, AVG(latitude) la, AVG(longitude) lo
                                FROM asset_geo_v0 WHERE latitude IS NOT NULL GROUP BY region, province"""):
            reg_meta.setdefault(r[0], (r[1] or "", r[2], r[3]))
        ref_rows = con.execute("""SELECT g.region, e.vector FROM asset_geo_v0 g
            JOIN embedding e ON e.subject_id=g.asset_id AND e.subject_type='asset'
            AND e.model_name=?
            WHERE g.location_source IN ('EXACT_GPS','MANUAL_SEED')""", (SIGLIP_MODEL_NAME,)).fetchall()
        refs = {}
        for r in ref_rows:
            refs.setdefault(r[0], []).append(_np.frombuffer(r[1], dtype='<f4'))
        sim_regions, sim_mats = [], []
        for reg, vecs in refs.items():
            if reg in ("其他", "") or reg not in reg_meta or len(vecs) < 5:
                continue  # 「其他」无地理意义不传播；参照样本太少的不传播（质心不可信）
            if len(vecs) > 300:
                vecs = vecs[:300]
            c = _np.vstack(vecs).mean(axis=0)
            n = _np.linalg.norm(c)
            if not n:
                continue
            sim_regions.append(reg)
            sim_mats.append(c / n)
        sim_cand = con.execute("""SELECT e.subject_id, e.vector FROM embedding e
            WHERE e.subject_type='asset' AND e.model_name=?
            AND e.subject_id NOT IN (SELECT asset_id FROM asset_geo_v0)""", (SIGLIP_MODEL_NAME,)).fetchall()
        if sim_regions and sim_cand:
            M = _np.vstack(sim_mats)
            X = _np.vstack([_np.frombuffer(r[1], dtype='<f4') for r in sim_cand])
            X = X / (_np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
            S = X @ M.T
            best_idx = S.argmax(axis=1)
            srt = _np.sort(S, axis=1)
            second = srt[:, -2] if S.shape[1] > 1 else _np.zeros(len(S))
            sim_count = 0
            sim_assigned = set()
            for i, r in enumerate(sim_cand if sim_on else []):
                aid = r[0]
                if aid in handled:
                    continue
                if S[i, best_idx[i]] < sim_min_score or S[i, best_idx[i]] - second[i] < 0.03:
                    continue
                reg = sim_regions[best_idx[i]]
                province, la, lo = reg_meta.get(reg, ("", None, None))
                con.execute(
                    """INSERT INTO asset_geo_v0
                       (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (aid, la, lo, reg, province, 0, 0, "SIMILAR", now),
                )
                sim_assigned.add(aid)
                sim_count += 1
            if sim_count:
                inserted += sim_count
                print(f"[rebuild-geo] SIGLIP 相似度传播: {sim_count} 张", flush=True)

            # 6) 人物共现传播（2026-09-07 v7，合照逻辑）：无位置照片里的人物与某高置信
            #    地区（EXACT_GPS+MANUAL_SEED）参照照人物重合 ≥50%（≥1 人），且画面最优
            #    方向就是该地区（余弦 ≥0.50、领先第二名 ≥0.01）→ 归入该地区。
            #    家人合照视觉 margin 天然小（同景同衣着），人物重合才是判别器，故阈值放宽
            #    （source=PERSON_COOP，坐标用地区均值；不参与视觉复核撤销）
            person_map = {}
            for pid_row in con.execute(
                    """SELECT asset_id, person_id FROM face_instance_v0
                       WHERE person_id IS NOT NULL"""):
                person_map.setdefault(pid_row[0], set()).add(pid_row[1])
            reg_persons = {}
            for reg in sim_regions:
                ps = con.execute("""SELECT f.person_id, COUNT(DISTINCT f.asset_id) c
                    FROM face_instance_v0 f JOIN asset_geo_v0 g ON g.asset_id=f.asset_id
                    WHERE g.region=? AND f.person_id IS NOT NULL
                    GROUP BY 1 HAVING c>=2""", (reg,)).fetchall()
                reg_persons[reg] = {p[0] for p in ps}
            coop_count = 0
            for i, r in enumerate(sim_cand if coop_on else []):
                aid = r[0]
                if aid in handled or aid in sim_assigned:
                    continue
                P = person_map.get(aid)
                if not P:
                    continue
                reg = sim_regions[best_idx[i]]
                score = float(S[i, best_idx[i]])
                if score < 0.50 or score - float(second[i]) < 0.01:
                    continue
                RP = reg_persons.get(reg)
                if not RP:
                    continue
                hit = len(P & RP)
                if hit < 1 or hit / len(P) < coop_ratio:
                    continue
                province, la, lo = reg_meta.get(reg, ("", None, None))
                con.execute(
                    """INSERT INTO asset_geo_v0
                       (asset_id,latitude,longitude,region,province,is_coastal,is_mountainous,location_source,created_at)
                       VALUES(?,?,?,?,?,?,?,?,?)""",
                    (aid, la, lo, reg, province, 0, 0, "PERSON_COOP", now),
                )
                coop_count += 1
            if coop_count:
                inserted += coop_count
                print(f"[rebuild-geo] 人物共现传播: {coop_count} 张", flush=True)
    except Exception as exc:
        print(f"[rebuild-geo] 相似度传播失败(不影响其他来源): {exc}", flush=True)

    con.commit()
    # 统计
    summary = {}
    for r in con.execute("""SELECT region, province, COUNT(*) c FROM asset_geo_v0 GROUP BY region, province ORDER BY c DESC"""):
        summary[r["region"]] = r["c"]
    con.close()
    # 有同日互证赋值 → 后台跑视觉模型辅助验证（结果按天缓存，增量只验新天）
    # 2026-09-08：可在「算法偏好」关闭（algo_vision_geo_verify_enabled）
    if time_neighbors and get_algo("vision_geo_verify_enabled", "1") == "1":
        try:
            threading.Thread(target=vision_verify_geo, daemon=True).start()
        except Exception:
            pass
    return {"assigned": inserted, "summary": summary}


# ---------- 手动定位种子（2026-09-07 半自动地理闭环） ----------
# 用户在查看器里给某张照片指定地区 → 该张成为 MANUAL_SEED 锚点写入 geo_manual_v0；
# rebuild_geo 把该天整体赋给同组地区（当天无 GPS 的照片走 TIME_NEIGHBOR 同日互证），
# vision_verify_geo 再对 manual_seed_day 证据的组做视觉复核（不符撤销传播行，锚点保留）。

def _init_geo_manual_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS geo_manual_v0 (
        asset_id TEXT PRIMARY KEY REFERENCES media_asset(asset_id),
        region TEXT NOT NULL,
        province TEXT,
        latitude REAL,
        longitude REAL,
        created_at TEXT NOT NULL)""")
    # 2026-09-07 v5：起止日期——一次旅行标一个区间，不再只管拍摄当天
    cols = {r[1] for r in con.execute("PRAGMA table_info(geo_manual_v0)")}
    if "start_date" not in cols:
        con.execute("ALTER TABLE geo_manual_v0 ADD COLUMN start_date TEXT DEFAULT ''")
        con.execute("ALTER TABLE geo_manual_v0 ADD COLUMN end_date TEXT DEFAULT ''")
    # 2026-09-09 v6：省市区第三级（产品决策：手动定位填省市区）
    if "district" not in cols:
        con.execute("ALTER TABLE geo_manual_v0 ADD COLUMN district TEXT DEFAULT ''")
    con.commit()


def _geo_region_coords():
    """地区下拉数据源：asset_geo_v0 里各地区的 GPS 均值坐标 + 旅行预设兜底。"""
    con = sqlite3.connect(DB, timeout=10)
    rows = con.execute(
        """SELECT region, province, AVG(latitude) la, AVG(longitude) lo, COUNT(*) c
           FROM asset_geo_v0 WHERE latitude IS NOT NULL AND latitude!=0
           GROUP BY region, province ORDER BY c DESC""").fetchall()
    con.close()
    m = {}
    for region, province, la, lo, c in rows:
        if region and region not in ("其他",) and la and lo:
            m[region] = {"region": region, "province": province or "",
                         "lat": round(la, 4), "lon": round(lo, 4), "gps_n": c}
    for region, p in GEO_CITY_PRESETS.items():
        if region not in m:
            m[region] = {"region": region, "province": p.get("province", ""),
                         "lat": p["lat"], "lon": p["lon"], "gps_n": 0}
    # 全国地级市（2026-09-07）：库内地区/预设之外的所有市，保证二级下拉全国覆盖
    for prov, cities in CITY_COORDS.items():
        for city, (la, lo) in cities.items():
            if city not in m:
                m[city] = {"region": city, "province": prov,
                           "lat": la, "lon": lo, "gps_n": 0}
    return sorted(m.values(), key=lambda r: (-r["gps_n"], r["region"]))


def _geo_region_groups():
    """省→市两级联动数据源（2026-09-07）：省份分组，组内含库内已有地区（GPS 计数）+ 内置预设。
    库内地区 province 可能是「沿海/未知」等非标准省名，原样保留为一组。"""
    con = sqlite3.connect(DB, timeout=10)
    rows = con.execute(
        """SELECT region, province, COUNT(*) c, AVG(latitude) la, AVG(longitude) lo
           FROM asset_geo_v0 WHERE region IS NOT NULL AND region<>''
           GROUP BY region, province ORDER BY c DESC""").fetchall()
    con.close()
    groups = {}
    for region, province, c, la, lo in rows:
        p = province or "其他"
        groups.setdefault(p, {})[region] = {
            "region": region, "gps_n": c,
            "lat": round(la, 4) if la else None, "lon": round(lo, 4) if lo else None}
    for region, p in GEO_CITY_PRESETS.items():
        pd = p.get("province") or "其他"
        groups.setdefault(pd, {}).setdefault(region, {
            "region": region, "gps_n": 0, "lat": p["lat"], "lon": p["lon"]})
    # 全国地级市兜底（2026-09-07）：每个省组补齐全部地级市，二级下拉全国覆盖
    for prov, cities in CITY_COORDS.items():
        g = groups.setdefault(prov, {})
        for city, (la, lo) in cities.items():
            if city not in g:
                g[city] = {"region": city, "gps_n": 0, "lat": la, "lon": lo, "city": 1}
    ordered = sorted(groups.items(),
                     key=lambda kv: -sum(r["gps_n"] for r in kv[1].values()))
    return [{"province": prov,
             "regions": sorted(rs.values(), key=lambda r: (-r["gps_n"], r["region"]))}
            for prov, rs in ordered]


def geo_seed(body):
    """手动定位：set（标记某张照片的地区并触发重建+互证）/ get / remove。"""
    action = body.get("action", "set")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _init_geo_manual_table(con)
    asset_id = str(body.get("asset_id") or "").strip()
    if action == "get":
        row = con.execute("""SELECT region, province, start_date, end_date, created_at
                             FROM geo_manual_v0 WHERE asset_id=?""", (asset_id,)).fetchone()
        con.close()
        return {"seeded": bool(row), "region": row["region"] if row else None,
                "province": row["province"] if row else "",
                "start_date": row["start_date"] if row else "",
                "end_date": row["end_date"] if row else ""}
    if action == "remove":
        con.execute("DELETE FROM geo_manual_v0 WHERE asset_id=?", (asset_id,))
        con.commit()
        con.close()
        r = request_rebuild_geo("geo_seed remove")
        return {"ok": True, "removed": True, "assigned": r.get("assigned"),
                "queued": r.get("queued")}
    region = str(body.get("region") or "").strip()
    if not asset_id:
        con.close()
        raise ValueError("缺少 asset_id")
    if not region:
        con.close()
        raise ValueError("缺少 region")
    info = next((r for r in _geo_region_coords() if r["region"] == region), None)
    if info:
        province, lat, lon = info.get("province"), info.get("lat"), info.get("lon")
    else:
        # 自定义新地区（2026-09-07）：库里有 GPS 的地区之外，允许直接新建（如老照片的青岛）。
        # 坐标优先取内置城市表；都查不到则置空——地点类目/互证照常生效，仅地图暂无堆点
        preset = GEO_CITY_PRESETS.get(region)
        province = str(body.get("province") or "").strip() or (preset or {}).get("province", "")
        lat = preset["lat"] if preset else body.get("latitude")
        lon = preset["lon"] if preset else body.get("longitude")
        try:
            lat = float(lat) if lat is not None else None
            lon = float(lon) if lon is not None else None
        except (TypeError, ValueError):
            lat = lon = None
    exists = con.execute("SELECT 1 FROM media_asset WHERE asset_id=?", (asset_id,)).fetchone()
    if not exists:
        con.close()
        raise ValueError("资产不存在：" + asset_id)
    # 起止日期（2026-09-07 v5）：默认拍摄当天；一次旅行可标一个区间（上限 90 天防手滑）
    import datetime as _dt
    own_day = con.execute("SELECT substr(capture_time,1,10) d FROM media_asset WHERE asset_id=?",
                          (asset_id,)).fetchone()
    own_day = own_day[0] if own_day else ""
    def _clean_day(v, fallback):
        v = str(v or "").strip()[:10]
        try:
            _dt.date.fromisoformat(v)
            return v
        except ValueError:
            return fallback
    start = _clean_day(body.get("start_date"), own_day)
    end = _clean_day(body.get("end_date"), start or own_day)
    if start and end and start > end:
        start, end = end, start
    try:
        if start and end and (_dt.date.fromisoformat(end) - _dt.date.fromisoformat(start)).days > 90:
            end = (_dt.date.fromisoformat(start) + _dt.timedelta(days=90)).isoformat()
    except ValueError:
        pass
    con.execute("""INSERT OR REPLACE INTO geo_manual_v0
                   (asset_id,region,province,district,latitude,longitude,created_at,start_date,end_date)
                   VALUES(?,?,?,?,?,?,?,?,?)""",
                (asset_id, region, province, str(body.get("district") or ""), lat, lon,
                 now_iso(), start, end))
    con.commit()
    con.close()
    r = request_rebuild_geo("geo_seed set")
    queued = r.get("queued")
    return {"ok": True, "region": region, "province": province,
            "assigned": r.get("assigned"), "queued": queued,
            "note": ("已入库，全库重建排队中（30s 窗口内多次标注自动合并）"
                     if queued else
                     "同日互证已生效，视觉复核在后台运行（不符会自动撤销，手动锚点保留）")}


# ---------- 视觉模型辅助验证（同日互证二次确认） ----------
_VISION_GEO_RUNNING = False
_VISION_GEO_LOCK = threading.Lock()   # 2026-09-02: 并发触发竞态锁


def _init_vision_geo_cache(con):
    con.executescript("""
    CREATE TABLE IF NOT EXISTS vision_geo_check_v0 (
      day TEXT NOT NULL,
      region TEXT NOT NULL,
      verdict TEXT NOT NULL,
      votes_json TEXT,
      sample_count INTEGER,
      created_at TEXT NOT NULL,
      PRIMARY KEY(day, region)
    );
    """)


def _place_desc(region, province):
    """地区 → 给视觉模型看的场景描述；无法描述的地区返回 None（跳过验证）。"""
    if not region or not province or province in ("未知", "沿海"):
        return None
    if region == "江西其他":
        return "中国江西省景德镇一带（江南小城、乡村、丘陵环境）"
    return f"中国{province}{region}一带"


# SIGLIP 视觉降级：地区 → 场景正类描述（LLM 视觉不可用时用本地模型投票）
REGION_SCENE_TEXTS = {
    # 注意：本地 SIGLIP 文本超过 20 字会被截断，描述控制在 20 字内
    "新疆": "新疆草原雪山戈壁沙漠风光",
    "武夷山": "武夷山丹霞山地、竹林与溪流风光",
    "庐山/九江": "庐山山地云雾与森林风光",
    "三清山/上饶": "三清山花岗岩山峰与云海风光",
    "西安": "西安古建筑、城墙或北方城市街景",
    "南昌": "南昌城市街景、赣江或江南城市风貌",
    "东南沿海": "海岸线、沙滩、渔港或海岛风光",
    "黄山": "黄山花岗岩奇峰、云海与奇松风光",
    "乐平": "江南小城街道、公园或室内生活场景",
    "景德镇": "景德镇城市街景、瓷器或陶艺场景",
    "萍乡": "萍乡山地城镇与丘陵风光",
    "重庆": "重庆山城高楼、长江或城市夜景",
    "恩施": "恩施峡谷、喀斯特石壁与山地风光",
    "长沙": "长沙城市街景、湘江或都市夜景",
    "川西北/九寨沟": "九寨沟彩林、高原海子与雪山风光",
    "陇南/川甘界": "陇南山区峡谷、河谷与山地风光",
}

_SIGLIP_NEG_TEXTS = [
    "室内家庭聚餐或聚会的照片",
    "城市街道或商场内部的照片",
    "电脑屏幕、文档或聊天记录截图",
    "人物面部特写或自拍照",
]


def siglip_region_votes(sample_ids, region):
    """本地 SIGLIP 视觉降级验证：地区场景正类 vs 泛化负类。
    正类需严格最高分才算 match；负类领先 >0.03 才算 mismatch，否则 uncertain。
    返回票数列表（与 sample_ids 等长）或 None（该地区无场景描述）。"""
    pos = REGION_SCENE_TEXTS.get(region)
    if not pos:
        return None
    texts = [pos] + _SIGLIP_NEG_TEXTS
    scores = local_siglip_scores(texts, sample_ids)
    votes = []
    for aid in sample_ids:
        s = scores.get(aid)
        if not s:
            votes.append(None)
            continue
        pos_score = s[0]
        margin = max(s[1:]) - pos_score
        if margin > 0.03:
            votes.append("mismatch")
        elif pos_score >= max(s[1:]):
            votes.append("match")
        else:
            votes.append("uncertain")
    return votes


def vision_verify_geo(samples_per_day=None, max_days=None):
    """视觉模型辅助验证 TIME_NEIGHBOR 同日互证：
    每个待验证 (day, region) 抽 N 张无 GPS 样本（N=算法偏好 vision_geo_sample_n，默认 3），
    问视觉模型画面场景是否与推断地区相符；
    ≥2 张「不符」→ 撤销该天赋值（从 asset_geo_v0 / location_inference_v0 删除）；
    「相符」占多 → 置信度 0.75 → 0.8。结果缓存进 vision_geo_check_v0，重建 geo 时直接生效。"""
    global _VISION_GEO_RUNNING
    if samples_per_day is None:
        try:
            samples_per_day = max(1, int(float(get_algo("vision_geo_sample_n", "3"))))
        except Exception:
            samples_per_day = 3
    with _VISION_GEO_LOCK:
        if _VISION_GEO_RUNNING or not API_KEY:
            return {"skipped": "busy_or_no_key"}
        _VISION_GEO_RUNNING = True
    stats = {"days_total": 0, "days_verified": 0, "days_new": 0, "revoked_days": 0,
             "revoked_assets": 0, "boosted_assets": 0, "errors": 0}
    try:
        con = sqlite3.connect(DB, timeout=30)
        con.row_factory = sqlite3.Row
        _init_vision_geo_cache(con)
        cache = {(r["day"], r["region"]): r["verdict"]
                 for r in con.execute("SELECT day, region, verdict FROM vision_geo_check_v0")}
        # 分组：day+region -> [asset_id, ...]，按拍摄时间排序便于全天均匀抽样
        groups = {}
        for r in con.execute(
            """SELECT ag.asset_id, ag.region, ag.province,
                      json_extract(li.evidence_json,'$.day') AS day
               FROM asset_geo_v0 ag
               JOIN location_inference_v0 li ON li.asset_id = ag.asset_id
               WHERE ag.location_source='TIME_NEIGHBOR'
                 AND (li.evidence_json LIKE '%same_day_majority%'
                      OR li.evidence_json LIKE '%manual_seed_day%')
               ORDER BY day, ag.asset_id"""
        ):
            if r["day"]:
                groups.setdefault((r["day"], r["region"], r["province"]), []).append(r["asset_id"])
        stats["days_total"] = len(groups)

        def sample_ids(ids):
            n = len(ids)
            if n <= samples_per_day:
                return ids
            step = n / samples_per_day
            return [ids[int(i * step)] for i in range(samples_per_day)]

        def asset_path(asset_id):
            row = con.execute(
                "SELECT absolute_path FROM media_file WHERE asset_id=? ORDER BY variant_kind='original' DESC LIMIT 1",
                (asset_id,)).fetchone()
            return row[0] if row else None

        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=4)
        use_llm = True  # DeepSeek 视觉可用性；余额不足时整轮降级本地 SIGLIP
        # SIGLIP 只能给已有向量的资产打分（照片），预先载入向量覆盖集
        emb_ids = {r[0] for r in con.execute(
            "SELECT DISTINCT subject_id FROM embedding WHERE subject_type='asset'")}
        try:
            for (day, region, province), ids in groups.items():
                if max_days and stats["days_new"] >= max_days:
                    break
                desc = _place_desc(region, province)
                if not desc:
                    continue  # 场景描述不明确的地区不做视觉验证
                stats["days_verified"] += 1
                verdict = cache.get((day, region))
                if verdict:
                    pass  # 已有缓存，直接应用
                else:
                    sids = sample_ids(ids)
                    source, votes = None, []
                    # 通道 1：DeepSeek LLM 视觉（对地区人文场景判断更准）
                    if use_llm:
                        paths = [(aid, asset_path(aid)) for aid in sids]
                        paths = [(aid, p) for aid, p in paths if p]
                        if paths:
                            futs = [pool.submit(vision_region_verify, p, desc) for _, p in paths]
                            votes = [f.result() for f in futs]
                        if any(v == "no_balance" for v in votes):
                            use_llm = False
                            print("[vision-geo] DeepSeek 视觉不可用（余额/鉴权），本轮降级本地 SIGLIP",
                                  flush=True)
                            votes = []
                        elif votes and any(v is not None for v in votes):
                            source = "llm"
                    # 通道 2：本地 SIGLIP 视觉降级（免费、不出网，仅对已有向量的资产有效）
                    if source is None:
                        cand = [i for i in ids if i in emb_ids]
                        sv = siglip_region_votes(sample_ids(cand), region) if cand else None
                        if sv is not None:
                            source, votes = "siglip", sv
                    if source is None or not votes or all(v is None for v in votes):
                        stats["skipped"] = stats.get("skipped", 0) + 1
                        continue  # 两条通道都没出结果：保持原判，下次再试
                    stats[f"{source}_days"] = stats.get(f"{source}_days", 0) + 1
                    n_match = sum(1 for v in votes if v == "match")
                    n_miss = sum(1 for v in votes if v == "mismatch")
                    if n_miss >= 2 and n_miss > n_match:
                        verdict = "mismatch"
                    elif n_match > 0 and n_match >= n_miss:
                        verdict = "match"
                    else:
                        verdict = "uncertain"
                    con.execute(
                        """INSERT OR REPLACE INTO vision_geo_check_v0
                           (day, region, verdict, votes_json, sample_count, created_at)
                           VALUES(?,?,?,?,?,?)""",
                        (day, region, verdict,
                         json.dumps({"source": source, "votes": votes}, ensure_ascii=False),
                         len(votes), now_iso()))
                    con.commit()
                    cache[(day, region)] = verdict
                    stats["days_new"] += 1
                    print(f"[vision-geo] {day} {region} -> {verdict} ({source} votes={votes})", flush=True)
                # 应用判定到当前行
                is_manual_day = bool(con.execute(
                    """SELECT 1 FROM location_inference_v0 WHERE evidence_json LIKE '%manual_seed_day%'
                       AND json_extract(evidence_json,'$.day')=? LIMIT 1""", (day,)).fetchone())
                if verdict == "mismatch" and is_manual_day:
                    # 2026-09-07：手动标注天的复核仅供参考，不撤销——
                    # 用户明确声明过地区，模型看走眼不该推翻人的记忆
                    stats["manual_kept"] = stats.get("manual_kept", 0) + 1
                    continue
                if verdict == "mismatch":
                    # 手动定位锚点是用户明确指定的，不撤销（只撤销同日互证传播行）
                    protected = {p[0] for p in con.execute(
                        "SELECT asset_id FROM asset_geo_v0 WHERE location_source='MANUAL_SEED'")}
                    del_ids = [i for i in ids if i not in protected]
                    for i in range(0, len(del_ids), 500):
                        chunk = del_ids[i:i + 500]
                        ph = ",".join("?" * len(chunk))
                        con.execute(f"DELETE FROM asset_geo_v0 WHERE asset_id IN ({ph})", chunk)
                        con.execute(f"DELETE FROM location_inference_v0 WHERE asset_id IN ({ph})", chunk)
                    stats["revoked_days"] += 1
                    stats["revoked_assets"] += len(del_ids)
                elif verdict == "match":
                    for i in range(0, len(ids), 500):
                        chunk = ids[i:i + 500]
                        for aid in chunk:
                            row = con.execute(
                                "SELECT evidence_json FROM location_inference_v0 WHERE asset_id=?",
                                (aid,)).fetchone()
                            if row and '"vision"' not in (row[0] or ""):
                                try:
                                    ev = json.loads(row[0])
                                except Exception:
                                    ev = {}
                                ev["vision"] = "match"
                                con.execute(
                                    "UPDATE location_inference_v0 SET confidence=0.8, evidence_json=? WHERE asset_id=?",
                                    (json.dumps(ev, ensure_ascii=False), aid))
                    stats["boosted_assets"] += len(ids)
                con.commit()
        finally:
            pool.shutdown(wait=False)
        con.close()
    except Exception as exc:
        print(f"[vision-geo] 异常: {exc}", flush=True)
        stats["error"] = str(exc)
    finally:
        _VISION_GEO_RUNNING = False
    print(f"[vision-geo] 完成: {stats}", flush=True)
    return stats


def vision_geo_status():
    """视觉验证进度与缓存统计。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    try:
        _init_vision_geo_cache(con)
        by_verdict = {r["verdict"]: r["c"] for r in con.execute(
            "SELECT verdict, COUNT(*) c FROM vision_geo_check_v0 GROUP BY verdict")}
        revoked = con.execute(
            """SELECT COUNT(*) FROM vision_geo_check_v0 WHERE verdict='mismatch'""").fetchone()[0]
    finally:
        con.close()
    return {"running": _VISION_GEO_RUNNING, "cache": by_verdict,
            "revoked_days": revoked}


# ---------- 多标签体系（scene_tag_v0）：一张照片可同时挂多个标签 ----------
# 「旅行」判定的家基准：设置 → 算法偏好（家纬度/家经度）。
# 产品化要求（2026-09-17）：个人数据一律不写死，未配置时不做旅行判定。
def _home_coords():
    try:
        la = float(get_setting("algo_home_lat", "") or "")
        lo = float(get_setting("algo_home_lon", "") or "")
        return la, lo
    except Exception:
        return None


def _init_scene_tag_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS scene_tag_v0 (
        asset_id TEXT NOT NULL,
        tag TEXT NOT NULL,
        source TEXT NOT NULL DEFAULT 'RULE',
        confidence REAL NOT NULL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        PRIMARY KEY(asset_id, tag)
    )""")
    con.commit()


def _init_user_category_table(con):
    con.execute("""CREATE TABLE IF NOT EXISTS user_category_v0 (
        category_id TEXT PRIMARY KEY,
        family_id TEXT NOT NULL,
        name TEXT NOT NULL,
        sort_order INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS user_category_member_v0 (
        category_id TEXT NOT NULL,
        asset_id TEXT NOT NULL,
        added_by TEXT,
        added_at TEXT NOT NULL,
        PRIMARY KEY(category_id, asset_id)
    )""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_ucm_asset ON user_category_member_v0(asset_id)")
    con.commit()


def merge_recycle_into_filter(con):
    """2026-09-07 回收站并入已过滤内容：回收站 UI 已下线（与已过滤内容功能重复）。
    启动时把 recycle_v0 历史数据一次性并入 asset_filter_v0（reason=MANUAL），
    照片改由「已过滤内容 · 管理」统一管理，恢复即进白名单。幂等：可重复执行。"""
    has = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='recycle_v0'").fetchone()
    if not has:
        return 0
    con.execute("""CREATE TABLE IF NOT EXISTS asset_allowlist_v0 (
        asset_id TEXT PRIMARY KEY, note TEXT, created_at TEXT)""")
    con.execute("""INSERT OR IGNORE INTO asset_filter_v0
        (asset_id, filter_reason, evidence_kind, evidence_value, confidence, rule_version, created_at)
        SELECT r.asset_id, 'MANUAL', 'user', 'trash-merge', 1.0, 'manual-v1', r.created_at
        FROM recycle_v0 r""")
    con.execute("DELETE FROM recycle_v0")
    con.commit()
    n = con.execute("SELECT count(*) FROM asset_filter_v0 WHERE evidence_value='trash-merge'").fetchone()[0]
    return n


def _init_recycle_table(con):
    """回收站：用户「删除」的照片进这里（软删除，原片不动）。
    回收站内的照片从全部正常分类/搜索/照片墙消失，只在 cat='recycle' 可见，可恢复。"""
    con.execute("""CREATE TABLE IF NOT EXISTS recycle_v0 (
        asset_id TEXT PRIMARY KEY,
        reason TEXT NOT NULL DEFAULT 'USER',
        created_at TEXT NOT NULL
    )""")
    con.commit()


def recycle_action(body):
    """回收站操作：add（移入回收站）/ restore（恢复）/ empty（清空）。
    软删除不碰 NAS 原片、不删索引，随时可恢复，避免「真删后下次扫描又回来」。"""
    action = body.get("action")
    if action not in ("add", "restore", "empty"):
        raise ValueError("未知 action")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _init_recycle_table(con)
    now = now_iso()
    if action == "add":
        asset_ids = body.get("asset_ids") or []
        if not asset_ids:
            raise ValueError("缺少 asset_ids")
        rows = [(a, "USER", now) for a in asset_ids if a]
        con.executemany("INSERT OR IGNORE INTO recycle_v0(asset_id,reason,created_at) VALUES(?,?,?)", rows)
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM recycle_v0").fetchone()[0]
        return {"recycled": len(rows), "total": total}
    if action == "restore":
        asset_ids = body.get("asset_ids") or []
        if not asset_ids:
            raise ValueError("缺少 asset_ids")
        con.executemany("DELETE FROM recycle_v0 WHERE asset_id=?", [(a,) for a in asset_ids if a])
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM recycle_v0").fetchone()[0]
        return {"restored": len(asset_ids), "total": total}
    # empty：清空回收站（仅从回收站表移除，照片索引仍在；如需彻底删除请走来源移除）
    con.execute("DELETE FROM recycle_v0")
    con.commit()
    return {"emptied": True, "total": 0}


def _init_privacy_table(con):
    """隐私相册：用户「藏起来」的照片进这里。
    隐私照片从全部正常分类/搜索/照片墙/回收站消失，仅 PIN 解锁后在隐私视图可见。
    软隐藏：不碰原片、不删索引，随时可移出。"""
    con.execute("""CREATE TABLE IF NOT EXISTS privacy_v0 (
        asset_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL
    )""")
    con.execute("""CREATE TABLE IF NOT EXISTS privacy_pin_v0 (
        id INTEGER PRIMARY KEY CHECK (id=1),
        pin_hash TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )""")
    con.commit()


def _privacy_pin_ok(con, pin):
    """校验 PIN：返回 True/False。未设置过 PIN 时返回 None（区分「还没设」和「错」）。"""
    row = con.execute("SELECT pin_hash FROM privacy_pin_v0 WHERE id=1").fetchone()
    if not row:
        return None
    import hashlib
    h = hashlib.sha256(("wb-privacy:" + str(pin or "")).encode("utf-8")).hexdigest()
    return h == row["pin_hash"]


def privacy_action(body):
    """隐私相册操作：set_pin / verify / add / remove / list / count。
    - set_pin(pin, old_pin?)：首次设置或修改 PIN（首次无需 old_pin）
    - verify(pin)：校验 PIN 是否正确（has_pin 告知是否已设置过）
    - add(asset_ids, pin)：移入隐私相册（需 PIN）
    - remove(asset_ids, pin)：移出（需 PIN）
    - list(pin)：列出隐私资产（需 PIN）
    - count()：仅返回数量与 has_pin，用于侧栏角标，无需 PIN"""
    action = body.get("action")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _init_privacy_table(con)
    now = now_iso()

    if action == "count":
        total = con.execute("SELECT COUNT(*) FROM privacy_v0").fetchone()[0]
        has_pin = con.execute("SELECT COUNT(*) FROM privacy_pin_v0 WHERE id=1").fetchone()[0] > 0
        return {"total": total, "has_pin": has_pin}

    if action == "set_pin":
        pin = str(body.get("pin") or "").strip()
        if not (4 <= len(pin) <= 12) or not pin.isdigit():
            raise ValueError("PIN 须为 4-12 位数字")
        ok = _privacy_pin_ok(con, body.get("old_pin"))
        existing = con.execute("SELECT COUNT(*) FROM privacy_pin_v0 WHERE id=1").fetchone()[0]
        if existing and ok is not True:
            raise ValueError("旧 PIN 不正确")
        import hashlib
        h = hashlib.sha256(("wb-privacy:" + pin).encode("utf-8")).hexdigest()
        con.execute("INSERT INTO privacy_pin_v0(id,pin_hash,updated_at) VALUES(1,?,?) "
                    "ON CONFLICT(id) DO UPDATE SET pin_hash=excluded.pin_hash, updated_at=excluded.updated_at", (h, now))
        con.commit()
        return {"ok": True}

    if action == "verify":
        ok = _privacy_pin_ok(con, body.get("pin"))
        if ok is None:
            return {"ok": False, "has_pin": False}
        return {"ok": bool(ok), "has_pin": True}

    if action == "add":
        asset_ids = body.get("asset_ids") or []
        if not asset_ids:
            raise ValueError("缺少 asset_ids")
        if _privacy_pin_ok(con, body.get("pin")) is not True:
            raise ValueError("PIN 不正确")
        rows = [(a, now) for a in asset_ids if a]
        # 已在回收站的先恢复出来再藏（两表互斥，避免状态混乱）
        con.executemany("DELETE FROM recycle_v0 WHERE asset_id=?", [(a,) for a, _ in rows])
        con.executemany("INSERT OR IGNORE INTO privacy_v0(asset_id,created_at) VALUES(?,?)", rows)
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM privacy_v0").fetchone()[0]
        return {"hidden": len(rows), "total": total}

    if action == "remove":
        asset_ids = body.get("asset_ids") or []
        if not asset_ids:
            raise ValueError("缺少 asset_ids")
        if _privacy_pin_ok(con, body.get("pin")) is not True:
            raise ValueError("PIN 不正确")
        con.executemany("DELETE FROM privacy_v0 WHERE asset_id=?", [(a,) for a in asset_ids if a])
        # 自动移入过的资产被用户移出 → 决策改 manual_removed，之后自动检测不再复藏
        _init_privacy_auto_table(con)
        con.executemany(
            "UPDATE privacy_auto_v0 SET decision='manual_removed', created_at=? WHERE asset_id=?",
            [(now, a) for a in asset_ids if a])
        con.commit()
        total = con.execute("SELECT COUNT(*) FROM privacy_v0").fetchone()[0]
        return {"removed": len(asset_ids), "total": total}

    if action == "list":
        if _privacy_pin_ok(con, body.get("pin")) is not True:
            raise ValueError("PIN 不正确")
        rows = con.execute(
            """SELECT ma.asset_id, ma.capture_time FROM privacy_v0 p
               JOIN media_asset ma USING(asset_id)
               ORDER BY p.created_at DESC, ma.capture_time IS NULL, ma.capture_time DESC""").fetchall()
        media_info = {r["asset_id"]: (r["media_type"], r["width"], r["height"])
                      for r in con.execute("SELECT asset_id, media_type, width, height FROM media_asset")}
        geo_info = {r["asset_id"]: r["region"] for r in con.execute("SELECT asset_id, region FROM asset_geo_v0")}
        assets = [{"id": r["asset_id"], "time": r["capture_time"],
                   "type": media_info.get(r["asset_id"], ("photo", None, None))[0],
                   "width": media_info.get(r["asset_id"], (None, None, None))[1],
                   "height": media_info.get(r["asset_id"], (None, None, None))[2],
                   "region": geo_info.get(r["asset_id"])} for r in rows]
        return {"assets": assets, "total": len(assets)}

    raise ValueError("未知 action")


def _init_crop_table(con):
    """非破坏性裁切：只存归一化裁切框 (x,y,w,h ∈ 0..1)，原片文件永不改动。
    展示端（网格缩略图/大图查看器）按裁切框做 CSS 构图呈现，可随时恢复原图。"""
    con.execute("""CREATE TABLE IF NOT EXISTS crop_v0 (
        asset_id TEXT PRIMARY KEY,
        x REAL NOT NULL, y REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
        updated_at TEXT NOT NULL
    )""")


def crop_action(body):
    """裁切/二次构图：set / clear / map。
    - set(asset_id, rect=[x,y,w,h])：保存裁切框（归一化 0..1）
    - clear(asset_id)：恢复原图（删掉裁切框）
    - map()：返回全部裁切 {id: [x,y,w,h]}，前端加载后本地应用"""
    action = body.get("action")
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    _init_crop_table(con)

    def _valid_rect(r):
        if not isinstance(r, (list, tuple)) or len(r) != 4:
            raise ValueError("rect 须为 [x,y,w,h]")
        vals = []
        for v in r:
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                raise ValueError("rect 含非数值")
        x, y, w, h = vals
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError("rect 数值超出 0..1 范围")
        if x + w > 1.001 or y + h > 1.001:
            raise ValueError("裁切框超出原图范围")
        return [min(x, 1 - w), min(y, 1 - h), w, h]

    if action == "set":
        asset_id = str(body.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("缺少 asset_id")
        x, y, w, h = _valid_rect(body.get("rect"))
        con.execute("INSERT INTO crop_v0(asset_id,x,y,w,h,updated_at) VALUES(?,?,?,?,?,?) "
                    "ON CONFLICT(asset_id) DO UPDATE SET x=excluded.x,y=excluded.y,w=excluded.w,h=excluded.h,updated_at=excluded.updated_at",
                    (asset_id, x, y, w, h, now_iso()))
        con.commit()
        return {"ok": True, "rect": [x, y, w, h]}

    if action == "clear":
        asset_id = str(body.get("asset_id") or "").strip()
        if not asset_id:
            raise ValueError("缺少 asset_id")
        con.execute("DELETE FROM crop_v0 WHERE asset_id=?", (asset_id,))
        con.commit()
        return {"ok": True}

    if action == "map":
        # 2026-09-13 缩略图"自动放大好多倍"修复：map 只返回用户手动裁切（method 为
        # NULL 或 'manual'）。crop_v0 被外部智能裁切批处理写入了 1.2 万+ 条
        # method='center/face/horizon/saliency' 的自动裁切框，原查询不加过滤全部下发，
        # 前端卡片把每条都当"二次构图"套用 → 大量缩略图变成放大数倍的局部特写。
        # 自动策略行保留在表内（批处理自身数据，不删），只是不再进展示通道；
        # 兼容旧库无 method 列的情况（那时全部行都是手动裁切）。
        cols = [r[1] for r in con.execute("PRAGMA table_info(crop_v0)")]
        if "method" in cols:
            rows = con.execute("SELECT asset_id,x,y,w,h FROM crop_v0 "
                               "WHERE method IS NULL OR method='manual'").fetchall()
        else:
            rows = con.execute("SELECT asset_id,x,y,w,h FROM crop_v0").fetchall()
        return {"crops": {r["asset_id"]: [r["x"], r["y"], r["w"], r["h"]] for r in rows}}

    raise ValueError("未知 action")


def _haversine_km(lat1, lon1, lat2, lon2):
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


# 场景标签族：正类文本(SIGLIP 文本上限 20 字) → 标签名
# 所有文本互为竞争类：某标签成立须其正类得分为全场最高且领先次高 ≥0.02
SCENE_TEXTS = {
    "山景": "山峦 群山 山峰 自然风景",
    "海景": "大海 海浪 沙滩 海岸线",
    "美食": "食物 美食 菜品 餐桌",
    "花草": "花朵 花草 植物 花园",
    "夜景": "夜晚 夜景 灯光 灯火",
    "雪景": "积雪 雪地 冬季 雪天",
    "建筑": "建筑物 房屋 街道 古镇",
    "江河": "湖面 江河 河流 水面",
    "宠物": "动物 宠物 小猫 小狗",
    "儿童": "儿童 小孩 孩子 幼儿",
    "生日": "生日蛋糕 蜡烛 庆祝 派对",
    "车辆": "汽车 马路 车内 交通",
}
# 恒定负类（只竞争不出标签）；沙漠/雪峰负类用于压住 新疆沙丘误判海景、天山雪峰误判雪景
_SCENE_NEG_TEXTS = ["室内 房间 家里", "人物 人像 自拍 特写", "屏幕截图 文档 界面",
                   "沙漠 沙丘 戈壁 荒漠", "雪山 雪峰 山顶积雪"]

# 物品维度词表（SIGLIP 预计算标签, 与场景共用打分管线）。
# 阈值经全库校准: top 命中语义高度合理(书包top=开学日/灯笼top=春节跨年/帐篷top=单次露营),
# 默认 margin>0.04, 灯笼语义特征过强(0.02 时命中1481张)单独收紧到 0.06。
OBJECT_TEXTS = {
    "书包": "背包 书包 双肩包",
    "行李箱": "行李箱 拉杆箱 旅行箱",
    "自行车": "自行车 单车 骑行",
    "火车": "火车 高铁 动车 车厢",
    "飞机": "飞机 舷窗 机场",
    "游船": "游船 轮船 游艇",
    "婴儿车": "婴儿车 推车",
    "玩具": "玩具 积木 玩偶",
    "气球": "气球 彩色气球",
    "风筝": "风筝 天空飞的风筝",
    "帐篷": "帐篷 露营 野餐垫",
    "烧烤": "烧烤 烤架 烤串",
    "火锅": "火锅 涮锅 餐桌火锅",
    "灯笼": "红灯笼 春联 红包 过年",
    "滑雪": "滑雪 滑雪板 雪橇",
    "乐器": "乐器 吉他 钢琴",
}
_OBJECT_THRESHOLDS = {"灯笼": 0.06}
_OBJECT_DEFAULT_THRESHOLD = 0.04
_OBJ_NEG_TEXTS = ["空旷风景 没有任何突出物体"]


def object_tag_for_word(word):
    """用户查询词 → 预计算物品标签（词表标签名或空格分隔的任一同义词）。
    2026-09-08: 合并用户自定义词表（settings custom_objects, JSON [{tag,text}]）。"""
    if not word:
        return None
    w = word.strip()
    for tag, text in _all_object_texts().items():
        if w == tag or w in text.split():
            return tag
    return None


def _all_object_texts():
    """内置物品词表 + 用户自定义（custom_objects）。"""
    merged = dict(OBJECT_TEXTS)
    try:
        custom = json.loads(get_setting("custom_objects", "[]"))
        for item in custom:
            tag = (item.get("tag") or "").strip()
            text = (item.get("text") or "").strip()
            if tag and text:
                merged[tag] = text
    except Exception:
        pass
    return merged


def objects_vocabulary():
    """词表管理接口数据：内置（只读）+ 自定义。"""
    builtin = [{"tag": t, "text": txt, "builtin": True} for t, txt in OBJECT_TEXTS.items()]
    custom = []
    try:
        custom = [{"tag": i.get("tag"), "text": i.get("text"), "builtin": False}
                  for i in json.loads(get_setting("custom_objects", "[]"))]
    except Exception:
        pass
    return {"builtin": builtin, "custom": custom,
            "hint": "自定义词加入后需点「重建场景标签」重新打分（全库 SIGLIP 本地跑，约几分钟）"}


def objects_vocabulary_action(body):
    """自定义物品词增删。action ∈ {add, delete}。"""
    action = body.get("action", "")
    try:
        custom = json.loads(get_setting("custom_objects", "[]"))
    except Exception:
        custom = []
    if action == "add":
        tag = (body.get("tag") or "").strip()
        text = (body.get("text") or "").strip()
        if not tag or not text:
            return {"error": "标签名和描述词都必填（描述词用空格分隔同义词）"}
        if tag in OBJECT_TEXTS or any(i.get("tag") == tag for i in custom):
            return {"error": f"标签「{tag}」已存在"}
        custom.append({"tag": tag, "text": text})
    elif action == "delete":
        tag = (body.get("tag") or "").strip()
        custom = [i for i in custom if i.get("tag") != tag]
    else:
        return {"error": "未知 action"}
    set_setting("custom_objects", json.dumps(custom, ensure_ascii=False))
    return {"ok": True, "count": len(custom)}


_SCENE_REBUILD_LOCK = threading.Lock()
_SCENE_REBUILD_RUNNING = False   # 2026-09-02: 供面板显示真实 running 状态


def rebuild_scene_tags(trigger="manual"):
    """全量重建多标签表。一张照片可同时挂多个标签。
    合影 = 同片 ≥2 位已确认人物；旅行 = 坐标距家 >100km（含互证坐标）；
    其余 12 类 = SIGLIP 全库一次打分，正类分需为全部文本最高且领先次高 ≥0.02。

    2026-09-02 修复（三个坑一起修）：
      1) 原实现 DELETE 全表 —— 会把 VLM 重判(更准)的标签一起冲掉, SIGLIP 错标签复活,
         还让已描述资产重新变"无标签", 触发 VLM 重复描述。改为只删 source='SIGLIP'。
      2) SIGLIP 打分 INSERT OR REPLACE —— 会覆盖 VLM 已打标签。改 OR IGNORE,
         VLM 标签永远优先, SIGLIP 只做补充。
      3) 同步跑在 HTTP 线程 + 无锁 —— 全库打分几分钟内阻塞服务, 且与 VLM/人脸
         进程并发写库。改为加锁, 由调用方放后台线程执行。
    2026-09-10 二次锁死修复：纳入全局后台互斥 _BG_TASK_LOCK（与 autoscan/vlm/
    privacy 互斥）+ 全程日志。此前它游离在互斥体系外，DELETE+批量 INSERT 的
    长写事务无日志、无上限。
    """
    global _SCENE_REBUILD_RUNNING
    if not _SCENE_REBUILD_LOCK.acquire(blocking=False):
        return {"ok": False, "error": "already_running"}
    if not _BG_TASK_LOCK.acquire(timeout=30 if trigger.startswith("manual") else 0):
        _SCENE_REBUILD_LOCK.release()
        print(f"[scene] 与其他后台任务冲突，跳过重建（trigger={trigger}）", flush=True)
        return {"ok": False, "error": "busy"}
    _SCENE_REBUILD_RUNNING = True
    print(f"[scene] 场景标签重建开始（trigger={trigger}）", flush=True)
    try:
        result = _rebuild_scene_tags_locked()
        print(f"[scene] 场景标签重建完成: {result.get('tags', {})}", flush=True)
        return result
    finally:
        _SCENE_REBUILD_RUNNING = False
        _SCENE_REBUILD_LOCK.release()
        _BG_TASK_LOCK.release()


def _rebuild_scene_tags_locked():
    now = now_iso()
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=30000")
    _init_scene_tag_table(con)
    con.execute("DELETE FROM scene_tag_v0 WHERE source='SIGLIP'")  # 只清 SIGLIP, 保留 VLM/FACE/GEO
    counts = {}

    # 1) 合影：同片 ≥2 位已确认人物
    for r in con.execute("""
        SELECT fi.asset_id, COUNT(DISTINCT fi.person_id) n
        FROM face_instance_v0 fi
        JOIN person p ON p.person_id=fi.person_id AND p.identity_status='confirmed'
        GROUP BY fi.asset_id HAVING n >= 2"""):
        con.execute("INSERT OR REPLACE INTO scene_tag_v0 VALUES(?,?,?,?,?)",
                    (r["asset_id"], "合影", "FACE", 1.0, now))
        counts["合影"] = counts.get("合影", 0) + 1

    # 2) 旅行：坐标（实拍 GPS + 互证中值都在 asset_geo_v0）距家 >100km；家坐标未配置则跳过
    _home = _home_coords()
    if _home:
        for r in con.execute("SELECT asset_id, latitude, longitude FROM asset_geo_v0"):
            if r["latitude"] and r["longitude"]:
                if _haversine_km(_home[0], _home[1], r["latitude"], r["longitude"]) > 100:
                    con.execute("INSERT OR REPLACE INTO scene_tag_v0 VALUES(?,?,?,?,?)",
                                (r["asset_id"], "旅行", "GEO", 0.95, now))
                    counts["旅行"] = counts.get("旅行", 0) + 1

    con.commit()

    # 3) SIGLIP 场景 12 类 + 物品 16 类+用户自定义：一次全库打分（本地向量已在库，无照片外发）
    pos_texts = list(SCENE_TEXTS.values())
    tag_names = list(SCENE_TEXTS.keys())
    all_obj = _all_object_texts()
    obj_texts = list(all_obj.values())
    obj_tag_names = list(all_obj.keys())
    all_texts = pos_texts + obj_texts + _SCENE_NEG_TEXTS + _OBJ_NEG_TEXTS
    n_pos = len(pos_texts)
    n_obj = len(obj_texts)
    neg_texts = _SCENE_NEG_TEXTS + _OBJ_NEG_TEXTS
    n_neg = len(neg_texts)
    # 地理硬规则：内陆坐标(如赛里木湖视觉极似海)不给海景标签；无地理信息不限制
    inland_ids = {r[0] for r in con.execute(
        "SELECT asset_id FROM asset_geo_v0 WHERE COALESCE(is_coastal,0)=0")}
    try:
        scores = local_siglip_scores(all_texts)
        for aid, v in scores.items():
            vals = [float(x) for x in v]
            pos_vals = vals[:n_pos]
            # 场景负类(前5)与物品负类(含"空旷风景")分开算:
            # "空旷风景"恰好压制山水照, 不能进场景竞争集(否则山景821→457)
            scene_neg_max = max(vals[n_pos + n_obj:n_pos + n_obj + len(_SCENE_NEG_TEXTS)])
            obj_neg_max = max(vals[n_pos + n_obj:]) if n_neg else 0.0
            for i, tag in enumerate(tag_names):
                s = pos_vals[i]
                if tag == "海景" and aid in inland_ids:
                    continue  # 内陆大湖(赛里木湖等)视觉似海, 用地理规则压制
                # 正类之间不互斥(山+河同框可双挂), 只需压过全部负类(SIGLIP 原始 logit 很小,
                # 绝对阈值无意义, 判别信号是相对负类的领先幅度)
                if s > scene_neg_max + 0.02:
                    con.execute("INSERT OR IGNORE INTO scene_tag_v0 VALUES(?,?,?,?,?)",
                                (aid, tag, "SIGLIP", round(s, 4), now))
                    counts[tag] = counts.get(tag, 0) + 1
            # 物品: 压过全部负类(含"无突出物体"泛化负类), 按标签独立阈值
            for j, otag in enumerate(obj_tag_names):
                s = vals[n_pos + j]
                if s > obj_neg_max + _OBJECT_THRESHOLDS.get(otag, _OBJECT_DEFAULT_THRESHOLD):
                    con.execute("INSERT OR IGNORE INTO scene_tag_v0 VALUES(?,?,?,?,?)",
                                (aid, otag, "SIGLIP", round(s, 4), now))
                    counts[otag] = counts.get(otag, 0) + 1
        con.commit()
    except Exception as exc:
        print(f"[scene] SIGLIP 场景打分失败: {exc}")

    # 孤儿清理（资产已删）
    con.execute("DELETE FROM scene_tag_v0 WHERE asset_id NOT IN (SELECT asset_id FROM media_asset)")
    con.commit()
    con.close()
    return {"ok": True, "tags": counts}


def ensure_scene_tags_async():
    """启动时若标签表为空则后台构建，不阻塞服务起来。
    2026-09-10 二次锁死修复：延迟 120 秒再触发（原实现启动即跑），把启动
    窗口期完全让给服务与用户首屏；日志标注自动触发。"""
    con = sqlite3.connect(DB, timeout=10)
    _init_scene_tag_table(con)
    n = con.execute("SELECT COUNT(*) FROM scene_tag_v0").fetchone()[0]
    con.close()
    if n == 0:
        def _delayed():
            time.sleep(120)
            print("[scene] 标签表为空，启动 120s 后自动重建", flush=True)
            rebuild_scene_tags(trigger="auto-startup")
        threading.Thread(target=_delayed, daemon=True).start()


# ---------- 人脸全量补跑（由 launchd 托管的本服务拉起, 子进程不随工具会话回收） ----------
_FACES_PY = FACES_PYTHON
_FACES_SCRIPT = ROOT / "backfill_faces_full.py"
_SIGLIP_PY = SIGLIP_PYTHON
_SIGLIP_SCRIPT = ROOT / "backfill_siglip_embedding.py"
_ENRICH_SCRIPT = ROOT / "enrich.py"


def _running_pids(pattern, exclude=()):
    """找出命令行匹配 pattern 的进程 PID（跨平台）。

    2026-09-16 修：容器镜像里**没有 pgrep**（python 官方精简基础镜像不带 procps）。
    `subprocess.run(["pgrep", ...])` 抛 FileNotFoundError，而调用点普遍写成
    `except Exception: pass` → 幂等检查永远返回「没在跑」→ 每次扫描都重复拉起
    子进程，多个人脸/SigLIP/富化任务同时写同一个 SQLite。
    Linux 走 /proc 自己扫；macOS（无 /proc）退回 pgrep。
    """
    import re
    rx = re.compile(pattern)
    if os.path.isdir("/proc"):
        me = str(os.getpid())
        out = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit() or entry == me:
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    cmd = f.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
            except Exception:
                continue
            if not cmd or not rx.search(cmd):
                continue
            if any(x in cmd for x in exclude):
                continue
            out.append(entry)
        return out
    try:
        pids = subprocess.run(["pgrep", "-f", pattern],
                              capture_output=True, text=True).stdout.split()
    except Exception:
        return []
    return [p for p in pids if p != str(os.getpid())]


def _spawn_worker(worker_key, script, log_name, pgrep_pattern, skip_if=None, extra_args=()):
    """统一拉起后台 worker：进程幂等 + 日志目录兜底 + 异常不静默。

    2026-09-16 修三个坑：
    1. `open(ROOT/"logs"/...)` 在目录缺失时抛 FileNotFoundError，线程体内异常
       无人接管 → 静默死亡。这里 makedirs 兜底。
    2. 拉起失败只返回 dict，调用方（在后台线程里）看不到 → 统一 print 到 stdout，
       让 `docker logs` 里能看见，别再出现「跑了三个月没人知道从没成功过」。
    3. script 不存在时给出明确原因而不是空跑。
    """
    running = _running_pids(pgrep_pattern, exclude=("--pending", "--status", "--dry-run"))
    if running:
        return {"ok": True, "already_running": True, "pids": running}
    if skip_if and not skip_if():
        return {"ok": True, "skipped": "条件不满足"}
    if not script.exists():
        print(f"[{worker_key}] 跳过：脚本不存在 {script}", flush=True)
        return {"ok": False, "error": f"script missing: {script}"}
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        logf = open(LOGS_DIR / log_name, "a")
    except Exception as exc:
        print(f"[{worker_key}] 无法打开日志 {LOGS_DIR/log_name}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return {"ok": False, "error": f"log open failed: {exc}"}
    try:
        subprocess.Popen(_py_cmd(worker_key, *extra_args), cwd=str(ROOT),
                         stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
    except Exception as exc:
        print(f"[{worker_key}] 拉起失败：{type(exc).__name__}: {exc}", flush=True)
        return {"ok": False, "error": str(exc)}
    finally:
        logf.close()
    print(f"[{worker_key}] 已拉起后台任务 {' '.join(extra_args)}", flush=True)
    return {"ok": True, "started": True}


def siglip_backfill_run():
    """拉起 SIGLIP 图像向量回填(幂等: 已在跑不重复)。
    新照片没有向量时语义检索/场景打分都会漏掉它, 导入后自动补。"""
    return _spawn_worker("siglip", _SIGLIP_SCRIPT, "siglip_backfill.log",
                         r"backfill_siglip_embedding\.py|--ff-worker siglip")


def faces_backfill_run():
    """拉起人脸检测+身份归属后台任务(幂等: 已处理资产自动跳过, 已在跑不重复)。"""
    return _spawn_worker("faces", _FACES_SCRIPT, "faces_full.log",
                         r"backfill_faces_full\.py|--ff-worker faces")


# ---------- 增量富化流水线（2026-09-16 新增）----------
# 补齐「新照片导入后从没自动跑过」的环节：精确去重 / 画质 / 相似分组 / 择优 / 语义过滤。
# 幂等：只处理缺数据的资产，重复跑无损（详见 enrich.py 头部设计说明）。
def enrich_run(trigger="auto"):
    """拉起增量富化流水线（幂等：已在跑则不重复）。

    手动触发时可带 `enrich_repick_all` 开关（一次性生效后自动清除），
    用于刻意全量重算代表张 —— 默认只算从没算过择优分的组，
    避免把用户在界面上手动指定的代表张覆盖掉。
    """
    if get_setting("enrich_enabled", "1") != "1" and trigger == "auto":
        return {"ok": True, "skipped": "enrich_enabled=0"}
    extra = []
    if get_setting("enrich_repick_all", "0") == "1":
        extra = ["--all", "--repick-all"]
        set_setting("enrich_repick_all", "0")
    r = _spawn_worker("enrich", _ENRICH_SCRIPT, "enrich.log",
                      r"enrich\.py|--ff-worker enrich", extra_args=tuple(extra))
    if r.get("started"):
        set_setting("enrich_last_run_at", now_iso())
        set_setting("enrich_last_trigger", trigger + ("+repick_all" if extra else ""))
    return r


def enrich_status():
    """富化欠账体检（只读）。界面与排障用。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA busy_timeout=10000")

    def q(sql, *a):
        try:
            return con.execute(sql, a).fetchone()[0]
        except Exception:
            return -1
    st = {
        "enabled": get_setting("enrich_enabled", "1") == "1",
        "last_run_at": get_setting("enrich_last_run_at", ""),
        "last_trigger": get_setting("enrich_last_trigger", ""),
        "running": bool(_pgrep(r"enrich\.py|--ff-worker enrich")),
        "blur_missing": q("""SELECT count(*) FROM media_asset ma
            LEFT JOIN asset_quality_v0 q ON q.asset_id=ma.asset_id
            WHERE ma.media_type='photo' AND (q.asset_id IS NULL OR q.blur_label IS NULL)"""),
        "sha256_done": q("SELECT count(*) FROM media_file WHERE sha256 IS NOT NULL AND sha256<>''"),
        "files_total": q("SELECT count(*) FROM media_file"),
        "ungrouped_photos": q("""SELECT count(*) FROM media_asset ma WHERE ma.media_type='photo'
            AND ma.asset_id NOT IN (SELECT asset_id FROM asset_similar_member_v0)"""),
        "similar_groups": q("SELECT count(*) FROM asset_similar_group_v0"),
        "filtered_assets": q("SELECT count(DISTINCT asset_id) FROM asset_filter_v0"),
        "allowlist": q("SELECT count(*) FROM asset_allowlist_v0"),
        "vision_checked": q("SELECT count(*) FROM asset_vision_check_v0"),
        "vision_hit": q("SELECT count(*) FROM asset_vision_check_v0 WHERE junk=1"),
    }
    try:
        row = con.execute("""SELECT name FROM api_providers_v0
            WHERE COALESCE(vision,1)=1 AND COALESCE(enabled,1)=1
              AND api_key IS NOT NULL AND api_key<>'' LIMIT 1""").fetchone()
        st["vision_provider"] = row[0] if row else ""
    except Exception:
        st["vision_provider"] = ""
    try:
        with open(LOGS_DIR / "enrich.log", "r", errors="replace") as f:
            st["log_tail"] = "".join(f.readlines()[-15:])
    except Exception:
        st["log_tail"] = ""
    con.close()
    return st


def _pgrep(pattern):
    """兼容旧调用名；实现在 _running_pids（容器内没有 pgrep 命令）。"""
    return _running_pids(pattern)


def _enrich_pending():
    """问 enrich.py 还有多少欠账（唯一真源在 enrich.pending_work，不在 server 里重写 SQL）。"""
    try:
        out = subprocess.run(_py_cmd("enrich", "--pending"), cwd=str(ROOT),
                             capture_output=True, text=True, timeout=180).stdout
        for line in reversed((out or "").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
    except Exception as exc:
        print(f"[enrich] 待办检查失败: {type(exc).__name__}: {exc}", flush=True)
    return None



def faces_assign_run():
    """人工标注后拉起 kNN 自动归类(只归属不检测)。
    幂等：归属进程已在跑则跳过；全量检测进程在跑时也跳过(它收尾会统一归属)，
    避免与检测写库互相竞争。"""
    running = _running_pids(r"backfill_faces_full\.py|--ff-worker faces")
    if running:
        return {"ok": True, "already_running": True, "pids": running,
                "note": "检测/归属任务进行中，新标注将在其收尾时生效"}
    if not _FACES_SCRIPT.exists():
        return {"ok": False, "error": "script missing"}
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOGS_DIR / "faces_full.log", "a")
    except Exception as exc:
        return {"ok": False, "error": f"log open failed: {exc}"}
    try:
        subprocess.Popen(_py_cmd("faces", "--assign-only"), cwd=str(ROOT),
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    return {"ok": True, "started": True}


_CAPTIONS_SCRIPT = ROOT / "caption_days.py"


def captions_vlm_run():
    """拉起 VLM 叙事描述补跑(余额不足脚本内整轮跳过, 充值后重跑即可)。"""
    out = _running_pids(r"caption_days\.py")
    if out:
        return {"ok": True, "already_running": True, "pids": out}
    if not _CAPTIONS_SCRIPT.exists():
        return {"ok": False, "error": "script missing"}
    try:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log = open(LOGS_DIR / "captions.log", "a")
    except Exception as exc:
        return {"ok": False, "error": f"log open failed: {exc}"}
    try:
        subprocess.Popen(_py_cmd("captions", "--vlm"), cwd=str(ROOT),
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    finally:
        log.close()
    return {"ok": True, "started": True}


def captions_status():
    con = sqlite3.connect(DB, timeout=10)
    total_days = con.execute(
        "SELECT COUNT(DISTINCT substr(capture_time,1,10)) FROM media_asset "
        "WHERE capture_time IS NOT NULL").fetchone()[0]
    try:
        cap = con.execute("SELECT COUNT(*), SUM(caption_vlm IS NOT NULL) FROM day_caption_v0").fetchone()
        captioned, vlm = cap[0] or 0, cap[1] or 0
    except Exception:
        captioned, vlm = 0, 0
    con.close()
    running = bool(_running_pids(r"caption_days\.py"))
    return {"total_days": total_days, "captioned_days": captioned,
            "vlm_days": vlm, "vlm_running": running}


def get_asset_faces(asset_id):
    """单张照片的全部人脸(供照片标注界面)：框位置 + 已归属人物 + 质量。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    asset = con.execute(
        """SELECT asset_id, capture_time, media_type FROM media_asset WHERE asset_id=?""",
        (asset_id,)).fetchone()
    if not asset:
        con.close()
        raise ValueError("照片不存在")
    faces = []
    for r in con.execute(
        """SELECT fi.face_instance_id, fi.bbox_json, fi.detection_score, fi.quality_class,
                  fi.person_id, p.display_name person_name, p.relationship_label
           FROM face_instance_v0 fi LEFT JOIN person p USING(person_id)
           WHERE fi.asset_id=? ORDER BY fi.face_width*fi.face_height DESC""",
        (asset_id,)):
        item = dict(r)
        item["bbox"] = json.loads(item.pop("bbox_json"))
        faces.append(item)
    con.close()
    return {"asset": dict(asset), "faces": faces}


def search_face_assets(q, include_all=False):
    """按日期前缀/文件名搜照片(照片标注模式选片用)。
    include_all=True 时搜全部照片——漏检人脸/0 脸的照片也能打开, 在页面上拖框补脸。"""
    q = (q or "").strip()
    if not q:
        return {"assets": []}
    like = f"%{q}%"
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    join_faces = "" if include_all else "JOIN face_instance_v0 fi USING(asset_id)"
    rows = con.execute(
        f"""SELECT DISTINCT ma.asset_id, ma.capture_time, ma.media_type,
                  (SELECT mf.filename FROM media_file mf WHERE mf.asset_id=ma.asset_id LIMIT 1) filename,
                  (SELECT COUNT(*) FROM face_instance_v0 fc WHERE fc.asset_id=ma.asset_id) face_count
           FROM media_asset ma {join_faces}
           WHERE (ma.capture_time LIKE ? AND ma.capture_time IS NOT NULL)
              OR EXISTS (SELECT 1 FROM media_file mf WHERE mf.asset_id=ma.asset_id AND mf.filename LIKE ?)
           ORDER BY ma.capture_time DESC LIMIT 60""",
        (f"{q}%", like)).fetchall()
    con.close()
    return {"assets": [dict(r) for r in rows]}


def get_frequent_unlabeled_clusters(limit=20, samples_per_cluster=6):
    """照片标注页默认推荐：按未标注人物出现频率排序的候选面孔。

    取 recurring-v1 候选簇（路人已过滤），按「墙面可见成员数」降序，
    每个簇返回若干张样例人脸（含 bbox 裁切，前端只展示脸部区域，
    避免整图多脸/无人脸造成误判）。已确认人物、过滤表/相似组非最佳资产均不展示。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    hidden = get_hidden_ids(con)
    con.execute("CREATE TEMP TABLE IF NOT EXISTS _hidden_ids_v1(asset_id TEXT PRIMARY KEY)")
    con.execute("DELETE FROM _hidden_ids_v1")
    con.executemany("INSERT OR IGNORE INTO _hidden_ids_v1 VALUES(?)", [(a,) for a in hidden])
    # 先算每个候选簇的可见成员数，按可见频率排序
    cluster_rows = con.execute(
        """SELECT c.cluster_id, c.member_count, c.asset_count,
                  COUNT(DISTINCT fi.face_instance_id) AS visible_member_count
           FROM anonymous_person_cluster_v0 c
           JOIN anonymous_person_membership_v0 m USING(cluster_id)
           JOIN face_instance_v0 fi USING(face_instance_id)
           JOIN media_asset ma ON ma.asset_id=fi.asset_id
           WHERE c.hypothesis_status='candidate' AND fi.person_id IS NULL
             AND NOT EXISTS(SELECT 1 FROM _hidden_ids_v1 h WHERE h.asset_id=ma.asset_id)
           GROUP BY c.cluster_id
           HAVING visible_member_count > 0
           ORDER BY visible_member_count DESC, c.member_count DESC, c.asset_count DESC"""
    ).fetchall()
    clusters = []
    cluster_samples = []
    seen_faces = set()  # 按人脸去重(同一张照片的不同脸可在不同簇各出现一次)
    for idx, c in enumerate(cluster_rows[:limit], start=1):
        rows = con.execute(
            """SELECT DISTINCT ma.asset_id, ma.capture_time, ma.media_type,
                      (SELECT COUNT(*) FROM face_instance_v0 fc WHERE fc.asset_id=ma.asset_id) face_count,
                      fi.face_instance_id, fi.bbox_json
               FROM anonymous_person_membership_v0 m
               JOIN face_instance_v0 fi USING(face_instance_id)
               JOIN media_asset ma ON ma.asset_id=fi.asset_id
               WHERE m.cluster_id=? AND fi.person_id IS NULL
                 AND NOT EXISTS(SELECT 1 FROM _hidden_ids_v1 h WHERE h.asset_id=ma.asset_id)
               ORDER BY fi.face_width*fi.face_height DESC, fi.detection_score DESC
               LIMIT ?""",
            (c["cluster_id"], samples_per_cluster * 2)
        ).fetchall()
        samples = []
        for r in rows:
            if r["face_instance_id"] in seen_faces:
                continue
            seen_faces.add(r["face_instance_id"])
            item = dict(r)
            item["bbox"] = json.loads(item.pop("bbox_json"))
            item["cluster_rank"] = idx
            item["cluster_id"] = c["cluster_id"]
            item["cluster_member_count"] = c["visible_member_count"]  # 可见成员数
            item["cluster_asset_count"] = c["asset_count"]
            samples.append(item)
            cluster_samples.append(item)
            if len(samples) >= samples_per_cluster:
                break
        if not samples:
            continue
        clusters.append({
            "rank": idx,
            "cluster_id": c["cluster_id"],
            "member_count": c["visible_member_count"],
            "asset_count": c["asset_count"],
            "sample_assets": [s["asset_id"] for s in samples],
        })
    con.execute("DROP TABLE IF EXISTS _hidden_ids_v1")
    con.close()
    return {"clusters": clusters, "assets": cluster_samples}


def faces_backfill_status():
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    done = con.execute("SELECT COUNT(*) FROM person_asset_processing_v0 WHERE status='success'").fetchone()[0]
    total = con.execute("SELECT COUNT(*) FROM media_asset").fetchone()[0]
    persons = [dict(r) for r in con.execute(
        """SELECT p.display_name name, COUNT(DISTINCT fi.asset_id) assets
           FROM person p LEFT JOIN face_instance_v0 fi USING(person_id) GROUP BY 1 ORDER BY 2 DESC""")]
    con.close()
    return {"processed": done, "total_assets": total, "persons": persons}


def _faces_running():
    """人脸检测/归属子进程(backfill_faces_full.py)是否在跑。"""
    return bool(_running_pids(r"backfill_faces_full\.py"))


def models_status():
    """「模型与任务」面板聚合状态：模型清单 + 后台任务进度/开关 + 资源冲突提示。"""
    con = sqlite3.connect(DB, timeout=10)
    con.row_factory = sqlite3.Row
    total = con.execute("SELECT COUNT(*) FROM media_asset").fetchone()[0]
    faces_done = con.execute("SELECT COUNT(*) FROM person_asset_processing_v0 WHERE status='success'").fetchone()[0]
    faces_total = con.execute("SELECT COUNT(*) FROM face_instance_v0").fetchone()[0]
    desc_n = con.execute("SELECT COUNT(*) FROM asset_description_v0").fetchone()[0]
    try:
        scene_n = con.execute("SELECT COUNT(*) FROM scene_tag_v0").fetchone()[0]
    except Exception:
        scene_n = 0
    con.close()

    faces_running = _faces_running()
    vlm = vlm_autorun_config()
    ac = autoscan_config()
    vlm_running = VLM_JOB["running"]
    geo_running = _VISION_GEO_RUNNING

    models = [
        {"key": "yunet", "name": "YuNet 人脸检测", "kind": "本地 ONNX", "resource": "CPU",
         "desc": "检测照片/视频里的人脸框（阈值 0.72）", "always_ready": True},
        {"key": "sface", "name": "SFace 人脸识别", "kind": "本地 ONNX", "resource": "CPU",
         "desc": "把人脸转成 128 维向量，做身份 kNN 归属", "always_ready": True},
        {"key": "ollama", "name": "qwen2.5vl:7b", "kind": "本地 Ollama", "resource": "GPU/内存",
         "desc": "图像描述、意图解析、问答（本地优先）", "online": _ollama_up()},
        {"key": "cloudapi", "name": "云端 API", "kind": "OpenAI 兼容", "resource": "网络",
         "desc": "在「API 服务」里自选添加（下拉选服务商，填 Key 即用）",
         "online": bool(_enabled_providers())},
        {"key": "siglip", "name": "SIGLIP", "kind": "本地 transformers", "resource": "CPU",
         "desc": "语义检索、场景/物品标签（本地向量，照片不外发）", "online": _siglip_available()},
    ]

    faces_pct = round(faces_done * 100 / total, 1) if total else 0
    tasks = [
        {"key": "faces", "name": "人脸检测 + 身份归属", "resource": "CPU",
         "running": faces_running, "progress": {"done": faces_done, "total": total, "pct": faces_pct},
         "extra": f"已检出 {faces_total} 张人脸 · 处理 {faces_done}/{total}", "manual": True},
        {"key": "vlm", "name": "VLM 图像描述", "resource": "GPU/内存",
         "running": vlm_running, "pending": vlm_pending_count(), "desc_done": desc_n,
         "auto_enabled": vlm["enabled"], "auto_interval_min": vlm["interval_min"],
         "last_result": VLM_JOB["last_result"], "manual": True},
        {"key": "scene", "name": "场景/物品标签重建", "resource": "CPU",
         "running": _SCENE_REBUILD_RUNNING, "tags": scene_n, "extra": f"已打 {scene_n} 个标签", "manual": True},
        {"key": "geo", "name": "视觉地理位置验证", "resource": "网络/CPU",
         "desc": "GPS 缺失时的补救：调用链上的视觉模型复核推断地区，不可用降级本地 SIGLIP",
         "running": geo_running, "manual": True},
        {"key": "autoscan", "name": "自动扫描新照片", "resource": "磁盘 IO",
         "running": False, "auto_enabled": ac["enabled"], "auto_interval_min": ac["interval_min"],
         "last_scan_at": ac["last_scan_at"], "last_indexed": ac["last_scan_indexed"], "manual": True},
    ]

    # 资源冲突检测：同一资源上有多个任务在跑时告警
    busy = {}
    for t in tasks:
        if t["running"]:
            busy.setdefault(t["resource"], []).append(t["name"])
    conflicts = []
    if faces_running:
        conflicts.append("人脸检测正在占用 CPU（12 核），此时跑「场景标签重建」或「VLM 描述」会互相拖慢")
    if vlm_running:
        conflicts.append("VLM 描述正在占用 Ollama，其它 LLM 请求（意图/问答/地理验证）会排队等待")
    if faces_running and vlm_running:
        conflicts.append("人脸检测(CPU) 与 VLM 描述(GPU) 已同时运行，整机资源紧张，建议错峰")
    return {"models": models, "tasks": tasks, "busy": busy, "conflicts": conflicts}


def models_toggle(body):
    """开关自动任务。key ∈ {vlm_auto, autoscan}。"""
    key = (body.get("key") or "").strip()
    # 2026-09-02 修复: bool("0") 是 True(非空串), 原实现传 enabled=0 时开关反而被打开。
    raw = body.get("enabled")
    enabled = str(raw).lower() in ("1", "true", "yes", "on") if raw is not None else False
    if key == "vlm_auto":
        return vlm_autorun_config(enabled=1 if enabled else 0)
    if key == "autoscan":
        return autoscan_config(enabled=1 if enabled else 0)
    return {"error": f"未知开关 {key}"}


# ---- API 服务 / 本地模型管理（2026-09-07 models.html 重设计） ----

def _mask_key(k):
    k = k or ""
    return (k[:4] + "****" + k[-4:]) if len(k) > 8 else ("****" if k else "")


def llm_providers():
    """「API 服务」板块聚合状态：优先级 + 内置 DeepSeek + 自定义服务 + 本地 Ollama。"""
    ds_key = get_setting("deepseek_api_key", "") or API_KEY
    installed = [m["name"] for m in ollama_models()["models"]]
    return {
        "priority": get_setting("llm_priority", "local_first"),
        "deepseek": {"configured": bool(ds_key), "key_masked": _mask_key(ds_key),
                     "vision_model": get_setting("ds_vision_model", ""), "text_model": get_setting("ds_text_model", "")},
        "ollama_sel": {"vision_model": get_setting("ollama_vision_model", ""),
                       "text_model": get_setting("ollama_text_model", "")},
        "installed_models": installed,
        "providers": _provider_list(),
        "ollama": ollama_models(),
        "presets": PROVIDER_PRESETS,
        "algos": algo_settings_list(),
    }


LLM_MODEL_KEYS = {"ollama_vision_model", "ollama_text_model", "ds_vision_model", "ds_text_model"}


def llm_model_sel_action(body):
    key = body.get("key") or ""
    if key not in LLM_MODEL_KEYS:
        return {"error": "非法 key"}
    val = (body.get("value") or "").strip()
    if not val:
        set_setting(key, "")  # 清空 = 回到内置默认
        return {"ok": True, "key": key, "value": ""}
    set_setting(key, val)
    return {"ok": True, "key": key, "value": val}


def llm_provider_action(body):
    """自定义 API 服务的增删启停测。action ∈ {add, delete, enable, test}。"""
    action = body.get("action", "")
    con = sqlite3.connect(DB, timeout=15)
    con.execute("PRAGMA busy_timeout=15000")
    con.row_factory = sqlite3.Row
    _ensure_api_provider_table(con)
    if action == "add":
        name = (body.get("name") or "").strip()
        base_url = (body.get("base_url") or "").strip()
        model = (body.get("model") or "").strip()
        if not name or not base_url or not model:
            con.close()
            return {"error": "名称 / Base URL / 模型名 必填"}
        pid = "prov_" + uuid.uuid4().hex[:10]
        max_sort = con.execute("SELECT COALESCE(MAX(sort_order),0)+1 FROM api_providers_v0").fetchone()[0]
        con.execute("INSERT INTO api_providers_v0 VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (pid, name, base_url, (body.get("api_key") or "").strip(), model,
                     1 if body.get("vision") else 0, 1 if body.get("free") else 0,
                     1 if body.get("enabled", True) else 0, "", now_iso(), max_sort))
        con.commit()
        con.close()
        return {"ok": True, "provider_id": pid}
    pid = body.get("provider_id") or ""
    if action == "delete":
        con.execute("DELETE FROM api_providers_v0 WHERE provider_id=?", (pid,))
        con.commit()
        con.close()
        return {"ok": True}
    if action == "enable":
        en = str(body.get("enabled")).lower() in ("1", "true", "yes", "on")
        con.execute("UPDATE api_providers_v0 SET enabled=? WHERE provider_id=?", (1 if en else 0, pid))
        con.commit()
        con.close()
        return {"ok": True}
    if action == "edit":
        # 2026-09-17：卡片内直接编辑（原先换模型只能删了重加）
        row = con.execute("SELECT * FROM api_providers_v0 WHERE provider_id=?", (pid,)).fetchone()
        if not row:
            con.close()
            return {"error": "服务不存在"}
        upd, args = [], []
        for f in ("name", "base_url", "model"):
            v = (body.get(f) or "").strip()
            if v:
                upd.append(f"{f}=?"); args.append(v)
        if (body.get("api_key") or "").strip():
            upd.append("api_key=?"); args.append(body["api_key"].strip())
        if body.get("vision") is not None:
            upd.append("vision=?"); args.append(1 if body["vision"] else 0)
        if body.get("free") is not None:
            upd.append("free=?"); args.append(1 if body["free"] else 0)
        if not upd:
            con.close()
            return {"error": "没有可更新的字段"}
        args.append(pid)
        con.execute(f"UPDATE api_providers_v0 SET {', '.join(upd)} WHERE provider_id=?", args)
        con.commit()
        con.close()
        return {"ok": True}
    if action == "move":
        direction = body.get("dir", "up")
        rows = con.execute("SELECT provider_id, sort_order FROM api_providers_v0 ORDER BY sort_order, rowid").fetchall()
        idx = next((i for i, r in enumerate(rows) if r[0] == pid), -1)
        j = idx - 1 if direction == "up" else idx + 1
        if idx < 0 or j < 0 or j >= len(rows):
            con.close()
            return {"ok": True}
        a_id, a_sort = rows[idx][0], rows[idx][1]
        b_id, b_sort = rows[j][0], rows[j][1]
        con.execute("UPDATE api_providers_v0 SET sort_order=? WHERE provider_id=?", (b_sort, a_id))
        con.execute("UPDATE api_providers_v0 SET sort_order=? WHERE provider_id=?", (a_sort, b_id))
        con.commit()
        con.close()
        return {"ok": True}
    if action == "test":
        row = con.execute("SELECT * FROM api_providers_v0 WHERE provider_id=?", (pid,)).fetchone()
        con.close()
        if not row:
            return {"error": "服务不存在"}
        row = dict(zip(row.keys(), row))
        # DeepSeek 种子服务 Key 为空时回退内置 Key（与调用链 _enabled_providers 同逻辑）
        if not (row.get("api_key") or "").strip() and "deepseek" in (row.get("base_url") or ""):
            row["api_key"] = get_setting("deepseek_api_key", "") or API_KEY
        try:
            text, dt = _call_openai_compat(row["base_url"], row["api_key"], row["model"],
                                           [{"role": "user", "content": "回复 OK 两个字母即可"}],
                                           max_tokens=512, timeout=60)
            status = f"可用（{dt}s）"
            con = sqlite3.connect(DB, timeout=15)
            con.execute("UPDATE api_providers_v0 SET status=? WHERE provider_id=?", (status, pid))
            con.commit()
            con.close()
            return {"ok": True, "latency": dt, "reply": (text or "")[:120]}
        except urllib.error.HTTPError as e:
            err = {401: "API Key 无效(401)", 402: "余额不足(402)", 403: "无权限(403)",
                   404: "地址或模型不存在(404)", 429: "限流(429)"}.get(e.code, f"HTTP {e.code}")
            try:
                con = sqlite3.connect(DB, timeout=15)
                con.execute("UPDATE api_providers_v0 SET status=? WHERE provider_id=?", (f"不可用：{err}", pid))
                con.commit()
                con.close()
            except Exception:
                pass
            return {"ok": False, "error": err}
        except Exception as e:
            try:
                con = sqlite3.connect(DB, timeout=15)
                con.execute("UPDATE api_providers_v0 SET status=? WHERE provider_id=?",
                            (f"不可用：连接失败 {type(e).__name__}", pid))
                con.commit()
                con.close()
            except Exception:
                pass
            return {"ok": False, "error": f"连接失败: {type(e).__name__}"}
    con.close()
    return {"error": "未知 action"}


def llm_priority_action(body):
    mode = body.get("mode", "local_first")
    if mode not in ("local_first", "api_first"):
        return {"error": "mode 须为 local_first 或 api_first"}
    set_setting("llm_priority", mode)
    return {"ok": True, "priority": mode}


def deepseek_key_action(body):
    key = (body.get("api_key") or "").strip()
    if not key:
        return {"error": "API Key 为空"}
    set_setting("deepseek_api_key", key)
    try:
        req = urllib.request.Request(
            "https://api.deepseek.com/chat/completions",
            data=json.dumps({"model": DS_TEXT_MODEL,
                             "messages": [{"role": "user", "content": "回复 OK"}],
                             "max_tokens": 8}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=30) as resp:
            json.loads(resp.read())
        return {"ok": True, "verified": True}
    except urllib.error.HTTPError as e:
        return {"ok": True, "saved": True, "verified": False,
                "error": {401: "Key 无效(401)", 402: "余额不足(402)"}.get(e.code, f"HTTP {e.code}")}
    except Exception as e:
        return {"ok": True, "saved": True, "verified": False, "error": str(e)}


def ollama_models():
    # OLLAMA_BASE 带 /v1 后缀（OpenAI 兼容层），tags 接口要用原始地址
    raw = OLLAMA_BASE[:-3] if OLLAMA_BASE.endswith("/v1") else OLLAMA_BASE
    try:
        with _DIRECT_OPENER.open(raw + "/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
        models = [{"name": m.get("name", ""), "size_gb": round((m.get("size") or 0) / 1e9, 1)}
                  for m in data.get("models", [])]
    except Exception:
        models = []
    return {"up": _ollama_up(), "models": models,
            "pulling": [k for k, v in _OLLAMA_PULLS.items() if v.get("running")],
            "pull_done": [k for k, v in _OLLAMA_PULLS.items() if not v.get("running") and v.get("done")],
            "pull_error": {k: v.get("error") for k, v in _OLLAMA_PULLS.items() if v.get("error")}}


def ollama_action(body):
    action = body.get("action", "list")
    model = (body.get("model") or "").strip()
    if action == "list":
        return ollama_models()
    if not model:
        return {"error": "缺少模型名"}
    ollama_bin = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
    if action == "pull":
        if _OLLAMA_PULLS.get(model, {}).get("running"):
            return {"started": False, "already_running": True}

        def _pull():
            _OLLAMA_PULLS[model] = {"running": True, "started_at": now_iso()}
            try:
                subprocess.run([ollama_bin, "pull", model], capture_output=True, timeout=7200)
                _OLLAMA_PULLS[model]["running"] = False
                _OLLAMA_PULLS[model]["done"] = True
            except Exception as exc:
                _OLLAMA_PULLS[model]["running"] = False
                _OLLAMA_PULLS[model]["error"] = str(exc)[:200]

        threading.Thread(target=_pull, daemon=True).start()
        return {"started": True, "model": model}
    if action == "delete":
        try:
            subprocess.run([ollama_bin, "rm", model], capture_output=True, timeout=120)
            return {"deleted": model}
        except Exception as exc:
            return {"error": str(exc)}
    return {"error": "未知 action"}


# ---- 推荐模型目录 + 设备检测（2026-09-08：下载改下拉推荐，按内存自动过滤） ----

MODEL_CATALOG = [
    {"name": "qwen2.5vl:7b", "purpose": "视觉主力：照片描述 / 地理验证 / 意图解析",
     "size_gb": 6, "min_ram_gb": 16, "vision": True, "star": True},
    {"name": "minicpm-v:8b", "purpose": "视觉备选：中文照片描述效果也不错",
     "size_gb": 5.5, "min_ram_gb": 16, "vision": True, "star": False},
    {"name": "llava:7b", "purpose": "视觉轻量：老牌图文模型，速度较快",
     "size_gb": 4.7, "min_ram_gb": 8, "vision": True, "star": False},
    {"name": "qwen3:8b", "purpose": "文本：问答 / 意图解析（不支持看图）",
     "size_gb": 5.2, "min_ram_gb": 16, "vision": False, "star": False},
    {"name": "qwen3:4b", "purpose": "文本轻量：小内存机器也能跑",
     "size_gb": 2.6, "min_ram_gb": 8, "vision": False, "star": True},
    {"name": "llama3.2:3b", "purpose": "文本轻量：Meta 小模型，英文强中文一般",
     "size_gb": 2, "min_ram_gb": 8, "vision": False, "star": False},
    {"name": "deepseek-r1:8b", "purpose": "推理：复杂问题深度思考，速度较慢",
     "size_gb": 5.2, "min_ram_gb": 16, "vision": False, "star": False},
]


_SIGLIP_OK = None  # None=未探测；部署到无 torch/transformers 的机器(NAS)时显示离线而非误导


def _siglip_available():
    """SIGLIP 跑在独立子进程解释器（默认 /usr/bin/python3，可用 SIGLIP_PYTHON 覆盖），
    探测必须针对同一解释器；探不到 torch/transformers 时前端显示离线（优雅降级）。"""
    global _SIGLIP_OK
    if _SIGLIP_OK is None:
        try:
            p = subprocess.run([os.environ.get("SIGLIP_PYTHON") or "/usr/bin/python3",
                                "-c", "import torch, transformers"],
                               capture_output=True, timeout=60)
            _SIGLIP_OK = (p.returncode == 0)
        except Exception:
            _SIGLIP_OK = False
    return _SIGLIP_OK


def device_profile():
    prof = {}
    ram_bytes = None
    try:  # macOS
        ram_bytes = int(subprocess.run(["sysctl", "-n", "hw.memsize"],
                                       capture_output=True, text=True, timeout=5).stdout.strip())
    except Exception:
        try:  # Linux (NAS)
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_bytes = int(line.split()[1]) * 1024
                        break
        except Exception:
            pass
    prof["ram_gb"] = round(ram_bytes / 1e9) if ram_bytes else None
    chip = ""
    try:  # macOS
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        pass
    if not chip:
        try:  # Linux
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        chip = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
    prof["chip"] = chip
    return prof


def models_catalog():
    dev = device_profile()
    installed = {m["name"] for m in ollama_models()["models"]}
    ram = dev.get("ram_gb")
    items = []
    for c in MODEL_CATALOG:
        d = dict(c)
        d["installed"] = c["name"] in installed
        d["fits"] = bool(ram is None or ram >= c.get("min_ram_gb", 8))
        d["recommended"] = bool(d["fits"] and not d["installed"] and c.get("star"))
        items.append(d)
    items.sort(key=lambda x: (not x["recommended"], not x["fits"], x["size_gb"] or 0))
    return {"device": dev, "catalog": items}


# ---- 云端服务商预设（2026-09-08：下拉即填，不用手抄 URL；2026-09-17 更新现役模型名） ----

PROVIDER_PRESETS = [
    {"name": "硅基流动 SiliconFlow", "base_url": "https://api.siliconflow.cn/v1",
     "model": "Qwen/Qwen3-VL-30B-A3B-Instruct", "free": False, "vision": True,
     "note": "注册送额度；Qwen3-VL-30B-A3B 快且准（全量 1 万张约 10 元），想更省可用 Qwen/Qwen3-VL-8B-Instruct。注意 Qwen2.5-VL 系列已下架"},
    {"name": "OpenRouter", "base_url": "https://openrouter.ai/api/v1",
     "model": "qwen/qwen2.5-vl-7b-instruct:free", "free": True, "vision": True,
     "note": "全球模型聚合，:free 后缀的模型不花钱（免费模型不稳定，建议付费档）"},
    {"name": "智谱 GLM", "base_url": "https://open.bigmodel.cn/api/paas/v4",
     "model": "glm-4.6v-flash", "free": True, "vision": True,
     "note": "GLM-4.6V-Flash 官方免费但有严格限流（~1张/3-5分钟），只适合小批量试；文本免费可用 glm-4.7-flash"},
    {"name": "通义千问 DashScope", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "model": "qwen-vl-plus", "free": False, "vision": True,
     "note": "阿里云，新用户有免费额度"},
    {"name": "月之暗面 Kimi", "base_url": "https://api.moonshot.cn/v1",
     "model": "moonshot-v1-8b-vision-preview", "free": False, "vision": True,
     "note": "长文本见长，按量付费"},
]

# ---- 算法偏好（2026-09-08：把关键逻辑阈值开放给用户调） ----

ALGO_SETTINGS = [
    {"key": "face_knn_min_sim", "label": "人脸归属严格度", "type": "range",
     "min": 0.45, "max": 0.70, "step": 0.01, "default": "0.55",
     "desc": "越高越严格：认错人更少，但更多脸留待你手标（当前参照库下 0.50~0.60 都安全）"},
    {"key": "face_child_strict", "label": "儿童加严归属", "type": "bool", "default": "1",
     "desc": "12 岁以下人物用更高阈值 + 更高投票比例，防止大人和小孩互相认错"},
    {"key": "person_coop_enabled", "label": "人物共现传播（合照归地）", "type": "bool", "default": "1",
     "desc": "合照里的人物与某地区参照照重合 ≥50% 且画面相符 → 自动归入该地区"},
    {"key": "sim_propagate_enabled", "label": "视觉相似传播", "type": "bool", "default": "1",
     "desc": "长得像某地区高置信照片的无位置照片自动归入"},
    {"key": "sim_min_score", "label": "相似传播门槛", "type": "range",
     "min": 0.45, "max": 0.75, "step": 0.01, "default": "0.60",
     "desc": "画面与参照地区的相似度需达到此值才传播；调高更保守"},
    {"key": "coop_ratio", "label": "合照归地·人物重合率", "type": "range",
     "min": 0.30, "max": 0.90, "step": 0.05, "default": "0.50",
     "desc": "照片里的人物与该地区参照照人物的重合比例要求；调高更保守"},
    {"key": "vision_geo_sample_n", "label": "地理复核抽样数", "type": "range",
     "min": 1, "max": 10, "step": 1, "default": "3",
     "desc": "每个待验证天数抽几张照片核验；抽得越多越准、耗时越长"},
    {"key": "vision_geo_verify_enabled", "label": "地理视觉复核", "type": "bool", "default": "1",
     "desc": "每天抽 3 张核验地点，不符自动撤销（手动标注永不撤销）"},
    # ---- 2026-09-17 产品化：个人数据/模型行为全部界面可配，不写死 ----
    {"key": "home_lat", "label": "家纬度", "type": "text",
     "desc": "「旅行」标签的家的基准纬度（十进制度，如 29.26）；留空则不生成旅行标签"},
    {"key": "home_lon", "label": "家经度", "type": "text",
     "desc": "家的基准经度（如 117.17），与纬度配套填写"},
    {"key": "vision_min_conf", "label": "语义判定严格度", "type": "range",
     "min": 0.50, "max": 0.95, "step": 0.01, "default": "0.80",
     "desc": "视觉模型判垃圾的置信度门槛，调高更保守（误杀更少、漏判更多）"},
    {"key": "vision_min_interval_sec", "label": "语义判定间隔(秒)", "type": "range",
     "min": 0, "max": 10, "step": 1, "default": "1",
     "desc": "两次视觉判定的最小间隔，防 API 限流；0 = 不限速"},
    {"key": "vision_prompt", "label": "语义判定提示词", "type": "textarea",
     "desc": "视觉模型的判定提示词，改后下一轮判定生效；JSON 输出格式那段勿删"},
    # ---- 2026-09-23 大疆/影石 Log 原片自动还原（灰片救回，原片不动） ----
    {"key": "logcolor_enabled", "label": "Log 原片自动还原", "type": "bool", "default": "1",
     "desc": "大疆 D-Log / 影石 Flat 那种灰蒙蒙的原片自动提亮补色；只改缩略图和预览，原片文件永不修改"},
    {"key": "logcolor_strength", "label": "还原力度", "type": "range",
     "min": 0.30, "max": 1.50, "step": 0.05, "default": "1",
     "desc": "1 = 默认实测最自然；嫌过艳往小调，还觉得发灰往大调（0.30 很轻 / 1.50 很浓）"},
]


def get_algo(key, default):
    return get_setting("algo_" + key, default)


def algo_settings_list():
    vals = []
    for s in ALGO_SETTINGS:
        d = dict(s)
        # 2026-09-18 修复：text/textarea 项（home_lat 等）没有 default 字段，
        # 硬取 s["default"] 会 KeyError 导致整个 /api/llm/status 挂掉
        d["value"] = get_algo(s["key"], s.get("default", ""))
        vals.append(d)
    return vals


def algo_settings_action(body):
    if body.get("action") == "reset":
        for s in ALGO_SETTINGS:
            set_setting("algo_" + s["key"], s.get("default", ""))
        return {"ok": True, "algos": algo_settings_list()}
    key = body.get("key") or ""
    meta = next((s for s in ALGO_SETTINGS if s["key"] == key), None)
    if not meta:
        return {"error": f"未知设置 {key}"}
    val = body.get("value")
    if meta["type"] == "range":
        try:
            val = round(float(val), 3)
            assert meta["min"] <= val <= meta["max"]
        except Exception:
            return {"error": f"取值须在 {meta['min']} ~ {meta['max']} 之间"}
    elif meta["type"] in ("text", "textarea"):
        val = str(val if val is not None else "").strip()
    else:
        val = "1" if str(val).lower() in ("1", "true", "yes", "on") else "0"
    set_setting("algo_" + key, str(val))
    return {"ok": True, "key": key, "value": str(val)}


def save_trip(body):
    con = sqlite3.connect(DB, timeout=20)
    con.row_factory = sqlite3.Row
    _init_user_trip_table(con)
    trip_id = body.get("trip_id") or ("trip_" + uuid.uuid4().hex[:12])
    name = body.get("name", "").strip()
    region = body.get("region", "").strip()
    province = body.get("province", "").strip()
    lat = body.get("latitude")
    lon = body.get("longitude")
    start = body.get("start_date", "").strip()
    end = body.get("end_date", "").strip()
    if not name or not region or not start or not end:
        con.close()
        return {"error": "名称、地区、开始/结束日期不能为空"}
    # 自动补全省份/坐标
    preset = GEO_CITY_PRESETS.get(region)
    if not province and preset:
        province = preset["province"]
    if lat is None and preset:
        lat = preset["lat"]
    if lon is None and preset:
        lon = preset["lon"]
    now = now_iso()
    district = str(body.get("district") or "").strip()
    con.execute(
        """INSERT OR REPLACE INTO user_trip
           (trip_id,name,region,province,district,latitude,longitude,start_date,end_date,created_at,updated_at)
           VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
        (trip_id, name, region, province, district, lat, lon, start, end,
         body.get("created_at") or now, now),
    )
    con.commit()
    con.close()
    result = request_rebuild_geo("trip save")
    result["trip_id"] = trip_id
    return result


def delete_trip(trip_id):
    con = sqlite3.connect(DB, timeout=20)
    con.execute("DELETE FROM user_trip WHERE trip_id=?", (trip_id,))
    con.commit()
    con.close()
    return request_rebuild_geo("trip delete")


# ============ HTTP ============
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.parse


def safe_static_path(url_path):
    """把 URL 路径映射为 STATIC 目录内的文件路径；路径穿越（越出 STATIC）返回 None。

    2026-09-04 从 do_GET 内联逻辑抽出，行为不变，便于单测安全边界。
    """
    f = (STATIC / "index.html") if url_path == "/" else (STATIC / url_path.lstrip("/"))
    # 2026-09-02 安全修复: 防路径穿越(如 /../server.py 读出源码)。
    # 用 resolve 归一化后必须仍落在 STATIC 目录内, 否则视为越界。
    try:
        f = f.resolve()
        f.relative_to(STATIC.resolve())
    except (ValueError, OSError):
        return None
    return f


class Handler(BaseHTTPRequestHandler):
    # 2026-09-12 keep-alive 升级：HTTP/1.0 下每张缩略图都新建 TCP+线程，浏览器对
    # 无 keep-alive 主机只开 6 条并发，快速滚动时上千请求排队，可视区域迟迟不渲染
    # （体感"局域网也卡"）。升 HTTP/1.1 复用连接；兜底：本响应没发 Content-Length
    # 就强制 close_connection（EOF 定界），任何旧发送路径都不会悬挂。
    protocol_version = "HTTP/1.1"

    def send_response(self, *a, **k):
        self._has_cl = False
        super().send_response(*a, **k)

    def send_header(self, keyword, value):
        if str(keyword).lower() == "content-length":
            self._has_cl = True
        super().send_header(keyword, value)

    def end_headers(self):
        if not getattr(self, "_has_cl", False):
            self.close_connection = True
        super().end_headers()

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # 浏览器取消缩略图或视频请求属于正常行为。

    def _send_auth_result(self, gate):
        """鉴权门的统一响应：302 重定向（页面）或 JSON 错误（API）。"""
        if "redirect" in gate:
            self.send_response(302)
            self.send_header("Location", gate["redirect"])
            self.end_headers(); return
        self.send_response(gate["status"])
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(gate["json"], ensure_ascii=False).encode())

    def do_POST(self):
        # ===== 施工图#9 鉴权门（在一切路由之前；公开路径在 _auth_gate 内放行）=====
        gate = _auth_gate(self, "POST")
        if gate is not None:
            self._send_auth_result(gate); return
        _strip_token_param(self)
        # ---- 在线升级：下载远程包 / 配置更新源（仅 admin）----
        if self.path == "/api/update/fetch":
            user, _ = _current_session_user(self)
            if not user or user.get("role") != "admin":
                self.send_response(403)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "需要管理员权限"}, ensure_ascii=False).encode()); return
            man = update_check_remote()
            if not man:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "远程更新源未配置或不可达"}, ensure_ascii=False).encode()); return
            ok, msg = update_fetch_remote(man)
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "message": msg}, ensure_ascii=False).encode()); return
        if self.path == "/api/update/feed":
            user, _ = _current_session_user(self)
            if not user or user.get("role") != "admin":
                self.send_response(403)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "需要管理员权限"}, ensure_ascii=False).encode()); return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
            except Exception:
                body = {}
            url = str(body.get("url") or "").strip()
            if url and not url.startswith(("http://", "https://")):
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "更新源地址必须以 http(s):// 开头"}, ensure_ascii=False).encode()); return
            set_setting("update_feed_url", url)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "feed": url}, ensure_ascii=False).encode()); return
        # ---- 在线升级应用（仅 admin；替换文件后自动重启服务）----
        if self.path == "/api/update/apply":
            user, _ = _current_session_user(self)
            if not user or user.get("role") != "admin":
                self.send_response(403)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                self.wfile.write(json.dumps({"ok": False, "error": "需要管理员权限"}, ensure_ascii=False).encode()); return
            try:
                ok, msg = update_apply()
            except Exception as exc:   # 2026-09-14 审计修复：任何异常回 400 JSON 而不是断连
                print(f"[api-error] /api/update/apply: {exc}", flush=True)
                ok, msg = False, f"更新失败: {exc}"
            self.send_response(200 if ok else 400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps({"ok": ok, "message": msg}, ensure_ascii=False).encode()); return
        # ---- 登录 ----
        if self.path == "/api/auth/login":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            _login_username = str(body.get("username") or "").strip()
            _login_ip = self.client_address[0]
            _login_wait = _login_rate_check(_login_ip, _login_username)
            if _login_wait is not None:
                self.send_response(429)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Retry-After", str(_login_wait))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(json.dumps(
                    {"ok": False, "error": f"登录尝试过于频繁，请约 {max(1, _login_wait // 60)} 分钟后再试"},
                    ensure_ascii=False).encode())
                return
            ok, info = auth_login(_login_username,
                                  str(body.get("password") or ""), bool(body.get("remember")))
            _login_rate_record(_login_ip, _login_username, ok)
            if ok:
                cookie = (f"{AUTH_COOKIE}={info['token']}; Path=/; HttpOnly; SameSite=Lax; "
                          f"Max-Age={info['max_age']}")
                payload = {"ok": True, "role": info["role"],
                           "username": info["username"], "token": info["token"]}
                status = 200
            else:
                cookie, payload, status = None, {"ok": False, "error": info}, 401
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            if cookie:
                self.send_header("Set-Cookie", cookie)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode())
            return
        # ---- 品牌标题（/api/people 特例同款：公开读，写在 handler 内校验 admin）----
        if self.path == "/api/brand":
            user, _ = _current_session_user(self)
            if not user or user.get("role") != "admin":
                self.send_response(403)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                self.wfile.write(json.dumps({"error": "需要管理员权限"}, ensure_ascii=False).encode()); return
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0) or 0)) or b"{}")
            except Exception:
                body = {}
            text = str(body.get("text") or "").strip()
            if not text or len(text) > 60:
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                self.wfile.write(json.dumps({"error": "标题需 1-60 字符"}, ensure_ascii=False).encode()); return
            set_setting("brand_text", text)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "text": text}, ensure_ascii=False).encode()); return
        # ---- 登出 ----
        if self.path == "/api/auth/logout":
            _, token = _current_session_user(self)
            auth_logout(token)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Set-Cookie", f"{AUTH_COOKIE}=; Path=/; HttpOnly; Max-Age=0")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"ok": true}')
            return
        # ---- 本人修改密码（登录即可，非 admin 专用；admin 重置走 /api/auth/users action=passwd）----
        if self.path == "/api/auth/passwd":
            user, token = _current_session_user(self)
            if not user:
                self.send_response(401)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write('{"ok": false, "error": "未登录"}'.encode())
                return
            if user["role"] == "guest":
                # 访客账号多为共享：不让访客改密码，避免一个人改完全家被锁外
                self.send_response(403)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write('{"ok": false, "error": "访客账户请联系管理员重置密码"}'.encode())
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            ok, msg = auth_change_own_password(user["user_id"],
                                               str(body.get("old_password") or ""),
                                               str(body.get("new_password") or ""), token)
            result, status = ({"ok": True} if ok else {"ok": False, "error": msg}), 200
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode())
            return
        # ---- 账号管理（admin 专用，前缀已在门里拒绝非 admin）----
        if self.path == "/api/auth/users":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            action = str(body.get("action") or "create")
            if action == "create":
                ok, msg = auth_user_create(str(body.get("username") or ""),
                                           str(body.get("password") or ""),
                                           str(body.get("role") or "family"))
                result, status = ({"ok": True} if ok else {"ok": False, "error": msg}), 200
            elif action == "delete":
                ok, msg = auth_user_delete(str(body.get("user_id") or ""))
                result, status = ({"ok": True} if ok else {"ok": False, "error": msg}), 200
            elif action == "passwd":
                ok, msg = auth_user_set_password(str(body.get("user_id") or ""),
                                                 str(body.get("new_password") or ""))
                result, status = ({"ok": True} if ok else {"ok": False, "error": msg}), 200
            else:
                result, status = {"ok": False, "error": "未知操作"}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode())
            return
        # ===== 施工图#5 初始化向导 API（向导完成后永久禁用）=====
        if self.path in ("/api/init/admin", "/api/init/validate-dir", "/api/init/finish",
                         "/api/init/browse-dir"):
            # admin 接口不预检（原子 SQL 自防并发）；validate/finish 以"已登记照片源"为完成门禁
            if self.path != "/api/init/admin":
                # 2026-09-14 审计修复：browse-dir/validate/finish 还要求管理员尚不存在——
                # admin 已建但还没配照片源的窗口期，匿名用户不得浏览服务器目录
                _has_admin, has_source = wizard_state()
                if _has_admin or has_source:
                    self.send_response(403); self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.end_headers()
                    self.wfile.write(json.dumps({"ok": False, "msg": "相册已初始化，向导已禁用"},
                                                ensure_ascii=False).encode()); return
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            result, status = {"ok": False, "msg": "未知操作"}, 400
            if self.path == "/api/init/browse-dir":
                try:
                    result, status = init_browse_dir(str(body.get("path") or "")), 200
                except Exception as exc:
                    result, status = {"ok": False, "msg": f"浏览失败：{exc}"}, 200
            elif self.path == "/api/init/admin":
                result["ok"], result["msg"] = init_create_admin(
                    str(body.get("username") or ""), str(body.get("password") or ""))
                status = 200
            elif self.path == "/api/init/validate-dir":
                ok, info = init_validate_dir(str(body.get("path") or ""))
                result = ({"ok": True, "info": info} if ok else {"ok": False, "msg": info})
                status = 200
            elif self.path == "/api/init/finish":
                try:
                    ok, info = init_validate_dir(str(body.get("path") or ""))
                    if not ok:
                        result, status = {"ok": False, "msg": info}, 200
                    else:
                        init_add_source(str(body.get("path")))
                        init_start_scan(str(body.get("path")))
                        result, status = {"ok": True}, 200
                except Exception as exc:
                    print(f"[init-wizard] finish 异常: {exc}", flush=True)
                    result, status = {"ok": False, "msg": f"登记失败: {exc}"}, 500
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/people":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if body.get("action") == "detail":
                    result = get_person_detail(body.get("person_id"), confirmed_limit=body.get("confirmed_limit", 16))
                    status = 200
                elif not body.get("action"):
                    result = get_people_payload()
                    status = 200
                else:
                    # 施工图#9：update_people 是写操作，仅 admin
                    user, _ = _current_session_user(self)
                    if not user or user["role"] != "admin":
                        result, status = {"error": "只读账户，无权修改人物"}, 403
                    else:
                        result, status = update_people(body), 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/face/upload-preview":
            try:
                result, status = handle_face_upload_preview(self)
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/face/create-from-upload":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result, status = handle_face_create_from_upload(body)
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/ab-review":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                result = save_ab_review_label(json.loads(self.rfile.read(length) or b"{}"))
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode()); return
        if self.path in ("/api/asset/faces", "/api/asset/search", "/api/unlabeled/frequent", "/api/unlabeled/similar"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/asset/faces":
                    _aid = (body.get("asset_id") or "").strip()
                    # 2026-09-14 隐私/回收站门：隐藏照片的人脸框不下发
                    if _aid:
                        privacy_ids, recycle_ids = _hidden_media_ids()
                        if _aid in privacy_ids and not _privacy_session_unlocked(self):
                            result, status = {"error": "照片不可访问"}, 403
                            self.send_response(status)
                            self.send_header("Content-Type", "application/json; charset=utf-8")
                            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                            self.wfile.write(json.dumps(result, ensure_ascii=False).encode()); return
                        if _aid in recycle_ids:
                            user, _ = _current_session_user(self)
                            if (user or {}).get("role") not in ("admin", "family"):
                                result, status = {"error": "照片不可访问"}, 403
                                self.send_response(status)
                                self.send_header("Content-Type", "application/json; charset=utf-8")
                                self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
                                self.wfile.write(json.dumps(result, ensure_ascii=False).encode()); return
                    result = get_asset_faces(_aid)
                elif self.path == "/api/unlabeled/frequent":
                    limit = max(1, min(50, int(body.get("limit") or 20)))
                    samples = max(1, min(12, int(body.get("samples") or 4)))
                    result = get_frequent_unlabeled_clusters(limit=limit, samples_per_cluster=samples)
                elif self.path == "/api/unlabeled/similar":
                    # threshold=0（或负数）= 不过滤，加载全部未标注脸仅按相似度排序
                    _th = body.get("threshold")
                    if _th is None:
                        _th = 0.50  # 2026-09-07：0.40 会把成年男性推荐进儿童池
                    result = get_similar_unlabeled_faces(
                        body.get("person_id"),
                        threshold=float(_th),
                        limit=max(1, min(9000, int(body.get("limit") or 1000))))
                else:
                    result = search_face_assets(body.get("q"), include_all=bool(body.get("include_all")))
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode()); return
        if self.path == "/api/check-new":
            result = check_new_files()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/categories":
            result = get_categories()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path in ("/api/trip/list", "/api/trip/save", "/api/trip/delete", "/api/trip/rebuild",
                         "/api/scene/rebuild", "/api/geo/vision-verify",
                         "/api/faces/run", "/api/faces/status",
                         "/api/geo/seed", "/api/geo/regions",
                         "/api/models/status", "/api/models/toggle",
                         "/api/llm/status", "/api/llm/provider", "/api/llm/priority",
                         "/api/llm/deepseek-key", "/api/llm/ollama",
                         "/api/llm/model-sel",
                         "/api/models/catalog", "/api/settings/algos",
                         "/api/settings/objects",
                         "/api/vlm/autorun", "/api/vlm/run", "/api/vlm/status",
                         "/api/tasks/config", "/api/tasks/warm", "/api/tasks/videolc"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/trip/list":
                    result = list_trips()
                elif self.path == "/api/trip/save":
                    result = save_trip(body)
                elif self.path == "/api/trip/delete":
                    result = delete_trip(body.get("trip_id", ""))
                elif self.path == "/api/scene/rebuild":
                    # 2026-09-02: 全库 SIGLIP 打分耗时分钟级, 原实现同步阻塞 HTTP 线程;
                    # 改后台线程执行, 立即返回 started, 面板轮询 _SCENE_REBUILD_RUNNING。
                    if _SCENE_REBUILD_RUNNING:
                        result = {"started": False, "already_running": True}
                    else:
                        threading.Thread(target=rebuild_scene_tags, daemon=True).start()
                        result = {"started": True}
                elif self.path == "/api/models/status":
                    result = models_status()
                elif self.path == "/api/models/toggle":
                    result = models_toggle(body)
                elif self.path == "/api/llm/status":
                    result = llm_providers()
                elif self.path == "/api/llm/provider":
                    result = llm_provider_action(body)
                elif self.path == "/api/llm/priority":
                    result = llm_priority_action(body)
                elif self.path == "/api/llm/deepseek-key":
                    result = deepseek_key_action(body)
                elif self.path == "/api/llm/model-sel":
                    result = llm_model_sel_action(body)
                elif self.path == "/api/llm/ollama":
                    result = ollama_action(body)
                elif self.path == "/api/models/catalog":
                    result = models_catalog()
                elif self.path == "/api/settings/algos":
                    result = algo_settings_action(body)
                elif self.path == "/api/settings/objects":
                    result = (objects_vocabulary_action(body) if body.get("action")
                              else objects_vocabulary())
                elif self.path == "/api/faces/run":
                    result = faces_backfill_run()
                elif self.path == "/api/faces/status":
                    result = faces_backfill_status()
                elif self.path == "/api/vlm/autorun":
                    en = body.get("enabled")
                    iv = body.get("interval_min")
                    result = vlm_autorun_config(
                        None if en is None else int(en),
                        None if iv is None else int(iv))
                elif self.path == "/api/vlm/run":
                    result = vlm_run("manual")
                elif self.path == "/api/vlm/status":
                    result = vlm_status()
                elif self.path == "/api/tasks/config":
                    result = tasks_config_action(body)
                elif self.path == "/api/tasks/warm":
                    result = warm_run("manual")
                elif self.path == "/api/tasks/videolc":
                    result = videolc_run("manual")
                elif self.path == "/api/geo/seed":
                    result = geo_seed(body)
                elif self.path == "/api/geo/regions":
                    result = {"provinces": _geo_region_groups(), "regions": _geo_region_coords()}
                elif self.path == "/api/geo/vision-verify":
                    # 视觉模型辅助验证：立即返回状态，验证在后台跑（每天抽 3 张，结果按天缓存）
                    if body.get("force"):
                        c = sqlite3.connect(DB, timeout=10)
                        c.execute("DELETE FROM vision_geo_check_v0")
                        c.commit()
                        c.close()
                    if not _VISION_GEO_RUNNING:
                        threading.Thread(target=vision_verify_geo, daemon=True).start()
                    result = {"started": True, **vision_geo_status()}
                else:
                    result = rebuild_geo()
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/asset":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                action = body.get("action")
                if action == "set_source":
                    result = set_asset_source(body)
                elif action == "reassign_person":
                    result = reassign_person(body)
                elif action == "move_scene":
                    result = move_scene_tags(body)
                elif action == "add_to_category":
                    result = add_to_category(body)
                else:
                    raise ValueError(f"未知 action: {action}")
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/category":
            length = int(self.headers.get("Content-Length", 0) or 0)
            status = 200
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                # body 可能是 null/list/str 等非法结构, 统一归零为 {} 防 AttributeError 断连
                if not isinstance(body, dict):
                    body = {}
                result = search_by_category(body.get("cat", ""), body.get("value", ""), body.get("order"),
                                            slim=bool(body.get("slim")),
                                            offset=body.get("offset"), limit=body.get("limit"),
                                            quality=body.get("quality"))
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            payload = json.dumps(result, ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            # 2026-09-06：分类列表动辄几百 KB～数 MB，客户端支持时 gzip 压缩（压到 1/8 左右）。
            # 浏览器 fetch 对 Content-Encoding: gzip 自动解压，前端无感知。
            if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                payload = gzip.compress(payload, 5)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/api/geo/map":
            status = 200
            try:
                result = geo_map_data()
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            payload = json.dumps(result, ensure_ascii=False, default=str).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            if "gzip" in (self.headers.get("Accept-Encoding") or ""):
                payload = gzip.compress(payload, 5)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if self.path == "/api/custom":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = custom_category_action(body)
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/catman":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = catman_action(body)
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path in ("/api/privacy/auto",):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = privacy_auto_action(body)
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path in ("/api/privacy",):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = privacy_action(body)
                status = 200
                # 2026-09-14 隐私门：PIN 验证成功 → 该会话 30 分钟内可读隐私媒体
                if body.get("action") == "verify" and result.get("ok"):
                    _privacy_mark_unlocked(self)
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path in ("/api/crop",):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = crop_action(body)
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/recycle":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                result = recycle_action(body)
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path in ("/api/filter/list", "/api/filter/add", "/api/filter/remove", "/api/filter/count"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    body = {}
                if self.path == "/api/filter/count":
                    result = get_filter_count()
                elif self.path == "/api/filter/list":
                    result = get_filter_list(body.get("order", "desc"))
                elif self.path == "/api/filter/add":
                    items = body.get("items") or [body]
                    result = filter_add(items)
                else:
                    ids = body.get("asset_ids") or ([body["asset_id"]] if body.get("asset_id") else [])
                    result = filter_remove(ids)
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        # 2026-09-16 新增：增量富化流水线的状态查询与手动触发
        if self.path in ("/api/enrich/status", "/api/enrich/run"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    body = {}
                if self.path == "/api/enrich/status":
                    result = enrich_status()
                else:
                    if self.path == "/api/enrich/run" and body.get("repick_all"):
                        set_setting("enrich_repick_all", "1")
                    result = enrich_run(trigger="manual")
                    result["status"] = enrich_status()
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path in ("/api/similar/list", "/api/similar/scan", "/api/similar/set-best",
                         "/api/similar/repick", "/api/similar/ungroup"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/similar/list":
                    result = get_similar_groups(body.get("engine"))
                elif self.path == "/api/similar/scan":
                    result = compute_similar_groups(
                        time_window=body.get("time_window", SIMILAR_TIME_WINDOW_SECONDS),
                        min_group_size=int(body.get("min_group_size", 2)),
                        engine=body.get("engine", "timewin-v1")
                    )
                elif self.path == "/api/similar/set-best":
                    result = similar_set_best(body.get("group_id", ""), body.get("asset_id", ""))
                elif self.path == "/api/similar/repick":
                    result = similar_repick(body.get("group_id", ""))
                else:
                    result = similar_ungroup(body.get("group_id", ""))
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path in ("/api/source/list", "/api/source/add", "/api/source/remove",
                         "/api/source/toggle", "/api/source/scan", "/api/source/scan_status",
                         "/api/source/scanning", "/api/source/discover", "/api/source/autoscan"):
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/source/list":
                    result = source_list()
                elif self.path == "/api/source/add":
                    result = source_add(body.get("path", ""), body.get("label", ""),
                                        body.get("source_type", "local_folder"))
                elif self.path == "/api/source/remove":
                    result = source_remove(body.get("source_id", ""))
                elif self.path == "/api/source/toggle":
                    result = source_set_enabled(body.get("source_id", ""), bool(body.get("enabled", True)))
                elif self.path == "/api/source/scan":
                    result = source_scan(body.get("source_id", ""))
                elif self.path == "/api/source/scan_status":
                    result = source_scan_status(body.get("scan_id", ""))
                elif self.path == "/api/source/scanning":
                    result = source_scanning()
                elif self.path == "/api/source/discover":
                    result = source_discover()
                else:
                    result = autoscan_config(
                        body.get("enabled") if "enabled" in body else None,
                        body.get("interval_min") if "interval_min" in body else None)
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
            return
        if self.path == "/api/ask":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            # 2026-09-04 修复: body 可能是 null/list/str/int（非法 JSON 结构），
            # 此时 body.get 直接 AttributeError 断连。统一归零为 {}。
            if not isinstance(body, dict):
                body = {}
            # 2026-09-04 修复: body={"question":null} 时 get 的默认值 "" 不生效,
            # question 仍为 None, 下游 fallback_parse 里 "x" in question 直接
            # TypeError('NoneType' is not iterable)。统一 or "" 兜底。
            question = body.get("question") if isinstance(body.get("question"), str) else ""
            t0 = time.time()
            # 2026-09-02 修复: search/LLM 内部异常时原实现直接抛 500 HTML 断连,
            # 前端拿到非 JSON。包一层返回 JSON 错误。
            try:
                result = search(question)
                llm_text = llm_synthesize_answer(question, result)
                if llm_text:
                    result["answer"] = llm_text
                    result["answer_source"] = "llm"
                else:
                    result["answer_source"] = "local"
            except Exception as exc:
                result = {"error": str(exc), "answer": f"查询出错: {exc}", "assets": []}
            result["elapsed_ms"] = int((time.time() - t0) * 1000)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
        elif self.path == "/api/captions":
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                body = {}
            try:
                if body.get("action") == "run":
                    result = captions_vlm_run()
                else:
                    result = captions_status()
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
        elif self.path == "/api/daily_best":
            # T4 接线（2026-09-15）：每日精选 —— 按美学分取某天前 N 张（默认最新那天取 5 张）
            length = int(self.headers.get("Content-Length", 0) or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    body = {}
                result = daily_best(body.get("day"), body.get("limit", 5))
                status = 200
            except Exception as exc:
                print(f"[api-error] {self.path}: {exc}", flush=True)
                result, status = {"error": str(exc)}, 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_HEAD(self):
        """HEAD 探活支持：复用 do_GET 的完整路由与响应头，丢弃响应体。
        背景：BaseHTTPRequestHandler 默认对 HEAD 返回 501，offline-guard.js
        等前端探活会把 501（>=500）误判为服务掉线，导致顶部一直挂掉线横幅。"""
        real_wfile = self.wfile
        state = {"headers_sent": False}

        class _HeadFilter:
            """头部照常写出（含 Content-Length，符合 RFC 7231 HEAD 语义），body 丢弃。"""
            def write(self, b):
                if not state["headers_sent"]:
                    real_wfile.write(b)
                return len(b)
            def flush(self):
                try: real_wfile.flush()
                except Exception: pass
            def close(self): pass

        _orig_flush = self.flush_headers
        def _flush_headers():
            _orig_flush()
            state["headers_sent"] = True
        self.flush_headers = _flush_headers
        self.wfile = _HeadFilter()
        try:
            self.do_GET()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        # ===== 施工图#5 初始化向导门禁（先于鉴权门：空库首启必须直达向导）=====
        _wiz_path = urllib.parse.urlparse(self.path).path
        _WIZ_HTML = {"/", "/index.html", "/library.html", "/people.html", "/person.html",
                     "/faces.html", "/ab-review.html", "/models.html"}
        if not is_initialized():
            # 向导未走完（含进行中）：HTML 页面一律留在向导页；API 与图片资源不受影响
            if _wiz_path in _WIZ_HTML:
                self.send_response(302)
                self.send_header("Location", "/init_wizard.html")
                self.end_headers(); return
        elif _wiz_path == "/init_wizard.html":
            # 已初始化：向导页永久禁用，回首页
            self.send_response(302)
            self.send_header("Location", "/")
            self.end_headers(); return
        # ===== 施工图#9 鉴权门（公开路径在 _auth_gate 内放行）=====
        gate = _auth_gate(self, "GET")
        if gate is not None:
            self._send_auth_result(gate); return
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/auth/me":
            user, _ = _current_session_user(self)
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            payload = ({"user": {"username": user["username"], "role": user["role"]}}
                       if user else {"user": None})
            self.wfile.write(json.dumps(payload, ensure_ascii=False).encode()); return
        if parsed.path == "/api/auth/users":
            result = auth_users_list()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps(result, ensure_ascii=False).encode()); return
        if parsed.path == "/api/brand":
            # 品牌标题（公开读；写入在 do_POST 内校验 admin）
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps({"text": get_setting("brand_text", "")}, ensure_ascii=False).encode()); return
        # token 已被上面公开路径消费完，摘掉避免干扰后续精确匹配路由
        _strip_token_param(self)
        parsed = urllib.parse.urlparse(self.path)
        # ===== 施工图#5 初始化向导门禁（已上移至 do_GET 开头、先于鉴权门）=====
        if parsed.path == "/api/init/status":
            has_admin, has_source = wizard_state()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps({"initialized": is_initialized(), "has_admin": has_admin,
                                         "has_source": has_source, "scan": INIT_SCAN["progress"]},
                                        ensure_ascii=False).encode()); return
        if parsed.path == "/api/update/check":
            # 在线升级检查（登录即可看；应用仅 admin，见 do_POST）：本地包 + 远程 feed 取高版本
            cand = update_best_package()
            remote = update_check_remote()
            avail = None; src = None
            if cand and _ver_tuple(cand["version"]) > _ver_tuple(APP_VERSION):
                avail, src = cand["version"], "local"
            if remote and _ver_tuple(remote["version"]) > _ver_tuple(APP_VERSION):
                if avail is None or _ver_tuple(remote["version"]) > _ver_tuple(avail):
                    avail, src = remote["version"], "remote"
            user, _ = _current_session_user(self)
            notes = (cand or {}).get("notes", "") if src == "local" else (remote or {}).get("notes", "") if src == "remote" else ""
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            self.wfile.write(json.dumps({"current": APP_VERSION, "available": avail, "source": src,
                                         "notes": notes,
                                         "remote_feed": remote["version"] if remote else None,
                                         "feed_url": get_update_feed_url(),
                                         "is_admin": bool(user and user.get("role") == "admin")},
                                        ensure_ascii=False).encode()); return
        if parsed.path == "/api/tasks":
            # 后台重活状态（2026-09-24）：Log 视频转码 + 缩略图预热。
            # 两者互斥串行（都是 4K 解码，并发即内存风暴），这里给界面/排障看进度。
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*"); self.end_headers()
            total, avail, load1 = res_snapshot()
            _, warn_mb, danger_mb = res_levels()
            self.wfile.write(json.dumps({
                "warm": warm_status(), "videolc": videolc_status(),
                "res": {"enabled": get_setting("res_guard_enabled", "1") == "1",
                        "total_mb": total, "avail_mb": avail, "load1": load1,
                        "warn_mb": warn_mb, "danger_mb": danger_mb,
                        "warn_mb_override": get_setting("res_warn_mb", ""),
                        "danger_mb_override": get_setting("res_danger_mb", ""),
                        "ffmpeg_rlimit_mb": res_ffmpeg_rlimit() // 1048576,
                        "tier": machine_profile()["tier"],
                        "video_concurrency": machine_profile()["video_concurrency"],
                        "transcode_rlimit_mb": machine_profile()["transcode_rlimit_mb"]}},
                                        ensure_ascii=False).encode()); return
        if parsed.path == "/api/asset_info":
            # 2026-09-07 查看器信息条：时间/地点/标签/人物，小字追加在页码后
            aid = urllib.parse.parse_qs(parsed.query).get("asset", [""])[0]
            con = sqlite3.connect(DB, timeout=10)
            con.row_factory = sqlite3.Row
            try:
                row = con.execute("SELECT capture_time FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
                g = con.execute("SELECT region, province FROM asset_geo_v0 WHERE asset_id=?", (aid,)).fetchone()
                tags = [r[0] for r in con.execute(
                    "SELECT tag FROM scene_tag_v0 WHERE asset_id=? ORDER BY confidence DESC LIMIT 8", (aid,))]
                persons = [r[0] for r in con.execute(
                    """SELECT DISTINCT p.display_name FROM face_instance_v0 fi
                       JOIN person p ON p.person_id=fi.person_id
                       WHERE fi.asset_id=? ORDER BY p.display_name LIMIT 8""", (aid,))]
            finally:
                con.close()
            result = {"asset_id": aid,
                      "time": (row["capture_time"] or "") if row else "",
                      "region": (g["region"] if g and g["region"] else ""),
                      "province": (g["province"] if g and g["province"] else ""),
                      "tags": tags, "persons": persons}
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False).encode()); return
        if parsed.path == "/api/ab-review":
            result = get_ab_review_payload()
            self.send_response(200); self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers(); self.wfile.write(json.dumps(result, ensure_ascii=False, default=str).encode()); return
        if parsed.path == "/view":
            aid = urllib.parse.parse_qs(parsed.query).get("asset", [""])[0]
            aid_q = urllib.parse.quote(aid, safe="")  # 2026-09-04 修复反射型 XSS：aid 未转义直接拼进 HTML 属性
            path = get_orig_path(aid)
            if not path:
                self.send_response(404); self.end_headers(); return
            con = sqlite3.connect(DB, timeout=10)
            row = con.execute("SELECT media_type FROM media_asset WHERE asset_id=?", (aid,)).fetchone()
            con.close()
            media_type = row[0] if row else "photo"
            name = html.escape(os.path.basename(path), quote=True)
            media = (f'<video src="/orig?asset={aid_q}" controls autoplay playsinline></video>' if media_type == "video" else f'<img src="/preview?asset={aid_q}" alt="{name}">')
            page = f'''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{name}</title><style>html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#111;color:#eee;font-family:-apple-system,"PingFang SC",sans-serif}}main{{width:100vw;height:100vh;display:grid;grid-template-rows:minmax(0,1fr) auto}}.media{{width:100%;height:100%;min-width:0;min-height:0;overflow:hidden;display:grid;place-items:center}}img,video{{display:block;width:100%;height:100%;max-width:100vw;max-height:calc(100vh - 44px);object-fit:contain;background:#000}}footer{{min-height:44px;padding:10px 18px;background:#1b1b1b;display:flex;align-items:center;justify-content:space-between;gap:15px;font-size:12px;overflow:hidden}}footer span{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}a{{color:#fff;white-space:nowrap}}</style><main><div class="media">{media}</div><footer><span>{name}</span><a href="/orig?asset={aid_q}" target="_blank">打开原始文件</a></footer></main>'''.encode("utf-8")
            self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(page))); self.end_headers(); self.wfile.write(page); return
        if parsed.path == "/preview":
            aid = urllib.parse.parse_qs(parsed.query).get("asset", [""])[0]
            if not media_gate_allows(self, aid):     # 2026-09-14 隐私/回收站门
                self.send_response(404); self.end_headers(); return
            # 2026-08-29 点开即原图：浏览器原生可显示的格式（JPG/PNG/WebP/GIF/BMP）直通原片，
            # 不再压 2200px 转码；HEIC/视频封面等仍走转码管线
            orig_path = get_orig_path(aid)
            # 2026-09-23 Log 原片不能走「直通原片」这条快路：直通就等于把没还原
            # 的灰片直接发出去，前面的还原全白做。命中还原则落到下面 get_preview
            # 走还原链（缓存名带 _lc，不会跟这条快路互相污染）。
            if orig_path and not logcolor_for(aid):
                ext = os.path.splitext(orig_path)[1].lower()
                if ext in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
                    size = os.path.getsize(orig_path)
                    if size <= 30 * 1024 * 1024:   # 超大原片（>30MB）仍走转码，避免加载卡死
                        self.send_response(302)
                        self.send_header("Location", f"/orig?asset={aid}")
                        self.send_header("Cache-Control", "max-age=86400")
                        self.end_headers(); return
            preview = get_preview(aid)
            if not preview:
                self.send_response(404); self.end_headers(); return
            data = preview.read_bytes()
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "max-age=86400"); self.end_headers(); self.wfile.write(data); return
        if parsed.path == "/orig":
            qs = urllib.parse.parse_qs(parsed.query)
            aid = qs.get("asset", [""])[0]
            if not media_gate_allows(self, aid):     # 2026-09-14 隐私/回收站门
                self.send_response(404); self.end_headers(); return
            path = get_orig_path(aid)
            if not path:
                self.send_response(404)
                self.end_headers()
                return
            # 2026-09-24 Log 视频转码缓存优先：DJI D-Log 是 4K HEVC Main 10（10bit），
            # Chrome 解不动 → video onerror（用户实锤「log视频打不开」）。videos_lc/
            # 里有转好的 H.264 8bit 版（色彩已还原、faststart 流式友好）就优先给；
            # 没有则照旧播原片（Safari/iPhone 能解 10bit）。转好一条生效一条。
            if logcolor_for(aid):
                _lc_file = VIDEO_LC_DIR / (aid[6:] + "_lc.mp4")
                try:
                    if _lc_file.exists() and _lc_file.stat().st_size > 0:
                        path = str(_lc_file)
                except OSError:
                    pass
            # 原片流式返回，支持 Range（大文件可分块加载）
            size = os.path.getsize(path)
            content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
            if path.lower().endswith(".mov"):
                content_type = "video/quicktime"
            range_header = self.headers.get("Range")
            start = end = None
            invalid_range = False
            if range_header:
                # 解析 Range: bytes=start-end / bytes=start- / bytes=-suffix
                # 2026-09-02 修复: 原实现把 bytes=-N 解析成 start=0,end=N-1(错),
                # 后缀语义是"末尾 N 字节", Safari 拖进度条会发这种请求导致画面错位。
                # 2026-09-13 修复: 非法 Range(如 bytes=abc)按 RFC 忽略返回 200 全量；
                # start 超出文件尾(播放器 seek 越界)返回 416, 旧实现会发
                # 206 + 负数 Content-Length, 可能导致播放器异常。
                # 2026-09-13 补充: bytes=5-2 (start>end) 旧实现发 206+负数
                # Content-Length → 改判 416; 多段 Range (bytes=0-1,5-6) 不支持
                # (旧实现静默只发第一段) → 按 RFC 当无效 Range 忽略, 返回 200 全量。
                import re
                m = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
                if m and (m.group(1) or m.group(2)):
                    if m.group(1) == "":
                        suffix = int(m.group(2))
                        start = max(0, size - suffix)
                        end = size - 1
                    else:
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else size - 1
                else:
                    invalid_range = True
            if start is not None and (start >= size or start > end):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Type", content_type)
                self.end_headers()
                return
            if start is not None and not invalid_range:
                end = min(end, size - 1)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(end - start + 1))
            else:
                start = end = None
                self.send_response(200)
                self.send_header("Content-Length", str(size))
                self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "max-age=3600")
            self.end_headers()
            self._stream_file(path, start, end)
            return
        if parsed.path == "/api/facepos":
            result = get_face_positions()
            body = json.dumps(result, ensure_ascii=False, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "max-age=600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/thumb":
            qs = urllib.parse.parse_qs(parsed.query)
            aid = qs.get("asset", [""])[0]
            if not media_gate_allows(self, aid):     # 2026-09-14 隐私/回收站门
                self.send_response(404); self.end_headers(); return
            try:                                    # 分档：全站统一 480 档（1600 已废弃）
                edge = int(qs.get("edge", [str(THUMB_EDGE)])[0])
            except ValueError:
                edge = THUMB_EDGE
            # 2026-09-14 审计修复：edge 量化到固定档位——连续值会以 1px 步进灌满缓存目录
            _EDGES = (120, 160, 240, 320, 400, 480, 600, 800, 960, 1200, 1600)
            edge = min(_EDGES, key=lambda s: abs(s - edge))
            thumb = get_thumb(aid, edge)
            if thumb and thumb.exists():
                data = thumb.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=3600")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
            return
        if parsed.path == "/face_crop":
            qs = urllib.parse.parse_qs(parsed.query)
            fid = qs.get("face", [""])[0]
            if fid:                                  # 2026-09-14 隐私/回收站门（fid→asset 反查）
                _c = sqlite3.connect(DB, timeout=10)
                try:
                    _row = _c.execute("SELECT asset_id FROM face_instance_v0 WHERE face_instance_id=?",
                                      (fid,)).fetchone()
                finally:
                    _c.close()
                if _row and not media_gate_allows(self, _row[0]):
                    fid = ""
            try:
                k = float(qs.get("k", ["3.6"])[0])
                size = int(qs.get("size", ["460"])[0])
            except ValueError:
                k, size = 3.6, 460
            k = max(1.5, min(k, 6.0))
            size = max(160, min(size, 800))
            # 2026-09-14 审计修复：size 量化到固定档位——连续值会让恶意请求以
            # 1px 步进灌满缓存目录（缓存键含 size）
            _SZ = (160, 240, 320, 460, 480, 640, 800)
            size = min(_SZ, key=lambda s: abs(s - size))
            crop = get_face_crop(fid, k=k, size=size) if fid else None
            if crop and crop.exists():
                data = crop.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            else:
                self.send_response(404)
                self.end_headers()
            return
        # 静态文件（路径穿越防护收敛在 safe_static_path，2026-09-04 抽出以便单测）
        f = safe_static_path(parsed.path)
        if f is None:
            self.send_response(404)
            self.end_headers()
            return
        if f.exists() and f.is_file():
            ctype = {".html": "text/html", ".js": "application/javascript", ".css": "text/css"}.get(f.suffix, "application/octet-stream")
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            # 2026-08-29 页面/脚本禁止缓存：否则浏览器留着旧版 library.html，新功能看起来"没反应"
            # 2026-09-08 no-cache 仍允许协商缓存(304)个别场景漏更新, 升级 no-store 彻底禁掉
            if f.suffix in (".html", ".js", ".css"):
                self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(f.read_bytes())
        else:
            self.send_response(404)
            self.end_headers()

    # 原片下载分块大小。把 64KB 提到 1MB 并不是为了提速 —— 实测过三种传输方式
    # （64KB chunk / 1MB chunk / sendfile 零拷贝），在同一批文件上吞吐分别是
    # 42.0 / 44.9 / 42.5 MB/s，差异不足 7%，落在误差范围内：瓶颈在 SMB 读路径本身，
    # 不在 chunk 大小，也不在 read/write 的系统调用次数。
    # 仍然保留 1MB：往返次数更少，慢链路更稳，且实测不比 64KB 差。
    # sendfile 实测没有可测收益，故不启用，免得多一条失败回退路径。
    ORIG_CHUNK = 1 << 20

    def _stream_file(self, path, start=None, end=None):
        """把文件（或 Range 片段）写进响应体。调用前响应头必须已经 end_headers() 发出。

        start 为 None 表示整个文件；否则只发 [start, end] 闭区间，绝不越界一个字节。
        浏览器切换图片/拖动视频进度条时会主动掐断连接，BrokenPipe/ConnectionReset 静默吞掉。
        """
        offset = start or 0
        remaining = None if start is None else (end - start + 1)
        if remaining is not None and remaining <= 0:
            return  # 空 Range 或 start > end 的畸形请求：按原实现一样不发任何字节
        with open(path, "rb") as f:
            f.seek(offset)
            try:
                while True:
                    chunk = f.read(self.ORIG_CHUNK if remaining is None
                                   else min(self.ORIG_CHUNK, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    if remaining is not None:
                        remaining -= len(chunk)
                        if remaining <= 0:
                            break
            except (BrokenPipeError, ConnectionResetError):
                pass

    def log_message(self, fmt, *args):
        pass


def _fd_watchdog_loop():
    """FD 泄漏看门狗：每 60s 检查进程文件描述符数，超上限 70% 主动退出，
    由 launchd KeepAlive 拉起新进程，避免 FD 耗尽后服务假死一整天。"""
    try:
        import resource as _r
        soft, _hard = _r.getrlimit(_r.RLIMIT_NOFILE)
    except Exception:
        soft = 0
    sys.stderr.write(f"[fd-watchdog] RLIMIT_NOFILE soft={soft}，启动 FD 监控\n")
    while True:
        time.sleep(60)
        try:
            n = len(os.listdir("/dev/fd"))
        except Exception:
            continue
        if soft and n > soft * 0.7:
            sys.stderr.write(f"[fd-watchdog] FD {n}/{soft} 超 70%，主动重启\n")
            os._exit(3)


def _wrap_busy_503(orig):
    """2026-09-10 修 .app 启动锁死配套：do_GET/do_POST 顶层兜底。
    偶发 SQLite 锁等待超时（OperationalError: database is locked）时，
    返回 503 + JSON 提示并当场释放，绝不让异常沿 ThreadingHTTPServer
    冒泡拖垮主服务；前端可见「服务忙」而不是无限转圈。"""
    def inner(self, *args, **kwargs):
        try:
            return orig(self, *args, **kwargs)
        except sqlite3.OperationalError as exc:
            print(f"[busy-guard] {getattr(self, 'command', '?')} "
                  f"{getattr(self, 'path', '?')} → 503: {exc}", flush=True)
            try:
                body = json.dumps({"error": "服务忙，请稍后重试（数据库锁等待超时）"},
                                  ensure_ascii=False).encode()
                self.send_response(503)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except Exception:
                pass  # 响应头可能已发出，丢弃即可
    return inner


Handler.do_GET = _wrap_busy_503(Handler.do_GET)
Handler.do_POST = _wrap_busy_503(Handler.do_POST)


if __name__ == "__main__":
    # 2026-09-11 自查修复：端口占用防护（必须在 schema/WAL 初始化之前）。
    # 此前双开 .app 时，第二实例要先跑完建库/WAL/后台线程启动，最后才在
    # bind 处抛 OSError 整进程静默退出——用户毫无感知，还会误碰同一 DB。
    # 探测用与 _Server 完全相同的 bind 语义（SO_REUSEADDR + 127.0.0.1）：
    # - 另一个 .app 实例占着环回端口 → 探测失败，弹通知明确退出
    # - Docker 版占通配 *:PORT（更宽的监听）→ 环回仍可绑定，放行
    #   （桌面版 127.0.0.1 与 Docker 版 0.0.0.0 共存是合法用法，不能误杀）
    import socket as _socket
    _probe = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    _probe.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    try:
        _probe.bind(("127.0.0.1", PORT))
    except OSError:
        _msg = f"端口 {PORT} 已被占用，可能已有一个相册实例在运行。请先退出它。"
        print(f"[startup] {_msg}", flush=True)
        try:
            subprocess.run(["osascript", "-e",
                            f'display notification "{_msg}" with title "家庭记忆相册"'],
                           timeout=5, capture_output=True)
        except Exception:
            pass
        sys.exit(0)
    finally:
        _probe.close()
    print(f"家庭记忆 MVP 启动: http://localhost:{PORT}/")
    print(f"数据库: {DB}")
    # 施工图#4：自动建库（幂等）——必须在任何 DB 访问和线程启动之前。
    # 解决试装崩点 2：新用户首次启动 DB 不存在 → 主线程查表崩溃服务起不来。
    # 行为约定：现有完整库静默跳过（created=0 无输出，启动日志逐行不变）；
    # 空库/缺表时逐表补建 + 应用 migrations，绝不重建已有表。
    try:
        DB.parent.mkdir(parents=True, exist_ok=True)  # 无 data 目录时先建目录
        import schema as _schema
        _created, _mig = _schema.ensure_schema(DB)
        if _created or _mig:
            print(f"[schema] 自动建库: 新建 {_created} 张表, 应用迁移 {_mig}", flush=True)
        # 2026-09-10 二次锁死修复（修改单#3）：切 WAL 模式（持久属性，设一次即可）。
        # delete 模式的经典死结：写事务 PENDING 状态挡住一切新读者，而老读者
        # （跨 SMB 慢 IO 的缩略图线程）不退出 → 写者饿死 → 全站 503 含静态资源。
        # WAL 下读写互不阻塞，读者再多也挡不住写者。
        _wcon = sqlite3.connect(DB, timeout=30)
        _wmode = _wcon.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        _wcon.execute("PRAGMA synchronous=NORMAL")
        _wcon.close()
        print(f"[db] journal_mode={_wmode} synchronous=NORMAL（读写分离防锁死）", flush=True)
    except Exception as _exc:
        print(f"[schema] 自动建库失败: {_exc}", flush=True)
        raise
    # 启动时确保来源表/设置表就绪，并启动自动扫描线程
    _con0 = sqlite3.connect(DB, timeout=10)
    _ensure_source_schema(_con0)
    try:
        _nmerged = merge_recycle_into_filter(_con0)
        if _nmerged:
            print(f"[trash-merge] 回收站已并入已过滤内容: {_nmerged} 项", flush=True)
    except Exception as _exc:
        print(f"[trash-merge] 失败(忽略,不影响启动): {_exc}", flush=True)
        _con0.rollback()
    _con0.close()
    _cfg = autoscan_config()
    print(f"[autoscan] enabled={_cfg['enabled']} interval={_cfg['interval_min']}min", flush=True)
    # 多标签表为空时后台构建（合影/旅行/山景/海景），不阻塞服务启动
    ensure_scene_tags_async()
    threading.Thread(target=_autoscan_loop, daemon=True, name="autoscan").start()
    threading.Thread(target=_cleanup_orphan_cache, daemon=True, name="cache-cleanup").start()
    threading.Thread(target=_fd_watchdog_loop, daemon=True, name="fd-watchdog").start()
    _vcfg = vlm_autorun_config()
    print(f"[vlm-autorun] enabled={_vcfg['enabled']} interval={_vcfg['interval_min']}min", flush=True)
    threading.Thread(target=_vlm_autorun_loop, daemon=True, name="vlm-autorun").start()
    threading.Thread(target=_videolc_watchdog, daemon=True, name="videolc-watchdog").start()
    threading.Thread(target=_warm_watchdog, daemon=True, name="thumb-warm-wd").start()
    threading.Thread(target=_res_guard_watchdog, daemon=True, name="res-guard").start()
    _pauto = privacy_auto_config()
    # 2026-09-10 二次锁死修复（修改单#4）：启动序列不再预载 pending 统计
    # （1.3 万资产的大 NOT IN 查询），首次后台轮询时再查。
    print(f"[privacy-auto] enabled={_pauto['enabled']} interval={_pauto['interval_min']}min", flush=True)
    threading.Thread(target=_privacy_auto_loop, daemon=True, name="privacy-auto").start()
    # 绑定地址可配（2026-09-09 Docker）：容器内必须 0.0.0.0 否则端口转发 reset；
    # 宿主机直起默认 127.0.0.1 不对外暴露
    bind_host = os.environ.get("FF_BIND", "127.0.0.1")
    # 2026-09-10 二次锁死修复：listen backlog 默认仅 5，首屏并发连接溢出会被
    # 直接丢弃（客户端表现为超时/失败）。提到 128。
    class _Server(ThreadingHTTPServer):
        request_queue_size = 128
        daemon_threads = True
    server = _Server((bind_host, PORT), Handler)
    server.serve_forever()
