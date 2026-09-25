#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""家庭回忆相册发布流水线：开发目录 → 公开仓 → 在线升级包 → GitHub Release。

一次命令走完「同步代码 / 脱敏改版本号 / 打包 / 写 manifest / 提交打标签 / 建 Release」，
避免每次发版手敲七八条命令、漏改 manifest 或忘传包。

用法：
    python3 tools/release.py <版本号> "<更新说明>" [--with-worker] [--deploy]

  <版本号>        形如 1.0.10，必须同时写进 mvp 的 APP_VERSION 与 manifest
  --with-worker    把后台 worker 脚本（transcode_log_videos.py）一并随包下发；
                   注意目标机必须是 **已装过新白名单的版本**，否则整包会被 update_apply 拒收
                   （这是 2026-09-24 的血案，本脚本会就地校验白名单）
  --deploy         额外把更新包投放到运行中的服务器 updates 目录，
                   需准备环境变量 FM_SSH_HOST / FM_SSH_USER / FM_SSH_PASS，
                   **口令只从环境变量读，绝不写进本文件/仓库**（fork 出去给别人也能用）

环境变量：
    FM_MVP_DIR      开发目录（默认：公开仓的 ../mvp，即同级目录下的 mvp/）
"""

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REL = ROOT
# 开发目录与公开仓同级：…/家庭回忆相册/mvp 与 …/家庭回忆相册/release/ 并列
MVP = Path(os.environ.get("FM_MVP_DIR", str(ROOT.parent.parent / "mvp"))).resolve()

# 每次发版必须带上来的文件（相对路径）。static/ 由脚本自动收进去，不用手写。
DEFAULT_FILES = ["server.py", "static/library.html"]
WORKER = "transcode_log_videos.py"

# 公开仓红线：命中任何一条就中止（真名 / 内网 IP / 个人路径 / 口令 / 家庭坐标）
PATTERNS = {
    "真实姓名": r"徐国庆|徐鹏|xp03",
    "内网IP": r"\b10\.31\.\d+\.\d+\b",
    "个人路径": r"/Users/xupeng|/volume1/homes/xp03|/volume1/homes",
    "口令": r"Abc12345",
    "家庭坐标": r"home_lat|home_lon|家坐标",
}
# 上面「家庭坐标」命中的其实是设置项键名与界面文案，属安全项，单独放行
SAFE_HITS = {"家庭坐标": {"home_lat", "home_lon", "家坐标"}}

EXCLUDE_FILE = {".DS_Store", "quality-demo.html", "test-similar.html", "ab-review.html"}
EXCLUDE_DIR = {"design-preview", "design-notes"}


def is_junk(p: Path) -> bool:
    n = p.name
    if n in EXCLUDE_FILE or any(part in EXCLUDE_DIR for part in p.parts):
        return True
    if ".bak" in n or n.startswith("XX") or n.endswith(".pyc"):
        return True
    return False


def need(cond, msg):
    if not cond:
        print("✗ " + msg)
        sys.exit(1)


def sensitive_scan(files) -> bool:
    """公开仓红线扫描。返回 True = 有风险项需人工确认。"""
    risky = False
    for rel in files:
        f = REL / rel
        if not f.exists():
            continue
        txt = io.open(f, encoding="utf8", errors="ignore").read()
        for name, pat in PATTERNS.items():
            hits = [(i, l.strip()[:120]) for i, l in enumerate(txt.split("\n"), 1)
                    if re.search(pat, l)]
            if not hits:
                continue
            safe = all(any(s in l for s in SAFE_HITS.get(name, ())) for _, l in hits)
            if safe:
                print("· [%s] %s：%d 处（判定安全：%s）" % (name, rel, len(hits), "/".join(SAFE_HITS[name])))
                continue
            risky = True
            print("!! [%s] %s 命中 %d 处" % (name, rel, len(hits)))
            for i, l in hits[:5]:
                print("   L%d: %s" % (i, l))
    return risky


def sync_from_mvp(files, version):
    """mvp → release：复制 + 脱敏 + 改版本号。"""
    for rel in files:
        src, dst = MVP / rel, REL / rel
        need(src.is_file(), "开发目录里没有 %s（FM_MVP_DIR=%s）" % (rel, MVP))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        print("copy:", rel)

    p = REL / "server.py"
    s = io.open(p, encoding="utf8").read()
    n = s.count("徐国庆")
    s = s.replace("徐国庆", "某人")
    print("desens: 真名 %d 处 → 某人" % n)

    s2, cnt = re.subn(r'(?m)^APP_VERSION\s*=\s*"[^"]+"',
                      'APP_VERSION = "%s"' % version, s, count=1)
    need(cnt == 1, "server.py 里没找到唯一的 APP_VERSION 赋值，开发目录结构变了？")
    io.open(p, "w", encoding="utf8").write(s2)
    print("version ->", version)


def build_zip(version, notes, with_worker):
    server_py = (REL / "server.py").read_text(encoding="utf-8")
    m = re.search(r'APP_VERSION\s*=\s*["\']([^"\']+)"', server_py)
    need(m and m.group(1) == version, "server.py 里的 APP_VERSION 与命令行不一致")

    allow = re.search(r'UPDATE_ALLOWED_FILES\s*=\s*\{([^}]*)\}', server_py)
    need(allow, "server.py 里找不到 UPDATE_ALLOWED_FILES 白名单")
    allowed = set(re.findall(r'"([^"]+)"', allow.group(1)))

    files = [REL / "server.py"]
    if with_worker:
        need(WORKER in allowed,
             "%s 不在白名单里，打包下发会让旧版本整包拒收，先单独发一版铺白名单" % WORKER)
        need((REL / WORKER).is_file(), "release 仓里没有 %s" % WORKER)
        files.append(REL / WORKER)
        print("· 随包下发 worker:", WORKER)
    for p in sorted((REL / "static").rglob("*")):
        if p.is_file() and not is_junk(p.relative_to(REL)):
            files.append(p)

    out = ROOT / ("update_pkg")
    out.mkdir(exist_ok=True)
    name = "family_memory_update_v%s.zip" % version
    zpath = out / name
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, f.relative_to(REL).as_posix())
    raw = zpath.read_bytes()
    print("✓ %s: %d 文件, %.0f KB"
          % (name, len(files), len(raw) / 1024))
    return name, hashlib.md5(raw).hexdigest(), len(raw)


def write_manifest(version, name, md5, size, notes):
    man = {
        "version": version,
        "file": name,
        "md5": md5,
        "size": size,
        "notes": notes,
        "url": "https://github.com/pengxu795/family-memory-album/releases/download/%s/%s"
               % (version, name),
    }
    (REL / "update" / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("manifest ->", version, md5)


def run(cmd, **kw):
    print("$", " ".join(cmd)[:160])
    return subprocess.run(cmd, cwd=str(REL), **kw)


def git_release(version, notes, name):
    msg = "release v%s: %s" % (version, notes)
    if run(["git", "add", "-A"]).returncode:
        return False
    if run(["git", "commit", "-m", msg]).returncode:
        print("· 没有新改动可提交（或提交被拦），继续")
    run(["git", "tag", "-f", version if version.startswith("v") else "v" + version])
    run(["git", "push", "origin", "main", "--follow-tags"])
    gh = shutil.which("gh")
    if not gh:
        print("! 本机没有 gh，跳过 GitHub Release（手动上传 %s 即可）" % name)
        return False
    tag = version if version.startswith("v") else "v" + version
    r = subprocess.run([gh, "release", "list", "--limit", "3"], cwd=str(REL),
                       capture_output=True, text=True)
    exists = tag in (r.stdout or "")
    cmd = [gh, "release", "create" if not exists else "upload", tag, "-t", "v%s · %s" % (version, notes)]
    if not exists:
        cmd += ["-n", notes]
    cmd += [str(REL / "update_pkg" / name)]
    run(cmd)
    return True


def deploy_to_server(name):
    host = os.environ.get("FM_SSH_HOST")
    user = os.environ.get("FM_SSH_USER")
    pw = os.environ.get("FM_SSH_PASS")
    remote_dir = os.environ.get("FM_UPDATE_DIR", "/volume1/homes/xp03/family-album/data/updates")
    need(host and user and pw,
         "缺 FM_SSH_HOST / FM_SSH_USER / FM_SSH_PASS 任一环境变量，跳过 --deploy")
    local = (ROOT / "update_pkg" / name).read_bytes()
    tmp = "/tmp/%s" % name
    try:
        p = subprocess.run(["base64", "-i", "-"], input=local, capture_output=True, check=True)
        base64_pipe = p.stdout
    except Exception as e:
        print("! base64 失败:", e)
        return
    # 两步式：先落 /tmp（普通用户可写），再 sudo cp 到 root 属主的 updates 目录。
    # 不能把 sudo 和管道放同一条命令——sudo -S 会吃掉 stdin，出来是个空文件。
    subprocess.run(["ssh", "-o", "ConnectTimeout=15", "%s@%s" % (user, host),
                    "base64 -d > %s" % tmp], input=base64_pipe)
    subprocess.run(["ssh", "-o", "ConnectTimeout=15", "%s@%s" % (user, host),
                    "echo '%s' | sudo -S -p '' cp %s %s/%s && md5sum %s/%s"
                    % (pw, tmp, remote_dir, name, remote_dir, name)],
                   capture_output=True, text=True)
    print("· 已尝试投放到 %s@%s:%s（看上面的 md5sum 结果确认）" % (user, host, remote_dir))


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("version")
    ap.add_argument("notes")
    ap.add_argument("--with-worker", action="store_true")
    ap.add_argument("--deploy", action="store_true")
    ap.add_argument("--no-git", action="store_true", help="只打包，不改 git")
    a = ap.parse_args()

    ver, notes = a.version, a.notes
    files = DEFAULT_FILES + ([WORKER] if a.with_worker else [])
    sync_from_mvp(files, ver)
    if sensitive_scan(files):
        print("!! 敏感扫描命中可疑项，确认上面列出的行都是安全文案再继续")
        sys.exit(1)
    name, md5, size = build_zip(ver, notes, a.with_worker)
    write_manifest(ver, name, md5, size, notes)
    if not a.no_git:
        git_release(ver, notes, name)
    if a.deploy:
        deploy_to_server(name)
    print("\n完成。用户侧：设置 → 软件更新，点一下即可升到 v%s" % ver)


if __name__ == "__main__":
    main()
