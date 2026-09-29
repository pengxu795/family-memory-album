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

# 每次发版必须带上来的文件（相对路径）。static/ 由脚本**整目录**收进去（跳过 .bak/草稿），
# 不用手写清单——2026-09-29 的教训：只列 library.html 漏掉 index.html，用户升了版 bug 原样还在。
DEFAULT_FILES = ["server.py"]
WORKER = "transcode_log_videos.py"

# 仓库级文件：**不进更新包**（更新包只换 server.py / static / worker），但必须随版本
# 同步到公开仓，否则「别人 clone 下来自己构建」拿到的是旧脚手架。
# 2026-09-29 补：此前只同步 server.py + static/，于是 launcher 少 videolc 分发表、
#   spec 少 4 个 worker、schema.py 少 busy_timeout 的修复 **永远到不了公开仓**
#   —— 只在本地 mvp 里修好了，等于没修。
#
# ★★ 但「能同步」的前提是「mvp 版与公开版除本次修复外内容一致」。
#   下面这几个文件在公开仓是**人工策展**过的，mvp 版反而是旧的/含个人信息的，
#   **绝不能被 mvp 整文件覆盖**（2026-09-29 预检实测出来的）：
#     README.md                → 公开版有「NAS 一键部署」+ ghcr.io 拉取章节；
#                                mvp 版已删掉该章节，且示例句里写着真名
#     LICENSE                  → 公开版署名 pengxu795；mvp 版是 xupeng
#     Dockerfile               → 公开版 FF_FACE_BACKEND=opencv（SFace，Apache-2.0 可分发）；
#                                mvp 版是 adaface（WebFace4M 非商用，不可随镜像分发）
#     FamilyMemoryAlbum.spec   → 公开版 bundle_identifier=io.github.*；
#                                mvp 版是 com.xupeng.*
#   这几处要改代码时，**手工打补丁**到公开仓（脚本只告警，不覆盖）。
REPO_FILES = [
    "launcher.py",
    "requirements.txt",
    "requirements-docker.txt",
    "docker-compose.yml",
    ".dockerignore",
    "schema.py",
    "geo_cities_cn.py",
]
# 只校验存在性 / 一致性并告警，**绝不覆盖**（原因见上）
REPO_PROTECTED = [
    "Dockerfile",
    "FamilyMemoryAlbum.spec",
    "README.md",
    "LICENSE",
]
# 整目录同步（同上：只进公开仓，不进更新包）
REPO_DIRS = ["migrations"]

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
    """mvp → release：复制 + 脱敏 + 改版本号。static/ 与 REPO_DIRS 整目录同步。"""
    copy_list = list(files) + list(REPO_FILES)
    for d in REPO_DIRS:
        src_dir = MVP / d
        if src_dir.is_dir():
            for p in sorted(src_dir.rglob("*")):
                if p.is_file() and not is_junk(p.relative_to(MVP)):
                    rel = p.relative_to(MVP).as_posix()
                    if rel not in copy_list:
                        copy_list.append(rel)
    static_src = MVP / "static"
    if static_src.is_dir():
        for p in sorted(static_src.rglob("*")):
            if p.is_file() and not is_junk(p.relative_to(MVP)):
                rel = p.relative_to(MVP).as_posix()
                if rel not in copy_list:
                    copy_list.append(rel)
    # 硬护栏：公开仓人工策展的文件绝不允许进同步清单。
    # （防止将来又有人图省事把 README.md / LICENSE / Dockerfile 加回 REPO_FILES —— 
    #   2026-09-29 预检实测：那样会把公开 README 的安装章节删掉、泄露真名、
    #   并把非商用的 AdaFace 权重配置推进公开仓）
    clash = sorted(set(copy_list) & set(REPO_PROTECTED))
    need(not clash, "同步清单里出现「公开仓人工策展、禁止覆盖」的文件：%s —— "
                    "这些要手工打补丁到公开仓" % clash)

    missing = [rel for rel in copy_list if not (MVP / rel).is_file()]
    if missing:
        # 仓库级文件允许缺失（不同人手上的开发目录可能没有 .app 打包产物），但不能是
        # server.py 这类必需件 —— 那种情况必须报错停下来。
        hard = [rel for rel in missing if rel in DEFAULT_FILES or rel == WORKER]
        need(not hard, "开发目录里没有必需文件 %s（FM_MVP_DIR=%s）" % (hard, MVP))
        if missing:
            print("· 跳过开发目录里没有的仓库级文件：%s" % ", ".join(missing))
        copy_list = [rel for rel in copy_list if rel not in missing]
    for rel in copy_list:
        src, dst = MVP / rel, REL / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print("copy: %d 个文件（server.py + static/ + 仓库级 %d 个）"
          % (len(copy_list), len([r for r in copy_list if r in REPO_FILES or r.split('/')[0] in REPO_DIRS])))

    # 脱敏：代码与注释里的真实姓名（必须在打包/扫描前做）
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
    return copy_list


def check_worker_sync():
    """校验「worker 脚本四处同步」铁律（2026-09-29 血案固化成断言）。

    `server.py:WORKER_SCRIPTS` 是**唯一真源**；凡登记在册、靠 subprocess 拉起的 .py
    必须同时出现在另外三处，否则「别人 clone 下来自己构建」会**静默缺功能**：
        Dockerfile 的 COPY 清单 / launcher.py 的 scripts 字典 / FamilyMemoryAlbum.spec 的 datas
    缺文件不报错、不崩溃，只是对应后台任务永远起不来 —— 历史上这么瞒了 9 天。
    """
    s = (REL / "server.py").read_text(encoding="utf-8")
    m = re.search(r"WORKER_SCRIPTS\s*=\s*\{.*?\{(.*?)\}\s*\.items\(\)", s, re.S)
    need(m, "server.py 里找不到 WORKER_SCRIPTS 字面量，解析规则要跟着源码改")
    workers = sorted(set(re.findall(r'"([^"]+\.py)"', m.group(1))))
    need(workers, "WORKER_SCRIPTS 里没解析出任何 .py")
    places = ["Dockerfile", "launcher.py", "FamilyMemoryAlbum.spec"]
    texts = {p: ((REL / p).read_text(encoding="utf-8") if (REL / p).exists() else "")
             for p in places}
    bad = []
    print("worker 四处同步自检（%d 个脚本 × 3 处脚手架）:" % len(workers))
    for w in workers:
        miss = [p for p in places if w not in texts[p]]
        bad += [(w, p) for p in miss]
        print("  %s %-28s %s" % ("✓" if not miss else "✗", w,
                                 "三处齐备" if not miss else "缺：" + ", ".join(miss)))
    need(not bad, "worker 脚本没在四处同步 —— 补完 Dockerfile / launcher.py / .spec 再发版")


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
    if not out.exists():      # 不用 mkdir(exist_ok=True)：部分沙箱包装会把已存在目录的
        out.mkdir(parents=True)   # mkdir 当越权操作拦掉（EEXIST→PermissionError）
    name = "family_memory_update_v%s.zip" % version
    zpath = out / name
    # ★ 包内必须带 manifest.json：服务端 update_best_package() 就是靠读 zip 里的
    #   manifest.json 认包的（认不到就跳过这个包），少了它用户永远升不上新版本。
    inner = {"version": version, "notes": notes, "files": []}
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            rel = f.relative_to(REL).as_posix()
            z.write(f, rel)
            inner["files"].append({"path": rel, "md5": hashlib.md5(f.read_bytes()).hexdigest()})
        z.writestr("manifest.json", json.dumps(inner, ensure_ascii=False, indent=1))
    raw = zpath.read_bytes()
    print("✓ %s: %d 文件, %.0f KB"
          % (name, len(files), len(raw) / 1024))
    return name, hashlib.md5(raw).hexdigest(), len(raw)


def sign_package(zip_path, key_path):
    """用 Ed25519 私钥给更新包签名，返回 (base64 签名, 公钥 PEM)。

    2026-09-29 安全审计配套：更新包的完整性原先只有 manifest 自带的 md5 自证，
    而 manifest 与包同源 —— 更新源被劫持就等于远程代码执行（update_apply 会用
    包里的 server.py 覆盖自身并重启）。加上非对称签名后，服务端只要内置公钥，
    就能拒收任何非本发布流程产出的包。

    私钥**绝不进仓库**（默认放 tools/update_signing.key，已在 .gitignore 忽略）。
    用法：
        python3 tools/release.py 1.0.14 "..." --gen-key       # 首次生成密钥对
        python3 tools/release.py 1.0.14 "..." --sign-key tools/update_signing.key
    启用步骤（顺序重要）：目标机必须先具备 cryptography 库，
    再把公钥写到 <数据目录>/update_pubkey.txt；否则服务端会 fail-closed 拒收。
    """
    try:
        import base64
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except Exception as e:
        print("! 本机缺少 cryptography，无法签名：%s" % e)
        print("  macOS 可用系统 python：/usr/bin/python3 tools/release.py ...")
        return None, None
    data = Path(key_path).read_bytes()
    try:
        key = serialization.load_pem_private_key(data, password=None)
    except Exception:
        key = Ed25519PrivateKey.from_private_bytes(data)
    if not isinstance(key, Ed25519PrivateKey):
        print("! 私钥不是 Ed25519 类型，拒绝用它签名")
        return None, None
    digest = hashlib.sha256(zip_path.read_bytes()).digest()
    sig = base64.b64encode(key.sign(digest)).decode()
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    return sig, pub_pem


def gen_signing_key(key_path):
    """生成 Ed25519 密钥对，写入 key_path（权限 600），返回公钥 PEM。"""
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except Exception as e:
        print("! 本机缺少 cryptography，无法生成密钥：%s" % e)
        return None
    p = Path(key_path)
    if p.exists():
        print("! 私钥已存在，不覆盖：%s" % p)
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    key = Ed25519PrivateKey.generate()
    p.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()))
    os.chmod(p, 0o600)
    pub_pem = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    print("✓ 已生成签名私钥 %s（权限 600，务必不要提交进仓库）" % p)
    print("  对应公钥（贴到目标机 <数据目录>/update_pubkey.txt）：\n%s" % pub_pem)
    return pub_pem


def write_manifest(version, name, md5, size, notes, sig=None):
    man = {
        "version": version,
        "file": name,
        "md5": md5,
        "size": size,
        "notes": notes,
        "url": "https://github.com/pengxu795/family-memory-album/releases/download/%s/%s"
               % (version, name),
    }
    if sig:
        man["sig"] = sig          # Ed25519(sha256(zip))，服务端 verify_package_signature 校验
    (REL / "update" / "manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("manifest ->", version, md5, ("+sig" if sig else "(无签名)"))


def run(cmd, **kw):
    print("$", " ".join(cmd)[:160])
    return subprocess.run(cmd, cwd=str(REL), **kw)


def git_release(version, notes, name):
    msg = "release v%s: %s" % (version, notes)
    if run(["git", "add", "-A"]).returncode:
        return False
    if run(["git", "commit", "-m", msg]).returncode:
        print("· 没有新改动可提交（或提交被拦），继续")
    tag = version if version.startswith("v") else "v" + version
    run(["git", "tag", "-f", tag])
    run(["git", "push", "origin", "main"])
    # 注意：--follow-tags 在这台机器/这个远端上不会把新建的轻量 tag 推上去，
    # 少了这一步 gh release create 会报「tag has not been pushed」，故显式推一次。
    run(["git", "push", "origin", tag])
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
    ap.add_argument("--sign-key", default=os.environ.get("FM_SIGN_KEY", ""),
                    help="Ed25519 私钥路径（PEM 或 32 字节 raw）；给包加 sig 字段")
    ap.add_argument("--gen-key", action="store_true",
                    help="生成签名密钥对（默认 tools/update_signing.key）后退出")
    a = ap.parse_args()

    if a.gen_key:
        gen_signing_key(a.sign_key or "tools/update_signing.key")
        return

    ver, notes = a.version, a.notes
    files = DEFAULT_FILES + ([WORKER] if a.with_worker else [])
    files = sync_from_mvp(files, ver)
    check_worker_sync()          # 四处同步铁律：缺一个就停，别把静默失效发出去
    if sensitive_scan(files):
        print("!! 敏感扫描命中可疑项，确认上面列出的行都是安全文案再继续")
        sys.exit(1)
    name, md5, size = build_zip(ver, notes, a.with_worker)

    # 2026-09-29：签名（可选）。没给 --sign-key 就跳过，但会明确提示当前是「无签名」模式。
    sig = None
    if a.sign_key:
        sig, pub_pem = sign_package(ROOT / "update_pkg" / name, a.sign_key)
        if pub_pem:
            print("  签名完成。目标机启用验签：把下面公钥写到 <数据目录>/update_pubkey.txt\n%s" % pub_pem)
    else:
        print("· 未签名（未提供 --sign-key）。目标机将退化为「域名白名单」校验级别。")

    write_manifest(ver, name, md5, size, notes, sig)
    if not a.no_git:
        git_release(ver, notes, name)
    if a.deploy:
        deploy_to_server(name)
    print("\n完成。用户侧：设置 → 软件更新，点一下即可升到 v%s" % ver)


if __name__ == "__main__":
    main()
