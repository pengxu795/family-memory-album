#!/usr/bin/env python3
"""Log 视频播放转码缓存（2026-09-24）。

背景：DJI D-Log 原片是 4K HEVC Main 10（10bit），Chrome 解不动 →
查看器 video onerror（用户实锤「log视频打不开」）。缩略图/预览在
09-23 已走还原链，但 /orig 播放仍是原片。

本脚本把 is_log=1 的视频逐条转成浏览器通吃的 H.264 8bit（≤1920 边）+
D-Log 色彩还原（与 server.py 缩略图同一套参数），存 /data/videos_lc/。
server.py 的 /orig 路由检测到转码文件后自动优先返回 → 转好一条生效一条。

设计约束：
- 幂等：产物已存在且 >0 字节则跳过；.tmp 半成品视为未完成（重跑重转）
- 护栏：RLIMIT_AS 2.2GB 起 + nice 19（09-23/24 swap 风暴教训，
  ffmpeg 在 3.9GB NAS 上不加护栏就是事故）
- 速度：按时段自动档（白天 1 核防卡站，23:00–08:00 夜间 2 核提速）
- 原片永不修改；音频尽量 -c:a copy（DJI 全是 AAC）
- 还原参数经 import server 复用，与缩略图链严格同源，不复制数值

用法（容器内）：
    python3 /app/transcode_log_videos.py --status     # 看待办/已完成
    python3 /app/transcode_log_videos.py --limit 1    # 只转一条（验证用）
    python3 /app/transcode_log_videos.py              # 全量跑（后台 nohup）
"""
import argparse
import os
import resource
import sqlite3
import subprocess
import sys
import time

DATA_DIR = os.environ.get("FF_DATA_DIR", "/data")
DB = os.path.join(DATA_DIR, "family_memory.db")
OUT_DIR = os.path.join(DATA_DIR, "videos_lc")
FFMPEG = os.environ.get("FFMPEG_BIN") or "ffmpeg"

# 复用 server.py 的还原参数（_logcolor_vf / logcolor_for / VIDEO_LC_DIR）
sys.path.append(os.path.dirname(os.path.abspath(__file__)) or "/app")
import server as S  # noqa: E402


def lc_video_rows():
    """所有 is_log=1 的视频 (asset_id, filename, src_path|None)。"""
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    rows = con.execute("""
        SELECT ma.asset_id, mf.filename, mf.absolute_path
        FROM asset_log_color_v0 lc
        JOIN media_asset ma ON ma.asset_id = lc.asset_id AND ma.media_type='video'
        LEFT JOIN media_file mf ON mf.asset_id = ma.asset_id
        WHERE lc.is_log = 1
        ORDER BY mf.byte_size ASC
    """).fetchall()
    con.close()
    return [(r["asset_id"], r["filename"], r["absolute_path"]) for r in rows]


def _speed_profile():
    """按时段自动选转码速度（2026-09-24 用户拍板：夜间自动全速）。

    白天（08:00–23:00）单核：4 核 NAS 全速转码吃 2.8 核 → load 3.8 整站卡顿
    （用户实锤），threads=1 压到 ~1 核留 3 核给相册服务；
    夜间（23:00–08:00）自动提 2 核，速度约翻倍，睡一觉转完。
    **每条转码前现取时段**——进程常驻跨过午夜也会自动切换，无需重启。
    """
    h = time.localtime().tm_hour
    if 8 <= h < 23:
        return 1, 2_200_000_000
    return 2, 2_400_000_000


def _make_limit(rlimit_as):
    """ffmpeg 子进程护栏工厂：RLIMIT_AS 按档位走 + nice 19（风暴血案教训）。
    2026-09-24 实测 1.8GB 偏紧：4K HEVC 软解 DPB + x264 frame-threads 的
    VSZ（含线程栈/mmap）轻松超限 → x264 报 "Error submitting video frame"
    大面积失败。2.2GB 起步（RSS 远小于此），夜间 2 核档放宽到 2.4GB。"""
    def _f():
        try:
            resource.setrlimit(resource.RLIMIT_AS, (rlimit_as, rlimit_as))
        except Exception:
            pass
        try:
            os.nice(19)
        except Exception:
            pass
    return _f


def transcode_one(asset_id, src):
    """转一条 → 产物路径；失败返回 None。"""
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, asset_id[6:] + "_lc.mp4")
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out                      # 已转好
    tmp = out + f".tmp{os.getpid()}"
    lc = S.logcolor_for(asset_id)
    vf = ("scale=1920:1920:force_original_aspect_ratio=decrease,format=yuv420p,"
          + (S._logcolor_vf(lc[1]) + "," if lc else ""))
    th, rl = _speed_profile()
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error", "-strict", "unofficial",
           # 线程数按时段动态选（见 _speed_profile）：解码 + 滤镜 + x264 三处必须同步改，
           # 漏一处该线程池仍按默认核数开（2026-09-24 实锤单 ffmpeg 吃 200% CPU）
           "-threads", str(th), "-i", src,
           "-filter_threads", str(th),
           "-vf", vf + "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
           # x264 内存限流（2026-09-24 实锤）：frame-threads/lookahead 默认值
           # 在护栏下 malloc 失败。frame-threads=档位数 + lookahead 10 帧够用。
           "-x264-params", f"threads={th}:lookahead-threads=1:lookahead=10:sync-lookahead=0",
           "-c:a", "copy", "-movflags", "+faststart",
           # -f mp4 必须显式指定：tmp 文件名以 .tmpN 结尾，ffmpeg 按最后
           # 扩展名猜 muxer 会报 "use a standard extension"（2026-09-24 实锤）
           "-f", "mp4", tmp]
    try:
        subprocess.run(cmd, capture_output=True, timeout=4 * 3600, check=True,
                       preexec_fn=_make_limit(rl))
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        err = ""
        if isinstance(e, subprocess.CalledProcessError) and e.stderr:
            err = e.stderr.decode(errors="replace")[-400:]
        print(f"  [FAIL] {asset_id} {err}", flush=True)
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    if not os.path.exists(tmp) or os.path.getsize(tmp) < 100_000:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return None
    os.replace(tmp, out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="最多转 N 条（0=不限）")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    rows = lc_video_rows()
    todo, done, miss = [], 0, []
    for aid, fn, src in rows:
        if not src or not os.path.exists(src):
            miss.append((aid, fn))
            continue
        out = os.path.join(OUT_DIR, aid[6:] + "_lc.mp4")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            done += 1
        else:
            todo.append((aid, fn, src))
    print(f"Log 视频共 {len(rows)}：已转 {done}，待转 {len(todo)}，源缺失 {len(miss)}", flush=True)
    if args.status:
        for aid, fn, src in todo[:10]:
            print(f"  待转 {aid} {fn} {os.path.getsize(src)/1e9:.1f}GB" if os.path.exists(src) else "")
        return

    n = 0
    t0 = time.time()
    for aid, fn, src in todo:
        if args.limit and n >= args.limit:
            break
        sz = os.path.getsize(src) / 1e9
        t1 = time.time()
        out = transcode_one(aid, src)
        dt = time.time() - t1
        # 成败都计数：--limit 是「尝试条数」，失败不限流会一路烧穿全量
        n += 1
        if out:
            print(f"[{done+n}/{len(rows)}] {fn} ({sz:.1f}GB) -> "
                  f"{os.path.getsize(out)/1e6:.0f}MB 用时 {dt/60:.1f}min", flush=True)
    print(f"本轮尝试 {n} 条（成功见上），总用时 {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
