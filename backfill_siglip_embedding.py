#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SigLIP 图像向量全库回填（算法线 A4：SigLIP v1 → SigLIP2）
为所有还没有 google/siglip2-base-patch16-224 图像向量的资产补嵌入。
- 数据源用本地缩略图（thumbs_mvp/{id}_t480.jpg），无缩略图才读原片/抽视频帧
- 断点续跑：已有该 model 向量的跳过
- 维度/BLOB 协议严格校验（768 维 little-endian float32）
- 由导入管线（server.py 扫描后）自动触发；也可手动运行
- A/B 实测（eval/ab_siglip2_*.md）：v2 中文检索 Recall@10 0.733→0.792 胜出

历史：原为 archived/ab_embed_siglip.py --all（A/B 实验时代），2026-09-05 清理归档时
被误删——新资产因此缺向量，语义检索/场景打分都会漏掉它们。恢复为正式脚本并接入管线。
2026-09-08 算法线 A2/A4：切换 SigLIP2；--limit 小样本冒烟；FF_DB_PATH 测试隔离。
"""
import json, os, sqlite3, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("FF_DATA_DIR") or (ROOT / "data"))
DB = Path(os.environ.get("FF_DB_PATH") or (DATA_DIR / "family_memory.db"))
REPO = "google/siglip2-base-patch16-224"
# SigLIP2 权重为本地普通目录（models/siglip2-base-patch16-224），非 hub 缓存布局
MODEL_DIR = os.environ.get("FF_SIGLIP2_DIR") or str(
    ROOT.parent / "models" / "siglip2-base-patch16-224")
PY = "/usr/bin/python3"  # 系统 python：transformers + torch 就绪


def main():
    limit = 0
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    con = sqlite3.connect(DB, timeout=60)
    con.row_factory = sqlite3.Row
    source_sql = """WITH ranked AS (
                        SELECT mf.asset_id, mf.absolute_path AS path,
                               ROW_NUMBER() OVER (
                                   PARTITION BY mf.asset_id
                                   ORDER BY mf.byte_size DESC, mf.absolute_path) rn
                        FROM media_file mf
                    )
                    SELECT ranked.asset_id, ranked.path, ma.media_type
                    FROM ranked JOIN media_asset ma USING(asset_id)
                    WHERE ranked.rn=1"""
    rows = con.execute(source_sql).fetchall()
    done = {r["subject_id"] for r in con.execute(
        "SELECT subject_id FROM embedding WHERE subject_type='asset' AND model_name=?", (REPO,))}
    todo = [r for r in rows if r["asset_id"] not in done]
    if limit:
        todo = todo[:limit]
    print(f"全库 {len(rows)}, 已有向量 {len(done)}, 待算 {len(todo)}", flush=True)
    con.close()
    if not todo:
        print("无待计算")
        return

    manifest = f"/tmp/siglip_backfill_manifest.json"
    payload = []
    for r in todo:
        stem = r["asset_id"][6:]
        base = ROOT / "data" / "thumbs_mvp"
        path = base / f"{stem}_t480.jpg"
        if not path.exists():
            path = base / f"{stem}.jpg"
        if not path.exists():
            path = r["path"]  # 无缩略图：读原片（worker 内有 sips/ffmpeg 兜底）
        payload.append({"asset_id": r["asset_id"], "path": str(path),
                        "media_type": "photo"})  # 缩略图已是代表帧
    with open(manifest, "w") as f:
        json.dump(payload, f)

    code = r'''
import json, os, time, sys, sqlite3, shutil
from pathlib import Path
os.environ["HF_HUB_OFFLINE"] = "1"
from transformers import AutoModel, AutoProcessor
import torch
from PIL import Image

EXPECTED_DIM = 768
EXPECTED_BYTES = EXPECTED_DIM * 4

model_dir = MODEL_DIR
repo = REPO
db = DB_PATH
manifest_path = MANIFEST

t0 = time.time()
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
proc = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True, local_files_only=True)
model.eval()
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"模型加载 {time.time()-t0:.1f}s device={device}", flush=True)

items = json.load(open(manifest_path))
con = sqlite3.connect(db, timeout=60)
now = __import__("datetime").datetime.now().astimezone().isoformat()
times = []
ok = fail = 0
BATCH = 8
for i in range(0, len(items), BATCH):
    batch = items[i:i+BATCH]
    imgs = []; valid = []; batch_tmps = []
    for it in batch:
        p = it["path"]
        try:
            if it["media_type"] == "video":
                import subprocess, tempfile
                tmp = tempfile.mktemp(suffix=".jpg")
                last_error = None
                for seek in ("0.5", "0"):
                    try:
                        subprocess.run([os.environ.get("FFMPEG_BIN") or "ffmpeg","-hide_banner","-loglevel","error",
                            "-ss",seek,"-i",p,"-frames:v","1","-vf","scale=224:-2","-y",tmp],
                            capture_output=True, timeout=15, check=True)
                        if os.path.exists(tmp) and os.path.getsize(tmp):
                            last_error = None
                            break
                    except Exception as exc:
                        last_error = exc
                if last_error is not None or not os.path.exists(tmp):
                    raise last_error or RuntimeError("video frame extraction failed")
                im = Image.open(tmp).convert("RGB")
                batch_tmps.append(tmp)
            else:
                try:
                    im = Image.open(p).convert("RGB")
                except Exception:
                    # Pillow 默认不能读取部分 HEIC；macOS 用 sips 只读转换，容器/Linux 用 ffmpeg
                    import subprocess, tempfile
                    tmp = tempfile.mktemp(suffix=".jpg")
                    if shutil.which("sips"):
                        subprocess.run(["sips", "-Z", "1200", "-s", "format", "jpeg", p, "--out", tmp],
                                       capture_output=True, timeout=30, check=True)
                    else:
                        subprocess.run([os.environ.get("FFMPEG_BIN") or "ffmpeg", "-hide_banner",
                                        "-loglevel", "error", "-y", "-i", p, "-frames:v", "1",
                                        "-vf", "scale=1200:-2", tmp],
                                       capture_output=True, timeout=30, check=True)
                    im = Image.open(tmp).convert("RGB")
                    batch_tmps.append(tmp)
            imgs.append(im); valid.append(it)
        except Exception as e:
            print(f"  [open-fail] {it['asset_id']} {e}", flush=True)
            fail += 1
    if not imgs: continue
    t1 = time.time()
    with torch.no_grad():
        inp = proc(images=imgs, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        feats = model.get_image_features(**inp)
        feats = feats.cpu()
        feats = torch.nn.functional.normalize(feats, dim=-1)
    for tmp in batch_tmps:
        try: os.unlink(tmp)
        except Exception: pass
    times.append((time.time() - t1) / len(imgs))
    for it, fv in zip(valid, feats):
        try:
            # embedding.vector 的存储协议：little-endian float32 BLOB，768 维=3072 字节
            vec = fv.detach().cpu().float().contiguous().numpy().astype("<f4", copy=False)
            if vec.ndim != 1 or vec.shape[0] != EXPECTED_DIM:
                raise ValueError(f"unexpected embedding shape: {vec.shape}, expected ({EXPECTED_DIM},)")
            if not torch.isfinite(fv).all().item():
                raise ValueError("embedding contains NaN or Inf")
            blob = vec.tobytes(order="C")
            if len(blob) != EXPECTED_BYTES:
                raise ValueError(f"unexpected BLOB size: {len(blob)}, expected {EXPECTED_BYTES}")
            con.execute("INSERT OR REPLACE INTO embedding(subject_type,subject_id,model_name,dimension,vector,created_at) VALUES ('asset',?,?,?,?,?)",
                (it["asset_id"], repo, EXPECTED_DIM, sqlite3.Binary(blob), now))
            ok += 1
        except Exception as e:
            print(f"  [db-fail] {it['asset_id']} {e}", flush=True)
            fail += 1
    con.commit()
    if (i+BATCH) % 40 == 0 or i+BATCH >= len(items):
        print(f"  进度 {i+BATCH}/{len(items)} ok={ok} fail={fail} 单张均速={(sum(times)/len(times)*1000 if times else 0):.0f}ms", flush=True)
con.close()
print(f"完成: ok={ok} fail={fail} 平均单张 {(sum(times)/len(times)*1000 if times else 0):.0f}ms", flush=True)
'''
    src = (code
           .replace("MODEL_DIR", json.dumps(MODEL_DIR))
           .replace("REPO", json.dumps(REPO))
           .replace("DB_PATH", json.dumps(str(DB)))
           .replace("MANIFEST", json.dumps(manifest)))
    worker = "/tmp/siglip_backfill_worker.py"
    with open(worker, "w") as f:
        f.write(src)
    r = os.system(f"{PY} {worker}")
    sys.exit(r >> 8 if r >= 0 else 1)


if __name__ == "__main__":
    main()
