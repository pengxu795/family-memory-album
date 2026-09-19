#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A2 SigLIP v1 vs v2 中文检索 A/B（施工图·算法线 A2）
用 scene_tag_v0 的人工/规则标签做弱真值，在抽样照片上对比两个模型的
中文文本→图像检索能力。赢了才值得全库重算 1.6 万张。

- v1: ~/.cache/huggingface hub 缓存里的 google/siglip-base-patch16-224
- v2: 家庭回忆相册/models/siglip2-base-patch16-224（本地目录）
- 图像源：data/thumbs_mvp/{stem}_t480.jpg（缺失用 {stem}.jpg）
- 指标：每标签 Recall@10、正/负样本平均相似度差（margin），宏平均汇总

用法（必须用系统 python，transformers+torch 在那）：
  /usr/bin/python3 ab_siglip2.py [--sample 200] [--tags 10]
输出：控制台摘要 + eval/ab_siglip2_<ts>.md
"""
import argparse, json, os, sqlite3, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
OUT = ROOT / "eval"
HUB = os.path.expanduser("~/.cache/huggingface/hub")
V1_DIR = os.path.join(HUB, "models--google--siglip-base-patch16-224", "snapshots")
V2_DIR = str(ROOT.parent / "models" / "siglip2-base-patch16-224")
THUMBS = ROOT / "data" / "thumbs_mvp"
V1_NAME = "google/siglip-base-patch16-224"
V2_NAME = "google/siglip2-base-patch16-224"


def resolve_snapshot(base):
    subs = [d for d in os.listdir(base) if not d.startswith(".")]
    return os.path.join(base, subs[0])


def thumb_path(asset_id):
    stem = asset_id[6:]
    for cand in (f"{stem}_t480.jpg", f"{stem}.jpg"):
        p = THUMBS / cand
        if p.exists():
            return p
    return None


def pick_assets(limit_per_tag, tag_limit):
    """按标签分层抽样：每标签最多 limit_per_tag 张，最多 tag_limit 个标签"""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tags = [r[0] for r in con.execute(
        "SELECT tag FROM scene_tag_v0 GROUP BY tag ORDER BY count(*) DESC LIMIT ?", (tag_limit,))]
    asset2tags = {}
    for t in tags:
        rows = con.execute(
            "SELECT asset_id FROM scene_tag_v0 WHERE tag=? ORDER BY RANDOM() LIMIT ?",
            (t, limit_per_tag)).fetchall()
        for (aid,) in rows:
            asset2tags.setdefault(aid, set()).add(t)
    con.close()
    # 过滤有缩略图的
    items = []
    for aid in asset2tags:
        tp = thumb_path(aid)
        if tp:
            items.append((aid, tp))
    return tags, items


class Encoder:
    def __init__(self, model_dir, name):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch = torch
        self.name = name
        self.model = AutoModel.from_pretrained(model_dir).eval()
        self.processor = AutoProcessor.from_pretrained(model_dir)
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        print(f"[{name}] 加载完成 → {self.device}", flush=True)

    def encode_images(self, paths, batch=16):
        from PIL import Image
        vecs = []
        with self.torch.no_grad():
            for i in range(0, len(paths), batch):
                imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
                inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
                f = self.model.get_image_features(**inputs)
                f = f / f.norm(dim=-1, keepdim=True)
                vecs.append(f.cpu().numpy())
        return np.concatenate(vecs)

    def encode_texts(self, texts):
        with self.torch.no_grad():
            inputs = self.processor(text=texts, padding="max_length", max_length=64,
                                    return_tensors="pt").to(self.device)
            f = self.model.get_text_features(**inputs)
            f = f / f.norm(dim=-1, keepdim=True)
            return f.cpu().numpy()


def evaluate(enc, img_vecs, tags, asset_ids, tag_sets):
    """对每标签计算 Recall@10 与 margin"""
    text_vecs = enc.encode_texts(tags)
    img_mat = img_vecs.T  # (D, N)
    sims = text_vecs @ img_mat  # (T, N)
    aid_index = {aid: i for i, aid in enumerate(asset_ids)}
    rows = []
    for ti, tag in enumerate(tags):
        pos_idx = [aid_index[a] for a in asset_ids if tag in tag_sets[a]]
        if len(pos_idx) < 3:
            continue
        order = np.argsort(-sims[ti])
        top10 = set(order[:10].tolist())
        hits = sum(1 for i in pos_idx if i in top10)
        recall = hits / min(10, len(pos_idx))
        pos_sim = float(np.mean(sims[ti][pos_idx]))
        neg_sim = float(np.mean(np.delete(sims[ti], pos_idx)))
        rows.append({"tag": tag, "pos_n": len(pos_idx), "recall@10": recall,
                     "pos_sim": pos_sim, "neg_sim": neg_sim, "margin": pos_sim - neg_sim})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=25, help="每标签抽样上限")
    ap.add_argument("--tags", type=int, default=12, help="参与评测的标签数")
    args = ap.parse_args()

    tags, items = pick_assets(args.sample, args.tags)
    asset_ids = [aid for aid, _ in items]
    paths = [str(p) for _, p in items]
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    tag_sets = {}
    for aid in asset_ids:
        rows = con.execute("SELECT tag FROM scene_tag_v0 WHERE asset_id=?", (aid,)).fetchall()
        tag_sets[aid] = {r[0] for r in rows}
    con.close()
    print(f"抽样 {len(asset_ids)} 张照片 × {len(tags)} 个标签", flush=True)

    results = {}
    for name, d in ((V1_NAME, resolve_snapshot(V1_DIR)), (V2_NAME, V2_DIR)):
        if not os.path.isdir(d):
            print(f"[{name}] 模型目录缺失: {d}，跳过")
            continue
        enc = Encoder(d, name)
        t0 = time.time()
        img_vecs = enc.encode_images(paths)
        print(f"[{name}] 图像编码 {len(paths)} 张耗时 {time.time()-t0:.0f}s", flush=True)
        results[name] = evaluate(enc, img_vecs, tags, asset_ids, tag_sets)

    # 汇总
    ts = time.strftime("%Y%m%d-%H%M")
    lines = [f"# SigLIP v1 vs v2 中文检索 A/B {ts}", "",
             f"- 抽样 {len(asset_ids)} 张（每标签≤{args.sample}，标签数 {len(tags)}）", ""]
    summary = {}
    for name, rows in results.items():
        rec = float(np.mean([r["recall@10"] for r in rows]))
        marg = float(np.mean([r["margin"] for r in rows]))
        summary[name] = (rec, marg)
        lines.append(f"## {name}", )
        lines.append(f"- 宏平均 Recall@10: **{rec:.3f}** | 宏平均 margin: **{marg:.4f}**")
        lines.append("")
        lines.append("| 标签 | 正样本 | Recall@10 | margin |")
        lines.append("|---|---|---|---|")
        for r in rows:
            lines.append(f"| {r['tag']} | {r['pos_n']} | {r['recall@10']:.2f} | {r['margin']:+.4f} |")
        lines.append("")
    if len(summary) == 2:
        (n1, r1), (n2, r2) = summary.items()
        delta = r2[0] - r1[0]
        verdict = "v2 更好" if delta > 0.02 else ("v2 略好" if delta > 0 else "v2 无优势，暂不换")
        lines += ["## 结论", "", f"- Recall@10 变化: {r1[0]:.3f} → {r2[0]:.3f}（{delta:+.3f}）→ **{verdict}**"]

    os.makedirs(OUT, exist_ok=True)
    out_md = OUT / f"ab_siglip2_{ts}.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[3:8]))
    print(f"报告: {out_md}")


if __name__ == "__main__":
    main()
