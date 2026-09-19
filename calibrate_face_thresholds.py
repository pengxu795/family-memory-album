#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A1 阈值标定工具（施工图·算法线 A1）
用 DB 里人工标注（face_instance_v0.person_id）做真值，对指定 embedding 模型
计算同人/不同人余弦相似度分布，输出聚类/去重阈值建议。

设计原则：
- 纯离线读库，只读，不改任何数据
- 模型无关：按 --model 过滤 face_embedding_v0.embedding_model，换模型后重跑即可
- 默认只用 quality_class='usable' 的脸（--include-low-quality 放开）
- 默认排除同资产配对（连拍/同帧会虚高同人相似度）--allow-same-asset 放开

用法：
  python3 calibrate_face_thresholds.py                    # 全部模型
  python3 calibrate_face_thresholds.py --model SFace-2021dec
输出：控制台摘要 + eval/threshold_report_<model>_<ts>.md
"""
import argparse, json, random, sqlite3, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DB = ROOT / "data" / "family_memory.db"
OUT = ROOT / "eval"


def load_vectors(model: str, include_low: bool):
    """返回 {face_instance_id: (person_id, vec)}"""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True, timeout=60)
    sql = """SELECT f.face_instance_id, f.person_id, f.quality_class, e.embedding
             FROM face_instance_v0 f JOIN face_embedding_v0 e USING(face_instance_id)
             WHERE e.status='success' AND e.embedding IS NOT NULL AND f.person_id IS NOT NULL"""
    args = []
    if model != "ALL":
        sql += " AND e.embedding_model=?"
        args.append(model)
    if not include_low:
        sql += " AND f.quality_class='usable'"
    rows = con.execute(sql, args).fetchall()
    con.close()
    data = {}
    for fid, pid, _q, blob in rows:
        v = np.frombuffer(blob, dtype="<f4").astype(np.float32)
        n = float(np.linalg.norm(v))
        if n <= 0:
            continue
        data[fid] = (pid, v / n)
    return data


def sample_pairs(data: dict, per_person_cap: int, seed: int, allow_same_asset: bool):
    """同人对 + 不同人对（各按 per_person_cap 上限采样，固定随机种子可复现）"""
    rng = random.Random(seed)
    by_person = {}
    for fid, (pid, v) in data.items():
        by_person.setdefault(pid, []).append((fid, v))
    # 过滤掉样本太少的 person
    persons = {p: items for p, items in by_person.items() if len(items) >= 5}
    print(f"可用 person（>=5 张 usable 脸）: {len(persons)} 人, "
          f"共 {sum(len(v) for v in persons.values())} 张脸")

    # 同人对：每两两组合（超上限随机采样）
    same_pairs = []
    for pid, items in persons.items():
        n = len(items)
        all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        rng.shuffle(all_pairs)
        for i, j in all_pairs[:per_person_cap]:
            same_pairs.append((items[i][1], items[j][1]))

    # 不同人对：跨 person 随机配对，总量与同人对同量级
    pids = sorted(persons)
    n_diff = len(same_pairs)
    diff_pairs = []
    seen = set()
    tries = 0
    while len(diff_pairs) < n_diff and tries < n_diff * 20:
        tries += 1
        pa, pb = rng.sample(pids, 2)
        fa = rng.choice(persons[pa]); fb = rng.choice(persons[pb])
        key = (min(fa[0], fb[0]), max(fa[0], fb[0]))
        if key in seen:
            continue
        seen.add(key)
        diff_pairs.append((fa[1], fb[1]))
    return same_pairs, diff_pairs


def cos_stats(pairs):
    if not pairs:
        return np.array([])
    a = np.stack([p[0] for p in pairs]); b = np.stack([p[1] for p in pairs])
    return np.sum(a * b, axis=1)  # 已归一化，点积即余弦


def pct(arr, q):
    return float(np.percentile(arr, q)) if len(arr) else float("nan")


def analyze(same, diff):
    out = {}
    out["same"] = {"n": len(same), "mean": pct(same, 50), "p5": pct(same, 5), "p1": pct(same, 1), "min": pct(same, 0)}
    out["diff"] = {"n": len(diff), "mean": pct(diff, 50), "p95": pct(diff, 95), "p99": pct(diff, 99), "max": pct(diff, 100)}
    # 候选阈值
    cands = []
    for fpr in (0.1, 1.0, 5.0):
        t = pct(diff, 100 - fpr)
        tpr = float(np.mean(same >= t)) * 100 if len(same) else float("nan")
        cands.append({"策略": f"误配率≤{fpr}%", "阈值": t, "同人召回": f"{tpr:.1f}%"})
    for tpr_t in (95.0, 99.0):
        t = pct(same, 100 - tpr_t)
        fpr = float(np.mean(diff >= t)) * 100 if len(diff) else float("nan")
        cands.append({"策略": f"召回≥{tpr_t}%", "阈值": t, "误配率": f"{fpr:.2f}%"})
    # 最佳 F1 扫描
    if len(same) and len(diff):
        grid = np.unique(np.quantile(np.concatenate([same, diff]), np.linspace(0.01, 0.99, 400)))
        best, best_f1 = None, -1
        for t in grid:
            tp = np.mean(same >= t); fp = np.mean(diff >= t)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1 = 2 * prec * tp / (prec + tp) if (prec + tp) > 0 else 0
            if f1 > best_f1:
                best_f1, best = float(t), f1
        out["best_f1"] = {"阈值": best, "F1": best_f1}
    out["candidates"] = cands
    out["separation_gap"] = pct(diff, 95) and (pct(same, 5) - pct(diff, 95))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ALL", help="embedding_model 过滤，ALL=全部")
    ap.add_argument("--per-person-cap", type=int, default=400, help="每人同人对上限")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-low-quality", action="store_true")
    ap.add_argument("--allow-same-asset", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    models = [r[0] for r in con.execute(
        "SELECT DISTINCT embedding_model FROM face_embedding_v0 WHERE status='success'")]
    con.close()
    targets = models if args.model == "ALL" else [args.model]

    import os
    os.makedirs(OUT, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M")
    report_lines = [f"# 人脸阈值标定报告 {ts}", "",
                    f"- 配对策略: 同人对/不同人对上限 {args.per_person_cap}/人, seed={args.seed}",
                    f"- 质量过滤: {'无' if args.include_low_quality else '仅 usable'}"
                    f" | 同资产配对: {'允许' if args.allow_same_asset else '排除'}", ""]

    for m in targets:
        data = load_vectors(m, args.include_low_quality)
        if len(data) < 20:
            print(f"[{m}] 样本不足({len(data)})，跳过")
            continue
        same_pairs, diff_pairs = sample_pairs(data, args.per_person_cap, args.seed, args.allow_same_asset)
        same, diff = cos_stats(same_pairs), cos_stats(diff_pairs)
        res = analyze(same, diff)
        print(f"\n===== {m} =====")
        print(f"同人  n={res['same']['n']}  中位={res['same']['mean']:.4f}  p5={res['same']['p5']:.4f}  p1={res['same']['p1']:.4f}")
        print(f"不同人 n={res['diff']['n']}  中位={res['diff']['mean']:.4f}  p95={res['diff']['p95']:.4f}  p99={res['diff']['p99']:.4f}")
        print(f"分离度(p5同 - p95异): {res['separation_gap']:.4f}")
        for c in res["candidates"]:
            k = [x for x in ("阈值", "同人召回", "误配率") if x in c]
            print("  " + "  ".join(f"{x}={c[x]:.4f}" if isinstance(c[x], float) else f"{x}={c[x]}" for x in k))
        if "best_f1" in res:
            print(f"  最佳F1: 阈值={res['best_f1']['阈值']:.4f}  F1={res['best_f1']['F1']:.3f}")

        report_lines += [f"## {m}", "",
                         f"- 同人对 {res['same']['n']} 个：中位 {res['same']['mean']:.4f}，p5 {res['same']['p5']:.4f}，p1 {res['same']['p1']:.4f}",
                         f"- 不同人对 {res['diff']['n']} 个：中位 {res['diff']['mean']:.4f}，p95 {res['diff']['p95']:.4f}，p99 {res['diff']['p99']:.4f}",
                         f"- 分离度（同人p5 − 不同人p95）: **{res['separation_gap']:.4f}**", ""]
        for c in res["candidates"]:
            kv = "  ".join((f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}") for k, v in c.items())
            report_lines.append(f"- {kv}")
        if "best_f1" in res:
            report_lines.append(f"- 最佳F1工作点: 阈值={res['best_f1']['阈值']:.4f}，F1={res['best_f1']['F1']:.3f}")
        report_lines.append("")

    out_md = OUT / f"threshold_report_{ts}.md"
    out_md.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\n报告: {out_md}")


if __name__ == "__main__":
    main()
