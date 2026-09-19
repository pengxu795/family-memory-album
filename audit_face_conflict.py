#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人物池冲突检测（本地 kNN 一致性，2026-09-07）
逻辑：被归到某人的脸，理应最像那人自己的参照脸。若它明显更像**别人**的参照脸
（top3 均值差 >= 0.03 且最高 >= 0.50），则高度疑似错标 → 交给 VLM 复核。
用法: python audit_face_conflict.py [--apply]
不带 --apply 只打印报告（dry-run）。"""
import json, sqlite3, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'data/family_memory.db'
MARGIN = 0.03
FLOOR = 0.50

TARGETS = None  # #16：动态取自 DB（12 岁内 confirmed 人物），懒加载见 main()


def _load_targets():
    con = sqlite3.connect(DB, timeout=30)
    rows = con.execute(
        """SELECT display_name FROM person
           WHERE identity_status='confirmed' AND birth_date IS NOT NULL
             AND (julianday('now') - julianday(birth_date)) / 365.25 < 12"""
    ).fetchall()
    con.close()
    return [r[0] for r in rows]


def main():
    apply = '--apply' in sys.argv
    con = sqlite3.connect(DB, timeout=60)
    con.row_factory = sqlite3.Row
    # 参照库：全部人工确认脸（含低质量——它们也是人, 不影响均值判断）
    refs = {}
    for r in con.execute("""SELECT fi.person_id, p.display_name, fe.embedding
            FROM face_instance_v0 fi JOIN face_embedding_v0 fe USING(face_instance_id)
            JOIN person p USING(person_id)
            WHERE fi.person_id IS NOT NULL AND fi.sample_role='manual' AND fe.status='success'"""):
        refs.setdefault(r['person_id'], {'name': r['display_name'],
                                         'vecs': [np.frombuffer(r['embedding'], dtype='<f4')]})
        refs[r['person_id']]['vecs'].append(np.frombuffer(r['embedding'], dtype='<f4'))
    # 修正：上面循环里首条 append 了两次, 重建
    refs = {}
    for r in con.execute("""SELECT fi.person_id, p.display_name, fe.embedding
            FROM face_instance_v0 fi JOIN face_embedding_v0 fe USING(face_instance_id)
            JOIN person p USING(person_id)
            WHERE fi.person_id IS NOT NULL AND fi.sample_role='manual' AND fe.status='success'"""):
        refs.setdefault(r['person_id'], {'name': r['display_name'], 'vecs': []})
        refs[r['person_id']]['vecs'].append(np.frombuffer(r['embedding'], dtype='<f4'))
    for pid, d in refs.items():
        d['M'] = np.vstack(d['vecs'])
        d['M'] = d['M'] / (np.linalg.norm(d['M'], axis=1, keepdims=True) + 1e-9)
    print(f"参照库: {len(refs)} 人, {sum(len(d['vecs']) for d in refs.values())} 张脸")

    total_flagged = 0
    for name in (TARGETS or _load_targets()):
        prow = con.execute("SELECT person_id FROM person WHERE display_name=?", (name,)).fetchone()
        if not prow or prow['person_id'] not in refs:
            print(f"{name}: 无参照脸, 跳过")
            continue
        own_pid = prow['person_id']
        rows = con.execute("""SELECT fi.face_instance_id, fe.embedding
            FROM face_instance_v0 fi JOIN face_embedding_v0 fe USING(face_instance_id)
            WHERE fi.person_id=? AND fi.sample_role != 'manual' AND fi.person_locked=0
            AND fe.status='success'""", (own_pid,)).fetchall()
        flagged = []
        for r in rows:
            v = np.frombuffer(r['embedding'], dtype='<f4')
            v = v / (np.linalg.norm(v) + 1e-9)
            sims = {}
            for pid, d in refs.items():
                top3 = np.sort(d['M'] @ v)[-3:]
                sims[pid] = float(top3.mean())
            own = sims.get(own_pid, 0.0)
            other_pid = max((p for p in sims if p != own_pid), key=lambda p: sims[p], default=None)
            other = sims.get(other_pid, 0.0) if other_pid else 0.0
            if other - own >= MARGIN and other >= FLOOR:
                flagged.append((r['face_instance_id'], other_pid, refs[other_pid]['name'],
                                round(own, 3), round(other, 3)))
        print(f"\n{name}: 池内自动脸 {len(rows)}, 冲突(更像别人) {len(flagged)}")
        from collections import Counter
        cc = Counter(f[2] for f in flagged)
        print(f"  更像谁: {cc.most_common(6)}")
        for f in flagged[:5]:
            print(f"  例: own={f[3]} vs {f[2]}={f[4]}")
        if apply:
            for fid, opid, oname, own, other in flagged:
                con.execute("""INSERT OR IGNORE INTO face_unassign_audit_v0
                               (face_instance_id, person_id, display_name, capture_time, birth_date, reason, created_at)
                               SELECT ?, ?, ?, ma.capture_time, p.birth_date, 'conflict_more_like_'||?,
                               datetime('now')
                               FROM face_instance_v0 fi JOIN media_asset ma USING(asset_id)
                               JOIN person p ON p.person_id=fi.person_id
                               WHERE fi.face_instance_id=?""", (fid, own_pid, name, oname, fid))
            con.executemany("UPDATE face_instance_v0 SET person_id=NULL WHERE face_instance_id=? AND person_locked=0",
                            [(f[0],) for f in flagged])
            con.commit()
            print(f"  → 已解除 {len(flagged)} 张（审计留档）")
        total_flagged += len(flagged)
    print(f"\n总冲突: {total_flagged} {'(已apply)' if apply else '(dry-run)'}")
    con.close()


if __name__ == '__main__':
    main()
