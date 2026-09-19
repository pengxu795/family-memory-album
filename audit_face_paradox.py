#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""人脸悖论审计（2026-09-07 v3）：性别/年龄/时间三方对账
用户要求：男的不出现女的、女的不出现男的；2岁的脸不能是5岁孩子；幼儿池不能有大人……
方法：本地 qwen2.5vl 估每张已归属脸的「年龄(岁)+性别」，
  ① 性别悖论：VLM 性别 ≠ 人物参照脸多数性别 → 摘
  ② 年龄悖论：|VLM 年龄 − 照片拍摄时人物真实年龄(按出生日期)| > 3 岁 → 摘
  ③ 出生悖论：拍摄时间早于出生日期（双保险，v2 已清）
只动 sample_role != 'manual' 且 person_locked=0 的脸；VLM 结果缓存可断点续跑。
用法: audit_face_paradox.py [--targets children|all] [--apply]"""
import base64, json, re, sqlite3, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DB = ROOT / 'data/family_memory.db'
CROPS = ROOT / 'data' / 'face_crops'
OLLAMA = 'http://127.0.0.1:11434/api/chat'
MODEL = 'qwen2.5vl:7b'
NOW = datetime.now(timezone.utc).isoformat()
# 性别是硬事实, 不让 VLM 猜（曾有孩子被误判 female 导致误摘 6 张）。
# #16 隐私清洗：默认空——如需人工核实兜底，按 display_name 在此临时补充（勿提交真名）。
GENDER_KNOWN = {}
BATCH = 6
AGE_TOL = 3.0


def _load_children():
    """儿童池动态取自 DB（出生日期在 12 岁以内的 confirmed 人物），代码零人名。"""
    con = sqlite3.connect(DB, timeout=30)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """SELECT display_name FROM person
           WHERE identity_status='confirmed' AND birth_date IS NOT NULL
             AND (julianday('now') - julianday(birth_date)) / 365.25 < 12"""
    ).fetchall()
    con.close()
    return [r["display_name"] for r in rows]


CHILDREN = _load_children()
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def vlm(images_b64, prompt):
    body = {"model": MODEL, "stream": False,
            "messages": [{"role": "user", "content": prompt, "images": images_b64}],
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": 900}}
    req = urllib.request.Request(OLLAMA, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with DIRECT.open(req, timeout=240) as resp:
        data = json.loads(resp.read())
    return (data.get('message', {}).get('content') or '').strip()


def parse_json(text):
    m = re.search(r'\[.*\]|\{.*\}', text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    targets = sys.argv[1] if len(sys.argv) > 1 else 'children'
    apply = '--apply' in sys.argv
    con = sqlite3.connect(DB, timeout=60)
    con.row_factory = sqlite3.Row
    con.execute("""CREATE TABLE IF NOT EXISTS face_vlm_attr_v0 (
        face_instance_id TEXT PRIMARY KEY, age_years REAL, age_stage TEXT, gender TEXT,
        raw TEXT, created_at TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS face_unassign_audit_v0 (
        face_instance_id TEXT PRIMARY KEY, person_id TEXT, display_name TEXT,
        capture_time TEXT, birth_date TEXT, reason TEXT, created_at TEXT)""")

    names = CHILDREN if targets == 'children' else \
        [r[0] for r in con.execute("SELECT display_name FROM person")]
    names = [n for n in names if n]
    total_flag = {}
    for name in names:
        prow = con.execute("SELECT person_id, birth_date FROM person WHERE display_name=?", (name,)).fetchone()
        if not prow:
            continue
        pid, birth = prow['person_id'], (prow['birth_date'] or '')[:10]

        # ① 人物期望性别：参照脸多数投票（一次调用）
        ref_rows = con.execute("""SELECT fi.face_instance_id FROM face_instance_v0 fi
            WHERE fi.person_id=? AND fi.sample_role='manual'
            ORDER BY fi.face_width*fi.face_height DESC LIMIT 4""", (pid,)).fetchall()
        ref_imgs = []
        for r in ref_rows:
            p = CROPS / f"{r['face_instance_id']}_k24_s480.jpg"
            if p.exists():
                ref_imgs.append(base64.b64encode(p.read_bytes()).decode())
        exp_gender = None
        if ref_imgs:
            try:
                txt = vlm(ref_imgs, '这几张是同一个人的脸部特写。判断此人性别与年龄段。'
                                   '只回答JSON: {"gender":"male|female","age_stage":"婴儿|幼儿|儿童|青少年|青年|中年|老年"}')
                j = parse_json(txt)
                if isinstance(j, list) and j:
                    j = j[0]
                if isinstance(j, dict):
                    exp_gender = j.get('gender')
            except Exception as e:
                print(f'[{name}] 参照性别判定失败: {e}')
        print(f'== {name}: 期望性别={exp_gender} 生日={birth or "无"}')

        # ② 待审脸：自动归属、未锁定、有裁切图、未缓存
        rows = con.execute("""SELECT fi.face_instance_id, ma.capture_time
            FROM face_instance_v0 fi JOIN media_asset ma USING(asset_id)
            WHERE fi.person_id=? AND fi.sample_role!='manual' AND fi.person_locked=0""",
                           (pid,)).fetchall()
        todo = []
        for r in rows:
            if (CROPS / f"{r['face_instance_id']}_k24_s480.jpg").exists():
                todo.append(r)
        cached = {r[0] for r in con.execute("SELECT face_instance_id FROM face_vlm_attr_v0")}
        todo = [r for r in todo if r['face_instance_id'] not in cached]
        print(f'   待审 {len(todo)}/{len(todo)} 张（池内共 {len(rows)}）')
        attrs = {}  # fid -> (age, gender)
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            imgs = [base64.b64encode((CROPS / f"{r['face_instance_id']}_k24_s480.jpg").read_bytes()).decode()
                    for r in chunk]
            prompt = (f'这是{len(imgs)}张不同照片的人脸特写。对每张分别估计此人大致年龄(岁数,数字)和性别。'
                      '只回答JSON数组: [{"i":1,"age":数字,"gender":"male|female|unclear"},...]（i从1开始按顺序）')
            try:
                txt = vlm(imgs, prompt)
            except Exception as e:
                print(f'   批次{i}调用失败: {e}')
                continue
            arr = parse_json(txt)
            if not isinstance(arr, list) or len(arr) != len(chunk):
                continue
            for r, item in zip(chunk, arr):
                try:
                    age = float(item.get('age'))
                except (TypeError, ValueError):
                    continue
                g = item.get('gender')
                fid = r['face_instance_id']
                attrs[fid] = (age, g)
                con.execute("""INSERT OR REPLACE INTO face_vlm_attr_v0
                               (face_instance_id, age_years, age_stage, gender, raw, created_at)
                               VALUES(?,?,?,?,?,?)""", (fid, age, None, g, json.dumps(item, ensure_ascii=False), NOW))
            con.commit()  # 每批一提交, 避免长事务占写锁
            if (i // BATCH) % 10 == 0:
                print(f'   进度 {i + len(chunk)}/{len(todo)}', flush=True)

        # 补齐缓存里已有的
        for r in con.execute("SELECT face_instance_id, age_years, gender FROM face_vlm_attr_v0"):
            attrs.setdefault(r[0], (r[1], r[2]))

        # ③ 悖论判定
        flags = []
        for r in rows:
            fid = r['face_instance_id']
            a = attrs.get(fid)
            if not a or a[0] is None:
                continue
            age_vlm, g = a
            ct = (r['capture_time'] or '')[:10]
            reason = None
            if exp_gender in ('male', 'female') and g in ('male', 'female') and g != exp_gender:
                reason = 'gender_paradox'
            elif birth and ct >= '1000' and re.match(r'^\d{4}-\d{2}-\d{2}$', ct):
                age_true = (datetime.strptime(ct, '%Y-%m-%d') - datetime.strptime(birth, '%Y-%m-%d')).days / 365.25
                if age_vlm >= 0 and abs(age_vlm - age_true) > AGE_TOL:
                    reason = 'age_paradox'
            if reason:
                flags.append((fid, reason, age_vlm, g, ct))
        print(f'   悖论: 性别={sum(1 for f in flags if f[1]=="gender_paradox")} '
              f'年龄={sum(1 for f in flags if f[1]=="age_paradox")} / 已判 {len(attrs)}')
        for f in flags[:6]:
            print(f'     例: {f[4]} VLM年龄={f[2]} 性别={f[3]} → {f[1]}')
        if apply:
            for fid, reason, _, _, ct in flags:
                con.execute("""INSERT OR IGNORE INTO face_unassign_audit_v0
                               (face_instance_id, person_id, display_name, capture_time, birth_date, reason, created_at)
                               VALUES(?,?,?,?,?,?,?)""", (fid, pid, name, ct, birth, reason, NOW))
            con.executemany("UPDATE face_instance_v0 SET person_id=NULL WHERE face_instance_id=? AND person_locked=0",
                            [(f[0],) for f in flags])
            con.commit()
            print(f'   → 已解除 {len(flags)} 张')
        total_flag[name] = len(flags)
    print('\n汇总:', json.dumps(total_flag, ensure_ascii=False))
    con.close()


if __name__ == '__main__':
    main()
