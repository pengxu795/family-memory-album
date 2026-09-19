# 快审模糊带：|own-other|<=0.03 的脸 VLM 判年龄性别，悖论即摘
import base64, json, re, sqlite3, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_face_paradox import vlm, parse_json, CROPS, DB, AGE_TOL, NOW, GENDER_KNOWN

APPLY = '--apply' in sys.argv
con = sqlite3.connect(DB, timeout=60)
con.row_factory = sqlite3.Row
con.execute("""CREATE TABLE IF NOT EXISTS face_vlm_attr_v0 (
    face_instance_id TEXT PRIMARY KEY, age_years REAL, age_stage TEXT, gender TEXT,
    raw TEXT, created_at TEXT)""")
con.execute("""CREATE TABLE IF NOT EXISTS face_unassign_audit_v0 (
    face_instance_id TEXT PRIMARY KEY, person_id TEXT, display_name TEXT,
    capture_time TEXT, birth_date TEXT, reason TEXT, created_at TEXT)""")

refs = {}
for r in con.execute("""SELECT fi.person_id, p.display_name, fe.embedding
        FROM face_instance_v0 fi JOIN face_embedding_v0 fe USING(face_instance_id)
        JOIN person p USING(person_id)
        WHERE fi.person_id IS NOT NULL AND fi.sample_role='manual' AND fe.status='success'"""):
    refs.setdefault(r['person_id'], {'name': r['display_name'], 'vecs': []})
    refs[r['person_id']]['vecs'].append(np.frombuffer(r['embedding'], dtype='<f4'))
for pid, d in refs.items():
    M = np.vstack(d['vecs']); d['M'] = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)

total = 0
# #16：目标人物动态取自 DB（12 岁内 confirmed 人物），代码零人名
_children = [r[0] for r in con.execute(
    """SELECT display_name FROM person
       WHERE identity_status='confirmed' AND birth_date IS NOT NULL
         AND (julianday('now') - julianday(birth_date)) / 365.25 < 12""")]
for name in _children:
    prow = con.execute("SELECT person_id, birth_date FROM person WHERE display_name=?", (name,)).fetchone()
    pid, birth = prow['person_id'], (prow['birth_date'] or '')[:10]
    # 期望性别: 硬事实优先, VLM 参照脸兜底
    exp_gender = GENDER_KNOWN.get(name)
    try:
        if exp_gender:
            raise StopIteration
        ref_imgs = [base64.b64encode((CROPS / f"{r['face_instance_id']}_k24_s480.jpg").read_bytes()).decode()
                    for r in con.execute("""SELECT face_instance_id FROM face_instance_v0
                        WHERE person_id=? AND sample_role='manual' ORDER BY face_width*face_height DESC LIMIT 4""", (pid,))
                    if (CROPS / f"{r['face_instance_id']}_k24_s480.jpg").exists()]
        txt = vlm(ref_imgs, '这几张是同一个人的脸部特写。只回答JSON: {"gender":"male|female"}')
        j = parse_json(txt)
        if isinstance(j, list) and j: j = j[0]
        if isinstance(j, dict): exp_gender = j.get('gender')
    except StopIteration:
        pass
    except Exception as e:
        print(f'{name} 参照性别失败: {e}')
    print(f'== {name} 期望性别={exp_gender}')

    rows = con.execute("""SELECT fi.face_instance_id, ma.capture_time, fe.embedding
        FROM face_instance_v0 fi JOIN face_embedding_v0 fe USING(face_instance_id)
        JOIN media_asset ma USING(asset_id)
        WHERE fi.person_id=? AND fi.sample_role!='manual' AND fi.person_locked=0 AND fe.status='success'""",
        (pid,)).fetchall()
    band = []
    for r in rows:
        v = np.frombuffer(r['embedding'], dtype='<f4'); v = v / (np.linalg.norm(v) + 1e-9)
        sims = {p2: float(np.sort(d['M'] @ v)[-3:].mean()) for p2, d in refs.items()}
        own = sims.get(pid, 0.0)
        other = max((s for p2, s in sims.items() if p2 != pid), default=0.0)
        if abs(own - other) <= 0.03:
            band.append(r)
    print(f'   模糊带 {len(band)} 张，VLM 审判中...')
    flags = []
    for i in range(0, len(band), 6):
        chunk = band[i:i + 6]
        imgs = [base64.b64encode((CROPS / f"{r['face_instance_id']}_k24_s480.jpg").read_bytes()).decode() for r in chunk]
        try:
            txt = vlm(imgs, f'这是{len(imgs)}张不同照片的人脸特写。对每张分别估计此人大致年龄(岁数,数字)和性别。'
                            '只回答JSON数组: [{"i":1,"age":数字,"gender":"male|female|unclear"},...]（i从1按顺序）')
        except Exception as e:
            print(f'   批次失败: {e}'); continue
        arr = parse_json(txt)
        if not isinstance(arr, list) or len(arr) != len(chunk):
            print(f'   解析失败: {txt[:80]}'); continue
        for r, item in zip(chunk, arr):
            try:
                age = float(item.get('age'))
            except (TypeError, ValueError):
                continue
            g = item.get('gender')
            fid = r['face_instance_id']
            con.execute("""INSERT OR REPLACE INTO face_vlm_attr_v0
                           (face_instance_id, age_years, age_stage, gender, raw, created_at)
                           VALUES(?,?,?,?,?,?)""", (fid, age, None, g, json.dumps(item, ensure_ascii=False), NOW))
            ct = (r['capture_time'] or '')[:10]
            reason = None
            if exp_gender in ('male', 'female') and g in ('male', 'female') and g != exp_gender:
                reason = 'gender_paradox'
            elif birth and ct >= '1000' and re.match(r'^\d{4}-\d{2}-\d{2}$', ct):
                age_true = (datetime.strptime(ct, '%Y-%m-%d') - datetime.strptime(birth, '%Y-%m-%d')).days / 365.25
                if abs(age - age_true) > AGE_TOL:
                    reason = 'age_paradox'
            if reason:
                flags.append((fid, reason, age, g, ct))
    con.commit()
    print(f'   悖论 {len(flags)} 张:')
    for f in flags:
        print(f'     {f[4]} VLM年龄={f[2]} 性别={f[3]} → {f[1]}')
    if APPLY:
        for fid, reason, _, _, ct in flags:
            con.execute("""INSERT OR IGNORE INTO face_unassign_audit_v0
                           (face_instance_id, person_id, display_name, capture_time, birth_date, reason, created_at)
                           VALUES(?,?,?,?,?,?,?)""", (fid, pid, name, ct, birth, reason, NOW))
        con.executemany("UPDATE face_instance_v0 SET person_id=NULL WHERE face_instance_id=? AND person_locked=0",
                        [(f[0],) for f in flags])
        con.commit()
        print(f'   → 已解除 {len(flags)} 张')
    total += len(flags)
print(f'\n合计悖论 {total} 张{"（已apply）" if APPLY else "（dry-run）"}')
