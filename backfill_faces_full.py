#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全库人脸补跑 v0.2-full
======================
POC 阶段 person_intelligence_v0.py 只处理了 cohort_poc_v01 的 2,272 个资产。
本脚本把剩余全库资产(约 7,876 个)跑完 YuNet 检测 + SFace 嵌入，
并把新脸直接对齐已确认 6 人的脸向量质心完成身份归属(不再走人工聚类确认)。

用法:
  python backfill_faces_full.py --calibrate   # 只做阈值校准(秒级)
  python backfill_faces_full.py               # 全量检测+归属(小时级, 建议后台)
"""
import argparse, hashlib, json, os, sqlite3, subprocess, tempfile, time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get('FF_DATA_DIR') or (ROOT / 'data'))
DB = Path(os.environ.get('FF_DB_PATH') or (DATA_DIR / 'family_memory.db'))
# A4 终裁（2026-09-09 晚）：出厂默认回退 SFace（pair-F1 0.511<0.542 + WebFace4M 权重
# 非商业许可风险），详见 eval/threshold_addendum_20260909.md。
# 本地库已全量 AdaFace 向量，launchd plist 用 FF_FACE_BACKEND=adaface 钉住本地一致性。
# 2026-09-19 开源合规默认：opencv（YuNet+SFace，Apache-2.0 权重随分发）。
# adaface 权重（WebFace4M 非商业）不随分发，需 --adaface 自行下载并显式设 FF_FACE_BACKEND=adaface。
BACKEND = os.environ.get('FF_FACE_BACKEND', 'opencv').strip().lower()
OPENCV_MODEL_DIR = Path(os.environ.get(
    'FF_OPENCV_MODEL_DIR', str(ROOT / 'baked_models')))
INSIGHTFACE_MODEL_DIR = Path(os.environ.get(
    'FF_INSIGHTFACE_MODEL_DIR', str(ROOT / 'baked_models' / 'insightface')))
# A6 修复（2026-09-09）：PIPELINE 名绑定后端——断点表 person_asset_processing_v0
# 按 PIPELINE 记 success，旧版常量名不带模型导致「换后端重算」时全库被断点跳过
# （A4 首跑 todo=4 的根因）。绑定后：换后端=全库重检测，同后端重跑=幂等续跑。
PIPELINE = f'person-intelligence-v0.3-{BACKEND}'
MAX_DIM = 1280
SAMPLE_ROLE = 'library'
FFMPEG = '/opt/homebrew/bin/ffmpeg'

# 归属策略: kNN 投票(实测 leave-one-out 准确率 97.2%), 已弃质心方案(母女/兄妹相似度过高)
KNN_K = 10
# ① 阈值按模型查表（2026-09-09 标定，别再用一刀切 0.55）：
#   SFace 0.5437 = pair 口径 best-F1（eval/threshold_report_20260908-1608）；
#   AdaFace 0.51 = kNN top-1 生产语义标定（1079 manual 脸 leave-one-out，
#   同人召回 60.4% / 外人误过 3.3% / 精度 0.948，见 eval/threshold_addendum_20260909.md §3）
KNN_MIN_SIM_BY_MODEL = {
    'SFace-2021dec': 0.5437,
    'AdaFace-IR18-WebFace4M': 0.51,
}
KNN_MIN_SIM = 0.55                # 兜底默认（未知模型时）
KNN_VOTE_RATIO = 0.6


def stamp():
    return datetime.now(timezone.utc).isoformat()


def stable(prefix, value):
    return prefix + '_' + hashlib.sha256(value.encode()).hexdigest()[:24]


def resize_max(image):
    h, w = image.shape[:2]
    scale = min(1.0, MAX_DIM / max(h, w))
    if scale < 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return image


def load_photo(path, ext):
    if ext and ext.lower() in ('.heic', '.heif'):
        with tempfile.TemporaryDirectory(dir='/tmp') as td:
            out = Path(td) / 'image.jpg'
            subprocess.run(['sips', '-s', 'format', 'jpeg', '-Z', str(MAX_DIM), path, '--out', str(out)],
                           capture_output=True, check=True, timeout=45)
            image = cv2.imread(str(out), cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
    return resize_max(image) if image is not None else None


def load_video_frame(path, t):
    last_error = None
    for seek_time in (max(0, t), max(0, t * 0.75), max(0, t * 0.50)):
        try:
            p = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-ss', str(seek_time), '-i', path,
                                '-frames:v', '1', '-vf', f'scale={MAX_DIM}:{MAX_DIM}:force_original_aspect_ratio=decrease',
                                '-f', 'image2pipe', '-vcodec', 'mjpeg', '-'],
                               capture_output=True, check=True, timeout=30)
            image = cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)
            if image is not None:
                return image
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            last_error = exc
    if last_error:
        raise last_error
    return None


def frame_specs(media_type, duration):
    if media_type == 'photo':
        return [(None, None)]
    duration = duration or 0
    if duration <= 0:
        return [(0.0, None)]
    return [(round(duration * f, 3), None) for f in (0.10, 0.50, 0.90)]


def quality(det, image):
    x, y, w, h = det['bbox']
    x, y, w, h = int(x), int(y), max(1, int(round(w))), max(1, int(round(h)))
    crop = image[max(0, y):y + h, max(0, x):x + w]
    blur = float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) if crop.size else 0.0
    le = np.asarray(det['kps'][0]); re = np.asarray(det['kps'][1]); nose = np.asarray(det['kps'][2])
    eyes_mid = (le + re) / 2
    eye_dist = max(float(np.linalg.norm(le - re)), 1.0)
    pose = float(abs(nose[0] - eyes_mid[0]) / eye_dist)
    if min(w, h) < 40:
        cls = 'small'
    elif blur < 25:
        cls = 'blurry'
    elif pose > 0.38:
        cls = 'nonfrontal_candidate'
    else:
        cls = 'usable'
    return blur, pose, cls


def detect_asset(row, backend, con):
    # 人工标注的脸(sample_role='manual')是参照库样本, 重跑检测时必须保留
    # 2026-09-02 修复: 表情表(face_expression_v0)也联动删除, 否则旧脸被删后留下外键孤儿
    con.execute("DELETE FROM face_embedding_v0 WHERE face_instance_id IN (SELECT face_instance_id FROM face_instance_v0 WHERE asset_id=? AND sample_role != 'manual')", (row['asset_id'],))
    con.execute("DELETE FROM face_expression_v0 WHERE face_instance_id IN (SELECT face_instance_id FROM face_instance_v0 WHERE asset_id=? AND sample_role != 'manual')", (row['asset_id'],))
    con.execute("DELETE FROM face_instance_v0 WHERE asset_id=? AND sample_role != 'manual'", (row['asset_id'],))
    frames = 0
    faces_total = 0
    for frame_time, _ in frame_specs(row['media_type'], row['duration_seconds']):
        image = load_photo(row['absolute_path'], row['extension']) if frame_time is None else load_video_frame(row['absolute_path'], frame_time)
        if image is None:
            continue
        frames += 1
        h, w = image.shape[:2]
        dets = backend.detect(image)
        if not dets:
            continue
        frame_key = f"{row['asset_id']}:{'photo' if frame_time is None else f'{frame_time:.3f}'}"
        for idx, det in enumerate(dets):
            x, y, bw, bh = det['bbox']
            score = det['score']
            blur, pose, qclass = quality(det, image)
            norm_box = {'x': x / w, 'y': y / h, 'w': bw / w, 'h': bh / h, 'image_width': w, 'image_height': h}
            landmarks = [{'x': float(k[0]) / w, 'y': float(k[1]) / h} for k in det['kps']]
            fid = stable('face', f'{frame_key}:{idx}:{x:.2f}:{y:.2f}:{bw:.2f}:{bh:.2f}')
            con.execute('''INSERT OR IGNORE INTO face_instance_v0
              (face_instance_id,asset_id,frame_time_seconds,frame_key,bbox_json,landmarks_json,
               detection_score,face_width,face_height,blur_score,pose_score,quality_class,sample_role,detection_model,created_at)
              VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''', (
                fid, row['asset_id'], frame_time, frame_key, json.dumps(norm_box), json.dumps(landmarks),
                score, round(bw), round(bh), blur, pose, qclass, SAMPLE_ROLE, backend.DETECTION_MODEL, stamp()))
            try:
                emb = backend.embed(image, det).astype(np.float32).reshape(-1)
                norm = float(np.linalg.norm(emb))
                emb = emb / norm if norm else emb
                con.execute('INSERT OR IGNORE INTO face_embedding_v0 VALUES(?,?,?,?,?,?,?)',
                            (fid, backend.EMBEDDING_MODEL, len(emb), emb.astype('<f4').tobytes(), norm, 'success', stamp()))
            except Exception:
                con.execute('INSERT OR IGNORE INTO face_embedding_v0 VALUES(?,?,?,?,?,?,?)',
                            (fid, backend.EMBEDDING_MODEL, 0, None, None, 'failed', stamp()))
            faces_total += 1
    return frames, faces_total


def reembed_manual_faces(con, backend):
    """换嵌入模型后，人工标注(manual)脸按已存 bbox+landmarks 重嵌。

    A4 修复（2026-09-09）：manual 脸是 kNN 归属的参照库，detect_asset 故意不删
    它们（保 person 标注），但旧模型向量若不换，参照库(SFace 128 维)与候选
    (AdaFace 512 维)混模型算 cosine 全是垃圾——首跑 assignment assigned=10 /
    abstain=6854 的根因。bbox/landmarks/quality 原样保留，只重算向量。
    幂等：该脸当前模型已有 success 向量则跳过。
    返回 (重嵌数, 跳过数, 失败数)。"""
    rows = con.execute('''
        SELECT fi.face_instance_id, fi.bbox_json, fi.landmarks_json, fi.frame_time_seconds,
               mf.absolute_path, mf.extension, ma.media_type, ma.duration_seconds
        FROM face_instance_v0 fi
        JOIN media_file mf ON mf.asset_id = fi.asset_id
        JOIN media_asset ma USING(asset_id)
        WHERE fi.sample_role='manual' AND fi.bbox_json IS NOT NULL''').fetchall()
    todo = []
    for r in rows:
        ok = con.execute("""SELECT 1 FROM face_embedding_v0
            WHERE face_instance_id=? AND embedding_model=? AND status='success'""",
            (r['face_instance_id'], backend.EMBEDDING_MODEL)).fetchone()
        if not ok:
            todo.append(r)
    done = fail = 0
    for i, r in enumerate(todo, 1):
        try:
            bbox = json.loads(r['bbox_json'])
            lms = json.loads(r['landmarks_json']) if r['landmarks_json'] else None
            if r['media_type'] == 'video' and r['frame_time_seconds'] is not None:
                image = load_video_frame(r['absolute_path'], r['frame_time_seconds'])
            else:
                image = load_photo(r['absolute_path'], r['extension'])
            if image is None:
                raise RuntimeError('image load failed')
            h, w = image.shape[:2]
            kps = None
            if lms:
                kps = np.array([[p['x'] * w, p['y'] * h] for p in lms], dtype=np.float32)
            det = {'bbox': [bbox['x'] * w, bbox['y'] * h, bbox['w'] * w, bbox['h'] * h],
                   'score': 1.0, 'kps': kps}
            emb = backend.embed(image, det).astype(np.float32).reshape(-1)
            norm = float(np.linalg.norm(emb))
            emb = emb / norm if norm else emb
            con.execute("DELETE FROM face_embedding_v0 WHERE face_instance_id=?", (r['face_instance_id'],))
            con.execute('INSERT INTO face_embedding_v0 VALUES(?,?,?,?,?,?,?)',
                        (r['face_instance_id'], backend.EMBEDDING_MODEL, len(emb),
                         emb.astype('<f4').tobytes(), norm, 'success', stamp()))
            done += 1
        except Exception as exc:
            fail += 1
            print(f'[faces-full] manual-reembed fail {r["face_instance_id"]}: {exc}', flush=True)
        if i % 100 == 0:
            con.commit()
            print(f'[faces-full] manual reembed {i}/{len(todo)}', flush=True)
    con.commit()
    print(f'[faces-full] manual reembed: done={done} skipped={len(rows)-len(todo)} fail={fail}', flush=True)
    return done, len(rows) - len(todo), fail


def calibrate(con, emb_model=None):
    """用已标注脸测: 脸 vs 本人质心(leave-one-out) 与 vs 他人质心 的相似度分布"""
    if emb_model is None:
        from face_backends import embedding_model_of
        emb_model = embedding_model_of(BACKEND)
    rows = con.execute('''SELECT fi.face_instance_id, fi.person_id, fe.embedding
        FROM face_instance_v0 fi JOIN face_embedding_v0 fe USING(face_instance_id)
        WHERE fi.person_id IS NOT NULL AND fe.status='success' AND fe.embedding_model=? ''',
        (emb_model,)).fetchall()
    by_pid = {}
    for r in rows:
        by_pid.setdefault(r['person_id'], []).append((r['face_instance_id'], np.frombuffer(r['embedding'], dtype='<f4')))
    own, other = [], []
    for pid, items in by_pid.items():
        X = np.vstack([v for _, v in items])
        for i, (_, v) in enumerate(items):
            rest = np.delete(X, i, axis=0)
            c = rest.mean(axis=0); c /= max(np.linalg.norm(c), 1e-9)
            own.append(float(v @ c))
            for opid, oitems in by_pid.items():
                if opid == pid:
                    continue
                oc = np.vstack([v for _, v in oitems]).mean(axis=0)
                oc /= max(np.linalg.norm(oc), 1e-9)
                other.append(float(v @ oc))
    own = np.array(own); other = np.array(other)
    print(json.dumps({
        'labeled_faces': len(rows),
        'own_sim': {'p1': round(float(np.percentile(own, 1)), 3), 'p5': round(float(np.percentile(own, 5)), 3),
                    'p25': round(float(np.percentile(own, 25)), 3), 'mean': round(float(own.mean()), 3)},
        'other_sim': {'p99': round(float(np.percentile(other, 99)), 3), 'p95': round(float(np.percentile(other, 95)), 3),
                      'mean': round(float(other.mean()), 3)},
        'own_p5_gt_other_p99': bool(np.percentile(own, 5) > np.percentile(other, 99)),
    }, ensure_ascii=False, indent=2))


def assign_persons(con, k=KNN_K, min_sim=None, vote_ratio=KNN_VOTE_RATIO, emb_model=None):
    """未归属脸 → kNN 投票归属。返回统计。

    2026-09-07 v2 修复（用户反馈：2019 年出生的女儿被匹配到成年男性）：
    1. 参照库只收人工确认的脸(sample_role='manual')——'auto' 脸是上轮自动归属的
       产物，混进参照库会滚雪球放大错误（实测某家庭成员 24 张 auto 脚本脸当了参照）
    2. 出生日期硬门：候选照片拍摄时间早于人物出生日期 → 物理不可能，该参照票作废
    3. 投票 margin：top1 票数须领先 top2 ≥2 票，接近时弃权（防家人间模糊互抢）
    4. 儿童加严：当前年龄<12 的人物 min_sim+0.07、vote_ratio+0.1
       （SFace 以成人为主训练，儿童脸特征弱、跨年龄/性别交叉匹配显著更多）"""
    from collections import Counter
    # ① 换模型必须重标定：未显式给门槛时按嵌入模型查表（eval/threshold_addendum_20260909.md）
    if min_sim is None:
        min_sim = KNN_MIN_SIM_BY_MODEL.get(emb_model, KNN_MIN_SIM)
    # 2026-09-08：算法偏好（models.html「算法偏好」）——用户值覆盖脚本默认
    try:
        r = con.execute("SELECT value FROM app_setting_v0 WHERE key='algo_face_knn_min_sim'").fetchone()
        if r and r[0]:
            min_sim = float(r[0])
    except Exception:
        pass
    try:
        r2 = con.execute("SELECT value FROM app_setting_v0 WHERE key='algo_face_child_strict'").fetchone()
        child_strict = (r2 is None) or (str(r2[0]) == '1')
    except Exception:
        child_strict = True
    labeled = con.execute('''SELECT fi.person_id, p.display_name, p.birth_date,
        fi.quality_class, fi.sample_role, fe.embedding
        FROM face_instance_v0 fi JOIN face_embedding_v0 fe USING(face_instance_id)
        JOIN person p USING(person_id)
        WHERE fi.person_id IS NOT NULL AND fe.status='success' AND fe.embedding_model=?''',
        (emb_model,)).fetchall()
    births = {}
    for r in labeled:
        births.setdefault(r['person_id'], (r['birth_date'] or '')[:10])
    today = datetime.now(timezone.utc).date()
    child_pids = set()
    if child_strict:
        for pid, b in births.items():
            if not b:
                continue
            try:
                age = (today - datetime.strptime(b, '%Y-%m-%d').date()).days / 365.25
                if age < 12:
                    child_pids.add(pid)
            except ValueError:
                pass
    # 标注脸只用人工确认(manual)的高质量子集做参照库;
    # library=全量检测产物、auto=上轮自动归属产物, 都不进参照(防误差滚雪球)
    ref = [r for r in labeled if r['quality_class'] in ('usable', 'small') and r['sample_role'] == 'manual']
    if not ref:
        return {'assigned': 0, 'error': 'no manually confirmed reference faces'}
    X = np.vstack([np.frombuffer(r['embedding'], dtype='<f4') for r in ref])
    y = np.array([r['person_id'] for r in ref])
    y_birth = np.array([births.get(r['person_id'], '') for r in ref])
    names = {r['person_id']: r['display_name'] for r in labeled}
    rows = con.execute('''SELECT fi.face_instance_id, fe.embedding, ma.capture_time
        FROM face_instance_v0 fi
        JOIN face_embedding_v0 fe USING(face_instance_id)
        JOIN media_asset ma USING(asset_id)
        WHERE fi.person_id IS NULL AND fe.status='success' AND fe.embedding_model=?''',
        (emb_model,)).fetchall()
    assigned = 0
    abstain = 0
    birth_blocked = 0
    per_person = {}
    for r in rows:
        v = np.frombuffer(r['embedding'], dtype='<f4')
        sims = X @ v
        ct = (r['capture_time'] or '')[:10]
        # 出生日期硬门：拍这张时还没出生的人, 参照票直接作废
        if ct >= '1000':
            dead = np.array([(b != '') and (ct < b) for b in y_birth], dtype=bool)
            if dead.any():
                sims[dead] = -1.0
        order = np.argsort(-sims)[:k]
        top_pid = y[order[0]]
        if sims[order[0]] < 0:
            birth_blocked += 1
            abstain += 1
            continue
        eff_min = min_sim + (0.07 if top_pid in child_pids else 0.0)
        if sims[order[0]] < eff_min:
            abstain += 1
            continue
        votes = Counter(y[order])
        ranked = votes.most_common(2)
        pid, cnt = ranked[0]
        eff_ratio = min(0.85, vote_ratio + (0.1 if pid in child_pids else 0.0))
        if len(ranked) > 1 and cnt - ranked[1][1] < 2:
            abstain += 1
            continue
        if cnt / min(k, len(order)) >= eff_ratio:
            con.execute('UPDATE face_instance_v0 SET person_id=? WHERE face_instance_id=?', (pid, r['face_instance_id']))
            per_person[names[pid]] = per_person.get(names[pid], 0) + 1
            assigned += 1
        else:
            abstain += 1
    con.commit()
    return {'assigned': assigned, 'abstain_left_null': abstain, 'birth_blocked': birth_blocked,
            'per_person': per_person, 'ref_faces': len(ref), 'ref_source': 'manual_only'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--calibrate', action='store_true')
    ap.add_argument('--k', type=int, default=KNN_K)
    ap.add_argument('--min-sim', type=float, default=None,
                    help='kNN 门槛；缺省按嵌入模型查 KNN_MIN_SIM_BY_MODEL')
    ap.add_argument('--vote-ratio', type=float, default=KNN_VOTE_RATIO)
    ap.add_argument('--assign-only', action='store_true', help='只做归属, 不检测')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 个资产(冒烟测试)')
    ap.add_argument('--shard-index', type=int, default=0, help='分片索引(0-based), 与 --shard-count 配合并行检测')
    ap.add_argument('--shard-count', type=int, default=1, help='分片总数(1=不分片); >1 时本进程只检测不归属')
    args = ap.parse_args()

    con = sqlite3.connect(DB, timeout=120)
    con.row_factory = sqlite3.Row
    # autocommit: 每条语句即时提交, 写锁只在单条语句上短暂持有,
    # 避免长事务在 ffmpeg/检测期间一直占住写锁导致并行分片 database is locked。
    con.isolation_level = None
    con.execute('PRAGMA journal_mode=WAL')
    con.execute('PRAGMA synchronous=NORMAL')
    # 索引已在首次运行建好(idx_fi_asset / idx_fi_person), 此处不再重复 CREATE, 避免并发启动竞争 schema 锁。

    if args.calibrate:
        calibrate(con)
        return

    if not args.assign_only:
        # A4 裁定后三后端：adaface(默认/出厂,MIT) | insightface(本地可选) | opencv(Apache 2.0)
        from face_backends import create_backend
        if BACKEND == 'insightface':
            mdir = INSIGHTFACE_MODEL_DIR
        elif BACKEND == 'adaface':
            mdir = os.environ.get('FF_ADAFACE_DIR') or str(ROOT.parent / 'models' / 'adaface')
        else:
            mdir = OPENCV_MODEL_DIR
        backend = create_backend(BACKEND, mdir)
        print(f'[faces-full] backend={BACKEND} ({backend.DETECTION_MODEL} + {backend.EMBEDDING_MODEL})', flush=True)
        excluded = {r[0] for r in con.execute('''WITH ranked AS (SELECT group_id,asset_id,ROW_NUMBER() OVER(PARTITION BY group_id ORDER BY canonical_score DESC) rn FROM canonical_candidate_v01) SELECT asset_id FROM ranked WHERE rn>1''')}
        rows = con.execute('''WITH ranked AS (SELECT mf.*,ROW_NUMBER() OVER(PARTITION BY asset_id ORDER BY byte_size DESC,absolute_path) rn FROM media_file mf)
            SELECT r.*,ma.media_type,ma.duration_seconds FROM ranked r JOIN media_asset ma USING(asset_id)
            WHERE r.rn=1 ORDER BY r.absolute_path''').fetchall()
        todo = [r for r in rows if r['asset_id'] not in excluded and not con.execute(
            "SELECT 1 FROM person_asset_processing_v0 WHERE asset_id=? AND pipeline_version=? AND status='success'",
            (r['asset_id'], PIPELINE)).fetchone()]
        if args.limit:
            todo = todo[:args.limit]
        if args.shard_count > 1:
            todo = [r for i, r in enumerate(todo) if i % args.shard_count == args.shard_index]
        t0 = time.time()
        print(f'[faces-full] shard={args.shard_index}/{args.shard_count} todo={len(todo)}', flush=True)
        failed = 0
        for i, row in enumerate(todo, 1):
            try:
                frames, faces = detect_asset(row, backend, con)
                status, err = 'success', None
            except Exception as exc:
                frames = faces = 0
                status, err = 'failed', f'{type(exc).__name__}: {exc}'[:500]
                failed += 1
            con.execute('INSERT OR REPLACE INTO person_asset_processing_v0 VALUES(?,?,?,?,?,?,?)',
                        (row['asset_id'], PIPELINE, frames, faces, status, err, stamp()))
            if i % 25 == 0:
                con.commit()
                speed = i / (time.time() - t0 + 1e-9)
                eta_min = (len(todo) - i) / max(speed, 1e-9) / 60
                print(f'[faces-full] {i}/{len(todo)} failed={failed} faces={con.execute("SELECT count(*) FROM face_instance_v0").fetchone()[0]} speed={speed:.1f}/s eta={eta_min:.0f}min', flush=True)
        con.commit()
        print(f'[faces-full] detection done: {len(todo)} assets, failed={failed}, {time.time()-t0:.0f}s', flush=True)

    # 归属: 单进程(不分片)时检测完自动归属; 分片模式(--shard-count>1)只检测, 归属由单独 --assign-only 跑一次
    do_assign = args.assign_only or args.shard_count <= 1
    if do_assign:
        # A4 修复：归属前确保 manual 参照脸已重嵌为当前模型向量（混模型 kNN=垃圾）
        from face_backends import create_backend, embedding_model_of
        emb_model = embedding_model_of(BACKEND)
        try:
            backend  # 检测分支里已建（--assign-only 时未建）
        except NameError:
            mdir = (INSIGHTFACE_MODEL_DIR if BACKEND == 'insightface'
                    else (os.environ.get('FF_ADAFACE_DIR') or str(ROOT.parent / 'models' / 'adaface'))
                    if BACKEND == 'adaface' else OPENCV_MODEL_DIR)
            backend = create_backend(BACKEND, mdir)
        reembed_manual_faces(con, backend)
        stats = assign_persons(con, args.k, args.min_sim, args.vote_ratio, emb_model=emb_model)
        print('[faces-full] assignment:', json.dumps(stats, ensure_ascii=False), flush=True)
        # 每人覆盖资产数
        for r in con.execute('''SELECT p.display_name, COUNT(DISTINCT fi.asset_id) c FROM person p
            LEFT JOIN face_instance_v0 fi USING(person_id) GROUP BY 1 ORDER BY c DESC'''):
            print(f'[faces-full]   {r["display_name"]}: {r["c"]} 张资产', flush=True)
    else:
        print(f'[faces-full] shard={args.shard_index} detection done (assign skipped)', flush=True)
    con.close()


if __name__ == '__main__':
    main()
