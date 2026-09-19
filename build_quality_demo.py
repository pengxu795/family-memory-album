#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 static/quality-demo.html —— 新功能①-⑤ 前端展示方案演示页

纯静态：数据从 data/*.jsonl 内嵌进 HTML，不写库、不改 server.py。
图片走现有 /thumb?asset=<id>&edge=480 接口（需 server 在 8788 运行）。

用法：python3 build_quality_demo.py   （重跑可随时刷新演示数据）
"""
import json, random, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
THUMBS = DATA / "thumbs_mvp"


def has_t480(asset_id):
    """只用 t480 缩略图在线的资产（缺档的原图在 NAS，演示页不赌网络）。"""
    return (THUMBS / f"{asset_id[6:]}_t480.jpg").exists()


def load_jsonl(p, limit=0):
    f = DATA / p
    if not f.exists():
        return []
    rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows[:limit] if limit else rows


def sample(rows, n, seed=42):
    random.seed(seed)
    return random.sample(rows, min(n, len(rows)))


# 全量真实数据（T2 产物），按 tab 分层抽样控制页面体积
_blur_all = [r for r in load_jsonl("blur_scores_full.jsonl") if has_t480(r["asset_id"])]
blur_rows = []
for _lab in ("sharp", "soft", "blurry"):
    blur_rows += sample([r for r in _blur_all if r.get("label") == _lab], 20)

dedup_rows = sample([g for g in load_jsonl("dedup_groups_full.jsonl")
                     if has_t480(g["keep"]) and all(has_t480(d) for d in g["duplicates"])], 12)

# 每日五张：aesthetic 全量分 × DB 拍摄日 → 最近 4 天每天 top5（不重扫图）
con = sqlite3.connect(f"file:{DATA / 'family_memory.db'}?mode=ro", uri=True)
day_of = dict(con.execute(
    """SELECT asset_id, substr(capture_time,1,10) FROM media_asset
       WHERE media_type='photo' AND capture_time NOT LIKE '0000%'"""))
con.close()
_aeth = [{"asset_id": r["asset_id"], "aesthetic": r["aesthetic"], "day": day_of[r["asset_id"]]}
         for r in load_jsonl("aesthetic_full.jsonl")
         if r["asset_id"] in day_of and has_t480(r["asset_id"])]
_by_day = {}
for r in _aeth:
    _by_day.setdefault(r["day"], []).append(r)
top5_rows = []
_full_days = sorted((d for d, rs in _by_day.items() if len(rs) >= 5), reverse=True)[:4]
for d in _full_days:
    top5_rows += sorted(_by_day[d], key=lambda x: -x["aesthetic"])[:5]

by_m = {}
for r in load_jsonl("crop_params_full.jsonl"):
    if has_t480(r["asset_id"]):
        by_m.setdefault(r["method"], []).append(r)
crop_rows = []
for m in ("face", "saliency", "horizon", "center"):
    crop_rows += sample(by_m.get(m, []), 6)

# junk：全量扫描 + 深扫结果合并（已解码分类优先展示）
_deep = {r["asset_id"]: r for r in load_jsonl("junk_qr_deep.jsonl")}
decoded, undec = [], []
for r in load_jsonl("junk_scan_full.jsonl"):
    if not has_t480(r["asset_id"]):
        continue
    d = _deep.get(r["asset_id"])
    if d and d.get("kind") != "qrcode_undecoded":
        decoded.append(dict(r, deep_kind=d["kind"], deep_content=(d.get("content") or "")[:40]))
    else:
        undec.append(r)
junk_rows = sample(decoded, 6) + sample(undec, 8)

payload = {
    "blur": blur_rows,
    "dedup": dedup_rows,
    "top5": top5_rows,
    "crop": crop_rows,
    "junk": junk_rows,
}

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>画质与整理 · 新功能演示</title>
<style>
:root{--card:#fff;--bg:#f4f3ef;--line:#e4e2dc;--muted:#8a877f;--ink:#26241f;--accent:#c8541c;
--ok:#2e7d32;--warn:#b26a00;--bad:#c62828}
*{box-sizing:border-box;margin:0}
body{background:var(--bg);color:var(--ink);font:14px/1.5 -apple-system,"PingFang SC",sans-serif}
header{padding:18px 22px 10px}
h1{font-size:19px;font-weight:600}
.sub{color:var(--muted);font-size:12px;margin-top:3px}
.tabs{display:flex;gap:6px;padding:10px 22px;flex-wrap:wrap;border-bottom:1px solid var(--line)}
.tabs button{border:1px solid var(--line);background:var(--card);padding:8px 16px;border-radius:20px;
cursor:pointer;font-size:13px;color:var(--ink)}
.tabs button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
main{padding:18px 22px;max-width:1200px}
.note{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 14px;
font-size:12.5px;color:var(--muted);margin-bottom:16px}
.note b{color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;position:relative}
.card .ph{position:relative;width:100%;aspect-ratio:4/3;background:#e8e6e0;overflow:hidden}
.card .ph img{width:100%;height:100%;object-fit:cover;display:block}
.badge{position:absolute;left:8px;top:8px;padding:3px 9px;border-radius:12px;font-size:11px;color:#fff}
.b-sharp{background:var(--ok)} .b-soft{background:var(--warn)} .b-blurry{background:var(--bad)}
.score{position:absolute;right:8px;bottom:8px;background:#000a;color:#fff;padding:2px 8px;
border-radius:10px;font-size:11px}
.meta{padding:8px 10px;font-size:11.5px;color:var(--muted);display:flex;justify-content:space-between}
.id{font-family:ui-monospace,Menlo,monospace;opacity:.7}
.filters{display:flex;gap:8px;margin-bottom:14px}
.filters button{border:1px solid var(--line);background:var(--card);border-radius:16px;
padding:5px 13px;font-size:12px;cursor:pointer}
.filters button.on{background:var(--accent);color:#fff;border-color:var(--accent)}
/* 去重组 */
.dgroup{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:14px}
.dgroup .head{display:flex;gap:8px;align-items:center;font-size:12.5px;margin-bottom:10px}
.dgroup .engine{background:#ece9f5;color:#5b4a9e;border-radius:10px;padding:2px 9px;font-size:11px}
.drow{display:flex;gap:10px;flex-wrap:wrap}
.ditem{width:150px}
.ditem .ph{aspect-ratio:1;border-radius:8px;overflow:hidden;position:relative;background:#e8e6e0}
.ditem .ph img{width:100%;height:100%;object-fit:cover}
.ditem.keep .ph{outline:3px solid var(--ok);outline-offset:-3px}
.ditem.dup .ph img{opacity:.55}
.ditem .tag{position:absolute;left:6px;top:6px;font-size:10.5px;color:#fff;border-radius:9px;padding:2px 8px}
.ditem.keep .tag{background:var(--ok)} .ditem.dup .tag{background:#9e9e9e}
.ditem .sz{text-align:center;font-size:10.5px;color:var(--muted);padding-top:4px}
/* 五张选优 */
.dayhead{font-size:14px;font-weight:600;margin:6px 0 12px}
.dayhead span{color:var(--muted);font-weight:400;font-size:12px;margin-left:8px}
.top5{display:grid;grid-template-columns:repeat(5,1fr);gap:10px}
.top5 .card .rank{position:absolute;left:8px;top:8px;width:26px;height:26px;border-radius:50%;
background:var(--accent);color:#fff;display:flex;align-items:center;justify-content:center;
font-size:13px;font-weight:600}
/* 裁切预览 */
.cropgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.cropbox{position:relative;display:inline-block}
.mwin{position:absolute;border:2.5px solid var(--accent);box-shadow:0 0 0 999px #0006;
border-radius:3px;pointer-events:none}
.mwin.v{border-color:#7b1fa2}
.mlabel{position:absolute;z-index:2;left:8px;top:8px}
.m-label{background:#000b;color:#fff;border-radius:10px;padding:2px 9px;font-size:11px}
.m-label.v{background:#7b1fa2}
/* junk */
.jcard .ph img{filter:grayscale(.2)}
.legend{font-size:12px;color:var(--muted);margin:8px 0 14px}
.hide{display:none!important}
</style>
</head>
<body>
<header>
  <h1>画质与整理 · 新功能①-⑤ 展示方案</h1>
  <div class="sub">纯静态演示 · 数据内嵌自 data/*.jsonl · 不写库 · 缩略图走 /thumb 接口（需 server:8788）</div>
</header>
<div class="tabs" id="tabs">
  <button data-t="blur" class="on">① 模糊检测</button>
  <button data-t="dedup">② 相似去重</button>
  <button data-t="top5">③ 每日五张</button>
  <button data-t="crop">⑤ 智能裁切</button>
  <button data-t="junk">④ 截图/收款码</button>
</div>
<main id="main"></main>
<script>
const DATA = __PAYLOAD__;
const thumb = a => `/thumb?asset=${a}&edge=480`;
const ph = (a, cls) => `<img loading="lazy" src="${thumb(a)}" onerror="this.replaceWith(Object.assign(document.createElement('div'),{style:'padding:40% 8px;text-align:center;color:#999;font-size:11px',textContent:'缩略图不可用'}))">`;

const views = {
/* ① 模糊检测 ------------------------------------------------ */
blur(){const m=document.createElement('div');
 m.innerHTML=`<div class="note"><b>标记规则：</b>清晰(绿) / 偏软(黄) / 模糊(红)，取分块 top-25% Laplacian 方差。
 全量跑完后回写 quality_class，相册可一键"只看清晰的"。评分越高越清晰。</div>
 <div class="filters" id="bf">
   <button data-f="all" class="on">全部 ${DATA.blur.length}</button>
   <button data-f="sharp">清晰 ${DATA.blur.filter(r=>r.label==='sharp').length}</button>
   <button data-f="soft">偏软 ${DATA.blur.filter(r=>r.label==='soft').length}</button>
   <button data-f="blurry">模糊 ${DATA.blur.filter(r=>r.label==='blurry').length}</button>
 </div><div class="grid" id="bg"></div>`;
 const draw=f=>{m.querySelector('#bg').innerHTML=DATA.blur
   .filter(r=>f==='all'||r.label===f)
   .map(r=>`<div class="card"><div class="ph">${ph(r.asset_id)}
     <span class="badge b-${r.label}">${({sharp:'清晰',soft:'偏软',blurry:'模糊'})[r.label]}</span>
     <span class="score">${r.sharp_score}</span></div>
     <div class="meta"><span class="id">…${r.asset_id.slice(-6)}</span><span>中位 ${r.med_score}</span></div></div>`).join('');};
 draw('all');
 m.querySelector('#bf').onclick=e=>{const b=e.target.closest('button');if(!b)return;
   m.querySelectorAll('#bf button').forEach(x=>x.classList.toggle('on',x===b));draw(b.dataset.f);};
 return m;},

/* ② 相似去重 ------------------------------------------------ */
dedup(){const m=document.createElement('div');
 m.innerHTML=`<div class="note"><b>展示规则：</b>绿框=建议保留（组内文件最大），灰暗=清理候选。
 点击灰图切换选定状态——确认后的清理清单走 asset_filter_v0（A4 联调），本页不写库。</div>` +
 DATA.dedup.map((g,i)=>`<div class="dgroup">
   <div class="head"><span class="engine">${g.engine}</span>
     <span>组 ${i+1} · 保留 1 张 + 候选 ${g.duplicates.length} 张</span></div>
   <div class="drow">
     <div class="ditem keep"><div class="ph">${ph(g.keep)}<span class="tag">保留</span></div>
       <div class="sz">${(g.sizes[g.keep]/1024).toFixed(0)} KB · …${g.keep.slice(-6)}</div></div>
     ${g.duplicates.map(d=>`<div class="ditem dup" onclick="this.classList.toggle('dup')">
       <div class="ph">${ph(d)}<span class="tag">候选</span></div>
       <div class="sz">${(g.sizes[d]/1024).toFixed(0)} KB · …${d.slice(-6)}</div></div>`).join('')}
   </div></div>`).join('');
 return m;},

/* ③ 每日五张 ------------------------------------------------ */
top5(){const m=document.createElement('div');
 const by={};DATA.top5.forEach(r=>{(by[r.day]=by[r.day]||[]).push(r)});
 m.innerHTML=`<div class="note"><b>展示规则：</b>LAION Aesthetic V2 打分，每天取前 5。
 排名徽标 + 原始分数。可做"每日精选"独立 tab 或时间轴置顶折叠。</div>` +
 Object.entries(by).map(([d,rs])=>`<div class="dayhead">${d}<span>共 ${rs.length} 张精选（当日全量打分排序）</span></div>
   <div class="top5">${rs.map((r,i)=>`<div class="card"><div class="ph">${ph(r.asset_id)}
     <span class="rank">${i+1}</span><span class="score">${r.aesthetic}</span></div>
     <div class="meta"><span class="id">…${r.asset_id.slice(-6)}</span></div></div>`).join('')}</div>`).join('<br>');
 return m;},

/* ⑤ 智能裁切 ------------------------------------------------ */
crop(){const m=document.createElement('div');
 m.innerHTML=`<div class="note"><b>展示规则：</b>橙窗=4:3 横窗，紫窗=3:4 竖窗（人像特写自动转竖）。
 蒙版=裁掉区域。DB 只存 {x,y,w,h,method} 归一化参数，前端按参数渲染，不生成新图。
 method：face 人脸 / saliency 显著主体 / horizon 地平线 / center 中心。</div>
 <div class="cropgrid">` + DATA.crop.map(r=>{
   const style=`left:${r.x*100}%;top:${r.y*100}%;width:${r.w*100}%;height:${r.h*100}%`;
   const v=r.w<r.h?' v':'';
   return `<div class="card cropbox"><div class="ph" style="aspect-ratio:auto">
     <img loading="lazy" src="${thumb(r.asset_id)}" style="width:100%;display:block"
       onerror="this.closest('.card').style.display='none'">
     <div class="mwin${v}" style="${style}"></div>
     <span class="mlabel"><span class="m-label${v}">${({face:'人脸',saliency:'主体',horizon:'地平线',center:'中心'})[r.method]}${v?' · 3:4':''}</span></span>
     </div><div class="meta"><span class="id">…${r.asset_id.slice(-6)}</span><span>${({face:'人脸',saliency:'主体',horizon:'地平线',center:'中心'})[r.method]}</span></div></div>`;
 }).join('') + '</div>';
 return m;},

/* ④ 截图/收款码 --------------------------------------------- */
junk(){const m=document.createElement('div');
 const tag=r=>({payment_qrcode_wechat:'微信收款码',payment_qrcode_alipay:'支付宝收款码',
   qrcode:'二维码已解码',qrcode_undecoded:'QR 待精扫'})[r.deep_kind||'qrcode_undecoded'];
 const color=r=>r.deep_kind&&r.deep_kind!=='qrcode_undecoded'?'#c62828':'#5b4a9e';
 m.innerHTML=`<div class="note"><b>展示规则：</b>检出但解不出内容的二维码 → "待精扫"标记；
 深扫（原片 4x 重试）解出后分类为 微信收款码/支付宝码/普通码（红标）。
 命中项进 asset_filter_v0（reason=JUNK_XXX）默认从墙面隐藏，可恢复。本页不写库。</div>
 <div class="legend">全量 12491 张：检出 QR ${'1130'} 张待精扫，深扫已解码 ${DATA.junk.filter(r=>r.deep_kind).length} 张（本页抽样展示）：</div>
 <div class="grid">` + DATA.junk.map(r=>`<div class="card jcard"><div class="ph">${ph(r.asset_id)}
   <span class="badge" style="background:${color(r)}">${tag(r)}</span></div>
   <div class="meta"><span class="id">…${r.asset_id.slice(-6)}</span><span>${(r.deep_content||'').slice(0,18)}</span></div></div>`).join('') + '</div>';
 return m;}
};

const tabs=document.getElementById('tabs'), main=document.getElementById('main');
function show(t){main.replaceChildren(views[t]());}
tabs.onclick=e=>{const b=e.target.closest('button');if(!b)return;
 tabs.querySelectorAll('button').forEach(x=>x.classList.toggle('on',x===b));show(b.dataset.t);};
show('blur');
</script>
</body>
</html>"""

html = HTML.replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False))
out = ROOT / "static" / "quality-demo.html"
out.write_text(html, encoding="utf-8")
print(f"生成 {out}（{len(html)//1024} KB）")
print(f"数据: blur={len(blur_rows)} dedup={len(dedup_rows)} top5={len(top5_rows)} crop={len(crop_rows)} junk={len(junk_rows)}")
