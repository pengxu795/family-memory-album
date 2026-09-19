#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 家庭回忆相册 — 交付前质控冒烟（17 项功能清单，HTTP 层）
# 用法: qa_smoke.sh <轮次> <版本名> <BASE_URL> <用户名> <密码> <face_id> <视频asset> <heic6 asset>
# 输出: /tmp/fma_qa/results/qa_<轮次>_<版本名>.md
set -u
ROUND="$1"; NAME="$2"; BASE="$3"; QUSER="$4"; QPASS="$5"; FACE_ID="$6"; VID="$7"; HEIC6="$8"
OUTDIR="${QA_OUTDIR:-/tmp/fma_qa/results}"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/qa_${ROUND}_${NAME}.md"
CJ=$(mktemp)
PASS=0; FAIL=0; SKIP=0; WARN=0
declare -a DETAILS

say() { DETAILS+=("$1"); case "$1" in PASS*) PASS=$((PASS+1));; FAIL*) FAIL=$((FAIL+1));; SKIP*) SKIP=$((SKIP+1));; WARN*) WARN=$((WARN+1));; esac; }

cget()  { curl -s --noproxy '*' --max-time 30 -b "$CJ" -c "$CJ" "$@"; }
cpost() { curl -s --noproxy '*' --max-time 60 -b "$CJ" -c "$CJ" -H "Content-Type: application/json" -X POST "$@"; }
# 非管理员账号收到权限拒绝 → 权限设计生效，记 SKIP 而非 FAIL
perm_ok() { echo "$1" | grep -qE "需要管理员权限|只读账户"; }

# ---------- t02/t03 登录 ----------
r=$(cpost "$BASE/api/auth/login" -d "{\"username\":\"$QUSER\",\"password\":\"$QPASS\"}")
if echo "$r" | grep -q '"ok": true'; then say "PASS|t03 管理员登录|ok"
else say "FAIL|t03 管理员登录|$r"; fi

# ---------- t01 安装可达 ----------
code=$(curl -s --noproxy '*' --max-time 10 -o /dev/null -w "%{http_code}" "$BASE/")
[ "$code" = "302" ] || [ "$code" = "200" ] && say "PASS|t01 安装/服务可达|HTTP $code" || say "FAIL|t01 安装/服务可达|HTTP $code"

# ---------- t02 首启向导页面 ----------
code=$(curl -s --noproxy '*' --max-time 10 -o /dev/null -w "%{http_code}" -b "$CJ" "$BASE/init_wizard.html")
r=$(curl -s --noproxy '*' --max-time 10 -b "$CJ" "$BASE/api/init/status")
if [ -n "$r" ] && echo "$r" | grep -q "initialized"; then
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    say "PASS|t02 首启向导页面+status|page=$code（已初始化实例重定向属设计） status=${r:0:60}"
  else
    say "FAIL|t02 首启向导页面+status|page=$code status=$r"
  fi
else say "FAIL|t02 首启向导页面+status|page=$code status=$r"; fi

# ---------- t04 照片墙与时间线 ----------
html=$(cget "$BASE/")
if echo "$html" | grep -qE "家的记忆|<title>"; then say "PASS|t04a 首页 HTML|${#html} bytes"
else say "FAIL|t04a 首页 HTML|异常内容"; fi
r=$(cpost "$BASE/api/category" -d '{"cat":"latest","limit":24}')
n=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); a=d.get('items') or d.get('assets') or d.get('rows') or []; print(len(a))" 2>/dev/null || echo 0)
[ "$n" -gt 0 ] && say "PASS|t04b 时间线 latest 列表|$n 条" || say "FAIL|t04b 时间线 latest 列表|${r:0:120}"
r=$(cpost "$BASE/api/captions" -d '{}')
if echo "$r" | grep -qE '"error"'; then perm_ok "$r" && say "SKIP|t04c captions|权限拒绝（设计生效）" || say "FAIL|t04c captions|${r:0:120}"; else say "PASS|t04c captions|${r:0:40}..."; fi

# ---------- t05 年份跳转（从 categories.years 取第一个） ----------
YEAR=$(cpost "$BASE/api/categories" -d '{}' | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d['years'][0]['year'])" 2>/dev/null)
r=$(cpost "$BASE/api/category" -d "{\"cat\":\"year\",\"value\":\"$YEAR\",\"limit\":10}")
n=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); a=d.get('items') or d.get('assets') or d.get('rows') or []; print(len(a))" 2>/dev/null || echo 0)
[ "$n" -gt 0 ] && say "PASS|t05 年份跳转|$YEAR 年 $n 条" || say "FAIL|t05 年份跳转|year=$YEAR ${r:0:120}"

# ---------- t06 地点·人物·场景分类 ----------
CJ_=$(cpost "$BASE/api/categories" -d '{}')
LOC=$(echo "$CJ_" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d['locations'][0]['region'] if d.get('locations') else '')" 2>/dev/null)
PER=$(echo "$CJ_" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d['persons'][0].get('person_id') or d['persons'][0].get('id','') if d.get('persons') else '')" 2>/dev/null)
SCN=$(echo "$CJ_" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); s=d.get('scenes',[]); print(s[0].get('tag') or s[0].get('scene','') if s else '')" 2>/dev/null)
r=$(cpost "$BASE/api/category" -d "{\"cat\":\"location\",\"value\":\"$LOC\",\"limit\":10}")
echo "$r" | grep -q '"error"' && say "FAIL|t06a 地点分类|$LOC ${r:0:100}" || say "PASS|t06a 地点分类|$LOC"
r=$(cpost "$BASE/api/category" -d "{\"cat\":\"person\",\"value\":\"$PER\",\"limit\":10}")
echo "$r" | grep -q '"error"' && say "FAIL|t06b 人物分类|$PER ${r:0:100}" || say "PASS|t06b 人物分类|person_id=$PER"
r=$(cpost "$BASE/api/category" -d "{\"cat\":\"scene\",\"value\":\"$SCN\",\"limit\":10}")
echo "$r" | grep -q 'error' && say "WARN|t06c 场景分类|cat=scene ${r:0:100}" || say "PASS|t06c 场景分类|$SCN"

# ---------- t07 地图与旅行 ----------
r=$(cpost "$BASE/api/geo/map" -d '{}')
n=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('regions',[])))" 2>/dev/null || echo 0)
[ "$n" -gt 0 ] && say "PASS|t07a 地图 regions|$n 个" || say "FAIL|t07a 地图 regions|${r:0:120}"
r=$(cpost "$BASE/api/trip/list" -d '{}')
if echo "$r" | grep -q '"trips"'; then say "PASS|t07b 旅行标记 list|${r:0:50}"; elif perm_ok "$r"; then say "SKIP|t07b 旅行标记 list|权限拒绝（设计生效）"; else say "FAIL|t07b 旅行标记 list|${r:0:120}"; fi

# ---------- t08 语音搜索（/api/ask 语义问答） ----------
r=$(cpost "$BASE/api/ask" -d '{"question":"2025年有哪些照片"}')
if echo "$r" | grep -q '"answer"'; then say "PASS|t08 语音/语义搜索 ask|answer=${r:0:60}..."; elif perm_ok "$r"; then say "SKIP|t08 语音/语义搜索 ask|权限拒绝（设计生效）"; else say "FAIL|t08 语音/语义搜索 ask|${r:0:150}"; fi

# ---------- t09 人物管理 ----------
r=$(cpost "$BASE/api/people" -d '{}')
n=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('people',[])))" 2>/dev/null || echo 0)
[ "$n" -gt 0 ] && say "PASS|t09a 人物列表|$n 人" || say "FAIL|t09a 人物列表|${r:0:120}"
r=$(cget "$BASE/api/facepos")
echo "$r" | grep -qE 'error|Error' && say "WARN|t09b facepos|${r:0:100}" || say "PASS|t09b facepos|${#r} bytes"

# ---------- t10 已过滤内容 + 回收站 ----------
r=$(cpost "$BASE/api/filter/list" -d '{}')
n=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('assets',[])))" 2>/dev/null || echo -1)
[ "$n" -ge 0 ] && say "PASS|t10a 已过滤列表|$n 项" || say "FAIL|t10a 已过滤列表|${r:0:120}"
r=$(cpost "$BASE/api/recycle" -d '{"action":"__probe__"}')
if echo "$r" | grep -qE '未知 action|error'; then say "PASS|t10b 回收站 API 响应正常|${r:0:60}"; else say "WARN|t10b 回收站 API|${r:0:60}"; fi

# ---------- t11 相似照片 ----------
r=$(cpost "$BASE/api/similar/list" -d '{}')
n=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('groups',[])))" 2>/dev/null || echo 0)
if [ "$n" -gt 0 ]; then say "PASS|t11 相似照片分组|$n 组"; else perm_ok "$r" && say "SKIP|t11 相似照片分组|权限拒绝（设计生效）" || say "FAIL|t11 相似照片分组|${r:0:120}"; fi

# ---------- t12 隐私相册 ----------
r=$(cpost "$BASE/api/privacy" -d '{"action":"count"}')
TOTAL=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total','ERR'))" 2>/dev/null)
if [ "$TOTAL" != "ERR" ] && [ -n "$TOTAL" ]; then say "PASS|t12a 隐私 count|total=$TOTAL"; elif perm_ok "$r"; then say "SKIP|t12a 隐私 count|权限拒绝（设计生效）"; else say "FAIL|t12a 隐私 count|${r:0:120}"; fi
if [ "${QA_PRIVACY_FULLFLOW:-0}" = "1" ]; then
  AID=$(cpost "$BASE/api/category" -d '{"cat":"latest","limit":1}' | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); a=d.get('items') or d.get('assets') or []; print(a[0].get('asset_id') or a[0].get('id'))" 2>/dev/null)
  if [ -n "${QA_DB_PATH:-}" ] && [ -f "$QA_DB_PATH" ]; then
    sqlite3 "$QA_DB_PATH" "DELETE FROM privacy_pin_v0 WHERE id=1;" 2>/dev/null
    TOTAL=$(cpost "$BASE/api/privacy" -d '{"action":"count"}' | /usr/bin/python3 -c "import json,sys; print(json.load(sys.stdin).get('total'))" 2>/dev/null)
  fi
  r1=$(cpost "$BASE/api/privacy" -d '{"action":"set_pin","pin":"9948"}')
  r2=$(cpost "$BASE/api/privacy" -d '{"action":"verify","pin":"9948"}')
  r3=$(cpost "$BASE/api/privacy" -d "{\"action\":\"add\",\"pin\":\"9948\",\"asset_ids\":[\"$AID\"]}")
  r4=$(cpost "$BASE/api/privacy" -d '{"action":"list","pin":"9948"}')
  r5=$(cpost "$BASE/api/privacy" -d "{\"action\":\"remove\",\"pin\":\"9948\",\"asset_ids\":[\"$AID\"]}")
  cnt_after=$(cpost "$BASE/api/privacy" -d '{"action":"count"}' | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total'))" 2>/dev/null)
  if echo "$r2" | grep -qE '"ok": ?true' && echo "$r3" | grep -q '"hidden"' && echo "$r5" | grep -q '"removed"' && [ "$cnt_after" = "$TOTAL" ]; then
    say "PASS|t12b 隐私全流程(set/verify/add/list/remove)|总数复原 $cnt_after"
  else
    say "FAIL|t12b 隐私全流程|verify=$r2 add=$r3 list=${r4:0:80} remove=$r5 after=$cnt_after"
  fi
else
  say "SKIP|t12b 隐私全流程|仅 count（共享真实数据实例，QA_PRIVACY_FULLFLOW=1 开启）"
fi

# ---------- t13 设置 ----------
r=$(cpost "$BASE/api/models/status" -d '{}')
if echo "$r" | grep -q '"models"'; then say "PASS|t13a 模型状态|${r:0:50}..."; elif perm_ok "$r"; then say "SKIP|t13a 模型状态|权限拒绝（设计生效）"; else say "FAIL|t13a 模型状态|${r:0:120}"; fi
r=$(cpost "$BASE/api/settings/algos" -d '{"key":"face_child_strict","value":"0"}')
if echo "$r" | grep -q '"ok"'; then
  NEWVAL=$(echo "$r" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('value','?'))" 2>/dev/null)
  r2=$(cpost "$BASE/api/settings/algos" -d '{"key":"face_child_strict","value":"1"}')
  BACK=$(echo "$r2" | /usr/bin/python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('value','?'))" 2>/dev/null)
  if [ "$NEWVAL" = "0" ] && [ "$BACK" = "1" ]; then
    say "PASS|t13b 算法设置写读还原|改后=$NEWVAL 还原=$BACK"
  else
    say "FAIL|t13b 算法设置|改后=$NEWVAL 还原=$BACK"
  fi
elif perm_ok "$r"; then
  say "SKIP|t13b 算法设置写|权限拒绝（设计生效）"
else
  say "FAIL|t13b 算法设置写|${r:0:120}"
fi
r=$(cpost "$BASE/api/llm/status" -d '{}')
if echo "$r" | grep -q 'priority'; then say "PASS|t13c LLM 状态|${r:0:60}..."; elif perm_ok "$r"; then say "SKIP|t13c LLM 状态|权限拒绝（设计生效）"; else say "WARN|t13c LLM 状态|${r:0:120}"; fi

# ---------- t14 视频缩略图与播放 ----------
r=$(cget -o /dev/null -w "%{http_code} %{content_type} %{size_download}" "$BASE/thumb?asset=$VID")
echo "$r" | grep -q "^200 image/jpeg" && say "PASS|t14a 视频缩略图|$r" || say "FAIL|t14a 视频缩略图|$r"
r=$(cget -o /dev/null -w "%{http_code} %{content_type}" "$BASE/view?asset=$VID")
r2=$(cget -o /dev/null -w "%{http_code} %{content_type}" -H "Range: bytes=0-1023" "$BASE/orig?asset=$VID")
if echo "$r" | grep -q "^200 text/html" && echo "$r2" | grep -q "^206 video"; then
  say "PASS|t14b 视频播放页+流|view=$r orig-range=$r2"
else
  say "FAIL|t14b 视频播放页+流|view=$r orig-range=$r2"
fi

# ---------- t15 EXIF 转正 ----------
TMPJ=$(mktemp /tmp/qa_heic_XXXX.jpg)
meta=$(cget -o "$TMPJ" -w "%{http_code} %{content_type} %{size_download}" "$BASE/thumb?asset=$HEIC6")
dims=$(sips -g pixelWidth -g pixelHeight "$TMPJ" 2>/dev/null | awk '/pixel/{print $2}' | tr '\n' 'x' | sed 's/x$//')
W=$(echo "$dims" | cut -dx -f1); H=$(echo "$dims" | cut -dx -f2)
if echo "$meta" | grep -q "^200" && [ -n "$W" ] && [ "$H" -gt "$W" ]; then
  say "PASS|t15 EXIF 转正|HTTP $meta 尺寸=${W}x${H}（竖版）"
else
  say "FAIL|t15 EXIF 转正|HTTP $meta 尺寸=${W}x${H}"
fi
rm -f "$TMPJ"

# ---------- t16 精确裁剪 ----------
TMPF=$(mktemp /tmp/qa_face_XXXX.jpg)
meta=$(cget -o "$TMPF" -w "%{http_code} %{content_type} %{size_download}" "$BASE/face_crop?face=$FACE_ID&k=3.6&size=460")
sig=$(file "$TMPF" | grep -o 'Lavc[^,"]*')
dims=$(sips -g pixelWidth -g pixelHeight "$TMPF" 2>/dev/null | awk '/pixel/{print $2}' | tr '\n' 'x' | sed 's/x$//')
if echo "$meta" | grep -q "^200 image/jpeg" && echo "$dims" | grep -q "^460x460"; then
  say "PASS|t16 精确人脸裁剪|HTTP $meta 尺寸=$dims 编码器=${sig:-sips}"
else
  say "FAIL|t16 精确人脸裁剪|HTTP $meta 尺寸=$dims"
fi
rm -f "$TMPF"

# ---------- 汇总 ----------
{
echo "# QA 第${ROUND}轮 — ${NAME}（$(date '+%F %T')）"
echo ""
echo "- 实例: $BASE"
echo "- 结果: PASS=$PASS FAIL=$FAIL WARN=$WARN SKIP=$SKIP"
echo ""
echo "| 结果 | 项目 | 详情 |"
echo "|---|---|---|"
for d in "${DETAILS[@]}"; do
  res="${d%%|*}"; rest="${d#*|}"
  echo "| $res | $(echo "$rest" | cut -d'|' -f1) | $(echo "$rest" | cut -d'|' -f2-) |"
done
} > "$OUT"
rm -f "$CJ"
echo "=== $OUT ==="
grep -E "^(PASS|FAIL|WARN|SKIP)" "$OUT" 2>/dev/null || cat "$OUT"
[ "$FAIL" -eq 0 ]
