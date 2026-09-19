#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Beta 0.5 UI 改版 — 页面暖化批量替换器
用法: python3 scripts/ui_retheme.py static/xxx.html [更多.html ...]
- 大小写不敏感替换旧色值为暖色令牌系
- 在 </head> 前插入 theme.css 引用（幂等）
"""
import re, sys

# 旧值 → 新值（暖色令牌系，见 static/theme.css）
MAP = {
    # 底色/分区（灰白 → 暖米白）
    "#f4f4f2": "#f6f1e8", "#f6f6f4": "#f6f1e8", "#f0f0ec": "#f0eadf",
    "#efefeb": "#f0eadf", "#fafaf8": "#fbf7f0", "#f2f2f2": "#f4efe6",
    "#fbf3ec": "#fbf5ec", "#f3f2fb": "#f6f1e8",
    # 线/分隔（灰 → 暖沙）
    "#e7e7e3": "#e8ddcb", "#e8e8e4": "#e8ddcb", "#e5e5e0": "#e8ddcb",
    "#e0e0dc": "#e8ddcb", "#ececec": "#ecdfcd", "#e2e2de": "#ecdfcd",
    "#dfdfda": "#e8ddcb", "#deded8": "#e8ddcb", "#ddd": "#e3d8c5",
    "#eee": "#f0e9dc", "#d9dce1": "#e8ddcb", "#e6e6e2": "#e8ddcb",
    # accent（刺眼橙红/朱红 → 暖棕）
    "#c65f3c": "#9a6b45", "#b3341f": "#9a6b45", "#b3402f": "#a8563f",
    "#d85a30": "#a86440", "#c0392b": "#a8563f", "#a53f2b": "#9a6b45",
    "#d33": "#b5543f",
    # 文字（冷黑灰 → 暖深棕）
    "#1b1b1a": "#3d3226", "#1c1c1c": "#3d3226", "#141414": "#322a1f",
    "#222": "#3d3226", "#111": "#322a1f",
    # 次级文字（冷灰 → 暖灰棕）
    "#6f6f6a": "#8a7a63", "#7a7a75": "#9c8b74", "#b0b0ac": "#b3a48d",
    "#888": "#9c8b74", "#666": "#8a7a63", "#999": "#a5967f",
    "#777": "#9c8b74", "#555": "#7c6b53", "#444": "#6b5b44",
    "#aaa": "#c3b49c", "#bbb": "#c9bba3",
    # 纯白卡片底 → 奶白（仅 background 上下文单独处理）
}

# rgba 同步替换
RGBA_MAP = {
    "198,95,60": "154,107,69",     # c65f3c
    "179,52,31": "154,107,69",     # b3341f
    "27,27,26": "61,50,38",        # 1b1b1a
    "74,125,255": "154,107,69",    # 4a7dff login 蓝
}

BG_WHITE = re.compile(r"background\s*:\s*#fff\b", re.I)
BG_WHITE2 = re.compile(r"background:\s*#ffffff", re.I)

def process(path):
    s = open(path, encoding="utf-8").read()
    orig = s
    low_cases = []
    def repl(m):
        v = m.group(0).lower()
        return MAP.get(v, v)
    # 6 位与 3 位 hex（词边界由正则组保证不吞关键字）
    s = re.sub(r"#[0-9a-fA-F]{6}\b", repl, s)
    s = re.sub(r"#[0-9a-fA-F]{3}\b", repl, s)
    for old, new in RGBA_MAP.items():
        s = s.replace("rgba(%s" % old, "rgba(%s" % new)
    # 白底卡片微暖（不影响 color:#fff 白字）
    s = BG_WHITE.sub("background:#fffdf8", s)
    s = BG_WHITE2.sub("background:#fffdf8", s)
    # 插入 theme.css（</head> 前，幂等）
    if "theme.css" not in s:
        s = s.replace("</head>", '<link rel="stylesheet" href="/theme.css?v=05">\n</head>', 1)
    if s != orig:
        open(path, "w", encoding="utf-8").write(s)
        n = sum(1 for a, b in zip(orig.split("\n"), s.split("\n")) if a != b)
        print(f"  {path}: {n} 行改动")
    else:
        print(f"  {path}: 无改动")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        process(p)
