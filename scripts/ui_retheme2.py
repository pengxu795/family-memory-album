#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全站换装第二期：暖棕家庭风 → 参考图六色现代极简风。
严格映射：电蓝 #2f31f5 / 黄绿 #dbf059 / 中灰 #797979 / 铂灰 #ebebe9 / 纯黑 #131314 / 纯白 #ffffff。
功能性色（红/绿/琥珀）统一为现代标准值，不再一家一个色。
同时：SVG 图标笔画统一 1.6（现代极简线宽）、蓝→黄绿渐变降噪为纯电蓝、缓存版本 v=05→v=06。
幂等：跑多遍结果一致。"""
import re, glob, os

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")

HEX_MAP = {
    # 核心六色映射（暖棕 → 新色板）
    "#f6f1e8": "#ffffff",  # 暖米白底 → 纯白
    "#fffdf8": "#ffffff",  # 奶白卡片 → 纯白
    "#e8ddcb": "#ebebe9",  # 暖沙描边 → 铂灰
    "#9a6b45": "#2f31f5",  # 暖棕主色 → 电蓝
    "#c98d5f": "#dbf059",  # 柔橙辅助 → 黄绿
    "#3d3226": "#131314",  # 暖深棕文字 → 纯黑
    "#9c8b74": "#797979",  # 暖灰棕次级 → 中灰
    # 扩展暖色 → 中性灰阶
    "#8a7a63": "#6b6b6b",
    "#7c6b53": "#6b6b6b",
    "#6b5b44": "#6b6b6b",
    "#b3a48d": "#a3a3a0",
    "#a5967f": "#a3a3a0",
    "#c9bba3": "#d5d5d2",
    "#322a1f": "#131314",  # 深棕 chrome（theme-color 等）
    "#ecdfcd": "#ebebe9",
    "#f0e9dc": "#f0f0ee",
    "#e3d8c5": "#e3e3e0",
    "#1d1812": "#131314",
    "#33291c": "#131314",
    "#1b1b1a": "#131314",
    "#26241f": "#131314",
    # 特殊页遗留紫 → 电蓝
    "#5b4a9e": "#2f31f5",
    "#7b1fa2": "#2f31f5",
    # 暖红系功能色 → 统一现代红
    "#b5543f": "#e5484d",
    "#c55b55": "#e5484d",
    "#c62828": "#e5484d",
    "#d4471f": "#e5484d",
    "#a8563f": "#e5484d",
    "#c65f3c": "#e5484d",
    "#fef6f3": "#fdeeee",
    "#f3ece8": "#f4f4f2",
    # 蓝灰徽章 → 纯黑
    "#4a5568": "#131314",
    # 旧绿 → 现代绿
    "#2e7d32": "#30a46c",
    "#7a9a6d": "#30a46c",
    # ---- 二期长尾（暖色残留中和；地图场景分类色/功能琥珀保留不动）----
    "#7a5a3a": "#6b6b6b",   # index 帮助文字
    "#fbf5ec": "#f4f4f2",   # index 帮助面板底
    "#a86440": "#e5484d",   # wizard --err
    "#d9553f": "#e5484d",   # library 来源掉线红点
    "#a84c2e": "#d13438",   # library 恢复按钮 hover（深一档红）
    "#7a4a35": "#6b6b6b",   # models 列表文字
    "#c3b49c": "#b3b3b0",   # person 禁用态文字
    "#a08d74": "#8a8a87",   # tv 暖灰
    "#c69b3b": "#ffb300",   # ab-review unsure 边（并入标准琥珀）
    "#f0e8cc": "#fff6e0",   # ab-review unsure 底（琥珀浅底）
    "#f0eadf": "#f4f4f2",
    "#fdf1dc": "#fff6e0",
    "#fbe9e2": "#fdeeee",
    "#eec3b2": "#f4f4f2",
    "#e5c9c3": "#efefed",
    "#f0dcd3": "#f4f4f2",
    "#f6ede8": "#f4f4f2",
    "#faece6": "#fdeeee",
    "#f1d9d7": "#fdeeee",
    "#e6b0aa": "#fdeeee",
    "#f4efe6": "#f4f4f2",
    "#c8541c": "#e5484d",
}

RGBA_MAP = {
    "154,107,69": "47,49,245",   # accent
    "201,141,95": "219,240,89",  # accent-2
    "93,74,54":   "19,19,20",    # 暖影 → 黑影
    "61,50,38":   "19,19,20",    # ink
    "50,42,31":   "19,19,20",    # 深棕
    "29,24,18":   "19,19,20",
}

def remap_css_text(txt: str) -> str:
    for old, new in HEX_MAP.items():
        txt = re.sub(re.escape(old), new, txt, flags=re.IGNORECASE)
    for old, new in RGBA_MAP.items():
        txt = re.sub(r"rgba\(" + re.escape(old) + r"\b", "rgba(" + new, txt)
    # 蓝→黄绿渐变降噪为纯电蓝（按钮/横幅等高亮场景，黄绿配白字不可读）
    txt = re.sub(r"linear-gradient\([^)]*#2f31f5[^)]*#dbf059[^)]*\)", "#2f31f5", txt, flags=re.IGNORECASE)
    txt = re.sub(r"linear-gradient\([^)]*#dbf059[^)]*#2f31f5[^)]*\)", "#2f31f5", txt, flags=re.IGNORECASE)
    # SVG 图标笔画统一 1.6（只动 <svg> 开标签上的属性）
    txt = re.sub(r'(<svg[^>]*?stroke-width=")[0-9.]+(")', r"\g<1>1.6\g<2>", txt)
    # 缓存版本号
    txt = txt.replace("theme.css?v=05", "theme.css?v=06")
    return txt

def main():
    files = sorted(glob.glob(os.path.join(HERE, "*.html"))) + [os.path.join(HERE, "theme.css")]
    for f in files:
        with open(f, encoding="utf-8") as fh:
            src = fh.read()
        out = remap_css_text(src)
        if out != src:
            with open(f, "w", encoding="utf-8") as fh:
                fh.write(out)
            print(f"updated  {os.path.basename(f)}")
        else:
            print(f"no-change {os.path.basename(f)}")

if __name__ == "__main__":
    main()
