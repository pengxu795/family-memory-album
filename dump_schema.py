#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从真实库 dump 完整 schema → schema_raw.json（施工图 #2）。

用途：真实库是权威 schema。本脚本只读连接，输出表/索引的 CREATE 语句
到 schema_raw.json，作为 schema.py 幂等建库的生成源。

⚠️ 基线锁定（2026-09-08 工作约定）：schema_raw.json 是唯一初始基线，已冻结。
之后所有表结构变更（加表/加列/改表）一律走 migrations/002+，禁止重跑本脚本
覆盖基线——重 dump 会让"权威 schema"漂移，用户已部署库的结构就对不上了。
本脚本仅用于最初生成；如确需重新对账，输出到别处手动比对。
"""
import json
import sqlite3
import sys
from pathlib import Path
import os

DB = Path(os.environ.get("FF_DB_PATH") or "data/family_memory.db")
OUT = Path(__file__).resolve().parent / "schema_raw.json"

# 只 dump 永久对象，排除 SQLite 内部表
SKIP_PREFIXES = ("sqlite_",)


def main():
    if OUT.exists():
        print("基线已冻结：schema_raw.json 已存在，禁止覆盖（见模块 docstring 的基线锁定约定）。", file=sys.stderr)
        print("如需对账，请临时修改 OUT 指向新文件，用 diff 比对，勿覆盖基线。", file=sys.stderr)
        return 1
    if not DB.exists():
        print(f"真实库不存在: {DB}", file=sys.stderr)
        return 1
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)  # 只读，服务运行中安全
    try:
        rows = con.execute(
            "SELECT type, name, sql FROM sqlite_master "
            "WHERE sql IS NOT NULL AND name NOT LIKE 'sqlite_%' "
            "ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name"
        ).fetchall()
    finally:
        con.close()

    tables, indexes = [], []
    for typ, name, sql in rows:
        entry = {"name": name, "sql": sql.strip().rstrip(";") + ";"}
        (tables if typ == "table" else indexes).append(entry)

    payload = {
        "source_db": str(DB),
        "dumped_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "table_count": len(tables),
        "index_count": len(indexes),
        "tables": tables,
        "indexes": indexes,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {len(tables)} 表, {len(indexes)} 索引 → {OUT}")
    print("表清单:")
    for t in tables:
        print(f"  {t['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
