#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""幂等建库 + 版本化迁移（施工图 #3）。

解决试装报告崩点 2「新用户首次启动 no such table 主线程崩溃」：
- ensure_schema(db)：空库自动建齐 schema_raw.json 里的 71 张表 + 15 索引（幂等，
  旧库缺表时补建），然后按序执行 migrations/ 里未应用的迁移（鉴权表等增量）。
- 重复执行无副作用（CREATE IF NOT EXISTS + schema_migrations 版本记录）。
- 本模块不读取任何硬编码路径，db 路径由调用方传入（配合 #6 配置外置）。
"""
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA_RAW = ROOT / "schema_raw.json"
MIGRATIONS_DIR = ROOT / "migrations"

_TABLE_RE = re.compile(
    r'CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["\'\[]?(\w+)', re.I
)
_INDEX_RE = re.compile(
    r'CREATE\s+(?:UNIQUE\s+)?INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?["\'\[]?(\w+)', re.I
)


def _object_name(stmt):
    """提取 CREATE TABLE/INDEX 语句的对象名。"""
    m = _TABLE_RE.match(stmt) or _INDEX_RE.match(stmt)
    return m.group(1) if m else None


def _load_core_statements():
    """从 schema_raw.json 读核心建表/索引语句（生成源勿手改，重跑 dump_schema.py 更新）。"""
    payload = json.loads(SCHEMA_RAW.read_text(encoding="utf-8"))
    stmts = [t["sql"] for t in payload["tables"]] + [i["sql"] for i in payload["indexes"]]
    return stmts


def _ensure_migrations_table(con):
    con.execute(
        "CREATE TABLE IF NOT EXISTS schema_migrations ("
        " version TEXT PRIMARY KEY,"
        " applied_at TEXT NOT NULL)"
    )


def _pending_migrations(con):
    """migrations/ 下按文件名排序、尚未应用的 .sql 文件列表 [(version, sql), ...]。"""
    applied = {r[0] for r in con.execute("SELECT version FROM schema_migrations")}
    out = []
    if MIGRATIONS_DIR.is_dir():
        for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if f.stem not in applied:
                out.append((f.stem, f.read_text(encoding="utf-8")))
    return out


def ensure_schema(db_path):
    """确保 db_path 的库具备完整 schema（核心 71 表幂等 + migrations 增量）。
    返回 (created_tables_count, applied_migrations)；库文件不存在时自动创建。"""
    con = sqlite3.connect(str(db_path))
    try:
        before = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        # 「存在即跳过」而非 IF NOT EXISTS：dump 含 FTS5 虚拟表（如 memory_search），
        # 建虚拟表时 SQLite 自动创建影子表，影子表的独立 dump 语句必须跳过。
        have = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table','index')"
            )
        }
        for stmt in _load_core_statements():
            name = _object_name(stmt)
            if name and name not in have:
                try:
                    con.execute(stmt)
                except sqlite3.OperationalError as e:
                    # FTS5 虚拟表（memory_search）建成时 SQLite 自动创建影子表
                    # （_config/_content/_data/_docsize/_idx），其独立 dump 语句必然
                    # 撞 already exists —— 幂等语义下视为成功跳过。
                    if "already exists" not in str(e):
                        raise
                have.add(name)
        _ensure_migrations_table(con)
        applied = []
        for version, sql in _pending_migrations(con):
            # 逐语句执行 + 容忍「列已存在」：手动 sqlite3 < file 执行过迁移但忘登记
            # schema_migrations 时（2026-09-09 004 就是），重放不再炸启动。
            # 迁移文件均为简单 DDL（无触发器/过程体），先剥注释行再按分号切分安全。
            # （2026-09-10 修复：旧逻辑按分号切完再跳过以 -- 开头的段，导致
            #   「注释块与 CREATE TABLE 同段无分号」的迁移（003）整段被跳过，
            #   空库自动建库时 person_alias 表没建出来，下一条 INDEX 直接炸。）
            stripped = "\n".join(
                l for l in sql.splitlines() if not l.strip().startswith("--"))
            for stmt in [s.strip() for s in stripped.split(";") if s.strip()]:
                try:
                    con.execute(stmt)
                except sqlite3.OperationalError as e:
                    if "duplicate column name" not in str(e) \
                            and "already exists" not in str(e):
                        raise
            con.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, datetime.now(timezone.utc).isoformat(timespec="seconds")),
            )
            applied.append(version)
        con.commit()
        after = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        return len(after - before), applied
    finally:
        con.close()


def missing_core_tables(db_path):
    """诊断用：返回 db_path 缺失的核心表名列表（全齐则空列表）。"""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        have = {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        core = {t["name"] for t in json.loads(SCHEMA_RAW.read_text(encoding="utf-8"))["tables"]}
        return sorted(core - have)
    finally:
        con.close()


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else ":memory:"
    created, migrations = ensure_schema(target)
    print(f"OK: 新建 {created} 张表, 应用迁移 {migrations or '无'} → {target}")
