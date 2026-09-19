#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""schema.py 验收固化（施工图 #3/#4）：空库建齐、幂等、旧库补缺、篡改 fail-fast、生产库兼容。

生产库兼容用例依赖 backups/ 下的 DB 备份，缺失时自动跳过（干净环境跑其余 4 项）。
"""
import os
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import schema  # noqa: E402

MVP_DIR = Path(__file__).resolve().parent
BACKUP_DB = MVP_DIR.parent / "backups" / "family_memory_20260908_1320.db"


def _tables(db):
    con = sqlite3.connect(db)
    try:
        return {
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
    finally:
        con.close()


def test_fresh_db_full_schema():
    """空库首跑：建齐核心表 + user/session + migration 001 应用。"""
    db = os.path.join(tempfile.mkdtemp(), "fresh.db")
    created, mig = schema.ensure_schema(db)
    tables = _tables(db)
    assert created >= 71
    assert {"user", "session", "media_asset", "media_file", "person", "face"} <= tables
    assert "001_auth_tables" in mig and "002_single_admin" in mig


def test_idempotent_rerun():
    """幂等：重复执行零新建、零迁移。"""
    db = os.path.join(tempfile.mkdtemp(), "fresh.db")
    schema.ensure_schema(db)
    t1 = _tables(db)
    created, mig = schema.ensure_schema(db)
    assert created == 0 and mig == []
    assert _tables(db) == t1


def test_partial_legacy_db_repair():
    """旧库补缺：完整库 DROP 数张表后自动补齐 + 迁移补应用（真实迁移中断场景）。"""
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "partial.db")
    shutil.copy(BACKUP_DB, db)
    con = sqlite3.connect(db)
    for t in ("user", "session", "privacy_pin_v0", "geo_manual_v0", "recycle_v0", "schema_migrations"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    con.commit()
    con.close()
    created, mig = schema.ensure_schema(db)
    assert created >= 3 and "001_auth_tables" in mig and "002_single_admin" in mig
    assert schema.missing_core_tables(db) == []


def test_corrupt_table_fail_fast():
    """结构被篡改的表 → fail-fast 报错，绝不静默生成半残库。"""
    db = os.path.join(tempfile.mkdtemp(), "corrupt.db")
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE media_asset (asset_id TEXT PRIMARY KEY)")
    con.commit()
    con.close()
    with pytest.raises(sqlite3.OperationalError):
        schema.ensure_schema(db)


@pytest.mark.skipif(not BACKUP_DB.exists(), reason="生产库备份不存在（干净环境）")
def test_prod_copy_zero_damage():
    """生产库副本：只补鉴权表，存量资产零变动。"""
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "prod_copy.db")
    shutil.copy(BACKUP_DB, db)
    before = sqlite3.connect(db).execute("SELECT COUNT(*) FROM media_asset").fetchone()[0]
    created, _ = schema.ensure_schema(db)
    after = sqlite3.connect(db).execute("SELECT COUNT(*) FROM media_asset").fetchone()[0]
    assert before == after
    assert created <= 3  # user / session / schema_migrations
