#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""路径前缀迁移工具（施工图 #6，为 #15 旧库迁移演练准备）。

场景：旧部署的 DB 里 absolute_path/root_path 存的是旧机器路径前缀
（如 Mac 挂载点 /Volumes/homes），换到 NAS/容器后照片挂载点变了
（如 /photos）。本工具把 DB 中所有媒体路径做"旧前缀 → 新前缀"批量改写。

安全设计：
- 默认 dry-run（只打印将改动的行数和示例），加 --apply 才真正写库
- 写库前自动做一份 SQLite 在线备份到 <db>.pre-migrate-<时间戳>.db
- 事务执行，失败整体回滚
- 幂等：前缀已替换过的路径不会再被匹配（以旧前缀开头才改）

用法：
  python3 migrate_paths.py <db路径> <旧前缀> <新前缀> [--apply]
示例：
  python3 migrate_paths.py data/family_memory.db /Volumes/homes /photos --apply
"""
import os
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(argv):
    args = [a for a in argv if a != "--apply"]
    apply_mode = "--apply" in argv
    if len(args) != 3:
        print(__doc__)
        return 1
    db, old_prefix, new_prefix = args
    if not Path(db).is_file():
        print(f"DB 不存在: {db}", file=sys.stderr)
        return 1
    if old_prefix == new_prefix or not old_prefix:
        print("旧/新前缀无效", file=sys.stderr)
        return 1
    if not apply_mode:
        print("== DRY-RUN 模式（加 --apply 才真正写库）==")

    if apply_mode:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup = f"{db}.pre-migrate-{stamp}.db"
        src = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        dst = sqlite3.connect(backup)
        src.backup(dst)
        dst.close()
        src.close()
        print(f"已备份: {backup}")

    con = sqlite3.connect(db, timeout=30)
    con.execute("PRAGMA busy_timeout=30000")
    try:
        total = 0
        for table, col in (("media_file", "absolute_path"), ("source", "root_path")):
            rows = con.execute(
                f"SELECT rowid, {col} FROM {table} WHERE {col} LIKE ?", (old_prefix + "%",)
            ).fetchall()
            total += len(rows)
            samples = [r[1] for r in rows[:3]]
            print(f"{table}.{col}: {len(rows)} 行待改写")
            for s in samples:
                print(f"   例: {s}  →  {new_prefix}{s[len(old_prefix):]}")
            if apply_mode and rows:
                con.executemany(
                    f"UPDATE {table} SET {col} = ? || substr({col}, ?) "
                    f"WHERE rowid = ? AND {col} LIKE ?",
                    [(new_prefix, len(old_prefix) + 1, rid, old_prefix + "%")
                     for rid, _ in rows],
                )
        if apply_mode:
            con.commit()
            print(f"完成：共改写 {total} 行")
        else:
            con.rollback()
            print(f"共 {total} 行将被改写（未写入）")
        return 0
    except Exception as exc:
        con.rollback()
        print(f"失败已回滚: {exc}", file=sys.stderr)
        return 1
    finally:
        con.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
