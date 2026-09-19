-- #16 隐私清洗：人名词典表——家庭人名/别名/称呼全部数据化，出厂代码零人名。
-- kind='name'：ASR/拼音/口语变体 → 标准姓名（canonical=person.display_name）
-- kind='role'：称呼 → 姓名（补充；常规称呼优先从 person.relationship_label 动态派生）
-- 数据不入库迁移（隐私）：由管理员按家庭实际自行添加，或从旧版一次性导入。
CREATE TABLE IF NOT EXISTS person_alias (
    alias      TEXT PRIMARY KEY,
    canonical  TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'name',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_person_alias_canonical ON person_alias(canonical);
