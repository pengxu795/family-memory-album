-- Migration 001: 鉴权表（施工图 #3，为 #9 两级鉴权前置数据模型）
-- 最小化设计：user + session 两表；访客白名单用相册粒度标记（复用现有相册概念），不做权限矩阵。
-- role: admin = 管理员(配置/全库), family = 家人(全库浏览问答), guest = 访客(仅白名单相册)

CREATE TABLE IF NOT EXISTS user (
    user_id       TEXT PRIMARY KEY,
    username      TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,              -- PBKDF2-SHA256, 格式: iterations$salt$hash
    role          TEXT NOT NULL DEFAULT 'family' CHECK (role IN ('admin','family','guest')),
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS session (
    token      TEXT PRIMARY KEY,              -- 随机 token, Cookie 携带(电视端浏览器同样适用)
    user_id    TEXT NOT NULL REFERENCES user(user_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_user ON session(user_id);
CREATE INDEX IF NOT EXISTS idx_session_expires ON session(expires_at);
