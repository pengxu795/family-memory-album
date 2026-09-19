-- Migration 002: 管理员唯一性（施工图 #5 初始化向导互斥的 DB 层保障）
-- 部分唯一索引：role='admin' 全库只能有一行。网页层 INSERT 失败即回"已有管理员"，
-- 双浏览器竞态下也只有一个成功——互斥不依赖网页判断。

CREATE UNIQUE INDEX IF NOT EXISTS idx_user_single_admin ON user(role) WHERE role = 'admin';
