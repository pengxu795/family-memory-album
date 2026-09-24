-- 005_log_color_v0.sql —— 大疆/影石 Log 原片自动色彩还原（2026-09-23）
-- 原则：只加表，不动现有数据；可回滚（DROP TABLE 即回滚）。
-- 原片文件永不修改：还原只发生在缩略图/预览层（get_thumb 生成时挂滤镜），
-- 查看器看原图时由前端 CSS filter 模拟，开关随时可关。

CREATE TABLE IF NOT EXISTS asset_log_color_v0 (
    asset_id     TEXT PRIMARY KEY REFERENCES media_asset(asset_id),
    is_log       INTEGER NOT NULL,          -- 1=Log/Flat 灰片（需要还原） 0=已检查、不是
    camera       TEXT,                      -- DJI / Insta360 / GoPro / unknown
    profile      TEXT,                      -- dlog / dlogm / flat / unknown
    confidence   REAL,                      -- 0~1
    source       TEXT NOT NULL,             -- auto_filename / auto_visual / user
    strength     REAL NOT NULL DEFAULT 1.0, -- 还原强度 0.5~1.5（喂给滤镜链插值）
    thumb_ver    INTEGER NOT NULL DEFAULT 1,-- 还原参数版本：参数变更 +1 → 旧缓存缩略图失效
    detected_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alc_log ON asset_log_color_v0(is_log);
