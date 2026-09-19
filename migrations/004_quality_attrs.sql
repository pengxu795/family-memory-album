-- 004_quality_attrs.sql —— 新功能①-⑥ A4 联调迁移（设计定稿，执行时机：A4 全量重算完成后）
-- 原则：只加列/加表，不动现有数据；全部可回滚（DROP 即回滚）。

-- ① 画质属性表（模糊 + 美学，一张表管一个资产）
--    media_asset.quality_score 列已存在，保留为「综合展示分」；
--    原始分与模型版本进新表，避免魔改核心表。
CREATE TABLE IF NOT EXISTS asset_quality_v0 (
    asset_id      TEXT PRIMARY KEY REFERENCES media_asset(asset_id),
    blur_label    TEXT,             -- sharp / soft / blurry（blur_detector）
    sharp_score   REAL,             -- 分块 top-25% Laplacian 方差
    aesthetic     REAL,             -- LAION Aesthetic V2 原始连续分
    model_version TEXT NOT NULL,    -- 如 'quality-v1-20260909'（重跑换代用）
    created_at    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_aq_blur ON asset_quality_v0(blur_label);

-- ② 智能裁切：crop_v0 已存在（手动裁切），加 method 区分人手/自动 + 保留参数版本
ALTER TABLE crop_v0 ADD COLUMN method TEXT DEFAULT 'manual';
--     自动裁切写入 method IN (face/saliency/horizon/center)；
--     前端渲染顺序：手动裁切 > 自动裁切参数；表内同 asset 只保留一行（现有约定）。

-- ③ 性别年龄 + 表情：face_instance_v0 加列（genderage ⑥ / face_expr ⑪ 回写）
ALTER TABLE face_instance_v0 ADD COLUMN gender VARCHAR(8);      -- male / female
ALTER TABLE face_instance_v0 ADD COLUMN age_est INTEGER;        -- 预测年龄（岁）
ALTER TABLE face_instance_v0 ADD COLUMN age_label VARCHAR(8);   -- child/teen/adult/elder
--     模型版本记录在 analysis_run（pipeline='genderage-v1'），不重复建列。
ALTER TABLE face_instance_v0 ADD COLUMN eyes_open INTEGER;      -- 1/0（2d106det 眼开合比阈值判定）
ALTER TABLE face_instance_v0 ADD COLUMN eye_open_score REAL;    -- 眼开合比（两眼均值）
ALTER TABLE face_instance_v0 ADD COLUMN smile_ratio REAL;       -- 嘴宽/瞳距
ALTER TABLE face_instance_v0 ADD COLUMN is_smiling INTEGER;     -- 1/0

-- ③b 照片内文字 OCR（⑩）：可搜索文本
CREATE TABLE IF NOT EXISTS asset_ocr_v0 (
    asset_id      TEXT PRIMARY KEY REFERENCES media_asset(asset_id),
    n_regions     INTEGER,          -- 识别出的文字块数
    mean_conf     REAL,             -- 平均置信度
    text          TEXT,             -- 拼接文本（≤600 字）
    model_version TEXT NOT NULL,    -- 'ocr-v1-20260909'
    created_at    TEXT DEFAULT (datetime('now'))
);
--    检索接入（A4 联调）：memory_search 建索引时 JOIN 本表，或此处直接建 FTS：
CREATE VIRTUAL TABLE IF NOT EXISTS asset_ocr_fts USING fts5(
    asset_id UNINDEXED, text, tokenize='unicode61'
);

-- ④ 相似去重：asset_similar_group_v0 加 engine 标记（新旧管线共存，便于对比回滚）
ALTER TABLE asset_similar_group_v0 ADD COLUMN engine TEXT DEFAULT 'phash-v1';
--     双引擎新组写 engine='dual-v1'（phash + siglip2 同源窗口）；
--     旧 493 组保留 engine='phash-v1'，UI 先按 engine 过滤展示，人工确认后退役旧组。

-- ⑤ 截图/收款码：不加表。复用 asset_filter_v0，新增 filter_reason 枚举：
--     JUNK_QR_PAYMENT_WECHAT   微信收款码（wxp://）
--     JUNK_QR_PAYMENT_ALIPAY   支付宝收款码
--     JUNK_SCREENSHOT_VISUAL   视觉截图（纵横比+纯边+状态栏，文件名法漏网的）
--     JUNK_QR_UNDECODED        二维码解不出内容 → 不隐藏，只打标待精扫
--     evidence_kind='qr' / 'visual'，evidence_value=解码内容或分块特征摘要，
--     rule_version='junk-v1'，confidence 0~1。
