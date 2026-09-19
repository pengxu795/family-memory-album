# 在线更新源（Update Feed）

本目录是应用内「设置 → 软件更新」的远程更新源（feed）。

## 格式

`manifest.json`（服务端每 8 秒超时拉取一次，字段必填 `version` / `file` / `md5`）：

```json
{
  "version": "1.0.0",
  "file": "family_memory_update_v1.0.0.zip",
  "md5": "<zip 整体 md5>",
  "size": 1393743,
  "notes": "版本说明",
  "url": "https://... （可选；缺省则为 feed 地址 + file 名）"
}
```

更新包 zip 内包含 `manifest.json`（`{version, notes, files:[{path, md5}]}`）与
`server.py`、`static/**`。客户端校验链：zip 整体 md5 → 包内 version 与声明一致 →
逐文件 md5 → 只允许覆盖 `server.py` 与 `static/` 下文件 → 备份后两阶段替换 → 自动重启。

## 配置方法

应用内「设置 → 软件更新」填入 feed 地址：

```
https://raw.githubusercontent.com/pengxu795/family-memory-album/main/update
```

发布新版本时：改动 `server.py` 的 `APP_VERSION` → 用 `make_update.py` 打包 →
把 zip 挂到 GitHub Release 资产 → 更新本目录 `manifest.json`。
