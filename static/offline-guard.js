/* 家庭相册 · 掉线兜底提示（引入到主要页面 <body> 末尾）
 * 定期探测服务可达性；掉线时顶部提示条，恢复自动消失。
 * NAS 场景：掉线常因 NAS 休眠/断网/Tailscale 未连——给出对应排查提示。 */
(function () {
  "use strict";
  var OFFLINE_ID = "ff-offline-bar";
  var CHECK_EVERY = 20000;      // 掉线后重试间隔
  var PING_EVERY  = 120000;     // 在线时的例行探测间隔
  var bar = null, timer = null, consecutiveFail = 0;

  function ensureBar() {
    if (bar) return bar;
    bar = document.createElement("div");
    bar.id = OFFLINE_ID;
    bar.style.cssText = "position:fixed;top:0;left:0;right:0;z-index:99999;" +
      "background:#D85A30;color:#fff;font-size:13px;line-height:1.5;" +
      "padding:8px 14px;text-align:center;font-family:-apple-system,'PingFang SC',sans-serif;" +
      "box-shadow:0 2px 8px rgba(0,0,0,.25)";
    bar.innerHTML = "<b>⚠️ 无法连接相册服务</b> — NAS 可能离线或正在重启，正在自动重试…" +
      " <span style=\"opacity:.85\">（排查：NAS 是否开机/休眠 · 同一局域网 · Tailscale 是否已连接）</span>";
    document.body.appendChild(bar);
    return bar;
  }

  function removeBar() {
    if (bar && bar.parentNode) bar.parentNode.removeChild(bar);
    bar = null;
  }

  function ping() {
    // 用 GET 探活：部分部署形态里的轻量 HTTP 服务不实现 HEAD（回 501），
    // 会被误判为掉线；GET /login.html 是静态资源，无需登录态，任何 <500 响应都算活着
    var t0 = Date.now();
    return fetch("/login.html", { method: "GET", cache: "no-store" })
      .then(function (r) { return r.status < 500; })
      .catch(function () { return false; })
      .then(function (ok) {
        if (ok) {
          consecutiveFail = 0;
          removeBar();
          schedule(PING_EVERY);
        } else {
          consecutiveFail += 1;
          if (consecutiveFail >= 2) ensureBar();   // 连续 2 次失败才报，避免瞬时抖动
          schedule(CHECK_EVERY);
        }
      });
  }

  function schedule(ms) {
    if (timer) clearTimeout(timer);
    timer = setTimeout(ping, ms);
  }

  function start() { schedule(3000); }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else { start(); }
})();
