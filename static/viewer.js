/* ============================================================
 * viewer.js — 家庭相册统一查看器（2026-09-03 抽取）
 * 首页 index.html 与总库 library.html 共用同一实现，
 * 杜绝「改一处漏一处」。页面只需引入本文件并调用 WBViewer API。
 *
 * API：
 *   WBViewer.open(items, index, opts)  打开；items 为资产数组
 *   WBViewer.close()                   关闭（触发 opts.onClose）
 *   WBViewer.step(+1/-1)               上一张 / 下一张
 *   WBViewer.render()                  列表增删后重渲染当前项（索引自动收敛）
 *   WBViewer.current()                 -> {item,index,items} | null
 *   WBViewer.isOpen()                  -> bool
 *
 * items 元素字段：{id, type:'photo'|'video', time:'YYYY-MM-DD…'}
 * opts（均可选）：
 *   meta(item, i, n)      底部元信息文本；缺省 "i / n · 日期"
 *   filterBtn(item)       -> {text, onClick(item)} 显示动作按钮；返回 null 隐藏
 *   actionBtn(item)       -> {text, onClick(item), danger=true} 第二个动作按钮（红色，用于删除类操作）；返回 null 隐藏
 *   cropBtn(item)         -> {text, onClick(item)} 第三个动作按钮（用于裁切/二次构图）；返回 null 隐藏
 *   afterRender(item,i,n) 每次渲染完成后的回调（首页用它预取相邻原片）
 *   onClose()             关闭回调（页面清理自身状态，如 currentGroupId）
 *
 * 裁切/二次构图（2026-09-06 收编进本文件，首页/总库共用一份）：
 *   WBViewer.loadCropMap()           启动时拉取裁切框映射 -> Promise<cropMap>
 *   WBViewer.cropAt(id)              查某资产的裁切框 [x,y,w,h] | undefined
 *   WBViewer.openCropEditor(id,type) 打开裁切编辑器（type='video' 会拒绝）
 *   WBViewer.setCropChangeHandler(fn) 裁切保存/恢复后回调 fn(id, rect|null)，
 *                                    页面在此同步自己的卡片/列表视图
 *
 * item.crop = [x,y,w,h]（归一化 0..1）：非破坏性裁切框，存在时查看器
 * 只显示裁切区域（.vcrop 容器按裁切比例适配视口），原图文件不受影响。
 *
 * 交互：滚轮缩放 / 双击 1x↔2.5x / 右上按钮组 / 放大后拖拽平移 /
 *       左右方向键翻页、ESC 关闭、点击两侧热区翻页
 * ============================================================ */
(function () {
  'use strict';

  var CSS = [
    '.viewer{position:fixed;inset:0;z-index:1000;display:none;background:#080808;overflow:hidden}',
    '.viewer.open{display:block}',
    '.viewer-media{position:absolute;inset:0;display:grid;place-items:center;overflow:hidden}',
    '.viewer-media img,.viewer-media video{display:block;width:100vw;height:100vh;max-width:none;max-height:none;object-fit:contain;background:#000;box-shadow:none}',
    '.viewer-back{position:absolute;left:18px;top:18px;z-index:1004;width:40px;height:40px;padding:0;display:flex;align-items:center;justify-content:center;border:1px solid #ffffff55;background:#111b;color:#fff;border-radius:50%;cursor:pointer}',
    '.viewer-back svg{width:19px;height:19px;stroke:#fff}',
    '.viewer-back:hover{background:#000e}',
    '.viewer-zoom{position:absolute;right:18px;top:18px;z-index:1004;display:flex;flex-direction:column;gap:9px}',
    '.viewer-zoom button{width:40px;height:40px;border-radius:50%;border:1px solid #ffffff55;background:#111b;color:#fff;display:flex;align-items:center;justify-content:center;cursor:pointer;padding:0}',
    '.viewer-zoom button:hover{background:#000e}',
    '.viewer-zoom button svg{width:17px;height:17px}',
    '.viewer-zone{position:absolute;top:64px;bottom:64px;z-index:1003;border:0;background:transparent;cursor:pointer}',
    '.viewer-prev{left:0;width:22%}',
    '.viewer-next{right:0;width:22%}',
    '.viewer-meta{position:absolute;left:0;right:0;bottom:0;z-index:1004;min-height:44px;padding:10px 18px;display:flex;align-items:center;justify-content:center;color:#aaa;background:linear-gradient(transparent,#000b);font-size:12px;pointer-events:none}',
    '.viewer-actions{position:absolute;left:50%;top:16px;transform:translateX(-50%);z-index:1005;display:flex;gap:8px}',
    '.viewer-filter{border:1px solid #ffffff55;background:#111b;color:#fff;border-radius:14px;padding:5px 10px;font-size:10px;cursor:pointer;position:static}',
    '.viewer-filter:disabled{opacity:.6;cursor:default}',
    '.viewer-filter.danger{border-color:#c65f3cbb;color:#ffb59d}',
    '.viewer-filter.danger:hover{background:#c65f3c;color:#fff}',
    '#viewerMedia img.pz{cursor:grab;touch-action:none;transition:transform .12s ease-out;user-select:none;-webkit-user-drag:none}',
    '#viewerMedia img.pz.dragging{cursor:grabbing;transition:none}',
    /* 2026-09-08 修「缩略图→原图过渡闪烁」：原图作为独立图层盖在缩略图上，解码完成后淡入，
     * 全程不置换任何已显示位图；.pz-hi 阶段 pointer-events:none，交互仍落在垫底缩略图上 */
    '#viewerMedia img.pz-hi{position:absolute;inset:0;opacity:0;pointer-events:none;z-index:1;transition:opacity .28s ease, transform .12s ease-out}',
    '#viewerMedia img.pz-hi.ready{opacity:1}',
    /* 2026-09-10 兜底：.vcrop 默认占满视口。原 max-width/max-height 写法在
     * applyCrop 应用前容器塌缩成 0×0 并居中，absolute img 从视口中心向右下
     * 铺开，加载期间出现「右下四分之一错位画面」（切图必现）。默认全屏后，
     * 加载帧=整屏显示，applyCrop 到货后再收拢为裁切比例。 */
    '.vcrop{position:relative;width:100vw;height:100vh}',
    '.vcrop img.pz{position:absolute;object-fit:fill}',
    '.viewer.zoomed .viewer-zone{pointer-events:none}',
    /* ===== 裁切/二次构图编辑器（2026-09-06 从 library.html 收编进统一查看器）=====
     * 首页/总库共用这一份：改这里两边同时生效。非破坏性——只存裁切框，原图不动。 */
    '.crop-modal{position:fixed;inset:0;z-index:1500;background:#0a0a0ad9;backdrop-filter:blur(8px);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:14px}',
    '.crop-modal .crop-title{color:#fff;font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}',
    '.crop-modal .crop-title .sub{font-size:11px;color:#999;font-weight:400}',
    '.crop-stage{position:relative;max-width:90vw;line-height:0;touch-action:none}',
    '.crop-stage img{display:block;max-width:90vw;max-height:66vh;object-fit:contain}',
    '.crop-box{position:absolute;border:1.5px solid #fff;box-shadow:0 0 0 9999px #000a;cursor:move}',
    '.crop-hdl{position:absolute;width:14px;height:14px;background:#fff;border-radius:3px;border:1px solid #0008}',
    '.crop-hdl.tl{left:-8px;top:-8px;cursor:nwse-resize}',
    '.crop-hdl.tr{right:-8px;top:-8px;cursor:nesw-resize}',
    '.crop-hdl.bl{left:-8px;bottom:-8px;cursor:nesw-resize}',
    '.crop-hdl.br{right:-8px;bottom:-8px;cursor:nwse-resize}',
    '.crop-grid{position:absolute;inset:0;pointer-events:none;opacity:.5}',
    '.crop-grid::before,.crop-grid::after{content:"";position:absolute;background:#ffffff88}',
    '.crop-grid::before{left:33.3%;right:33.3%;top:0;bottom:0;width:1px}',
    '.crop-grid::after{top:33.3%;bottom:33.3%;left:0;right:0;height:1px}',
    '.crop-presets{display:flex;gap:6px;flex-wrap:wrap;justify-content:center}',
    '.crop-presets button{border:1px solid #ffffff44;background:#ffffff14;color:#ddd;border-radius:14px;padding:5px 12px;font-size:11px;cursor:pointer}',
    '.crop-presets button.on{background:#fff;color:#111;border-color:#fff}',
    '.crop-btns{display:flex;gap:10px;align-items:center}',
    '.crop-btns button{border:1px solid #ffffff55;background:#ffffff18;color:#fff;border-radius:16px;padding:8px 18px;font-size:12px;cursor:pointer}',
    '.crop-btns button.primary{background:#fff;color:#111;border-color:#fff;font-weight:600}',
    '.crop-btns button.danger{border-color:#c65f3c88;color:#ffb59d}',
    '.crop-btns button.danger:hover{background:#c65f3c;color:#fff}',
    '.crop-btns .hint{color:#888;font-size:11px;max-width:260px}',
    /* ===== 手动定位（2026-09-07 半自动地理闭环）：锚点 → 同日互证 → 视觉复核 ===== */
    '.geo-body{display:flex;flex-direction:column;gap:12px;align-items:center}',
    '.geo-body input,.geo-body select{min-width:300px;max-width:82vw;background:#1b1b1b;color:#fff;border:1px solid #ffffff44;border-radius:10px;padding:10px 12px;font-size:13px;outline:none}',
    '.geo-body input:focus,.geo-body select:focus{border-color:#fff}',
    '.geo-body input::placeholder{color:#666}',
    '.geo-body .geo-row2{display:flex;gap:10px;min-width:300px;max-width:82vw}',
    '.geo-body .geo-row2 select{min-width:0;flex:1.4}',
    '.geo-body .geo-row2 input{min-width:0;flex:1}',
    '.geo-body .geo-date{display:flex;align-items:center;gap:6px;flex:1;color:#999;font-size:11px}',
    '.geo-body .geo-date input{flex:1;min-width:0;color-scheme:dark}',
    '.geo-body .geo-cur{color:#8fd48f;font-size:11px}',
    '.geo-body .geo-cur.none{color:#777}',
    '.geo-flow{color:#aaa;font-size:11px;line-height:1.7;text-align:left;background:#ffffff0d;border-radius:10px;padding:10px 14px;max-width:420px}',
    '.geo-flow b{color:#fff}',
    /* 裁切操作结果提示（viewer 内统一 toast，页面无需自备 notice 栏） */
    '.crop-toast{position:fixed;left:50%;bottom:70px;transform:translateX(-50%);z-index:1600;background:#111d;color:#fff;border:1px solid #ffffff44;border-radius:16px;padding:8px 18px;font-size:12px;pointer-events:none}'
  ].join('');

  var HTML = ''
    + '<section id="viewer" class="viewer" aria-hidden="true">'
    + '<div id="viewerMedia" class="viewer-media"></div>'
    + '<button class="viewer-back" onclick="WBViewer.close()" title="返回" aria-label="返回"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 12H5M12 19l-7-7 7-7"/></svg></button>'
    + '<div class="viewer-zoom" id="vZoom">'
    + '<button onclick="WBViewer.zoomIn()" title="放大" aria-label="放大"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg></button>'
    + '<button onclick="WBViewer.zoomOut()" title="缩小" aria-label="缩小"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M5 12h14"/></svg></button>'
    + '<button onclick="WBViewer.resetZoom()" title="复位到 1:1" aria-label="复位"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 14 10 14 10 20"/><polyline points="20 10 14 10 14 4"/><line x1="14" y1="10" x2="21" y2="3"/><line x1="3" y1="21" x2="10" y2="14"/></svg></button>'
    + '<button onclick="WBViewer.rotate()" title="旋转 90°（仅本次查看）" aria-label="旋转"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/></svg></button>'
    + '</div>'
    + '<button class="viewer-zone viewer-prev" aria-label="上一张" onclick="WBViewer.step(-1)"></button>'
    + '<button class="viewer-zone viewer-next" aria-label="下一张" onclick="WBViewer.step(1)"></button>'
    + '<div id="viewerMeta" class="viewer-meta"></div>'
    + '<div class="viewer-actions"><button class="viewer-filter" id="vFilter">移出墙面</button><button class="viewer-filter" id="vGeo">📍 定位</button><button class="viewer-filter" id="vCrop" style="display:none"></button><button class="viewer-filter danger" id="vAction" style="display:none"></button></div>'
    + '</section>';

  // ---- 内部状态（单向数据流：open/step 改状态 -> render 读状态刷 DOM） ----
  var st = { items: [], index: -1, opts: null };
  var vz = { s: 1, x: 0, y: 0, r: 0 };
  var el = null;

  function ensure() {
    if (el) return;
    var style = document.createElement('style');
    style.textContent = CSS;
    document.head.appendChild(style);
    document.body.insertAdjacentHTML('beforeend', HTML);
    el = document.getElementById('viewer');
    document.getElementById('viewerMedia').addEventListener('wheel', function (e) {
      if (!document.querySelector('#viewerMedia img.pz')) return; // 视频交给原生控制条
      e.preventDefault();
      zoomBy(e.deltaY < 0 ? 1.18 : 1 / 1.18);
    }, { passive: false });
    document.addEventListener('keydown', function (e) {
      if (!isOpen()) return;
      if (document.querySelector('.crop-modal')) return; // 裁切/定位弹窗打开时，按键交给弹窗（ESC 只关弹窗）
      if (e.key === 'Escape') close();
      if (e.key === 'ArrowLeft') step(-1);
      if (e.key === 'ArrowRight') step(1);
    });
  }

  function isOpen() { return !!(el && el.classList.contains('open')); }

  function fmtDay(t) {
    return (t && !/^0000/.test(String(t))) ? String(t).slice(0, 10) : '时间待确认';
  }

  // ---- 缩放 / 拖动 ----
  function vzApply() {
    // 2026-09-08：双层淡入期间缩略图与原图图层并存，transform 要同步到所有 .pz，防止错位
    var imgs = document.querySelectorAll('#viewerMedia img.pz');
    for (var i = 0; i < imgs.length; i++) {
      imgs[i].style.transform = 'translate(' + vz.x + 'px,' + vz.y + 'px) scale(' + vz.s + ') rotate(' + (vz.r || 0) + 'deg)';
    }
    el.classList.toggle('zoomed', vz.s > 1.001);
  }
  function resetZoom() { vz = { s: 1, x: 0, y: 0, r: 0 }; vzApply(); }
  function rotate() { vz.r = ((vz.r || 0) + 90) % 360; vzApply(); }
  function zoomBy(f) {
    vz.s = Math.min(8, Math.max(1, vz.s * f));
    if (vz.s <= 1.0001) { vz.s = 1; vz.x = 0; vz.y = 0; }
    vzApply();
  }
  function bindPan(img) {
    if (!img || img.__vzInit) return;
    img.__vzInit = 1;
    img.addEventListener('dblclick', function (e) {
      e.preventDefault();
      if (vz.s > 1.001) { resetZoom(); } else { vz.s = 2.5; vzApply(); }
    });
    img.addEventListener('pointerdown', function (e) {
      if (!img.classList.contains('pz')) return;
      e.preventDefault();
      try { img.setPointerCapture(e.pointerId); } catch (_) {}
      img.classList.add('dragging');
      var lx = e.clientX, ly = e.clientY;
      var mv = function (ev) {
        vz.x += ev.clientX - lx; vz.y += ev.clientY - ly;
        lx = ev.clientX; ly = ev.clientY;
        vzApply();
      };
      var up = function () {
        img.classList.remove('dragging');
        img.removeEventListener('pointermove', mv);
        img.removeEventListener('pointerup', up);
        img.removeEventListener('pointercancel', up);
      };
      img.addEventListener('pointermove', mv);
      img.addEventListener('pointerup', up);
      img.addEventListener('pointercancel', up);
    });
  }

  // ---- 邻图预取（2026-09-08）：当前照片稳定后，后台串行预取相邻两张的
  // 1600 档与原图。HTTP 层 max-age=3600，翻到时直接命中浏览器缓存 → 秒开。
  // 串行队列：40MB 级原片一张张来，不并发轰炸 NAS/慢链路。
  var _pfQueue = [];
  var _pfBusy = false;
  var _pfTimer = null;
  function _pfNext() {
    if (_pfBusy || !_pfQueue.length) return;
    _pfBusy = true;
    var url = _pfQueue.shift();
    var im = new Image();
    im.onload = im.onerror = function () { _pfBusy = false; _pfNext(); };
    im.src = url;
  }
  // 2026-09-23 Log 原片（大疆 D-Log / 影石 Flat）：/orig 是未还原的灰片，
  // 所以最高清层改用 /preview（服务端已过还原链）。代价是 2200px 封顶，
  // 换来「点开跟墙上看到的一致」——对灰片这个取舍比满分辨率更值。
  function topURL(a) { return (a && a.logcolor) ? '/preview?asset=' + a.id : '/orig?asset=' + a.id; }
  function schedulePrefetch(index) {
    if (_pfTimer) { clearTimeout(_pfTimer); _pfTimer = null; }
    _pfQueue.length = 0;                       // 丢弃上一张的未跑完队列
    var up = st.items[index + 1], down = st.items[index - 1], far = st.items[index + 2];
    [up, down, far].forEach(function (x) {
      if (x && x.type !== 'video') _pfQueue.push('/thumb?asset=' + x.id + '&edge=1600');
    });
    [up, down].forEach(function (x) {          // 原图大，只预取紧邻两张
      if (x && x.type !== 'video') _pfQueue.push(topURL(x));
    });
    _pfTimer = setTimeout(function () { _pfTimer = null; _pfNext(); }, 1500);
  }

  // ---- 渲染（唯一刷 DOM 的函数；索引越界自动收敛，列表删空自动关闭） ----
  function render() {
    if (!st.items.length) { close(); return; }
    if (st.index < 0) st.index = 0;
    if (st.index >= st.items.length) st.index = st.items.length - 1;
    var a = st.items[st.index];
    if (!a) return;
    resetZoom();
    var vm = document.getElementById('viewerMedia');
    if (a.type === 'video') {
      // 2026-09-24 不再静默关闭：4K HEVC 10bit 等编码 Chrome 解不动，静默 close
      // 表现就是「点开即关，打不开」（用户实锤）。改为给出可见提示，转码缓存
      // （videos_lc）转好后 /orig 自动换成转码文件，同 URL 即可恢复播放。
      vm.innerHTML = '<video src="/orig?asset=' + a.id + '" controls autoplay playsinline onerror="WBViewer.videoFail()"></video>';
    } else if (a.crop && a.crop.length === 4) {
      // 非破坏性裁切：容器按裁切后比例适配视口，img 放大平移到裁切框
      // 2026-09-10 修「切图错位到右下 + 长时间黑屏」：原实现只挂一张原图，
      // 加载期间 .vcrop 无尺寸塌缩成 0×0 且被 grid 居中在视口正中心，
      // absolute img 从该点向右下铺开 → 只露出右下四分之一（用户截图实锤）。
      // 改为与普通照片同款渐进：层1 = 1600 档（秒开，同一套裁切数学定位），
      // 层2 = 原图 decode 完成淡入转正。缩略图与原图等比，归一化裁切框通用。
      vm.innerHTML = '<div class="vcrop"><img class="pz" decoding="async" src="/thumb?asset=' + a.id + '&edge=1600" draggable="false" onerror="WBViewer.close()"></div>';
      var wrap = vm.querySelector('.vcrop');
      var cim = vm.querySelector('.vcrop img');
      var applyCrop = function (im) {
        var c = a.crop, nw = im.naturalWidth, nh = im.naturalHeight;
        if (!nw || !nh || !c) return false;
        var A = (nw * c[2]) / (nh * c[3]);
        wrap.style.width = 'min(100vw, calc(100vh * ' + A + '))';
        wrap.style.height = 'min(100vh, calc(100vw / ' + A + '))';
        im.style.width = (100 / c[2]) + '%';
        im.style.height = (100 / c[3]) + '%';
        im.style.left = (-100 * c[0] / c[2]) + '%';
        im.style.top = (-100 * c[1] / c[3]) + '%';
        return true;
      };
      if (cim.complete && cim.naturalWidth) applyCrop(cim);
      else cim.addEventListener('load', function () { applyCrop(cim); });
      bindPan(cim);
      // 原图层：同裁切数学定位，decode 后淡入盖住缩略层，再撤缩略层转正
      var oim = document.createElement('img');
      oim.className = 'pz pz-hi';
      oim.decoding = 'async';
      oim.draggable = false;
      oim.alt = '';
      oim.onerror = function () { if (oim.isConnected) oim.remove(); };
      var promoteCrop = function () {
        if (!oim.isConnected || !applyCrop(oim)) { if (oim.isConnected) oim.remove(); return; }
        oim.classList.add('ready');
        setTimeout(function () {
          if (!oim.isConnected) return;
          cim.remove();
          oim.classList.remove('pz-hi', 'ready');
          oim.style.opacity = '';
          bindPan(oim);
          vzApply();
        }, 320);
      };
      oim.onload = function () {
        if (oim.decode) oim.decode().then(promoteCrop).catch(promoteCrop);
        else promoteCrop();
      };
      wrap.appendChild(oim);   // 必须挂进 .vcrop：裁切 % 数学与 .vcrop img.pz 绝对定位都以它为基准
      oim.src = topURL(a);
      schedulePrefetch(st.index);
    } else {
      // 2026-09-06 修复「点开大图黑屏几秒」：先挂缩略档再后台换原图。
      // 2026-09-08 二版·三级渐进 + 邻图预取（用户反馈"加载还是有点慢"）：
      // 层1 = 480 档（与墙/网格卡片同 URL，浏览器缓存命中 → 查看器 0ms 秒开）；
      // 层2 = 1600 档；层3 = 原图。各层 decode() 完成后淡入并撤掉全部下层；
      // 若低优先级层在高优层之后才到货，直接丢弃（已兜底）。同一时刻只有一个
      // 交互层（.pz 无 pz-hi），其余图层 pointer-events:none 不抢事件。
      vm.innerHTML = '<img class="pz" decoding="async" src="/thumb?asset=' + a.id + '&edge=480" draggable="false" onerror="WBViewer.close()">';
      var base = vm.querySelector('img.pz');
      var bestRank = 1;                       // 已转正层优先级：480=1 1600=2 原图=3
      bindPan(base);
      var mkLayer = function (rank, url) {
        var im = document.createElement('img');
        im.className = 'pz pz-hi';
        im.decoding = 'async';
        im.draggable = false;
        im.alt = '';
        im.dataset.rank = rank;               // 供转正时区分层级：只撤比自己低的
        var promoted = false;
        var promote = function () {
          if (promoted || !im.isConnected) return;
          if (rank <= bestRank) { im.remove(); return; }   // 已有更优层兜底，到货即弃
          promoted = true;
          bestRank = rank;
          im.classList.add('ready');                        // 淡入盖住下层
          setTimeout(function () {
            if (!im.isConnected) return;
            var all = vm.querySelectorAll('img');
            for (var k = 0; k < all.length; k++) {
              // 只撤比自己优先级低的层；更高层（如 1600 转正时仍在加载的原图）必须留活
              if (all[k] !== im && parseInt(all[k].dataset.rank || '1', 10) < rank) all[k].remove();
            }
            im.classList.remove('pz-hi', 'ready');           // 转正为唯一交互层
            im.style.opacity = '';
            bindPan(im);
            vzApply();
          }, 320);
        };
        im.onload = function () {
          if (!im.isConnected) return;
          if (im.decode) im.decode().then(promote).catch(promote);
          else promote();
        };
        // 加载失败（SMB 抖动等）：只撤本层，保留下层继续显示，不闪不关
        im.onerror = function () { if (im.isConnected) im.remove(); };
        vm.appendChild(im);
        im.src = url;
      };
      mkLayer(2, '/thumb?asset=' + a.id + '&edge=1600');
      mkLayer(3, topURL(a));
      schedulePrefetch(st.index);
    }
    var zv = document.getElementById('vZoom');
    if (zv) zv.style.display = a.type === 'video' ? 'none' : 'flex';
    var metaEl = document.getElementById('viewerMeta');
    if (metaEl) {
      metaEl.textContent = (st.opts && st.opts.meta)
        ? st.opts.meta(a, st.index, st.items.length)
        : ((st.index + 1) + ' / ' + st.items.length + ' · ' + fmtDay(a.time));
    }
    updateMetaInfo(a.id);
    var vb = document.getElementById('vFilter');
    if (vb) {
      var conf = null;
      try { conf = (st.opts && st.opts.filterBtn) ? st.opts.filterBtn(a) : null; } catch (_) {}
      if (conf) {
        vb.style.display = 'inline-block';
        vb.disabled = false;
        vb.textContent = conf.text;
        vb.onclick = function () { conf.onClick(a); };
      } else {
        vb.style.display = 'none';
      }
    }
    var vc = document.getElementById('vCrop');
    if (vc) {
      var confc = null;
      try { confc = (st.opts && st.opts.cropBtn) ? st.opts.cropBtn(a) : null; } catch (_) {}
      if (confc) {
        vc.style.display = 'inline-block';
        vc.disabled = false;
        vc.textContent = confc.text;
        vc.classList.remove('danger');
        vc.onclick = function () { confc.onClick(a); };
      } else {
        vc.style.display = 'none';
      }
    }
    var va = document.getElementById('vAction');
    if (va) {
      var conf2 = null;
      try { conf2 = (st.opts && st.opts.actionBtn) ? st.opts.actionBtn(a) : null; } catch (_) {}
      if (conf2) {
        va.style.display = 'inline-block';
        va.disabled = false;
        va.textContent = conf2.text;
        va.classList.toggle('danger', conf2.danger !== false);
        va.onclick = function () { conf2.onClick(a); };
      } else {
        va.style.display = 'none';
      }
    }
    var vg = document.getElementById('vGeo');
    if (vg) {
      vg.style.display = 'inline-block';
      vg.onclick = function () { openGeoEditor(a.id, a.time); };
    }
    if (st.opts && st.opts.afterRender) {
      try { st.opts.afterRender(a, st.index, st.items.length); } catch (_) {}
    }
  }

  // ---- 查看器信息条（2026-09-07）：页码时间后追加 地点/人物/标签，小字同款式 ----
  var metaInfoCache = {};

  function updateMetaInfo(assetId) {
    var metaEl = document.getElementById('viewerMeta');
    if (!metaEl) return;
    var apply = function (info) {
      var c = WBViewer.current();
      if (!c || c.item.id !== assetId) return;   // 已翻到别的照片，丢弃过期结果
      var parts = [];
      if (info.region) parts.push(info.province && info.province !== info.region ? info.province + info.region : info.region);
      if (info.persons && info.persons.length) parts.push(info.persons.join('、'));
      if (info.tags && info.tags.length) parts.push(info.tags.join(' '));
      if (parts.length) metaEl.textContent += ' · ' + parts.join(' · ');
    };
    if (metaInfoCache[assetId]) { apply(metaInfoCache[assetId]); return; }
    fetch('/api/asset_info?asset=' + encodeURIComponent(assetId))
      .then(function (r) { return r.json(); })
      .then(function (info) { metaInfoCache[assetId] = info; apply(info); })
      .catch(function () {});
  }

  // ---- 裁切/二次构图（统一实现，首页/总库共用；数据存这里，页面通过回调同步自己的视图） ----
  var cropMap = {};
  var cropChangeFn = null;   // 页面注册的回调：(assetId, rect|null) => 同步页面自身卡片/列表

  function cropToast(msg) {
    var t = document.createElement('div');
    t.className = 'crop-toast'; t.textContent = msg;
    document.body.appendChild(t);
    setTimeout(function () { t.remove(); }, 2600);
  }
  function loadCropMap() {
    return fetch('/api/crop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'map' }) })
      .then(function (r) { return r.json(); })
      .then(function (d) { cropMap = d.crops || {}; return cropMap; });
  }
  function cropAt(id) { return cropMap[id]; }
  function afterCropChange(id, rect) {
    for (var i = 0; i < st.items.length; i++) {
      if (st.items[i].id === id) st.items[i].crop = rect || undefined;
    }
    if (isOpen()) render();
    if (cropChangeFn) { try { cropChangeFn(id, rect); } catch (_) {} }
  }
  function saveCrop(id, rect) {
    return fetch('/api/crop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'set', asset_id: id, rect: rect }) })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) throw new Error(d.error);
        cropMap[id] = rect;
        afterCropChange(id, rect);
      });
  }
  function clearCrop(id) {
    return fetch('/api/crop', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: 'clear', asset_id: id }) })
      .then(function (r) { return r.json(); })
      .then(function (d) {
        if (d.error) throw new Error(d.error);
        delete cropMap[id];
        afterCropChange(id, null);
      });
  }
  function openCropEditor(id, type) {
    if (type === 'video') { cropToast('视频暂不支持裁切'); return; }
    var ov = document.createElement('div');
    ov.className = 'crop-modal';
    ov.innerHTML = '<div class="crop-title">裁切 / 二次构图<span class="sub">只保存裁切框，原图不受影响，可随时恢复</span></div>'
      + '<div class="crop-stage"><img decoding="async" src="/orig?asset=' + String(id).replace(/"/g, '&quot;') + '" draggable="false" onerror="this.closest(\'.crop-modal\').remove()"><div class="crop-box"><div class="crop-grid"></div><i class="crop-hdl tl" data-h="tl"></i><i class="crop-hdl tr" data-h="tr"></i><i class="crop-hdl bl" data-h="bl"></i><i class="crop-hdl br" data-h="br"></i></div></div>'
      + '<div class="crop-presets"></div>'
      + '<div class="crop-btns"><span class="hint">拖四角调大小 · 拖中间移动 · 框外按下重新画框</span><button class="danger" id="cropReset">恢复原图</button><button id="cropCancel">取消</button><button class="primary" id="cropSave">保存裁切</button></div>';
    document.body.appendChild(ov);
    var img = ov.querySelector('img'), box = ov.querySelector('.crop-box'), stage = ov.querySelector('.crop-stage');
    var PRESETS = [['自由', null], ['原图', 'orig'], ['1:1', 1], ['4:3', 4 / 3], ['3:4', 3 / 4], ['16:9', 16 / 9], ['9:16', 9 / 16]];
    var rect = (cropMap[id] || [0.1, 0.1, 0.8, 0.8]).slice();
    var lock = null, natW = 0, natH = 0, drag = null;
    var clamp01 = function (v) { return Math.max(0, Math.min(1, v)); };
    function drawRect() {
      box.style.left = (rect[0] * 100) + '%'; box.style.top = (rect[1] * 100) + '%';
      box.style.width = (rect[2] * 100) + '%'; box.style.height = (rect[3] * 100) + '%';
    }
    function curLock() { return lock === 'orig' ? (natW && natH ? natW / natH : null) : lock; }
    function fitRectToLock() {
      var L = curLock(); if (!L || !natW || !natH) return;
      var r = L * natH / natW;                     // 归一化 w/h 比
      var cx = rect[0] + rect[2] / 2, cy = rect[1] + rect[3] / 2;
      var w = rect[2], h = w / r;
      if (h > 1) { h = 1; w = h * r; } if (w > 1) { w = 1; h = w / r; }
      rect = [clamp01(cx - w / 2), clamp01(cy - h / 2), w, h];
      rect[0] = Math.min(rect[0], 1 - rect[2]); rect[1] = Math.min(rect[1], 1 - rect[3]);
    }
    function markPreset() {
      ov.querySelectorAll('.crop-presets button').forEach(function (b) {
        b.classList.toggle('on', (lock === null && b.dataset.v === '') || (lock === 'orig' && b.dataset.v === 'orig') || (typeof lock === 'number' && String(lock) === b.dataset.v));
      });
    }
    var presets = ov.querySelector('.crop-presets');
    presets.innerHTML = PRESETS.map(function (p) { return '<button data-v="' + (p[1] === null ? '' : p[1] === 'orig' ? 'orig' : String(p[1])) + '">' + p[0] + '</button>'; }).join('');
    presets.querySelectorAll('button').forEach(function (b) {
      b.addEventListener('click', function () {
        var v = b.dataset.v;
        lock = v === '' ? null : v === 'orig' ? 'orig' : Number(v);
        fitRectToLock(); drawRect(); markPreset();
      });
    });
    function onImgLoad() { natW = img.naturalWidth; natH = img.naturalHeight; fitRectToLock(); drawRect(); markPreset(); }
    img.addEventListener('load', onImgLoad);
    if (img.complete && img.naturalWidth) onImgLoad(); else drawRect();
    var norm = function (e) { var r = stage.getBoundingClientRect(); return [clamp01((e.clientX - r.left) / r.width), clamp01((e.clientY - r.top) / r.height)]; };
    function resizeCorner(mode, o, px, py) {
      var ax, ay;
      if (mode === 'tl') { ax = o[0] + o[2]; ay = o[1] + o[3]; }
      else if (mode === 'tr') { ax = o[0]; ay = o[1] + o[3]; }
      else if (mode === 'bl') { ax = o[0] + o[2]; ay = o[1]; }
      else { ax = o[0]; ay = o[1]; }
      var w = Math.abs(px - ax), h = Math.abs(py - ay);
      var L = curLock();
      if (L && natW && natH) {
        var r = L * natH / natW;
        if (w / r > h) h = w / r; else w = h * r;
        var maxW = (mode === 'tl' || mode === 'bl') ? ax : 1 - ax;
        var maxH = (mode === 'tl' || mode === 'tr') ? ay : 1 - ay;
        if (w > maxW) { w = maxW; h = w / r; }
        if (h > maxH) { h = maxH; w = h * r; }
      } else { w = clamp01(w); h = clamp01(h); }
      if (w < 0.03 || h < 0.03) return o.slice();
      var rx = (mode === 'tl' || mode === 'bl') ? ax - w : ax;
      var ry = (mode === 'tl' || mode === 'tr') ? ay - h : ay;
      return [rx, ry, w, h];
    }
    function onMove(e) {
      if (!drag) return;
      var p = norm(e), px = p[0], py = p[1], o = drag.orig;
      if (drag.mode === 'move') {
        rect = [Math.min(Math.max(o[0] + px - drag.start[0], 0), 1 - o[2]), Math.min(Math.max(o[1] + py - drag.start[1], 0), 1 - o[3]), o[2], o[3]];
      } else if (drag.mode === 'new') {
        rect = [Math.min(drag.start[0], px), Math.min(drag.start[1], py), Math.abs(px - drag.start[0]), Math.abs(py - drag.start[1])];
      } else { rect = resizeCorner(drag.mode, o, px, py); }
      drawRect();
    }
    function onUp() {
      if (drag && drag.mode === 'new' && curLock()) fitRectToLock();
      drag = null; window.removeEventListener('pointermove', onMove); drawRect();
    }
    function startDrag(mode, e) {
      e.preventDefault(); e.stopPropagation();
      drag = { mode: mode, start: norm(e), orig: rect.slice() };
      window.addEventListener('pointermove', onMove);
      window.addEventListener('pointerup', onUp, { once: true });
    }
    box.addEventListener('pointerdown', function (e) { if (e.target.closest('.crop-hdl')) return; startDrag('move', e); });
    ov.querySelectorAll('.crop-hdl').forEach(function (hd) { hd.addEventListener('pointerdown', function (e) { startDrag(hd.dataset.h, e); }); });
    stage.addEventListener('pointerdown', function (e) { if (e.target === img || e.target === stage) startDrag('new', e); });
    var btnReset = ov.querySelector('#cropReset');
    btnReset.disabled = !cropMap[id]; btnReset.style.opacity = cropMap[id] ? 1 : .4;
    btnReset.addEventListener('click', function () {
      clearCrop(id).then(function () { ov.remove(); cropToast('已恢复原图构图'); })
        .catch(function (err) { cropToast('恢复失败：' + err.message); });
    });
    ov.querySelector('#cropCancel').addEventListener('click', function () { ov.remove(); });
    ov.addEventListener('click', function (e) { if (e.target === ov) ov.remove(); });
    ov.querySelector('#cropSave').addEventListener('click', function () {
      if (rect[2] < 0.03 || rect[3] < 0.03) { cropToast('裁切框太小了，拉大一点'); return; }
      saveCrop(id, rect.map(function (v) { return Math.round(v * 10000) / 10000; }))
        .then(function () { ov.remove(); cropToast('已保存裁切 · 原图保留，可随时调整/恢复'); })
        .catch(function (err) { cropToast('保存失败：' + err.message); });
    });
    var onKey = function (e) { if (e.key === 'Escape') { ov.remove(); document.removeEventListener('keydown', onKey); } };
    document.addEventListener('keydown', onKey);
  }

  // ---- 手动定位（半自动地理闭环）：锚点 → 同日互证 → 视觉复核，首页/总库共用 ----
  var geoRegionsCache = null;

  function geoSeedReq(payload) {
    return fetch('/api/geo/seed', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) })
      .then(function (r) { return r.json(); })
      .then(function (d) { if (d.error) throw new Error(d.error); return d; });
  }
  function loadGeoRegions() {
    if (geoRegionsCache) return Promise.resolve(geoRegionsCache);
    return fetch('/api/geo/regions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({}) })
      .then(function (r) { return r.json(); })
      .then(function (d) { geoRegionsCache = { provinces: d.provinces || [], regions: d.regions || [] }; return geoRegionsCache; });
  }
  function openGeoEditor(assetId, photoTime) {
    var ov = document.createElement('div');
    ov.className = 'crop-modal';
    ov.innerHTML = '<div class="crop-title">📍 手动定位<span class="sub">标记一张，系统自动完成同日互证 + 视觉复核</span></div>'
      + '<div class="geo-body">'
      + '<div class="geo-flow"><b>半自动流程：</b>① 选省→市/区（没有就「✎ 自定义」直接输入）→ ② 设<b>起止日期</b>（默认当天；一次旅行标一个区间，区间内每天的照片都自动归入）→ ③ 视觉模型抽张复核，不符自动撤销（锚点保留，可随时移除）</div>'
      + '<div class="geo-row2"><select id="geoProv"><option value="">— 选省份 —</option></select><select id="geoRegion" disabled><option value="">先选省份</option></select></div>'
      + '<div class="geo-row2"><label class="geo-date">开始<input type="date" id="geoStart"></label><label class="geo-date">结束<input type="date" id="geoEnd"></label></div>'
      + '<input id="geoCustom" placeholder="输入自定义地区名（如：北戴河）" autocomplete="off" style="display:none">'
      + '<input id="geoProvince" placeholder="省份（自定义地区选填，帮视觉复核更准）" autocomplete="off" style="display:none">'
      + '<div class="geo-cur none" id="geoCur">查询当前标记中…</div>'
      + '</div>'
      + '<div class="crop-btns"><button class="danger" id="geoRemove" style="display:none">移除标记</button><button id="geoCancel">取消</button><button class="primary" id="geoSave">保存并互证</button></div>';
    document.body.appendChild(ov);
    var provSel = ov.querySelector('#geoProv'), regSel = ov.querySelector('#geoRegion');
    var startIn = ov.querySelector('#geoStart'), endIn = ov.querySelector('#geoEnd');
    var customIn = ov.querySelector('#geoCustom'), provIn = ov.querySelector('#geoProvince');
    var cur = ov.querySelector('#geoCur'), btnRm = ov.querySelector('#geoRemove');
    var GROUPS = null, seeded = null;
    var photoDay = (photoTime || '').slice(0, 10);
    if (/^\d{4}-\d{2}-\d{2}$/.test(photoDay)) { startIn.value = photoDay; endIn.value = photoDay; }

    function fillRegions(provName) {
      var g = GROUPS.filter(function (x) { return x.province === provName; })[0];
      var opts = function (arr) { return arr.map(function (r) {
        return '<option value="' + String(r.region).replace(/"/g, '&quot;') + '">'
          + r.region + (r.gps_n ? '（已有' + r.gps_n + '张GPS）' : '') + '</option>';
      }).join(''); };
      var inlib = [], cities = [];
      (g ? g.regions : []).forEach(function (r) { (r.city ? cities : inlib).push(r); });
      regSel.innerHTML = '<option value="">— 选市/区 —</option>'
        + (inlib.length ? '<optgroup label="库内地区">' + opts(inlib) + '</optgroup>' : '')
        + (cities.length ? '<optgroup label="全国地级市">' + opts(cities) + '</optgroup>' : '');
      regSel.disabled = false;
    }
    function setCustomMode(on) {
      customIn.style.display = on ? 'block' : 'none';
      provIn.style.display = on ? 'block' : 'none';
      regSel.style.display = on ? 'none' : 'block';
      if (!on && provSel.value) fillRegions(provSel.value);
    }
    loadGeoRegions().then(function (d) {
      GROUPS = d.provinces || [];
      provSel.innerHTML = '<option value="">— 选省份 —</option>'
        + GROUPS.map(function (g) {
          var n = g.regions.reduce(function (s, r) { return s + (r.gps_n || 0); }, 0);
          return '<option value="' + String(g.province).replace(/"/g, '&quot;') + '">'
            + g.province + '（' + g.regions.length + '地' + (n ? ' · ' + n + '张GPS' : '') + '）</option>';
        }).join('')
        + '<option value="__custom__">✎ 自定义地区（直接输入）</option>';
      provSel.addEventListener('change', function () {
        if (provSel.value === '__custom__') setCustomMode(true);
        else { setCustomMode(false); if (provSel.value) fillRegions(provSel.value); }
      });
      return geoSeedReq({ action: 'get', asset_id: assetId });
    }).then(function (d) {
      seeded = d.seeded ? d.region : null;
      if (seeded) {
        var hit = GROUPS.filter(function (g) {
          return g.regions.some(function (r) { return r.region === seeded; });
        })[0];
        if (hit) {
          provSel.value = hit.province; fillRegions(hit.province); regSel.value = seeded;
        } else {
          provSel.value = '__custom__'; setCustomMode(true);
          customIn.value = seeded; if (d.province) provIn.value = d.province;
        }
        if (d.start_date) startIn.value = d.start_date;
        if (d.end_date) endIn.value = d.end_date;
        cur.textContent = '当前标记：' + seeded + (d.start_date ? '（' + d.start_date + ' ~ ' + (d.end_date || d.start_date) + '）' : '');
        cur.classList.remove('none');
        btnRm.style.display = 'inline-block';
      } else {
        cur.textContent = '当前未标记（若照片自带 GPS 则显示原 GPS 位置）';
      }
    }).catch(function (err) { cur.textContent = '加载失败：' + err.message + '（服务可能刚重启，关掉重试）'; });
    function close() { ov.remove(); document.removeEventListener('keydown', onKey); }
    function onKey(e) { if (e.key === 'Escape') close(); }
    document.addEventListener('keydown', onKey);
    ov.addEventListener('click', function (e) { if (e.target === ov) close(); });
    ov.querySelector('#geoCancel').addEventListener('click', close);
    btnRm.addEventListener('click', function () {
      geoSeedReq({ action: 'remove', asset_id: assetId }).then(function (d) {
        cropToast(d && d.queued ? '已移除定位标记 · 重建排队中（30 秒窗口合并）' : '已移除定位标记 · 地理表已重建');
      }).catch(function (err) { cropToast('移除失败：' + err.message); });
    });
    ov.querySelector('#geoSave').addEventListener('click', function () {
      var isCustom = provSel.value === '__custom__';
      var region = isCustom ? customIn.value.trim() : regSel.value;
      if (!region) { cropToast(isCustom ? '输入自定义地区名' : '先选省份和市/区'); return; }
      var sb = ov.querySelector('#geoSave');
      sb.disabled = true; sb.textContent = '互证中…';
      var payload = { action: 'set', asset_id: assetId, region: region,
                      start_date: startIn.value, end_date: endIn.value };
      if (isCustom) payload.province = provIn.value.trim();
      geoSeedReq(payload).then(function (d) {
        ov.remove();
        if (d.queued) {
          cropToast('已标记「' + region + '」· 已入库，全库重建排队中（30 秒窗口内多次标注自动合并）');
        } else {
          cropToast('已标记「' + region + '」· 同日互证生效（' + (d.assigned || 0) + ' 张有位置），视觉复核后台运行中');
        }
      }).catch(function (err) {
        sb.disabled = false; sb.textContent = '保存并互证';
        cropToast('标记失败：' + err.message + '（服务可能刚重启，稍后再试）');
      });
    });
  }

  // ---- 对外 API ----
  function open(items, index, opts) {
    if (!items || !items.length) return;
    ensure();
    st = { items: items, index: index | 0, opts: opts || null };
    el.classList.add('open');
    el.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
    render();
  }
  function close() {
    var opts = st.opts;
    document.getElementById('viewerMedia').innerHTML = '';
    resetZoom();
    if (el) {
      el.classList.remove('open');
      el.setAttribute('aria-hidden', 'true');
    }
    document.body.style.overflow = '';
    st = { items: [], index: -1, opts: null };
    if (opts && opts.onClose) { try { opts.onClose(); } catch (_) {} }
  }
  function step(delta) {
    if (!st.items.length) return;
    var video = document.querySelector('#viewerMedia video');
    if (video) video.pause();
    st.index = (st.index + delta + st.items.length) % st.items.length;
    render();
  }
  function current() {
    if (st.index < 0 || !st.items.length) return null;
    return { item: st.items[st.index], index: st.index, items: st.items };
  }

  // 2026-09-24 视频解码失败兜底：给出可见提示而不是静默关闭 viewer
  function videoFail() {
    var vm = document.getElementById('viewerMedia');
    if (!vm) return;
    vm.innerHTML = '<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;'
      + 'height:100vh;color:#9aa3af;font:14px/1.7 -apple-system,sans-serif;text-align:center;padding:0 32px">'
      + '<div>该视频编码当前浏览器无法播放（如 4K HEVC 10bit）</div>'
      + '<div style="opacity:.7">播放转码正在后台生成，完成后重新打开即可 · 手机 Safari 通常可直接播放</div></div>';
  }

  window.WBViewer = {
    open: open,
    close: close,
    step: step,
    render: render,
    current: current,
    isOpen: isOpen,
    videoFail: videoFail,
    zoomIn: function () { zoomBy(1.35); },
    zoomOut: function () { zoomBy(1 / 1.35); },
    rotate: rotate,
    resetZoom: resetZoom,
    /* 裁切/二次构图（统一实现）：页面启动时 loadCropMap()，需要裁切框数据用 cropAt(id)，
     * 页面要同步自己的卡片视图就 setCropChangeHandler((id, rect|null) => ...)。 */
    loadCropMap: loadCropMap,
    cropAt: cropAt,
    openCropEditor: openCropEditor,
    setCropChangeHandler: function (fn) { cropChangeFn = fn; },
    toast: cropToast
  };
})();
