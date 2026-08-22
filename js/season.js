/**
 * 年間予定 — その月らしいものが、画面に降る
 * ---------------------------------------------------------------------------
 * 読んでいる月に合わせて、鯉のぼりや雪や桜が画面を流れる。月が変わると、
 * 前のものは薄れて消え、新しいものが湧く。
 *
 * ■ canvas ひとつで描く
 *   絵は持たない。すべて図形で描くので、写真も SVG も読みに行かない。
 *   会員の多くは 70 代で、通信環境もまちまち。飾りのために待たせたくない。
 *
 * ■ 読むじゃまをしない
 *   文字の上を流れるので、濃さは控えめにしてある。指も通す
 *   （pointer-events: none）。動きが苦手な方の設定（prefers-reduced-motion）
 *   では、何も出さずに終わる。
 *
 * ■ 数は絞る
 *   古い端末でも詰まらないよう、同時に出るのは 18 個まで。画面を見ていない
 *   あいだ（別のタブなど）は止める。
 */
(function () {
  'use strict';

  var months = [].slice.call(document.querySelectorAll('.mo'));
  var canvas = document.getElementById('season');
  if (!months.length || !canvas || !canvas.getContext) return;

  var slow = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)');
  if (slow && slow.matches) return;

  var ctx = canvas.getContext('2d');
  var TAU = Math.PI * 2;
  var MAX = 14;

  /* ── 絵 ───────────────────────────────────────────────────────────
     どれも原点を中心に、大きさ s で描く。向きと濃さは呼ぶ側で決める。 */

  function koi(c, s, col) {                       /* 5月 鯉のぼり */
    c.fillStyle = col;
    c.beginPath();
    c.moveTo(-s, -0.34 * s);
    c.quadraticCurveTo(0.2 * s, -0.30 * s, 0.62 * s, -0.15 * s);
    c.lineTo(s, -0.44 * s);
    c.lineTo(0.84 * s, 0);
    c.lineTo(s, 0.44 * s);
    c.lineTo(0.62 * s, 0.15 * s);
    c.quadraticCurveTo(0.2 * s, 0.30 * s, -s, 0.34 * s);
    c.closePath();
    c.fill();
    c.strokeStyle = 'rgba(255,255,255,.85)';
    c.lineWidth = Math.max(1, s * 0.07);
    c.beginPath(); c.ellipse(-s, 0, s * 0.05, s * 0.34, 0, 0, TAU); c.stroke();
    for (var i = 0; i < 3; i++) {                 /* うろこ */
      c.beginPath();
      c.arc(-0.35 * s + i * 0.32 * s, 0, s * 0.2, -1.1, 1.1);
      c.stroke();
    }
    c.fillStyle = '#fff';
    c.beginPath(); c.arc(-0.7 * s, -0.12 * s, s * 0.12, 0, TAU); c.fill();
    c.fillStyle = '#20301F';
    c.beginPath(); c.arc(-0.7 * s, -0.12 * s, s * 0.055, 0, TAU); c.fill();
  }

  function teru(c, s) {                            /* 6月 てるてる坊主 */
    c.fillStyle = '#FFFDF7';
    c.beginPath();
    c.moveTo(-0.30 * s, -0.26 * s);
    c.quadraticCurveTo(-0.66 * s, 0.50 * s, -0.54 * s, 0.72 * s);
    c.quadraticCurveTo(-0.27 * s, 0.56 * s, 0, 0.74 * s);
    c.quadraticCurveTo(0.27 * s, 0.56 * s, 0.54 * s, 0.72 * s);
    c.quadraticCurveTo(0.66 * s, 0.50 * s, 0.30 * s, -0.26 * s);
    c.closePath(); c.fill();
    c.beginPath(); c.arc(0, -0.42 * s, 0.34 * s, 0, TAU); c.fill();
    c.strokeStyle = '#C8556A'; c.lineWidth = Math.max(1.4, s * 0.09);
    c.beginPath(); c.moveTo(-0.28 * s, -0.16 * s); c.lineTo(0.28 * s, -0.16 * s); c.stroke();
    c.fillStyle = '#3A4A38';
    c.beginPath(); c.arc(-0.12 * s, -0.46 * s, s * 0.055, 0, TAU); c.fill();
    c.beginPath(); c.arc(0.12 * s, -0.46 * s, s * 0.055, 0, TAU); c.fill();
    c.strokeStyle = '#3A4A38'; c.lineWidth = Math.max(1, s * 0.05);
    c.beginPath(); c.arc(0, -0.34 * s, s * 0.1, 0.25, Math.PI - 0.25); c.stroke();
  }

  function tanzaku(c, s, col) {                    /* 7月 短冊 */
    c.strokeStyle = 'rgba(90,110,86,.75)'; c.lineWidth = Math.max(1, s * 0.05);
    c.beginPath(); c.moveTo(0, -0.86 * s); c.quadraticCurveTo(0.16 * s, -0.72 * s, 0, -0.58 * s); c.stroke();
    c.fillStyle = col;
    var w = 0.30 * s, h = 0.58 * s, r = s * 0.07;
    c.beginPath();
    c.moveTo(-w + r, -h); c.lineTo(w - r, -h); c.quadraticCurveTo(w, -h, w, -h + r);
    c.lineTo(w, h - r); c.quadraticCurveTo(w, h, w - r, h);
    c.lineTo(-w + r, h); c.quadraticCurveTo(-w, h, -w, h - r);
    c.lineTo(-w, -h + r); c.quadraticCurveTo(-w, -h, -w + r, -h);
    c.closePath(); c.fill();
    c.strokeStyle = 'rgba(255,255,255,.7)'; c.lineWidth = Math.max(1, s * 0.045);
    for (var i = 0; i < 3; i++) {
      c.beginPath();
      c.moveTo(-w * 0.5, -h * 0.5 + i * h * 0.42);
      c.lineTo(w * 0.5, -h * 0.5 + i * h * 0.42);
      c.stroke();
    }
  }

  function kakigori(c, s) {                        /* 8月 かき氷 */
    c.fillStyle = '#FFFFFF';                       /* 氷の山 */
    c.beginPath();
    c.moveTo(-0.52 * s, -0.06 * s);
    c.quadraticCurveTo(-0.40 * s, -0.62 * s, -0.10 * s, -0.44 * s);
    c.quadraticCurveTo(0.02 * s, -0.80 * s, 0.20 * s, -0.44 * s);
    c.quadraticCurveTo(0.44 * s, -0.62 * s, 0.52 * s, -0.06 * s);
    c.closePath(); c.fill();
    c.fillStyle = 'rgba(226,74,74,.72)';           /* いちごのシロップ */
    c.beginPath();
    c.moveTo(-0.44 * s, -0.20 * s);
    c.quadraticCurveTo(-0.34 * s, -0.56 * s, -0.10 * s, -0.42 * s);
    c.quadraticCurveTo(0.02 * s, -0.72 * s, 0.19 * s, -0.42 * s);
    c.quadraticCurveTo(0.40 * s, -0.54 * s, 0.44 * s, -0.20 * s);
    c.quadraticCurveTo(0, -0.06 * s, -0.44 * s, -0.20 * s);
    c.closePath(); c.fill();
    c.fillStyle = 'rgba(120,170,200,.55)';         /* 器 */
    c.beginPath();
    c.moveTo(-0.54 * s, -0.06 * s); c.lineTo(0.54 * s, -0.06 * s);
    c.lineTo(0.22 * s, 0.62 * s); c.lineTo(-0.22 * s, 0.62 * s);
    c.closePath(); c.fill();
    c.strokeStyle = 'rgba(255,255,255,.8)'; c.lineWidth = Math.max(1, s * 0.05);
    c.beginPath(); c.moveTo(-0.54 * s, -0.06 * s); c.lineTo(0.54 * s, -0.06 * s); c.stroke();
  }

  function dango(c, s) {                           /* 9月 三色団子 */
    c.strokeStyle = '#C9A66B'; c.lineWidth = Math.max(1.4, s * 0.08);
    c.beginPath(); c.moveTo(0, -0.80 * s); c.lineTo(0, 0.86 * s); c.stroke();
    var cols = ['#F2B6C6', '#FFFAF2', '#B7D2A2'];
    for (var i = 0; i < 3; i++) {
      c.fillStyle = cols[i];
      c.beginPath(); c.arc(0, -0.42 * s + i * 0.42 * s, 0.27 * s, 0, TAU); c.fill();
      c.strokeStyle = 'rgba(120,110,90,.28)'; c.lineWidth = Math.max(1, s * 0.035);
      c.stroke();
    }
  }

  function leaf(c, s, col) {                       /* 10月 落ち葉 */
    c.fillStyle = col;
    c.beginPath();
    for (var i = 0; i < 5; i++) {
      var a = -Math.PI / 2 + (i - 2) * 0.62;
      var tipX = Math.cos(a) * 0.78 * s, tipY = Math.sin(a) * 0.78 * s;
      var lx = Math.cos(a - 0.34) * 0.3 * s, ly = Math.sin(a - 0.34) * 0.3 * s;
      var rx = Math.cos(a + 0.34) * 0.3 * s, ry = Math.sin(a + 0.34) * 0.3 * s;
      if (i === 0) c.moveTo(lx, ly); else c.lineTo(lx, ly);
      c.lineTo(tipX, tipY);
      c.lineTo(rx, ry);
    }
    c.closePath(); c.fill();
    c.strokeStyle = 'rgba(110,60,20,.45)'; c.lineWidth = Math.max(1, s * 0.05);
    c.beginPath(); c.moveTo(0, 0.2 * s); c.lineTo(0, 0.72 * s); c.stroke();
  }

  function lantern(c, s) {                         /* 11月 赤ちょうちん */
    c.strokeStyle = 'rgba(70,60,50,.6)'; c.lineWidth = Math.max(1, s * 0.05);
    c.beginPath(); c.moveTo(0, -0.92 * s); c.lineTo(0, -0.66 * s); c.stroke();
    c.fillStyle = '#4A3B2E';
    c.fillRect(-0.26 * s, -0.70 * s, 0.52 * s, 0.12 * s);
    c.fillRect(-0.26 * s, 0.58 * s, 0.52 * s, 0.12 * s);
    c.fillStyle = '#D0342C';
    c.beginPath(); c.ellipse(0, -0.02 * s, 0.44 * s, 0.62 * s, 0, 0, TAU); c.fill();
    c.strokeStyle = 'rgba(90,20,16,.4)'; c.lineWidth = Math.max(1, s * 0.035);
    for (var i = -2; i <= 2; i++) {
      c.beginPath();
      c.ellipse(0, -0.02 * s, 0.44 * s, 0.62 * s, 0, 0, TAU);
      c.save(); c.beginPath();
      c.rect(-0.5 * s, -0.02 * s + i * 0.22 * s - s * 0.012, s, s * 0.024);
      c.clip();
      c.beginPath(); c.ellipse(0, -0.02 * s, 0.44 * s, 0.62 * s, 0, 0, TAU); c.stroke();
      c.restore();
    }
    c.fillStyle = 'rgba(255,240,200,.55)';
    c.beginPath(); c.ellipse(-0.14 * s, -0.12 * s, 0.09 * s, 0.28 * s, 0, 0, TAU); c.fill();
  }

  function santa(c, s) {                           /* 12月 サンタクロース */
    c.fillStyle = '#F0C9A8';                       /* 顔 */
    c.beginPath(); c.arc(0, 0, 0.42 * s, 0, TAU); c.fill();
    c.fillStyle = '#FFFFFF';                       /* ひげ */
    c.beginPath();
    c.moveTo(-0.40 * s, 0.02 * s);
    c.quadraticCurveTo(-0.30 * s, 0.86 * s, 0, 0.78 * s);
    c.quadraticCurveTo(0.30 * s, 0.86 * s, 0.40 * s, 0.02 * s);
    c.quadraticCurveTo(0, 0.30 * s, -0.40 * s, 0.02 * s);
    c.closePath(); c.fill();
    c.fillStyle = '#C7302B';                       /* 帽子 */
    c.beginPath();
    c.moveTo(-0.46 * s, -0.24 * s);
    c.quadraticCurveTo(-0.10 * s, -1.02 * s, 0.52 * s, -0.72 * s);
    c.quadraticCurveTo(0.30 * s, -0.36 * s, 0.46 * s, -0.24 * s);
    c.closePath(); c.fill();
    c.fillStyle = '#FFFFFF';
    c.beginPath(); c.ellipse(0, -0.26 * s, 0.48 * s, 0.11 * s, 0, 0, TAU); c.fill();
    c.beginPath(); c.arc(0.55 * s, -0.74 * s, 0.14 * s, 0, TAU); c.fill();
    c.fillStyle = '#3A3028';                       /* 目 */
    c.beginPath(); c.arc(-0.14 * s, -0.02 * s, s * 0.05, 0, TAU); c.fill();
    c.beginPath(); c.arc(0.14 * s, -0.02 * s, s * 0.05, 0, TAU); c.fill();
  }

  function snow(c, s) {                             /* 1月 雪 */
    c.strokeStyle = '#FFFFFF';
    c.lineWidth = Math.max(1.2, s * 0.09);
    c.lineCap = 'round';
    for (var i = 0; i < 6; i++) {
      c.save(); c.rotate(i * Math.PI / 3);
      c.beginPath(); c.moveTo(0, 0); c.lineTo(0, -0.8 * s); c.stroke();
      c.beginPath();
      c.moveTo(0, -0.48 * s); c.lineTo(-0.22 * s, -0.66 * s);
      c.moveTo(0, -0.48 * s); c.lineTo(0.22 * s, -0.66 * s);
      c.stroke();
      c.restore();
    }
  }

  function oni(c, s) {                              /* 2月 節分の鬼 */
    c.fillStyle = '#D2442F';
    c.beginPath(); c.arc(0, 0.04 * s, 0.5 * s, 0, TAU); c.fill();
    c.fillStyle = '#F5E3B8';                        /* 角 */
    c.beginPath();
    c.moveTo(-0.34 * s, -0.36 * s); c.lineTo(-0.44 * s, -0.82 * s); c.lineTo(-0.14 * s, -0.44 * s);
    c.closePath(); c.fill();
    c.beginPath();
    c.moveTo(0.34 * s, -0.36 * s); c.lineTo(0.44 * s, -0.82 * s); c.lineTo(0.14 * s, -0.44 * s);
    c.closePath(); c.fill();
    c.fillStyle = '#FFFFFF';
    c.beginPath(); c.ellipse(-0.17 * s, -0.04 * s, 0.13 * s, 0.10 * s, 0, 0, TAU); c.fill();
    c.beginPath(); c.ellipse(0.17 * s, -0.04 * s, 0.13 * s, 0.10 * s, 0, 0, TAU); c.fill();
    c.fillStyle = '#33261F';
    c.beginPath(); c.arc(-0.16 * s, -0.03 * s, s * 0.055, 0, TAU); c.fill();
    c.beginPath(); c.arc(0.16 * s, -0.03 * s, s * 0.055, 0, TAU); c.fill();
    c.beginPath();                                  /* 口と牙 */
    c.moveTo(-0.22 * s, 0.22 * s); c.lineTo(0.22 * s, 0.22 * s);
    c.lineTo(0.12 * s, 0.36 * s); c.lineTo(-0.12 * s, 0.36 * s);
    c.closePath(); c.fill();
    c.fillStyle = '#FFFFFF';
    c.beginPath();
    c.moveTo(-0.14 * s, 0.22 * s); c.lineTo(-0.06 * s, 0.22 * s); c.lineTo(-0.10 * s, 0.32 * s);
    c.closePath(); c.fill();
    c.beginPath();
    c.moveTo(0.14 * s, 0.22 * s); c.lineTo(0.06 * s, 0.22 * s); c.lineTo(0.10 * s, 0.32 * s);
    c.closePath(); c.fill();
  }

  function mame(c, s) {                              /* 2月 豆 */
    c.fillStyle = '#D8B87E';
    c.beginPath(); c.ellipse(0, 0, 0.34 * s, 0.26 * s, 0.4, 0, TAU); c.fill();
    c.strokeStyle = 'rgba(120,90,50,.5)'; c.lineWidth = Math.max(1, s * 0.06);
    c.beginPath(); c.moveTo(-0.16 * s, -0.08 * s); c.lineTo(0.16 * s, 0.08 * s); c.stroke();
  }

  function sakura(c, s) {                            /* 3月 桜 */
    c.fillStyle = '#F7C6D4';
    for (var i = 0; i < 5; i++) {
      c.save(); c.rotate(i * TAU / 5);
      c.beginPath();
      c.moveTo(0, 0);
      c.quadraticCurveTo(-0.30 * s, -0.44 * s, -0.09 * s, -0.74 * s);
      c.quadraticCurveTo(0, -0.62 * s, 0.09 * s, -0.74 * s);
      c.quadraticCurveTo(0.30 * s, -0.44 * s, 0, 0);
      c.closePath(); c.fill();
      c.restore();
    }
    c.fillStyle = '#E58AA6';
    c.beginPath(); c.arc(0, 0, 0.12 * s, 0, TAU); c.fill();
  }

  /* ── 月ごとの決めごと ─────────────────────────────────────────────
     mode は流れ方。yoko は横に流れる（鯉のぼり、ちょうちん）、
     tate は上から落ちてくる。 */

  var KOI = ['#2F3A36', '#C7302B', '#2F6FA8'];
  var TAN = ['#E4718B', '#5FA9D8', '#F0C64B', '#8FBF7A', '#B39CD0'];
  var HA  = ['#C7692B', '#D9A227', '#A8452A', '#8E7B2E'];

  var SEASON = {
    5:  { mode: 'yoko', size: [22, 32], speed: [16, 30], draw: function (c, s, r) { koi(c, s, KOI[r % 3]); } },
    6:  { mode: 'tate', size: [21, 31], speed: [26, 44], draw: function (c, s) { teru(c, s); } },
    7:  { mode: 'tate', size: [21, 31], speed: [28, 48], draw: function (c, s, r) { tanzaku(c, s, TAN[r % 5]); } },
    8:  { mode: 'tate', size: [22, 32], speed: [22, 38], draw: function (c, s) { kakigori(c, s); } },
    9:  { mode: 'tate', size: [21, 30], speed: [26, 44], draw: function (c, s) { dango(c, s); } },
    10: { mode: 'tate', size: [18, 28], speed: [30, 52], spin: true, draw: function (c, s, r) { leaf(c, s, HA[r % 4]); } },
    11: { mode: 'yoko', size: [22, 32], speed: [10, 20], draw: function (c, s) { lantern(c, s); } },
    12: { mode: 'tate', size: [22, 32], speed: [24, 40], draw: function (c, s) { santa(c, s); } },
    1:  { mode: 'tate', size: [13, 22], speed: [22, 40], draw: function (c, s) { snow(c, s); } },
    2:  { mode: 'tate', size: [18, 29], speed: [30, 52], draw: function (c, s, r) { (r % 3 === 0 ? oni : mame)(c, s); } },
    3:  { mode: 'tate', size: [14, 23], speed: [26, 46], spin: true, draw: function (c, s) { sakura(c, s); } }
  };

  /* ── 画面 ─────────────────────────────────────────────────────── */

  var W = 0, H = 0, dpr = 1;

  function size() {
    var r = canvas.getBoundingClientRect();
    W = r.width; H = r.height;
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  /* ── 降るもの ─────────────────────────────────────────────────── */

  var bits = [];
  var now = 5;
  var rnd = function (a, b) { return a + Math.random() * (b - a); };

  function born(m, fresh) {
    var k = SEASON[m];
    if (!k) return null;
    var s = rnd(k.size[0], k.size[1]);
    var b = {
      m: m, s: s, r: (Math.random() * 1000) | 0,
      v: rnd(k.speed[0], k.speed[1]),
      sway: rnd(8, 26), w: rnd(0.6, 1.5), ph: Math.random() * TAU,
      spin: k.spin ? rnd(-1.1, 1.1) : 0,
      rot: k.spin ? Math.random() * TAU : 0,
      a: 0, to: rnd(0.5, 0.78)
    };
    if (k.mode === 'yoko') {
      b.yoko = true;
      b.dir = Math.random() < 0.5 ? 1 : -1;
      b.x = fresh ? rnd(0, W) : (b.dir > 0 ? -s * 1.6 : W + s * 1.6);
      b.y = rnd(s, H - s);
    } else {
      b.x = rnd(s, Math.max(s + 1, W - s));
      b.y = fresh ? rnd(-s, H) : -s * 1.8;
    }
    return b;
  }

  function fill(fresh) {
    var want = Math.min(MAX, Math.round(W / 46) + 4);
    var live = 0, i;
    for (i = 0; i < bits.length; i++) if (bits[i].m === now && !bits[i].bye) live++;
    for (i = live; i < want; i++) {
      var b = born(now, fresh);
      if (b) bits.push(b);
    }
  }

  function look() {
    /* 画面の真ん中にいちばん近い月を、いま読んでいる月とみなす */
    var mid = window.innerHeight / 2, best = null, near = Infinity;
    for (var i = 0; i < months.length; i++) {
      var r = months[i].getBoundingClientRect();
      var d = Math.abs(r.top + r.height / 2 - mid);
      if (d < near) { near = d; best = months[i]; }
    }
    if (!best) return;
    var tag = best.querySelector('.mo__badge b');
    var m = tag ? parseInt(tag.textContent, 10) : 0;
    if (!SEASON[m] || m === now) return;
    now = m;
    for (var j = 0; j < bits.length; j++) bits[j].bye = true;   /* 前の月は薄れて消える */
    fill(true);
  }

  /* ── 動かす ───────────────────────────────────────────────────── */

  var last = 0, run = 0;

  function frame(t) {
    run = requestAnimationFrame(frame);
    var dt = last ? Math.min((t - last) / 1000, 0.05) : 0.016;
    last = t;

    ctx.clearRect(0, 0, W, H);

    for (var i = bits.length - 1; i >= 0; i--) {
      var b = bits[i];
      b.a += (b.bye ? -1.4 : 1.4) * dt * (b.bye ? 1 : 1);
      if (b.a > b.to) b.a = b.to;
      if (b.bye && b.a <= 0) { bits.splice(i, 1); continue; }

      b.ph += b.w * dt;
      b.rot += b.spin * dt;

      if (b.yoko) {
        b.x += b.v * b.dir * dt;
        b.y += Math.sin(b.ph) * b.sway * dt;
        if (b.dir > 0 && b.x > W + b.s * 2) { b.x = -b.s * 2; }
        if (b.dir < 0 && b.x < -b.s * 2) { b.x = W + b.s * 2; }
      } else {
        b.y += b.v * dt;
        b.x += Math.sin(b.ph) * b.sway * dt;
        if (b.y > H + b.s * 1.6) {
          if (b.bye) { bits.splice(i, 1); continue; }
          b.y = -b.s * 1.6; b.x = rnd(b.s, Math.max(b.s + 1, W - b.s));
        }
      }

      var k = SEASON[b.m];
      if (!k) { bits.splice(i, 1); continue; }
      ctx.save();
      ctx.globalAlpha = Math.max(0, b.a);
      ctx.translate(b.x, b.y);
      if (b.yoko) { if (b.dir < 0) ctx.scale(-1, 1); ctx.rotate(Math.sin(b.ph) * 0.12); }
      /* 回すのは葉と花びらだけ。人や物は、揺れるだけにとどめる */
      else { ctx.rotate(b.spin ? b.rot : Math.sin(b.ph) * 0.16); }
      k.draw(ctx, b.s, b.r);
      ctx.restore();
    }
  }

  function start() { if (!run) { last = 0; run = requestAnimationFrame(frame); } }
  function stop() { if (run) { cancelAnimationFrame(run); run = 0; } }

  /* ── つなぐ ───────────────────────────────────────────────────── */

  var tick = 0;
  function onScroll() {
    if (tick) return;
    tick = requestAnimationFrame(function () { tick = 0; look(); });
  }

  size();
  fill(true);
  start();

  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', function () { size(); fill(true); }, { passive: true });
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) stop(); else start();
  });
  if (slow && slow.addEventListener) {
    slow.addEventListener('change', function (e) {
      if (e.matches) { stop(); ctx.clearRect(0, 0, W, H); } else { start(); }
    });
  }

  look();
})();
