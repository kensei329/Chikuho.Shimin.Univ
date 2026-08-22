/**
 * 筑豊市民大学 — 全ページ共通
 * ------------------------------------------------------------------------
 * ここでやるのは 2 つだけ。
 *
 *   1. タブバーとリンクの「押した手応え」
 *      指を置いた瞬間に沈み、離してから戻す。iOS の Safari は <a> の
 *      :active が効かないことがあるので、CSS だけに頼らずクラスでも持つ。
 *
 *   2. 次に開くページの先読み
 *      写真の多いページは、開いてから取りにいくと待たされる。回線が
 *      細いときと、通信量を節約する設定のときは何もしない。
 */
(function () {
  'use strict';

  /* ── 押した手応え ───────────────────────────────────────────────── */

  var PRESS = '.nav__link, .btn, .kaiPick__item, .card__goBtn';
  var pressedAt = 0;
  var releaseTimer = 0;

  document.addEventListener('pointerdown', function (e) {
    var t = e.target.closest && e.target.closest(PRESS);
    if (!t) return;
    window.clearTimeout(releaseTimer);
    pressedAt = Date.now();
    t.classList.add('is-pressed');
  }, { passive: true });

  function release(e) {
    var t = e.target.closest && e.target.closest(PRESS);
    if (!t) return;
    window.clearTimeout(releaseTimer);
    /* 一瞬で離されても沈んだことが見えるよう、最低 160ms は残す */
    releaseTimer = window.setTimeout(function () {
      t.classList.remove('is-pressed');
    }, Math.max(0, 160 - (Date.now() - pressedAt)));
  }
  document.addEventListener('pointerup', release, { passive: true });
  document.addEventListener('pointercancel', release, { passive: true });

  /* 押したまま指がずれたときに沈んだままにならないようにする */
  document.addEventListener('pointerleave', function (e) {
    var t = e.target.closest && e.target.closest(PRESS);
    if (t) t.classList.remove('is-pressed');
  }, true);

  /* ── 次のページの先読み ─────────────────────────────────────────── */

  var c = navigator.connection || {};
  if (c.saveData || /2g/.test(c.effectiveType || '')) return;

  var here = location.pathname.split('/').pop() || 'index.html';
  var order = ['katsudo.html', 'rinen.html', 'index.html', 'bunkakai.html', 'nenkan.html'];

  window.addEventListener('load', function () {
    window.setTimeout(function () {
      order.forEach(function (href) {
        if (href === here) return;
        var l = document.createElement('link');
        l.rel = 'prefetch';
        l.href = href;
        document.head.appendChild(l);
      });
    }, 1200);   // まず今のページを出し切ってから
  });
})();
