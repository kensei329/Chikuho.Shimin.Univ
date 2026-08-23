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
  var order = ['katsudo.html', 'rinen.html', '/', 'bunkakai.html', 'nenkan.html'];

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

/**
 * 横の案内板（デスクトップ）— いま見ているところを光らせる
 * ------------------------------------------------------------------------
 * 左の帯には、活動の講師や年間予定の月へ飛ぶリンクが並んでいる。
 * 押したときだけでなく、画面を送っているあいだも、いま読んでいるところが
 * 光って移り変わるようにする。5月から6月へ入れば、帯の色もついてくる。
 *
 * 光らせ方は aria-current。ページそのものを示す aria-current="page" と
 * 同じ見た目になるので、CSS は既にあるものがそのまま効く。
 *
 * 帯は幅 940px 以上でしか出さないが、DOM には常にある。リンクの飛び先が
 * このページに無ければ何もしないので、携帯で読み込んでも害はない。
 */
(function () {
  'use strict';

  var tree = document.querySelector('.siteRail--left .tree');
  if (!tree) return;

  var here = location.pathname.split('/').pop() || 'index.html';

  /* このページの中を指しているリンクだけを拾う */
  var spots = [];
  [].forEach.call(tree.querySelectorAll('a[href*="#"]'), function (a) {
    var href = a.getAttribute('href') || '';
    var cut = href.indexOf('#');
    if ((href.slice(0, cut) || here) !== here) return;
    var el = document.getElementById(href.slice(cut + 1));
    if (el) spots.push({ a: a, el: el });
  });
  if (!spots.length) return;

  /* 活動のカードは自前の帯（.reel）の中で送られる。ほかは画面ごと動く */
  var reel = document.querySelector('.reel');
  var head = document.querySelector('.head');
  var now = null;

  function look() {
    /* ヘッダーの少し下を基準の線にして、そこを最後に越えたものを選ぶ */
    var line = (head ? head.getBoundingClientRect().bottom : 0) + 24;
    var pick = spots[0];
    for (var i = 0; i < spots.length; i++) {
      if (spots[i].el.getBoundingClientRect().top <= line) pick = spots[i];
    }
    if (pick === now) return;
    if (now) now.a.removeAttribute('aria-current');
    pick.a.setAttribute('aria-current', 'true');
    now = pick;
  }

  var waiting = 0;
  function onScroll() {
    if (waiting) return;
    waiting = requestAnimationFrame(function () { waiting = 0; look(); });
  }

  (reel || window).addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  window.addEventListener('hashchange', onScroll);
  window.addEventListener('load', onScroll);
  look();
})();

/**
 * 写真が届かなかったとき
 * ------------------------------------------------------------------------
 * 以前は <img onerror="..."> と HTML に直接書いていた。読みやすかったが、
 * 「このページで動かしてよい JavaScript は、この置き場のものだけ」という
 * 決まり（Content-Security-Policy）を入れると、直書きは動かせない。
 * 同じことをここでやる。
 *
 * script は defer で後から動くので、走り出す前にもう失敗している写真が
 * ある。読み終わっているのに大きさが 0 のものは、そこで失敗したものなので
 * 拾い直す。あとから失敗するものは error で受ける。
 */
(function () {
  'use strict';

  function blank(img) {
    var box = img.closest('.card__photo, .slide__photo');
    if (box) box.classList.add('is-blank');
    img.remove();
  }

  [].forEach.call(document.querySelectorAll('.card__photo img, .slide__photo img'), function (img) {
    if (img.complete) { if (!img.naturalWidth) blank(img); return; }
    img.addEventListener('error', function () { blank(img); }, { once: true });
  });
})();
