/**
 * 実績ページ — 縦スワイプのカードフィード
 * ------------------------------------------------------------------------
 * 送り（スワイプでカードが切り替わる動き）は CSS の scroll-snap が
 * やっている。ここで面倒を見るのは 4 つだけ。
 *
 *   1. いいね  … 押せる。押した記録はこの端末のブラウザに残る
 *   2. コメント … そのカードの解説をシートで開く
 *   3. シェア  … そのカードへの直リンクを共有する
 *   4. 進み具合 … 左端の目盛りと、URL の #（戻る操作で同じ位置に戻れる）
 *
 * ■ いいねの数について
 *   このサイトはサーバーを持たない静的サイトなので、押した数を全員で
 *   共有することはできない。表示している数は「もとの数 + この端末で
 *   押したぶん」で、他の人の画面には反映されない。
 */
(function () {
  'use strict';

  var reel = document.getElementById('reel');
  if (!reel) return;

  var cards = [].slice.call(reel.querySelectorAll('.card'));
  if (!cards.length) return;

  var bar = document.getElementById('reelBar');
  var sheet = document.getElementById('sheet');
  var sheetBody = sheet && sheet.querySelector('.sheet__body');
  var sheetTitle = sheet && sheet.querySelector('.sheet__head h2');
  var sheetCount = sheet && sheet.querySelector('.sheet__count');

  var LIKE_KEY = 'chikuho:katsudo-liked';
  /* どのカードからでも、活動ページの入口を共有する */
  var SHARE_URL = location.href.split('#')[0];

  /* ── 保存（使えない環境でも落ちないようにする）──────────────────── */

  function readLiked() {
    try {
      return JSON.parse(localStorage.getItem(LIKE_KEY) || '{}') || {};
    } catch (e) {
      return {};   // プライベートモードなどで読めなくても素通りさせる
    }
  }

  function writeLiked(map) {
    try { localStorage.setItem(LIKE_KEY, JSON.stringify(map)); } catch (e) { /* 保存できないだけ */ }
  }

  var liked = readLiked();

  /* ── 数字の見せ方（1000 → 1,000）────────────────────────────────── */

  function comma(n) {
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  /** 1974000 + "K" → "1,974K"。K が付くものは 1 押しても見た目は変わらない */
  function countText(n, unit) {
    return unit === 'K' ? comma(Math.floor(n / 1000)) + 'K' : comma(n);
  }

  /* お礼の吹き出し。押したときだけ出し、少し置いて引っ込める */
  var thankTimer = 0;
  function thank(btn) {
    window.clearTimeout(thankTimer);
    for (var i = 0; i < cards.length; i++) {
      var other = cards[i].querySelector('.rail--like');
      if (other && other !== btn) other.classList.remove('is-thanking');
    }
    btn.classList.add('is-thanking');
    thankTimer = window.setTimeout(function () { btn.classList.remove('is-thanking'); }, 1800);
  }

  /* ── いいね ─────────────────────────────────────────────────────── */

  cards.forEach(function (card) {
    var id = card.id;
    var btn = card.querySelector('.rail--like');
    if (!btn) return;

    var out = btn.querySelector('b');
    var base = Number(btn.getAttribute('data-count')) || 0;
    var unit = btn.getAttribute('data-unit') || '';

    var paint = function () {
      var on = !!liked[id];
      var text = countText(base + (on ? 1 : 0), unit);
      btn.classList.toggle('is-on', on);
      btn.setAttribute('aria-pressed', on ? 'true' : 'false');
      btn.setAttribute('aria-label', (on ? 'いいねを取り消す' : 'いいねする') + '（現在 ' + text + '件）');
      out.textContent = text;
    };

    paint();

    btn.addEventListener('click', function () {
      if (liked[id]) delete liked[id];
      else liked[id] = 1;
      writeLiked(liked);
      paint();
      if (liked[id]) {
        btn.classList.remove('is-burst');
        void btn.offsetWidth;          // アニメーションをやり直させる
        btn.classList.add('is-burst');
        thank(btn);
      } else {
        btn.classList.remove('is-thanking');
      }
    });
  });

  /* ── コメント（解説）のシート ───────────────────────────────────── */

  var lastFocus = null;

  function openSheet(card) {
    if (!sheet) return;
    var src = card.querySelector('.card__sheet');
    if (!src) return;

    lastFocus = document.activeElement;
    sheetBody.innerHTML = src.innerHTML;
    sheetTitle.textContent = card.getAttribute('data-title') || '解説';
    /* 件数はカード側に書いてある。書いていないカードでは何も出さない
       （0 件と出すより、はじめから数えていない方が正直）。 */
    var n = card.querySelector('.rail--comment');
    var cnt = n && n.getAttribute('data-count');
    sheetCount.textContent = cnt ? comma(Number(cnt) || 0) + '件' : '';

    sheet.hidden = false;
    void sheet.offsetWidth;
    sheet.classList.add('is-open');
    sheetBody.scrollTop = 0;
    sheet.querySelector('.sheet__close').focus();
  }

  function closeSheet() {
    if (!sheet || sheet.hidden) return;
    sheet.classList.remove('is-open');
    var done = function () {
      sheet.hidden = true;
      sheet.removeEventListener('transitionend', done);
    };
    sheet.addEventListener('transitionend', done);
    window.setTimeout(done, 420);      // transition が来ない場合の保険
    if (lastFocus && lastFocus.focus) lastFocus.focus();
  }

  if (sheet) {
    sheet.querySelector('.sheet__close').addEventListener('click', closeSheet);
    sheet.querySelector('.sheet__veil').addEventListener('click', closeSheet);
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeSheet();
    });
  }

  cards.forEach(function (card) {
    var btn = card.querySelector('.rail--comment');
    if (btn) btn.addEventListener('click', function () { openSheet(card); });
  });

  /* ── シェア ─────────────────────────────────────────────────────── */

  cards.forEach(function (card) {
    var btn = card.querySelector('.rail--share');
    if (!btn) return;

    btn.addEventListener('click', function () {
      var url = SHARE_URL;
      var title = card.getAttribute('data-title') || document.title;
      var data = { title: title, text: title + '｜筑豊市民大学', url: url };

      // 共有シートを出せる端末はそれに任せる
      if (navigator.share && (!navigator.canShare || navigator.canShare(data))) {
        navigator.share(data).catch(function () { /* 取り消しただけ */ });
        return;
      }
      // 出せない場合はリンクを写す
      var told = function (ok) {
        var out = btn.querySelector('b');
        var was = out.textContent;
        out.textContent = ok ? 'コピー済' : 'コピー失敗';
        window.setTimeout(function () { out.textContent = was; }, 1600);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(url).then(function () { told(true); }, function () { told(false); });
      } else {
        told(false);
      }
    });
  });

  /* ── いま何枚目か ───────────────────────────────────────────────── */

  var dots = bar ? [].slice.call(bar.children) : [];
  var current = -1;

  function mark(i) {
    if (i === current || i < 0 || i >= cards.length) return;
    current = i;
    for (var k = 0; k < dots.length; k++) dots[k].classList.toggle('is-on', k <= i);
    if (bar) bar.setAttribute('aria-valuenow', String(i + 1));
    /* 戻る操作でも同じカードに戻れるようにする（履歴は増やさない）。
       1 枚目のときは # を付けない。開いただけで URL が
       achievements.html#top に変わると、そのまま貼られてしまうため。 */
    var id = cards[i].id;
    if (i === 0) {
      if (location.hash) history.replaceState(null, '', location.pathname + location.search);
    } else if (id && location.hash.slice(1) !== id) {
      history.replaceState(null, '', '#' + id);
    }
  }

  if ('IntersectionObserver' in window) {
    var io = new IntersectionObserver(function (entries) {
      // いちばん大きく映っているカードを「いま」とみなす
      var best = null;
      for (var i = 0; i < entries.length; i++) {
        if (!entries[i].isIntersecting) continue;
        if (!best || entries[i].intersectionRatio > best.intersectionRatio) best = entries[i];
      }
      if (best) mark(cards.indexOf(best.target));
    }, { root: reel, threshold: [0.5, 0.75] });
    cards.forEach(function (c) { io.observe(c); });
  } else {
    reel.addEventListener('scroll', function () {
      mark(Math.round(reel.scrollTop / reel.clientHeight));
    }, { passive: true });
  }

  /* ── キーボードと、開いたときの位置合わせ ───────────────────────── */

  /* 端まで来たら反対の端へ回す。
     ぐるっと戻る様子が見えないよう、行き先へは一瞬で飛ばし、
     その前後だけ軽く暗くして「1 枚送った」ように見せる。 */
  function wrapTo(i) {
    reel.classList.add('is-wrapping');
    window.setTimeout(function () {
      reel.scrollTo({ top: i * reel.clientHeight, behavior: 'auto' });
      mark(i);
      window.setTimeout(function () { reel.classList.remove('is-wrapping'); }, 30);
    }, 170);
  }

  function goTo(i, smooth) {
    var last = cards.length - 1;
    if (i > last) return wrapTo(0);
    if (i < 0) return wrapTo(last);
    reel.scrollTo({ top: i * reel.clientHeight, behavior: smooth ? 'smooth' : 'auto' });
  }

  /* 指でのスワイプは CSS のスナップが受け持つので、
     最後のカードで下向きに払われたことを別に拾って先頭へ回す。 */
  var atEnd = function () {
    return reel.scrollTop + reel.clientHeight >= reel.scrollHeight - 2;
  };
  var atStart = function () { return reel.scrollTop <= 2; };
  var wheelLock = 0;

  reel.addEventListener('wheel', function (e) {
    if (reel.classList.contains('is-wrapping')) return;
    var now = Date.now();
    if (now - wheelLock < 700) return;
    if (e.deltaY > 12 && atEnd()) { wheelLock = now; wrapTo(0); }
    else if (e.deltaY < -12 && atStart()) { wheelLock = now; wrapTo(cards.length - 1); }
  }, { passive: true });

  var touchY = 0;
  reel.addEventListener('touchstart', function (e) {
    touchY = e.touches[0].clientY;
  }, { passive: true });

  reel.addEventListener('touchend', function (e) {
    if (reel.classList.contains('is-wrapping')) return;
    var dy = touchY - (e.changedTouches[0] ? e.changedTouches[0].clientY : touchY);
    if (dy > 60 && atEnd()) wrapTo(0);
    else if (dy < -60 && atStart()) wrapTo(cards.length - 1);
  }, { passive: true });

  var prevBtn = document.getElementById('reelPrev');
  var nextBtn = document.getElementById('reelNext');
  var step = function (d) {
    return function () { goTo(Math.round(reel.scrollTop / reel.clientHeight) + d, true); };
  };
  if (prevBtn) prevBtn.addEventListener('click', step(-1));
  if (nextBtn) nextBtn.addEventListener('click', step(1));

  document.addEventListener('keydown', function (e) {
    if (sheet && !sheet.hidden) return;
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    var i = Math.round(reel.scrollTop / reel.clientHeight);
    if (e.key === 'ArrowDown' || e.key === 'PageDown') { e.preventDefault(); goTo(i + 1, true); }
    else if (e.key === 'ArrowUp' || e.key === 'PageUp') { e.preventDefault(); goTo(i - 1, true); }
    else if (e.key === 'Home') { e.preventDefault(); goTo(0, true); }
    else if (e.key === 'End') { e.preventDefault(); goTo(cards.length - 1, true); }
  });

  /* ── 開いた直後の案内 ───────────────────────────────────────────── */

  /* 指で送れることが伝わらないという声があったので、表紙にいる間だけ
     2 秒ほど案内を出し、カードを軽く上下に揺らす。
     触ったらその時点で消す（邪魔をしない）。 */
  /* 会員は 70 代が中心。2 秒では読み終わらないという声があったので、
     少し長めに出しておく。触ればその時点で消える。 */
  var HINT_MS = 3500;
  var hintTimer = 0;

  function stopHint() {
    if (!hintTimer) return;
    window.clearTimeout(hintTimer);
    hintTimer = 0;
    cards[0].classList.remove('is-hinting');
  }

  function startHint() {
    if (location.hash || reel.scrollTop >= 4) return;
    cards[0].classList.add('is-hinting');
    hintTimer = window.setTimeout(stopHint, HINT_MS);
    ['pointerdown', 'wheel', 'touchstart', 'keydown'].forEach(function (ev) {
      window.addEventListener(ev, stopHint, { once: true, passive: true });
    });
    reel.addEventListener('scroll', stopHint, { once: true, passive: true });
  }

  // #jinko のように名指しで開かれたときは、そのカードから始める
  var wanted = location.hash.slice(1);
  if (wanted) {
    var at = cards.map(function (c) { return c.id; }).indexOf(wanted);
    if (at > 0) goTo(at, false);
  }
  mark(Math.round(reel.scrollTop / reel.clientHeight));
  startHint();

  // 端末を回したときに中途半端な位置で止まらないようにする
  var fixTimer = 0;
  window.addEventListener('resize', function () {
    window.clearTimeout(fixTimer);
    var i = current;
    fixTimer = window.setTimeout(function () { goTo(i, false); }, 120);
  }, { passive: true });

  /* ── 押した手応え（タブバーや SNS アイコンと同じ作り）───────────── */

  var pressed = 0;
  var timer = 0;
  reel.addEventListener('pointerdown', function (e) {
    var t = e.target.closest && e.target.closest('.rail, .card__user, .card__goBtn');
    if (!t) return;
    window.clearTimeout(timer);
    pressed = Date.now();
    t.classList.add('is-pressed');
  });
  var release = function (e) {
    var t = e.target.closest && e.target.closest('.rail, .card__user, .card__goBtn');
    if (!t) return;
    window.clearTimeout(timer);
    timer = window.setTimeout(function () {
      t.classList.remove('is-pressed');
    }, Math.max(0, 160 - (Date.now() - pressed)));
  };
  reel.addEventListener('pointerup', release);
  reel.addEventListener('pointercancel', release);
})();
