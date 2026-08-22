/**
 * 政策ページ — Instagram のストーリー風
 * ------------------------------------------------------------------------
 * 実物に寄せた点（2026 年時点の仕様を調べたうえで）:
 *   ・1 枚あたり 5 秒で自動的に次へ（写真ストーリーの既定値）
 *   ・上部のゲージは本数ぶんに割り、いま見ているものだけが伸びる
 *   ・画面の右 1/3 をタップで次へ、左 1/3 で前へ
 *   ・押しっぱなしで一時停止し、そのあいだ上下の飾りを引っ込める
 *   ・左右のスワイプでも送れる
 *
 * ■ 止められるようにしておく
 *   勝手に進む見せ物は、止める手立てがないと WCAG 2.2.2 を満たせない。
 *   長押しは知らないと気づけないので、上に停止ボタンも出している。
 *   動きを減らす設定の環境では、最初から自動送りをしない。
 *
 * ■ ゲージの動かし方
 *   毎フレーム幅を書き換えるのではなく、CSS アニメーションに
 *   transform: scaleX を任せて animation-play-state で止める。
 *   合成だけで済むので、めくっている最中でも引っかからない。
 */
(function () {
  'use strict';

  var story = document.getElementById('story');
  if (!story) return;

  var track = document.getElementById('storyTrack');
  var slides = [].slice.call(track.querySelectorAll('.slide'));
  if (!slides.length) return;

  /* ゲージは JavaScript が動かない環境のために HTML に直接書いてある。
     枚数を足し引きしたときに書き忘れても崩れないよう、ここで数を合わせる。 */
  var barBox = document.querySelector('.story__bars');
  if (barBox) {
    while (barBox.children.length > slides.length) barBox.removeChild(barBox.lastElementChild);
    while (barBox.children.length < slides.length) {
      var b = document.createElement('div');
      b.className = 'story__bar';
      b.appendChild(document.createElement('i'));
      barBox.appendChild(b);
    }
  }
  var bars = [].slice.call(document.querySelectorAll('.story__bar'));
  var sect = document.getElementById('storySect');
  var pauseBtn = document.getElementById('storyPause');
  var heart = document.getElementById('storyHeart');
  var share = document.getElementById('storyShare');

  var STEP_MS = 5000;                       // 実物の写真ストーリーと同じ
  var GUIDE_MS = 2600;                      // 開いた直後の案内を出しておく長さ
  var guideTimer = 0;
  /* ストーリーは会ごとにページが分かれる。共有はいま開いているページを指し、
     いいねの記録も会ごとに分けて持つ。 */
  var SHARE_URL = location.href.split('#')[0];
  var LIKE_KEY = 'chikuho:kai-liked:' + (document.body.getAttribute('data-kai') || 'x');

  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  var index = 0;
  var timer = 0;
  var paused = reduce;                      // 動きを抑える設定なら最初から止めておく

  /* ── 位置の持ち方 ─────────────────────────────────────────────────
     以前は横スクロール＋scroll-snap で送り、その上に立方体の回転を
     重ねていた。ところが回転のために各スライドへ transform を掛けると、
     ブラウザがスナップの目印にしているのはその「変形後の箱」なので、
     どこが 1 枚分なのかを見失う。実際に測ると、2 枚目の左端が 390px の
     はずのところ -107px になっていた。結果、指を離したあとに半端な位置で
     止まることがあった。

     そこで、送りをスクロールに任せるのをやめた。いまは
     スライドを重ねて置き、progress（何枚目かを小数で持つ）だけを動かす。
     毎コマ書き換えるのは transform だけなので合成だけで済み、
     離したときは必ず整数の位置へ寄せるので途中で止まらない。

     JavaScript が動かない環境では、CSS の既定（横並び＋スナップ）のまま
     指で送れる。track に is-live が付いたときだけ重ねる形に変わる。 */

  var progress = 0;          // 0 = 1 枚目、0.5 = 1 枚目と 2 枚目の中間
  var SETTLE = 'transform .34s cubic-bezier(.22, .61, .36, 1)';
  var shown = -1;            // いま出しているスライドの範囲（付け外しを減らす）

  track.classList.add('is-live');

  function width() { return track.clientWidth || 1; }

  /**
   * @param {boolean} anim  なめらかに動かすか
   * @param {boolean} cube  立方体として回すか（指でなぞったときだけ true）
   */
  function render(anim, cube) {
    var w = width();
    var near = Math.round(progress);
    if (near !== shown) {
      shown = near;
      for (var k = 0; k < slides.length; k++) {
        slides[k].style.visibility = Math.abs(k - near) <= 1 ? '' : 'hidden';
      }
    }
    for (var i = 0; i < slides.length; i++) {
      var d = i - progress;
      if (d < -1.001 || d > 1.001) continue;
      var el = slides[i];
      el.style.transition = anim ? SETTLE : 'none';
      if (cube) {
        el.style.transformOrigin = d >= 0 ? '0% 50%' : '100% 50%';
        el.style.transform =
          'translateX(' + (d * w).toFixed(2) + 'px) rotateY(' + (-90 * d).toFixed(2) + 'deg)';
      } else {
        el.style.transformOrigin = '50% 50%';
        el.style.transform = 'translateX(' + (d * w).toFixed(2) + 'px)';
      }
    }
  }

  /* ── ゲージ ─────────────────────────────────────────────────────── */

  function paintBars() {
    for (var i = 0; i < bars.length; i++) {
      var fill = bars[i].firstElementChild;
      bars[i].classList.toggle('is-done', i < index);
      bars[i].classList.remove('is-live');
      if (i === index) {
        // 一度外してから付け直さないとアニメーションが最初から流れない
        fill.style.animation = 'none';
        void fill.offsetWidth;
        fill.style.animation = '';
        bars[i].classList.add('is-live');
        fill.style.animationDuration = (STEP_MS / 1000) + 's';
      }
    }
  }

  /* ── 送り ───────────────────────────────────────────────────────── */

  function apply() {
    paintBars();
    if (sect) sect.textContent = slides[index].getAttribute('data-sect') || '';
    story.setAttribute('aria-label', '政策 ' + (index + 1) + ' / ' + slides.length);
    for (var i = 0; i < slides.length; i++) {
      // 見えていないものは読み上げにも出さない
      slides[i].setAttribute('aria-hidden', i === index ? 'false' : 'true');
    }
    if (typeof paintHeart === 'function') paintHeart();
    restart();
  }

  /**
   * @param {number} i     行き先
   * @param {boolean} anim なめらかに動かすか
   * @param {boolean} cube 立方体として回すか
   */
  /* ── 写真の読み込み ─────────────────────────────────────────────
     17 枚を最初にまとめて取りにいくと 1.3MB になり、1 枚目が出るのが
     その分だけ遅れる。いま見ている枚の前後だけ入れておけば、送りに
     追いつくうえ、開いた直後に取りにいくのは 1 枚ぶんで済む。
     JavaScript が動かない環境には、HTML 側に noscript の控えがある。 */
  function loadPhotosAround(i) {
    // 端は回り込ませない。開いた直後に i-1 を回り込ませると、
    // まだ誰も見ていない最後の 1 枚を取りにいってしまう。
    for (var k = Math.max(0, i - 1); k <= Math.min(slides.length - 1, i + 2); k++) {
      var slide = slides[k];
      if (!slide) continue;
      var img = slide.querySelector('img[data-src]');
      if (!img) continue;
      img.src = img.getAttribute('data-src');
      img.removeAttribute('data-src');
    }
  }

  /* 前後だけでは、回線が遅いと送りに追いつけず白いまま出てしまう。
     いまの枚が出そろって手が空いたら、残りも 1 枚ずつ静かに集める。
     こうしておけば、どこへ飛んでも待たされない。 */
  function fillRestSlowly() {
    var rest = [].slice.call(track.querySelectorAll('img[data-src]'));
    (function next() {
      var img = rest.shift();
      if (!img) return;
      img.fetchPriority = 'low';
      img.onload = img.onerror = next;
      img.src = img.getAttribute('data-src');
      img.removeAttribute('data-src');
    })();
  }

  function goTo(i, anim, cube) {
    var last = slides.length - 1;
    var wrap = i < 0 || i > last;           // 端を越えたら反対の端へ
    index = wrap ? (i < 0 ? last : 0) : i;
    progress = index;
    loadPhotosAround(index);
    render(anim && !wrap, cube && !wrap);
    apply();
  }

  /* タップ・キー・自動送りは、実物と同じで「パッと切り替わる」。
     なめらかに滑らせるのは、指でなぞったときだけ（そのときは
     endDrag が anim と cube を立てて呼ぶ）。 */
  function next() { goTo(index + 1, false, false); }
  function prev() { goTo(index - 1, false, false); }

  function restart() {
    window.clearTimeout(timer);
    if (paused) return;
    timer = window.setTimeout(next, STEP_MS);
  }

  /* 開いた直後の案内。出ているあいだの最初の操作は「閉じるだけ」で、
     送りには使わない。案内を消そうとしただけで次の枚へ飛ぶと戸惑う。 */
  function guiding() { return guideTimer !== 0; }

  function stopGuide() {
    if (!guideTimer) return;
    window.clearTimeout(guideTimer);
    guideTimer = 0;
    story.classList.remove('is-guiding');
    if (!reduce && !paused) restart();
  }

  function setPaused(on) {
    paused = on;
    story.classList.toggle('is-paused', on);
    if (pauseBtn) {
      pauseBtn.setAttribute('aria-pressed', on ? 'true' : 'false');
      pauseBtn.setAttribute('aria-label', on ? '自動送りを再開する' : '自動送りを止める');
    }
    if (on) window.clearTimeout(timer);
    else restart();
  }

  if (pauseBtn) pauseBtn.addEventListener('click', function () { setPaused(!paused); });

  /* ── 指の操作（タップ・長押し・横スワイプ）─────────────────────
     以前は左右 34% に透明な要素を重ねてタップを受けていたが、その要素は
     スクロールできる箱の外側にあるため、その上で始めたなぞりが届かず、
     画面の両端ではスワイプできなかった。いまは track の上で見分ける。
       ・10px 以上動いた → なぞり（指について立方体が回る）
       ・動かないまま 220ms 以上 → 長押し（一時停止）
       ・どちらでもない → タップ（右 1/3 で次、左 1/3 で前） */

  var HOLD_MS = 220;
  var MOVE_PX = 10;
  var FLICK_MS = 300;        // これより速く払ったら、距離が短くても送る
  var holdTimer = 0;
  var holding = false;
  var moved = false;
  var down = false;
  var wasPaused = false;
  var startX = 0;
  var startY = 0;
  var startAt = 0;
  var startTime = 0;

  function inChrome(el) {
    // .slide__go はページを移るボタン。ここを押したときに送ってしまうと、
    // 移る前に次の 1 枚がちらついて見える。
    return !!(el && el.closest && el.closest('.story__foot, .story__head, .slide__go'));
  }

  function releaseHold() {
    if (!holding) return;
    holding = false;
    story.classList.remove('is-holding');
    if (!wasPaused) setPaused(false);
  }

  track.addEventListener('pointerdown', function (e) {
    if (inChrome(e.target)) return;
    // 案内が出ているあいだの 1 回目は、閉じるだけ。ここで送ってしまうと
    // 「案内を消そうとしただけなのに次へ行った」ことになる。
    if (guiding()) { stopGuide(); return; }
    down = true;
    holding = false;
    moved = false;
    wasPaused = paused;
    startX = e.clientX;
    startY = e.clientY;
    startAt = progress;
    startTime = Date.now();
    window.clearTimeout(holdTimer);
    holdTimer = window.setTimeout(function () {
      if (moved) return;
      holding = true;
      story.classList.add('is-holding');
      setPaused(true);
    }, HOLD_MS);
  });

  track.addEventListener('pointermove', function (e) {
    // 触れていないときの動きは見ない（触れる前のカーソル移動で
    // 「なぞった」と誤って判定してしまうため）
    if (!down) return;
    var dx = e.clientX - startX;
    if (!moved) {
      if (Math.abs(dx) <= MOVE_PX && Math.abs(e.clientY - startY) <= MOVE_PX) return;
      moved = true;
      window.clearTimeout(holdTimer);
      releaseHold();
      window.clearTimeout(timer);          // なぞっている間は進めない
      track.classList.add('is-cube');
    }
    // 端では引っぱられる感じを出す（行き止まりが分かる）
    var raw = startAt - dx / width();
    if (raw < 0) raw = raw / 3;
    else if (raw > slides.length - 1) raw = (slides.length - 1) + (raw - (slides.length - 1)) / 3;
    progress = raw;
    render(false, true);
  }, { passive: true });

  function endDrag(e) {
    window.clearTimeout(holdTimer);
    if (!down) return;
    down = false;

    if (holding) { releaseHold(); return; }

    if (moved) {
      // 半分を越えたか、素早く払ったら送る。どちらでもなければ元に戻す
      var dx = (e && typeof e.clientX === 'number' ? e.clientX : startX) - startX;
      var quick = Date.now() - startTime < FLICK_MS && Math.abs(dx) > 24;
      var moveBy = progress - startAt;
      var step = 0;
      if (quick) step = dx < 0 ? 1 : -1;
      else if (Math.abs(moveBy) > 0.5) step = moveBy > 0 ? 1 : -1;

      var to = Math.round(startAt) + step;
      var last = slides.length - 1;
      if (to < 0) to = 0;                 // なぞりでは端を越えて回さない
      if (to > last) to = last;
      index = to;
      progress = to;
      render(true, true);
      apply();
      // 回転を戻すのは、寄せ終わってから
      window.setTimeout(function () { track.classList.remove('is-cube'); render(false, false); }, 360);
      return;
    }

    if (inChrome(e.target)) return;
    var r = track.getBoundingClientRect();
    if (e.clientX - r.left < r.width / 3) prev();
    else next();
  }

  track.addEventListener('pointerup', endDrag);

  track.addEventListener('pointercancel', function () {
    window.clearTimeout(holdTimer);
    if (!down) return;
    down = false;
    releaseHold();
    if (moved) {
      index = Math.round(startAt);
      progress = index;
      render(true, true);
      window.setTimeout(function () { track.classList.remove('is-cube'); render(false, false); }, 360);
      apply();
    }
  });

  // 読み上げ・キーボード用（画面には出さない）と、PC でカラムの外に出す送り
  ['storyPrevSr', 'storySidePrev'].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener('click', function () { if (guiding()) { stopGuide(); return; } prev(); });
  });
  ['storyNextSr', 'storySideNext'].forEach(function (id) {
    var el = document.getElementById(id);
    if (el) el.addEventListener('click', function () { if (guiding()) { stopGuide(); return; } next(); });
  });

  /* ── キーボード ─────────────────────────────────────────────────── */

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { closeThanks(); return; }
    if (e.target && /^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) return;
    if (guiding() && /^(ArrowRight|ArrowLeft|ArrowUp|ArrowDown| |Spacebar|Enter)$/.test(e.key)) {
      e.preventDefault(); stopGuide(); return;
    }
    if (e.key === 'ArrowRight') { e.preventDefault(); next(); }
    else if (e.key === 'ArrowLeft') { e.preventDefault(); prev(); }
    else if (e.key === ' ' || e.key === 'Spacebar') { e.preventDefault(); setPaused(!paused); }
  });

  /* ── ハートと共有 ───────────────────────────────────────────────── */

  function readLiked() {
    try { return JSON.parse(localStorage.getItem(LIKE_KEY) || '{}') || {}; }
    catch (e) { return {}; }
  }

  var liked = readLiked();

  function paintHeart() {
    if (!heart) return;
    var on = !!liked[slides[index].id];
    heart.classList.toggle('is-on', on);
    heart.setAttribute('aria-pressed', on ? 'true' : 'false');
    heart.setAttribute('aria-label', on ? 'いいねを取り消す' : 'いいねする');
  }

  var thankTimer = 0;

  if (heart) {
    heart.addEventListener('click', function () {
      var id = slides[index].id;
      if (liked[id]) delete liked[id]; else liked[id] = 1;
      try { localStorage.setItem(LIKE_KEY, JSON.stringify(liked)); } catch (e) { /* 残せないだけ */ }
      paintHeart();
      // 押したときだけお礼の吹き出しを出す（取り消したときは出さない）
      window.clearTimeout(thankTimer);
      if (liked[id]) {
        heart.classList.add('is-thanking');
        thankTimer = window.setTimeout(function () { heart.classList.remove('is-thanking'); }, 1800);
      } else {
        heart.classList.remove('is-thanking');
      }
    });
  }

  if (share) {
    share.addEventListener('click', function () {
      var title = (document.body.getAttribute('data-kai-name') || '分科会') + '｜筑豊市民大学';
      var data = { title: title, text: title, url: SHARE_URL };
      if (navigator.share && (!navigator.canShare || navigator.canShare(data))) {
        navigator.share(data).catch(function () { /* 取り消しただけ */ });
        return;
      }
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(SHARE_URL).then(function () {
          share.setAttribute('aria-label', 'リンクをコピーしました');
          window.setTimeout(function () { share.setAttribute('aria-label', '共有する'); }, 1600);
        }, function () { /* 写せないだけ */ });
      }
    });
  }

  /* ── メッセージ ─────────────────────────────────────────────────
     送り先は無い（サーバーを持たない静的サイト）。送ったつもりで
     終わらせないよう、お礼とあわせてその旨も出している。 */

  var msgForm = document.getElementById('storyMsg');
  var msgIn = document.getElementById('storyMsgIn');
  var thanks = document.getElementById('thanks');
  var thanksEcho = document.getElementById('thanksEcho');
  var thanksTimer = 0;

  function closeThanks() {
    if (!thanks || thanks.hidden) return;
    thanks.classList.remove('is-open');
    window.setTimeout(function () { thanks.hidden = true; }, 300);
    if (!wasPaused) setPaused(false);
  }

  function openThanks(text) {
    if (!thanks) return;
    wasPaused = paused;
    setPaused(true);
    thanksEcho.textContent = text ? '「' + text + '」' : '';
    thanks.hidden = false;
    void thanks.offsetWidth;
    thanks.classList.add('is-open');
    window.clearTimeout(thanksTimer);
    thanksTimer = window.setTimeout(closeThanks, 3600);
  }

  if (thanks) {
    thanks.addEventListener('click', function () {
      window.clearTimeout(thanksTimer);
      closeThanks();
    });
  }

  if (msgForm && msgIn) {
    // 入力しているあいだは勝手に進めない
    msgIn.addEventListener('focus', function () { wasPaused = paused; setPaused(true); });
    msgIn.addEventListener('blur', function () { if (!msgIn.value.trim() && !wasPaused) setPaused(false); });
    msgIn.addEventListener('input', function () {
      msgForm.classList.toggle('is-filled', !!msgIn.value.trim());
    });
    msgForm.addEventListener('submit', function (e) {
      e.preventDefault();
      var text = msgIn.value.trim();
      if (!text) { msgIn.focus(); return; }
      msgIn.value = '';
      msgForm.classList.remove('is-filled');
      msgIn.blur();
      openThanks(text.slice(0, 60));
    });
  }

  /* ── 裏に回ったら止める ─────────────────────────────────────────── */

  document.addEventListener('visibilitychange', function () {
    if (document.hidden) window.clearTimeout(timer);
    else restart();
  });

  window.addEventListener('resize', function () {
    // 幅が変わったら、いまの 1 枚をぴったり合わせ直す
    progress = index;
    render(false, false);
  }, { passive: true });

  /* ── 起動 ───────────────────────────────────────────────────────── */

  var wanted = location.hash.slice(1);
  var at = wanted ? slides.map(function (s) { return s.id; }).indexOf(wanted) : -1;
  goTo(at > 0 ? at : 0, false, false);
  if (reduce) setPaused(true);

  /* ── 開いた直後の案内 ───────────────────────────────────────────── */

  /* これがストーリーで、横に送れるものだと分からないという声があった。
     最初の 1 枚にいるあいだだけ案内を出し、そのあいだ自動送りは止める。
     触られた時点で引っ込め、そこから再生を始める。 */
  // 読み終えて手が空いたら、残りの写真を静かに集めておく
  (function () {
    var started = false;
    var go = function () {
      if (started) return;
      started = true;
      if (window.requestIdleCallback) window.requestIdleCallback(fillRestSlowly, { timeout: 800 });
      else window.setTimeout(fillRestSlowly, 300);
    };
    if (document.readyState === 'complete') go();
    else window.addEventListener('load', go);
    /* 外部の読み込みが詰まると load は来ない。それでも白いままにしないため、
       1.2 秒たったら自分から集めはじめる（優先度は低いまま）。 */
    window.setTimeout(go, 1200);
  })();

  if (at <= 0 && document.getElementById('storyGuide')) {
    story.classList.add('is-guiding');
    window.clearTimeout(timer);           // 案内を読むあいだは進めない
    guideTimer = window.setTimeout(stopGuide, GUIDE_MS);
    // 画面の外（ヘッダーやタブバー）を触ったときにも閉じる
    ['wheel', 'keydown'].forEach(function (ev) {
      window.addEventListener(ev, stopGuide, { once: true, passive: true });
    });
  }
})();
