/**
 * 年間予定 — スクロールに合わせて、ぽこっと出す
 * ------------------------------------------------------------------------
 * 画面に入った月から順に is-in を付ける。動き自体は CSS のアニメーション
 * （ばね状の曲線で、行きすぎてから戻る）に任せる。
 *
 * 一度出したら消さない。行ったり来たりするたびに出たり消えたりすると、
 * 読んでいるあいだ落ち着かないため。
 */
(function () {
  'use strict';

  var months = [].slice.call(document.querySelectorAll('.mo'));
  if (!months.length) return;

  /* 見られない環境では、はじめから出しておく */
  if (!('IntersectionObserver' in window)) {
    months.forEach(function (el) { el.classList.add('is-in'); });
    return;
  }

  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (!e.isIntersecting) return;
      e.target.classList.add('is-in');
      io.unobserve(e.target);
    });
  }, {
    /* 下から 12% ぶん手前で出しはじめる。画面の端ちょうどで動くと、
       出てくる途中が見切れてしまう。 */
    rootMargin: '0px 0px -12% 0px',
    threshold: 0.15
  });

  months.forEach(function (el) { io.observe(el); });

  /* サイトマップから「11月」のように月を指して開いたとき。
     指した月まで一気に飛ぶので、その手前の月は一度も画面に入らない。
     観察は画面に入ったときしか動かないので、そのままだと透明のまま
     残ってしまう。指して開いたときだけ、全部出したうえで始める。 */
  /* hash はアドレス欄から来る。CSS の選び方として通らない字が混じっていると
     querySelector はその場で例外を投げ、ここから下が動かなくなる。
     id をそのまま探す getElementById なら、どんな字が来ても投げない。 */
  var wanted = decodeURIComponent(location.hash.slice(1) || '');
  if (wanted && document.getElementById(wanted)) {
    months.forEach(function (el) { el.classList.add('is-in'); io.unobserve(el); });
  }
})();
