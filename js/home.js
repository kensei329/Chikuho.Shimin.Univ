/**
 * ホーム — 年会費のその場計算
 * ------------------------------------------------------------------------
 * 1講座 10,000円。2講座目からは 1 講座ごとに 3,000円。
 * 4講座すべてで 19,000円が上限。
 *
 *   1講座 10,000 ／ 2講座 13,000 ／ 3講座 16,000 ／ 4講座 19,000
 */
(function () {
  'use strict';

  var boxes = [].slice.call(document.querySelectorAll('.calc__picks input[type="checkbox"]'));
  var yen = document.getElementById('feeYen');
  var note = document.getElementById('feeNote');
  if (!boxes.length || !yen || !note) return;

  var FIRST = 10000;
  var MORE = 3000;
  var CAP = 19000;

  function fee(n) {
    if (n <= 0) return 0;
    return Math.min(FIRST + (n - 1) * MORE, CAP);
  }

  function comma(n) {
    return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ',');
  }

  function update() {
    var n = 0;
    boxes.forEach(function (b) {
      if (b.checked) n++;
      /* :has() が使えない環境でも選んだことが分かるようにする */
      var row = b.closest('.pick');
      if (row) row.classList.toggle('is-on', b.checked);
    });

    yen.textContent = '¥' + comma(fee(n));
    note.textContent =
      n === 0 ? '受けたい会をえらんでください。' :
      n === 4 ? '4講座すべて。これが上限の金額です。' :
                n + '講座。もう1講座ふやすと +' + comma(MORE) + '円です。';
  }

  boxes.forEach(function (b) { b.addEventListener('change', update); });
  update();   // 戻ってきたときにブラウザが選択を覚えていることがある
})();
