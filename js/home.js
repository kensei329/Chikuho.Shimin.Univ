/**
 * ホーム — 入会申し込みフォーム
 * ------------------------------------------------------------------------
 * 見た目は自前。送り先は Google フォームだが、直接は投げない。
 * いったん同じサイトの /api/join に渡し、そこから Google へ送ってもらう。
 *
 * ■ なぜ間に一枚はさむのか
 *   ブラウザから Google へじかに投げる方式は、中身が一字一句正しくても
 *   受け取ってもらえなかった。しかも Google は別ドメインからの読み取りを
 *   許さないので、弾かれたことすら分からず、画面には「受け付けました」と
 *   出てしまう。会員の多くは 70 代で、申し込んだつもりで届いていない事態が
 *   いちばん困る。
 *
 *   サーバー同士のやりとりならブラウザの制限がかからず、Google の返事も
 *   読める。届いたと確かめてから、はじめて受付の画面を出す。
 *
 * ■ 選択肢の文字は、一字一句そろえる必要がある
 *   チェックボックスは、フォームに無い文字列を送ると弾かれる。
 *   HTML の value を書き換えるときは、必ずフォーム側と見比べること。
 */
(function () {
  'use strict';

  var form = document.getElementById('jform');
  if (!form) return;

  var FIRST = 10000;   /* 1講座 */
  var MORE  = 3000;    /* 2講座目から、1講座ごとに */
  var CAP   = 19000;   /* 4講座での上限 */

  var kais   = [].slice.call(form.querySelectorAll('input[name="entry.939298798"]'));
  var yen    = document.getElementById('feeYen');
  var note   = document.getElementById('feeNote');
  var feeOut = document.getElementById('jf-fee');

  var name  = document.getElementById('jf-name');
  var mail  = document.getElementById('jf-mail');
  var tel   = document.getElementById('jf-tel');
  var agree  = document.getElementById('jf-agree');
  var agree2 = document.getElementById('jf-agree2');

  var alertBox = document.getElementById('jf-alert');
  var sendBtn  = document.getElementById('jf-send');
  var done     = document.getElementById('jdone');

  function comma(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }

  /* ── 年会費 ─────────────────────────────────────────────────────── */

  function fee(n) { return n <= 0 ? 0 : Math.min(FIRST + (n - 1) * MORE, CAP); }

  function paintFee() {
    var n = 0;
    kais.forEach(function (b) {
      if (b.checked) n++;
      var row = b.closest('.pick');
      if (row) row.classList.toggle('is-on', b.checked);   /* :has() が無い環境の控え */
    });

    var amount = fee(n);
    yen.textContent = '¥' + comma(amount);
    note.textContent =
      n === 0 ? '受けたい会をえらんでください。' :
      n === 4 ? '4講座すべて。これが上限の金額です。' :
                n + '講座。もう1講座ふやすと +' + comma(MORE) + '円です。';

    /* 画面に出ている金額を、そのまま送る */
    feeOut.value = n === 0 ? '' : '¥' + comma(amount) + '（' + n + '講座）';
  }

  kais.forEach(function (b) { b.addEventListener('change', paintFee); });

  /* ── 入力の確かめ ───────────────────────────────────────────────── */

  var MAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

  function setErr(el, id, msg) {
    var box = document.getElementById(id);
    if (box) {
      box.textContent = msg || '';
      box.hidden = !msg;
    }
    if (el) el.setAttribute('aria-invalid', msg ? 'true' : 'false');
    return !msg;
  }

  function checkName()  { return setErr(name, 'jf-name-err', name.value.trim() ? '' : 'お名前をご記入ください。'); }
  function checkMail()  {
    var v = mail.value.trim();
    if (!v) return setErr(mail, 'jf-mail-err', 'メールアドレスをご記入ください。');
    if (!MAIL_RE.test(v)) return setErr(mail, 'jf-mail-err', 'メールアドレスの形をご確認ください（例：taro@example.com）。');
    return setErr(mail, 'jf-mail-err', '');
  }
  function checkTel() {
    var v = tel.value.trim();
    if (!v) return setErr(tel, 'jf-tel-err', '電話番号をご記入ください。');
    /* 数字が 10 桁に満たないものは、書き間違いとみなす */
    if ((v.replace(/\D/g, '')).length < 10) return setErr(tel, 'jf-tel-err', '電話番号をご確認ください（市外局番からご記入ください）。');
    return setErr(tel, 'jf-tel-err', '');
  }
  function checkAgree() { return setErr(agree, 'jf-agree-err', agree.checked ? '' : 'ご確認のうえ、チェックを入れてください。'); }
  function checkKai() {
    var any = kais.some(function (b) { return b.checked; });
    return setErr(null, 'jf-kai-err', any ? '' : '入会を希望する会を、1つ以上えらんでください。');
  }

  /* 赤い印が全部消えたら、上の注意書きも引っこめる。
     直したのに警告が出たままだと、どこが悪いのか分からなくなる。 */
  function sweep() {
    if (alertBox.hidden) return;
    if (!form.querySelector('.jf__err:not([hidden])')) alertBox.hidden = true;
  }

  /* 触ったところから順に、その場で直せるようにする */
  [[name, checkName], [mail, checkMail], [tel, checkTel]].forEach(function (p) {
    p[0].addEventListener('blur', function () { p[1](); sweep(); });
    p[0].addEventListener('input', function () {
      if (p[0].getAttribute('aria-invalid') === 'true') { p[1](); sweep(); }
    });
  });
  /* 確認事項は、フォーム側では2つの選択肢に分かれている。
     画面では1つのチェックにまとめてあるので、送るときだけ2つに戻す。
     外した項目は disabled にしておけば、送信の中身に入らない。 */
  function syncAgree() { agree2.disabled = !agree.checked; }

  agree.addEventListener('change', function () { syncAgree(); checkAgree(); sweep(); });
  kais.forEach(function (b) { b.addEventListener('change', function () { checkKai(); sweep(); }); });

  /* ── 送信 ───────────────────────────────────────────────────────── */

  var sending = false;

  /* 同じ欄が何度も出る（希望する会、確認事項）ので、組の並びのまま渡す */
  function collect() {
    var out = [];
    new FormData(form).forEach(function (v, k) { out.push([k, String(v)]); });
    return out;
  }

  function showDone() {
    form.hidden = true;
    done.hidden = false;
    done.scrollIntoView({ block: 'center' });
  }

  function showFailed() {
    sending = false;
    sendBtn.disabled = false;
    sendBtn.textContent = 'もう一度送る';
    alertBox.innerHTML =
      '送信できませんでした。恐れ入りますが、もう一度お試しください。' +
      'それでも送れないときは、下の「元のフォーム」からお願いします。';
    alertBox.hidden = false;
    alertBox.scrollIntoView({ block: 'center' });
  }

  form.addEventListener('submit', function (e) {
    e.preventDefault();
    if (sending) return;

    syncAgree();
    paintFee();

    var ok = [checkName(), checkMail(), checkTel(), checkKai(), checkAgree()]
             .every(function (v) { return v; });

    if (!ok) {
      alertBox.textContent = '未入力または確認が必要な項目があります。赤い印のところをご覧ください。';
      alertBox.hidden = false;
      var bad = form.querySelector('[aria-invalid="true"], .jf__err:not([hidden])');
      if (bad) bad.scrollIntoView({ block: 'center' });
      return;
    }

    alertBox.hidden = true;
    sending = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '送信しています…';

    fetch('/api/join', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fields: collect() })
    })
      .then(function (r) { return r.json().catch(function () { return {}; }); })
      .then(function (data) { if (data && data.ok) showDone(); else showFailed(); })
      .catch(showFailed);
  });

  syncAgree();
  paintFee();
})();
