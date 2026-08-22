/**
 * ホーム — 入会申し込みフォーム
 * ------------------------------------------------------------------------
 * 見た目は自前、送り先は Google フォーム。
 * 入力欄の name に Google フォームの entry ID をそのまま付けてあるので、
 * 隠した iframe を宛先にして、ふつうの <form> として送るだけで届く。
 *
 * ■ 送信の結果は、こちらからは分からない
 *   Google は /formResponse に CORS のヘッダを返さない。仕様上の制約で、
 *   回避する手立てはない。つまり Google 側で弾かれても、画面には
 *   「受け付けました」と出てしまう。
 *
 *   会員の多くは 70 代で、申し込んだつもりで届いていない事態は避けたい。
 *   そこで
 *     ・送る前にこちらで入力を確かめ、不備のまま送らせない
 *     ・必須の項目は Google フォーム側とそろえる
 *     ・受付の画面にも、元のフォームへの道を残す
 *   の三つで受けている。
 *
 * ■ 選択肢の文字は、一字一句そろえる必要がある
 *   ラジオやチェックボックスは、フォームに無い文字列を送ると弾かれる。
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
  var mail2 = document.getElementById('jf-mail2');
  var tel   = document.getElementById('jf-tel');
  var agree = document.getElementById('jf-agree');

  var alertBox = document.getElementById('jf-alert');
  var sendBtn  = document.getElementById('jf-send');
  var sink     = document.getElementById('jsink');
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
  agree.addEventListener('change', function () { checkAgree(); sweep(); });
  kais.forEach(function (b) { b.addEventListener('change', function () { checkKai(); sweep(); }); });

  /* ── 送信 ───────────────────────────────────────────────────────── */

  var sending = false;

  function showDone() {
    form.hidden = true;
    done.hidden = false;
    done.scrollIntoView({ block: 'center' });
  }

  form.addEventListener('submit', function (e) {
    /* メールは、フォーム側の設問にも同じ値を送る */
    mail2.value = mail.value.trim();
    paintFee();

    var ok = [checkName(), checkMail(), checkTel(), checkKai(), checkAgree()]
             .every(function (v) { return v; });

    if (!ok) {
      e.preventDefault();
      alertBox.textContent = '未入力または確認が必要な項目があります。赤い印のところをご覧ください。';
      alertBox.hidden = false;
      var bad = form.querySelector('[aria-invalid="true"], .jf__err:not([hidden])');
      if (bad) bad.scrollIntoView({ block: 'center' });
      return;
    }
    if (sending) { e.preventDefault(); return; }

    alertBox.hidden = true;
    sending = true;
    sendBtn.disabled = true;
    sendBtn.textContent = '送信しています…';
    /* ここで preventDefault はしない。<form> が iframe を宛先に送っていく */
  });

  /* 返事が返ってきたら受付の画面へ。中身は読めない（別のドメインなので）。
     返事が来ないことも考えて、時間でも受ける。 */
  var shown = false;
  function finish() {
    if (!sending || shown) return;
    shown = true;
    showDone();
  }
  sink.addEventListener('load', finish);
  form.addEventListener('submit', function () { window.setTimeout(finish, 8000); });

  paintFee();
})();
