/**
 * 入会申し込みフォーム
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
 * ■ 1 ページに何枚あってもよい
 *   ホームの本文と、デスクトップの右の帯と、同じフォームが並ぶ。
 *   id ではなく form の中を辿って部品を見つけるので、何枚あっても
 *   それぞれ独立して動く。
 *
 * ■ 選択肢の文字は、一字一句そろえる必要がある
 *   チェックボックスは、フォームに無い文字列を送ると弾かれる。
 *   HTML の value を書き換えるときは、必ずフォーム側と見比べること。
 */
(function () {
  'use strict';

  var FIRST = 10000;   /* 1講座 */
  var MORE  = 3000;    /* 2講座目から、1講座ごとに */
  var CAP   = 19000;   /* 4講座での上限 */

  var KAI   = 'entry.939298798';
  var AGREE = 'entry.1335746439';
  var MAIL  = 'entry.458126145';
  var TEL   = 'entry.2120756839';
  var NAME  = 'entry.1279542342';
  var FEE   = 'entry.1699267920';
  var BIRTH = 'entry.1074325515_';

  var MAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

  function comma(n) { return String(n).replace(/\B(?=(\d{3})+(?!\d))/g, ','); }
  function fee(n) { return n <= 0 ? 0 : Math.min(FIRST + (n - 1) * MORE, CAP); }

  function setup(form) {
    function q(sel) { return form.querySelector(sel); }
    function field(name) { return form.querySelector('[name="' + name + '"]'); }
    function err(key) { return form.querySelector('.jf__err[data-err="' + key + '"]'); }

    var kais   = [].slice.call(form.querySelectorAll('[name="' + KAI + '"]'));
    var yen    = q('.jfee__yen');
    var note   = q('.jfee__note');
    var feeOut = form.querySelector('input[name="' + FEE + '"]');

    var name  = field(NAME);
    var mail  = field(MAIL);
    var tel   = field(TEL);
    var agree = form.querySelector('input[type="checkbox"][name="' + AGREE + '"]');
    var agree2 = form.querySelector('input[type="hidden"][name="' + AGREE + '"]');

    var alertBox = q('.jform__alert');
    var sendBtn  = q('.jform__send');
    var done     = form.parentNode.querySelector('.jdone');

    if (!kais.length || !name || !mail || !tel || !agree || !sendBtn) return;

    /* ── 年会費 ───────────────────────────────────────────────────── */

    function paintFee() {
      var n = 0;
      kais.forEach(function (b) {
        if (b.checked) n++;
        var row = b.closest('.pick');
        if (row) row.classList.toggle('is-on', b.checked);   /* :has() が無い環境の控え */
      });

      var amount = fee(n);
      if (yen) yen.textContent = '¥' + comma(amount);
      if (note) {
        note.textContent =
          n === 0 ? '受けたい会をえらんでください。' :
          n === 4 ? '4講座すべて。これが上限の金額です。' :
                    n + '講座。もう1講座ふやすと +' + comma(MORE) + '円です。';
      }
      /* 画面に出ている金額を、そのまま送る */
      if (feeOut) feeOut.value = n === 0 ? '' : '¥' + comma(amount) + '（' + n + '講座）';
    }

    /* ── 入力の確かめ ─────────────────────────────────────────────── */

    function setErr(el, key, msg) {
      var box = err(key);
      if (box) {
        box.textContent = msg || '';
        box.hidden = !msg;
      }
      if (el) el.setAttribute('aria-invalid', msg ? 'true' : 'false');
      return !msg;
    }

    function checkName() {
      return setErr(name, 'name', name.value.trim() ? '' : 'お名前をご記入ください。');
    }
    function checkMail() {
      var v = mail.value.trim();
      if (!v) return setErr(mail, 'mail', 'メールアドレスをご記入ください。');
      if (!MAIL_RE.test(v)) return setErr(mail, 'mail', 'メールアドレスの形をご確認ください（例：taro@example.com）。');
      return setErr(mail, 'mail', '');
    }
    function checkTel() {
      var v = tel.value.trim();
      if (!v) return setErr(tel, 'tel', '電話番号をご記入ください。');
      /* 数字が 10 桁に満たないものは、書き間違いとみなす */
      if ((v.replace(/\D/g, '')).length < 10) return setErr(tel, 'tel', '電話番号をご確認ください（市外局番からご記入ください）。');
      return setErr(tel, 'tel', '');
    }
    function checkAgree() {
      return setErr(agree, 'agree', agree.checked ? '' : 'ご確認のうえ、チェックを入れてください。');
    }
    function checkKai() {
      var any = kais.some(function (b) { return b.checked; });
      return setErr(null, 'kai', any ? '' : '入会を希望する会を、1つ以上えらんでください。');
    }

    /* 赤い印が全部消えたら、上の注意書きも引っこめる。
       直したのに警告が出たままだと、どこが悪いのか分からなくなる。 */
    function sweep() {
      if (!alertBox || alertBox.hidden) return;
      if (!form.querySelector('.jf__err:not([hidden])')) alertBox.hidden = true;
    }

    [[name, checkName, 'name'], [mail, checkMail, 'mail'], [tel, checkTel, 'tel']].forEach(function (p) {
      p[0].addEventListener('blur', function () { p[1](); sweep(); });
      p[0].addEventListener('input', function () {
        if (p[0].getAttribute('aria-invalid') === 'true') { p[1](); sweep(); }
      });
    });

    /* 確認事項は、フォーム側では2つの選択肢に分かれている。
       画面では1つのチェックにまとめてあるので、送るときだけ2つに戻す。
       外した項目は disabled にしておけば、送信の中身に入らない。 */
    function syncAgree() { if (agree2) agree2.disabled = !agree.checked; }

    agree.addEventListener('change', function () { syncAgree(); checkAgree(); sweep(); });
    kais.forEach(function (b) {
      b.addEventListener('change', function () { paintFee(); checkKai(); sweep(); });
    });

    /* 生年月日。月と日は空から始め、選んだところだけ薄字をやめて濃くする */
    var dates = [].slice.call(form.querySelectorAll('select[name^="' + BIRTH + '"]'));
    function paintDate() {
      dates.forEach(function (s) { s.classList.toggle('is-empty', !s.value); });
    }
    dates.forEach(function (s) { s.addEventListener('change', paintDate); });

    /* 年だけは、いまから 60 年前を初めから出しておく。会員のおおよその世代で、
       そこから遠くへ回さずに済む。年を書き込まず毎回その場で数えるのは、
       年が明けても直さなくてよいようにするため。 */
    var yearSel = form.querySelector('select[name="' + BIRTH + 'year"]');
    if (yearSel && !yearSel.value) {
      var want = String(new Date().getFullYear() - 60);
      for (var i = 0; i < yearSel.options.length; i++) {
        if (yearSel.options[i].value === want) { yearSel.value = want; break; }
      }
    }

    /* ── 送信 ─────────────────────────────────────────────────────── */

    var sending = false;

    function collect() {
      var out = [];
      new FormData(form).forEach(function (v, k) { out.push([k, String(v)]); });
      /* 生年月日は任意。年・月・日がそろったときだけ送る。
         半端に送ると、フォーム側で日付として組み立てられない。 */
      var got = out.filter(function (p) { return p[0].indexOf(BIRTH) === 0 && p[1]; });
      if (got.length < 3) {
        out = out.filter(function (p) { return p[0].indexOf(BIRTH) !== 0; });
      }
      return out;
    }

    function showDone() {
      form.hidden = true;
      if (done) {
        done.hidden = false;
        done.scrollIntoView({ block: 'center' });
      }
    }

    function showFailed() {
      sending = false;
      sendBtn.disabled = false;
      sendBtn.textContent = 'もう一度送る';
      if (!alertBox) return;
      alertBox.textContent =
        '送信できませんでした。入力はそのまま残してあります。' +
        '恐れ入りますが、もう一度「送る」を押してください。';
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
        if (alertBox) {
          alertBox.textContent = '未入力または確認が必要な項目があります。赤い印のところをご覧ください。';
          alertBox.hidden = false;
        }
        var bad = form.querySelector('[aria-invalid="true"], .jf__err:not([hidden])');
        if (bad) bad.scrollIntoView({ block: 'center' });
        return;
      }

      if (alertBox) alertBox.hidden = true;
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
    paintDate();
    paintFee();
  }

  [].forEach.call(document.querySelectorAll('form.jform'), setup);
})();
