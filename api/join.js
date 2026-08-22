/**
 * 入会申し込みの中継
 * ---------------------------------------------------------------------------
 * ブラウザから Google フォームへ直接投げる方式は、二つの点で行き詰まった。
 *
 *   ・送信が通らない。中身は一字一句正しいのに、Google が 400 を返していた。
 *     原因はフォーム側の「メールアドレスを収集する」設定だった。あれは設問では
 *     なく収集設定が描く必須欄で、外から埋める手立てが無い。永久に未入力扱いに
 *     なり、すべての送信が「必須の質問です」で弾かれていた。設定を切って解決。
 *     メールアドレスは設問（entry.458126145）として別にあるので、失うものはない。
 *   ・通ったかどうかも読めない。Google は別ドメインからの読み取りを許さない
 *     ので、弾かれても画面には「受け付けました」と出てしまう。
 *
 * そこで、間にこの中継を一枚はさむ。ブラウザ → ここ（Vercel のサーバー）→
 * Google の順に渡す。サーバー同士のやりとりにはブラウザの制限がかからず、
 * Google の返事もそのまま読める。つまり本当に届いたかを確かめてから、
 * 画面に返せる。
 *
 * 会員の多くは 70 代で、申し込んだつもりで届いていない事態がいちばん困る。
 * 迷ったときは「成功した」と言わない側に倒すこと。
 */

const FORM_ID = '1FAIpQLSf4ndb0q_DS74N1hV--7RFzZG_3PwCfU23aEpLBHUWar4KGtw';
const ENDPOINT = `https://docs.google.com/forms/d/e/${FORM_ID}/formResponse`;

/* 受け取ってよい欄。これ以外は捨てる。 */
const ALLOWED = new Set([
  'entry.1279542342',        /* お名前 */
  'entry.1074325515_year',   /* 生年月日 */
  'entry.1074325515_month',
  'entry.1074325515_day',
  'entry.458126145',         /* メールアドレス（設問） */
  'entry.2120756839',        /* 電話番号 */
  'entry.939298798',         /* 入会を希望する会 */
  'entry.1699267920',        /* 年会費 */
  'entry.1335746439',        /* 確認事項 */
  'entry.939298798_sentinel',   /* チェックボックスに添える連れ */
  'entry.1335746439_sentinel',
]);

const MAX_FIELDS = 40;
const MAX_LEN = 2000;

/* 送信が通ったときに Google が返す文面。日本語と英語の両方を見る。 */
const SUCCESS_MARKS = [
  'freebirdFormviewerViewResponseConfirmationMessage',
  'Your response has been recorded',
  '回答を記録しました',
  'フォームを送信しました',
];

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > 100000) reject(new Error('body too large'));
    });
    req.on('end', () => resolve(raw));
    req.on('error', reject);
  });
}

/* JS 経由なら JSON、<form> が素で来たなら読める画面を返す */
function reply(res, asForm, code, payload) {
  if (!asForm) return res.status(code).json(payload);

  const done = payload.ok;
  const head = done ? 'お申し込みを受け付けました。' : '送信できませんでした。';
  const body = done
    ? '事務局に届きしだい、担当者からご連絡し、お支払いの方法をご案内します。'
    : '恐れ入りますが、前の画面に戻って、もう一度お試しください。';

  res.setHeader('content-type', 'text/html; charset=utf-8');
  return res.status(code).send(
    '<!doctype html><meta charset="utf-8">' +
    '<meta name="viewport" content="width=device-width,initial-scale=1">' +
    '<title>' + head + '</title>' +
    '<style>body{margin:0;padding:56px 24px;background:#F4F5EE;color:#20301F;' +
    'font:17px/1.9 system-ui,"Hiragino Sans","Noto Sans JP",sans-serif}' +
    'div{max-width:34em;margin:0 auto}h1{font-size:22px;line-height:1.6;margin:0 0 12px}' +
    'a{display:inline-block;margin-top:24px;color:#1C3D24;font-weight:700}</style>' +
    '<div><h1>' + head + '</h1><p>' + body + '</p>' +
    '<a href="/#join">筑豊市民大学のページへもどる</a></div>'
  );
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ ok: false, reason: 'method' });
  }


  /* ふだんは画面の JS が JSON で渡してくる。
     JS が動かない場合は <form> がそのまま来るので、どちらも受ける。 */
  const asForm = !String(req.headers?.['content-type'] || '').includes('json');
  const raw = typeof req.body === 'string' ? req.body : await readBody(req);

  /* 同じ欄が何度も出る（希望する会、確認事項）ので、
     連想配列ではなく組の並びで受け取る。 */
  let fields;
  if (asForm) {
    fields = [...new URLSearchParams(raw)];
  } else {
    try {
      const parsed = JSON.parse(raw || '{}');
      fields = Array.isArray(parsed.fields) ? parsed.fields : null;
    } catch (e) {
      return reply(res, asForm, 400, { ok: false, reason: 'bad-json' });
    }
  }
  if (!fields || !fields.length || fields.length > MAX_FIELDS) {
    return reply(res, asForm, 400, { ok: false, reason: 'bad-fields' });
  }

  const form = new URLSearchParams();
  form.set('fvv', '1');
  form.set('pageHistory', '0');
  for (const pair of fields) {
    if (!Array.isArray(pair) || pair.length !== 2) continue;
    const [name, value] = pair;
    if (typeof name !== 'string' || typeof value !== 'string') continue;
    if (!ALLOWED.has(name)) continue;
    form.append(name, value.slice(0, MAX_LEN));
  }
  if (!form.has('entry.458126145') || !form.has('entry.939298798')) {
    return reply(res, asForm, 400, { ok: false, reason: 'missing-required' });
  }

  let upstream;
  try {
    upstream = await fetch(ENDPOINT, {
      method: 'POST',
      headers: {
        'content-type': 'application/x-www-form-urlencoded',
        /* 素っ気ない相手だと別の画面を返してくることがある */
        'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) ' +
                      'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36',
        'accept-language': 'ja,en;q=0.8',
        referer: `https://docs.google.com/forms/d/e/${FORM_ID}/viewform`,
      },
      body: form.toString(),
      redirect: 'follow',
    });
  } catch (e) {
    return reply(res, asForm, 502, { ok: false, reason: 'unreachable' });
  }

  const html = await upstream.text().catch(() => '');
  const recorded = SUCCESS_MARKS.some((m) => html.includes(m));

  if (upstream.ok && recorded) {
    return reply(res, asForm, 200, { ok: true });
  }

  /* 届いていない。何が起きたかは残す。画面には出さないが、
     Vercel の記録から後で追える。 */
  console.error('[join] 送信できず', {
    status: upstream.status,
    recorded,
    head: html.slice(0, 400).replace(/\s+/g, ' '),
  });

  return reply(res, asForm, 502, {
    ok: false,
    reason: upstream.ok ? 'rejected' : 'status-' + upstream.status,
  });
}
