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
 *
 * ■ 悪用への構え
 *   この入口は誰でも叩ける。放っておくと、他所のページに置いた偽フォームから
 *   投げられたり、機械で連射されたりして、事務局の受信箱が偽の申し込みで
 *   埋まる。処理するのは人手なので、これがいちばん実害の大きい攻撃になる。
 *   そこで三重に受ける。
 *
 *     1. 自分のサイトから来たものだけ通す（Origin / Referer）
 *     2. 同じ相手からの連射を止める（IP ごと・入口ぜんたい）
 *     3. 人には見えない欄を一つ置き、埋まっていたら機械とみなす
 *
 *   どれも決定打ではない。1 は名乗りを騙れるし、2 は入口が増えれば数え直しに
 *   なるし、3 は狙って作られた道具には見抜かれる。それでも、その場かぎりの
 *   いたずらと出来合いの巡回botはここで止まる。会員に手間を増やさずに
 *   打てる手は、ここまでと考えている。人に謎かけ（CAPTCHA）はさせない。
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

/* 人には見えない欄。画面では読み上げからも外し、タブでも止まらない。
   埋まっているということは、欄という欄を埋める機械が来たということ。 */
const TRAP = 'kakunin';

const MAX_FIELDS = 40;
const MAX_LEN = 2000;
const MAX_BODY = 100000;

/* 連射よけ。人が申し込むのは、ふつう一度きり。
   公民館などから何人かがまとめて申し込む日もあるので、少し余裕をみる。

   相手を替えながら投げてこられると IP ごとの数えは効かないので、
   入口ぜんたいの上限も置く。この会に届く申し込みは年に数十件なので、
   10分で30件・1時間で60件も通れば、本物が詰まることはまず無い。
   攻められている最中は本物の人も待たされるが、事務局の受信箱が
   偽の申し込みで埋まるほうが困る。 */
const LIMITS = [
  { name: 'ip-10min',   span:  10 * 60 * 1000, max:  8, perIp: true  },
  { name: 'ip-1hour',   span:  60 * 60 * 1000, max: 20, perIp: true  },
  { name: 'all-10min',  span:  10 * 60 * 1000, max: 30, perIp: false },
  { name: 'all-1hour',  span:  60 * 60 * 1000, max: 60, perIp: false },
];

/* 送信が通ったときに Google が返す文面。日本語と英語の両方を見る。 */
const SUCCESS_MARKS = [
  'freebirdFormviewerViewResponseConfirmationMessage',
  'Your response has been recorded',
  '回答を記録しました',
  'フォームを送信しました',
];

/* この入口を置いてよい場所。プレビュー用の URL と、手元での確認も通す。 */
function ours(host) {
  if (!host) return false;
  const h = host.toLowerCase();
  return h === 'chikuho.uk' || h === 'www.chikuho.uk' ||
         h.endsWith('.vercel.app') ||
         h === 'localhost' || h === '127.0.0.1' ||
         h.startsWith('localhost:') || h.startsWith('127.0.0.1:');
}

function hostOf(url) {
  try { return new URL(url).host; } catch (e) { return ''; }
}

/**
 * 自分のサイトから来たかを見る。
 *
 * Origin は POST なら今どきのブラウザは必ず付ける。付いていないのは
 * ブラウザ以外か、よほど古いもの。名乗りは騙れるので、これは
 * 「偽サイトに置かれたフォームからの投稿」を止めるためのもので、
 * 直に叩いてくる相手には効かない。そちらは連射よけで受ける。
 */
function fromOurSite(req) {
  const h = req.headers || {};
  const site = h['sec-fetch-site'];
  if (site && site !== 'same-origin' && site !== 'same-site' && site !== 'none') return false;
  if (h.origin) return ours(hostOf(h.origin));
  if (h.referer) return ours(hostOf(h.referer));
  return true;   /* 名乗りが無いものは、ここでは弾かない */
}

/* 誰から来たか。
   x-forwarded-for は送り手が勝手に名乗れる（先頭に好きな値を足せば、
   数える相手を毎回すり替えられるし、他人の番号を騙って他人を締め出せる）。
   Vercel が自分で付ける x-vercel-forwarded-for / x-real-ip は騙れないので、
   そちらを先に見る。どちらも無いのは手元で動かしているときだけ。 */
function ipOf(req) {
  const h = req.headers || {};
  const trusted = h['x-vercel-forwarded-for'] || h['x-real-ip'];
  if (trusted) return String(trusted).split(',')[0].trim();
  return String(h['x-forwarded-for'] || '').split(',')[0].trim() || 'unknown';
}

/* 数えた跡。同じ処理が温まっているあいだだけ残る。
   入口が増えれば数え直しになるが、それでも連射の大半はここで止まる。 */
const seen = new Map();

/* 弾いたことは記録に残したいが、連射されているときに一件ずつ書くと、
   記録そのものが攻撃の道連れになる（量も費用もかさむ）。
   同じ理由については、1分に1度だけ書く。 */
const lastCry = new Map();
function cry(why, extra) {
  const now = Date.now();
  if (now - (lastCry.get(why) || 0) < 60000) return;
  lastCry.set(why, now);
  console.warn('[join] ' + why, extra);
}

function tooMany(ip, now) {
  const longest = Math.max(...LIMITS.map((l) => l.span));

  for (const [key, times] of seen) {
    const live = times.filter((t) => now - t < longest);
    if (live.length) seen.set(key, live); else seen.delete(key);
  }

  for (const limit of LIMITS) {
    const key = limit.perIp ? ip : '*';
    const times = seen.get(key) || [];
    if (times.filter((t) => now - t < limit.span).length >= limit.max) return limit.name;
  }

  for (const limit of LIMITS) {
    const key = limit.perIp ? ip : '*';
    if (!seen.has(key)) seen.set(key, []);
  }
  seen.get(ip).push(now);
  seen.get('*').push(now);
  return '';
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let raw = '';
    req.on('data', (chunk) => {
      raw += chunk;
      if (raw.length > MAX_BODY) reject(new Error('body too large'));
    });
    req.on('end', () => resolve(raw));
    req.on('error', reject);
  });
}

/* JS 経由なら JSON、<form> が素で来たなら読める画面を返す */
function reply(res, asForm, code, payload) {
  /* 申し込みの結果は、どこにも溜めない */
  res.setHeader('cache-control', 'no-store');

  if (!asForm) return res.status(code).json(payload);

  const done = payload.ok;
  const head = done ? 'お申し込みを受け付けました。' : '送信できませんでした。';
  const body = done
    ? '事務局に届きしだい、担当者からご連絡し、お支払いの方法をご案内します。'
    : payload.reason === 'too-many'
      ? '短いあいだに何度も送られています。恐れ入りますが、しばらく経ってからもう一度お試しください。'
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

/* 素の <form> と JSON の両方から、同じ「組の並び」を取り出す。
   同じ欄が何度も出る（希望する会、確認事項）ので、連想配列にはしない。 */
function pairsFrom(raw, asForm) {
  if (asForm) return [...new URLSearchParams(raw)];
  const parsed = JSON.parse(raw || '{}');
  return Array.isArray(parsed.fields) ? parsed.fields : null;
}

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    res.setHeader('cache-control', 'no-store');
    return res.status(405).json({ ok: false, reason: 'method' });
  }

  /* ふだんは画面の JS が JSON で渡してくる。
     JS が動かない場合は <form> がそのまま来るので、どちらも受ける。 */
  const asForm = !String(req.headers?.['content-type'] || '').includes('json');

  if (!fromOurSite(req)) {
    cry('よそのサイトから', { origin: req.headers?.origin, referer: req.headers?.referer });
    return reply(res, asForm, 403, { ok: false, reason: 'not-ours' });
  }

  const limited = tooMany(ipOf(req), Date.now());
  if (limited) {
    cry('連射', { limit: limited });
    return reply(res, asForm, 429, { ok: false, reason: 'too-many' });
  }

  let raw;
  if (typeof req.body === 'string') {
    /* 先に読まれて渡ってくることもある。その道でも長さは見ておく。 */
    if (req.body.length > MAX_BODY) return reply(res, asForm, 413, { ok: false, reason: 'too-large' });
    raw = req.body;
  }
  else if (req.body && typeof req.body === 'object') raw = null;   /* 先に解かれている */
  else {
    try { raw = await readBody(req); }
    catch (e) { return reply(res, asForm, 413, { ok: false, reason: 'too-large' }); }
  }

  let fields;
  try {
    fields = raw === null
      ? (asForm ? Object.entries(req.body).map(([k, v]) => [k, String(v)])
                : (Array.isArray(req.body.fields) ? req.body.fields : null))
      : pairsFrom(raw, asForm);
  } catch (e) {
    return reply(res, asForm, 400, { ok: false, reason: 'bad-json' });
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
    /* 人には見えない欄が埋まっている。機械とみなして、Google へは出さない。
       ここで「送れた」と嘘をつくと、万一これが人だったときに取り返しが
       つかない。落ちたことは正直に返す。 */
    if (name === TRAP && value.trim()) {
      cry('見えない欄が埋まっていた', {});
      return reply(res, asForm, 400, { ok: false, reason: 'trap' });
    }
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
