/**
 * 一時的な調べもの用。原因が分かったら消すこと。
 * （ファイル名を _ で始めると Vercel が関数として拾わない）
 *
 * viewform が出している隠し欄（fbzx, submissionTimestamp, チェックボックスの
 * _sentinel など）をそのまま真似て送ってみる。通った組み合わせで打ち切る。
 */

const FORM_ID = '1FAIpQLSf4ndb0q_DS74N1hV--7RFzZG_3PwCfU23aEpLBHUWar4KGtw';
const VIEW = `https://docs.google.com/forms/d/e/${FORM_ID}/viewform`;
const POST = `https://docs.google.com/forms/d/e/${FORM_ID}/formResponse`;
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
           '(KHTML, like Gecko) Chrome/126.0 Safari/537.36';

const ANSWERS = [
  ['entry.1279542342', '【動作確認】削除してください'],
  ['entry.1074325515_year', '1958'],
  ['entry.1074325515_month', '3'],
  ['entry.1074325515_day', '15'],
  ['entry.458126145', 'test@example.com'],
  ['entry.2120756839', '0947-00-0000'],
  ['entry.939298798', '歴史をたどる会'],
  ['entry.1699267920', '¥10,000（1講座）'],
  ['entry.1335746439', '入力した内容を確認し、筑豊市民大学への入会を申し込みます。'],
  ['entry.1335746439', '個人情報の取り扱いに同意します。'],
];

const SENTINELS = [['entry.939298798_sentinel', ''], ['entry.1335746439_sentinel', '']];
const MARKS = ['freebirdFormviewerViewResponseConfirmationMessage',
               'Your response has been recorded', '回答を記録しました', 'フォームを送信しました'];

function hidden(html, name) {
  const re = new RegExp('name="' + name + '"[^>]*value="([^"]*)"');
  const m = html.match(re) ||
            html.match(new RegExp('value="([^"]*)"[^>]*name="' + name + '"'));
  return m ? m[1] : null;
}

export default async function handler(req, res) {
  const key = new URL(req.url, 'http://x').searchParams.get('key');
  if (key !== 'shirabe') return res.status(404).json({ ok: false });

  const view = await fetch(VIEW, { headers: { 'user-agent': UA, 'accept-language': 'ja' } });
  const html = await view.text();
  const cookie = (view.headers.getSetCookie?.() || []).map((c) => c.split(';')[0]).join('; ');

  const fbzx = hidden(html, 'fbzx');
  const partial = hidden(html, 'partialResponse');
  const found = { fbzx, partial, cookie: cookie ? cookie.slice(0, 60) + '…' : null,
                  mentionsEmailAddress: html.includes('emailAddress') };

  const V = [
    ['_sentinel を足す', [...SENTINELS], {}],
    ['本物の fbzx を足す', [], { fbzx: fbzx || '' }],
    ['_sentinel + fbzx', [...SENTINELS], { fbzx: fbzx || '' }],
    ['+ submissionTimestamp', [...SENTINELS], { fbzx: fbzx || '', submissionTimestamp: '-1' }],
    ['+ partialResponse', [...SENTINELS],
      { fbzx: fbzx || '', submissionTimestamp: '-1', partialResponse: partial ?? '[null,null,"' + (fbzx || '') + '"]' }],
    ['ぜんぶ + メール欄', [...SENTINELS],
      { fbzx: fbzx || '', submissionTimestamp: '-1', emailAddress: 'test@example.com' }],
    ['ぜんぶ + emailReceipt', [...SENTINELS],
      { fbzx: fbzx || '', submissionTimestamp: '-1', emailAddress: 'test@example.com', emailReceipt: 'false' }],
  ];

  const tried = [];
  for (const [label, extraPairs, ctrl] of V) {
    const body = new URLSearchParams();
    body.set('fvv', '1');
    body.set('pageHistory', '0');
    for (const [k, v] of Object.entries(ctrl)) body.set(k, v);
    for (const [k, v] of ANSWERS) body.append(k, v);
    for (const [k, v] of extraPairs) body.append(k, v);

    const r = await fetch(POST, {
      method: 'POST',
      headers: {
        'content-type': 'application/x-www-form-urlencoded',
        'user-agent': UA, 'accept-language': 'ja',
        referer: VIEW, origin: 'https://docs.google.com',
        ...(cookie ? { cookie } : {}),
      },
      body: body.toString(),
      redirect: 'follow',
    });
    const back = await r.text();
    const recorded = MARKS.some((m) => back.includes(m));
    tried.push({ label, status: r.status, recorded });
    if (r.status === 200 && recorded) break;
  }

  return res.status(200).json({ found, tried });
}
