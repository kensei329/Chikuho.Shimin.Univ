/**
 * 一時的な調べもの用。原因が分かったら消すこと。
 *
 * Google が 400 を返す理由を突き止めるため、制御用パラメータの組み合わせを
 * 順に試し、それぞれの返事を持ち帰る。弾かれた分は記録に残らないので、
 * 通った組み合わせだけが 1 行できる。通った時点で打ち切る。
 */

const FORM_ID = '1FAIpQLSf4ndb0q_DS74N1hV--7RFzZG_3PwCfU23aEpLBHUWar4KGtw';
const ENDPOINT = `https://docs.google.com/forms/d/e/${FORM_ID}/formResponse`;

const ANSWERS = [
  ['entry.1279542342', '【動作確認】削除してください'],
  ['entry.1074325515_year', '1958'],
  ['entry.1074325515_month', '3'],
  ['entry.1074325515_day', '15'],
  ['emailAddress', 'test@example.com'],
  ['entry.458126145', 'test@example.com'],
  ['entry.2120756839', '0947-00-0000'],
  ['entry.939298798', '歴史をたどる会'],
  ['entry.1699267920', '¥10,000（1講座）'],
  ['entry.1335746439', '入力した内容を確認し、筑豊市民大学への入会を申し込みます。'],
  ['entry.1335746439', '個人情報の取り扱いに同意します。'],
];

const VARIANTS = [
  ['いまの形（fvv=1, pageHistory=0）', { fvv: '1', pageHistory: '0' }],
  ['制御なし', {}],
  ['fvv だけ', { fvv: '1' }],
  ['pageHistory だけ', { pageHistory: '0' }],
  ['fbzx を足す', { fvv: '1', pageHistory: '0', fbzx: '-1234567890123456789' }],
  ['submit を足す', { fvv: '1', pageHistory: '0', submit: 'Submit' }],
  ['partialResponse を足す', { fvv: '1', pageHistory: '0', partialResponse: '[null,null,"-1"]' }],
  ['メール欄なし', { fvv: '1', pageHistory: '0' }, (p) => p.filter(([k]) => k !== 'emailAddress')],
  ['会の設問だけ', { fvv: '1', pageHistory: '0' },
    (p) => p.filter(([k]) => k === 'entry.939298798')],
];

const MARKS = ['freebirdFormviewerViewResponseConfirmationMessage',
               'Your response has been recorded', '回答を記録しました', 'フォームを送信しました'];

export default async function handler(req, res) {
  if (req.query?.key !== 'shirabe') return res.status(404).json({ ok: false });

  const out = [];
  for (const [label, ctrl, filter] of VARIANTS) {
    const body = new URLSearchParams();
    for (const [k, v] of Object.entries(ctrl)) body.set(k, v);
    for (const [k, v] of (filter ? filter(ANSWERS) : ANSWERS)) body.append(k, v);

    let status = 0, recorded = false, head = '';
    try {
      const r = await fetch(ENDPOINT, {
        method: 'POST',
        headers: {
          'content-type': 'application/x-www-form-urlencoded',
          'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
                        '(KHTML, like Gecko) Chrome/126.0 Safari/537.36',
          'accept-language': 'ja,en;q=0.8',
        },
        body: body.toString(),
        redirect: 'follow',
      });
      status = r.status;
      const html = await r.text();
      recorded = MARKS.some((m) => html.includes(m));
      head = html.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 220);
    } catch (e) {
      head = 'つながらず: ' + e.message;
    }

    out.push({ label, status, recorded, head });
    if (status === 200 && recorded) break;   /* 通ったら、それ以上は送らない */
  }

  return res.status(200).json({ tried: out });
}
