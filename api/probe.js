/**
 * 一時的な調べもの用。原因が分かったら消すこと。
 * （ファイル名を _ で始めると Vercel が関数として拾わない）
 *
 * Google はフォームの定義を viewform の中に FB_PUBLIC_LOAD_DATA_ という
 * かたまりで埋め込んでいる。設問の種類・entry ID・必須かどうか・選択肢が
 * そのまま入っているので、こちらの送り方と突き合わせられる。
 */

const FORM_ID = '1FAIpQLSf4ndb0q_DS74N1hV--7RFzZG_3PwCfU23aEpLBHUWar4KGtw';
const VIEW = `https://docs.google.com/forms/d/e/${FORM_ID}/viewform`;
const POST = `https://docs.google.com/forms/d/e/${FORM_ID}/formResponse`;

const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
           '(KHTML, like Gecko) Chrome/126.0 Safari/537.36';

const KIND = {
  0: '記述（短文）', 1: '記述（段落）', 2: 'ラジオ', 3: 'プルダウン', 4: 'チェックボックス',
  5: '均等目盛', 7: 'グリッド', 8: '★セクション区切り', 9: '日付', 10: '時刻',
  6: '見出し', 11: '画像', 13: '動画', 18: 'ファイル添付',
};

function pull(html) {
  const m = html.match(/FB_PUBLIC_LOAD_DATA_\s*=\s*(\[[\s\S]*?\])\s*;\s*<\/script>/);
  if (!m) return null;
  try { return JSON.parse(m[1]); } catch (e) { return null; }
}

export default async function handler(req, res) {
  const key = new URL(req.url, 'http://x').searchParams.get('key');
  if (key !== 'shirabe') return res.status(404).json({ ok: false });

  /* ── 1. フォームの定義を読む ─────────────────────────────── */
  const view = await fetch(VIEW, { headers: { 'user-agent': UA, 'accept-language': 'ja' } });
  const html = await view.text();
  const data = pull(html);

  const report = { viewStatus: view.status, parsed: !!data, items: [], flags: {} };

  if (data) {
    const items = data[1]?.[1] || [];
    for (const it of items) {
      const row = { title: it[1], kind: KIND[it[3]] || ('種類' + it[3]), fields: [] };
      for (const e of (it[4] || [])) {
        row.fields.push({
          name: 'entry.' + e[0],
          required: e[2] === 1,
          options: (e[1] || []).map((o) => o[0]).filter((o) => o !== null && o !== ''),
        });
      }
      report.items.push(row);
    }
    /* 末尾のほうに、メール収集や1回限りなどの設定が入っている */
    report.flags.pageCount = 1 + report.items.filter((i) => i.kind === '★セクション区切り').length;
    report.flags.raw1_10 = data[1]?.[10] ?? null;
    report.flags.raw1_8 = data[1]?.[8] ?? null;
  }

  /* ── 2. いまの送り方で弾かれたとき、どこが赤くなるか ───────── */
  const body = new URLSearchParams();
  body.set('fvv', '1');
  body.set('pageHistory', '0');
  for (const [k, v] of [
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
  ]) body.append(k, v);

  const r = await fetch(POST, {
    method: 'POST',
    headers: {
      'content-type': 'application/x-www-form-urlencoded',
      'user-agent': UA, 'accept-language': 'ja',
    },
    body: body.toString(),
    redirect: 'follow',
  });
  const back = await r.text();
  report.postStatus = r.status;

  /* 弾かれた画面には、日本語のエラー文が素で入っている */
  const phrases = ['必須の質問です', '回答が正しくありません', '有効なメールアドレス',
                   'この質問は必須です', '不正な回答', 'エラー', 'もう一度'];
  report.errorsSeen = phrases.filter((p) => back.includes(p));

  const back2 = pull(back);
  if (back2) {
    /* 弾かれた側の定義にも、どの設問で転んだかの手がかりが乗る */
    report.postDataDiff = JSON.stringify(back2[1]?.[1]?.length) + '問';
  }
  report.backHead = back.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim().slice(0, 300);

  return res.status(200).json(report);
}
