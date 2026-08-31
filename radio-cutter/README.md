# radio-cutter

「AI活用法実験ラジオ」の収録動画を、決まり文句をアンカーに自動分割し、
YouTube 公開に必要な成果物一式まで作るローカル CLI です。

収録したままの mp4 を1本渡すと、次が揃います。

| 成果物 | ファイル | 中身 |
|---|---|---|
| ハイライト動画 | `01_highlight.mp4` | 本編から抜いた冒頭フック（既定30秒） |
| 本編動画 | `02_main.mp4` | アンカーA〜Bの区間 |
| エンディング動画 | `03_ending.mp4` | アンカーB〜終端 |
| 結合済み完成動画 | `final.mp4` | ①＋②＋③を連結 |
| YouTube概要欄 | `description.txt` | チャプター込み、そのまま貼れる |
| タイトル候補 | `titles.md` | 30個、6方向に分けて |
| 判断ログ | `decisions.json` | 検出したアンカー位置・カット秒・採用理由 |
| 文字起こし | `transcript.json` | 単語レベルのタイムスタンプ付き |
| 確認用プレビュー | `preview/*.mp4` | カット点前後2秒 |

詳しい仕様は [SPEC.md](SPEC.md) を見てください。
ターミナルに慣れていない方は [はじめかた.md](はじめかた.md) から読んでください。

入力は ffmpeg が読める形式ならそのまま渡せます。iPhone の `.MOV`（HEVC・H.264 どちらでも）、
大文字の拡張子、日本語やスペースを含むファイル名、縦向きの映像でも変換は要りません。
出力は常に `.mp4` です。

## 必要なもの

- Python 3.11 以上
- `ffmpeg` と `ffprobe` が PATH にあること
  （Apple Silicon なら VideoToolbox 対応ビルドだと書き出しが速い）
- 文字起こしのバックエンド（どちらか）
  - `mlx-whisper` — Apple Silicon 向け。こちらが主
  - `whisperx` — それ以外・フォールバック
- Claude Code（ハイライト選定と概要欄・タイトル生成に使う）
  既定ではこのパソコンに入っている Claude Code をそのまま呼ぶので、**APIキーは要りません**。
  `curl -fsSL https://claude.ai/install.sh | bash` で入れ、一度 `claude` を起動して
  ログインしておいてください。

## 入れる

```bash
cd radio-cutter
python3 -m pip install -e '.[llm,asr]'      # Apple Silicon
python3 -m pip install -e '.[llm,asr-x]'    # それ以外
claude          # 一度起動してログイン（APIキーは要りません）
```

Anthropic API を直接叩きたい場合だけ、`config/ai-radio.json` の `llm.provider` を
`"anthropic"`、`llm.model` を `"claude-opus-5"` などにして
`pip install -e '.[api]'` と `export ANTHROPIC_API_KEY=...` を足してください。

入ったかどうかは `doctor` が全部見ます。

```bash
radio-cutter doctor
```

足りないものは「何をどう入れればいいか」まで出ます。
`h264_videotoolbox` が無い環境では `libx264`（CPU エンコード）に自動で落ちます。

## 使う

普段はこの二段構えです。**先に判断を確認してから、書き出す。**

```bash
# 1. カット点とハイライトを決めるところまで（動画は作らない）
radio-cutter run ~/rec/ep42.mp4 --dry-run

# 2. decisions.json と preview/ で切り口を確かめる
open out/ep42/preview/cut_A.mp4
cat out/ep42/decisions.json

# 3. 問題なければ書き出す
radio-cutter run ~/rec/ep42.mp4 --from-step 7
```

60分の動画を切り直すたびに文字起こしからやり直すのは無駄なので、
中間ファイルは `work/<エピソードID>/` に残り、`--from-step N` で途中から再開できます。
文字起こしは入力の SHA-256 と ASR 設定でキャッシュされるので、同じ動画なら二度と走りません。

### 切り口がずれたとき

```bash
# 収録環境のノイズが大きいとき（無音と判定されにくい）
radio-cutter run ep42.mp4 --from-step 4 --silence-db=-28 --silence-dur 0.2

# 静かすぎて無音が長く続くとき
radio-cutter run ep42.mp4 --from-step 4 --silence-db=-38 --silence-dur 0.08
```

アンカーが見つからないときは、勝手に別の場所を選ばずに止まります。
そのとき、しきい値を下回った候補の上位3件とその前後の文脈が出るので、
`config/ai-radio.json` の `phrase` を実際の発話に寄せるか、`fuzzy_threshold` を下げてください。

### タイトルだけ作り直す

```bash
radio-cutter titles ep42
```

### コマンド一覧

```
radio-cutter run <input.mp4> [options]
  --config PATH        既定: config/ai-radio.json
  --out DIR            既定: out/
  --work DIR           既定: work/
  --from-step N        Nから再開（中間ファイルを再利用）
  --only-step N        Nだけ実行
  --dry-run            Step 6 まで実行し、書き出しを行わない
  --preview-only       Step 8 のプレビューだけ生成
  --silence-db DB      既定: -32（負の数は --silence-db=-30 の形で）
  --silence-dur SEC    既定: 0.12
  --episode-id ID      既定: 入力ファイル名の stem
  --stub-llm PATH      LLM の代わりに使う JSON（APIキー無しで通しの確認ができる）
  --force-transcribe   キャッシュを無視して文字起こしをやり直す

radio-cutter doctor            環境チェック
radio-cutter transcribe <f>    文字起こしのみ
radio-cutter titles <ep-id>    タイトルだけ再生成
```

## 何をしているか

```
入力 episode.mp4
  ↓ Step 1  音声抽出（ffmpeg）
  ↓ Step 2  文字起こし＋単語タイムスタンプ
  ↓ Step 3  アンカー検出（あいまい一致）
  ↓ Step 4  カット点の精密化（無音の谷に寄せる）
  ↓ Step 5  ハイライト区間の選定（LLM）
  ↓ Step 6  メタデータ生成（LLM：概要欄・チャプター・タイトル30個）
  ↓ Step 7  書き出し（ffmpeg）
  ↓ Step 8  確認用プレビュー生成
出力 out/<エピソードID>/
```

要になるのは Step 3 と Step 4 です。

Step 3 は、全単語をつないだ一本の文字列に正規化（NFKC・記号除去）をかけ、
アンカー語と同じ長さの窓を1文字ずつずらして似ている場所を拾います。
探索窓の外のもの、指定フレーズが続かないもの（`must_follow`）を落とし、
残りから `occurrence` に従って1つを確定します。

Step 4 は、そのままの位置で切ると語頭が削れるので、
直前の無音区間の終わりから50ミリ秒手前に寄せます。

## 設定

アンカー語はコードに入れず、チャンネルごとの JSON に置きます（`config/ai-radio.json`）。

```json
{
  "anchors": [
    { "id": "A", "phrase": "このチャンネルは", "occurrence": "first",
      "search_window_sec": [0, 600], "cut": "before", "fuzzy_threshold": 0.82 },
    { "id": "B", "phrase": "ということで", "occurrence": "last",
      "must_follow": { "phrase": "木原", "within_sec": 4.0 },
      "cut": "before", "fuzzy_threshold": 0.85 }
  ],
  "segments": [
    { "name": "main",   "file": "02_main.mp4",   "from": "A", "to": "B" },
    { "name": "ending", "file": "03_ending.mp4", "from": "B", "to": "end" }
  ]
}
```

- `occurrence` は `first` / `last` / `nth`（`nth` のときは `nth: 2` のように順位も書く）
- `must_follow` は「候補の終端から `within_sec` 以内にこのフレーズが続くもの」だけ残すフィルタ。
  「ということで」は本編中に何度も出るので、これで本物だけを残します
- `search_window_sec` は探す時間帯。オープニングの決まり文句を後半で拾わないための枠
- `fuzzy_threshold` は 0〜1。下げるほど拾いやすく、誤検出も増えます

`segments` は2本に限りません。`from` / `to` にアンカーIDか `start` / `end` を書けば、
3本以上にも分けられます。

プロンプトも `src/radio_cutter/llm/prompts/*.md` に外出ししてあるので、
言い回しを変えるのにコードを触る必要はありません。

## 開発

```bash
python3 -m pip install -e '.[dev]'
PYTHONPATH=src python3 -m pytest tests -q
```

テストは合成した60秒のエピソード（`tests/fixtures.py`）を使います。
音声・動画・文字起こしが同じ時刻表を共有しているので、
「無音の終わりに寄せたカット点が期待どおりの秒数になるか」まで実際に ffmpeg を叩いて確かめられます。
ffmpeg が無い環境では、それを使うテストだけ自動で飛びます。

APIキーが無くても、`--stub-llm` に応答の JSON を渡せば通しで動きます。

```bash
PYTHONPATH=src:tests python3 -c "
import fixtures, json
print(json.dumps(fixtures.stub_responses(), ensure_ascii=False))" > /tmp/stub.json
radio-cutter run ep42.mp4 --from-step 3 --stub-llm /tmp/stub.json
```
