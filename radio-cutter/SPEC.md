# radio-cutter 実装仕様書

「AI活用法実験ラジオ」の収録動画を、決まり文句をアンカーに自動分割し、
YouTube公開に必要な成果物一式まで生成するローカルCLIツール。

---

## 1. ゴール

収録したままのmp4を1本渡すと、以下がワンコマンドで揃う。

| 成果物 | ファイル | 内容 |
|---|---|---|
| ハイライト動画 | `01_highlight.mp4` | 本編から抽出した冒頭フック（既定30秒） |
| 本編動画 | `02_main.mp4` | アンカーA〜Bの区間 |
| エンディング動画 | `03_ending.mp4` | アンカーB〜終端 |
| 結合済み完成動画 | `final.mp4` | ①＋②＋③を連結したもの |
| YouTube概要欄 | `description.txt` | チャプター（タイムスタンプ）込み、そのまま貼れる状態 |
| タイトル候補 | `titles.md` | 30個、方向性別に分類 |
| 判断ログ | `decisions.json` | 検出したアンカー位置・カット秒・採用理由 |
| 文字起こし | `transcript.json` | 単語レベルのタイムスタンプ付き |
| 確認用プレビュー | `preview/cut_*.mp4` | カット点前後2秒の切り出し |

### 非ゴール（このバージョンではやらない）

- 字幕の焼き込み
- 縦型（9:16）へのリフレーム
- YouTubeへの自動アップロード
- BGM・SE・テロップの付与
- 動画の再エンコード品質最適化（見られる品質で十分）

---

## 2. 前提環境

- macOS（Apple Silicon / Mac Mini M4）
- Python 3.11+
- `ffmpeg`（VideoToolbox対応ビルド必須）と `ffprobe` がPATH上にあること
- LLM APIキー（環境変数）

初回に環境チェックコマンドを用意すること（`radio-cutter doctor`）。
`ffmpeg -encoders | grep videotoolbox` でハードウェアエンコーダの有無を確認し、
無ければCPUエンコードにフォールバックする旨を警告表示する。

---

## 3. 全体パイプライン

```
入力 episode.mp4
  ↓ Step 1  音声抽出（ffmpeg）
  ↓ Step 2  文字起こし＋単語タイムスタンプ（mlx-whisper系）
  ↓ Step 3  アンカー検出（あいまい一致）
  ↓ Step 4  カット点の精密化（silencedetect）
  ↓ Step 5  ハイライト区間の選定（LLM）
  ↓ Step 6  メタデータ生成（LLM：概要欄・チャプター・タイトル30個）
  ↓ Step 7  書き出し（ffmpeg）
  ↓ Step 8  確認用プレビュー生成
出力 out/<episode_id>/
```

各ステップは中間ファイルを `work/<episode_id>/` に残し、
`--from-step N` で途中から再実行できること。
文字起こしは最も重い工程なので、同じ入力ファイルに対しては必ずキャッシュを使う。

---

## 4. ディレクトリ構成

```
radio-cutter/
├─ SPEC.md
├─ pyproject.toml
├─ config/
│  └─ ai-radio.json
├─ src/radio_cutter/
│  ├─ cli.py
│  ├─ pipeline.py
│  ├─ steps/
│  │  ├─ s1_extract_audio.py
│  │  ├─ s2_transcribe.py
│  │  ├─ s3_find_anchors.py
│  │  ├─ s4_refine_cuts.py
│  │  ├─ s5_pick_highlight.py
│  │  ├─ s6_metadata.py
│  │  ├─ s7_render.py
│  │  └─ s8_preview.py
│  ├─ llm/
│  │  ├─ client.py
│  │  └─ prompts/
│  │     ├─ highlight.md
│  │     ├─ metadata.md
│  │     └─ titles.md
│  └─ util/
│     ├─ text_normalize.py
│     ├─ timeline.py
│     └─ ffmpeg.py
├─ work/     # 中間ファイル（.gitignore）
└─ out/      # 成果物（.gitignore）
```

---

## 5. 設定ファイル仕様

アンカー語をコードにハードコードしないこと。チャンネルごとにJSONで持つ。

`config/ai-radio.json`:

```json
{
  "channel": "AI活用法実験ラジオ",
  "anchors": [
    {
      "id": "A",
      "phrase": "このチャンネルは",
      "occurrence": "first",
      "search_window_sec": [0, 600],
      "cut": "before",
      "fuzzy_threshold": 0.82
    },
    {
      "id": "B",
      "phrase": "ということで",
      "occurrence": "last",
      "must_follow": { "phrase": "木原", "within_sec": 4.0 },
      "cut": "before",
      "fuzzy_threshold": 0.85
    }
  ],
  "segments": [
    { "name": "main",   "file": "02_main.mp4",   "from": "A", "to": "B" },
    { "name": "ending", "file": "03_ending.mp4", "from": "B", "to": "end" }
  ],
  "highlight": {
    "file": "01_highlight.mp4",
    "source_segment": "main",
    "target_duration_sec": 30,
    "min_duration_sec": 20,
    "max_duration_sec": 45,
    "position": "prepend",
    "allow_multi_cut": false
  },
  "asr": {
    "model": "mlx-community/whisper-large-v3-mlx",
    "language": "ja",
    "initial_prompt": "AI活用法実験ラジオ。このチャンネルは。ということで、木原さん。"
  },
  "llm": {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "max_retries": 3
  },
  "render": {
    "video_codec": "h264_videotoolbox",
    "video_bitrate": "12M",
    "audio_codec": "aac",
    "audio_bitrate": "192k",
    "fallback_video_codec": "libx264"
  },
  "youtube": {
    "channel_links": [],
    "fixed_footer": "",
    "hashtags": ["#AI", "#AI活用", "#生成AI"]
  }
}
```

`occurrence` は `first` / `last` / `nth` を受け付ける。
`must_follow` は指定フレーズが `within_sec` 以内に続く候補だけを残すフィルタ。

---

## 6. ステップ詳細

### Step 1 — 音声抽出

```bash
ffmpeg -y -i "{input}" -vn -ac 1 -ar 16000 -c:a pcm_s16le "{work}/audio.wav"
```

`ffprobe` で元動画の総尺・fps・解像度も取得し、`work/probe.json` に保存する。

### Step 2 — 文字起こし

`mlx-whisper` を使い、単語レベルのタイムスタンプを取得する。
単語タイムスタンプが必要なので、素の `mlx-whisper` CLIではなく
`whispermlx`（WhisperXのMLXバックエンド版）相当のAPIを使うこと。
利用できない場合は `whisperx`（faster-whisperバックエンド）にフォールバックする。

`config.asr.initial_prompt` を必ず渡す。アンカー語をモデルにバイアスさせるのが目的。

出力 `work/transcript.json`:

```json
{
  "language": "ja",
  "duration": 3612.4,
  "segments": [
    {
      "start": 0.42, "end": 5.10,
      "text": "このチャンネルはAIの活用法を実験する番組です",
      "words": [
        { "word": "この",     "start": 0.42, "end": 0.66 },
        { "word": "チャンネル", "start": 0.66, "end": 1.18 },
        { "word": "は",       "start": 1.18, "end": 1.30 }
      ]
    }
  ]
}
```

**キャッシュ**：入力ファイルのSHA-256とASR設定のハッシュをキーにして、
同一ならこのステップを丸ごとスキップする。ここが全工程の8割の時間を占めるため必須。

### Step 3 — アンカー検出

ここが最重要かつ最も壊れやすい箇所。以下の順で処理する。

1. **フラット化**：全 `words` を連結して1本の文字列 `flat` を作る。
   同時に「flat上の文字インデックス → 元の単語index」の対応表を持つ。

2. **正規化**：`flat` とアンカー語の両方に同じ正規化をかける。
   - 全角/半角の統一（NFKC正規化）
   - 句読点・記号・空白の除去（`。、,.!?！？「」…・ ` など）
   - カタカナ長音の揺れは触らない（過剰正規化は誤検出を招く）
   正規化後のインデックスから元インデックスへ戻せるようにマッピングを保持する。

3. **あいまい一致**：正規化後の `flat` に対し、アンカー語と同じ長さのウィンドウを
   1文字ずつスライドさせ、`rapidfuzz.fuzz.ratio` でスコアリング。
   `fuzzy_threshold` を超えたものを候補とする。
   隣接する候補（開始位置が3文字以内）は最高スコアのものに統合する。

4. **候補の絞り込み**：
   - `search_window_sec` の範囲外の候補を除外
   - `must_follow` が指定されていれば、候補終端から `within_sec` 以内に
     指定フレーズが出現するかを同じあいまい一致で確認し、満たさない候補を除外
   - `occurrence` に従って1つを確定

5. **時刻の取得**：確定した候補の先頭文字が属する単語の `start` を
   **raw_cut_time** とする。「こ」「と」の発話開始時刻がここ。

出力 `work/anchors.json`:

```json
{
  "A": {
    "phrase": "このチャンネルは",
    "matched_text": "このチャンネルは",
    "score": 100.0,
    "raw_cut_time": 12.84,
    "candidates_found": 1,
    "context": "...えー、じゃあ始めます。このチャンネルはAIの活用法を..."
  },
  "B": {
    "phrase": "ということで",
    "matched_text": "ということで",
    "score": 96.2,
    "raw_cut_time": 3218.06,
    "candidates_found": 7,
    "candidates_rejected": 6,
    "context": "...という話でした。ということで、木原さん、今日は..."
  }
}
```

**失敗時の挙動**：候補が0件なら例外を投げて停止する。
勝手に代替位置を選ばないこと。エラーメッセージには、しきい値を下回った
最高スコアの候補とその前後30文字の文脈を出し、
「しきい値を下げるか、config のフレーズを実際の発話に合わせて修正せよ」と案内する。

### Step 4 — カット点の精密化

`raw_cut_time` のままカットすると語の立ち上がりが削れる。無音の谷に寄せる。

```bash
ffmpeg -i "{work}/audio.wav" -af \
  "atrim=start={raw-1.5}:end={raw+0.5},asetpts=PTS-STARTPTS,silencedetect=n=-32dB:d=0.12" \
  -f null - 2>&1
```

標準エラー出力から `silence_start` / `silence_end` をパースし、
`raw_cut_time` より前にある最後の無音区間を採用。
その区間の**終了時刻マイナス50ms**を最終カット点 `cut_time` とする。

無音が検出されなければ `cut_time = raw_cut_time - 0.08` にフォールバックし、
`decisions.json` に `"silence_found": false` を記録する。

しきい値 `-32dB` と `d=0.12` はCLIオプションで上書き可能にする
（収録環境のノイズフロアによって最適値が変わるため）。

### Step 5 — ハイライト選定

本編区間（`cut_time_A` 〜 `cut_time_B`）の文字起こしのみをLLMに渡す。
プロンプトは `llm/prompts/highlight.md` に外出しする。

LLMには**候補を3つ**返させ、スコア最上位を採用する。
（1つだけ返させると外したときに手直しの起点が無くなるため。
残り2つは `decisions.json` に残し、UI追加時の差し替え候補にする。）

期待するJSON:

```json
{
  "candidates": [
    {
      "start": 842.5,
      "end": 871.2,
      "score": 92,
      "hook_line": "実は、AIに議事録を書かせるのは一番もったいない使い方なんです",
      "reason": "結論が先に来ていて、単体で意味が通る。数字と逆説を含む。"
    }
  ]
}
```

**採用後の後処理（必須）**：LLMが返す秒数は文の途中で切れていることが多い。

1. `start` / `end` を最寄りの単語境界にスナップ
2. `start` はその単語が属する文の先頭まで前方に拡張、
   `end` は文末まで後方に拡張（`。？！` で判定）
3. Step 4 と同じ `silencedetect` を前後1.5秒に適用し、無音の谷に寄せる
4. 結果が `max_duration_sec` を超える場合は末尾の文を1つ落として再計算

この3段スナップを飛ばすと、語尾が千切れて視聴に耐えないものが出る。

### Step 6 — メタデータ生成

LLM呼び出しは2回に分ける。同時に投げると片方の品質が落ちる。

#### 6-a. チャプター＋概要欄

入力：本編＋エンディングの文字起こし（タイムスタンプ付き、
トークン節約のため30秒単位に丸めたセグメント要約でよい）。

**チャプターの時刻は必ず `final.mp4` のタイムラインに変換すること。**
ハイライトを先頭に足しているため、元動画の秒数をそのまま書くと全部ずれる。

```
Dh = ハイライトの尺
Dm = 本編の尺（cut_time_B - cut_time_A）

元動画の時刻 t が本編内       → final_time = Dh + (t - cut_time_A)
元動画の時刻 t がエンディング内 → final_time = Dh + Dm + (t - cut_time_B)
```

YouTubeチャプターの成立条件を満たすこと。

- 最初のチャプターは必ず `0:00`
- 3つ以上
- 各チャプターは10秒以上
- 動画内の時刻の昇順

`0:00` は必ずハイライト部分に割り当てる（例：`0:00 今回の結論`）。

期待するJSON:

```json
{
  "summary_lead": "検索結果とモバイルの「もっと見る」前に表示される2〜3行。",
  "body": "本文。3〜5段落。",
  "chapters": [
    { "time_sec": 0,   "label": "今回の結論" },
    { "time_sec": 32,  "label": "オープニング" },
    { "time_sec": 118, "label": "AI議事録の落とし穴" }
  ],
  "keywords": ["AI議事録", "文字起こし", "業務自動化"]
}
```

`description.txt` の組み立てはコード側で行う（LLMに最終フォーマットまで
作らせない。順序が崩れるため）。

```
{summary_lead}

{body}

━━━━━━━━━━━━━━━
■ チャプター
0:00 今回の結論
0:32 オープニング
1:58 AI議事録の落とし穴
...

━━━━━━━━━━━━━━━
{fixed_footer}
{channel_links}

{hashtags}
```

時刻の書式は1時間未満なら `M:SS`、1時間以上なら `H:MM:SS`。

#### 6-b. タイトル候補30個

入力：`summary_lead` + `keywords` + ハイライトの `hook_line` + 本編要約。

30個を**6方向 × 5個**で生成させる。指定しないと同じ言い回しの30変奏になる。

| 方向性 | 狙い |
|---|---|
| 結論直球型 | 動画の主張をそのまま言い切る |
| 逆説・否定型 | 「〜はもうやめました」「実は逆効果」 |
| 数字型 | 「3つの」「90%が知らない」 |
| 疑問型 | 視聴者の疑問をそのまま置く |
| 実験・検証型 | 「試してみた」「1ヶ月使った結果」（番組名と整合） |
| ターゲット明示型 | 「非エンジニアのための」「中小企業の」 |

制約：

- 全角28〜32文字を主戦場とする（モバイル一覧で末尾が切れにくい範囲）
- 上限は全角45文字。YouTubeの仕様上限は100文字だが、そこまでは使わない
- 過剰な煽り（「衝撃」「ヤバい」「神」）は各方向1個までに制限
- 絵文字は使わない
- 【】は使ってよいが、30個中10個まで

出力 `titles.md`:

```markdown
# タイトル候補

## 結論直球型
1. ...
2. ...

## 逆説・否定型
6. ...
```

各行に想定文字数を併記する（例：`1. AIに議事録を書かせるのは一番もったいない（全角22字）`）。

### Step 7 — 書き出し

**キーフレーム問題に注意。** `-c copy` は最寄りのキーフレームまでカット位置が
ずれるため、この用途では使えない。必ず再エンコードすること。

```bash
ffmpeg -y -ss {start} -to {end} -i "{input}" \
  -c:v {video_codec} -b:v {video_bitrate} \
  -c:a {audio_codec} -b:a {audio_bitrate} \
  -movflags +faststart "{out}/{file}"
```

`-ss` は `-i` の**前**に置く（高速シーク）。ただし再エンコードするため
フレーム精度は保たれる。

`h264_videotoolbox` が使えない環境では `libx264 -preset veryfast -crf 20` に落とす。

連結は、3本を同一パラメータでエンコード済みなので concat demuxer を使う。

```bash
ffmpeg -y -f concat -safe 0 -i "{work}/concat.txt" -c copy \
  -movflags +faststart "{out}/final.mp4"
```

`final.mp4` の実尺を `ffprobe` で検算し、
`Dh + Dm + De` との差が0.5秒を超えたら警告を出す。

### Step 8 — 確認用プレビュー

カット点2箇所について、前後2秒（計4秒）を切り出す。
ハイライトの始点・終点も同様に出す。

```
out/<episode_id>/preview/
  cut_A.mp4        # cut_time_A ± 2秒
  cut_B.mp4        # cut_time_B ± 2秒
  highlight_in.mp4
  highlight_out.mp4
```

60分をレンダリングし直す前にここで確認できるようにするのが目的。

---

## 7. CLI仕様

```bash
radio-cutter run <input.mp4> [options]

  --config PATH        既定: config/ai-radio.json
  --out DIR            既定: out/
  --from-step N        Nから再開（中間ファイルを再利用）
  --only-step N        Nだけ実行
  --dry-run            Step 6まで実行し、書き出しを行わない
  --preview-only       Step 8のプレビューだけ生成
  --silence-db DB      既定: -32
  --silence-dur SEC    既定: 0.12
  --episode-id ID      既定: 入力ファイル名のstem

radio-cutter doctor            環境チェック
radio-cutter transcribe <f>    文字起こしのみ
radio-cutter titles <ep-id>    タイトルだけ再生成
```

`--dry-run` を既定の運用フローに想定する。
先に `decisions.json` とプレビューでカット点を確認し、
問題なければ `--from-step 7` で書き出す。

---

## 8. decisions.json スキーマ

すべての判断を1ファイルに集約する。あとから何が起きたか追えるようにする。

```json
{
  "episode_id": "ep42",
  "input": "/path/to/ep42.mp4",
  "input_sha256": "...",
  "duration": 3612.4,
  "generated_at": "2026-08-30T14:20:11+09:00",
  "anchors": {
    "A": { "raw_cut_time": 12.84, "cut_time": 12.61, "silence_found": true, "score": 100.0 },
    "B": { "raw_cut_time": 3218.06, "cut_time": 3217.82, "silence_found": true, "score": 96.2 }
  },
  "highlight": {
    "selected": { "start": 840.2, "end": 870.9, "score": 92, "reason": "..." },
    "alternatives": [ { "start": 1502.0, "end": 1531.4, "score": 85, "reason": "..." } ],
    "snapped_from": { "start": 842.5, "end": 871.2 }
  },
  "durations": { "highlight": 30.7, "main": 3205.21, "ending": 394.58, "final": 3630.49 },
  "llm_calls": [
    { "step": "highlight", "model": "claude-sonnet-4-6", "input_tokens": 24810, "retries": 0 }
  ],
  "warnings": []
}
```

---

## 9. エラーハンドリング方針

- **アンカー未検出** → 停止。候補スコア上位3件と文脈を表示。自動的に代替を選ばない。
- **アンカーBがAより前** → 停止。設定ミスの可能性が高い。
- **LLMがJSON以外を返す** → 3回までリトライ。失敗したらそのステップだけ落とし、
  動画の書き出しは続行する（`description.txt` が無くても動画は出す）。
- **ハイライト候補が本編の範囲外** → その候補を破棄し、次点を採用。全滅なら停止。
- **ffmpegが非ゼロ終了** → stderrをそのまま表示して停止。握りつぶさない。

---

## 10. 実装順序

一気に作らず、この順で動くものを積み上げること。
各フェーズの受け入れ基準を満たしてから次へ進む。

### Phase 1 — 分割まで（最優先）

Step 1〜4 と Step 7 の segment 書き出しのみ。ハイライトもLLMもまだ使わない。

受け入れ基準：
- 実際の収録動画1本で `cut_time_A` / `cut_time_B` が正しい位置に出る
- `02_main.mp4` の冒頭が「このチャンネルは」で始まっている
- `03_ending.mp4` の冒頭が「ということで」で始まっている
- 語頭が欠けていない

**ここが通らなければ先に進んでも意味がない。** まずここだけを検証する。

### Phase 2 — ハイライトと結合

Step 5、Step 7 の highlight/final、Step 8 を追加。

受け入れ基準：
- ハイライトが単体で意味の通る30秒になっている
- 語尾が千切れていない
- `final.mp4` が3本の合計尺と一致する

### Phase 3 — メタデータ

Step 6 を追加。

受け入れ基準：
- チャプターの時刻が `final.mp4` 上で正しい位置を指す（実際に再生して確認）
- 概要欄がそのままコピペで使える
- タイトル30個が6方向にきちんと分かれていて、同じ言い回しの重複が無い

### Phase 4 — 運用改善

- 文字起こしキャッシュ
- `doctor` コマンド
- 複数エピソードのバッチ処理

### Phase 5 —（必要になったら）確認UI

FastAPI + ブラウザ。波形とカット点、前後の文字起こしを表示し、
カット点をドラッグで微調整して書き出しを叩けるようにする。
Phase 1〜4がCLIで安定して回るまで着手しないこと。

---

## 11. 実装上の注意

- 秒数は全て `float`（小数点以下3桁）で扱う。フレーム番号に変換しない。
- 文字列比較は必ず正規化後に行う。生の文字起こしテキストで比較しない。
- LLMのプロンプトは `.md` ファイルに外出しし、コードに埋め込まない。
  プロンプト調整のたびにコードを触ることになるため。
- LLMには必ずJSON Schemaかそれに準ずる出力形式指定を渡し、
  返り値をスキーマ検証してからパースする。
- 中間ファイルは消さない。デバッグの起点になる。
- ログは各ステップの開始・終了・所要秒数を必ず出す。
  文字起こしが何分かかったかが分かるだけで運用判断がしやすくなる。
