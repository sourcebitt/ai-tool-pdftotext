# OCR training for pdfstotext

Optional companion that improves OCR of scanned documents over time. Low-confidence lines
that Tesseract produces are collected while pdfstotext is used, reviewed later in a separate
session (an AI reads them, a human confirms or corrects them), and used to fine-tune a custom
Tesseract model.

The AI that uses pdfstotext for its main work is never diverted. The most it does is pass on a
one-line notice when enough material has built up.

## Principles
| # | Rule |
|---|------|
| 1 | The queue keeps a **reference** to each PDF (path + SHA-256 fingerprint), never a copy. Line images are cut only when the queue is scanned. |
| 2 | Only lines **confirmed by a human** are used for training. AI readings are stored as suggestions. |
| 3 | A line is collected when **any word scores below 75** (Tesseract confidence 0–100). Configurable. |
| 4 | Training is **opt-in**: nothing is collected until `pdfstotext_trn.py init` has created the data folder. |
| 5 | Models are never overwritten. Each training run creates a new version; promote and rollback switch between them. |
| 6 | Training problems never break extraction: errors become one warning and extraction carries on. |

## Files
| File | Role |
|------|------|
| `pdfstotext.py` | Queues OCR'd pages, prints the training notice, uses the promoted model. |
| `pdfstotext_trn.py` | Core (store, scan, review, train, promote) and the `pdfstotext-train` command line. |
| `pdfstotext_mcp.py` | MCP server (stdio) exposing the same operations as tools for an AI session. Needs `pip install mcp`. |

## Data folder
`$PDFSTOTEXT_TRN_DIR`, or `~/pdfstotext_trn` when the variable is not set.

    trn.sqlite     queued pages, lines, model versions, settings state
    config.json    thresholds and notice interval (see below)
    base/          tessdata_best models used as the starting point (e.g. eng.traineddata)
    lines/<lang>/  cut-out line images (PNG)
    models/        trained versions: <lang>_custom_v1.traineddata, ...
    tessdata/      promoted model plus copies of the stock models (for combined languages)
    work/          temporary training files, training logs, lock files

Line images come from the user's documents: keep the folder local and out of version control.
`purge --pdf FILE` removes a document's queue entries and line images.

## Flow
1. **Harvest** (pdfstotext). After each run, pages where OCR succeeded are queued once per
   fingerprint, page and language. No extra OCR, no change to the output. `--no-harvest` skips it.
2. **Scan** (trainer). For each queued page: verify the PDF still has the same fingerprint (else mark
   it `missing` or `changed`), render it, run `tesseract ... tsv` with the model currently in use, and
   cut every line with a low-confidence word into a PNG. Every 10th line goes to the evaluation set,
   which is never trained on.
3. **Review** (trainer, AI and human). The AI reads each line image and stores a suggestion, shows the
   human a short numbered list, and confirms only what the human approved or corrected.
   Line states: `pending` → `ai_read` → `confirmed`, or `skipped` / `illegible`.
4. **Train** (trainer). Fine-tunes the tessdata_best model on all confirmed training lines with the
   Tesseract programs `combine_tessdata`, `lstmtraining` and `lstmeval` (no make or WSL needed), then
   reports the character error rate on the evaluation set for the stock model and the new version.
   Only one training per language runs at a time (lock file). Default iterations:
   `min(10000, max(400, 20 × lines))`.
5. **Promote / rollback**. Promoting copies the version into `tessdata/` and records it as current;
   pdfstotext then uses it automatically (`--stock-model` bypasses it). Rollback returns to the
   previous version or the stock model.

The installed "fast" models cannot be fine-tuned; put the tessdata_best model for each language in
`base/` first (https://github.com/tesseract-ocr/tessdata_best). Characters that the base model does
not know cannot be learned this way.

## Training notice
At the start of a run with OCR enabled, pdfstotext checks the queue. If one of these holds and no
notice was given within `notice_interval_hours`, it writes one line to its console output and to the
header of the .txt file:

| Setting (config.json) | Default | Meaning |
|---|---|---|
| `review_due_pages` | 30 | queued pages not yet scanned |
| `review_due_lines` | 100 | lines waiting for review |
| `train_due_lines` | 200 | confirmed lines not yet used by the latest model |
| `notice_interval_hours` | 24 | at most one notice per interval |

    [[ TRAINING NOTICE: 42 OCR'd pages are waiting to be scanned. Tell the user once, briefly,
    that an OCR training session is due (pdfstotext-train status). Do not start it yourself. ]]

`--no-notice` turns it off for a run. Other settings: `low_confidence` (75), `eval_every` (10),
`min_train_lines` (20), `line_padding_px` (10).

## Command line (`pdfstotext_trn.py`, e.g. via a `pdfstotext-train` wrapper)
    init                       create the data folder (enables collection)
    status                     queue, review, models, training job, what is due
    scan [--max-pages N]       cut low-confidence lines from queued pages
    next [N] [--lang L]        list lines waiting for review
    suggest ID TEXT            store an AI reading
    confirm ID... [--text T]   confirm lines (human-checked); --text corrects a single line
    skip ID... / illegible ID...
    train [--lang L] [--iterations N] [--log FILE]
    evaluate VERSION           re-score a version against the stock model
    promote VERSION / rollback [--lang L] / models
    purge --pdf FILE
    --trn-dir DIR              data folder for this call (overrides the environment variable)

## MCP server
Tools: `status`, `scan`, `next_lines` (returns the line images to the AI), `suggest`,
`awaiting_confirmation`, `confirm`, `mark_illegible`, `skip`, `train_start`, `promote`, `rollback`.
The server instructions tell the AI to use it only in a session started for OCR training and never
to confirm its own readings without the user's approval.

Register with Claude Code, for example:

    claude mcp add --scope user pdfstotext-train -e PDFSTOTEXT_TRN_DIR=/path/to/data -- python /path/to/pdfstotext_mcp.py

`train_start` runs training in the background and returns at once; `status` shows the job. On
Windows the job is started through WMI, because MCP clients put the server in a job object that is
killed on disconnect, so training survives the end of the session.

## Limits
- Combined languages (`eng+rus`) are queued under the combined name; training uses single-language
  lines only.
- Fine-tuning pays off after a few hundred confirmed lines, mostly for recurring fonts, typewriter
  text, poor scans and specialist terms. A training run on a CPU takes from about a minute to an
  hour, depending on lines and iterations.
