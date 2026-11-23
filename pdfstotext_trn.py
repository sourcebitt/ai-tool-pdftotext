"""
pdfstotext_trn.py — OCR training companion for pdfstotext
==========================================================

Collects low-confidence OCR lines from PDFs that pdfstotext has OCR'd, lets an AI
session and a human review them, and fine-tunes a custom Tesseract model from the
lines the human has confirmed. See DESIGN-training.md.

Data folder: $PDFSTOTEXT_TRN_DIR, or ~/pdfstotext_trn when the variable is not set.
Training is opt-in: nothing is collected until `pdfstotext_trn.py init` has created
the data folder.

Command line:
    python pdfstotext_trn.py init | status | scan | next | suggest | confirm | skip |
                             illegible | train | promote | rollback | models | purge

Only lines confirmed by a human are used for training. AI readings are stored as
suggestions until then.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import unicodedata
from pathlib import Path

try:
    import pymupdf
except ImportError:
    sys.exit("ERROR: PyMuPDF is not installed. Install it with: pip install pymupdf")


ENV_DIR = "PDFSTOTEXT_TRN_DIR"
DB_NAME = "trn.sqlite"

DEFAULT_CONFIG = {
    "low_confidence": 75,          # a line is collected if any word scores below this (0-100)
    "review_due_pages": 30,        # notice: queued pages not yet scanned
    "review_due_lines": 100,       # notice: lines waiting for review
    "train_due_lines": 200,        # notice: confirmed lines not yet used in training
    "notice_interval_hours": 24,   # at most one notice per interval
    "eval_every": 10,              # every Nth line goes to the evaluation set
    "min_train_lines": 20,         # refuse to train on fewer confirmed lines
    "line_padding_px": 10,         # margin around a cut-out line image
}

LINE_STATES = ("pending", "ai_read", "confirmed", "skipped", "illegible")
LOCK_STALE_HOURS = 12   # a training lock older than this is treated as left over from a crash

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    id           INTEGER PRIMARY KEY,
    pdf_path     TEXT NOT NULL,
    fingerprint  TEXT NOT NULL,
    page_no      INTEGER NOT NULL,          -- 1-based
    dpi          INTEGER NOT NULL,
    lang         TEXT NOT NULL,
    harvested_at TEXT NOT NULL,
    status       TEXT NOT NULL DEFAULT 'queued',   -- queued|scanned|missing|changed|failed
    scanned_at   TEXT,
    note         TEXT,
    UNIQUE (fingerprint, page_no, lang)
);
CREATE TABLE IF NOT EXISTS lines (
    id           INTEGER PRIMARY KEY,
    page_id      INTEGER NOT NULL REFERENCES pages(id),
    lang         TEXT NOT NULL,
    image        TEXT,                      -- path relative to the data folder
    bbox         TEXT NOT NULL,             -- pixel bbox at page dpi: x0,y0,x1,y1
    tess_text    TEXT NOT NULL,
    min_conf     REAL NOT NULL,
    mean_conf    REAL NOT NULL,
    ai_text      TEXT,
    final_text   TEXT,
    status       TEXT NOT NULL DEFAULT 'pending',
    split        TEXT NOT NULL DEFAULT 'train',     -- train|eval
    created_at   TEXT NOT NULL,
    reviewed_at  TEXT
);
CREATE INDEX IF NOT EXISTS lines_status ON lines(status, lang);
CREATE TABLE IF NOT EXISTS models (
    version      TEXT PRIMARY KEY,          -- e.g. eng_custom_v1
    lang         TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    train_lines  INTEGER NOT NULL,
    eval_lines   INTEGER NOT NULL,
    iterations   INTEGER NOT NULL,
    cer_base     REAL,                      -- % character error rate, stock model
    cer_new      REAL,                      -- % character error rate, this model
    status       TEXT NOT NULL DEFAULT 'candidate'  -- candidate|promoted|retired
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


class TrainingError(Exception):
    """A user-facing error from the training system."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def data_dir() -> Path:
    value = os.environ.get(ENV_DIR)
    return Path(value).expanduser() if value else Path.home() / "pdfstotext_trn"


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _tesseract_exe() -> Path:
    found = shutil.which("tesseract")
    if not found:
        raise TrainingError("Tesseract is not on PATH; install it or add its folder to PATH.")
    return Path(found)


def _tool(name: str) -> str:
    """Path of a Tesseract training program (installed next to tesseract)."""
    folder = _tesseract_exe().parent
    for candidate in (folder / f"{name}.exe", folder / name):
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if not found:
        raise TrainingError(f"Tesseract training program '{name}' not found next to tesseract.")
    return found


def stock_tessdata() -> Path:
    """The tessdata folder of the installed Tesseract."""
    env = os.environ.get("TESSDATA_PREFIX")
    candidates = [Path(env)] if env else []
    try:
        candidates.append(Path(pymupdf.get_tessdata()))
    except Exception:
        pass
    try:
        candidates.append(_tesseract_exe().parent / "tessdata")
    except TrainingError:
        pass
    for folder in candidates:
        if folder.is_dir() and any(folder.glob("*.traineddata")):
            return folder
    raise TrainingError("Cannot find the Tesseract tessdata folder; set TESSDATA_PREFIX.")


def _run(cmd: list[str], *, cwd: Path | None = None, log=None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if log is not None:
        for line in (result.stdout + result.stderr).splitlines():
            log(line)
    return result


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class Store:
    """SQLite-backed training data in one folder."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.db = sqlite3.connect(self.root / DB_NAME, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        columns = {r["name"] for r in self.db.execute("PRAGMA table_info(models)")}
        if "base_sha" not in columns:   # added later: fingerprint of the base model used
            self.db.execute("ALTER TABLE models ADD COLUMN base_sha TEXT")
            self.db.commit()
        self.config = self._load_config()

    # -- lifecycle --------------------------------------------------------
    @classmethod
    def init(cls, root: Path) -> "Store":
        root = Path(root)
        for sub in ("lines", "base", "work", "models", "tessdata"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        config_path = root / "config.json"
        if not config_path.exists():
            config_path.write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n", encoding="utf-8")
        return cls(root)

    @classmethod
    def open_existing(cls, root: Path | None = None) -> "Store":
        root = Path(root) if root else data_dir()
        if not (root / DB_NAME).is_file():
            raise TrainingError(
                f"No training data folder at {root}. Create it with: pdfstotext_trn.py init "
                f"(or set {ENV_DIR})."
            )
        return cls(root)

    @classmethod
    def open_if_enabled(cls, root: Path | None = None) -> "Store | None":
        """The store if training was initialised, else None (training is opt-in)."""
        root = Path(root) if root else data_dir()
        return cls(root) if (root / DB_NAME).is_file() else None

    def close(self) -> None:
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _load_config(self) -> dict:
        config = dict(DEFAULT_CONFIG)
        path = self.root / "config.json"
        if path.is_file():
            try:
                config.update(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError) as exc:
                print(f"WARNING: ignoring unreadable {path}: {exc}", file=sys.stderr)
        return config

    # -- meta -------------------------------------------------------------
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str | None) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    # -- harvest (called by pdfstotext) -----------------------------------
    def harvest(self, pdf_path: Path, page_numbers: list[int], dpi: int, lang: str) -> int:
        """Queue OCR'd pages of a PDF. Returns the number of newly queued pages."""
        if not page_numbers:
            return 0
        fingerprint = file_fingerprint(pdf_path)
        stamp = _now()
        with self.db:
            before = self.db.total_changes
            self.db.executemany(
                "INSERT OR IGNORE INTO pages(pdf_path, fingerprint, page_no, dpi, lang, harvested_at) "
                "VALUES(?, ?, ?, ?, ?, ?)",
                [(str(pdf_path), fingerprint, n, dpi, lang, stamp) for n in page_numbers],
            )
            return self.db.total_changes - before

    # -- counts and notice ------------------------------------------------
    def _count(self, sql: str, params: tuple = ()) -> int:
        return self.db.execute(sql, params).fetchone()[0]

    def latest_model(self, lang: str) -> sqlite3.Row | None:
        return self.db.execute(
            "SELECT * FROM models WHERE lang = ? ORDER BY created_at DESC, rowid DESC LIMIT 1",
            (lang,),
        ).fetchone()

    def untrained_confirmed(self, lang: str) -> int:
        confirmed = self._count(
            "SELECT COUNT(*) FROM lines WHERE lang = ? AND status = 'confirmed' AND split = 'train'",
            (lang,),
        )
        latest = self.latest_model(lang)
        # Approximate after a purge removed lines a model was trained on; never negative.
        return max(0, confirmed - (latest["train_lines"] if latest else 0))

    def due_reasons(self) -> list[str]:
        cfg = self.config
        reasons = []
        queued = self._count("SELECT COUNT(*) FROM pages WHERE status = 'queued'")
        if queued >= cfg["review_due_pages"]:
            reasons.append(f"{queued} OCR'd pages are waiting to be scanned")
        waiting = self._count("SELECT COUNT(*) FROM lines WHERE status IN ('pending', 'ai_read')")
        if waiting >= cfg["review_due_lines"]:
            reasons.append(f"{waiting} low-confidence lines are waiting for review")
        for (lang,) in self.db.execute("SELECT DISTINCT lang FROM lines"):
            ready = self.untrained_confirmed(lang)
            if ready >= cfg["train_due_lines"]:
                reasons.append(f"{ready} confirmed '{lang}' lines are ready for training")
        return reasons

    def training_notice(self) -> str | None:
        """A one-line notice if a session is due and none was given recently."""
        reasons = self.due_reasons()
        if not reasons:
            return None
        last = self.get_meta("last_notice_at")
        if last:
            try:
                elapsed = dt.datetime.now() - dt.datetime.fromisoformat(last)
                if elapsed < dt.timedelta(hours=float(self.config["notice_interval_hours"])):
                    return None
            except ValueError:
                pass
        self.set_meta("last_notice_at", _now())
        return (
            "[[ TRAINING NOTICE: " + "; ".join(reasons) + ". Tell the user once, briefly, that "
            "an OCR training session is due (pdfstotext-train status). Do not start it yourself. ]]"
        )

    # -- promoted models (used by pdfstotext) -----------------------------
    def current_model(self, lang: str) -> str | None:
        return self.get_meta(f"current_model:{lang}")

    def ocr_model_for(self, language: str) -> tuple[str, str] | None:
        """(language string, tessdata dir) using promoted models, or None if none apply."""
        tessdata = self.root / "tessdata"
        parts, changed = [], False
        for part in language.split("+"):
            current = self.current_model(part)
            if current and (tessdata / f"{current}.traineddata").is_file():
                parts.append(current)
                changed = True
            else:
                parts.append(part)
        if not changed:
            return None
        if not all((tessdata / f"{p}.traineddata").is_file() for p in parts):
            return None
        return "+".join(parts), str(tessdata)

    # -- scan ---------------------------------------------------------------
    def scan(self, max_pages: int = 20, log=print) -> dict:
        """Find low-confidence lines on queued pages and cut them out as images."""
        rows = self.db.execute(
            "SELECT * FROM pages WHERE status = 'queued' ORDER BY id LIMIT ?", (max_pages,)
        ).fetchall()
        result = {"pages": 0, "lines": 0, "missing": 0, "changed": 0, "failed": 0}
        fingerprints: dict[str, str | None] = {}
        documents: dict[str, pymupdf.Document] = {}
        try:
            for row in rows:
                status, note, found = self._scan_page(row, fingerprints, documents, log)
                with self.db:
                    self.db.execute(
                        "UPDATE pages SET status = ?, scanned_at = ?, note = ? WHERE id = ?",
                        (status, _now(), note, row["id"]),
                    )
                if status == "scanned":
                    result["pages"] += 1
                    result["lines"] += found
                else:
                    result[status] += 1
        finally:
            for document in documents.values():
                document.close()
        return result

    def _scan_page(self, row, fingerprints, documents, log) -> tuple[str, str | None, int]:
        path = row["pdf_path"]
        if path not in fingerprints:
            fingerprints[path] = file_fingerprint(Path(path)) if Path(path).is_file() else None
        if fingerprints[path] is None:
            return "missing", "PDF not found", 0
        if fingerprints[path] != row["fingerprint"]:
            return "changed", "PDF content changed since it was queued", 0
        try:
            if path not in documents:
                documents[path] = pymupdf.open(path)
            page = documents[path][row["page_no"] - 1]
            lines = self._low_confidence_lines(page, row["dpi"], row["lang"])
        except Exception as exc:   # one bad page must not stop the scan
            log(f"WARNING: scan failed for {Path(path).name} page {row['page_no']}: {exc}")
            return "failed", str(exc)[:500], 0

        pad = int(self.config["line_padding_px"])
        dpi = row["dpi"]
        line_dir = self.root / "lines" / row["lang"]
        line_dir.mkdir(parents=True, exist_ok=True)
        for (x0, y0, x1, y1), text, min_conf, mean_conf in lines:
            with self.db:
                cur = self.db.execute(
                    "INSERT INTO lines(page_id, lang, bbox, tess_text, min_conf, mean_conf, created_at) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (row["id"], row["lang"], f"{x0},{y0},{x1},{y1}", text, min_conf, mean_conf, _now()),
                )
                line_id = cur.lastrowid
                split = "eval" if line_id % int(self.config["eval_every"]) == 0 else "train"
                rel = Path("lines") / row["lang"] / f"{line_id}.png"
                clip = pymupdf.Rect(x0 - pad, y0 - pad, x1 + pad, y1 + pad) * (72 / dpi)
                pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, clip=clip & page.rect)
                pix.set_dpi(dpi, dpi)
                pix.save(self.root / rel)
                self.db.execute("UPDATE lines SET image = ?, split = ? WHERE id = ?",
                                (str(rel), split, line_id))
        return "scanned", f"{len(lines)} low-confidence lines", len(lines)

    def _ocr_language(self, lang: str) -> tuple[str, str]:
        promoted = self.ocr_model_for(lang)
        return promoted if promoted else (lang, str(stock_tessdata()))

    def _low_confidence_lines(self, page, dpi: int, lang: str):
        """[(pixel bbox, text, min_conf, mean_conf)] for lines with a word below threshold."""
        language, tessdata = self._ocr_language(lang)
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
        pix.set_dpi(dpi, dpi)
        work = self.root / "work"
        work.mkdir(exist_ok=True)
        image = work / f"scan_{os.getpid()}.png"
        pix.save(image)
        try:
            proc = _run([str(_tesseract_exe()), str(image), "stdout", "-l", language,
                         "--tessdata-dir", tessdata, "--dpi", str(dpi), "tsv"])
        finally:
            image.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise TrainingError(f"tesseract failed: {proc.stderr.strip()[:300]}")

        groups: dict[tuple, list[dict]] = {}
        for rec in csv.DictReader(io.StringIO(proc.stdout), delimiter="\t", quoting=csv.QUOTE_NONE):
            if rec.get("level") != "5" or not (rec.get("text") or "").strip():
                continue
            key = (rec["block_num"], rec["par_num"], rec["line_num"])
            groups.setdefault(key, []).append(rec)

        threshold = float(self.config["low_confidence"])
        found = []
        for words in groups.values():
            confs = [float(w["conf"]) for w in words if float(w["conf"]) >= 0]
            if not confs or min(confs) >= threshold:
                continue
            x0 = min(int(w["left"]) for w in words)
            y0 = min(int(w["top"]) for w in words)
            x1 = max(int(w["left"]) + int(w["width"]) for w in words)
            y1 = max(int(w["top"]) + int(w["height"]) for w in words)
            if y1 - y0 < 8 or x1 - x0 < 8:
                continue   # specks, not text lines
            text = " ".join(w["text"].strip() for w in words)
            found.append(((x0, y0, x1, y1), text, min(confs), sum(confs) / len(confs)))
        return found

    # -- review -------------------------------------------------------------
    def next_lines(self, count: int = 10, lang: str | None = None) -> list[dict]:
        """Lines needing attention: unread ones first, then AI-read ones awaiting confirmation."""
        sql = ("SELECT * FROM lines WHERE status IN ('pending', 'ai_read')"
               + (" AND lang = ?" if lang else "")
               + " ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, id LIMIT ?")
        params = (lang, count) if lang else (count,)
        return [self._line_dict(r) for r in self.db.execute(sql, params)]

    def awaiting_confirmation(self, lang: str | None = None, limit: int = 50) -> list[dict]:
        sql = ("SELECT * FROM lines WHERE status = 'ai_read'"
               + (" AND lang = ?" if lang else "") + " ORDER BY id LIMIT ?")
        params = (lang, limit) if lang else (limit,)
        return [self._line_dict(r) for r in self.db.execute(sql, params)]

    def _line_dict(self, row) -> dict:
        return {
            "id": row["id"], "lang": row["lang"], "status": row["status"],
            "image": str(self.root / row["image"]) if row["image"] else None,
            "tesseract": row["tess_text"], "min_conf": round(row["min_conf"], 1),
            "ai_text": row["ai_text"], "final_text": row["final_text"],
        }

    def _get_line(self, line_id: int) -> sqlite3.Row:
        row = self.db.execute("SELECT * FROM lines WHERE id = ?", (line_id,)).fetchone()
        if row is None:
            raise TrainingError(f"No line with id {line_id}.")
        return row

    def suggest(self, line_id: int, text: str) -> None:
        """Store an AI reading (a suggestion, not used for training until confirmed)."""
        row = self._get_line(line_id)
        if row["status"] == "confirmed":
            raise TrainingError(f"Line {line_id} is already confirmed.")
        text = _clean_text(text)
        if not text:
            raise TrainingError("Suggested text is empty; use 'illegible' instead.")
        with self.db:
            self.db.execute("UPDATE lines SET ai_text = ?, status = 'ai_read' WHERE id = ?",
                            (text, line_id))

    def confirm(self, line_id: int, text: str | None = None) -> str:
        """Record the human-confirmed text (the given text, else the AI suggestion)."""
        row = self._get_line(line_id)
        final = _clean_text(text) if text is not None else row["ai_text"]
        if not final:
            raise TrainingError(f"Line {line_id} has no suggestion; give the text to confirm.")
        with self.db:
            self.db.execute(
                "UPDATE lines SET final_text = ?, status = 'confirmed', reviewed_at = ? WHERE id = ?",
                (final, _now(), line_id),
            )
        return final

    def set_status(self, line_id: int, status: str) -> None:
        if status not in ("skipped", "illegible", "pending"):
            raise TrainingError(f"Unsupported status: {status}")
        self._get_line(line_id)
        with self.db:
            self.db.execute("UPDATE lines SET status = ?, reviewed_at = ? WHERE id = ?",
                            (status, _now(), line_id))

    # -- training -----------------------------------------------------------
    def train(self, lang: str = "eng", iterations: int | None = None, log=print) -> dict:
        """Fine-tune a new model version. Only one training per language at a time."""
        lock = self.root / "work" / f"train_{lang}.lock"
        lock.parent.mkdir(exist_ok=True)
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age_h = (dt.datetime.now().timestamp() - lock.stat().st_mtime) / 3600
            if age_h < LOCK_STALE_HOURS:
                raise TrainingError(
                    f"A '{lang}' training is already running (lock {lock}, {age_h:.1f} h old). "
                    "If none is running, delete the lock file."
                ) from None
            lock.unlink(missing_ok=True)
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        try:
            os.write(fd, f"{os.getpid()} {_now()}\n".encode())
            os.close(fd)
            return self._train(lang, iterations, log)
        finally:
            lock.unlink(missing_ok=True)

    def _train(self, lang: str, iterations: int | None, log) -> dict:
        base = self.root / "base" / f"{lang}.traineddata"
        if not base.is_file():
            raise TrainingError(
                f"Missing {base}. Put the tessdata_best '{lang}.traineddata' there "
                "(the 'fast' models cannot be fine-tuned)."
            )
        rows = self.db.execute(
            "SELECT * FROM lines WHERE lang = ? AND status = 'confirmed' ORDER BY id", (lang,)
        ).fetchall()
        train_rows = [r for r in rows if r["split"] == "train"]
        eval_rows = [r for r in rows if r["split"] == "eval"]
        if len(train_rows) < int(self.config["min_train_lines"]):
            raise TrainingError(
                f"Only {len(train_rows)} confirmed '{lang}' training lines; "
                f"at least {self.config['min_train_lines']} are needed."
            )

        number = self._count("SELECT COUNT(*) FROM models WHERE lang = ?", (lang,)) + 1
        version = f"{lang}_custom_v{number}"
        while (self.root / "models" / f"{version}.traineddata").exists():
            number += 1
            version = f"{lang}_custom_v{number}"
        work = self.root / "work" / version
        if work.exists():
            shutil.rmtree(work)
        (work / "lines").mkdir(parents=True)
        (work / "ckpt").mkdir()

        log(f"Preparing {len(train_rows)} training and {len(eval_rows)} evaluation lines ...")
        stock = str(stock_tessdata())
        train_list = self._make_lstmf(train_rows, work, stock, log)
        eval_list = self._make_lstmf(eval_rows, work, stock, log)
        if len(train_list) < int(self.config["min_train_lines"]):
            raise TrainingError(f"Only {len(train_list)} lines could be prepared for training.")
        (work / "train.txt").write_text("\n".join(train_list) + "\n", encoding="utf-8", newline="\n")
        (work / "eval.txt").write_text("\n".join(eval_list) + "\n", encoding="utf-8", newline="\n")

        iterations = iterations or min(10000, max(400, 20 * len(train_list)))
        start = work / f"{lang}.lstm"
        proc = _run([_tool("combine_tessdata"), "-e", str(base), str(start)], log=log)
        if proc.returncode != 0 or not start.is_file():
            raise TrainingError("combine_tessdata could not extract the LSTM model.")

        log(f"Fine-tuning {version} for {iterations} iterations ...")
        proc = _run([
            _tool("lstmtraining"),
            "--model_output", str(work / "ckpt" / "out"),
            "--continue_from", str(start),
            "--traineddata", str(base),
            "--train_listfile", str(work / "train.txt"),
            "--max_iterations", str(iterations),
        ], log=log)
        checkpoint = work / "ckpt" / "out_checkpoint"
        if proc.returncode != 0 or not checkpoint.is_file():
            raise TrainingError("lstmtraining failed; see the log above.")

        output = self.root / "models" / f"{version}.traineddata"
        proc = _run([
            _tool("lstmtraining"), "--stop_training",
            "--continue_from", str(checkpoint),
            "--traineddata", str(base),
            "--model_output", str(output),
        ], log=log)
        if proc.returncode != 0 or not output.is_file():
            raise TrainingError("Could not write the trained model.")

        cer_base = cer_new = None
        if eval_list:
            cer_base = self._evaluate(base, work / "eval.txt", log)
            cer_new = self._evaluate(output, work / "eval.txt", log)
        with self.db:
            self.db.execute(
                "INSERT INTO models(version, lang, created_at, train_lines, eval_lines, iterations, "
                "cer_base, cer_new, base_sha) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (version, lang, _now(), len(train_rows), len(eval_list), iterations, cer_base, cer_new,
                 file_fingerprint(base)),
            )
        shutil.rmtree(work / "ckpt", ignore_errors=True)
        return {"version": version, "train_lines": len(train_list), "eval_lines": len(eval_list),
                "iterations": iterations, "cer_base": cer_base, "cer_new": cer_new,
                "model": str(output)}

    def _make_lstmf(self, rows, work: Path, tessdata: str, log) -> list[str]:
        """Line image + WordStr box file -> .lstmf training file. Returns the .lstmf paths."""
        made = []
        for row in rows:
            source = self.root / (row["image"] or "")
            if not source.is_file():
                log(f"WARNING: image of line {row['id']} is missing; skipped.")
                continue
            base = work / "lines" / str(row["id"])
            shutil.copyfile(source, base.with_suffix(".png"))
            pix = pymupdf.Pixmap(str(source))
            width, height = pix.width, pix.height
            text = _clean_text(row["final_text"])
            base.with_suffix(".gt.txt").write_text(text + "\n", encoding="utf-8", newline="\n")
            box = [f"WordStr 0 0 {width} {height} 0 #{text}",
                   f"\t {width} {height} {width + 1} {height + 1} 0"]
            base.with_suffix(".box").write_text("\n".join(box) + "\n", encoding="utf-8", newline="\n")
            proc = _run([str(_tesseract_exe()), base.with_suffix(".png").name, base.name,
                         "--tessdata-dir", tessdata, "--psm", "13", "lstm.train"],
                        cwd=base.parent)
            lstmf = base.with_suffix(".lstmf")
            if proc.returncode == 0 and lstmf.is_file():
                made.append(str(lstmf))
            else:
                log(f"WARNING: line {row['id']} could not be prepared: {proc.stderr.strip()[:200]}")
        return made

    def _evaluate(self, model: Path, eval_list: Path, log) -> float | None:
        proc = _run([_tool("lstmeval"), "--model", str(model), "--eval_listfile", str(eval_list)])
        output = proc.stdout + proc.stderr
        match = (re.search(r"BCER eval=\s*([\d.]+)", output)
                 or re.search(r"Char error rate=\s*([\d.]+)", output))
        if not match:
            log(f"WARNING: could not read the evaluation result for {model.name}.")
            return None
        return float(match.group(1))

    def base_changed(self, lang: str) -> str | None:
        """A note if base/<lang>.traineddata differs from the one the latest model was trained on."""
        latest = self.latest_model(lang)
        base = self.root / "base" / f"{lang}.traineddata"
        if not latest or not latest["base_sha"] or not base.is_file():
            return None
        if file_fingerprint(base) == latest["base_sha"]:
            return None
        return (f"NOTE: base/{base.name} has changed since {latest['version']} was trained; "
                f"run train to build a version on the new base and compare.")

    def evaluate_version(self, version: str, log=print) -> tuple[float | None, float | None]:
        """Re-score a model and the stock base on that version's evaluation lines."""
        row = self.db.execute("SELECT * FROM models WHERE version = ?", (version,)).fetchone()
        model = self.root / "models" / f"{version}.traineddata"
        eval_list = self.root / "work" / version / "eval.txt"
        if row is None or not model.is_file():
            raise TrainingError(f"No trained model {version}.")
        if not eval_list.is_file() or not eval_list.read_text(encoding="utf-8").strip():
            raise TrainingError(f"No evaluation lines kept for {version}.")
        base = self.root / "base" / f"{row['lang']}.traineddata"
        cer_base = self._evaluate(base, eval_list, log)
        cer_new = self._evaluate(model, eval_list, log)
        with self.db:
            self.db.execute("UPDATE models SET cer_base = ?, cer_new = ? WHERE version = ?",
                            (cer_base, cer_new, version))
        return cer_base, cer_new

    # -- promote / rollback ---------------------------------------------------
    def promote(self, version: str) -> str:
        row = self.db.execute("SELECT * FROM models WHERE version = ?", (version,)).fetchone()
        source = self.root / "models" / f"{version}.traineddata"
        if row is None or not source.is_file():
            raise TrainingError(f"No trained model {version}.")
        tessdata = self.root / "tessdata"
        tessdata.mkdir(exist_ok=True)
        # The OCR folder must also hold the stock models used in combined languages (eng+rus).
        for stock in stock_tessdata().glob("*.traineddata"):
            target = tessdata / stock.name
            if not target.exists():
                shutil.copyfile(stock, target)
        shutil.copyfile(source, tessdata / source.name)
        lang = row["lang"]
        previous = self.current_model(lang)
        history = json.loads(self.get_meta(f"history:{lang}", "[]"))
        if previous and previous != version:
            history.append(previous)
        with self.db:
            self.db.execute("UPDATE models SET status = 'retired' WHERE lang = ? AND status = 'promoted'",
                            (lang,))
            self.db.execute("UPDATE models SET status = 'promoted' WHERE version = ?", (version,))
        self.set_meta(f"history:{lang}", json.dumps(history))
        self.set_meta(f"current_model:{lang}", version)
        return version

    def rollback(self, lang: str = "eng") -> str | None:
        """Go back to the previously promoted model (or the stock model). Returns the new current."""
        current = self.current_model(lang)
        if not current:
            raise TrainingError(f"No promoted '{lang}' model; the stock model is already in use.")
        history = json.loads(self.get_meta(f"history:{lang}", "[]"))
        previous = history.pop() if history else None
        with self.db:
            self.db.execute("UPDATE models SET status = 'retired' WHERE version = ?", (current,))
            if previous:
                self.db.execute("UPDATE models SET status = 'promoted' WHERE version = ?", (previous,))
        self.set_meta(f"history:{lang}", json.dumps(history))
        self.set_meta(f"current_model:{lang}", previous)
        return previous

    # -- purge ------------------------------------------------------------------
    def purge_pdf(self, pdf_path: Path) -> tuple[int, int]:
        """Remove a PDF's queue entries and line images. Returns (pages, lines) removed.
        Models already trained on those lines keep what they learned."""
        pdf_path = Path(pdf_path).resolve()
        conditions, params = ["pdf_path = ?"], [str(pdf_path)]
        if pdf_path.is_file():
            conditions.append("fingerprint = ?")
            params.append(file_fingerprint(pdf_path))
        pages = self.db.execute(
            f"SELECT id FROM pages WHERE {' OR '.join(conditions)}", params
        ).fetchall()
        page_ids = [r["id"] for r in pages]
        removed_lines = 0
        with self.db:
            for page_id in page_ids:
                for line in self.db.execute("SELECT image FROM lines WHERE page_id = ?", (page_id,)):
                    if line["image"]:
                        (self.root / line["image"]).unlink(missing_ok=True)
                removed_lines += self.db.execute("DELETE FROM lines WHERE page_id = ?",
                                                 (page_id,)).rowcount
                self.db.execute("DELETE FROM pages WHERE id = ?", (page_id,))
        return len(page_ids), removed_lines

    # -- status -----------------------------------------------------------------
    def status_text(self) -> str:
        out = [f"Training data folder: {self.root}"]
        pages = dict(self.db.execute("SELECT status, COUNT(*) FROM pages GROUP BY status").fetchall())
        out.append("Pages:  " + (", ".join(f"{k} {v}" for k, v in sorted(pages.items())) or "none"))
        langs = [r[0] for r in self.db.execute("SELECT DISTINCT lang FROM lines ORDER BY lang")]
        for lang in langs:
            counts = dict(self.db.execute(
                "SELECT status, COUNT(*) FROM lines WHERE lang = ? GROUP BY status", (lang,)
            ).fetchall())
            line = ", ".join(f"{s} {counts[s]}" for s in LINE_STATES if counts.get(s))
            out.append(f"Lines ({lang}): {line or 'none'}; "
                       f"confirmed and not yet trained: {self.untrained_confirmed(lang)}")
        for row in self.db.execute("SELECT * FROM models ORDER BY created_at"):
            cer = ""
            if row["cer_base"] is not None and row["cer_new"] is not None:
                cer = f", char error rate {row['cer_base']:.2f}% -> {row['cer_new']:.2f}%"
            out.append(f"Model {row['version']}: {row['status']}, {row['train_lines']} lines{cer}")
        for lang in sorted({*langs, "eng"}):
            current = self.current_model(lang)
            out.append(f"OCR model in use ({lang}): {current or 'stock'}")
            base_note = self.base_changed(lang)
            if base_note:
                out.append(base_note)
        job = self.get_meta("training_job")
        if job:
            info = json.loads(job)
            out.append(f"Training job: {info.get('state')} ({info.get('lang')}, started "
                       f"{info.get('started')}) {info.get('message', '')}".rstrip())
        reasons = self.due_reasons()
        out.append("Due: " + ("; ".join(reasons) if reasons else "nothing"))
        return "\n".join(out)


def _clean_text(text: str | None) -> str:
    return " ".join(unicodedata.normalize("NFC", text or "").split())


# ---------------------------------------------------------------------------
# Background training (used by the MCP server; long runs must not block a tool call)
# ---------------------------------------------------------------------------

def start_background_training(store: Store, lang: str, iterations: int | None) -> str:
    job = store.get_meta("training_job")
    if job and json.loads(job).get("state") == "running":
        raise TrainingError("A training job is already running; check status.")
    log_path = store.root / "work" / f"train_{lang}.log"
    log_path.parent.mkdir(exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    python = Path(sys.executable)
    if os.name == "nt" and (python.parent / "pythonw.exe").is_file():
        python = python.parent / "pythonw.exe"   # no console window
    cmd = [str(python), str(Path(__file__).resolve()), "--trn-dir", str(store.root),
           "train", "--lang", lang, "--job", "--log", str(log_path)]
    if iterations:
        cmd += ["--iterations", str(iterations)]
    store.set_meta("training_job", json.dumps(
        {"state": "running", "lang": lang, "started": _now(), "log": str(log_path)}))
    try:
        how = _spawn_detached(cmd, store.root)
    except Exception as exc:
        _finish_job(store, "failed", f"could not start: {exc}")
        raise TrainingError(f"Could not start training: {exc}") from exc
    return f"{log_path} ({how})"


def _spawn_detached(cmd: list[str], cwd: Path) -> str:
    """Start a process that outlives this one and its caller.

    MCP clients on Windows put the server's process tree in a job object that is killed
    when they disconnect, and do not allow breaking away from it. A process created by
    the WMI service is outside that tree, so training survives the session."""
    if os.name != "nt":
        subprocess.Popen(cmd, cwd=cwd, start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return "detached"
    command_line = subprocess.list2cmdline(cmd).replace("'", "''")
    folder = str(cwd).replace("'", "''")
    script = (
        "$r = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments "
        f"@{{CommandLine='{command_line}'; CurrentDirectory='{folder}'}}; "
        "Write-Output \"$($r.ReturnValue) $($r.ProcessId)\""
    )
    import base64
    encoded = base64.b64encode(script.encode("utf-16-le")).decode()
    proc = _run(["powershell", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded])
    parts = proc.stdout.split()
    if proc.returncode == 0 and len(parts) == 2 and parts[0] == "0":
        return f"process {parts[1]}"
    # Fallback: a detached child; it survives unless the caller kills its process tree.
    subprocess.Popen(cmd, cwd=cwd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL, close_fds=True,
                     creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS)
    return "fallback: stops if the session's process tree is closed"


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def _finish_job(store: Store, state: str, message: str) -> None:
    job = json.loads(store.get_meta("training_job") or "{}")
    job.update(state=state, finished=_now(), message=message)
    store.set_meta("training_job", json.dumps(job))


def _print_lines(lines: list[dict]) -> None:
    if not lines:
        print("No lines waiting for review.")
    for line in lines:
        print(f"[{line['id']}] {line['status']}  min conf {line['min_conf']}  ({line['lang']})")
        print(f"    image     : {line['image']}")
        print(f"    tesseract : {line['tesseract']}")
        if line["ai_text"]:
            print(f"    suggested : {line['ai_text']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pdfstotext-train",
        description="OCR training companion for pdfstotext. Data folder: "
                    f"${ENV_DIR} or ~/pdfstotext_trn.",
    )
    parser.add_argument("--trn-dir", type=Path, help=f"Data folder (overrides ${ENV_DIR}).")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init", help="Create the data folder (enables collection by pdfstotext).")
    sub.add_parser("status", help="Show queue, review and model status.")
    p = sub.add_parser("scan", help="Cut low-confidence lines from queued pages.")
    p.add_argument("--max-pages", type=int, default=20)
    p = sub.add_parser("next", help="List lines waiting for review.")
    p.add_argument("count", nargs="?", type=int, default=10)
    p.add_argument("--lang")
    p = sub.add_parser("suggest", help="Store an AI reading for a line.")
    p.add_argument("id", type=int)
    p.add_argument("text")
    p = sub.add_parser("confirm", help="Confirm lines (human-checked). Use --text to correct one line.")
    p.add_argument("ids", type=int, nargs="+")
    p.add_argument("--text")
    for name in ("skip", "illegible"):
        p = sub.add_parser(name, help=f"Mark lines as {name}.")
        p.add_argument("ids", type=int, nargs="+")
    p = sub.add_parser("train", help="Fine-tune a new model version from confirmed lines.")
    p.add_argument("--lang", default="eng")
    p.add_argument("--iterations", type=int)
    p.add_argument("--job", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--log", type=Path, help="Also write progress to this file.")
    p = sub.add_parser("evaluate", help="Re-score a model version against the stock model.")
    p.add_argument("version")
    p = sub.add_parser("promote", help="Make a model version the one pdfstotext uses.")
    p.add_argument("version")
    p = sub.add_parser("rollback", help="Return to the previous model (or the stock model).")
    p.add_argument("--lang", default="eng")
    sub.add_parser("models", help="List trained model versions.")
    p = sub.add_parser("purge", help="Remove a PDF's queue entries and line images.")
    p.add_argument("--pdf", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.trn_dir:
        os.environ[ENV_DIR] = str(args.trn_dir)

    if args.command == "init":
        store = Store.init(data_dir())
        print(f"Training data folder ready: {store.root}")
        print(f"Put tessdata_best models (e.g. eng.traineddata) in {store.root / 'base'}.")
        return 0

    with Store.open_existing() as store:
        cmd = args.command
        if cmd == "status":
            print(store.status_text())
        elif cmd == "scan":
            r = store.scan(args.max_pages)
            print(f"Scanned {r['pages']} pages, {r['lines']} low-confidence lines; "
                  f"missing {r['missing']}, changed {r['changed']}, failed {r['failed']}.")
        elif cmd == "next":
            _print_lines(store.next_lines(args.count, args.lang))
        elif cmd == "suggest":
            store.suggest(args.id, args.text)
            print(f"Line {args.id}: suggestion stored.")
        elif cmd == "confirm":
            if args.text is not None and len(args.ids) != 1:
                raise TrainingError("--text can only be used with a single line id.")
            for line_id in args.ids:
                print(f"Line {line_id} confirmed: {store.confirm(line_id, args.text)}")
        elif cmd in ("skip", "illegible"):
            for line_id in args.ids:
                store.set_status(line_id, "skipped" if cmd == "skip" else "illegible")
            print(f"{len(args.ids)} line(s) marked {cmd}.")
        elif cmd == "train":
            log_file = open(args.log, "a", encoding="utf-8") if args.log else None

            def log(message: str) -> None:
                print(message, flush=True)   # no-op under pythonw (no console)
                if log_file:
                    log_file.write(message + "\n")
                    log_file.flush()

            try:
                r = store.train(args.lang, args.iterations, log=log)
                message = (f"{r['version']}: {r['train_lines']} lines, {r['iterations']} iterations"
                           + (f", char error rate {r['cer_base']:.2f}% -> {r['cer_new']:.2f}%"
                              if r["cer_base"] is not None and r["cer_new"] is not None else ""))
                if args.job:
                    _finish_job(store, "done", message)
                log(f"Trained {message}")
                log(f"Promote with: pdfstotext-train promote {r['version']}")
            except Exception as exc:
                log(f"ERROR: {exc}")
                if args.job:
                    _finish_job(store, "failed", str(exc))
                raise
            finally:
                if log_file:
                    log_file.close()
        elif cmd == "evaluate":
            cer_base, cer_new = store.evaluate_version(args.version)
            print(f"{args.version}: char error rate stock {cer_base}% -> this model {cer_new}%")
        elif cmd == "promote":
            print(f"{store.promote(args.version)} is now used by pdfstotext.")
        elif cmd == "rollback":
            current = store.rollback(args.lang)
            print(f"Now using: {current or 'stock model'}")
        elif cmd == "models":
            for row in store.db.execute("SELECT * FROM models ORDER BY created_at"):
                print(dict(row))
        elif cmd == "purge":
            pages, lines = store.purge_pdf(args.pdf)
            print(f"Removed {pages} pages and {lines} lines.")
    return 0


if __name__ == "__main__":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    try:
        raise SystemExit(main())
    except TrainingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
