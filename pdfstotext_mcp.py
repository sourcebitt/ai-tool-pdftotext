"""
pdfstotext_mcp.py — MCP server for pdfstotext OCR training
===========================================================

Exposes the training store (pdfstotext_trn.py) as tools, so an AI session can work
through low-confidence OCR lines with the user and train a custom Tesseract model.

Run (stdio):  python pdfstotext_mcp.py
Requires:     pip install mcp
Data folder:  $PDFSTOTEXT_TRN_DIR, or ~/pdfstotext_trn
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP, Image  # noqa: E402

import pdfstotext_trn as trn  # noqa: E402

MAX_BATCH = 25

mcp = FastMCP(
    "pdfstotext-train",
    log_level="WARNING",
    instructions=(
        "OCR training for pdfstotext. Use only in a session the user started for OCR training; "
        "never as a side task during other work.\n"
        "Workflow: status -> scan (cuts low-confidence lines from queued pages) -> next_lines "
        "(look at each line image) -> suggest (your reading) -> show the user a short numbered "
        "list of your readings, plus the lines you could not read -> confirm only what the user "
        "approved or corrected -> mark unreadable lines illegible -> train_start when status says "
        "training is due -> report the error rates -> promote only if the user agrees.\n"
        "Only user-confirmed text is used for training; never confirm your own reading without "
        "the user's approval in this conversation."
    ),
)


def _store() -> trn.Store:
    return trn.Store.open_existing()


def _err(exc: Exception) -> str:
    return f"ERROR: {exc}"


@mcp.tool()
def status() -> str:
    """Queue, review and model status, and whether review or training is due."""
    try:
        with _store() as store:
            return store.status_text()
    except trn.TrainingError as exc:
        return _err(exc)


@mcp.tool()
def scan(max_pages: int = 20) -> str:
    """Run Tesseract on queued pages and cut out lines with a low-confidence word.
    Takes roughly as long as OCR of those pages."""
    try:
        with _store() as store:
            r = store.scan(max(1, max_pages), log=lambda m: print(m, file=sys.stderr))
        return (f"Scanned {r['pages']} pages: {r['lines']} low-confidence lines. "
                f"Missing {r['missing']}, changed {r['changed']}, failed {r['failed']}.")
    except trn.TrainingError as exc:
        return _err(exc)


@mcp.tool(structured_output=False)
def next_lines(count: int = 10, lang: str | None = None) -> list:
    """The next lines to review, each with its image, Tesseract's reading and any earlier
    suggestion. Read each image yourself, then call suggest for it."""
    try:
        with _store() as store:
            lines = store.next_lines(min(max(1, count), MAX_BATCH), lang)
    except trn.TrainingError as exc:
        return [_err(exc)]
    if not lines:
        return ["No lines waiting for review. Run scan if pages are queued."]
    content: list = []
    for line in lines:
        content.append(json.dumps({k: line[k] for k in
                                   ("id", "lang", "status", "tesseract", "min_conf", "ai_text")},
                                  ensure_ascii=False))
        if line["image"] and Path(line["image"]).is_file():
            content.append(Image(path=line["image"]))
    return content


@mcp.tool()
def suggest(line_id: int, text: str) -> str:
    """Store your reading of a line. It is only a suggestion until the user confirms it."""
    try:
        with _store() as store:
            store.suggest(line_id, text)
        return f"Line {line_id}: suggestion stored."
    except trn.TrainingError as exc:
        return _err(exc)


@mcp.tool()
def awaiting_confirmation(lang: str | None = None, limit: int = 50) -> str:
    """Lines with a suggestion that the user has not confirmed yet (id, suggestion)."""
    try:
        with _store() as store:
            lines = store.awaiting_confirmation(lang, limit)
    except trn.TrainingError as exc:
        return _err(exc)
    return "\n".join(f"{l['id']}: {l['ai_text']}" for l in lines) or "Nothing awaiting confirmation."


@mcp.tool()
def confirm(line_ids: list[int], corrections: dict[str, str] | None = None) -> str:
    """Confirm lines as correct. Call ONLY after the user has approved these readings in this
    conversation. `corrections` maps a line id (as a string) to the text the user gave;
    other lines are confirmed with their stored suggestion."""
    corrections = corrections or {}
    done, errors = [], []
    try:
        with _store() as store:
            for line_id in line_ids:
                try:
                    store.confirm(line_id, corrections.get(str(line_id)))
                    done.append(line_id)
                except trn.TrainingError as exc:
                    errors.append(str(exc))
    except trn.TrainingError as exc:
        return _err(exc)
    return f"Confirmed {len(done)} line(s)." + ("" if not errors else " Errors: " + "; ".join(errors))


def _mark(line_ids: list[int], state: str) -> str:
    try:
        with _store() as store:
            for line_id in line_ids:
                store.set_status(line_id, state)
        return f"{len(line_ids)} line(s) marked {state}."
    except trn.TrainingError as exc:
        return _err(exc)


@mcp.tool()
def mark_illegible(line_ids: list[int]) -> str:
    """Mark lines that neither you nor the user can read. They are not used for training."""
    return _mark(line_ids, "illegible")


@mcp.tool()
def skip(line_ids: list[int]) -> str:
    """Skip lines that are not worth training on (not text, cut off, mixed content)."""
    return _mark(line_ids, "skipped")


@mcp.tool()
def train_start(lang: str = "eng", iterations: int | None = None) -> str:
    """Start fine-tuning a new model version in the background. Check progress with status.
    Takes minutes to about an hour."""
    try:
        with _store() as store:
            log = trn.start_background_training(store, lang, iterations)
        return f"Training started. Log: {log}. Check with status."
    except trn.TrainingError as exc:
        return _err(exc)


@mcp.tool()
def promote(version: str) -> str:
    """Make a trained model version the one pdfstotext uses. Only with the user's agreement."""
    try:
        with _store() as store:
            return f"{store.promote(version)} is now used by pdfstotext."
    except trn.TrainingError as exc:
        return _err(exc)


@mcp.tool()
def rollback(lang: str = "eng") -> str:
    """Return to the previously promoted model, or the stock model."""
    try:
        with _store() as store:
            current = store.rollback(lang)
        return f"Now using: {current or 'stock model'}"
    except trn.TrainingError as exc:
        return _err(exc)


if __name__ == "__main__":
    mcp.run()
