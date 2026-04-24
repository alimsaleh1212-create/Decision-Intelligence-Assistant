#!/usr/bin/env python3
"""
ingest_all.py — Drive the full DIA ingest pipeline with live progress.

Loops POST /api/ingest → polls GET /api/ingest/status until done.
Stdlib only (urllib + json) — no extra dependencies needed.

Usage:
    python scripts/ingest_all.py                        # run full ingest
    python scripts/ingest_all.py --host http://host:8000
    python scripts/ingest_all.py --status               # show status and exit
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime


# ── ANSI colours (disabled automatically if not a TTY) ───────────────────────

_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def gold(t: str) -> str:   return _c("93", t)
def teal(t: str) -> str:   return _c("96", t)
def red(t: str) -> str:    return _c("91", t)
def dim(t: str) -> str:    return _c("2",  t)
def bold(t: str) -> str:   return _c("1",  t)
def green(t: str) -> str:  return _c("92", t)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


def _post(url: str, body: dict | None = None) -> dict:
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


# ── Progress bar ──────────────────────────────────────────────────────────────

def _bar(cursor: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return dim("[" + "░" * width + "]")
    filled = int(width * cursor / total)
    pct = cursor / total * 100
    bar = gold("█" * filled) + dim("░" * (width - filled))
    return f"[{bar}] {pct:5.1f}%"


# ── Status display ────────────────────────────────────────────────────────────

def _fmt_status(st: dict) -> str:
    status    = st.get("status", "?")
    cursor    = st.get("cursor", 0)
    total     = st.get("total_chunks", 0)
    batches   = st.get("batch_count", 0)
    embedded  = st.get("threads_embedded", 0)
    qdrant    = st.get("qdrant_count", 0)
    error_msg = st.get("error")

    bar = _bar(cursor, total)
    status_col = {
        "done":    green(status),
        "error":   red(status),
        "running": teal(status),
        "loading": teal(status),
        "ready":   gold(status),
        "idle":    dim(status),
    }.get(status, status)

    parts = [
        f"status={status_col}",
        bar,
        f"cursor={teal(str(cursor))}/{dim(str(total))}",
        f"batches={gold(str(batches))}",
        f"embedded={gold(str(embedded))}",
        f"qdrant={teal(str(qdrant))}",
    ]
    line = "  ".join(parts)
    if error_msg:
        line += f"\n  {red('ERROR:')} {error_msg}"
    return line


# ── Poll until not running/loading ────────────────────────────────────────────

def _wait_for_batch(base: str, poll_interval: float = 2.0) -> dict:
    """Poll /api/ingest/status until status is no longer running or loading."""
    spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    i = 0
    while True:
        st = _get(f"{base}/api/ingest/status")
        status = st.get("status", "")
        if status not in ("running", "loading"):
            # Clear spinner line
            if _USE_COLOR:
                sys.stdout.write("\r\033[K")
                sys.stdout.flush()
            return st
        # Spinner
        spin = teal(spinner[i % len(spinner)])
        cursor = st.get("cursor", 0)
        total  = st.get("total_chunks", 0)
        pct = f"{cursor/total*100:.1f}%" if total else "…"
        sys.stdout.write(f"\r  {spin} {dim('processing…')}  {pct}")
        sys.stdout.flush()
        i += 1
        time.sleep(poll_interval)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Run DIA full ingest with live progress")
    parser.add_argument("--host", default="http://localhost:8000",
                        help="Backend base URL (default: http://localhost:8000)")
    parser.add_argument("--status", action="store_true",
                        help="Print current ingest status and exit")
    parser.add_argument("--poll", type=float, default=2.0,
                        help="Status poll interval in seconds (default: 2)")
    args = parser.parse_args()

    base = args.host.rstrip("/")

    # ── Connectivity check ────────────────────────────────────────────────────
    try:
        st = _get(f"{base}/api/ingest/status")
    except (urllib.error.URLError, ConnectionRefusedError) as exc:
        print(red("✗ Cannot reach backend:"), str(exc))
        print(dim(f"  Is the backend running at {base}?"))
        return 1

    # ── --status mode: just print and exit ───────────────────────────────────
    if args.status:
        print(bold("DIA Ingest Status"))
        print(_fmt_status(st))
        return 0

    # ── Guard: don't start if already running ─────────────────────────────────
    current_status = st.get("status", "idle")
    if current_status == "running":
        print(gold("⚡ Ingest is already running:"))
        print(_fmt_status(st))
        print(dim("  Re-run with --status to monitor, or wait for it to finish."))
        return 0

    if current_status == "done":
        print(green("✓ Ingest already complete."))
        print(_fmt_status(st))
        print(dim("  Use POST /api/ingest/reset to restart from scratch."))
        return 0

    if current_status == "error":
        print(red("✗ Previous ingest ended in error:"))
        print(_fmt_status(st))
        print(dim("  Use POST /api/ingest/reset to clear and retry."))
        return 1

    # ── Run ───────────────────────────────────────────────────────────────────
    print(bold("DIA Full Ingest") + "  " + dim(f"→ {base}"))
    print(dim(f"  Started at {datetime.now().strftime('%H:%M:%S')}"))
    print(dim("  Press Ctrl-C to stop (ingest state is preserved)"))
    print()

    t_start = time.monotonic()
    batch_num = 0

    try:
        while True:
            t_batch = time.monotonic()

            # Trigger next batch
            try:
                _post(f"{base}/api/ingest")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                print(red(f"✗ POST /api/ingest failed ({exc.code}):"), body)
                return 1

            # Wait for it to finish
            st = _wait_for_batch(base, poll_interval=args.poll)
            status = st.get("status", "")
            batch_num += 1
            elapsed_batch = time.monotonic() - t_batch
            elapsed_total = time.monotonic() - t_start

            cursor   = st.get("cursor", 0)
            total    = st.get("total_chunks", 0)
            embedded = st.get("threads_embedded", 0)
            qdrant   = st.get("qdrant_count", 0)
            bar      = _bar(cursor, total)

            status_icon = {
                "ready": gold("✓"),
                "done":  green("✓✓"),
                "error": red("✗"),
            }.get(status, dim("?"))

            print(
                f"  {status_icon} "
                f"{bold(f'Batch {batch_num:>3}')}  "
                f"{bar}  "
                f"cursor={teal(str(cursor))}/{dim(str(total))}  "
                f"embedded={gold(str(embedded))}  "
                f"qdrant={teal(str(qdrant))}  "
                f"{dim(f'+{elapsed_batch:.1f}s / {elapsed_total:.0f}s total')}"
            )

            if status == "done":
                total_elapsed = time.monotonic() - t_start
                print()
                print(green(bold("✓ Ingest complete!")))
                print(
                    f"  {gold(str(batch_num))} batches  ·  "
                    f"{gold(str(embedded))} chunks embedded  ·  "
                    f"{teal(str(qdrant))} points in Qdrant  ·  "
                    f"{dim(f'{total_elapsed:.0f}s elapsed')}"
                )
                return 0

            if status == "error":
                print()
                print(red("✗ Ingest failed:"), st.get("error", "unknown error"))
                return 1

    except KeyboardInterrupt:
        print()
        print(dim("  Interrupted — fetching final state…"))
        try:
            st = _get(f"{base}/api/ingest/status")
            print(_fmt_status(st))
        except Exception:
            pass
        return 130  # conventional SIGINT exit code


if __name__ == "__main__":
    sys.exit(main())
