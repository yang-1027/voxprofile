"""Command-line interface for voxprofile."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from . import __version__
from .model import load_turns
from .render import Palette, render_replay, render_stats
from .stats import aggregate


def _color_enabled(no_color: bool, stream) -> bool:
    if no_color or os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def _reason(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _cmd_replay(args: argparse.Namespace) -> int:
    try:
        turns = load_turns(args.events)
    except OSError as exc:
        print(f"voxprofile: cannot read {args.events}: {_reason(exc)}", file=sys.stderr)
        return 2

    pal = Palette(_color_enabled(args.no_color, sys.stdout))
    print(render_replay(turns, args.target, args.events, pal))

    if args.html:
        if not turns:
            print(
                "voxprofile: no complete turns; skipping HTML export.",
                file=sys.stderr,
            )
        else:
            from .html import render_html

            # HTML is always fully styled, regardless of --no-color.
            doc = render_html(turns, args.target, args.events)
            try:
                with open(args.html, "w", encoding="utf-8") as fh:
                    fh.write(doc)
            except OSError as exc:
                print(
                    f"voxprofile: cannot write {args.html}: {_reason(exc)}",
                    file=sys.stderr,
                )
                return 2
            print(f"\nvoxprofile: wrote HTML report to {args.html}", file=sys.stderr)

    return 0 if turns else 1


def _cmd_stats(args: argparse.Namespace) -> int:
    pal = Palette(_color_enabled(args.no_color, sys.stdout))
    turns = []
    sources = []
    had_error = False
    for path in args.events:
        try:
            file_turns = load_turns(path)
        except OSError as exc:
            print(f"voxprofile: cannot read {path}: {_reason(exc)}", file=sys.stderr)
            had_error = True
            continue
        turns.extend(file_turns)
        sources.append(path)

    if not turns:
        print("voxprofile: no complete turns found.", file=sys.stderr)
        return 2 if had_error else 1

    rows = aggregate(turns)
    print(render_stats(rows, sources, len(turns), pal))
    return 2 if had_error else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voxprofile",
        description="Latency waterfall profiler for voice AI agent pipelines.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser(
        "replay",
        help="render per-turn latency waterfalls from a JSONL event log",
    )
    replay.add_argument("events", help="path to an events JSONL file")
    replay.add_argument(
        "--target",
        type=float,
        default=800.0,
        metavar="MS",
        help="total latency target in ms (default: 800)",
    )
    replay.add_argument(
        "--html",
        metavar="PATH",
        help="also write a self-contained HTML report to PATH",
    )
    replay.add_argument(
        "--no-color", action="store_true", help="disable ANSI colors"
    )
    replay.set_defaults(func=_cmd_replay)

    stats = sub.add_parser(
        "stats",
        help="aggregate p50/p95/min/max across one or more JSONL runs",
    )
    stats.add_argument(
        "events", nargs="+", help="one or more events JSONL files"
    )
    stats.add_argument(
        "--no-color", action="store_true", help="disable ANSI colors"
    )
    stats.set_defaults(func=_cmd_stats)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
