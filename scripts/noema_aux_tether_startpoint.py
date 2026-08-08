#!/usr/bin/env python3
"""NOEMA-Telekom host startpoint for the live 36D AUX stream.

Intended invocation is the same runpy/python3 -c pattern used by the NOEMA
session bootstrap. This process MUST run in the host namespace that owns the
real /dev/shm/ciel_noema surface. It never fabricates or reconstructs a surface.
"""
from __future__ import annotations

import argparse
import json
import signal
import sys
from pathlib import Path
from threading import Event

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from noema_telekom.aux_stream import (  # noqa: E402
    AuxStreamPublisher,
    FilePacketSink,
    NoemaSurfaceReader,
    TetherBlocked,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NOEMA-Telekom live AUX tether startpoint")
    p.add_argument("--root", default="/dev/shm/ciel_noema")
    p.add_argument("--outbox", default=None)
    p.add_argument("--interval-ms", type=float, default=50.0)
    p.add_argument("--socket", default="/run/user/1000/noema/ciel_headless.sock")
    p.add_argument("--socket-optional", action="store_true")
    p.add_argument("--max-frames", type=int, default=None)
    return p


def _emit_status(status: str, **fields: object) -> None:
    print(json.dumps({"TETHER_STATUS": status, **fields}, sort_keys=True), flush=True)


def main() -> int:
    args = build_parser().parse_args()
    if args.interval_ms <= 0:
        _emit_status("BLOCKED", reason="interval-ms must be > 0")
        return 2
    if args.max_frames is not None and args.max_frames <= 0:
        _emit_status("BLOCKED", reason="max-frames must be > 0")
        return 2

    root = Path(args.root)
    outbox = Path(args.outbox) if args.outbox else root / "telekom" / "aux_stream"
    required_socket = None if args.socket_optional else args.socket
    stop = Event()

    def _stop(_signum: int, _frame: object) -> None:
        stop.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        reader = NoemaSurfaceReader(root, required_socket=required_socket)
        sink = FilePacketSink(outbox)
        publisher = AuxStreamPublisher(
            reader,
            sink,
            interval_s=args.interval_ms / 1000.0,
        )

        # Fail closed before declaring ACTIVE: the first frame must be read from
        # the real surface, validated and durably emitted by Telekom.
        first = publisher.emit_once()
        _emit_status(
            "ACTIVE",
            root=str(root),
            outbox=str(outbox),
            stream_id=first.stream_id,
            source_host_id=first.source_host_id,
            seq=first.seq,
            tick=first.tick,
            frame_sha256=first.frame_sha256,
            read_set_sha256=first.read_set_sha256,
            dim=36,
        )

        remaining = None if args.max_frames is None else args.max_frames - 1
        if remaining == 0:
            return 0
        publisher.run(stop, max_frames=remaining)
        return 0
    except TetherBlocked as exc:
        _emit_status("BLOCKED", root=str(root), reason=str(exc))
        return 2
    except Exception as exc:
        _emit_status("BLOCKED", root=str(root), reason=f"unexpected: {type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
