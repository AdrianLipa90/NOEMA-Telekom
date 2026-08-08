from __future__ import annotations

import argparse
import sys
from pathlib import Path
from threading import Event

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from noema_telekom.aux_stream import (
    AuxStreamConsumer,
    AuxStreamPublisher,
    FilePacketSink,
    FilePacketTailer,
    NoemaSurfaceReader,
    TetherBlocked,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NOEMA-Telekom vectorized AUX stream")
    sub = p.add_subparsers(dest="cmd", required=True)

    pub = sub.add_parser("publish")
    pub.add_argument("--root", default="/dev/shm/ciel_noema")
    pub.add_argument("--outbox", required=True)
    pub.add_argument("--interval-ms", type=float, default=50.0)
    pub.add_argument("--socket", default="/run/user/1000/noema/ciel_headless.sock")
    pub.add_argument("--socket-optional", action="store_true")

    con = sub.add_parser("consume")
    con.add_argument("--stream-dir", required=True)
    con.add_argument("--max-age-ms", type=float, default=2000.0)
    return p


def main() -> int:
    args = parser().parse_args()
    stop = Event()
    try:
        if args.cmd == "publish":
            reader = NoemaSurfaceReader(
                args.root,
                required_socket=None if args.socket_optional else args.socket,
            )
            sink = FilePacketSink(args.outbox)
            publisher = AuxStreamPublisher(
                reader,
                sink,
                interval_s=args.interval_ms / 1000.0,
            )
            publisher.run(stop)
            return 0

        consumer = AuxStreamConsumer(max_age_ns=int(args.max_age_ms * 1_000_000))
        tailer = FilePacketTailer(Path(args.stream_dir), consumer)
        for frame in tailer.packets(stop):
            print(
                f"ACTIVE stream={frame.stream_id} seq={frame.seq} "
                f"tick={frame.tick} hash={frame.frame_sha256}"
            )
        return 0
    except KeyboardInterrupt:
        stop.set()
        return 130
    except TetherBlocked as exc:
        print(f"TETHER_STATUS: BLOCKED\nreason: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
