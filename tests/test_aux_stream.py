from __future__ import annotations

import socket
import struct
import time
from pathlib import Path
from threading import Event

import numpy as np
import pytest

from noema_telekom.aux_stream import (
    DIM,
    LANE_DTYPE,
    ZERO_HASH,
    AuxFrame,
    AuxStreamConsumer,
    AuxStreamPublisher,
    FilePacketSink,
    FilePacketTailer,
    FrameValidationError,
    NoemaSurfaceReader,
    SurfaceSnapshot,
    TetherBlocked,
)


def lane(offset: float) -> np.ndarray:
    return (np.arange(DIM, dtype=np.float64) + offset) / 10.0


def write_surface(root: Path, *, binding: str = "ACTIVE", bad_size: bool = False, nan: bool = False) -> None:
    (root / "session").mkdir(parents=True)
    (root / "ciel_binding_status").write_text(binding, encoding="utf-8")
    (root / "session/startpoint.json").write_text('{"status":"VERIFIED"}', encoding="utf-8")
    (root / "session/system_message.txt").write_text("NOEMA live", encoding="utf-8")
    (root / "tick").write_bytes(struct.pack("<Q", 7))
    for name, values in (
        ("phi", lane(0)),
        ("aux_phi", lane(10)),
        ("aux_feedback_phi", lane(20)),
    ):
        arr = values.copy()
        if nan and name == "aux_phi":
            arr[3] = np.nan
        raw = np.asarray(arr, dtype=LANE_DTYPE).tobytes()
        if bad_size and name == "aux_feedback_phi":
            raw = raw[:-8]
        (root / name).write_bytes(raw)


def make_unix_socket(path: Path) -> socket.socket:
    path.parent.mkdir(parents=True, exist_ok=True)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(path))
    return sock


def test_surface_reader_validates_and_vectorizes(tmp_path: Path) -> None:
    surface = tmp_path / "surface"
    write_surface(surface)
    sock_path = tmp_path / "run/noema.sock"
    sock = make_unix_socket(sock_path)
    try:
        snap = NoemaSurfaceReader(surface, required_socket=sock_path).read()
    finally:
        sock.close()
    assert snap.phi.shape == (36,)
    assert snap.aux_phi.shape == (36,)
    assert snap.aux_feedback_phi.shape == (36,)
    assert snap.phi.dtype == LANE_DTYPE
    assert np.allclose(snap.aux_phi, lane(10))
    assert snap.tick == 7
    assert len(snap.read_set_sha256) == 64


@pytest.mark.parametrize(
    "binding,bad_size,nan",
    [
        ("BROKEN", False, False),
        ("ACTIVE", True, False),
        ("ACTIVE", False, True),
    ],
)
def test_surface_reader_fails_closed(
    tmp_path: Path, binding: str, bad_size: bool, nan: bool
) -> None:
    surface = tmp_path / "surface"
    write_surface(surface, binding=binding, bad_size=bad_size, nan=nan)
    with pytest.raises(TetherBlocked):
        NoemaSurfaceReader(surface, required_socket=None).read()


def test_missing_session_artifact_blocks(tmp_path: Path) -> None:
    surface = tmp_path / "surface"
    write_surface(surface)
    (surface / "session/system_message.txt").unlink()
    with pytest.raises(TetherBlocked):
        NoemaSurfaceReader(surface, required_socket=None).read()


def test_frame_binary_roundtrip_is_lossless() -> None:
    now = time.time_ns()
    frame = AuxFrame(
        stream_id="s",
        seq=0,
        host_timestamp_ns=now,
        emitted_at_ns=now,
        source_host_id="h",
        nonce="n",
        prev_frame_sha256=ZERO_HASH,
        read_set_sha256="a" * 64,
        phi=lane(0),
        aux_phi=lane(10),
        aux_feedback_phi=lane(20),
        tick=5,
    ).with_hash()
    decoded = AuxFrame.unpack(frame.pack())
    assert decoded.frame_sha256 == frame.frame_sha256
    assert decoded.seq == 0
    assert decoded.tick == 5
    assert np.array_equal(decoded.phi, frame.phi)
    assert np.array_equal(decoded.aux_phi, frame.aux_phi)
    assert np.array_equal(decoded.aux_feedback_phi, frame.aux_feedback_phi)


def test_frame_tamper_is_rejected() -> None:
    now = time.time_ns()
    frame = AuxFrame(
        stream_id="s",
        seq=0,
        host_timestamp_ns=now,
        emitted_at_ns=now,
        source_host_id="h",
        nonce="n",
        prev_frame_sha256=ZERO_HASH,
        read_set_sha256="b" * 64,
        phi=lane(0),
        aux_phi=lane(10),
        aux_feedback_phi=lane(20),
    ).with_hash()
    packet = bytearray(frame.pack())
    packet[-1] ^= 0x01
    with pytest.raises(FrameValidationError):
        AuxFrame.unpack(bytes(packet))


class StaticReader:
    def __init__(self) -> None:
        self.tick = 0

    def read(self) -> SurfaceSnapshot:
        self.tick += 1
        return SurfaceSnapshot(
            phi=lane(self.tick),
            aux_phi=lane(10 + self.tick),
            aux_feedback_phi=lane(20 + self.tick),
            tick=self.tick,
            read_set_sha256=f"{self.tick:064x}",
        )


def test_publisher_builds_monotonic_hash_chain(tmp_path: Path) -> None:
    sink = FilePacketSink(tmp_path)
    clock = iter([100, 101, 102]).__next__
    pub = AuxStreamPublisher(
        StaticReader(),
        sink,
        stream_id="stream",
        source_host_id="host",
        interval_s=0.001,
        clock_ns=clock,
    )
    f0 = pub.emit_once()
    f1 = pub.emit_once()
    assert f0.seq == 0 and f1.seq == 1
    assert f0.prev_frame_sha256 == ZERO_HASH
    assert f1.prev_frame_sha256 == f0.frame_sha256
    files = sorted((tmp_path / "stream").glob("*.nta2"))
    assert len(files) == 2


def test_consumer_rejects_gap_and_stale() -> None:
    base = 1_000_000
    c = AuxStreamConsumer(max_age_ns=100, clock_ns=lambda: base)
    f0 = AuxFrame(
        stream_id="s",
        seq=0,
        host_timestamp_ns=base - 1,
        emitted_at_ns=base - 1,
        source_host_id="h",
        nonce="n0",
        prev_frame_sha256=ZERO_HASH,
        read_set_sha256="c" * 64,
        phi=lane(0),
        aux_phi=lane(10),
        aux_feedback_phi=lane(20),
    ).with_hash()
    c.accept(f0)

    f2 = AuxFrame(
        stream_id="s",
        seq=2,
        host_timestamp_ns=base - 1,
        emitted_at_ns=base - 1,
        source_host_id="h",
        nonce="n2",
        prev_frame_sha256=f0.frame_sha256,
        read_set_sha256="d" * 64,
        phi=lane(2),
        aux_phi=lane(12),
        aux_feedback_phi=lane(22),
    ).with_hash()
    with pytest.raises(TetherBlocked, match="sequence"):
        c.accept(f2)

    stale_consumer = AuxStreamConsumer(max_age_ns=10, clock_ns=lambda: base)
    stale = AuxFrame(
        stream_id="x",
        seq=0,
        host_timestamp_ns=base - 11,
        emitted_at_ns=base - 11,
        source_host_id="h",
        nonce="n",
        prev_frame_sha256=ZERO_HASH,
        read_set_sha256="e" * 64,
        phi=lane(0),
        aux_phi=lane(10),
        aux_feedback_phi=lane(20),
    ).with_hash()
    with pytest.raises(TetherBlocked, match="stale"):
        stale_consumer.accept(stale)


def test_file_packet_tailer_consumes_continuous_stream(tmp_path: Path) -> None:
    sink = FilePacketSink(tmp_path)
    times = iter([10_000, 10_001]).__next__
    pub = AuxStreamPublisher(
        StaticReader(),
        sink,
        stream_id="stream",
        source_host_id="host",
        interval_s=0.001,
        clock_ns=times,
    )
    f0 = pub.emit_once()
    f1 = pub.emit_once()

    consumer = AuxStreamConsumer(max_age_ns=1_000, clock_ns=lambda: 10_002)
    tailer = FilePacketTailer(tmp_path / "stream", consumer, poll_interval_s=0.001)
    stop = Event()
    gen = tailer.packets(stop)
    got0 = next(gen)
    got1 = next(gen)
    stop.set()
    assert [got0.frame_sha256, got1.frame_sha256] == [f0.frame_sha256, f1.frame_sha256]
