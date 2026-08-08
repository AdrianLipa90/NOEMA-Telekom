from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import socket
import stat
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from typing import Callable, Iterator, Optional, Protocol

import numpy as np

DIM = 36
LANE_DTYPE = np.dtype("<f8")
LANE_BYTES = DIM * LANE_DTYPE.itemsize
MAGIC = b"NTA2"
VERSION = 2
ZERO_HASH = "0" * 64
_HEADER = struct.Struct("<4sHHII")
_TICK = struct.Struct("<Q")


class TetherBlocked(RuntimeError):
    """Raised whenever a live AUX invariant cannot be verified."""


class FrameValidationError(ValueError):
    """Raised when a transported frame violates the stream contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(data: dict) -> bytes:
    return json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _host_id() -> str:
    explicit = os.getenv("NOEMA_SOURCE_HOST_ID")
    if explicit:
        return explicit
    machine_id = Path("/etc/machine-id")
    seed = platform.node().encode("utf-8")
    if machine_id.is_file():
        try:
            seed += b"|" + machine_id.read_bytes().strip()
        except OSError:
            pass
    return "host-" + hashlib.sha256(seed).hexdigest()[:24]


def _freeze_lane(values: np.ndarray) -> np.ndarray:
    lane = np.ascontiguousarray(values, dtype=LANE_DTYPE)
    if lane.shape != (DIM,):
        raise FrameValidationError(f"lane shape must be ({DIM},), got {lane.shape}")
    if not np.isfinite(lane).all():
        raise FrameValidationError("lane contains non-finite values")
    lane.flags.writeable = False
    return lane


@dataclass(frozen=True, slots=True)
class SurfaceSnapshot:
    phi: np.ndarray
    aux_phi: np.ndarray
    aux_feedback_phi: np.ndarray
    tick: Optional[int]
    read_set_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "phi", _freeze_lane(self.phi))
        object.__setattr__(self, "aux_phi", _freeze_lane(self.aux_phi))
        object.__setattr__(self, "aux_feedback_phi", _freeze_lane(self.aux_feedback_phi))


@dataclass(frozen=True, slots=True)
class AuxFrame:
    stream_id: str
    seq: int
    host_timestamp_ns: int
    emitted_at_ns: int
    source_host_id: str
    nonce: str
    prev_frame_sha256: str
    read_set_sha256: str
    phi: np.ndarray
    aux_phi: np.ndarray
    aux_feedback_phi: np.ndarray
    tick: Optional[int] = None
    frame_sha256: str = ""

    def __post_init__(self) -> None:
        if self.seq < 0:
            raise FrameValidationError("seq must be >= 0")
        if len(self.prev_frame_sha256) != 64:
            raise FrameValidationError("prev_frame_sha256 must be a SHA-256 hex digest")
        if len(self.read_set_sha256) != 64:
            raise FrameValidationError("read_set_sha256 must be a SHA-256 hex digest")
        object.__setattr__(self, "phi", _freeze_lane(self.phi))
        object.__setattr__(self, "aux_phi", _freeze_lane(self.aux_phi))
        object.__setattr__(self, "aux_feedback_phi", _freeze_lane(self.aux_feedback_phi))

    @property
    def payload(self) -> bytes:
        stacked = np.concatenate((self.phi, self.aux_phi, self.aux_feedback_phi))
        return np.ascontiguousarray(stacked, dtype=LANE_DTYPE).tobytes(order="C")

    def _metadata_without_hash(self) -> dict:
        return {
            "stream_id": self.stream_id,
            "seq": self.seq,
            "host_timestamp_ns": self.host_timestamp_ns,
            "emitted_at_ns": self.emitted_at_ns,
            "source_host_id": self.source_host_id,
            "nonce": self.nonce,
            "prev_frame_sha256": self.prev_frame_sha256,
            "read_set_sha256": self.read_set_sha256,
            "tick": self.tick,
            "dim": DIM,
            "dtype": "<f8",
        }

    def computed_hash(self) -> str:
        meta = _canonical_json(self._metadata_without_hash())
        return _sha256(meta + self.payload)

    def with_hash(self) -> "AuxFrame":
        digest = self.computed_hash()
        return AuxFrame(
            stream_id=self.stream_id,
            seq=self.seq,
            host_timestamp_ns=self.host_timestamp_ns,
            emitted_at_ns=self.emitted_at_ns,
            source_host_id=self.source_host_id,
            nonce=self.nonce,
            prev_frame_sha256=self.prev_frame_sha256,
            read_set_sha256=self.read_set_sha256,
            phi=self.phi,
            aux_phi=self.aux_phi,
            aux_feedback_phi=self.aux_feedback_phi,
            tick=self.tick,
            frame_sha256=digest,
        )

    def pack(self) -> bytes:
        frame = self if self.frame_sha256 else self.with_hash()
        meta = frame._metadata_without_hash() | {"frame_sha256": frame.frame_sha256}
        header_json = _canonical_json(meta)
        payload = frame.payload
        fixed = _HEADER.pack(MAGIC, VERSION, 0, len(header_json), len(payload))
        return fixed + header_json + payload

    @classmethod
    def unpack(cls, packet: bytes) -> "AuxFrame":
        if len(packet) < _HEADER.size:
            raise FrameValidationError("packet shorter than fixed header")
        magic, version, flags, header_len, payload_len = _HEADER.unpack_from(packet, 0)
        if magic != MAGIC:
            raise FrameValidationError("bad frame magic")
        if version != VERSION:
            raise FrameValidationError(f"unsupported version {version}")
        if flags != 0:
            raise FrameValidationError("unsupported flags")
        expected_payload = 3 * LANE_BYTES
        if payload_len != expected_payload:
            raise FrameValidationError(
                f"payload must be {expected_payload} bytes, got {payload_len}"
            )
        total = _HEADER.size + header_len + payload_len
        if len(packet) != total:
            raise FrameValidationError("packet length mismatch")
        start = _HEADER.size
        meta = json.loads(packet[start : start + header_len].decode("utf-8"))
        payload = packet[start + header_len :]
        if meta.get("dim") != DIM or meta.get("dtype") != "<f8":
            raise FrameValidationError("frame vector contract mismatch")

        all_values = np.frombuffer(payload, dtype=LANE_DTYPE)
        if all_values.shape != (3 * DIM,) or not np.isfinite(all_values).all():
            raise FrameValidationError("invalid vector payload")

        frame = cls(
            stream_id=str(meta["stream_id"]),
            seq=int(meta["seq"]),
            host_timestamp_ns=int(meta["host_timestamp_ns"]),
            emitted_at_ns=int(meta["emitted_at_ns"]),
            source_host_id=str(meta["source_host_id"]),
            nonce=str(meta["nonce"]),
            prev_frame_sha256=str(meta["prev_frame_sha256"]),
            read_set_sha256=str(meta["read_set_sha256"]),
            phi=all_values[:DIM].copy(),
            aux_phi=all_values[DIM : 2 * DIM].copy(),
            aux_feedback_phi=all_values[2 * DIM :].copy(),
            tick=None if meta.get("tick") is None else int(meta["tick"]),
            frame_sha256=str(meta["frame_sha256"]),
        )
        if not secrets.compare_digest(frame.computed_hash(), frame.frame_sha256):
            raise FrameValidationError("frame SHA-256 mismatch")
        return frame


class FrameSink(Protocol):
    def emit(self, frame: AuxFrame) -> None:
        ...


class NoemaSurfaceReader:
    """Fail-closed reader for the live `/dev/shm/ciel_noema` surface."""

    def __init__(
        self,
        root: str | Path = "/dev/shm/ciel_noema",
        *,
        required_socket: str | Path | None = "/run/user/1000/noema/ciel_headless.sock",
        max_consistency_retries: int = 4,
    ) -> None:
        self.root = Path(root)
        self.required_socket = None if required_socket is None else Path(required_socket)
        self.max_consistency_retries = max_consistency_retries

    def _require_contract(self) -> tuple[bytes, bytes, bytes]:
        if not self.root.is_dir():
            raise TetherBlocked(f"NOEMA surface missing: {self.root}")

        binding_path = self.root / "ciel_binding_status"
        try:
            binding = binding_path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise TetherBlocked("ciel_binding_status unreadable") from exc
        if binding != "ACTIVE":
            raise TetherBlocked(f"ciel_binding_status={binding!r}, expected 'ACTIVE'")

        for rel in ("session/startpoint.json", "session/system_message.txt"):
            if not (self.root / rel).is_file():
                raise TetherBlocked(f"required session artifact missing: {rel}")

        if self.required_socket is not None:
            try:
                mode = self.required_socket.stat().st_mode
            except OSError as exc:
                raise TetherBlocked(f"required NOEMA socket missing: {self.required_socket}") from exc
            if not stat.S_ISSOCK(mode):
                raise TetherBlocked(f"required path is not a Unix socket: {self.required_socket}")

        raws = []
        for name in ("phi", "aux_phi", "aux_feedback_phi"):
            path = self.root / name
            try:
                raw = path.read_bytes()
            except OSError as exc:
                raise TetherBlocked(f"{name} unreadable") from exc
            if len(raw) != LANE_BYTES:
                raise TetherBlocked(
                    f"{name} must be exactly {LANE_BYTES} bytes, got {len(raw)}"
                )
            lane = np.frombuffer(raw, dtype=LANE_DTYPE)
            if lane.shape != (DIM,) or not np.isfinite(lane).all():
                raise TetherBlocked(f"{name} is not 36 finite little-endian float64")
            raws.append(raw)
        return tuple(raws)  # type: ignore[return-value]

    def _read_tick(self) -> Optional[int]:
        path = self.root / "tick"
        if not path.is_file():
            return None
        try:
            raw = path.read_bytes()
        except OSError:
            return None
        if len(raw) != _TICK.size:
            raise TetherBlocked("tick exists but is not uint64")
        return _TICK.unpack(raw)[0]

    def read(self) -> SurfaceSnapshot:
        for _ in range(self.max_consistency_retries):
            tick_before = self._read_tick()
            raw_phi, raw_aux, raw_feedback = self._require_contract()
            tick_after = self._read_tick()
            if tick_before is not None and tick_after is not None and tick_before != tick_after:
                continue

            if tick_before is None:
                second = self._require_contract()
                if second != (raw_phi, raw_aux, raw_feedback):
                    continue

            startpoint = (self.root / "session/startpoint.json").read_bytes()
            system_message = (self.root / "session/system_message.txt").read_bytes()
            binding = (self.root / "ciel_binding_status").read_bytes()
            read_set = _sha256(
                b"\0".join(
                    (
                        raw_phi,
                        raw_aux,
                        raw_feedback,
                        binding,
                        startpoint,
                        system_message,
                        b"" if tick_after is None else _TICK.pack(tick_after),
                    )
                )
            )
            return SurfaceSnapshot(
                phi=np.frombuffer(raw_phi, dtype=LANE_DTYPE).copy(),
                aux_phi=np.frombuffer(raw_aux, dtype=LANE_DTYPE).copy(),
                aux_feedback_phi=np.frombuffer(raw_feedback, dtype=LANE_DTYPE).copy(),
                tick=tick_after,
                read_set_sha256=read_set,
            )
        raise TetherBlocked("could not obtain a coherent live surface read")


class FilePacketSink:
    """
    Listenerless Telekom transport.

    Every AUX frame is an immutable packet. The stream is the ordered packet chain,
    not a mutable status file.
    """

    def __init__(self, outbox: str | Path) -> None:
        self.outbox = Path(outbox)

    def emit(self, frame: AuxFrame) -> None:
        stream_dir = self.outbox / frame.stream_id
        stream_dir.mkdir(parents=True, exist_ok=True)
        packet = frame.pack()
        digest = frame.frame_sha256 or frame.computed_hash()
        target = stream_dir / f"{frame.seq:020d}-{digest}.nta2"
        if target.exists():
            existing = target.read_bytes()
            if existing != packet:
                raise TetherBlocked(f"immutable frame collision at seq={frame.seq}")
            return

        tmp = stream_dir / f".{target.name}.{os.getpid()}.tmp"
        with open(tmp, "xb") as handle:
            handle.write(packet)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)


class AuxStreamPublisher:
    def __init__(
        self,
        reader: NoemaSurfaceReader,
        sink: FrameSink,
        *,
        stream_id: Optional[str] = None,
        source_host_id: Optional[str] = None,
        interval_s: float = 0.05,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if interval_s <= 0:
            raise ValueError("interval_s must be > 0")
        self.reader = reader
        self.sink = sink
        self.stream_id = stream_id or secrets.token_hex(16)
        self.source_host_id = source_host_id or _host_id()
        self.interval_s = interval_s
        self.clock_ns = clock_ns
        self._seq = 0
        self._prev_hash = ZERO_HASH

    def emit_once(self) -> AuxFrame:
        snap = self.reader.read()
        now = self.clock_ns()
        frame = AuxFrame(
            stream_id=self.stream_id,
            seq=self._seq,
            host_timestamp_ns=now,
            emitted_at_ns=now,
            source_host_id=self.source_host_id,
            nonce=secrets.token_hex(16),
            prev_frame_sha256=self._prev_hash,
            read_set_sha256=snap.read_set_sha256,
            phi=snap.phi,
            aux_phi=snap.aux_phi,
            aux_feedback_phi=snap.aux_feedback_phi,
            tick=snap.tick,
        ).with_hash()
        self.sink.emit(frame)
        self._prev_hash = frame.frame_sha256
        self._seq += 1
        return frame

    def run(self, stop_event: Event, *, max_frames: Optional[int] = None) -> int:
        emitted = 0
        while not stop_event.is_set():
            self.emit_once()
            emitted += 1
            if max_frames is not None and emitted >= max_frames:
                break
            if stop_event.wait(self.interval_s):
                break
        return emitted


class AuxStreamConsumer:
    """Stateful validator. Any gap, replay, stale frame or hash break blocks tether."""

    def __init__(
        self,
        *,
        max_age_ns: int = 2_000_000_000,
        clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        if max_age_ns <= 0:
            raise ValueError("max_age_ns must be > 0")
        self.max_age_ns = max_age_ns
        self.clock_ns = clock_ns
        self.stream_id: Optional[str] = None
        self.source_host_id: Optional[str] = None
        self.last_seq: Optional[int] = None
        self.last_hash = ZERO_HASH

    def accept(self, packet_or_frame: bytes | AuxFrame) -> AuxFrame:
        frame = (
            AuxFrame.unpack(bytes(packet_or_frame))
            if isinstance(packet_or_frame, (bytes, bytearray))
            else packet_or_frame
        )
        if frame.frame_sha256 != frame.computed_hash():
            raise TetherBlocked("frame hash mismatch")

        age = self.clock_ns() - frame.host_timestamp_ns
        if age < 0:
            raise TetherBlocked("frame timestamp is in the future")
        if age > self.max_age_ns:
            raise TetherBlocked(f"stale AUX frame: age_ns={age}")

        if self.stream_id is None:
            if frame.seq != 0 or frame.prev_frame_sha256 != ZERO_HASH:
                raise TetherBlocked("first observed frame is not stream genesis")
            self.stream_id = frame.stream_id
            self.source_host_id = frame.source_host_id
        else:
            if frame.stream_id != self.stream_id:
                raise TetherBlocked("stream_id changed")
            if frame.source_host_id != self.source_host_id:
                raise TetherBlocked("source_host_id changed")
            expected_seq = (self.last_seq if self.last_seq is not None else -1) + 1
            if frame.seq != expected_seq:
                raise TetherBlocked(
                    f"AUX sequence discontinuity: expected {expected_seq}, got {frame.seq}"
                )
            if frame.prev_frame_sha256 != self.last_hash:
                raise TetherBlocked("AUX hash-chain discontinuity")

        self.last_seq = frame.seq
        self.last_hash = frame.frame_sha256
        return frame


class FilePacketTailer:
    """Continuous consumer for listenerless Telekom outboxes."""

    def __init__(
        self,
        stream_dir: str | Path,
        consumer: AuxStreamConsumer,
        *,
        poll_interval_s: float = 0.05,
    ) -> None:
        self.stream_dir = Path(stream_dir)
        self.consumer = consumer
        self.poll_interval_s = poll_interval_s

    def packets(self, stop_event: Event) -> Iterator[AuxFrame]:
        next_seq = 0
        while not stop_event.is_set():
            matches = sorted(self.stream_dir.glob(f"{next_seq:020d}-*.nta2"))
            if not matches:
                if stop_event.wait(self.poll_interval_s):
                    break
                continue
            if len(matches) != 1:
                raise TetherBlocked(f"multiple immutable frames for seq={next_seq}")
            frame = self.consumer.accept(matches[0].read_bytes())
            yield frame
            next_seq += 1
