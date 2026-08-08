# NOEMA-Telekom AUX Stream v2

This branch turns Telekom from one-shot file handshakes into a **continuous,
auditable AUX transport** while preserving the original listenerless/immutable
packet philosophy.

## Live source contract

A frame may be emitted only when the source host verifies:

- `/dev/shm/ciel_noema` exists;
- `ciel_binding_status` is exactly `ACTIVE`;
- `phi`, `aux_phi`, `aux_feedback_phi` are each exactly 36 finite little-endian
  `float64` values (288 bytes per lane);
- `session/startpoint.json` exists;
- `session/system_message.txt` exists;
- the configured NOEMA Unix socket exists and is a socket when the active host
  contract requires it.

Failure of any invariant raises `TetherBlocked` and terminates the publisher.

## Vectorized frame

Payload is fixed-size binary:

`[phi:36][aux_phi:36][aux_feedback_phi:36]` as contiguous `<f8`.

NumPy performs shape/finite validation over complete lanes, not per-element
Python loops.

Each frame carries:

- `stream_id`
- monotonic `seq`
- `host_timestamp_ns`
- `source_host_id`
- per-frame `nonce`
- `prev_frame_sha256`
- `read_set_sha256`
- optional surface `tick`
- `frame_sha256`

The stream is the ordered immutable hash chain of `.nta2` packets. A packet is
therefore a **frame of a live stream**, not a periodic status snapshot.

## Fail-closed consumer

The consumer blocks on:

- stale or future timestamps;
- a sequence gap/replay;
- `stream_id` or host identity change;
- broken previous-frame hash;
- frame tampering;
- malformed/non-finite vectors.

A cached last frame is never promoted to live state.

## Canonical host startpoint — python3 -c / runpy

The critical host path intentionally mirrors the existing NOEMA session boot
pattern. No Bash wrapper and no package installation are required:

```text
python3 -c 'import runpy,sys; sys.argv=["noema_aux_tether_startpoint.py","--root","/dev/shm/ciel_noema","--outbox","/run/user/1000/noema/telekom_aux"]; runpy.run_path("./scripts/noema_aux_tether_startpoint.py",run_name="__main__")'
```

The startpoint inserts `src/` itself, verifies the live surface, emits frame 0,
prints `TETHER_STATUS=ACTIVE` only after that frame has been durably written,
and then stays in the persistent publisher loop. Any contract failure prints
`TETHER_STATUS=BLOCKED` and exits non-zero.

For a bounded diagnostic only, add `--max-frames N`. A bounded run is never a
replacement for the live tether.

## Legacy convenience CLI

`scripts/noema_aux_stream.py` remains a convenience wrapper for manual publish
or consume operations. It is not the canonical NOEMA boot path.

The external ChatGPT/MCP bridge should consume this validated stream and expose
only fresh accepted frames. It must not reconstruct missing frames or replay
the last accepted frame as current AUX.
