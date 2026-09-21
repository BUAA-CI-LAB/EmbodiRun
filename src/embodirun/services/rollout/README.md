# Rollout communication and replay

This package is a model-independent boundary for the RLinf integration. It does
not use the physical `ControlRuntime` and contains no robot action executor.

- `ReplaySource` reads JSONL observations or supplies synthetic RGB bytes and
  state values. It has no dynamics or reward model.
- `StreamSession` adds a `embodirun.rollout.v1` envelope with a run session,
  route and contiguous sequence number. Both the Channel and Wireless comparison
  paths use the same envelope. A mismatch raises `StreamError`; the caller must
  reset its trajectory and start a new session after interruption.
- `WirelessEndpoint` owns a persistent asynchronous WirelessComm endpoint and
  returns concurrent futures. Callers register their own transition codecs.
  Unsafe pickle is disabled. A send future denotes local transmission, not a
  remote application acknowledgement. Close the endpoint when its owner exits.

RLinf supplies the native Channel adapter, tensor/transition types, routing keys,
weight synchronization, policy inference and experiment metrics. Deploy does not
import any of those frameworks. The replay benchmark uses fixed inputs and an
echo policy; real-model and LAN performance require separate measurements.

Run `pytest tests/test_rollout_stream.py tests/test_boundaries.py` with the
`wireless` extra installed. Tests use loopback peers, including disconnect,
healthy-peer progress, restart and rejection of old session/sequence messages.
These do not establish automatic recovery of a full RL training episode.
