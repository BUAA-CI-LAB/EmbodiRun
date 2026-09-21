"""Bounded STS3215 read transport and one-shot read-only diagnostics.

The caller owns the hardware I/O lock. Use single-device READ responses, not
GroupSyncRead (the deployed SDK drops device errors and can return missing data
as zero). The SDK validates packet ID/checksum; we additionally check its result,
device error and exact payload length before decoding any field. The block
reader also serves strict wheel preflight; diagnostics themselves grant no
control and make no claim of stationary state.
"""

from __future__ import annotations

import time
from typing import Any


def _word(data: list[int], offset: int = 0, *, signed: bool = False) -> int:
    value = data[offset] | (data[offset + 1] << 8)
    # STS3215 uses sign-magnitude, not two's complement. Preserve negative-zero
    # and reserved bytes in raw_bytes even though the decoded value is zero.
    return -(value & 0x7FFF) if signed and value & 0x8000 else value


def _read(packet: Any, port: Any, motor_id: int, label: str, address: int, length: int) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": label,
        "motor_id": motor_id,
        "address": address,
        "length": length,
        "started_timestamp_ns": time.time_ns(),
        "started_monotonic_ns": time.monotonic_ns(),
        "comm_result": None,
        "device_error": None,
        "raw_bytes": None,
        "ok": False,
        "errors": [],
        "fields": {},
    }
    try:
        data, comm, error = packet.readTxRx(port, motor_id, address, length)
        # These are SDK integer codes, not booleans or fabricated success values.
        if type(comm) is int:
            record["comm_result"] = comm
        if type(error) is int:
            record["device_error"] = error
        if type(comm) is not int or comm != 0:
            record["errors"].append(f"communication failed: {comm!r}")
        if type(error) is not int or error != 0:
            record["errors"].append(f"device error: {error!r}")
        if not isinstance(data, (list, tuple, bytes, bytearray)) or any(
            type(byte) is not int or not 0 <= byte <= 255 for byte in data
        ):
            record["errors"].append("invalid payload bytes")
        else:
            record["raw_bytes"] = list(data)
            if len(data) != length:
                record["errors"].append(f"payload length {len(data)}, expected {length}")
        record["ok"] = not record["errors"]
    except Exception as exc:  # noqa: BLE001 - diagnostics retain transport failures
        record["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        record["finished_monotonic_ns"] = time.monotonic_ns()
    return record


def read_sts3215_present_block(packet: Any, port: Any, motor_id: int) -> dict[str, Any]:
    """Read 56..66 in one response; do not claim firmware atomic latching."""
    block = _read(packet, port, motor_id, "present_block", 56, 11)
    if block["ok"]:
        data = block["raw_bytes"]
        block["fields"] = {
            "Present_Position": _word(data, signed=True),
            "Present_Velocity": _word(data, 2, signed=True),
            "Present_Load": _word(data, 4),
            "Present_Voltage": data[6],
            "Present_Temperature": data[7],
            "Status": data[9],
            "Moving": data[10],
        }
    return block


def read_sts3215_diagnostics(packet: Any, port: Any, motor_id: int) -> dict[str, Any]:
    """Eight fixed reads at most; no retries, register writes or state inference."""
    records = [_read(packet, port, motor_id, "identity", 3, 2)]
    result = {
        "purpose": "diagnostic_only",
        "read_only": True,
        "register_writes": 0,
        "cached": False,
        "atomic_snapshot": False,
        "field_units": "raw",
        "packet_validation": "SDK packet ID/checksum plus comm/error/payload validation",
        "transactions": records,
        "ok": False,
    }
    identity = records[0]
    if not identity["ok"]:
        return result
    model = _word(identity["raw_bytes"])
    identity["fields"] = {"Model_Number": model}
    if model != 777:
        identity["ok"] = False
        identity["errors"].append(f"model {model}, expected STS3215 model 777")
        return result

    block = read_sts3215_present_block(packet, port, motor_id)
    records.append(block)
    # Separate transactions: never present these as simultaneous with 56..66.
    for label, address, length, signed in (
        ("Goal_Velocity", 46, 2, True),
        ("Operating_Mode", 33, 1, False),
        ("Torque_Enable", 40, 1, False),
        ("Present_Current", 69, 2, False),
    ):
        record = _read(packet, port, motor_id, label, address, length)
        records.append(record)
        if record["ok"]:
            record["fields"] = {
                label: (_word(record["raw_bytes"], signed=signed) if length == 2 else record["raw_bytes"][0]),
            }
    for label, address, names in (
        ("firmware", 0, ("Firmware_Major_Version", "Firmware_Minor_Version")),
        ("velocity_estimator", 80, ("Moving_Velocity_Threshold", "DTs", "Velocity_Unit_factor")),
    ):
        record = _read(packet, port, motor_id, label, address, len(names))
        records.append(record)
        if record["ok"]:
            record["fields"] = dict(zip(names, record["raw_bytes"], strict=True))
        # Register interpretation may depend on firmware. Report bytes, never
        # infer that a zero value proves feature support or a velocity scale.
        record["interpretation"] = "raw SDK table fields; firmware support/units not inferred"
    result["ok"] = all(record["ok"] for record in records)
    return result
