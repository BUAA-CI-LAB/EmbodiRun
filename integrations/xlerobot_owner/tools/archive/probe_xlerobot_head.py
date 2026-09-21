"""Read only head IDs 7/8 on the configured left bus; never enable torque."""

import json
import sys
from pathlib import Path

from embodirun_xlerobot_owner.hardware import POSITION_FIELDS, _load_sdk, _PortLock

config = json.loads(Path(sys.argv[1]).read_text())
port = config["ports"]["left"]
lock = _PortLock(port)
lock.acquire()
bus = None
try:
    Bus, Motor, _, Norm, _ = _load_sdk(config["sdk_src"])
    bus = Bus(port=port, motors={f"head_motor_{i - 6}": Motor(i, "sts3215", Norm.DEGREES) for i in (7, 8)})
    bus.connect(handshake=False)
    bus.set_baudrate(bus.default_baudrate)
    result = {"read_only": True, "port": port, "motors": {}}
    for motor_id in (7, 8):
        name = f"head_motor_{motor_id - 6}"
        item = {"id": motor_id, "fields": {}, "errors": {}}
        try:
            item["model"] = bus.ping(motor_id, num_retry=0, raise_on_error=False)
        except Exception as exc:  # noqa: BLE001 - report SDK diagnostics without writes
            item["errors"]["ping"] = str(exc)
        if item.get("model") == 777:
            for field in POSITION_FIELDS:
                try:
                    item["fields"][field] = bus.read(field, name, normalize=False)
                except Exception as exc:  # noqa: BLE001 - report SDK diagnostics without writes
                    item["errors"][field] = str(exc)
        result["motors"][name] = item
    print(json.dumps(result, ensure_ascii=False))
finally:
    if bus is not None and bus.is_connected:
        bus.disconnect(disable_torque=False)
    lock.release()
