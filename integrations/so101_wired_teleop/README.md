# Wired SO-101 teleoperation

Optional integration package: run several SO-101 leader arms from one control
host and let them drive any number of follower arms over a dedicated wired link,
while recording episodes on the follower side.

It is installed separately from EmbodiRun:

```bash
pip install -e 'integrations/so101_wired_teleop[hardware,camera,test]'
```

## Why the port is the group key

A *leader* owns one arm, one UDP port and the addresses it broadcasts to. A
*follower* names the leader it obeys and listens on that leader's port. Because
the port — not the code — carries the grouping, every layout in the table below
is a configuration edit:

| Layout | What changes |
| --- | --- |
| One leader driving four followers | point four followers at that leader |
| Two leaders driving two followers each | point two followers at each |
| Any leader driving any follower | edit one `leader` field |
| A leader driving nothing | do not reference it |
| Every arm on one leader | give every follower the same `leader` |

`config.example.yaml` is the two-leaders-two-each case. It contains
`<PLACEHOLDER>` values on purpose: hosts, adapter serials and calibration paths
differ per site and must not be committed. `load_config` parses such a file, and
`TeleopConfig.require_resolved()` refuses to run while any placeholder remains,
so a half-edited file fails loudly instead of broadcasting to a literal
`<FOLLOWER_1_HOST>`.

## Layout

```yaml
leaders:
  - id: leader_a
    serial: /dev/serial/by-id/usb-1a86_USB_Single_Serial_<LEADER_A_SERIAL>-if00
    arm_id: <LEADER_A_ARM_ID>
    calibration_dir: <CALIBRATION_DIR>
    advertise: <LEADER_A_HOST>     # source address, and what followers filter on
    port: 55101                    # the group key; each leader needs its own
    fps: 20

followers:
  - id: follower_1
    leader: leader_a               # <- the whole mapping lives here
    bind: <FOLLOWER_1_HOST>
    ssh_host: <FOLLOWER_1_SSH_ALIAS>   # omit to run locally
    serial: /dev/serial/by-id/usb-1a86_USB_Single_Serial_<FOLLOWER_1_SERIAL>-if00
    arm_id: <FOLLOWER_1_ARM_ID>
    calibration_dir: <CALIBRATION_DIR>
    cameras: [0, 2]
    camera_roles: [main, wrist]
    record_dir: <FOLLOWER_1_RECORD_DIR>
```

## Running it

```bash
# One process per leader, on the host the leader arms are plugged into.
embodirun-so101-leader --config teleop.yaml --leader leader_a
embodirun-so101-leader --config teleop.yaml --leader leader_b

# One process per follower, on the host that follower is plugged into.
embodirun-so101-follower --config teleop.yaml --follower follower_1

# Operator console: pick a leader and a follower, then drive the recording.
embodirun-so101-collect --config teleop.yaml --task "pick up the block"
```

Before touching hardware, both ends can prove the path with `--network-test`,
which sends or waits for diagnostic packets and never opens an arm. Diagnostic
packets have a separate wire marker and are rejected by the motion receiver.

## Safety

The follower owns every limit, because the sender is not trusted to respect them:

| Limit | Default | Effect |
| --- | --- | --- |
| `max_step` | `3.0` | the commanded target advances at most this far per cycle |
| `max_lead` | `15.0` | the target may not run further ahead than this of the measured position |
| `watchdog_s` | `0.30` | a stalled stream holds the last position instead of releasing the arm |
| source filter | `advertise` | datagrams from any other address are discarded |
| queue drain | — | a burst is collapsed to its newest datagram, so stale targets are never replayed |

`--direct` bypasses `max_step` and `max_lead` and is for bench debugging only.

Duplicate and out-of-order sequence numbers are rejected, including across the
32-bit counter wrap. When restarting a leader process, restart its followers
as well to establish a new sequence baseline. Use a trusted, isolated wired
network: source-address filtering is not authentication.

## What an episode contains

```
<record_dir>/episode-<UTC stamp>/
    frames.jsonl     time, sequence, action (received), present, sent, cameras
    <role>/<ns>.jpg  one folder per configured camera role
    decision.json    keep or discard, written by the console
```

`action` is what the leader asked for, `sent` is what the follower actually
commanded after clamping. Keeping both is what makes a tracking error separable
from an operator error. Episode start and stop are driven by `SIGUSR1` and
`SIGUSR2`, so the console never has to hold a session open on the follower.

## Relationship to `integrations/xlerobot_owner`

Both packages control SO-101 followers from leader arms, and they are not
interchangeable:

| | `xlerobot_owner` | this package |
| --- | --- | --- |
| Placement | one host, leaders and followers together | leaders on one host, followers on many |
| Transport | the owner's own service | raw UDP, one port per leader |
| Leaders | a fixed left/right pair | any number, one process each |
| Mapping | fixed pairs | any leader to any follower, one to many, or none |
| Recording | owner-side episode recorder | follower-side, camera bytes never cross the link |

Pick `xlerobot_owner` for a single bimanual station; pick this package when the
arms are spread across hosts and the assignment has to change per session.

## Tests

```bash
pytest tests/so101_wired_teleop -q
```

The suite covers the layout rules, the placeholder guard, the packet framing, the
safety clamp and the episode bookkeeping. It needs no arm, camera or socket, so
it runs in the repository's CPU job alongside the other integration tests.
