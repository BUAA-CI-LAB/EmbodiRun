import re
import shutil
import subprocess
from pathlib import Path

import pytest

WEB_APP = (
    Path(__file__).parents[2]
    / "integrations"
    / "xlerobot_owner"
    / "src"
    / "embodirun_xlerobot_owner"
    / "web"
    / "app.js"
)


def _extract_function(source: str, name: str) -> str | None:
    """Extract one real top-level function without depending on its formatting."""
    match = re.search(
        rf"(?m)^[ \t]*(?:async\s+)?function\s+{re.escape(name)}\s*\(",
        source,
    )
    if match is None:
        return None
    opening = source.find("{", match.end())
    if opening < 0:
        return None

    depth = 0
    quote: str | None = None
    escaped = False
    line_comment = False
    block_comment = False
    index = opening
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if line_comment:
            if char == "\n":
                line_comment = False
            index += 1
            continue
        if block_comment:
            if char == "*" and next_char == "/":
                block_comment = False
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
            continue
        if char == "/" and next_char == "/":
            line_comment = True
            index += 2
            continue
        if char == "/" and next_char == "*":
            block_comment = True
            index += 2
            continue
        if char in "'\"`":
            quote = char
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[match.start() : index + 1].strip()
        index += 1
    return None


def _function_bundle(source: str, roots: tuple[str, ...], excluded: set[str] | None = None) -> str | None:
    """Collect actual functions and their function-call dependencies for Node."""
    names = dict.fromkeys(re.findall(r"(?m)^[ \t]*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(", source))
    functions = {name: _extract_function(source, name) for name in names}
    excluded = excluded or set()
    if any(not functions.get(root) for root in roots):
        return None

    selected: list[str] = []
    pending = list(roots)
    while pending:
        name = pending.pop()
        if name in selected or name in excluded:
            continue
        function = functions.get(name)
        if not function:
            continue
        selected.append(name)
        for dependency in re.findall(r"\b([A-Za-z_$][\w$]*)\s*\(", function):
            if dependency in functions and dependency not in selected and dependency not in excluded:
                pending.append(dependency)
    return "\n\n".join(functions[name] for name in selected)


def _run_node(script: str) -> None:
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for browser controller-input regression tests")
    result = subprocess.run(
        [node, "-e", script],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def _required_function_bundle(source: str, roots: tuple[str, ...], excluded: set[str] | None = None) -> str:
    bundle = _function_bundle(source, roots, excluded=excluded)
    assert bundle is not None, f"Required app.js function missing: {', '.join(roots)}"
    return bundle


def test_read_controller_preserves_named_dompoint_pose_and_rejects_nan_pose():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _function_bundle(
        source,
        (
            "readController",
            "makeController",
            "clamp",
            "finiteVector",
            "normalizedQuaternion",
            "readGamepadButton",
            "readGamepadButtons",
        ),
    )
    if bundle is None:
        pytest.skip("readController or a supported helper is not present in app.js")
    script = (
        "const assert = require('node:assert/strict');\n"
        f"{bundle}\n"
        "function button(pressed = false, value = 0) { return {pressed, value}; }\n"
        "function gamepad() { return {buttons: Array.from({length: 7}, () => button()), axes: [0, 0, 0, 0]}; }\n"
        "const rightSource = {handedness: 'right', gamepad: gamepad(), gripSpace: {id: 'right-grip'}};\n"
        "const translated = {x: 1.25, y: -0.5, z: 2.75};\n"
        "const rotated = {x: 0, y: 0.6, z: 0, w: 0.8};\n"
        "const frame = {getPose(space) {\n"
        "  assert.equal(space, rightSource.gripSpace);\n"
        "  return {transform: {position: translated, orientation: rotated}};\n"
        "}};\n"
        "const controller = readController(rightSource, frame, {});\n"
        "assert.equal(controller.tracked, true);\n"
        "assert.deepEqual(controller.position, [1.25, -0.5, 2.75]);\n"
        "assert.deepEqual(controller.orientation, [0, 0.6, 0, 0.8]);\n"
        "const nanFrame = {getPose() {\n"
        "  return {transform: {position: {x: Number.NaN, y: -0.5, z: 2.75}, orientation: rotated}};\n"
        "}};\n"
        "const nanController = readController(rightSource, nanFrame, {});\n"
        "assert.equal(nanController.tracked, false, 'NaN pose must remain untracked');\n"
        "const invalidQuaternionFrame = {getPose() {\n"
        "  return {transform: {position: translated, orientation: {x: 0, y: 0, z: 0, w: 2}}};\n"
        "}};\n"
        "assert.equal(readController(rightSource, invalidQuaternionFrame, {}).tracked, false);\n"
    )
    _run_node(script)


def test_both_quest_thumbsticks_survive_reading_and_serialization():
    source = WEB_APP.read_text()
    bundle = _required_function_bundle(source, ("readController", "sanitizeController"))
    _run_node(
        """
      const assert = require('node:assert/strict');
      const app = {gripperDirections:{left:'close', right:'open'}};
    """
        + bundle
        + """
      for (const hand of ['left', 'right']) {
        const source={handedness:hand,gamepad:{buttons:[],axes:[0,0,-0.7,0.3]},gripSpace:{}};
        const frame={getPose:()=>({transform:{position:{x:0,y:1,z:-.5},orientation:{x:0,y:0,z:0,w:1}}})};
        const c=readController(source,frame,{});
        assert.equal(c.tracked,true);
        assert.deepEqual(c.thumbstick,[-0.7,0.3]);
        assert.deepEqual(sanitizeController(c,hand).thumbstick,[-0.7,0.3]);
      }
    """
    )


def test_mapping_hint_separates_joint_limit_from_normal_speed_following():
    bundle = _required_function_bundle(WEB_APP.read_text(), ("mappingHint",))
    _run_node(
        """
      const assert = require('node:assert/strict');
      const app = {status:{mapping_status:{left:{active:true,trigger_ready:true,
        limited:true,speed_limited_joints:['left_arm_shoulder_lift.pos']}}}};
    """
        + bundle
        + """
      assert.match(mappingHint(),/按设定速度跟随中/);
      assert.doesNotMatch(mappingHint(),/边界|超过|限位/);
      app.status.mapping_status.left.joint_limit_joints=['left_arm_shoulder_lift.pos'];
      assert.match(mappingHint(),/左肩抬升到边界/);
      assert.match(mappingHint(),/其他轴仍可动/);
      app.status.mapping_status.left.joint_limit_joints=[];
      app.status.mapping_status.left.trigger_ready=false;
      assert.match(mappingHint(),/先松开扳机/);
    """
    )


def test_grip_edge_activates_without_hold_and_y_b_only_toggle_local_claws():
    bundle = _required_function_bundle(
        WEB_APP.read_text(),
        ("updateQuestButtonActions",),
        excluded={"manualStop", "requestArm"},
    )
    _run_node(
        """
      const assert = require('node:assert/strict');
      const XR_STOP_BUTTON_INDEX=4, XR_GRIPPER_TOGGLE_BUTTON_INDEX=5;
      let arms=0, stops=0;
      function c(grip=false) {return {grip, buttons:Array(7).fill(false)};}
      const app={status:{armed:false},armRequestPending:false,
        controllers:{left:c(true),right:c()},xr:{gripActivationReady:false,stopButtonWasDown:false},
        gripperDirections:{left:'close',right:'close'},gripperToggleWasDown:{left:false,right:false}};
      function requestArm(label, activation) {assert.equal(activation,'grip');arms++;}
      function manualStop() {stops++;}
    """
        + bundle
        + """
      updateQuestButtonActions(0);
      assert.equal(arms,0,'already-held Grip at entry must not activate');
      app.controllers.left.grip=false;
      updateQuestButtonActions(1);
      app.controllers.left.grip=true;
      updateQuestButtonActions(2);
      assert.equal(arms,1,'a new Grip edge activates immediately, no one-second timer');
      updateQuestButtonActions(3);
      assert.equal(arms,1,'held Grip must not repeatedly arm');
      app.controllers.left.buttons[5]=true;
      updateQuestButtonActions(4);
      assert.deepEqual(app.gripperDirections,{left:'open',right:'close'});
      assert.equal(stops,0,'Y/B no longer mean stop');
      updateQuestButtonActions(5);
      assert.equal(app.gripperDirections.left,'open','held Y must not toggle repeatedly');
      app.controllers.right.buttons[5]=true;
      updateQuestButtonActions(6);
      assert.deepEqual(app.gripperDirections,{left:'open',right:'open'});
      app.controllers.left.buttons[4]=true;
      updateQuestButtonActions(7);
      assert.equal(stops,1,'X must stop immediately');
      assert.equal(app.xr.gripActivationReady,false);
      app.controllers.left.buttons[4]=false;
      updateQuestButtonActions(8);
      assert.equal(arms,1,'continuing to hold Grip after Stop must not re-arm');
      app.controllers.left.grip=false;updateQuestButtonActions(9);
      app.controllers.left.grip=true;updateQuestButtonActions(10);
      assert.equal(arms,2);
    """
    )


def test_arm_readiness_hint_prioritizes_failure_pending_and_latched_chord():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_function_bundle(source, ("armReadinessHint",))
    script = (
        "const assert = require('node:assert/strict');\n"
        "global.WebSocket = {OPEN: 1};\n"
        "global.performance = {now: () => 2000};\n"
        "const CAMERA_NAMES = ['front', 'left_wrist', 'right_wrist'];\n"
        "const VIDEO_STALE_MS = 2500;\n"
        "const cameras = Object.fromEntries(CAMERA_NAMES.map(name => [name, {hasFrame:true, lastGoodAt:2000}]));\n"
        "function controller() {\n"
        "  return {tracked: true, grip: false, trigger: 0, thumbstick: [0, 0]};\n"
        "}\n"
        "const app = {\n"
        "  ws: {readyState: 1},\n"
        "  status: {stop_unconfirmed: false, control_owner: null, metadata: {allow_motion: true}},\n"
        "  armFailure: null,\n"
        "  armRequestPending: false,\n"
        "  controllers: {left: controller(), right: controller()},\n"
        "  xr: {safetyTripped: false, gripActivationReady: true},\n"
        "  video: {state:'connected'},\n"
        "};\n" + bundle + "\n"
        "app.armFailure = '机械臂启用被服务端拒绝';\n"
        "app.armRequestPending = true;\n"
        "const pendingHint = armReadinessHint();\n"
        "assert.equal(pendingHint, '启用请求已发送，等待 AGX 确认');\n"
        "assert.match(pendingHint, /等待/);\n"
        "assert.doesNotMatch(pendingHint, /0\\.0/);\n"
        "app.armRequestPending = false;\n"
        "const failureHint = armReadinessHint();\n"
        "assert.equal(failureHint, '启用被拒绝：机械臂启用被服务端拒绝');\n"
        "assert.doesNotMatch(failureHint, /0\\.0/);\n"
        "app.armFailure = null;\n"
        "app.xr.gripActivationReady = false; app.controllers.left.grip = true;\n"
        "const latchedHint = armReadinessHint();\n"
        "assert.equal(latchedHint, '先松开 Grip，再握住接管');\n"
        "assert.doesNotMatch(latchedHint, /0\\.0/);\n"
        "app.xr.session = {visibilityState:'visible', inputSources:[]};\n"
        "assert.match(armReadinessHint(), /浏览器未提供手柄输入源/);\n"
        "app.xr.session.inputSources = [{}, {}];\n"
        "cameras.front.hasFrame = false;\n"
        "assert.match(armReadinessHint(), /等待三路实时画面/);\n"
    )
    _run_node(script)


def test_apply_status_keeps_pending_for_unarmed_idle_feedback():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_function_bundle(
        source,
        ("applyStatus",),
        excluded={"updateServerIssues", "updateAllUi"},
    )
    script = (
        "const assert = require('node:assert/strict');\n"
        "global.WebSocket = {OPEN: 1};\n"
        "global.performance = {now: () => 123};\n"
        "const app = {\n"
        "  ws: {readyState: 1},\n"
        "  status: {armed: false, control_owner: null, stats: {}},\n"
        "  statusSource: 'none',\n"
        "  statusUpdatedAt: 0,\n"
        "  armRequestPending: true,\n"
        "  armFailure: '启用被拒绝：关节位置超限',\n"
        "  modeRequestPending: null,\n"
        "};\n"
        "function updateServerIssues() {}\n"
        "function updateAllUi() {}\n" + bundle + "\n"
        "applyStatus({armed: false}, 'ws');\n"
        "assert.equal(app.armRequestPending, true, 'idle feedback must not cancel a pending Arm request');\n"
        "assert.equal(app.armFailure, '启用被拒绝：关节位置超限');\n"
        "applyStatus({armed: true}, 'ws');\n"
        "assert.equal(app.armRequestPending, false);\n"
        "assert.equal(app.armFailure, null);\n"
    )
    _run_node(script)


def test_grip_input_bypasses_frame_interval_but_not_transport_failure():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_function_bundle(
        source,
        ("sendInputFrame",),
        excluded={
            "isWebSocketOpen",
            "failSafeStop",
            "xrInputDiagnostics",
            "sanitizeController",
        },
    )
    _run_node(
        """
      const assert = require('node:assert/strict');
      const INPUT_INTERVAL_MS = 1000/30, MAX_CONTROL_BUFFER_BYTES = 16384;
      let now = 100, stops = 0;
      global.performance = {now:()=>now};
      const packets=[];
      const app={lastInputSentAt:95, inputSeq:1, inputPackets:1,
        ws:{bufferedAmount:0,send:(p)=>packets.push(JSON.parse(p))},
        controllers:{left:{grip:true,gripper_direction:'open'},right:{grip:false}},
      };
      function isWebSocketOpen(){return true;}
      function failSafeStop(){stops++;}
      function xrInputDiagnostics(){return {};}
      function sanitizeController(c){return c;}
    """
        + bundle
        + """
      assert.equal(sendInputFrame(now),false);
      assert.equal(sendInputFrame(now,true),true);
      assert.equal(packets.length,1);
      assert.equal(packets[0].controllers.left.grip,true);
      assert.equal(packets[0].controllers.left.gripper_direction,'open');
      assert.equal(app.lastInputSentAt,100);
      now=110;
      assert.equal(sendInputFrame(now),false);
      app.ws.bufferedAmount=MAX_CONTROL_BUFFER_BYTES+1;
      assert.equal(sendInputFrame(now,true),false);
      assert.equal(stops,1);
      assert.equal(packets.length,1);
    """
    )


def test_request_arm_clears_failure_only_after_success_and_keeps_gates():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_function_bundle(
        source,
        ("requestArm", "canArm", "controllersAreNeutral"),
        excluded={"logEvent", "updateAllUi"},
    )
    script = (
        "const assert = require('node:assert/strict');\n"
        "global.WebSocket = {OPEN: 1};\n"
        "global.performance = {now: () => 200};\n"
        "const CAMERA_NAMES = ['front', 'left_wrist', 'right_wrist'];\n"
        "const VIDEO_STALE_MS = 2500;\n"
        "const cameras = Object.fromEntries(CAMERA_NAMES.map(name => [name, {hasFrame:true, lastGoodAt:100}]));\n"
        "const INPUT_FRESH_MS = 350;\n"
        "const sentMessages = [];\n"
        "let sendMode = 'ok';\n"
        "const ws = {readyState: 1, send(message) {\n"
        "  if (sendMode === 'throw') throw new Error('socket closed');\n"
        "  sentMessages.push(JSON.parse(message));\n"
        "}};\n"
        "function controller(overrides = {}) {\n"
        "  return {tracked: true, grip: false, trigger: 0, thumbstick: [0, 0], ...overrides};\n"
        "}\n"
        "const app = {\n"
        "  authenticated: true,\n"
        "  ws,\n"
        "  status: {armed: false, stop_unconfirmed: false, control_owner: null, metadata: {allow_motion: true}},\n"
        "  xr: {active: true, safetyTripped: false},\n"
        "  video: {state:'connected'},\n"
        "  inputSocket: ws,\n"
        "  lastInputSentAt: 100,\n"
        "  controllerClient: false,\n"
        "  controllers: {left: controller(), right: controller()},\n"
        "};\n"
        "const state = {armFailure: '上一次启用被拒绝', armRequestPending: false};\n"
        "const writes = [];\n"
        "Object.defineProperties(app, {\n"
        "  armFailure: {get: () => state.armFailure, set(value) {\n"
        "    writes.push(['armFailure', value]); state.armFailure = value;\n"
        "  }},\n"
        "  armRequestPending: {get: () => state.armRequestPending, set(value) {\n"
        "    writes.push(['armRequestPending', value]); state.armRequestPending = value;\n"
        "  }},\n"
        "});\n"
        "function logEvent() {}\n"
        "function updateAllUi() {}\n" + bundle + "\n"
        "assert.equal(controllersAreNeutral(), true);\n"
        "assert.equal(canArm(), true, 'an arm failure must not add a canArm gate');\n"
        "cameras.front.hasFrame = false;\n"
        "assert.equal(canArm(), false, 'idle recovery must not allow arming without video');\n"
        "cameras.front.hasFrame = true;\n"
        "app.video.state = 'error';\n"
        "assert.equal(canArm(), false, 'recent last frame does not make a failed video peer ready');\n"
        "app.video.state = 'connected';\n"
        "for (const blocked of [\n"
        "  {tracked: false},\n"
        "  {grip: true},\n"
        "  {trigger: 0.2},\n"
        "  {thumbstick: [0.2, 0]},\n"
        "]) {\n"
        "  app.controllers.left = controller(blocked);\n"
        "  app.controllers.right = controller();\n"
        "  sentMessages.length = 0; writes.length = 0;\n"
        "  state.armFailure = '保留失败提示'; state.armRequestPending = false;\n"
        "  assert.equal(controllersAreNeutral(), false);\n"
        "  assert.equal(canArm(), false);\n"
        "  assert.equal(requestArm('blocked'), false);\n"
        "  assert.deepEqual(sentMessages, []);\n"
        "  assert.equal(state.armRequestPending, false);\n"
        "}\n"
        "app.controllers.left = controller();\n"
        "app.controllers.right = controller();\n"
        "state.armFailure = '保留失败提示'; state.armRequestPending = false;\n"
        "writes.length = 0; sentMessages.length = 0; sendMode = 'ok';\n"
        "assert.equal(requestArm('retry'), true);\n"
        "assert.deepEqual(sentMessages, [{type: 'arm'}]);\n"
        "assert.deepEqual(writes.slice(0, 2), [\n"
        "  ['armFailure', null],\n"
        "  ['armRequestPending', true],\n"
        "]);\n"
        "assert.equal(state.armFailure, null);\n"
        "assert.equal(state.armRequestPending, true);\n"
        "state.armFailure = '新的失败提示'; state.armRequestPending = false;\n"
        "writes.length = 0; sentMessages.length = 0; sendMode = 'throw';\n"
        "assert.equal(requestArm('send failure'), false);\n"
        "assert.deepEqual(sentMessages, []);\n"
        "assert.equal(state.armRequestPending, false, 'a failed send must not become pending');\n"
        "assert.equal(writes.some(([name, value]) => name === 'armFailure' && value === null), false);\n"
    )
    _run_node(script)
