from textwrap import dedent

from .test_web_controller_input import WEB_APP, _function_bundle, _run_node


def _required_bundle(source: str, roots: tuple[str, ...], excluded: set[str]) -> str:
    bundle = _function_bundle(source, roots, excluded=excluded)
    assert bundle is not None, f"Required app.js function missing: {', '.join(roots)}"
    return bundle


def _lifecycle_prefix() -> str:
    return dedent(
        """
        const assert = require('node:assert/strict');
        const XR_STOP_BUTTON_INDEX = 4;
        const XR_GRIPPER_TOGGLE_BUTTON_INDEX = 5;
        let clock = 0;
        global.performance = {now: () => clock};
        global.document = {hidden: false};
        const events = [];
        const stopCalls = [];
        const inputCalls = [];
        const drawCalls = [];
        const updateCalls = [];
        const session = {
          visibilityState: 'visible',
          scheduled: 0,
          ended: 0,
          requestAnimationFrame(callback) { this.scheduled += 1; this.callback = callback; },
          end() { this.ended += 1; return Promise.resolve(); },
        };
        const pose = {
          transform: {matrix: new Float32Array(16)},
          views: [{projectionMatrix: new Float32Array(16), transform: {inverse: {matrix: new Float32Array(16)}}}],
        };
        function controller(overrides = {}) {
          return {
            position: [0, 0, 0], orientation: [0, 0, 0, 1], tracked: false,
            grip: false, trigger: 0, thumbstick: [0, 0], buttons: Array(7).fill(false),
            ...overrides,
          };
        }
        function makeController() { return controller(); }
        let nextControllers = {left: controller(), right: controller()};
        let viewerPose = pose;
        const app = {
          status: {armed: false, control_owner: null},
          armRequestPending: false,
          controllerClient: false,
          gripperDirections: {left:'close', right:'close'},
          gripperToggleWasDown: {left:false, right:false},
          inputPackets: 0,
          lastInputSentAt: 0,
          inputSocket: null,
          ws: {},
          controllers: {left: controller(), right: controller()},
          xr: {
            active: true, session, controllersReady: false, safetyTripped: false,
            gripActivationReady: false, stopButtonWasDown: false,
            ending: false, viewerTracked: false,
          },
        };
        const dom = {xrStatus: {textContent: ''}};
        function failSafeStop(reason, options = {}) {
          stopCalls.push({reason, options});
          app.xr.safetyTripped = true;
        }
        function sendInputFrame(now, force = false) {
          events.push('input');
          inputCalls.push({now, force, controllers: app.controllers});
          app.lastInputSentAt = now;
          app.inputPackets += 1;
        }
        function readControllers() { return nextControllers; }
        function drawXrScene(renderPose) {
          drawCalls.push(renderPose);
        }
        function updateImmersiveUi() { updateCalls.push('immersive'); }
        function updateInputUi() { updateCalls.push('input'); }
        function updateArmUi() { updateCalls.push('arm'); }
        function manualStop() { events.push('manual-stop'); }
        let armCalls = 0;
        function requestArm() {
          assert.equal(app.lastInputSentAt, clock, 'Grip activation must see fresh input first');
          events.push('arm');
          armCalls += 1;
        }
        function frame() {
          return {
            session,
            getViewerPose() { return viewerPose; },
          };
        }
        function resetSignals() {
          events.length = 0;
          stopCalls.length = 0;
          inputCalls.length = 0;
          drawCalls.length = 0;
          updateCalls.length = 0;
          armCalls = 0;
          session.scheduled = 0;
          session.ended = 0;
        }
        """
    )


def test_xr_focus_visibility_and_stale_session_safety():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_bundle(
        source,
        (
            "xrHasInputFocus",
            "hasRobotControlIntent",
            "pauseXrInput",
            "handleXrVisibilityChange",
        ),
        excluded={"makeController", "failSafeStop", "sendInputFrame", "updateAllUi"},
    )
    script = (
        _lifecycle_prefix()
        + bundle
        + dedent(
            """
            function sendInputFrame(now) {
              events.push('input');
              inputCalls.push({now, controllers: app.controllers});
            }
            function updateAllUi() { updateCalls.push('all'); }

            document.hidden = true;
            session.visibilityState = 'visible';
            assert.equal(xrHasInputFocus(), true, 'visible XR must ignore document.hidden');
            session.visibilityState = 'visible-blurred';
            assert.equal(xrHasInputFocus(), false, 'visible-blurred XR does not own input focus');
            session.visibilityState = 'hidden';
            assert.equal(xrHasInputFocus(), false, 'hidden XR does not own input focus');

            session.visibilityState = 'visible';
            const staleSession = {visibilityState: 'hidden'};
            app.controllers = {left: controller({tracked: true}), right: controller({tracked: true})};
            app.xr.controllersReady = true;
            app.xr.gripActivationReady = true;
            resetSignals();
            handleXrVisibilityChange(staleSession);
            assert.equal(stopCalls.length, 0, 'stale visibility events must be ignored');
            assert.equal(inputCalls.length, 0, 'stale visibility events must not send input');
            assert.equal(app.xr.controllersReady, true);
            assert.equal(app.xr.gripActivationReady, true);

            app.status = {armed: false, control_owner: null};
            app.armRequestPending = false;
            app.controllerClient = true;
            app.xr.safetyTripped = false;
            app.xr.controllersReady = true;
            app.xr.gripActivationReady = true;
            app.controllers = {left: controller({tracked: true}), right: controller({tracked: true})};
            session.visibilityState = 'hidden';
            resetSignals();
            handleXrVisibilityChange(session);
            assert.equal(stopCalls.length, 0, 'idle visibility loss must not latch safety');
            assert.equal(app.xr.safetyTripped, false);
            assert.equal(app.xr.controllersReady, false);
            assert.equal(app.xr.gripActivationReady, false);
            assert.equal(app.controllers.left.tracked, false);
            assert.equal(app.controllers.right.tracked, false);
            assert.equal(inputCalls.length, 1, 'visibility loss still reports zeroed input');
            assert.equal(updateCalls.length, 1);

            for (const intent of [
              {armRequestPending: true, status: {armed: false, control_owner: null}, controllerClient: false},
              {armRequestPending: false, status: {armed: false, control_owner: 'self'}, controllerClient: false},
              {armRequestPending: false, status: {armed: true, control_owner: null}, controllerClient: true},
            ]) {
              app.armRequestPending = intent.armRequestPending;
              app.status = intent.status;
              app.controllerClient = intent.controllerClient;
              app.xr.safetyTripped = false;
              app.xr.controllersReady = true;
              app.xr.gripActivationReady = true;
              session.visibilityState = 'visible-blurred';
              resetSignals();
              handleXrVisibilityChange(session);
              assert.equal(stopCalls.length, 1, 'each active control intent must fail safe');
              assert.equal(stopCalls[0].reason, 'VR 会话失去输入焦点');
              assert.deepEqual(stopCalls[0].options, {endSession: false});
              assert.equal(app.xr.safetyTripped, true);
              assert.equal(app.xr.controllersReady, false);
              assert.equal(inputCalls.length, 1);
            }

            app.armRequestPending = false;
            app.status = {armed: false, control_owner: null};
            app.controllerClient = false;
            app.xr.safetyTripped = false;
            session.visibilityState = 'hidden';
            resetSignals();
            handleXrVisibilityChange(session);
            assert.equal(stopCalls.length, 0, 'idle must remain non-safety after another focus loss');
            assert.equal(app.xr.safetyTripped, false);
            """
        )
    )
    _run_node(script)


def test_hand_or_gaze_sources_do_not_overwrite_touch_controllers():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_bundle(source, ("readControllers",), {"makeController", "readController"})
    _run_node(
        dedent("""
        const assert = require('node:assert/strict');
        function makeController() { return {tracked: false}; }
        function readController(source) { return {tracked: source.tracked, id: source.id}; }
        const touch = {id: 'touch', handedness: 'left', gamepad: {},
                       tracked: true, targetRayMode: 'tracked-pointer'};
        const hand = {id: 'hand', handedness: 'left', hand: {}, gamepad: {}, tracked: true};
        const gaze = {id: 'gaze', handedness: 'left', targetRayMode: 'gaze', gamepad: {}, tracked: true};
        const partial = {id: 'partial', handedness: 'left', gamepad: {}, tracked: false};
    """)
        + bundle
        + dedent("""
        for (const inputSources of [[touch, hand, gaze, partial], [partial, hand, gaze, touch]]) {
          const result = readControllers({}, {}, {inputSources});
          assert.deepEqual(result.left, {tracked: true, id: 'touch'});
          assert.equal(result.right.tracked, false);
        }
    """)
    )


def test_idle_video_pause_can_recover_but_active_video_loss_still_stops():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_bundle(
        source,
        ("checkVideoHealth", "cameraFramesFresh", "handleVideoPeerFailure"),
        {"updateCameraVisual", "failSafeStop", "stopVideoStats", "updateVideoPeerUi", "logEvent", "retryVideoPeer"},
    )
    _run_node(
        dedent("""
        const assert = require('node:assert/strict');
        global.performance = {now: () => 10000};
        const CAMERA_NAMES = ['front', 'left_wrist', 'right_wrist'];
        const VIDEO_STALE_MS = 2500;
        const cameras = Object.fromEntries(CAMERA_NAMES.map(name =>
          [name, {hasFrame: true, lastGoodAt: 100}]));
        const app = {authenticated:true, status:{armed:false, control_owner:null}, armRequestPending:false,
          controllerClient:true, inputPackets:300, xr:{active:true,safetyTripped:false},
          video:{state:'connected',generation:1}};
        const dom = {cameraHealth:{}};
        const stops = [];
        let retries = 0;
        function updateCameraVisual() {}
        function stopVideoStats() {}
        function updateVideoPeerUi() {}
        function logEvent() {}
        function retryVideoPeer() { retries++; }
        function failSafeStop(reason, options) {
          stops.push({reason, options}); app.xr.safetyTripped = true;
        }
    """)
        + bundle
        + dedent("""
        checkVideoHealth();
        assert.equal(cameraFramesFresh(),false);
        assert.equal(stops.length,0,'idle telemetry is not control ownership');
        assert.equal(app.xr.safetyTripped,false);
        handleVideoPeerFailure(new Error('sleep interrupted video'),null,1);
        assert.equal(stops.length,0,'idle WebRTC interruption must also remain recoverable');
        assert.equal(app.video.state,'error');
        assert.equal(retries,1,'viewing should reconnect after headset sleep');
        app.video.state = 'connected';
        for (const camera of Object.values(cameras)) camera.lastGoodAt = 9999;
        checkVideoHealth();
        assert.equal(cameraFramesFresh(),true);
        assert.equal(app.xr.safetyTripped,false);
        cameras.front.lastGoodAt = 100;
        for (const intent of [
          {pending:true,armed:false,owner:null},
          {pending:false,armed:false,owner:'self'},
          {pending:false,armed:true,owner:null},
        ]) {
          app.armRequestPending = intent.pending;
          app.status = {armed:intent.armed,control_owner:intent.owner};
          app.xr.safetyTripped = false;
          stops.length = 0;
          checkVideoHealth();
          assert.equal(stops.length,1);
          assert.equal(stops[0].reason,'视频帧过期');
          assert.equal(app.xr.safetyTripped,true);
          cameras.front.lastGoodAt = 9999;
          checkVideoHealth();
          assert.equal(app.xr.safetyTripped,true,'fresh video does not auto-rearm');
          cameras.front.lastGoodAt = 100;
          app.xr.safetyTripped = false;
          app.video.state = 'connected';
          stops.length = 0;
          handleVideoPeerFailure(new Error('connection lost'),null,1);
          assert.equal(stops.length,1,'active WebRTC interruption must stop immediately');
          assert.equal(stops[0].reason,'WebRTC 视频中断');
        }
    """)
    )


def test_on_xr_frame_keeps_startup_telemetry_renders_and_orders_input_before_arm():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_bundle(
        source,
        (
            "onXrFrame",
            "xrHasInputFocus",
            "hasRobotControlIntent",
            "pauseXrInput",
            "updateQuestButtonActions",
        ),
        excluded={
            "makeController",
            "failSafeStop",
            "sendInputFrame",
            "readControllers",
            "drawXrScene",
            "updateImmersiveUi",
            "updateInputUi",
            "updateArmUi",
            "manualStop",
            "requestArm",
        },
    )
    script = (
        _lifecycle_prefix()
        + bundle
        + dedent(
            """
            document.hidden = true;
            session.visibilityState = 'visible';
            viewerPose = pose;
            nextControllers = {
              left: controller({tracked: false}),
              right: controller({tracked: true}),
            };
            app.xr.controllersReady = false;
            app.xr.safetyTripped = false;
            app.status = {armed: false, control_owner: null};
            app.armRequestPending = false;
            app.controllerClient = false;
            resetSignals();
            clock = 100;
            onXrFrame(0, frame());
            assert.equal(session.scheduled, 1);
            assert.equal(inputCalls.length, 1, 'startup tracking loss must still send telemetry');
            assert.equal(inputCalls[0].controllers.left.tracked, false);
            assert.equal(inputCalls[0].controllers.right.tracked, true);
            assert.equal(drawCalls.length, 1, 'a viewer pose must still render the XR scene');
            assert.equal(stopCalls.length, 0);
            assert.equal(session.ended, 0);
            assert.equal(app.xr.controllersReady, false);
            assert.equal(app.xr.safetyTripped, false);
            assert.equal(armCalls, 0, 'unready startup input must not activate Grip');

            nextControllers = {
              left: controller({tracked: true, grip:true}),
              right: controller({tracked: true}),
            };
            app.xr.controllersReady = true;
            app.xr.safetyTripped = false;
            app.xr.gripActivationReady = true;
            resetSignals();
            clock = 2000;
            onXrFrame(0, frame());
            assert.deepEqual(events, ['input', 'arm'], 'fresh input must precede Grip activation');
            assert.equal(inputCalls[0].now, 2000);
            assert.equal(inputCalls[0].force, true, 'Grip edge bypasses normal input throttling');
            assert.equal(drawCalls.length, 1);
            assert.equal(armCalls, 1);
            """
        )
    )
    _run_node(script)


def test_on_xr_frame_stale_frame_and_control_tracking_loss_stop_without_auto_rearm():
    source = WEB_APP.read_text(encoding="utf-8")
    bundle = _required_bundle(
        source,
        (
            "onXrFrame",
            "xrHasInputFocus",
            "hasRobotControlIntent",
            "pauseXrInput",
        ),
        excluded={
            "makeController",
            "failSafeStop",
            "sendInputFrame",
            "readControllers",
            "drawXrScene",
            "updateImmersiveUi",
            "updateInputUi",
            "updateArmUi",
            "updateQuestButtonActions",
        },
    )
    script = (
        _lifecycle_prefix()
        + bundle
        + dedent(
            """
            function updateQuestButtonActions() { events.push('buttons'); }
            document.hidden = false;
            session.visibilityState = 'visible';
            viewerPose = pose;
            nextControllers = {
              left: controller({tracked: true}),
              right: controller({tracked: true}),
            };
            const staleFrame = {session: {visibilityState: 'visible'}, getViewerPose() { return pose; }};
            resetSignals();
            onXrFrame(0, staleFrame);
            assert.equal(session.scheduled, 0, 'stale frames must not schedule another frame');
            assert.equal(inputCalls.length, 0);
            assert.equal(drawCalls.length, 0);
            assert.equal(stopCalls.length, 0);

            nextControllers = {
              left: controller({tracked: false}),
              right: controller({tracked: true}),
            };
            app.xr.controllersReady = true;
            app.xr.safetyTripped = false;
            app.status = {armed: true, control_owner: 'self'};
            app.armRequestPending = false;
            app.controllerClient = true;
            app.inputPackets = 1;
            resetSignals();
            clock = 3000;
            onXrFrame(0, frame());
            assert.equal(stopCalls.length, 1, 'control tracking loss must stop active control');
            assert.equal(stopCalls[0].reason, '控制器追踪丢失');
            assert.deepEqual(stopCalls[0].options, {endSession: false});
            assert.equal(app.xr.safetyTripped, true);
            assert.equal(session.ended, 0, 'tracking loss must keep the XR session for diagnostics');
            assert.equal(inputCalls.length, 1, 'tracking loss may still report telemetry');
            assert.equal(events.includes('buttons'), false, 'unready tracking must not process buttons');

            nextControllers = {
              left: controller({tracked: true, buttons: Object.assign(Array(7).fill(false), {4: true})}),
              right: controller({tracked: true, buttons: Object.assign(Array(7).fill(false), {4: true})}),
            };
            resetSignals();
            clock = 4000;
            onXrFrame(0, frame());
            assert.equal(app.xr.safetyTripped, true, 'recovered tracking must retain the safety latch');
            assert.equal(events.includes('buttons'), false, 'safety recovery must never auto-arm');
            assert.equal(stopCalls.length, 0);
            assert.equal(inputCalls.length, 1);
            """
        )
    )
    _run_node(script)
