import shutil
import subprocess
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).parents[2] / "integrations" / "xlerobot_owner" / "src" / "embodirun_xlerobot_owner" / "web"


def asset(name: str) -> str:
    return (WEB_DIR / name).read_text(encoding="utf-8")


def test_drive_page_has_explicit_dual_leader_controls_without_run_text_entry():
    html = asset("drive.html")
    source = asset("drive.js")

    assert 'id="leader-start"' in html
    assert 'id="leader-stop"' in html
    assert "不再要求输入 RUN" in html
    assert 'send({ type: "leader_start" })' in source
    assert 'send({ type: "leader_stop" })' in source
    assert "prompt(" not in source


def test_video_transport_and_xr_layout_assets():
    # Static source checks only; this is not proof of browser/WebRTC media playback.
    html = asset("index.html")
    source = asset("app.js")
    style = asset("style.css")

    assert html.count("<video ") == 3
    for name in ("front", "left_wrist", "right_wrist"):
        assert f'id="camera-{name}"' in html
    assert html.count("autoplay muted playsinline") == 3
    assert "<img" not in html
    assert "WebRTC coded video" in html
    assert "return [app.mainCamera, others[0], others[1]];" in source
    assert 'new RTCPeerConnection({ iceServers: [], bundlePolicy: "max-bundle" });' in source
    assert 'peer.addTransceiver("video", { direction: "recvonly" })' in source
    assert "await waitForIceGatheringComplete(peer);" in source
    assert 'requestJson("/api/webrtc/offer"' in source
    assert 'body: JSON.stringify({ sdp: localDescription.sdp, type: "offer" })' in source
    assert "event.transceiver" in source
    assert "new MediaStream([event.track])" in source
    assert "const stream = event.streams" not in source
    assert "camera.video.srcObject = stream" in source
    assert "requestVideoFrameCallback" in source
    assert "gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, camera.video);" in source
    assert "/api/cameras/" not in source
    assert "XR_MENU_BUTTON_INDEX" not in source
    assert "const XR_STOP_BUTTON_INDEX = 4;" in source
    assert "const XR_GRIPPER_TOGGLE_BUTTON_INDEX = 5;" in source
    assert "XR_ARM_HOLD_MS" not in source
    assert "grid-template-columns: repeat(2, minmax(0, 1fr));" in style
    assert "aspect-ratio: 4 / 3;" in style
    assert "{ x: 0, y: 0.62, z: -2.9, width: 1.8, height: 1.35 }" in source
    assert "{ x: -0.51, y: -0.48, z: -2.9, width: 0.95, height: 0.7125 }" in source
    assert "{ x: 0.51, y: -0.48, z: -2.9, width: 0.95, height: 0.7125 }" in source
    assert "const headRelativeModel = multiplyMatrices(new Float32Array(pose.transform.matrix), model);" in source
    assert "const mvp = multiplyMatrices(viewProjection, headRelativeModel);" in source
    assert source.count("gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);") == 1


def test_xr_initial_tracking_waits_before_failsafe_stop():
    source = asset("app.js")
    # Runtime loss/recovery is exercised by test_web_xr_lifecycle.py. Keep
    # this asset check limited to the separation of XR focus and robot stop.
    assert "controllersReady: false" in source
    assert 'session.addEventListener("visibilitychange"' in source
    assert "if (hasRobotControlIntent() && !app.xr.safetyTripped)" in source
    assert "failSafeStop(reason, { endSession: false });" in source
    assert "if (pose) drawXrScene(pose, app.xr.layer);" in source


def test_auth_accepts_pairing_code_copy_without_changing_session_payload():
    html = asset("index.html")
    source = asset("app.js")

    assert "Token 或 6 位配对码" in html
    assert "输入长 Token 或 6 位配对码" in html
    assert "JSON.stringify({ token: value })" in source
    assert 'id="tokenInput"' in html
    assert "maxlength=" not in html.split('id="tokenInput"', 1)[1].split(">", 1)[0]


def test_recording_outcome_messages_describe_saved_record_and_task_result():
    source = asset("app.js")

    assert "JSON.stringify({ success })" in source
    assert "记录已保存；任务结果：成功。" in source
    assert "记录已保存；任务结果：失败。" in source
    assert "记录已保存；任务结果：未知。" in source


def test_robot_fault_status_does_not_become_auth_failure():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for browser request helper test")
    source = asset("app.js")
    helper = source.split("  async function requestJson(", 1)[1].split("  function readFragmentToken()", 1)[0]
    script = (
        """
    const assert = require('node:assert/strict');
    class ApiError extends Error {
      constructor(message, status, payload) { super(message); this.status = status; }
    }
    """
        + "async function requestJson("
        + helper
        + """
    (async () => {
      let responseStatus = 200;
      global.fetch = async () => ({
        ok: responseStatus === 200, status: responseStatus,
        text: async () => JSON.stringify({connected: false, error: 'AGX restarting'})
      });
      assert.equal((await requestJson('/api/status')).error, 'AGX restarting');
      await assert.rejects(requestJson('/api/session'), /AGX restarting/);
      responseStatus = 401;
      await assert.rejects(requestJson('/api/status'), error => error.status === 401);
    })().catch(error => { console.error(error); process.exitCode = 1; });
    """
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_shared_signalled_stream_is_split_into_one_track_per_camera():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for browser track binding test")
    helper = (
        asset("app.js").split("  function handleVideoTrack(", 1)[1].split("  function handleVideoPeerFailure(", 1)[0]
    )
    script = (
        """
    const assert = require('node:assert/strict');
    const CAMERA_NAMES = ['front', 'left_wrist', 'right_wrist'];
    const peer = {};
    const app = { video: { peer } };
    const cameras = Object.fromEntries(CAMERA_NAMES.map(name => [name, {
      video: { play: () => Promise.resolve() }
    }]));
    class MediaStream { constructor(tracks) { this.tracks = tracks; } }
    function scheduleVideoFrameCallback() {}
    function updateCameraVisual() {}
    function handleVideoPeerFailure(error) { throw error; }
    """
        + "function handleVideoTrack("
        + helper
        + """
    const transceivers = [{}, {}, {}];
    const tracks = CAMERA_NAMES.map(id => ({id, kind: 'video', addEventListener() {}}));
    const shared = new MediaStream(tracks);
    // Different arrival order, with the same three-track MSID on every event.
    for (const index of [2, 0, 1]) {
      handleVideoTrack({track: tracks[index], transceiver: transceivers[index],
        streams: [shared]}, peer, transceivers);
    }
    for (const [index, name] of CAMERA_NAMES.entries()) {
      assert.deepEqual(cameras[name].video.srcObject.tracks, [tracks[index]]);
      assert.notEqual(cameras[name].video.srcObject, shared);
    }
    """
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


def test_xr_draw_binds_each_camera_for_both_eyes_and_cached_frames():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required for XR draw-state test")
    functions = (
        asset("app.js").split("  function identityMatrix()", 1)[1].split("  async function checkXrSupport()", 1)[0]
    )
    script = (
        """
    const assert = require('node:assert/strict');
    const CAMERA_NAMES = ['front', 'left_wrist', 'right_wrist'];
    const textures = new Map();
    let bound = null, eye = null, center = null, uploads = 0;
    const draws = [];
    const gl = {
      TEXTURE0: 0, TEXTURE_2D: 3553,
      activeTexture(unit) { assert.equal(unit, this.TEXTURE0); },
      bindTexture(target, texture) { bound = texture; },
      texImage2D(...args) { textures.set(bound, args.at(-1).id); uploads++; },
      bindFramebuffer() {}, clearColor() {}, clear() {}, useProgram() {}, bindBuffer() {},
      viewport(x) { eye = x; },
      uniformMatrix4fv(location, transpose, matrix) { center = [matrix[12], matrix[13]]; },
      uniform1f() {},
      drawArrays() { draws.push({ eye, camera: textures.get(bound), center }); }
    };
    const app = { mainCamera: 'front', xr: { gl, program: {} } };
    const cameras = Object.fromEntries(CAMERA_NAMES.map(id => [id, {
      texture: {id}, video: {id, readyState: 2, videoWidth: 640, videoHeight: 480},
      frameVersion: 1, uploadedVersion: -1, hasFrame: true
    }]));
    function logEvent(message) { throw new Error(message); }
    """
        + "function identityMatrix()"
        + functions
        + """
    const matrix = identityMatrix();
    const pose = {transform: {matrix}, views: [0, 1].map(eye => ({
      eye, projectionMatrix: matrix, transform: {inverse: {matrix}}
    }))};
    const layer = {framebuffer: {}, getViewport(view) {
      return {x: view.eye, y: 0, width: 640, height: 480};
    }};
    for (const main of CAMERA_NAMES) {
      app.mainCamera = main;
      for (const cachedFrame of [false, true]) {
        if (!cachedFrame) CAMERA_NAMES.forEach(name => cameras[name].frameVersion++);
        const beforeUploads = uploads;
        draws.length = 0;
        drawXrScene(pose, layer);
        const names = [main, ...CAMERA_NAMES.filter(name => name !== main)];
        assert.deepEqual(draws.map(draw => draw.camera), [...names, ...names],
          `wrong XR camera binding: main=${main}, cached=${cachedFrame}`);
        assert.deepEqual(draws.map(draw => draw.eye), [0, 0, 0, 1, 1, 1]);
        assert.equal(uploads - beforeUploads, cachedFrame ? 0 : 3,
          'upload only new frames once, not once per eye');
        assert.ok(draws[0].center[1] > draws[1].center[1]);
        assert.ok(draws[1].center[0] < draws[2].center[0]);
      }
    }
    """
    )
    result = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
