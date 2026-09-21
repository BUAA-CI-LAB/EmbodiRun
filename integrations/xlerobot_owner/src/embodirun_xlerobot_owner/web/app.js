(() => {
  "use strict";

  const CAMERA_NAMES = ["front", "left_wrist", "right_wrist"];
  const CAMERA_LABELS = {
    front: "前视",
    left_wrist: "左腕",
    right_wrist: "右腕",
  };
  const VIDEO_STALE_MS = 1000;
  const VIDEO_HEALTH_INTERVAL_MS = 200;
  const ICE_GATHERING_TIMEOUT_MS = 5000;
  const VIDEO_RETRY_BASE_MS = 500;
  const VIDEO_RETRY_MAX_MS = 8000;
  const VIDEO_STATS_INTERVAL_MS = 1000;
  const STATUS_INTERVAL_MS = 1000;
  const INPUT_INTERVAL_MS = 1000 / 30;
  const INPUT_FRESH_MS = 350;
  const MAX_CONTROL_BUFFER_BYTES = 64 * 1024;
  // xr-standard: Grip=1, X/A=4 (stop), Y/B=5 (local claw direction).
  const XR_STOP_BUTTON_INDEX = 4;
  const XR_GRIPPER_TOGGLE_BUTTON_INDEX = 5;

  const app = {
    authenticated: false,
    authInFlight: false,
    status: {
      mode: null,
      armed: false,
      connected: false,
      control_owner: null,
      recording: false,
      cameras: [],
      state: {},
      error: null,
      stats: {},
    },
    statusSource: "none",
    statusUpdatedAt: 0,
    ws: null,
    wsState: "closed",
    wsReconnectTimer: null,
    shouldReconnect: true,
    mainCamera: "front",
    inputSeq: 0,
    lastInputSentAt: 0,
    inputPackets: 0,
    inputSocket: null,
    armRequestPending: false,
    armFailure: null,
    modeRequestPending: null,
    controllerClient: false,
    gripperDirections: {left: "close", right: "close"},
    gripperToggleWasDown: {left: false, right: false},
    videoHealthTimer: null,
    video: {
      peer: null,
      transceivers: [],
      generation: 0,
      connecting: false,
      state: "idle",
      error: "",
      retryTimer: null,
      retryAttempt: 0,
      retryDelayMs: 0,
      failurePeer: null,
      wasConnected: false,
      statsTimer: null,
      statsInFlight: false,
      statsPrevious: {},
      stats: {
        fpsMin: null,
        fpsMax: null,
        fpsAverage: null,
        streamCount: 0,
        bitrate: null,
        codec: "-",
      },
    },
    controllers: {
      left: makeController(),
      right: makeController(),
    },
    xr: {
      supported: false,
      supportChecked: false,
      gl: null,
      program: null,
      layer: null,
      session: null,
      referenceSpace: null,
      active: false,
      startedAt: 0,
      controllersReady: false,
      viewerTracked: false,
      gripActivationReady: false,
      stopButtonWasDown: false,
      safetyTripped: false,
      stopSent: false,
      ending: false,
    },
  };

  const dom = {};
  const cameras = {};

  class ApiError extends Error {
    constructor(message, status = 0, payload = null) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.payload = payload;
    }
  }

  function makeController() {
    return {
      position: [0, 0, 0],
      orientation: [0, 0, 0, 1],
      tracked: false,
      grip: false,
      trigger: 0,
      thumbstick: [0, 0],
      buttons: [false, false, false, false, false, false, false],
    };
  }

  function finite(value, fallback = 0) {
    return Number.isFinite(value) ? Number(value) : fallback;
  }

  function clamp(value, low, high) {
    return Math.max(low, Math.min(high, finite(value, low)));
  }

  function finiteVector(value, length, fallback) {
    if (!value || typeof value.length !== "number" || value.length < length) {
      return fallback.slice();
    }
    return Array.from({ length }, (_, index) => finite(value[index], fallback[index]));
  }

  function normalizedQuaternion(value) {
    const q = finiteVector(value, 4, [0, 0, 0, 1]);
    const norm = Math.hypot(q[0], q[1], q[2], q[3]);
    if (!Number.isFinite(norm) || norm < 0.0001) {
      return [0, 0, 0, 1];
    }
    return q.map((component) => component / norm);
  }

  function getElement(id) {
    return document.getElementById(id);
  }

  function initDom() {
    [
      "modeBadge",
      "connectionBadge",
      "connectionText",
      "themeToggle",
      "cameraHealth",
      "videoPeerStatus",
      "videoPeerError",
      "videoPeerStats",
      "videoPeerRetry",
      "mainCameraSelect",
      "authPanel",
      "authForm",
      "authState",
      "tokenInput",
      "authButton",
      "authHint",
      "controlHint",
      "controlModeSelect",
      "controlModeHint",
      "armBadge",
      "leaseValue",
      "armButton",
      "stopButton",
      "immersiveButton",
      "xrStatus",
      "statusMode",
      "statusConnected",
      "statusRecording",
      "statusInput",
      "statusAge",
      "statusError",
      "statFps",
      "statLatency",
      "statDropped",
      "jointState",
      "recordForm",
      "taskInput",
      "recordStart",
      "recordStop",
      "recordOutcome",
      "recordingBadge",
      "recordingMessage",
      "recordingPath",
      "eventLog",
      "toastRegion",
      "xrCanvas",
    ].forEach((id) => {
      dom[id] = getElement(id);
    });

    CAMERA_NAMES.forEach((name) => {
      cameras[name] = {
        name,
        video: getElement(`camera-${name}`),
        frame: getElement(`camera-${name}`)?.closest(".camera-frame"),
        state: getElement(`camera-state-${name}`),
        placeholder: getElement(`placeholder-${name}`),
        tile: document.querySelector(`[data-camera="${name}"]`),
        frameVersion: 0,
        uploadedVersion: -1,
        lastGoodAt: 0,
        lastFrameTimestamp: null,
        lastPresentedFrames: null,
        frameCallbackId: null,
        hasFrame: false,
        stale: false,
        track: null,
        error: null,
      };
    });
  }

  async function requestJson(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (options.body && !headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }

    let response;
    try {
      response = await fetch(path, {
        ...options,
        headers,
        credentials: "include",
        cache: "no-store",
      });
    } catch (error) {
      throw new ApiError(`网络请求失败: ${error.message || "无法连接"}`);
    }

    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch (_error) {
      payload = { error: text || `HTTP ${response.status}` };
    }

    // A successful status response can carry a robot fault. Keep that as
    // telemetry so the authenticated video/Stop UI remains available.
    const commandError = path !== "/api/status" && payload && typeof payload.error === "string";
    if (!response.ok || commandError) {
      throw new ApiError(
        payload && typeof payload.error === "string" ? payload.error : `HTTP ${response.status}`,
        response.status,
        payload,
      );
    }
    return payload;
  }

  function readFragmentToken() {
    const rawHash = window.location.hash.replace(/^#/, "");
    const params = new URLSearchParams(rawHash);
    const token = params.get("token");
    if (token) {
      const cleanUrl = `${window.location.pathname}${window.location.search}`;
      try {
        window.history.replaceState(window.history.state, document.title, cleanUrl);
      } catch (_error) {
        window.location.hash = "";
      }
    }
    return token || "";
  }

  function setAuthUi(message, state = "muted") {
    const trustedLan = app.status.browser_access === "trusted_lan";
    dom.authPanel.classList.toggle("is-authenticated", app.authenticated);
    dom.authState.textContent = app.authenticated ? (trustedLan ? "局域网直连" : "已认证") : message;
    setChip(dom.authState, app.authenticated ? "safe" : state);
    dom.authButton.disabled = app.authInFlight;
    dom.authHint.textContent = app.authInFlight
      ? "正在验证 Token 或 6 位配对码，等待服务端确认。"
      : app.authenticated
        ? trustedLan ? "当前局域网无需 Token 或配对码；AGX 控制端仍保留独立认证。"
          : "当前浏览器已获得会话 Cookie，后续请求不再提交令牌。"
        : "可输入长 Token 或服务器控制台显示的 6 位一次性配对码。地址栏的 #token=... 会自动取用并移除。";
  }

  async function authenticate(token) {
    const value = String(token || "").trim();
    if (!value) {
      setAuthUi("未认证", "muted");
      dom.tokenInput.focus();
      return false;
    }

    app.authInFlight = true;
    setAuthUi("认证中", "warning");
    try {
      const result = await requestJson("/api/session", {
        method: "POST",
        body: JSON.stringify({ token: value }),
      });
      if (!result || result.authenticated !== true) {
        throw new ApiError("服务端未确认会话认证");
      }
      app.authenticated = true;
      logEvent("会话认证成功，开始读取服务状态。", "safe");
      setAuthUi("已认证", "safe");
      startAuthenticatedPipelines();
      await refreshStatus();
      return true;
    } catch (error) {
      app.authenticated = false;
      setAuthUi(error.message || "认证失败", "danger");
      showToast(error.message || "认证失败", "error");
      logEvent(`会话认证失败: ${error.message || "未知错误"}`, "error");
      return false;
    } finally {
      app.authInFlight = false;
      setAuthUi(app.authenticated ? "已认证" : "未认证", app.authenticated ? "safe" : "muted");
    }
  }

  function setUnauthenticated(message = "需要重新认证") {
    if (isActualControllerClient()) {
      failSafeStop("认证失效");
    }
    stopVideoPeer();
    if (app.videoHealthTimer) {
      window.clearInterval(app.videoHealthTimer);
      app.videoHealthTimer = null;
    }
    app.authenticated = false;
    app.shouldReconnect = false;
    app.armRequestPending = false;
    app.modeRequestPending = null;
    if (app.ws) {
      try {
        app.ws.close(1000, "authentication required");
      } catch (_error) {
        // The browser may already have closed the socket.
      }
      app.ws = null;
    }
    app.wsState = "closed";
    setAuthUi(message, "danger");
    updateAllUi();
  }

  async function refreshStatus() {
    try {
      const status = await requestJson("/api/status");
      app.authenticated = true;
      applyStatus(status, "rest");
      setAuthUi("已认证", "safe");
      startAuthenticatedPipelines();
      return true;
    } catch (error) {
      if (error.status === 401) {
        setUnauthenticated("需要认证");
      } else {
        setStatusError(`状态读取失败: ${error.message || "未知错误"}`);
        logEvent(`状态读取失败: ${error.message || "未知错误"}`, "error");
      }
      return false;
    }
  }

  function startAuthenticatedPipelines() {
    app.shouldReconnect = true;
    startVideoPeer();
    startVideoHealth();
    openWebSocket();
  }

  function scheduleStatusPolling() {
    window.setInterval(() => {
      if (app.authenticated) {
        void refreshStatus();
      } else if (!app.authInFlight) {
        void refreshStatus();
      }
    }, STATUS_INTERVAL_MS);
  }

  function normalizeStatus(payload) {
    if (payload && payload.status && typeof payload.status === "object") {
      return payload.status;
    }
    return payload && typeof payload === "object" ? payload : {};
  }

  function applyStatus(payload, source) {
    const incoming = normalizeStatus(payload);
    const previousOwner = app.status.control_owner;
    const shouldTakeOwner = source === "ws" || !isWebSocketOpen();
    app.status = {
      ...app.status,
      ...incoming,
      stats: {
        ...(app.status.stats || {}),
        ...(incoming.stats && typeof incoming.stats === "object" ? incoming.stats : {}),
      },
    };
    if (!shouldTakeOwner && previousOwner !== null && previousOwner !== undefined) {
      app.status.control_owner = previousOwner;
    }
    if (source === "ws") {
      app.statusSource = "ws";
      app.statusUpdatedAt = performance.now();
    } else if (!app.statusUpdatedAt) {
      app.statusSource = "rest";
      app.statusUpdatedAt = performance.now();
    }
    if (app.status.armed === true) {
      app.armRequestPending = false;
      app.armFailure = null;
    }
    if (incoming.control_mode === app.modeRequestPending) app.modeRequestPending = null;
    updateServerIssues(source);
    updateAllUi();
  }

  function updateServerIssues(source) {
    const issues = [];
    if (app.status.error) issues.push(String(app.status.error));
    if (Array.isArray(app.status.observation_errors) && app.status.observation_errors.length) {
      issues.push(`观测问题: ${app.status.observation_errors.map((item) => String(item)).join("；")}`);
    }
    if (app.status.stop_unconfirmed === true) {
      issues.push("Stop 未获得服务端确认，请先验证停止反馈。");
    }
    const metadata = app.status.metadata && typeof app.status.metadata === "object"
      ? app.status.metadata
      : {};
    if (metadata.allow_motion === false) {
      issues.push("动作权限已关闭: allow_motion=false");
    }
    if (issues.length) {
      setStatusError(issues.join(" | "));
    } else if (source === "ws") {
      setStatusError("");
    }
  }

  function setChip(element, state) {
    if (!element) return;
    element.classList.remove(
      "status-chip-muted",
      "status-chip-safe",
      "status-chip-warning",
      "status-chip-danger",
    );
    element.classList.add(`status-chip-${state || "muted"}`);
  }

  function setStatusError(message) {
    const value = String(message || "");
    dom.statusError.textContent = value;
    dom.statusError.hidden = !value;
  }

  function modeLabel(mode) {
    if (mode === "demo") return "演示模式 / 合成数据";
    if (mode === "remote") return "远程模式";
    if (mode === "hardware") return "硬件模式";
    return "等待状态";
  }

  function formatValue(value) {
    if (typeof value === "boolean") return value ? "true" : "false";
    if (typeof value === "number" && Number.isFinite(value)) {
      return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
    }
    if (typeof value === "string") {
      const trimmed = value.trim();
      const numeric = Number(trimmed);
      if (trimmed && Number.isFinite(numeric)) return formatValue(numeric);
      return value;
    }
    if (value === null || value === undefined) return "-";
    if (Array.isArray(value)) return value.map((item) => formatValue(item)).join(", ");
    if (typeof value === "object") {
      return Object.entries(value)
        .map(([name, nestedValue]) => `${name}: ${formatValue(nestedValue)}`)
        .join(" · ");
    }
    return String(value);
  }

  function updateModeUi() {
    const mode = app.status.mode;
    dom.modeBadge.textContent = modeLabel(mode);
    setChip(dom.modeBadge, mode === "demo" ? "warning" : mode ? "safe" : "muted");
    dom.statusMode.textContent = modeLabel(mode);
    if (mode === "demo") {
      dom.statusMode.classList.add("is-danger");
    } else {
      dom.statusMode.classList.remove("is-danger");
    }
  }

  function isWebSocketOpen() {
    return Boolean(app.ws && app.ws.readyState === WebSocket.OPEN);
  }

  function updateConnectionUi() {
    const serverConnected = app.status.connected === true;
    let label = "未连接";
    let state = "muted";
    if (app.wsState === "connecting") {
      label = "WebSocket 连接中";
      state = "warning";
    } else if (isWebSocketOpen() && serverConnected) {
      label = "服务已连接";
      state = "safe";
    } else if (isWebSocketOpen()) {
      label = "链路已连接";
      state = "warning";
    } else if (app.authenticated) {
      label = "等待 WebSocket";
      state = "warning";
    }
    dom.connectionText.textContent = label;
    setChip(dom.connectionBadge, state);
    dom.statusConnected.textContent = serverConnected ? "已连接" : "未确认";
    dom.statusConnected.classList.toggle("is-safe", serverConnected);
    dom.statusConnected.classList.toggle("is-danger", !serverConnected);
  }

  function updateLeaseUi() {
    const owner = app.status.control_owner;
    if (owner === "self") {
      dom.leaseValue.textContent = "本页操作者";
      dom.leaseValue.style.color = "var(--accent-strong)";
    } else if (owner === "other") {
      dom.leaseValue.textContent = "其他操作者";
      dom.leaseValue.style.color = "var(--danger)";
    } else {
      dom.leaseValue.textContent = "无当前操作者";
      dom.leaseValue.style.color = "";
    }
  }

  function updateArmUi() {
    const armed = app.status.armed === true;
    dom.armBadge.textContent = armed ? "已启用" : app.armRequestPending ? "Arm 请求中" : "未启用";
    setChip(dom.armBadge, armed ? "warning" : app.armRequestPending ? "warning" : "muted");
    dom.armButton.textContent = app.armRequestPending ? "Arm 请求中" : "Arm";
    dom.armButton.disabled = !canArm() || app.armRequestPending;
    dom.stopButton.disabled = !isWebSocketOpen();
    if (dom.controlModeSelect) {
      dom.controlModeSelect.value = app.status.control_mode || "arms";
      dom.controlModeSelect.disabled = !canChangeControlMode();
      dom.controlModeSelect.querySelector('option[value="drive"]').disabled = app.status.drive_available !== true;
      dom.controlModeHint.textContent = app.modeRequestPending
        ? "模式切换请求已发送，等待确认；不会自动启用。"
        : app.status.drive_available !== true
          ? "底盘尚未开放：需完成轮向及底盘参数验证。三路视频仍可查看。"
          : "先 Stop 再切换，随后松开并重新握 Grip 接管。行驶：右 Grip＋右摇杆；松开即停。";
    }
    if (app.status.control_owner === "other") {
      dom.controlHint.textContent = "其他操作者持有控制权。观察端仍可点击 Stop，Arm 需等控制权释放。";
    } else if (armed) {
      dom.controlHint.textContent = app.status.control_mode === "drive"
        ? "行驶模式已启用：按住右 Grip，用右摇杆前后行驶、左右转向；松开 Grip 即停。"
        : "按住对应 Grip：移动手柄跟随位姿；左右推对应摇杆转动底部旋转轴，归中保持。横推摇杆时优先用摇杆转轴，其余位姿继续跟随。Y/B 切换爪子方向，扣扳机动作、松开保持；X/A 停止。";
    } else if (app.armFailure) {
      dom.controlHint.textContent = `启用被拒绝：${app.armFailure}`;
    } else if (app.status.metadata && app.status.metadata.allow_motion === false) {
      dom.controlHint.textContent = "服务端已关闭动作权限 allow_motion=false。前端保持未启用，先处理观测问题。";
    } else {
      dom.controlHint.textContent = "握住 Grip 接管并跟手；左右推对应摇杆转动底部旋转轴。默认扣扳机合爪，左 Y / 右 B 切换对应爪子开合，松开扳机保持；X / A 停止。首次接管请松开扳机、摇杆归中。";
    }
  }

  function updateRecordingUi() {
    const recording = app.status.recording === true;
    dom.recordingBadge.textContent = recording ? "记录中" : "未录制";
    setChip(dom.recordingBadge, recording ? "warning" : "muted");
    dom.statusRecording.textContent = recording ? "记录中" : "未录制";
    dom.statusRecording.classList.toggle("is-safe", recording);
    dom.recordStart.disabled = recording;
    dom.recordStop.disabled = !recording;
    if (app.status.recording_path) {
      dom.recordingPath.hidden = false;
      dom.recordingPath.textContent = `保存路径: ${String(app.status.recording_path)}`;
    }
  }

  function updateInputUi() {
    const age = app.lastInputSentAt ? performance.now() - app.lastInputSentAt : Infinity;
    const fresh = age <= INPUT_FRESH_MS && app.inputPackets > 0;
    const waitingForControllers = app.xr.active && !app.xr.controllersReady;
    dom.statusInput.textContent = waitingForControllers
      ? "等待双手控制器"
      : fresh
        ? "WebXR 输入新鲜"
        : app.xr.active
          ? "等待新鲜输入"
          : "未启动";
    dom.statusInput.classList.toggle("is-safe", fresh && !waitingForControllers);
    dom.statusInput.classList.toggle("is-danger", app.xr.active && !fresh && !waitingForControllers);
    const stats = app.status.stats || {};
    const fps = firstFinite(stats.sample_hz, stats.fps);
    const observationAge = firstFinite(stats.observation_age_ms);
    const inputStatus = app.status.input_status && typeof app.status.input_status === "object"
      ? app.status.input_status
      : {};
    const tracked = inputStatus.tracked && typeof inputStatus.tracked === "object"
      ? inputStatus.tracked
      : {};
    const trackedText = Object.keys(tracked).length
      ? `L:${tracked.left === true ? "ok" : "lost"} R:${tracked.right === true ? "ok" : "lost"}`
      : "追踪: -";
    const inputParts = [];
    if (Number.isFinite(inputStatus.clients)) inputParts.push(`${inputStatus.clients} 端`);
    inputParts.push(trackedText);
    if (Number.isFinite(inputStatus.age_ms)) inputParts.push(`${formatValue(inputStatus.age_ms)} ms`);
    if (Number.isInteger(inputStatus.seq)) inputParts.push(`seq ${inputStatus.seq}`);
    dom.statFps.textContent = fps === null ? "-" : `${formatValue(fps)} Hz`;
    dom.statLatency.textContent = observationAge === null ? "-" : `${formatValue(observationAge)} ms`;
    dom.statDropped.textContent = inputParts.length ? inputParts.join(" /") : "-";
  }

  function updateStatusAge() {
    if (!app.statusUpdatedAt) {
      dom.statusAge.textContent = "尚未回读";
      return;
    }
    const age = Math.max(0, performance.now() - app.statusUpdatedAt);
    dom.statusAge.textContent = `${Math.round(age)} ms 前回读`;
  }

  function firstFinite(...values) {
    for (const value of values) {
      if (Number.isFinite(value)) return value;
    }
    return null;
  }

  function updateAllUi() {
    updateModeUi();
    updateConnectionUi();
    updateLeaseUi();
    updateArmUi();
    updateRecordingUi();
    updateInputUi();
    updateStatusAge();
    updateJointState();
    updateImmersiveUi();
    updateVideoPeerUi();
  }

  function updateJointState() {
    const state = app.status.state && typeof app.status.state === "object" ? app.status.state : {};
    const entries = Object.entries(state);
    if (!entries.length) {
      dom.jointState.replaceChildren(createTextNode("等待 /api/status 的 state", "empty-inline"));
      return;
    }
    const fragment = document.createDocumentFragment();
    entries.slice(0, 32).forEach(([name, value]) => {
      const cell = document.createElement("div");
      cell.className = "joint-value";
      const nameNode = document.createElement("span");
      nameNode.textContent = name;
      const valueNode = document.createElement("strong");
      valueNode.textContent = formatValue(value);
      cell.append(nameNode, valueNode);
      fragment.appendChild(cell);
    });
    dom.jointState.replaceChildren(fragment);
  }

  function createTextNode(text, className = "") {
    const node = document.createElement("div");
    node.textContent = text;
    if (className) node.className = className;
    return node;
  }

  function updateImmersiveUi() {
    if (!app.xr.supportChecked) {
      dom.xrStatus.textContent = "正在检查 WebXR 支持";
      dom.xrStatus.className = "inline-status";
    } else if (!app.xr.supported) {
      dom.xrStatus.textContent = "此浏览器不支持 immersive-vr";
      dom.xrStatus.className = "inline-status is-error";
    } else if (app.xr.active && !app.xr.controllersReady) {
      dom.xrStatus.textContent = "WebXR immersive-vr 已启动，等待双手控制器追踪，当前不发送动作输入。";
      dom.xrStatus.className = "inline-status";
    } else if (app.xr.active) {
      dom.xrStatus.textContent = "按住对应 Grip 即可接管；X/A 停止；Y/B 切换对应夹爪开合方向。";
      dom.xrStatus.className = "inline-status is-ready";
    } else {
      dom.xrStatus.textContent = "WebXR 可用。进入后双眼共享同一组三路单目视频。";
      dom.xrStatus.className = "inline-status is-ready";
    }
    dom.immersiveButton.disabled = !app.authenticated || !app.xr.supported || !isWebSocketOpen() || app.xr.active;
    dom.immersiveButton.textContent = app.xr.active ? "VR 会话中" : "进入沉浸式 VR";
  }

  function logEvent(message, kind = "info") {
    const timeNode = document.createElement("span");
    timeNode.className = "event-time";
    timeNode.textContent = new Date().toLocaleTimeString("zh-CN", { hour12: false });
    const messageNode = document.createElement("span");
    messageNode.textContent = message;
    dom.eventLog.replaceChildren(timeNode, messageNode);
    dom.eventLog.dataset.kind = kind;
  }

  function showToast(message, kind = "info") {
    const toast = document.createElement("div");
    toast.className = `toast${kind === "error" ? " is-error" : kind === "safe" ? " is-safe" : ""}`;
    toast.textContent = message;
    dom.toastRegion.appendChild(toast);
    window.setTimeout(() => toast.remove(), 4200);
  }

  function handleApiError(error, context) {
    const message = `${context}: ${error.message || "未知错误"}`;
    if (error.status === 401) {
      setUnauthenticated("需要重新认证");
    } else {
      setStatusError(message);
      logEvent(message, "error");
    }
  }

  function openWebSocket() {
    if (!app.authenticated || app.ws || app.wsState === "connecting") return;
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const url = `${protocol}//${window.location.host}/ws`;
    app.wsState = "connecting";
    updateAllUi();
    let socket;
    try {
      socket = new WebSocket(url);
    } catch (error) {
      app.wsState = "closed";
      logEvent(`WebSocket 创建失败: ${error.message || "未知错误"}`, "error");
      scheduleWebSocketReconnect();
      updateAllUi();
      return;
    }
    app.ws = socket;
    socket.addEventListener("open", () => {
      if (app.ws !== socket) return;
      app.wsState = "open";
      logEvent("WebSocket 已连接，等待服务端状态。", "safe");
      updateAllUi();
    });
    socket.addEventListener("message", (event) => {
      if (app.ws !== socket) return;
      handleWebSocketMessage(event.data);
    });
    socket.addEventListener("error", () => {
      if (app.ws !== socket) return;
      logEvent("WebSocket 发生传输错误。", "error");
      updateAllUi();
    });
    socket.addEventListener("close", () => {
      if (app.ws !== socket) return;
      app.ws = null;
      app.wsState = "closed";
      app.modeRequestPending = null;
      if (app.xr.active || app.status.control_owner === "self") {
        failSafeStop("WebSocket 已断开", { endSession: true });
      }
      logEvent("WebSocket 已断开，控制输入已停止。", "error");
      updateAllUi();
      scheduleWebSocketReconnect();
    });
  }

  function scheduleWebSocketReconnect() {
    if (!app.authenticated || !app.shouldReconnect || app.wsReconnectTimer) return;
    app.wsReconnectTimer = window.setTimeout(() => {
      app.wsReconnectTimer = null;
      openWebSocket();
    }, 1500);
  }

  function handleWebSocketMessage(raw) {
    let message;
    try {
      message = typeof raw === "string" ? JSON.parse(raw) : JSON.parse(String(raw));
    } catch (_error) {
      logEvent("收到无法解析的 WebSocket 消息。", "error");
      return;
    }
    if (!message || typeof message !== "object") return;
    if (message.type === "status") {
      applyStatus(message, "ws");
      return;
    }
    if (message.type === "error") {
      const error = typeof message.error === "string" ? message.error : "服务端拒绝了请求";
      if (app.armRequestPending) app.armFailure = controlErrorHint(error);
      app.armRequestPending = false;
      app.modeRequestPending = null;
      setStatusError(error);
      showToast(error, "error");
      logEvent(`WebSocket 服务端错误: ${error}`, "error");
      updateAllUi();
    }
  }

  function sendWebSocketMessage(message) {
    if (!isWebSocketOpen()) {
      logEvent("WebSocket 未连接，未发送控制请求。", "error");
      return false;
    }
    try {
      app.ws.send(JSON.stringify(message));
      return true;
    } catch (error) {
      logEvent(`WebSocket 发送失败: ${error.message || "未知错误"}`, "error");
      return false;
    }
  }

  function sendStopPacket(reason, force = false) {
    if (!force && !isActualControllerClient()) return false;
    const sent = sendWebSocketMessage({ type: "stop" });
    if (sent) {
      app.xr.stopSent = true;
      logEvent(`已发送 Stop: ${reason}`, "safe");
    }
    return sent;
  }

  function manualStop(reason = "操作员 Stop") {
    const sent = sendStopPacket(reason, true);
    app.armRequestPending = false;
    app.xr.gripActivationReady = false;
    app.xr.stopSent = sent;
    updateAllUi();
    if (sent) showToast("Stop 已发送，等待服务端状态回读。", "safe");
  }

  function requestArm(source = "桌面 Arm", activation = "neutral") {
    if (!canArm(activation === "grip")) {
      logEvent("Arm 未发送: 需要同一 websocket 的新鲜双手中立输入。", "error");
      return false;
    }
    const sent = sendWebSocketMessage(activation === "grip" ? {type: "arm", activation: "grip"} : {type: "arm"});
    if (sent) {
      app.armFailure = null;
      app.armRequestPending = true;
      app.controllerClient = true;
      logEvent(`${source} 请求已发送，等待服务端确认。`, "info");
      updateAllUi();
    }
    return sent;
  }

  function controlErrorHint(error) {
    const message = String(error || "未知错误");
    const limit = message.match(/(left|right)_arm_(\w+): present position (-?\d+) is outside saved range \[(-?\d+),\s*(-?\d+)\]/);
    if (limit) {
      const joints = { shoulder_pan: "肩旋转", shoulder_lift: "肩抬升", elbow_flex: "肘部", wrist_flex: "腕俯仰", wrist_roll: "腕旋转", gripper: "夹爪" };
      return `${limit[1] === "left" ? "左" : "右"}${joints[limit[2]] || limit[2]}超限：${limit[3]}，允许 ${limit[4]}–${limit[5]}`;
    }
    if (message.includes("controller input delayed")) return "手柄输入传输延迟，控制已停止；松开后重新握 Grip 接管";
    if (message.includes("tracked neutral controllers")) return "需要双手追踪；请松开 Grip、扳机并将摇杆归中";
    return message;
  }

  function canChangeControlMode() {
    return ["arms", "drive"].includes(app.status.control_mode) &&
      app.authenticated && isWebSocketOpen() && !app.status.armed && !app.status.recording &&
      !app.status.control_owner && !app.status.stop_unconfirmed && !app.armRequestPending && !app.modeRequestPending;
  }

  function requestControlMode(mode) {
    if (!canChangeControlMode() || !["arms", "drive"].includes(mode) ||
        (mode === "drive" && app.status.drive_available !== true)) {
      updateAllUi();
      return false;
    }
    const sent = sendWebSocketMessage({ type: "set_control_mode", mode });
    if (sent) app.modeRequestPending = mode;
    updateAllUi();
    return sent;
  }

  function canArm(allowGrip = false) {
    if (!app.authenticated || !isWebSocketOpen() || app.status.armed || app.armRequestPending) return false;
    if (!app.xr.active || app.xr.safetyTripped || app.status.stop_unconfirmed) return false;
    if (app.xr.session && !xrHasInputFocus(app.xr.session)) return false;
    if (!cameraFramesFresh()) return false;
    if (app.status.control_owner === "other") return false;
    if (app.status.metadata && app.status.metadata.allow_motion === false) return false;
    if (
      !app.lastInputSentAt ||
      app.inputSocket !== app.ws ||
      performance.now() - app.lastInputSentAt > INPUT_FRESH_MS
    ) return false;
    return controllersAreNeutral(allowGrip);
  }

  function controllersAreNeutral(allowGrip = false) {
    return ["left", "right"].every((hand) => {
      const controller = app.controllers[hand];
      return (
        controller.tracked === true &&
        (allowGrip || controller.grip === false) &&
        controller.trigger < 0.1 &&
        Math.max(Math.abs(controller.thumbstick[0]), Math.abs(controller.thumbstick[1])) < 0.15
      );
    });
  }

  function isActualControllerClient() {
    if (app.status.control_owner === "self") return true;
    return Boolean(
      app.controllerClient &&
      app.xr.active &&
      app.inputPackets > 0 &&
      app.status.control_owner !== "other",
    );
  }

  function startVideoHealth() {
    if (!app.authenticated || app.videoHealthTimer) return;
    app.videoHealthTimer = window.setInterval(checkVideoHealth, VIDEO_HEALTH_INTERVAL_MS);
    checkVideoHealth();
  }

  function startVideoPeer() {
    if (!app.authenticated || app.video.peer || app.video.connecting || app.video.retryTimer) return;
    void connectVideoPeer();
  }

  function waitForIceGatheringComplete(peer) {
    if (peer.iceGatheringState === "complete") return Promise.resolve();
    return new Promise((resolve, reject) => {
      let settled = false;
      let timeoutId = null;
      const cleanup = () => {
        peer.removeEventListener("icegatheringstatechange", onStateChange);
        if (timeoutId !== null) window.clearTimeout(timeoutId);
      };
      const finish = (error = null) => {
        if (settled) return;
        settled = true;
        cleanup();
        if (error) reject(error);
        else resolve();
      };
      const onStateChange = () => {
        if (peer.iceGatheringState === "complete") finish();
      };
      peer.addEventListener("icegatheringstatechange", onStateChange);
      timeoutId = window.setTimeout(
        () => finish(new Error("ICE gathering 超时，未发送未完成的 offer。")),
        ICE_GATHERING_TIMEOUT_MS,
      );
      onStateChange();
    });
  }

  function isCurrentVideoPeer(peer, generation) {
    return Boolean(
      peer &&
      app.video.peer === peer &&
      app.video.generation === generation &&
      app.authenticated,
    );
  }

  async function connectVideoPeer() {
    if (!app.authenticated || app.video.peer || app.video.connecting || app.video.retryTimer) return;
    const generation = app.video.generation + 1;
    app.video.generation = generation;
    app.video.connecting = true;
    app.video.state = "connecting";
    app.video.error = "";
    app.video.failurePeer = null;
    app.video.wasConnected = false;
    updateVideoPeerUi();

    let peer = null;
    try {
      if (typeof RTCPeerConnection !== "function") {
        throw new Error("浏览器不支持 WebRTC");
      }
      peer = new RTCPeerConnection({ iceServers: [], bundlePolicy: "max-bundle" });
      app.video.peer = peer;
      const transceivers = CAMERA_NAMES.map(() =>
        peer.addTransceiver("video", { direction: "recvonly" }),
      );
      app.video.transceivers = transceivers;
      peer.addEventListener("track", (event) => handleVideoTrack(event, peer, transceivers));
      peer.addEventListener("connectionstatechange", () => {
        handleVideoPeerState(peer, generation);
      });
      peer.addEventListener("iceconnectionstatechange", () => {
        handleVideoPeerState(peer, generation);
      });

      const offer = await peer.createOffer();
      if (!isCurrentVideoPeer(peer, generation)) return;
      await peer.setLocalDescription(offer);
      await waitForIceGatheringComplete(peer);
      if (!isCurrentVideoPeer(peer, generation)) return;
      const localDescription = peer.localDescription;
      if (!localDescription || typeof localDescription.sdp !== "string" || !localDescription.sdp) {
        throw new Error("WebRTC 未生成本地 offer");
      }
      const answer = await requestJson("/api/webrtc/offer", {
        method: "POST",
        credentials: "include",
        body: JSON.stringify({ sdp: localDescription.sdp, type: "offer" }),
      });
      if (!isCurrentVideoPeer(peer, generation)) return;
      if (
        !answer ||
        answer.type !== "answer" ||
        typeof answer.sdp !== "string" ||
        !answer.sdp ||
        !Array.isArray(answer.cameras) ||
        answer.cameras.join(",") !== CAMERA_NAMES.join(",")
      ) {
        throw new Error("服务端返回的 WebRTC answer 或相机顺序无效");
      }
      await peer.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
      if (!isCurrentVideoPeer(peer, generation)) return;
      app.video.state = peer.connectionState === "connected" ||
        peer.iceConnectionState === "connected" ||
        peer.iceConnectionState === "completed"
        ? "connected"
        : "negotiating";
      startVideoStats(peer, generation);
      updateVideoPeerUi();
    } catch (error) {
      if (generation !== app.video.generation || (peer && app.video.peer !== peer)) return;
      if (error.status === 401) {
        setUnauthenticated("需要重新认证");
      } else {
        handleVideoPeerFailure(error, peer, generation);
      }
    } finally {
      if (generation === app.video.generation) {
        app.video.connecting = false;
        updateVideoPeerUi();
      }
    }
  }

  function handleVideoPeerState(peer, generation) {
    if (!isCurrentVideoPeer(peer, generation)) return;
    const connected = peer.connectionState === "connected" ||
      peer.iceConnectionState === "connected" ||
      peer.iceConnectionState === "completed";
    if (connected) {
      const wasConnected = app.video.wasConnected;
      app.video.wasConnected = true;
      app.video.state = "connected";
      app.video.error = "";
      app.video.failurePeer = null;
      app.video.retryAttempt = 0;
      app.video.retryDelayMs = 0;
      updateVideoPeerUi();
      if (!wasConnected) logEvent("WebRTC 视频已连接，正在接收三路编码视频。", "safe");
      return;
    }

    const failed = ["failed", "disconnected", "closed"].includes(peer.connectionState) ||
      ["failed", "disconnected", "closed"].includes(peer.iceConnectionState);
    if (failed) {
      handleVideoPeerFailure(new Error("WebRTC 视频连接已断开"), peer, generation);
      return;
    }
    app.video.state = "negotiating";
    updateVideoPeerUi();
  }

  function handleVideoTrack(event, peer, transceivers) {
    if (app.video.peer !== peer || !event || !event.track || event.track.kind !== "video") return;
    const index = transceivers.findIndex((transceiver) => transceiver === event.transceiver);
    if (index < 0) {
      handleVideoPeerFailure(new Error("收到未匹配到 offer transceiver 的视频轨道"), peer, app.video.generation);
      return;
    }
    const camera = cameras[CAMERA_NAMES[index]];
    if (!camera || !camera.video) return;
    // aiortc groups all three tracks into one signalled MediaStream. Binding
    // event.streams[0] to each video makes every element play its first track.
    // Give each camera a single-track stream selected by its transceiver.
    const stream = typeof MediaStream === "function"
      ? new MediaStream([event.track])
      : null;
    if (!stream) {
      handleVideoPeerFailure(new Error("视频轨道没有可播放的 MediaStream"), peer, app.video.generation);
      return;
    }
    if (camera.track !== event.track) {
      camera.track = event.track;
      if (camera.video.srcObject !== stream) camera.video.srcObject = stream;
      event.track.addEventListener("ended", () => {
        if (camera.track !== event.track) return;
        camera.error = "视频轨道已结束";
        updateCameraVisual(camera);
      }, { once: true });
      void camera.video.play().catch((error) => {
        if (camera.track !== event.track) return;
        camera.error = `视频播放失败: ${error.message || "浏览器拒绝播放"}`;
        updateCameraVisual(camera);
      });
    }
    camera.error = null;
    camera.stale = true;
    camera.lastGoodAt = 0;
    camera.lastFrameTimestamp = null;
    camera.lastPresentedFrames = null;
    scheduleVideoFrameCallback(camera);
    updateCameraVisual(camera);
  }

  function handleVideoPeerFailure(error, peer, generation) {
    if (generation !== app.video.generation) return;
    if (peer && app.video.peer && app.video.peer !== peer) return;
    if (app.video.failurePeer === peer && app.video.state === "error") return;
    const message = error && error.message ? error.message : "WebRTC 视频连接失败";
    app.video.failurePeer = peer;
    if (app.video.peer === peer) {
      app.video.peer = null;
      app.video.transceivers = [];
    }
    app.video.connecting = false;
    app.video.state = "error";
    app.video.error = message;
    stopVideoStats();
    if (peer) {
      try {
        peer.close();
      } catch (_error) {
        // The peer may already be closed after a connection failure.
      }
    }
    updateVideoPeerUi();
    logEvent(`WebRTC 视频失败: ${message}`, "error");
    if (hasRobotControlIntent()) {
      failSafeStop("WebRTC 视频中断", { endSession: true });
    }
    // Reconnect viewing after headset sleep using the existing backoff. This
    // does not request ownership or clear a control safety latch.
    if (app.authenticated) retryVideoPeer();
  }

  function retryVideoPeer() {
    if (!app.authenticated || app.video.connecting || app.video.retryTimer) return;
    const exponent = Math.min(app.video.retryAttempt, 4);
    const delay = Math.min(VIDEO_RETRY_MAX_MS, VIDEO_RETRY_BASE_MS * 2 ** exponent);
    app.video.retryAttempt += 1;
    app.video.retryDelayMs = delay;
    app.video.state = "retry-wait";
    app.video.error = app.video.error || "WebRTC 视频连接失败";
    updateVideoPeerUi();
    app.video.retryTimer = window.setTimeout(() => {
      app.video.retryTimer = null;
      app.video.retryDelayMs = 0;
      app.video.failurePeer = null;
      app.video.state = "connecting";
      void connectVideoPeer();
    }, delay);
  }

  function stopVideoPeer() {
    app.video.generation += 1;
    if (app.video.retryTimer) {
      window.clearTimeout(app.video.retryTimer);
      app.video.retryTimer = null;
    }
    const peer = app.video.peer;
    stopVideoStats();
    app.video.peer = null;
    app.video.transceivers = [];
    app.video.connecting = false;
    app.video.state = "idle";
    app.video.error = "";
    app.video.retryDelayMs = 0;
    app.video.failurePeer = null;
    app.video.wasConnected = false;
    if (peer) {
      try {
        peer.close();
      } catch (_error) {
        // The peer may already be closed during logout.
      }
    }
    CAMERA_NAMES.forEach((name) => {
      const camera = cameras[name];
      if (!camera) return;
      if (camera.frameCallbackId !== null && camera.video &&
        typeof camera.video.cancelVideoFrameCallback === "function") {
        try {
          camera.video.cancelVideoFrameCallback(camera.frameCallbackId);
        } catch (_error) {
          // The callback may already be running.
        }
      }
      camera.frameCallbackId = null;
      camera.track = null;
      camera.hasFrame = false;
      camera.stale = false;
      camera.error = null;
      camera.lastGoodAt = 0;
      camera.lastFrameTimestamp = null;
      camera.lastPresentedFrames = null;
      camera.frameVersion = 0;
      camera.uploadedVersion = -1;
      if (camera.video) {
        camera.video.pause();
        camera.video.srcObject = null;
        camera.video.classList.remove("has-frame", "is-stale");
      }
      if (camera.frame) camera.frame.classList.remove("has-frame");
      updateCameraVisual(camera);
    });
    updateVideoPeerUi();
  }

  function scheduleVideoFrameCallback(camera) {
    if (!camera || !camera.video || typeof camera.video.requestVideoFrameCallback !== "function") {
      if (camera && !camera.error) camera.error = "浏览器不支持视频帧回调";
      updateCameraVisual(camera);
      return;
    }
    if (camera.frameCallbackId !== null) return;
    const video = camera.video;
    camera.frameCallbackId = video.requestVideoFrameCallback((_now, metadata) => {
      camera.frameCallbackId = null;
      if (camera.video !== video || camera.track === null || !video.srcObject) return;
      const presentedFrames = Number.isFinite(metadata && metadata.presentedFrames)
        ? metadata.presentedFrames
        : null;
      const timestamp = firstFinite(
        metadata && metadata.captureTime,
        metadata && metadata.mediaTime,
        metadata && metadata.expectedDisplayTime,
      );
      if (
        presentedFrames === null ||
        presentedFrames !== camera.lastPresentedFrames ||
        timestamp !== camera.lastFrameTimestamp
      ) {
        camera.frameVersion += 1;
        camera.lastPresentedFrames = presentedFrames;
        camera.lastFrameTimestamp = timestamp;
      }
      camera.lastGoodAt = performance.now();
      camera.hasFrame = true;
      camera.stale = false;
      camera.error = null;
      updateCameraVisual(camera);
      scheduleVideoFrameCallback(camera);
    });
  }

  function cameraFramesFresh(now = performance.now()) {
    if (app.video.state !== "connected") return false;
    return CAMERA_NAMES.every(name => {
      const camera = cameras[name];
      return Boolean(camera && camera.hasFrame && camera.lastGoodAt &&
        now - camera.lastGoodAt <= VIDEO_STALE_MS);
    });
  }

  function checkVideoHealth() {
    const now = performance.now();
    let liveCount = 0;
    let staleOwnedVideo = false;
    CAMERA_NAMES.forEach((name) => {
      const camera = cameras[name];
      const live = Boolean(camera && camera.hasFrame && camera.lastGoodAt &&
        now - camera.lastGoodAt <= VIDEO_STALE_MS);
      if (live) liveCount += 1;
      if (camera && camera.hasFrame && !live) camera.stale = true;
      updateCameraVisual(camera);
      if (!live) staleOwnedVideo = true;
    });
    dom.cameraHealth.textContent = `${liveCount} / ${CAMERA_NAMES.length} 在线`;
    // Diagnostic packets do not mean the browser owns the robot. Sleep or a
    // system-menu transition may pause idle video; let that session recover.
    // Arming still requires fresh video, and active control still stops.
    if (staleOwnedVideo && hasRobotControlIntent() && !app.xr.safetyTripped) {
      failSafeStop("视频帧过期", { endSession: true });
    }
  }

  function updateCameraVisual(camera) {
    if (!camera || !camera.state || !camera.video) return;
    const age = camera.lastGoodAt ? performance.now() - camera.lastGoodAt : Infinity;
    const live = camera.hasFrame && age <= VIDEO_STALE_MS;
    camera.stale = camera.hasFrame && !live;
    camera.state.classList.remove("is-live", "is-error");
    camera.video.classList.toggle("has-frame", camera.hasFrame);
    camera.video.classList.toggle("is-stale", camera.stale);
    if (camera.frame) {
      camera.frame.classList.toggle("has-frame", camera.hasFrame);
    }
    if (live) {
      camera.state.textContent = `LIVE ${Math.max(0, Math.round(age))} ms`;
      camera.state.classList.add("is-live");
    } else if (camera.error) {
      camera.state.textContent = camera.error;
      camera.state.classList.add("is-error");
    } else if (camera.stale) {
      camera.state.textContent = "视频帧过期";
    } else {
      camera.state.textContent = "等待视频";
    }
  }

  function updateVideoPeerUi() {
    if (!dom.videoPeerStatus || !dom.videoPeerError || !dom.videoPeerRetry || !dom.videoPeerStats) return;
    const state = app.video.state;
    let status = "等待认证后建立 WebRTC 视频";
    if (state === "connecting") status = "正在建立 WebRTC 视频连接…";
    else if (state === "negotiating") status = "WebRTC 已协商，等待三路视频轨道…";
    else if (state === "connected") status = "WebRTC 视频已连接 · 3 路接收";
    else if (state === "retry-wait") status = "WebRTC 视频将在退避后重试";
    dom.videoPeerStatus.textContent = status;
    const showError = state === "error" || state === "retry-wait";
    dom.videoPeerError.hidden = !showError;
    dom.videoPeerError.textContent = state === "retry-wait"
      ? `${app.video.error}（${(app.video.retryDelayMs / 1000).toFixed(1)} s 后）`
      : app.video.error;
    dom.videoPeerRetry.hidden = !showError || !app.authenticated;
    dom.videoPeerRetry.disabled = state === "retry-wait" || app.video.connecting;
    dom.videoPeerRetry.textContent = state === "retry-wait" ? "视频重试退避中" : "重试 WebRTC 视频";
    const stats = app.video.stats;
    const fps = Number.isFinite(stats.fpsAverage)
      ? `每路 ${formatFpsRange(stats.fpsMin, stats.fpsMax)} FPS（均值 ${formatValue(stats.fpsAverage)}）`
      : "每路解码 FPS -";
    const bitrate = Number.isFinite(stats.bitrate) ? formatBitrate(stats.bitrate) : "总码率 -";
    dom.videoPeerStats.textContent = `实际解码 ${fps} · codec ${stats.codec || "-"} · ${bitrate}`;
  }

  function formatFpsRange(min, max) {
    if (!Number.isFinite(min) || !Number.isFinite(max)) return "-";
    if (Math.abs(max - min) < 0.05) return formatValue((min + max) / 2);
    return `${formatValue(min)}–${formatValue(max)}`;
  }

  function formatBitrate(bitsPerSecond) {
    if (!Number.isFinite(bitsPerSecond) || bitsPerSecond < 0) return "码率 -";
    if (bitsPerSecond >= 1_000_000) return `总码率 ${(bitsPerSecond / 1_000_000).toFixed(2)} Mbps`;
    return `总码率 ${(bitsPerSecond / 1_000).toFixed(0)} Kbps`;
  }

  function startVideoStats(peer, generation) {
    if (app.video.statsTimer || !peer || typeof peer.getStats !== "function") return;
    app.video.statsTimer = window.setInterval(() => {
      void refreshVideoStats(peer, generation);
    }, VIDEO_STATS_INTERVAL_MS);
    void refreshVideoStats(peer, generation);
  }

  function stopVideoStats() {
    if (app.video.statsTimer) {
      window.clearInterval(app.video.statsTimer);
      app.video.statsTimer = null;
    }
    app.video.statsInFlight = false;
    app.video.statsPrevious = {};
    app.video.stats = {
      fpsMin: null,
      fpsMax: null,
      fpsAverage: null,
      streamCount: 0,
      bitrate: null,
      codec: "-",
    };
  }

  async function refreshVideoStats(peer, generation) {
    if (
      app.video.statsInFlight ||
      !isCurrentVideoPeer(peer, generation) ||
      typeof peer.getStats !== "function"
    ) return;
    app.video.statsInFlight = true;
    try {
      const report = await peer.getStats();
      if (!isCurrentVideoPeer(peer, generation)) return;
      const codecs = new Map();
      report.forEach((stat) => {
        if (stat.type === "codec" && typeof stat.id === "string") codecs.set(stat.id, stat);
      });
      const previous = app.video.statsPrevious;
      const nextPrevious = {};
      const decodedFpsValues = [];
      let bitrate = 0;
      let hasBitrate = false;
      let codec = "-";
      report.forEach((stat) => {
        if (stat.type !== "inbound-rtp" || (stat.kind || stat.mediaType) !== "video") return;
        nextPrevious[stat.id] = {
          framesDecoded: stat.framesDecoded,
          bytesReceived: stat.bytesReceived,
          timestamp: stat.timestamp,
        };
        const old = previous[stat.id];
        const intervalMs = old && Number.isFinite(old.timestamp) && Number.isFinite(stat.timestamp)
          ? stat.timestamp - old.timestamp
          : 0;
        let streamFps = null;
        if (Number.isFinite(stat.framesPerSecond)) {
          streamFps = stat.framesPerSecond;
        } else if (
          intervalMs > 0 &&
          Number.isFinite(old && old.framesDecoded) &&
          Number.isFinite(stat.framesDecoded)
        ) {
          streamFps = (stat.framesDecoded - old.framesDecoded) * 1000 / intervalMs;
        }
        if (Number.isFinite(streamFps)) decodedFpsValues.push(streamFps);
        if (
          intervalMs > 0 &&
          Number.isFinite(old && old.bytesReceived) &&
          Number.isFinite(stat.bytesReceived)
        ) {
          bitrate += (stat.bytesReceived - old.bytesReceived) * 8 * 1000 / intervalMs;
          hasBitrate = true;
        }
        const codecStat = codecs.get(stat.codecId);
        if (codecStat && typeof codecStat.mimeType === "string") {
          codec = codecStat.mimeType.replace(/^video\//i, "").toUpperCase();
        }
      });
      app.video.statsPrevious = nextPrevious;
      const fpsMin = decodedFpsValues.length ? Math.min(...decodedFpsValues) : null;
      const fpsMax = decodedFpsValues.length ? Math.max(...decodedFpsValues) : null;
      app.video.stats = {
        fpsMin,
        fpsMax,
        fpsAverage: decodedFpsValues.length
          ? decodedFpsValues.reduce((total, value) => total + value, 0) / decodedFpsValues.length
          : null,
        streamCount: decodedFpsValues.length,
        bitrate: hasBitrate ? bitrate : null,
        codec,
      };
      updateVideoPeerUi();
    } catch (_error) {
      // Stats are observability only; the media and control paths remain independent.
    } finally {
      app.video.statsInFlight = false;
    }
  }

  function setMainCamera(name) {
    if (!CAMERA_NAMES.includes(name)) return;
    app.mainCamera = name;
    dom.mainCameraSelect.value = name;
    CAMERA_NAMES.forEach((cameraName) => {
      const tile = cameras[cameraName].tile;
      if (!tile) return;
      const selected = cameraName === name;
      tile.classList.toggle("is-main", selected);
      tile.setAttribute("aria-label", `${CAMERA_LABELS[cameraName]}相机，${selected ? "当前为" : "点击设为"} VR 主画面`);
    });
    logEvent(`VR 主画面已选择: ${CAMERA_LABELS[name]}`, "info");
  }

  function bindCameraControls() {
    dom.mainCameraSelect.addEventListener("change", () => setMainCamera(dom.mainCameraSelect.value));
    CAMERA_NAMES.forEach((name) => {
      const tile = cameras[name].tile;
      if (!tile) return;
      const select = () => setMainCamera(name);
      tile.addEventListener("click", select);
      tile.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          select();
        }
      });
    });
  }

  function bindRecordingControls() {
    dom.recordForm.addEventListener("submit", async (event) => {
      event.preventDefault();
      dom.recordOutcome.value = "null";
      const task = dom.taskInput.value.trim();
      if (!task) {
        dom.recordingMessage.textContent = "请先填写任务名称。";
        dom.taskInput.focus();
        return;
      }
      dom.recordStart.disabled = true;
      dom.recordingMessage.textContent = "正在请求服务端开始记录。";
      try {
        const result = await requestJson("/api/record/start", {
          method: "POST",
          body: JSON.stringify({ task }),
        });
        if (!result || result.recording !== true) {
          throw new ApiError("服务端未确认开始记录");
        }
        if (result.path) {
          dom.recordingPath.hidden = false;
          dom.recordingPath.textContent = `保存路径: ${String(result.path)}`;
        }
        dom.recordingMessage.textContent = "服务端已确认开始记录。";
        logEvent("数据记录已开始。", "safe");
        await refreshStatus();
      } catch (error) {
        dom.recordingMessage.textContent = `开始记录失败: ${error.message || "未知错误"}`;
        handleApiError(error, "开始记录失败");
      } finally {
        updateRecordingUi();
      }
    });

    dom.recordStop.addEventListener("click", async () => {
      dom.recordStop.disabled = true;
      dom.recordingMessage.textContent = "正在请求服务端停止记录。";
      try {
        const selectedOutcome = dom.recordOutcome.value;
        const success = selectedOutcome === "true" ? true : selectedOutcome === "false" ? false : null;
        const result = await requestJson("/api/record/stop", {
          method: "POST",
          body: JSON.stringify({ success }),
        });
        if (
          !result ||
          !Object.prototype.hasOwnProperty.call(result, "success") ||
          ![true, false, null].includes(result.success)
        ) {
          throw new ApiError("服务端未返回记录结果");
        }
        if (result.success === true) {
          dom.recordingMessage.textContent = "记录已保存；任务结果：成功。";
        } else if (result.success === false) {
          dom.recordingMessage.textContent = "记录已保存；任务结果：失败。";
        } else {
          dom.recordingMessage.textContent = "记录已保存；任务结果：未知。";
        }
        logEvent("数据记录已停止，结果已回读。", result.success === false ? "error" : "safe");
        await refreshStatus();
      } catch (error) {
        dom.recordingMessage.textContent = `停止记录失败: ${error.message || "未知错误"}`;
        handleApiError(error, "停止记录失败");
      } finally {
        updateRecordingUi();
      }
    });
  }

  function initTheme() {
    const stored = (() => {
      try {
        return window.localStorage.getItem("xlerobot-teleop-theme");
      } catch (_error) {
        return null;
      }
    })();
    if (stored === "light" || stored === "dark") {
      document.documentElement.dataset.theme = stored;
    }
    updateThemeButton();
    dom.themeToggle.addEventListener("click", () => {
      const current = document.documentElement.dataset.theme ||
        (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
      const next = current === "light" ? "dark" : "light";
      document.documentElement.dataset.theme = next;
      try {
        window.localStorage.setItem("xlerobot-teleop-theme", next);
      } catch (_error) {
        // Theme still applies for this page.
      }
      updateThemeButton();
    });
  }

  function updateThemeButton() {
    const current = document.documentElement.dataset.theme ||
      (window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    dom.themeToggle.textContent = current === "light" ? "深色界面" : "浅色界面";
  }

  function initWebGL() {
    if (app.xr.gl) return app.xr.gl;
    const canvas = dom.xrCanvas;
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: true, xrCompatible: true }) ||
      canvas.getContext("webgl", { alpha: false, antialias: true, xrCompatible: true });
    if (!gl) throw new Error("浏览器没有可用的 WebGL 上下文");

    const vertexSource = `
      attribute vec2 a_position;
      attribute vec2 a_uv;
      uniform mat4 u_mvp;
      varying vec2 v_uv;
      void main() {
        v_uv = a_uv;
        gl_Position = u_mvp * vec4(a_position, 0.0, 1.0);
      }
    `;
    const fragmentSource = `
      precision mediump float;
      uniform sampler2D u_texture;
      uniform float u_has_texture;
      varying vec2 v_uv;
      void main() {
        vec3 color;
        if (u_has_texture > 0.5) {
          color = texture2D(u_texture, v_uv).rgb;
        } else {
          float grid = step(0.94, fract(v_uv.x * 12.0)) + step(0.94, fract(v_uv.y * 8.0));
          color = mix(vec3(0.055, 0.075, 0.08), vec3(0.12, 0.15, 0.15), min(grid, 1.0));
        }
        float edge = min(min(v_uv.x, 1.0 - v_uv.x), min(v_uv.y, 1.0 - v_uv.y));
        float border = 1.0 - smoothstep(0.006, 0.022, edge);
        color = mix(color, vec3(0.84, 0.65, 0.35), border * 0.9);
        gl_FragColor = vec4(color, 1.0);
      }
    `;
    const program = createProgram(gl, vertexSource, fragmentSource);
    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([
        -0.5, -0.5, 0, 0,
        0.5, -0.5, 1, 0,
        -0.5, 0.5, 0, 1,
        0.5, 0.5, 1, 1,
      ]),
      gl.STATIC_DRAW,
    );
    gl.useProgram(program);
    const positionLocation = gl.getAttribLocation(program, "a_position");
    const uvLocation = gl.getAttribLocation(program, "a_uv");
    gl.enableVertexAttribArray(positionLocation);
    gl.enableVertexAttribArray(uvLocation);
    gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 16, 0);
    gl.vertexAttribPointer(uvLocation, 2, gl.FLOAT, false, 16, 8);
    gl.uniform1i(gl.getUniformLocation(program, "u_texture"), 0);
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
    gl.enable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);

    app.xr.gl = gl;
    app.xr.program = {
      program,
      buffer,
      mvp: gl.getUniformLocation(program, "u_mvp"),
      hasTexture: gl.getUniformLocation(program, "u_has_texture"),
    };
    CAMERA_NAMES.forEach((name) => {
      const texture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        1,
        1,
        0,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        new Uint8Array([15, 23, 25, 255]),
      );
      cameras[name].texture = texture;
    });
    const hudCanvas = document.createElement("canvas");
    hudCanvas.width = 1024;
    hudCanvas.height = 192;
    app.xr.hud = {
      canvas: hudCanvas, context: hudCanvas.getContext("2d"),
      texture: gl.createTexture(), key: "",
    };
    gl.bindTexture(gl.TEXTURE_2D, app.xr.hud.texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return gl;
  }

  function createProgram(gl, vertexSource, fragmentSource) {
    const vertex = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vertex, vertexSource);
    gl.compileShader(vertex);
    if (!gl.getShaderParameter(vertex, gl.COMPILE_STATUS)) {
      throw new Error(`WebGL 顶点着色器失败: ${gl.getShaderInfoLog(vertex) || "unknown"}`);
    }
    const fragment = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fragment, fragmentSource);
    gl.compileShader(fragment);
    if (!gl.getShaderParameter(fragment, gl.COMPILE_STATUS)) {
      throw new Error(`WebGL 片元着色器失败: ${gl.getShaderInfoLog(fragment) || "unknown"}`);
    }
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(`WebGL 程序链接失败: ${gl.getProgramInfoLog(program) || "unknown"}`);
    }
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    return program;
  }

  function identityMatrix() {
    return new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }

  function multiplyMatrices(a, b) {
    const out = new Float32Array(16);
    for (let column = 0; column < 4; column += 1) {
      for (let row = 0; row < 4; row += 1) {
        out[column * 4 + row] =
          a[row] * b[column * 4] +
          a[4 + row] * b[column * 4 + 1] +
          a[8 + row] * b[column * 4 + 2] +
          a[12 + row] * b[column * 4 + 3];
      }
    }
    return out;
  }

  function panelModel(x, y, z, width, height) {
    const model = identityMatrix();
    model[0] = width;
    model[5] = height;
    model[12] = x;
    model[13] = y;
    model[14] = z;
    return model;
  }

  function orderedCameraNames() {
    const others = CAMERA_NAMES.filter((name) => name !== app.mainCamera);
    return [app.mainCamera, others[0], others[1]];
  }

  function uploadCameraTexture(camera) {
    const gl = app.xr.gl;
    if (!gl || !camera.texture || !camera.video) return false;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, camera.texture);
    if (camera.uploadedVersion !== camera.frameVersion) {
      if (camera.hasFrame && camera.video.readyState >= 2 && camera.video.videoWidth && camera.video.videoHeight) {
        try {
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, camera.video);
          camera.uploadedVersion = camera.frameVersion;
        } catch (error) {
          logEvent(`VR 纹理上传失败: ${error.message || "未知错误"}`, "error");
        }
      }
    }
    return camera.hasFrame && camera.uploadedVersion >= 0;
  }

  function drawXrScene(pose, layer) {
    const gl = app.xr.gl;
    const xrProgram = app.xr.program;
    if (!gl || !xrProgram || !pose || !layer) return;
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    // Both eye viewports share the same framebuffer. Clear once, otherwise
    // rendering the second eye erases the first eye's image.
    gl.clearColor(0.035, 0.05, 0.055, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.useProgram(xrProgram.program);
    gl.bindBuffer(gl.ARRAY_BUFFER, xrProgram.buffer);
    const names = orderedCameraNames();
    const textureReady = Object.fromEntries(
      names.map((name) => [name, uploadCameraTexture(cameras[name])]),
    );
    updateXrHudTexture();
    const layouts = [
      { x: 0, y: 0.62, z: -2.9, width: 1.8, height: 1.35 },
      { x: -0.51, y: -0.48, z: -2.9, width: 0.95, height: 0.7125 },
      { x: 0.51, y: -0.48, z: -2.9, width: 0.95, height: 0.7125 },
    ];
    pose.views.forEach((view) => {
      const viewport = layer.getViewport(view);
      if (!viewport) return;
      gl.viewport(viewport.x, viewport.y, viewport.width, viewport.height);
      const projection = new Float32Array(view.projectionMatrix);
      const viewMatrix = new Float32Array(view.transform.inverse.matrix);
      const viewProjection = multiplyMatrices(projection, viewMatrix);

      names.forEach((name, index) => {
        const camera = cameras[name];
        const model = panelModel(
          layouts[index].x,
          layouts[index].y,
          layouts[index].z,
          layouts[index].width,
          layouts[index].height,
        );
        // A head-relative video monitor remains at eye level in either
        // local-floor or local reference space. This is a mono camera feed.
        const headRelativeModel = multiplyMatrices(new Float32Array(pose.transform.matrix), model);
        const mvp = multiplyMatrices(viewProjection, headRelativeModel);
        gl.uniformMatrix4fv(xrProgram.mvp, false, mvp);
        gl.uniform1f(xrProgram.hasTexture, textureReady[name] ? 1 : 0);
        // Uploading all cameras leaves the last texture bound. Select the
        // panel's texture on every draw, including cached frames in both eyes.
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, camera.texture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      });
      if (app.xr.hud && app.xr.hud.key) {
        const model = panelModel(0, -1.16, -2.9, 2.1, 0.394);
        const mvp = multiplyMatrices(viewProjection,
          multiplyMatrices(new Float32Array(pose.transform.matrix), model));
        gl.uniformMatrix4fv(xrProgram.mvp, false, mvp);
        gl.uniform1f(xrProgram.hasTexture, 1);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, app.xr.hud.texture);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      }
    });
  }

  function armReadinessHint() {
    if (!isWebSocketOpen()) return "控制连接未就绪";
    if (app.xr.session && !xrHasInputFocus(app.xr.session)) return "VR 未获得焦点：请关闭系统菜单或浏览器浮层";
    if (app.status.stop_unconfirmed) return "停止尚未确认；请先检查 Stop 反馈";
    if (app.xr.safetyTripped) return "安全停止：退出 VR 后重新进入";
    if (app.armRequestPending) return "启用请求已发送，等待 AGX 确认";
    if (app.armFailure) return `启用被拒绝：${app.armFailure}`;
    if (app.status.control_owner === "other") return "其他操作者占用控制权";
    if (app.status.metadata && app.status.metadata.allow_motion === false) return "AGX 未允许运动";
    if (app.xr.session && app.xr.session.inputSources.length === 0) return "浏览器未提供手柄输入源；用 Meta 菜单往返恢复";
    if (!cameraFramesFresh()) return "等待三路实时画面恢复；当前不可启用";
    const left = app.controllers.left, right = app.controllers.right;
    if (!left.tracked || !right.tracked) return "等待双手追踪；请把两只手柄放在头显前方";
    if (app.xr.gripActivationReady === false && (left.grip || right.grip)) return "先松开 Grip，再握住接管";
    if (left.trigger >= 0.1 || right.trigger >= 0.1) return "先松开食指扳机";
    if ([...left.thumbstick, ...right.thumbstick].some(value => Math.abs(value) >= 0.15)) return "请将摇杆归中";
    if (app.status.error?.includes("controller input delayed")) return controlErrorHint(app.status.error);
    return "握住对应 Grip 接管；松开保持；X / A 一按即停";
  }

  function mappingHint() {
    const status = app.status.mapping_status || {};
    const names = { left: "左", right: "右" };
    const joints = {shoulder_pan: "肩转向", shoulder_lift: "肩抬升", elbow_flex: "肘部", wrist_flex: "腕俯仰", wrist_roll: "腕旋转"};
    const limited = ["left", "right"].find(side => status[side]?.active && status[side]?.joint_limit_joints?.length);
    if (limited) {
      const affected = status[limited].joint_limit_joints.map(name => {
        const joint = name.replace(/^.*_arm_/, "").replace(/\.pos$/, "");
        return joints[joint] || joint;
      }).join("/");
      return `${names[limited]}${affected}到边界；换方向，其他轴仍可动`;
    }
    const workspace = ["left", "right"].find(side => status[side]?.active && status[side]?.workspace_limited);
    if (workspace) return `${names[workspace]}臂伸缩方向到工作范围边界；换方向，横转和手腕仍可动`;
    const pickup = ["left", "right"].filter(side => status[side]?.active && status[side]?.trigger_ready === false);
    if (pickup.length) return `${pickup.map(side => names[side]).join("/")}夹爪：先松开扳机，再扣下按所选方向动作`;
    const paced = ["left", "right"].some(side => status[side]?.active && status[side]?.speed_limited_joints?.length);
    if (paced) return "按设定速度跟随中；松开 Grip 保持当前位置";
    return "Grip 松开保持；重新握住重设参考；双臂模式摇杆不控制手臂";
  }

  function updateXrHudTexture() {
    const hud = app.xr.hud;
    if (!hud || !hud.context) return;
    const left = app.controllers.left, right = app.controllers.right;
    const armed = app.status.armed === true;
    const mode = app.status.control_mode === "drive" ? "行驶模式" : "机械臂模式";
    const lines = [
      `${mode} · ${armed ? "已启用" : "未启用"} · ${app.wsState === "open" ? "控制已连接" : "控制未连接"}`,
      `追踪 L:${left.tracked ? "OK" : "未就绪"} R:${right.tracked ? "OK" : "未就绪"} · 扳机方向 左:${app.gripperDirections.left === "open" ? "开" : "合"} 右:${app.gripperDirections.right === "open" ? "开" : "合"}`,
      armed ? (app.status.control_mode === "drive" ? "右 Grip + 右摇杆行驶；松开即停；X / A 停止" : "Grip 跟手＋横摇杆转底轴；扳机动作；Y/B 开合；X/A 停止") : armReadinessHint(),
      app.armFailure ? "先核查关节位置与标定；请勿硬掰机械臂或绕过限位"
        : app.status.error ? `最近提示：${controlErrorHint(app.status.error)}`
          : armed && app.status.control_mode !== "drive" ? mappingHint()
            : `VR:${app.xr.session?.visibilityState || "未知"} · 输入源:${app.xr.session?.inputSources?.length || 0} · 回传包:${app.inputPackets}`,
    ];
    const key = lines.join("\n");
    if (hud.key === key) return;
    const ctx = hud.context;
    ctx.fillStyle = "#101b21";
    ctx.fillRect(0, 0, 1024, 192);
    ctx.font = "30px sans-serif";
    ctx.textBaseline = "middle";
    lines.forEach((line, index) => {
      ctx.fillStyle = index === 0 ? (armed ? "#ffc96b" : "#a6ecc9") : "#f4f5f6";
      ctx.fillText(line, 18, 25 + index * 46, 988);
    });
    const gl = app.xr.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, hud.texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, hud.canvas);
    hud.key = key;
  }

  async function checkXrSupport() {
    if (!navigator.xr || typeof navigator.xr.isSessionSupported !== "function") {
      app.xr.supportChecked = true;
      app.xr.supported = false;
      updateImmersiveUi();
      return;
    }
    try {
      app.xr.supported = await navigator.xr.isSessionSupported("immersive-vr");
    } catch (error) {
      app.xr.supported = false;
      logEvent(`WebXR 检查失败: ${error.message || "未知错误"}`, "error");
    } finally {
      app.xr.supportChecked = true;
      updateImmersiveUi();
    }
  }

  async function enterXr() {
    if (!app.xr.supported || app.xr.active) return;
    if (!app.authenticated || !isWebSocketOpen()) {
      showToast("进入 VR 前需要完成认证并连接 WebSocket。", "error");
      return;
    }
    try {
      const gl = initWebGL();
      if (typeof gl.makeXRCompatible === "function") await gl.makeXRCompatible();
      const session = await navigator.xr.requestSession("immersive-vr", {
        optionalFeatures: ["local-floor", "bounded-floor"],
      });
      const layer = new XRWebGLLayer(session, gl);
      session.updateRenderState({ baseLayer: layer });
      let referenceSpace;
      try {
        referenceSpace = await session.requestReferenceSpace("local-floor");
      } catch (_error) {
        referenceSpace = await session.requestReferenceSpace("local");
      }
      app.xr.session = session;
      app.xr.layer = layer;
      app.xr.referenceSpace = referenceSpace;
      app.xr.active = true;
      app.xr.startedAt = performance.now();
      app.xr.controllersReady = false;
      app.xr.viewerTracked = false;
      app.xr.gripActivationReady = false;
      app.gripperDirections = {left: "close", right: "close"};
      app.gripperToggleWasDown = {left: false, right: false};
      app.xr.stopButtonWasDown = false;
      app.xr.safetyTripped = false;
      app.xr.stopSent = false;
      app.xr.ending = false;
      app.controllerClient = false;
      session.addEventListener("end", () => handleXrSessionEnd(session));
      session.addEventListener("visibilitychange", () => handleXrVisibilityChange(session));
      session.requestAnimationFrame(onXrFrame);
      app.controllers = { left: makeController(), right: makeController() };
      sendInputFrame(performance.now());
      logEvent("已进入 immersive-vr。先松开再握 Grip 接管；X/A 停止。", "safe");
      updateAllUi();
    } catch (error) {
      logEvent(`无法进入 immersive-vr: ${error.message || "未知错误"}`, "error");
      showToast(`无法进入 VR: ${error.message || "未知错误"}`, "error");
      updateAllUi();
    }
  }

  function handleXrSessionEnd(session) {
    if (app.xr.session !== session) return;
    const shouldStop = !app.xr.stopSent && (app.status.control_owner === "self" || app.controllerClient);
    if (shouldStop) sendStopPacket("WebXR 会话结束", true);
    app.xr.session = null;
    app.xr.layer = null;
    app.xr.referenceSpace = null;
    app.xr.active = false;
    app.xr.controllersReady = false;
    app.xr.gripActivationReady = false;
    app.xr.stopButtonWasDown = false;
    app.xr.ending = false;
    app.controllerClient = false;
    logEvent("WebXR 会话已结束，控制输入已停止。", "safe");
    updateAllUi();
  }

  function failSafeStop(reason, options = {}) {
    const endSession = options.endSession !== false;
    const shouldSend = isActualControllerClient() || (app.xr.active && app.xr.controllersReady);
    if (shouldSend && !app.xr.stopSent) sendStopPacket(reason, true);
    app.armRequestPending = false;
    app.xr.safetyTripped = true;
    app.xr.gripActivationReady = false;
    updateAllUi();
    if (endSession && app.xr.session && !app.xr.ending) {
      const session = app.xr.session;
      app.xr.ending = true;
      void session.end().catch(() => {
        app.xr.ending = false;
      });
    }
    logEvent(`安全停止: ${reason}`, "error");
  }

  function readGamepadButton(gamepad, index) {
    return Boolean(gamepad && gamepad.buttons && gamepad.buttons[index] && gamepad.buttons[index].pressed);
  }

  function readGamepadButtons(gamepad) {
    const count = Math.max(7, gamepad && gamepad.buttons ? gamepad.buttons.length : 0);
    return Array.from({ length: count }, (_, index) => readGamepadButton(gamepad, index));
  }

  function readController(source, frame, referenceSpace) {
    const controller = makeController();
    const gamepad = source.gamepad;
    const buttons = readGamepadButtons(gamepad);
    controller.buttons = buttons;
    controller.grip = Boolean(buttons[1]);
    const triggerButton = gamepad && gamepad.buttons ? gamepad.buttons[0] : null;
    controller.trigger = clamp(
      triggerButton && Number.isFinite(triggerButton.value)
        ? triggerButton.value
        : triggerButton && triggerButton.pressed
          ? 1
          : 0,
      0,
      1,
    );
    const axes = gamepad && gamepad.axes && typeof gamepad.axes.length === "number" ? gamepad.axes : [];
    controller.thumbstick = [clamp(axes[2], -1, 1), clamp(axes[3], -1, 1)];
    if (!source.gripSpace) return controller;
    let pose;
    try {
      pose = frame.getPose(source.gripSpace, referenceSpace);
    } catch (_error) {
      pose = null;
    }
    if (!pose || !pose.transform) return controller;
    // WebXR uses DOMPointReadOnly objects, not array-like values. Falling
    // back to zeros here hid every real hand translation and rotation.
    const p = pose.transform.position, q = pose.transform.orientation;
    if (!p || !q) return controller;
    const position = [p.x, p.y, p.z];
    const rotation = [q.x, q.y, q.z, q.w];
    const norm = Math.hypot(...rotation);
    if (!position.every(Number.isFinite) || !rotation.every(Number.isFinite) || norm <= 0.8 || norm >= 1.2) return controller;
    const orientation = rotation.map(value => value / norm);
    controller.position = position;
    controller.orientation = orientation;
    controller.tracked = position.every(Number.isFinite) && orientation.every(Number.isFinite);
    return controller;
  }

  function readControllers(frame, referenceSpace, session) {
    const result = { left: makeController(), right: makeController() };
    for (const source of session.inputSources) {
      if (source.handedness !== "left" && source.handedness !== "right") continue;
      // Hand/gaze pointers may coexist with real controllers. Never let a
      // later non-gamepad source overwrite a tracked Touch controller.
      if (!source.gamepad || source.gamepad.connected === false || source.hand) continue;
      if (source.targetRayMode && source.targetRayMode !== "tracked-pointer") continue;
      const controller = readController(source, frame, referenceSpace);
      if (!result[source.handedness].tracked) result[source.handedness] = controller;
    }
    return result;
  }

  function updateQuestButtonActions() {
    const left = app.controllers.left;
    const right = app.controllers.right;
    const stopDown = Boolean(left.buttons[XR_STOP_BUTTON_INDEX] || right.buttons[XR_STOP_BUTTON_INDEX]);
    if (stopDown && !app.xr.stopButtonWasDown) {
      manualStop("Quest X/A");
      app.xr.gripActivationReady = false;
    }
    app.xr.stopButtonWasDown = stopDown;

    for (const hand of ["left", "right"]) {
      const down = Boolean(app.controllers[hand].buttons[XR_GRIPPER_TOGGLE_BUTTON_INDEX]);
      if (down && !app.gripperToggleWasDown[hand]) {
        app.gripperDirections[hand] = app.gripperDirections[hand] === "close" ? "open" : "close";
      }
      app.gripperToggleWasDown[hand] = down;
    }
    if (stopDown) return;
    if (!left.grip && !right.grip) {
      app.xr.gripActivationReady = true;
    } else if (app.xr.gripActivationReady) {
      app.xr.gripActivationReady = false;
      if (!app.status.armed && !app.armRequestPending) requestArm("Grip 接管", "grip");
    }
  }

  function sanitizeController(controller, hand) {
    const safeStick = [clamp(controller.thumbstick[0], -1, 1), clamp(controller.thumbstick[1], -1, 1)];
    return {
      position: finiteVector(controller.position, 3, [0, 0, 0]),
      orientation: normalizedQuaternion(controller.orientation),
      tracked: controller.tracked === true,
      grip: controller.grip === true,
      trigger: clamp(controller.trigger, 0, 1),
      thumbstick: safeStick,
      buttons: Array.isArray(controller.buttons) ? controller.buttons.map(Boolean) : [],
      gripper_direction: app.gripperDirections[hand],
    };
  }

  function sendInputFrame(now, force = false) {
    if (!isWebSocketOpen()) return false;
    if (!force && now - app.lastInputSentAt < INPUT_INTERVAL_MS) return false;
    if (app.ws.bufferedAmount > MAX_CONTROL_BUFFER_BYTES) {
      failSafeStop("WebSocket 缓冲过高", { endSession: true });
      return false;
    }
    const packet = {
      type: "input",
      seq: app.inputSeq,
      timestamp_ms: finite(performance.now(), 0),
      xr: xrInputDiagnostics(),
      controllers: {
        left: sanitizeController(app.controllers.left, "left"),
        right: sanitizeController(app.controllers.right, "right"),
      },
    };
    try {
      app.ws.send(JSON.stringify(packet));
    } catch (error) {
      failSafeStop(`输入发送失败: ${error.message || "未知错误"}`, { endSession: true });
      return false;
    }
    app.inputSeq += 1;
    app.lastInputSentAt = now;
    app.inputPackets += 1;
    app.inputSocket = app.ws;
    app.controllerClient = true;
    return true;
  }

  function xrHasInputFocus(session = app.xr.session) {
    // Immersive visibility is independent of the owning HTML document.
    // Never treat visible-blurred (system UI owns input) as interactive.
    return session ? session.visibilityState === "visible" : !document.hidden;
  }

  function hasRobotControlIntent() {
    return app.armRequestPending || app.status.control_owner === "self" ||
      (app.status.armed && app.controllerClient && app.status.control_owner !== "other");
  }

  function pauseXrInput(reason) {
    app.xr.gripActivationReady = false;
    app.xr.controllersReady = false;
    if (hasRobotControlIntent() && !app.xr.safetyTripped) {
      // Stop immediately but keep rendering diagnostics. Restoring tracking
      // does not clear the safety latch or re-arm the robot.
      failSafeStop(reason, { endSession: false });
    }
  }

  function handleXrVisibilityChange(session) {
    if (!app.xr.active || app.xr.session !== session) return;
    if (!xrHasInputFocus(session)) {
      app.controllers = { left: makeController(), right: makeController() };
      pauseXrInput("VR 会话失去输入焦点");
      sendInputFrame(performance.now());
    }
    updateAllUi();
  }

  function xrInputDiagnostics() {
    const session = app.xr.session;
    return {
      active: app.xr.active,
      visibility: session?.visibilityState || "none",
      document_hidden: document.hidden,
      viewer_tracked: app.xr.viewerTracked,
      safety_tripped: app.xr.safetyTripped,
      sources: Array.from(session?.inputSources || []).slice(0, 6).map(source => ({
        hand: source.handedness,
        gamepad: Boolean(source.gamepad),
        grip_space: Boolean(source.gripSpace),
        profiles: Array.from(source.profiles || []).slice(0, 4).map(value => String(value).slice(0, 80)),
      })),
    };
  }

  function onXrFrame(time, frame) {
    const session = app.xr.session;
    if (!app.xr.active || !session || frame.session !== session) return;
    session.requestAnimationFrame(onXrFrame);
    let pose;
    try {
      pose = frame.getViewerPose(app.xr.referenceSpace);
    } catch (_error) {
      pose = null;
    }
    app.xr.viewerTracked = Boolean(pose);
    const focused = xrHasInputFocus(session);
    app.controllers = focused && pose
      ? readControllers(frame, app.xr.referenceSpace, session)
      : { left: makeController(), right: makeController() };
    const ready = focused && Boolean(pose) && app.controllers.left.tracked && app.controllers.right.tracked;
    const now = performance.now();
    if (!ready) {
      pauseXrInput(!focused ? "VR 会话失去输入焦点" : !pose ? "头显追踪丢失" : "控制器追踪丢失");
    } else {
      app.xr.controllersReady = true;
    }
    // Report even incomplete startup tracking. Otherwise the server cannot
    // distinguish an untracked hand from a disconnected browser. Input alone
    // never arms; the server rejects untracked frames while controlling.
    // A new Grip edge must reach the server before its activation request,
    // even between the normal 30 Hz input sends. Otherwise the server sees
    // the preceding neutral frame and rejects a valid brief squeeze.
    const activatingGrip = ready && !app.xr.safetyTripped && app.xr.gripActivationReady &&
      (app.controllers.left.grip || app.controllers.right.grip) &&
      !app.status.armed && !app.armRequestPending;
    sendInputFrame(now, activatingGrip);
    if (ready && !app.xr.safetyTripped) updateQuestButtonActions(now);
    if (pose) drawXrScene(pose, app.xr.layer);
    updateImmersiveUi();
    updateInputUi();
    updateArmUi();
  }

  function bindSafetyEvents() {
    document.addEventListener("visibilitychange", () => {
      if (app.xr.active && app.xr.session) {
        handleXrVisibilityChange(app.xr.session);
      } else if (document.hidden && isActualControllerClient()) {
        failSafeStop("页面隐藏", { endSession: false });
      }
    });
    window.addEventListener("pagehide", () => {
      if (isActualControllerClient()) {
        failSafeStop("控制端离开页面", { endSession: false });
      }
    });
    window.addEventListener("beforeunload", () => {
      if (isActualControllerClient()) {
        sendStopPacket("控制端页面卸载", true);
      }
    });
  }

  function bindControls() {
    dom.authForm.addEventListener("submit", (event) => {
      event.preventDefault();
      void authenticate(dom.tokenInput.value);
    });
    dom.armButton.addEventListener("click", () => requestArm("桌面 Arm"));
    dom.controlModeSelect.addEventListener("change", () => requestControlMode(dom.controlModeSelect.value));
    dom.stopButton.addEventListener("click", () => manualStop("操作员 Stop"));
    dom.immersiveButton.addEventListener("click", () => void enterXr());
    dom.videoPeerRetry.addEventListener("click", retryVideoPeer);
    bindCameraControls();
    bindRecordingControls();
    bindSafetyEvents();
  }

  async function boot() {
    initDom();
    initTheme();
    bindControls();
    setMainCamera("front");
    updateAllUi();
    void checkXrSupport();
    scheduleStatusPolling();
    const token = readFragmentToken();
    if (token) {
      await authenticate(token);
    } else {
      await refreshStatus();
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => void boot(), { once: true });
  } else {
    void boot();
  }
})();
