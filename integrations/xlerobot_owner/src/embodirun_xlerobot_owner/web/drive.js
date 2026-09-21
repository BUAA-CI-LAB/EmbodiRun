/* Wired keyboard driving. UI tests exercise the same key state machine. */
"use strict";
class DriveKeys {
  constructor() { this.active = false; this.keys = new Set(); }
  enable() { this.keys.clear(); this.active = true; }
  pause() { this.keys.clear(); }
  stop() { this.keys.clear(); this.active = false; }
  key(code, down, repeat = false) {
    if (!["KeyW", "KeyA", "KeyS", "KeyD"].includes(code)) return false;
    if (!down) this.keys.delete(code);
    else if (this.active && !repeat) this.keys.add(code);
    return true;
  }
}
if (typeof module !== "undefined") module.exports = { DriveKeys };
if (typeof document !== "undefined") (() => {
  const $ = id => document.getElementById(id);
  const names = ["front", "left_wrist", "right_wrist"];
  const input = new DriveKeys();
  const frames = Object.fromEntries(names.map(n => [n, 0]));
  let ws, peer, state, statusAt = 0, seq = 0, pending = false, connecting = false;
  let generation = 0, requestedMode = false, localError = "", lastStamp = -1;
  let leaderPending = "", leaderLocalError = "";
  let speedDirty = false, speedPending = false, speedError = "";
  const speedIds = ["drive-linear", "drive-angular"];
  const focused = () => !document.hidden && document.hasFocus();
  const videoReady = () => names.every(n => performance.now() - frames[n] < 800 && frames[n] > 0);
  const send = data => {
    if (ws?.readyState !== WebSocket.OPEN || ws.bufferedAmount > 8192) return false;
    ws.send(JSON.stringify(data)); return true;
  };
  const pulse = () => {
    lastStamp = Math.max(performance.now(), lastStamp + 0.001);
    return send({ type: "keyboard_input", seq: seq++, timestamp_ms: lastStamp,
      keys: [...input.keys], focused: focused(), video_ready: videoReady() });
  };
  function pause() {
    input.pause();
    localError = "输入短暂延迟，底盘已暂停；连接保留，松开后重新按 WASD 即可。";
  }
  function stop(reason, notify = true) {
    const owned = input.active || pending;
    input.stop(); pending = false; localError = reason;
    if (notify && owned) { pulse(); send({ type: "stop" }); }
    paint();
  }
  function ready() {
    return ws?.readyState === WebSocket.OPEN && state?.connected &&
      !state?.armed && !state?.control_owner && !state?.stop_unconfirmed &&
      !state?.keyboard_drive_unavailable_reason && state?.control_mode === "drive" &&
      performance.now() - statusAt < 500 && videoReady() && focused();
  }
  function paint() {
    const leader = state?.leader_follow;
    const leaderActive = ["starting", "aligning", "following", "stopping"].includes(leader?.state);
    $("connection").textContent = state?.connected && performance.now() - statusAt < 500 ? "AGX 已连接" : "等待 AGX";
    $("mode").textContent = input.active ? "键盘已启用" : pending ? "请求控制中" : "未启用";
    $("parallel").textContent = state?.parallel_control && state?.control_scope === "base"
      ? `独立通道已连接 · 双臂：${state.control_state?.arms ? "跟随中" : "未启用"} · 底盘：${state.control_state?.base ? "控制中" : "未启用"}`
      : "等待支持双臂 / 底盘并行控制的 AGX 服务。";
    $("enable").disabled = input.active || pending || !ready();
    const faults = state?.observation_errors || [];
    const hardwareFault = faults.length ? `硬件反馈异常（${faults.length} 项）：${faults[0]}` : "";
    $("message").textContent = hardwareFault || localError || state?.keyboard_drive_unavailable_reason ||
      state?.error || (state?.control_owner === "other" ? "另一个操作端正在控制，请先在该端停止。" :
      !videoReady() ? "等待三路实时视频；若画面未恢复，点击重新连接。" :
      input.active ? "按住 WASD 行驶，松键停止。Space / Esc 退出控制。" : "三路画面已就绪，点击启用后用 WASD 开车。");
    $("keys").textContent = [...input.keys].map(k => k.slice(3)).join(" + ") || "—";
    document.querySelectorAll("kbd[data-key]").forEach(k => k.classList.toggle("down", input.keys.has(k.dataset.key)));
    const linear = state?.state?.["x.vel"], angular = state?.state?.["theta.vel"];
    $("linear").textContent = Number.isFinite(linear) ? `${linear.toFixed(3)} m/s` : "—";
    $("angular").textContent = Number.isFinite(angular) ? `${angular.toFixed(1)} °/s` : "—";
    $("limits").textContent = state?.drive_limits ? `${state.drive_limits.linear_m_s} m/s · ${state.drive_limits.angular_deg_s} °/s` : "—";
    const canSetSpeed = state?.connected && ws?.readyState === WebSocket.OPEN &&
      performance.now() - statusAt < 500 && !state?.stop_unconfirmed &&
      state?.control_mode === "drive" && !state?.keyboard_drive_unavailable_reason &&
      state?.control_owner !== "other" && !input.keys.size && !pending && !speedPending;
    if (state?.drive_limits && state?.drive_speed) {
      ["linear_m_s", "angular_deg_s"].forEach((key, i) => {
        const slider = $(speedIds[i]);
        slider.max = String(state.drive_limits[key]);
        if (!speedDirty && !speedPending) slider.value = String(state.drive_speed[key]);
      });
    }
    speedIds.forEach(id => { $(id).disabled = !canSetSpeed; });
    const targetLinear = Number($("drive-linear").value), targetAngular = Number($("drive-angular").value);
    $("drive-linear-value").textContent = `${targetLinear.toFixed(2)} m/s · ${Math.round(targetLinear * 100)} cm/s`;
    $("drive-angular-value").textContent = `${targetAngular}°/s`;
    $("speed-apply").disabled = !canSetSpeed || !speedDirty;
    $("speed-apply").textContent = speedPending ? "正在应用…" : "应用速度";
    $("speed-message").textContent = speedError || (speedPending ? "等待服务端确认…" :
      speedDirty ? "尚未应用：松开 WASD 后点击应用速度。" :
      state?.drive_speed ? `当前生效：${state.drive_speed.linear_m_s.toFixed(2)} m/s · ${state.drive_speed.angular_deg_s}°/s` :
      "等待速度配置。");
    $("age").textContent = Number.isFinite(state?.stats?.observation_age_ms) ? `${state.stats.observation_age_ms} ms` : "—";
    const leaderLabels = {
      unavailable: "双主臂入口未配置。", stopped: "双主臂未启动。",
      starting: "正在连接并检查两只主臂；尚未启臂。",
      aligning: "正在平滑完成主从臂初始对齐，请保持主臂不动。",
      following: "同步完成；移动两只主臂，从臂会直接跟随。",
      stopping: "正在停止并等待 AGX 确认。", failed: "双主臂跟随启动或运行失败。"
    };
    $("leader-message").textContent = leaderLocalError || leader?.error ||
      (leader?.state === "stopped" ? leader?.unavailable_reason : "") ||
      leaderLabels[leader?.state] || "正在检查本机主臂与 AGX 双臂通道。";
    $("leader-start").disabled = leaderPending !== "" || leaderActive ||
      !leader?.available || Boolean(leader?.unavailable_reason);
    $("leader-stop").disabled = leaderPending !== "" || !leaderActive;
    $("leader-output").textContent = leader?.output?.length ? leader.output.join("\n") : "尚未启动";
    names.forEach(n => $(n + "-health").textContent = frames[n] && performance.now() - frames[n] < 800 ? "实时" : "等待 / 停更");
  }
  async function json(url, body) {
    const r = await fetch(url, { method: body ? "POST" : "GET", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: body ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(6000) });
    const value = await r.json();
    if (r.status === 401) $("auth").hidden = false;
    if (!r.ok) throw new Error(value.error || `HTTP ${r.status}`);
    return value;
  }
  async function video(myGeneration) {
    const current = new RTCPeerConnection({ iceServers: [], bundlePolicy: "max-bundle" });
    peer = current;
    const receivers = names.map(() => current.addTransceiver("video", { direction: "recvonly" }));
    current.addEventListener("track", event => {
      if (myGeneration !== generation) return;
      const name = names[receivers.indexOf(event.transceiver)];
      if (!name) return stop("相机轨道无法匹配");
      const element = $(name);
      element.srcObject = new MediaStream([event.track]);
      element.play().catch(() => stop("相机播放失败，请重新连接"));
      const onFrame = () => {
        if (myGeneration !== generation) return;
        frames[name] = performance.now();
        element.requestVideoFrameCallback(onFrame);
      };
      if (typeof element.requestVideoFrameCallback !== "function") {
        stop("请使用支持视频帧回调的新版本 Chrome / Edge / Safari"); return;
      }
      element.requestVideoFrameCallback(onFrame);
    });
    current.addEventListener("connectionstatechange", () => {
      if (myGeneration === generation && ["failed", "disconnected", "closed"].includes(current.connectionState)) {
        stop("视频连接中断，请重新连接");
      }
    });
    await current.setLocalDescription(await current.createOffer());
    if (current.iceGatheringState !== "complete") await new Promise((resolve, reject) => {
      const changed = () => { if (current.iceGatheringState === "complete") { cleanup(); resolve(); } };
      const cleanup = () => { clearTimeout(timer); current.removeEventListener("icegatheringstatechange", changed); };
      const timer = setTimeout(() => { cleanup(); reject(new Error("视频协商超时")); }, 5000);
      current.addEventListener("icegatheringstatechange", changed);
      changed();
    });
    if (myGeneration !== generation) return;
    const answer = await json("/api/webrtc/offer", { type: "offer", sdp: current.localDescription.sdp });
    if (answer.cameras?.join() !== names.join()) throw new Error("相机顺序不一致");
    if (myGeneration === generation) await current.setRemoteDescription({ type: answer.type, sdp: answer.sdp });
  }
  async function connect() {
    if (connecting) return;
    stop("正在连接", true); connecting = true;
    const currentGeneration = ++generation;
    ws?.close(); peer?.close(); peer = null;
    names.forEach(n => { frames[n] = 0; $(n).srcObject = null; });
    state = null; statusAt = 0; seq = 0; requestedMode = false;
    speedDirty = false; speedPending = false; speedError = "";
    try {
      await json("/api/status");
      ws = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
      ws.addEventListener("open", () => { if (generation === currentGeneration) { localError = ""; pulse(); } });
      ws.addEventListener("message", event => {
        if (generation !== currentGeneration) return;
        const data = JSON.parse(event.data);
        if (data.type === "drive_speed_result") {
          speedPending = false;
          speedError = data.ok ? "" : data.error;
          if (data.ok) { speedDirty = false; if (state) state.drive_speed = data.speed; }
          paint(); return;
        }
        if (data.type === "leader_error") {
          leaderPending = ""; leaderLocalError = data.error; paint(); return;
        }
        if (data.type === "error") { stop(data.error); return; }
        if (data.type !== "status") return;
        state = data; statusAt = performance.now();
        const leaderState = data.leader_follow?.state;
        if ((leaderPending === "start" && ["starting", "aligning", "following", "failed"].includes(leaderState)) ||
            (leaderPending === "stop" && !["starting", "aligning", "following", "stopping"].includes(leaderState))) {
          leaderPending = "";
        }
        if (input.active && data.keyboard_paused) { pause(); pulse(); }
        else if (input.active && localError.startsWith("输入短暂延迟")) localError = "";
        if (!requestedMode && !data.armed && !data.control_owner && !data.keyboard_drive_unavailable_reason) {
          requestedMode = true; send({ type: "set_control_mode", mode: "drive" });
        }
        if (pending && data.armed && data.control_owner === "self") {
          pending = false; input.enable(); localError = "";
          if (!focused() || !videoReady()) stop("窗口或画面未就绪");
        } else if (input.active && (!data.armed || data.control_owner !== "self")) {
          stop(data.error || "控制已停止，请重新启用", false);
        }
        paint();
      });
      ws.addEventListener("close", () => { if (generation === currentGeneration) { speedPending = false; stop("控制连接已断开，请重新连接", false); } });
      ws.addEventListener("error", () => { if (generation === currentGeneration) stop("控制连接异常"); });
      await video(currentGeneration);
    } catch (e) { stop(e.message); }
    finally { connecting = false; paint(); }
  }
  $("enable").addEventListener("click", () => {
    if (!ready() || pending) return;
    input.stop(); localError = ""; pending = true;
    if (!pulse() || !send({ type: "arm", activation: "keyboard" })) stop("发送失败，请重新连接");
    $("enable").blur(); paint();
  });
  speedIds.forEach(id => {
    $(id).addEventListener("input", () => { speedDirty = true; speedError = ""; paint(); });
    $(id).addEventListener("change", () => $(id).blur());
  });
  $("speed-apply").addEventListener("click", () => {
    if ($("speed-apply").disabled) return;
    speedPending = true; speedError = "";
    if (!pulse() || !send({ type: "set_drive_speed", linear_m_s: Number($("drive-linear").value),
      angular_deg_s: Number($("drive-angular").value) })) {
      speedPending = false; speedError = "发送失败，速度未改变；请重新连接。";
    }
    $("speed-apply").blur(); paint();
  });
  $("leader-start").addEventListener("click", () => {
    if ($("leader-start").disabled) return;
    leaderLocalError = ""; leaderPending = "start";
    if (!send({ type: "leader_start" })) {
      leaderPending = ""; leaderLocalError = "发送失败，请重新连接";
    }
    $("leader-start").blur(); paint();
  });
  $("leader-stop").addEventListener("click", () => {
    if ($("leader-stop").disabled) return;
    leaderLocalError = ""; leaderPending = "stop";
    if (!send({ type: "leader_stop" })) {
      leaderPending = ""; leaderLocalError = "发送失败，请重新连接";
    }
    $("leader-stop").blur(); paint();
  });
  $("stop").addEventListener("click", () => { stop("已请求停止"); send({ type: "stop" }); });
  $("stop-all").addEventListener("click", () => {
    stop("已请求整车停止", false); leaderPending = "stop"; leaderLocalError = "";
    send({ type: "stop_all" });
  });
  $("reconnect").addEventListener("click", connect);
  $("auth").addEventListener("submit", async event => {
    event.preventDefault();
    try { await json("/api/session", { token: $("token").value.trim() }); $("token").value = ""; $("auth").hidden = true; connect(); }
    catch (e) { stop(e.message); }
  });
  window.addEventListener("keydown", event => {
    if (["Space", "Escape"].includes(event.code)) { event.preventDefault(); stop("键盘停止"); send({ type: "stop" }); return; }
    if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName) || event.target.isContentEditable) return;
    if (event.ctrlKey || event.metaKey || event.altKey) { if (input.active || pending) stop("快捷键切换，已停止"); return; }
    if (input.key(event.code, true, event.repeat)) { event.preventDefault(); pulse(); paint(); }
  });
  window.addEventListener("keyup", event => { if (input.key(event.code, false)) { pulse(); paint(); } });
  window.addEventListener("blur", () => stop("窗口失焦，已停止；返回后重新启用"));
  document.addEventListener("visibilitychange", () => { if (document.hidden) stop("页面隐藏，已停止"); });
  document.addEventListener("focusin", event => {
    if (speedIds.includes(event.target.id)) return;
    if (event.target.matches("input,textarea,select,[contenteditable]")) stop("进入文字输入，已停止");
  });
  window.addEventListener("pagehide", () => { stop("页面关闭"); ws?.close(); peer?.close(); });
  setInterval(() => {
    if ((input.active || pending) && (!focused() || !videoReady() || performance.now() - statusAt > 2000)) stop("窗口、画面或控制状态过期，已停止");
    else if (input.active && performance.now() - statusAt > 500) pause();
    if (ws?.readyState === WebSocket.OPEN && !pulse() && (input.active || pending)) stop("发送队列拥堵，已停止");
    paint();
  }, 50);
  connect();
})();
