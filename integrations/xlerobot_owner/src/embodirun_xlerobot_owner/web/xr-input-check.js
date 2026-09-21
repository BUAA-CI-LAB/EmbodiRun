/* Independent WebXR input probe. Deliberately no video, robot Arm or command. */
(() => {
  "use strict";
  const button = document.getElementById("start-check");
  const nativeButton = document.getElementById("native-check");
  const output = document.getElementById("check-status");
  const canvas = document.getElementById("check-canvas");
  let session = null, socket = null, seq = 0, sentAt = 0, starting = false;
  const canOffer = typeof navigator.xr?.offerSession === "function";
  nativeButton.disabled = !canOffer;

  const empty = () => ({position: [0, 0, 0], orientation: [0, 0, 0, 1],
    tracked: false, grip: false, trigger: 0, thumbstick: [0, 0], buttons: []});

  function rawControllers(frame, reference, current) {
    const result = {left: empty(), right: empty()};
    // Original working path: enumerate the native array, no profile filters,
    // no readiness latch, no mapping or video dependency.
    for (const source of current.inputSources) {
      if (!(source.handedness in result)) continue;
      const c = empty(), gamepad = source.gamepad;
      c.buttons = Array.from(gamepad?.buttons || []).map(b => Boolean(b.pressed));
      c.grip = Boolean(c.buttons[1]);
      c.trigger = gamepad?.buttons?.[0]?.value || 0;
      c.thumbstick = [gamepad?.axes?.[2] || 0, gamepad?.axes?.[3] || 0];
      const pose = source.gripSpace ? frame.getPose(source.gripSpace, reference) : null;
      if (pose) {
        const p = pose.transform.position, q = pose.transform.orientation;
        c.position = [p.x, p.y, p.z];
        c.orientation = [q.x, q.y, q.z, q.w];
        c.tracked = [...c.position, ...c.orientation].every(Number.isFinite);
      }
      result[source.handedness] = c;
    }
    return result;
  }

  function drawStatus(gl, program, texture, textCanvas, lines, pose, layer) {
    const ctx = textCanvas.getContext("2d");
    ctx.fillStyle = "#102029";
    ctx.fillRect(0, 0, textCanvas.width, textCanvas.height);
    ctx.fillStyle = "#dcfff0";
    ctx.font = "32px sans-serif";
    lines.forEach((line, i) => ctx.fillText(line, 24, 52 + i * 54, 976));
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, textCanvas);
    gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
    gl.clearColor(0.02, 0.04, 0.05, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(program);
    for (const view of pose.views) {
      const vp = layer.getViewport(view);
      gl.viewport(vp.x, vp.y, vp.width, vp.height);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
  }

  async function start(entry = "request") {
    if (session || starting) return;
    starting = true;
    button.disabled = true;
    nativeButton.disabled = true;
    try {
      if (!navigator.xr) throw new Error("此浏览器没有 WebXR");
      const gl = canvas.getContext("webgl", {xrCompatible: true, alpha: false});
      if (!gl) throw new Error("WebGL 不可用");
      await gl.makeXRCompatible();
      // Same request parameters as the observed working 02:20 client.
      const options = {optionalFeatures: ["local-floor", "bounded-floor"]};
      if (entry === "offer") {
        if (!canOffer) throw new Error("此浏览器不支持原生 VR 入口");
        output.textContent = "请点击浏览器工具栏的 VR 图标，进入只读检测。此入口仍需实机对照，并非已确认的修复。";
        session = await navigator.xr.offerSession("immersive-vr", options);
      } else {
        session = await navigator.xr.requestSession("immersive-vr", options);
      }
      const current = session;
      const layer = new XRWebGLLayer(current, gl);
      current.updateRenderState({baseLayer: layer});
      let reference;
      try { reference = await current.requestReferenceSpace("local-floor"); }
      catch { reference = await current.requestReferenceSpace("local"); }
      const program = gl.createProgram();
      for (const [kind, code] of [
        [gl.VERTEX_SHADER, "attribute vec2 p; varying vec2 uv; void main(){uv=p*0.5+0.5; gl_Position=vec4(p*0.85,0.0,1.0);}"],
        [gl.FRAGMENT_SHADER, "precision mediump float; varying vec2 uv; uniform sampler2D tex; void main(){gl_FragColor=texture2D(tex,uv);}"],
      ]) {
        const shader = gl.createShader(kind);
        gl.shaderSource(shader, code); gl.compileShader(shader);
        if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
        gl.attachShader(program, shader);
      }
      gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
      gl.useProgram(program);
      gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW);
      const position = gl.getAttribLocation(program, "p");
      gl.enableVertexAttribArray(position);
      gl.vertexAttribPointer(position, 2, gl.FLOAT, false, 0, 0);
      const texture = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      const textCanvas = document.createElement("canvas");
      textCanvas.width = 1024; textCanvas.height = 512;
      let changes = 0, selected = 0, frameCount = 0;
      current.addEventListener("inputsourceschange", () => { changes++; });
      current.addEventListener("selectstart", () => { selected++; });
      current.addEventListener("end", () => {
        if (session !== current) return;
        session = null; button.disabled = false; nativeButton.disabled = !canOffer;
        if (socket) socket.close();
        socket = null;
      });
      // This socket only sends diagnostic input packets. It never requests
      // ownership; the server cannot arm or move from input packets alone.
      socket = new WebSocket(`wss://${location.host}/ws`);
      function frameLoop(_time, frame) {
        if (session !== current) return;
        current.requestAnimationFrame(frameLoop);
        const pose = frame.getViewerPose(reference);
        const controllers = rawControllers(frame, reference, current);
        const now = performance.now();
        const xr = {
          active: true, visibility: current.visibilityState, document_hidden: document.hidden,
          viewer_tracked: Boolean(pose), safety_tripped: false,
          sources: Array.from(current.inputSources).map(s => ({hand: s.handedness,
            gamepad: Boolean(s.gamepad), grip_space: Boolean(s.gripSpace), profiles: Array.from(s.profiles)})),
        };
        if (socket?.readyState === WebSocket.OPEN && now - sentAt >= 50) {
          socket.send(JSON.stringify({type: "input", seq: seq++, timestamp_ms: now, controllers, xr}));
          sentAt = now;
        }
        frameCount++;
        const lines = [
          `${entry === "offer" ? "浏览器原生入口" : "网页入口"} · 只读 · 无机器人动作`,
          `VR:${current.visibilityState}  原生输入源:${current.inputSources.length}  帧:${frameCount}`,
          `左:${controllers.left.tracked ? "有追踪" : "无追踪"}  右:${controllers.right.tracked ? "有追踪" : "无追踪"}`,
          `X:${controllers.left.buttons[4] ? "按下" : "松开"}  A:${controllers.right.buttons[4] ? "按下" : "松开"}`,
          `设备变化事件:${changes}  扳机事件:${selected}`,
          `回传:${socket?.readyState === WebSocket.OPEN ? "已连接" : "未连接"}`,
        ];
        output.textContent = lines.join("\n") + "\n" + navigator.userAgent;
        if (pose) drawStatus(gl, program, texture, textCanvas, lines, pose, layer);
      }
      current.requestAnimationFrame(frameLoop);
    } catch (error) {
      output.textContent = String(error);
      if (session) await session.end().catch(() => {});
      session = null; button.disabled = false;
      nativeButton.disabled = !canOffer;
    } finally {
      starting = false;
    }
  }
  button.addEventListener("click", () => void start());
  nativeButton.addEventListener("click", () => void start("offer"));
})();
