import json
from pathlib import Path

import pytest

from .test_web_controller_input import _run_node

PROBE = Path(__file__).parents[2] / "integrations/xlerobot_owner/src/embodirun_xlerobot_owner/web/xr-input-check.js"


@pytest.mark.parametrize("entry", ["request", "offer"])
def test_input_probe_uses_native_sources_and_never_sends_arm_or_robot_actions(entry):
    source = PROBE.read_text()
    _run_node(
        """
    const assert = require('node:assert/strict');
    let click, nativeClick, raf, now=100;
    const entry = ENTRY_VALUE;
    const sent=[];
    const gl = new Proxy({}, {get(_target, key) {
      if (/^[A-Z_0-9]+$/.test(key)) return 1;
      if (key==='getShaderParameter' || key==='getProgramParameter') return () => true;
      return () => ({});
    }});
    const context = {fillRect(){},fillText(){}};
    const elements = {
      'start-check': {addEventListener(_name, cb){click=cb;}},
      'native-check': {addEventListener(_name, cb){nativeClick=cb;}},
      'check-status': {},
      'check-canvas': {getContext(){return gl;}},
    };
    const session = {
      inputSources: [], visibilityState:'visible', updateRenderState(){},
      requestReferenceSpace: async () => ({}), addEventListener(){},
      requestAnimationFrame(cb){raf=cb;}, end:async()=>{},
    };
    global.document = {hidden:false,getElementById(id){return elements[id];},
      createElement(){return {getContext(){return context;}};}};
    function openSession(mode, options) {
      assert.equal(mode,'immersive-vr');
      assert.deepEqual(options.optionalFeatures,['local-floor','bounded-floor']);
      return session;
    }
    Object.defineProperty(globalThis, 'navigator', {configurable:true, value:{userAgent:'test', xr:{
      requestSession:async(...args)=>{assert.equal(entry,'request');return openSession(...args);},
      offerSession:async(...args)=>{assert.equal(entry,'offer');return openSession(...args);},
    }}});
    global.performance = {now:()=>now};
    global.location = {host:'test.invalid'};
    global.XRWebGLLayer = class {getViewport(){return {x:0,y:0,width:100,height:100};}};
    global.WebSocket = class {
      static OPEN=1;
      constructor(url){assert.equal(url,'wss://test.invalid/ws');this.readyState=1;}
      send(value){sent.push(JSON.parse(value));} close(){}
    };
    """.replace("ENTRY_VALUE", json.dumps(entry))
        + source
        + """
    (async()=>{
      (entry==='offer' ? nativeClick : click)();
      await new Promise(resolve=>setImmediate(resolve));
      assert.equal(typeof raf,'function',elements['check-status'].textContent);
      const frame = {getViewerPose(){return {views:[{}]};},getPose(){return {
        transform:{position:{x:1,y:2,z:3},orientation:{x:0,y:0,z:0,w:1}}
      };}};
      raf(0,frame);
      assert.equal(sent.length,1);
      assert.deepEqual(sent[0].xr.sources,[]);
      assert.equal(sent[0].controllers.left.tracked,false);
      assert.match(elements['check-status'].textContent,/原生输入源:0/);
      session.inputSources = ['left','right'].map(handedness=>({handedness,gripSpace:{},
        profiles:['meta-quest-touch-pro'],gamepad:{
          buttons:Array.from({length:12},(_,i)=>({pressed:i===4,value:0})),axes:[0,0,0,0]
        }}));
      now=200;
      raf(0,frame);
      assert.equal(sent[1].xr.sources.length,2);
      assert.equal(sent[1].controllers.left.tracked,true);
      assert.equal(sent[1].controllers.right.buttons[4],true);
      assert.deepEqual(sent[1].controllers.left.position,[1,2,3]);
      assert.equal(sent.every(packet=>packet.type==='input'),true);
      assert.equal(sent.some(packet=>'action' in packet),false);
      assert.match(elements['check-status'].textContent,/原生输入源:2/);
    })().catch(error=>{console.error(error);process.exitCode=1;});
    """
    )


def test_probe_is_a_standalone_page_without_video_or_external_scripts():
    page = PROBE.with_suffix(".html").read_text()
    assert '<script defer src="/static/xr-input-check.js"></script>' in page
    assert "<video" not in page
    assert 'src="https://' not in page
    assert "app.js" not in page
    assert json.dumps("arm") not in PROBE.read_text()
