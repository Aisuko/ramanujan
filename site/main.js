import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const REDUCE = matchMedia('(prefers-reduced-motion: reduce)').matches;

function field() {
  const canvas = document.getElementById('field');
  if (!canvas) return;
  const readout = document.getElementById('readout');
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  } catch (e) {
    canvas.style.display = 'none';
    return;
  }
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 200);
  camera.position.set(0, 3.0, 14.5);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.minPolarAngle = 0.5;
  controls.maxPolarAngle = 2.5;
  controls.autoRotate = !REDUCE;
  controls.autoRotateSpeed = 0.5;

  const world = new THREE.Group();
  scene.add(world);

  const N = 110, R = 6.2;
  const C = new THREE.Vector3(0, 0, 0);
  const P = [];
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = Math.PI * (3 - Math.sqrt(5)) * i;
    P.push(new THREE.Vector3(Math.cos(th) * r, y * 0.66, Math.sin(th) * r).multiplyScalar(R));
  }

  const nodes = new THREE.InstancedMesh(
    new THREE.OctahedronGeometry(0.12, 0),
    new THREE.MeshBasicMaterial(),
    N
  );
  const m = new THREE.Matrix4();
  for (let i = 0; i < N; i++) {
    m.makeTranslation(P[i].x, P[i].y, P[i].z);
    nodes.setMatrixAt(i, m);
    nodes.setColorAt(i, new THREE.Color(0x37e6c0));
  }
  nodes.instanceMatrix.needsUpdate = true;
  world.add(nodes);

  const spoke = new Float32Array(N * 6);
  for (let i = 0; i < N; i++) {
    spoke.set([0, 0, 0, P[i].x, P[i].y, P[i].z], i * 6);
  }
  const spokes = new THREE.LineSegments(
    new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(spoke, 3)),
    new THREE.LineBasicMaterial({ color: 0x1b2a37, transparent: true, opacity: 0.4 })
  );
  world.add(spokes);

  world.add(new THREE.LineSegments(
    new THREE.WireframeGeometry(new THREE.IcosahedronGeometry(1.15, 1)),
    new THREE.LineBasicMaterial({ color: 0x37e6c0, transparent: true, opacity: 0.32 })
  ));
  const core = new THREE.Mesh(
    new THREE.IcosahedronGeometry(0.5, 1),
    new THREE.MeshBasicMaterial({ color: 0x37e6c0 })
  );
  world.add(core);
  world.add(new THREE.Mesh(
    new THREE.TorusGeometry(R * 1.02, 0.006, 3, 160),
    new THREE.MeshBasicMaterial({ color: 0x37e6c0, transparent: true, opacity: 0.16 })
  ).rotateX(Math.PI / 2));

  const PARK = 1e4;
  function pulseLayer(color, size, opacity) {
    const a = new Float32Array(N * 3).fill(PARK);
    const g = new THREE.BufferGeometry().setAttribute('position', new THREE.BufferAttribute(a, 3));
    const pts = new THREE.Points(g, new THREE.PointsMaterial({ color, size, transparent: true, opacity, depthWrite: false }));
    pts.frustumCulled = false;
    world.add(pts);
    return { a, g, pts };
  }
  const up = pulseLayer(0x9fb3c8, 0.1, 0.85);
  const down = pulseLayer(0x37e6c0, 0.17, 1);

  const drift = new Float32Array(N);
  const rate = new Float32Array(N);
  const flash = new Float32Array(N);
  const uT = new Float32Array(N).fill(-1);
  const dT = new Float32Array(N).fill(-1);
  for (let i = 0; i < N; i++) {
    drift[i] = Math.random() * 0.3;
    rate[i] = 0.03 + Math.random() * 0.11;
  }

  const cOk = new THREE.Color(0x37e6c0);
  const cBad = new THREE.Color(0xf2a33c);
  const cHot = new THREE.Color(0xffffff);
  const tmp = new THREE.Color();
  const v = new THREE.Vector3();

  function setPoint(layer, i, x, y, z) {
    layer.a[i * 3] = x; layer.a[i * 3 + 1] = y; layer.a[i * 3 + 2] = z;
  }

  let running = true, last = performance.now(), acc = 0;

  function tick(now) {
    requestAnimationFrame(tick);
    const dt = Math.min((now - last) / 1000, 0.05);
    last = now;
    if (!running) return;

    let drifted = 0;
    for (let i = 0; i < N; i++) {
      drift[i] = Math.min(1, drift[i] + rate[i] * dt);
      if (drift[i] > 0.28) drifted++;

      if (drift[i] > 0.3 && uT[i] < 0 && dT[i] < 0 && Math.random() < 0.6 * dt) uT[i] = 0;

      if (uT[i] >= 0) {
        uT[i] += dt * 0.95;
        if (uT[i] >= 1) { uT[i] = -1; dT[i] = 0; setPoint(up, i, PARK, PARK, PARK); }
        else { v.copy(P[i]).lerp(C, uT[i]); setPoint(up, i, v.x, v.y, v.z); }
      }
      if (dT[i] >= 0) {
        dT[i] += dt * 1.6;
        if (dT[i] >= 1) { dT[i] = -1; drift[i] = 0; flash[i] = 1; setPoint(down, i, PARK, PARK, PARK); }
        else { v.copy(C).lerp(P[i], dT[i]); setPoint(down, i, v.x, v.y, v.z); }
      }

      flash[i] = Math.max(0, flash[i] - dt * 2.2);
      tmp.copy(cOk).lerp(cBad, drift[i]);
      if (flash[i] > 0) tmp.lerp(cHot, flash[i]);
      nodes.setColorAt(i, tmp);
    }
    nodes.instanceColor.needsUpdate = true;
    up.g.attributes.position.needsUpdate = true;
    down.g.attributes.position.needsUpdate = true;

    core.scale.setScalar(1 + Math.sin(now / 420) * 0.08);
    controls.update();
    renderer.render(scene, camera);

    acc += dt;
    if (acc > 0.25 && readout) {
      acc = 0;
      readout.innerHTML = `<span class="text-accent">●</span> RECONCILING · ${String(drifted).padStart(3, '0')} DRIFTED / ${N} EDGES`;
    }
  }

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    world.position.x = w > 1024 ? 4.2 : w > 720 ? 2.2 : 0;
    world.position.y = w > 720 ? 0 : -1.2;
  }
  addEventListener('resize', resize);
  resize();
  requestAnimationFrame(tick);

  const hero = document.getElementById('top');
  if (hero) {
    new IntersectionObserver((es) => { running = es[0].isIntersecting; last = performance.now(); }).observe(hero);
  }
}

function reveal() {
  const io = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); }
  }, { rootMargin: '-10% 0px -10% 0px' });
  document.querySelectorAll('.rise').forEach((n) => io.observe(n));
}

function ladder() {
  const box = document.getElementById('ladder');
  if (!box) return;
  new IntersectionObserver((es, o) => {
    if (!es[0].isIntersecting) return;
    o.disconnect();
    box.querySelectorAll('.bar').forEach((b, i) => {
      setTimeout(() => { b.style.width = b.dataset.w + '%'; }, 90 * i);
    });
  }, { threshold: 0.25 }).observe(box);
}

function rail() {
  const links = [...document.querySelectorAll('.rail a')];
  const io = new IntersectionObserver((es) => {
    for (const e of es) {
      if (!e.isIntersecting) continue;
      links.forEach((l) => l.setAttribute('aria-current', l.hash === '#' + e.target.id));
    }
  }, { threshold: 0.4 });
  links.forEach((l) => { const s = document.querySelector(l.hash); if (s) io.observe(s); });
}

function bubbleDemo() {
  const art = document.getElementById('reactive');
  const bub = document.getElementById('bubble');
  if (!art || !bub) return;
  const bHead = document.getElementById('b-head');
  const bText = document.getElementById('b-text');
  const bMs = document.getElementById('b-ms');
  const bConf = document.getElementById('b-conf');

  const STOP = new Set(['the', 'and', 'to', 'of', 'a', 'it', 'is', 'then', 'so', 'that', 'not', 'on', 'in']);
  const HITS = [
    ['mmr', 'maximal marginal relevance — ranks candidates by relevance minus redundancy against what you already have.', 92],
    ['redundanc', 'overlap with candidates already chosen. MMR subtracts it so the bubble does not repeat itself.', 81],
    ['pointer head', 'scores every other span on the page against your selection and returns the best one, verbatim.', 90],
    ['webgpu', "the browser's GPU API. forge compiles the encoder to WGSL and runs it there — no install, no endpoint.", 88],
    ['reconcile', 'the edge reports observed status upward; the centre returns the delta to the expected state. Continuous, never blocking.', 85],
    ['expected state', 'what the centralised network believes each edge should be running. The edge converges toward it.', 83],
    ['threshold', 'below it, the bubble says the page never explains this — instead of pointing somewhere wrong.', 79],
    ['dequantis', 'q4 codes unpack to f32 while loading into GPU buffers, so the size win costs no new kernels.', 76],
    ['weights', '5.94M parameters, 0.92 MB at the target quantisation. Bundled with the app, never fetched.', 80],
    ['warm', 'pipeline creation is the one-time cost. The tab keeps the device and buffers alive between reactions.', 74],
  ];

  function react(raw) {
    const t = raw.toLowerCase().trim();
    if (t.length < 3 || STOP.has(t)) {
      return { head: 'HEAD A · NULL CLASS', text: 'This page never explains that. Nothing to point at, so nothing is invented.', conf: 11, warn: true };
    }
    for (const [k, text, conf] of HITS) if (t.includes(k)) return { head: 'HEAD A · POINTER', text, conf };
    return { head: 'HEAD A · POINTER', text: 'Closest explaining span on this page: “trade relevance against redundancy”.', conf: 43 };
  }

  function show() {
    const sel = getSelection();
    if (!sel || sel.isCollapsed || !sel.rangeCount) return hide();
    const r = sel.getRangeAt(0);
    if (!art.contains(r.commonAncestorContainer)) return hide();
    const txt = sel.toString();
    if (!txt.trim()) return hide();

    const out = react(txt);
    bHead.textContent = out.head;
    bHead.className = 'mono text-[9px] tracking-[.18em] ' + (out.warn ? 'text-warn' : 'text-accent');
    bText.textContent = out.text;
    bMs.textContent = (6.8 + Math.random() * 4.6).toFixed(1) + ' ms';
    bConf.className = 'bar ' + (out.warn ? 'bg-warn' : 'bg-accent');

    const box = r.getBoundingClientRect();
    bub.classList.remove('hidden');
    const w = bub.offsetWidth, h = bub.offsetHeight;
    let x = box.left + box.width / 2 - w / 2;
    x = Math.max(12, Math.min(x, innerWidth - w - 12));
    const above = box.top > h + 16;
    bub.style.left = x + 'px';
    bub.style.top = (above ? box.top - h - 10 : box.bottom + 10) + 'px';
    bConf.style.width = '0%';
    requestAnimationFrame(() => { bConf.style.width = out.conf + '%'; });
  }

  function hide() { bub.classList.add('hidden'); }

  let t = 0;
  document.addEventListener('selectionchange', () => {
    clearTimeout(t);
    t = setTimeout(show, 140);
  });
  bub.addEventListener('mousedown', (e) => e.preventDefault());
  addEventListener('scroll', hide, { passive: true });
  addEventListener('keydown', (e) => { if (e.key === 'Escape') { hide(); getSelection().removeAllRanges(); } });

  new IntersectionObserver((es, o) => {
    if (!es[0].isIntersecting || REDUCE) return;
    o.disconnect();
    const seed = [...art.querySelectorAll('b')].find((b) => b.textContent.trim() === 'MMR');
    if (!seed) return;
    setTimeout(() => {
      const r = document.createRange();
      r.selectNodeContents(seed);
      const sel = getSelection();
      sel.removeAllRanges();
      sel.addRange(r);
      show();
    }, 700);
  }, { threshold: 0.55 }).observe(art);
}

field();
reveal();
ladder();
rail();
bubbleDemo();
