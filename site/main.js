const TOOLS = [
  { id: 'search-001', name: 'Web search agent', s: [
    { t: 'There are many open-weight language models available today, and the right choice depends on your needs.' },
    { t: 'Gemma 2 9B runs on a single 24 GB consumer GPU, and its licence allows commercial self-hosting.', m: 'gemma', f: ['selfhost', 'single-gpu'], c: 'runs on a single 24 GB GPU under a licence that allows commercial self-hosting' },
    { t: 'Qwen2.5 7B covers 29 languages, including Chinese, Arabic and Vietnamese.', m: 'qwen', f: ['multilingual'], c: 'covers 29 languages' },
    { t: 'Llama 3.1 70B leads several English chat benchmarks but needs multiple datacenter GPUs.', m: 'llama', f: ['english', 'datacenter'], c: 'leads English chat benchmarks on datacenter GPUs' },
    { t: 'Gemma 2 9B was trained mostly on English text, so English is its strongest language.', m: 'gemma', f: ['english'], c: 'is strongest in English' },
    { t: 'It is always a good idea to evaluate models on your own data before deploying them.' },
  ] },
  { id: 'rag-002', name: 'Docs retrieval tool', s: [
    { t: 'According to the Qwen2.5 documentation, the 7B model supports more than 29 languages.', m: 'qwen', f: ['multilingual'], dup: 'search-001:2' },
    { t: 'Qwen2.5 7B fits in 16 GB of GPU memory with 4-bit weights and can be self-hosted.', m: 'qwen', f: ['selfhost', 'single-gpu'], c: 'fits in 16 GB of GPU memory with 4-bit weights' },
    { t: 'Hosted APIs such as Gemini Flash need no infrastructure and are billed per token.', m: 'gemini', f: ['api'], c: 'needs no infrastructure and is billed per token' },
    { t: 'Gemini Flash handles more than 100 languages through the managed API.', m: 'gemini', f: ['api', 'multilingual'], c: 'handles more than 100 languages' },
  ] },
  { id: 'code-003', name: 'Coding assistant', s: [
    { t: 'Here is a general overview of how to deploy a chatbot.' },
    { t: 'vLLM can serve Gemma 2 9B on one GPU behind an OpenAI-compatible endpoint.', m: 'gemma', f: ['selfhost', 'single-gpu'], c: 'can be served by vLLM on one GPU' },
    { t: 'Running Llama 3.1 70B in production usually means tensor parallelism across four or more GPUs.', m: 'llama', f: ['selfhost', 'datacenter'], c: 'runs self-hosted with tensor parallelism across four or more GPUs' },
    { t: 'A managed API removes the need to patch drivers and maintain GPU servers.', m: 'gemini', f: ['api', 'single-gpu', 'datacenter'], c: 'removes the need to maintain GPU servers' },
    { t: 'As mentioned above, Gemma 2 9B fits on one 24 GB card.', m: 'gemma', f: ['selfhost', 'single-gpu'], dup: 'search-001:1' },
  ] },
];

const MODELS = { gemma: 'Gemma 2 9B', qwen: 'Qwen2.5 7B', llama: 'Llama 3.1 70B', gemini: 'Gemini Flash' };

const QAS = [
  { id: 'q-001', q: 'Where will the model run?', w: 3, a: [['selfhost', 'Self-hosted'], ['api', 'Managed API']] },
  { id: 'q-002', q: 'Which languages must it handle?', w: 2, a: [['english', 'English is enough'], ['multilingual', 'Many languages']] },
  { id: 'q-003', q: 'What hardware do you have?', w: 1, a: [['single-gpu', 'One consumer GPU'], ['datacenter', 'A GPU cluster']] },
];

const MISSING = { selfhost: 'self-hosting', api: 'a managed API', english: 'English quality', multilingual: 'multilingual support', 'single-gpu': 'single-GPU deployment', datacenter: 'cluster deployment' };
const OPPOSITE = { selfhost: 'api', api: 'selfhost', 'single-gpu': 'datacenter' };

const SENTENCES = TOOLS.flatMap(tool => tool.s.map((s, i) => ({ ...s, key: `${tool.id}:${i}`, source: tool.id })));
SENTENCES.forEach((s, i) => { s.index = i; });

const picked = {};
let request = 0;

const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  el.append(...kids);
  return el;
};

const join = xs => xs.length < 2 ? xs.join('') : `${xs.slice(0, -1).join(', ')} and ${xs[xs.length - 1]}`;

function select(facets) {
  if (!facets.length) return { model: null, evidence: [], missing: [] };
  const weight = Object.fromEntries(QAS.flatMap(q => q.a.map(([f]) => [f, q.w])));
  const conflicts = s => facets.some(f => OPPOSITE[f] && s.f.includes(OPPOSITE[f]) && !s.f.includes(f));
  const ranked = Object.keys(MODELS).map(m => {
    const own = SENTENCES.filter(s => s.m === m && !s.dup);
    const covered = facets.filter(f => own.some(s => s.f.includes(f) && !conflicts(s)));
    return {
      m, own, covered,
      cover: covered.reduce((a, f) => a + weight[f], 0),
      bad: own.filter(conflicts).length,
      hits: own.reduce((a, s) => a + s.f.filter(f => facets.includes(f)).length, 0),
    };
  }).sort((a, b) => b.cover - a.cover || a.bad - b.bad || b.hits - a.hits);
  const best = ranked[0];
  const evidence = best.own
    .filter(s => !conflicts(s))
    .map(s => ({ s, score: s.f.filter(f => facets.includes(f)).length / facets.length }))
    .filter(e => e.score > 0)
    .sort((a, b) => b.score - a.score || a.s.index - b.s.index)
    .slice(0, 3);
  return { model: best.m, evidence, missing: facets.filter(f => !best.covered.includes(f)) };
}

function summarise({ model, evidence, missing }) {
  if (!model) return [`The three tools compare ${join(Object.values(MODELS))}, but which one fits depends on how you deploy, which languages you need and what hardware you have.`];
  const claims = evidence.map((e, i) => `${e.s.c}<sup>[${i + 1}]</sup>`);
  const out = [`${MODELS[model]} best fits your answers: it ${join(claims)}.`];
  if (missing.length) out.push(`None of the tool outputs confirm ${join(missing.map(f => MISSING[f]))} for it.`);
  return out;
}

const truncate = t => t.length > 60 ? `${t.slice(0, 57)}…` : t;
const words = t => t.split(/\s+/).filter(Boolean).length;

function render() {
  const facets = QAS.map(q => picked[q.id]).filter(Boolean);
  const result = select(facets);
  const sentences = summarise(result);
  const chosen = new Map(result.evidence.map((e, i) => [e.s.key, i + 1]));
  request += 1;

  document.querySelectorAll('.s').forEach(el => {
    const s = SENTENCES[+el.dataset.i];
    const rank = chosen.get(s.key);
    const score = s.f ? s.f.filter(f => facets.includes(f)).length / Math.max(facets.length, 1) : 0;
    el.classList.toggle('ev', !!rank);
    el.classList.toggle('dup', !!s.dup && facets.length > 0 && s.f.some(f => facets.includes(f)));
    el.classList.toggle('gen', !s.f);
    if (rank) el.dataset.ref = `[${rank}]`;
    el.querySelector('.bar').style.width = `${Math.round(score * 100)}%`;
  });

  document.getElementById('summary').innerHTML = sentences.join(' ');
  const evidence = document.getElementById('evidence');
  evidence.replaceChildren(...result.evidence.map(e => h('li', {}, h('code', {}, `${e.s.source}`), ` ${e.s.t}`)));

  const refining = request > 1;
  document.getElementById('stats').replaceChildren(
    h('span', {}, `req-${String(request).padStart(3, '0')}`),
    h('span', { class: refining ? 'on' : '' }, refining ? `refinement ${request - 1}` : 'first request'),
    h('span', { class: 'on' }, 'tools rerun: 0'),
    h('span', { class: refining ? 'on' : '' }, refining ? 'source encoding: cached' : 'source encoding: computed'),
    h('span', {}, `evidence: ${result.evidence.length}`),
    h('span', {}, `sentences: ${sentences.length}`),
  );

  const id = n => `req-${String(n).padStart(3, '0')}`;
  const req = {
    request_id: id(request),
    task: 'focused_summarisation',
    tool_outputs: TOOLS.map(t => ({ source_id: t.id, content: truncate(t.s.map(s => s.t).join(' ')) })),
    selected_qa: QAS.filter(q => picked[q.id]).map(q => ({ question_id: q.id, question: q.q, answer: q.a.find(([f]) => f === picked[q.id])[1] })),
    constraints: { min_sentences: 1, max_sentences: 3, style: 'concise', preserve_factuality: true },
    previous_request_id: refining ? id(request - 1) : null,
  };
  const plain = sentences.join(' ').replace(/<sup>\[\d+\]<\/sup>/g, '');
  const res = {
    request_id: id(request),
    summary: plain,
    sentence_count: sentences.length,
    evidence: result.evidence.map(e => ({ source_id: e.s.source, text_span: e.s.t, sentence_index: e.s.index, score: +e.score.toFixed(2) })),
    metrics: { selected_evidence: result.evidence.length, input_tokens: Math.round(words(SENTENCES.map(s => s.t).join(' ')) * 1.3), output_tokens: Math.round(words(plain) * 1.3) },
  };
  document.getElementById('req').textContent = JSON.stringify(req, null, 2);
  document.getElementById('res').textContent = JSON.stringify(res, null, 2);
}

document.getElementById('toolmeta').textContent =
  `Agent called ${TOOLS.length} tools → ${SENTENCES.length} sentences, ${words(SENTENCES.map(s => s.t).join(' '))} words`;

document.getElementById('tools').replaceChildren(...TOOLS.map(tool => h('div', { class: 'tool' },
  h('h3', {}, tool.id, h('span', {}, tool.name)),
  ...tool.s.map((s, i) => h('span', { class: 's', 'data-i': SENTENCES.find(x => x.key === `${tool.id}:${i}`).index }, s.t, h('span', { class: 'bar' }))),
)));

document.getElementById('qas').replaceChildren(...QAS.map(q => h('div', { class: 'qa' },
  h('p', {}, q.q),
  h('div', { class: 'chips' }, ...q.a.map(([f, label]) => {
    const b = h('button', { class: 'chip', type: 'button', 'aria-pressed': 'false' }, label);
    b.addEventListener('click', () => {
      picked[q.id] = picked[q.id] === f ? undefined : f;
      b.parentElement.querySelectorAll('.chip').forEach(c => c.setAttribute('aria-pressed', String(c === b && picked[q.id] === f)));
      render();
    });
    return b;
  })),
)));

render();
