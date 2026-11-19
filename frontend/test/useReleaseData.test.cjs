const assert = require('node:assert/strict')
const fs = require('node:fs')
const { test } = require('node:test')
const ts = require('typescript')
const React = require('react')
const { act, create } = require('react-test-renderer')

// Run the actual TypeScript hook and validators with Node's test runner.
require.extensions['.ts'] = (module, filename) => {
  module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
  }).outputText, filename)
}
const { useReleaseData } = require('../src/hooks/useReleaseData.ts')

function deferred() {
  let resolve, reject
  const promise = new Promise((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}

const curves = precision => ({ NK_biological_process_Naive: [{ tau: 0.5, precision, recall: 0.5 }] })
const response = value => ({ ok: true, json: async () => value })
function essential(releaseId, name) {
  if (name === 'methods.json') return { methods: {} }
  if (name === 'best.json') return {}
  return {
    releaseId, startTimepoint: 'Mar_2026', endTimepoint: releaseId,
    dates: { goaStart: 'start', goaEnd: 'end', uniprotStart: 'start', uniprotEnd: 'end' },
    targetCounts: { uniqueAcrossSubsets: 1, ...Object.fromEntries(
      ['NK', 'LK', 'PK'].map(s => [s, { total: 1, byAspect: {} }]),
    ) },
  }
}

async function mount(t, initial = 'June', holdMeta = false) {
  const originalFetch = global.fetch
  const requests = [], renders = []
  let current, renderer
  global.fetch = (url, options) => {
    const [, , , releaseId, name] = url.split('/')
    const pending = deferred()
    requests.push({ url, releaseId, name, signal: options.signal, ...pending })
    // Deliberately ignore abort: verify the hook discards late responses even
    // when a mock, cached response, or response-body parse finishes afterward.
    if (name !== 'curves.json' && !(holdMeta && name === 'meta.json')) {
      pending.resolve(response(essential(releaseId, name)))
    }
    return pending.promise
  }
  function Probe({ id }) {
    current = useReleaseData(id)
    renders.push({ id, ...current })
    return null
  }
  await act(async () => { renderer = create(React.createElement(Probe, { id: initial })) })
  t.after(async () => {
    await act(async () => renderer.unmount())
    global.fetch = originalFetch
  })
  return {
    get current() { return current }, requests, renders,
    async select(id) { await act(async () => renderer.update(React.createElement(Probe, { id }))) },
    async load() { await act(async () => { void current.loadCurves() }) },
    async unmount() { await act(async () => renderer.unmount()) },
  }
}

test('loaded curves are cleared immediately and fetched for the new selection', async t => {
  const h = await mount(t)
  assert.equal(h.requests.filter(r => r.name === 'curves.json').length, 0)
  await h.load()
  await act(async () => h.requests.at(-1).resolve(response(curves(0.2))))
  assert.deepEqual(h.current.data.curves, curves(0.2))
  await h.select('September')
  for (const render of h.renders.filter(r => r.id === 'September')) {
    assert.equal(render.data.curves, null)
    assert.ok(render.data.meta === null || render.data.meta.releaseId === 'September')
  }
  await h.load()
  assert.equal(h.requests.at(-1).releaseId, 'September')
  await act(async () => h.requests.at(-1).resolve(response(curves(0.8))))
  assert.deepEqual(h.current.data.curves, curves(0.8))
})

test('old curve response cannot overwrite a new request or clear its loading state', async t => {
  const h = await mount(t)
  await h.load()
  const old = h.requests.at(-1)
  await h.select('September')
  assert.equal(old.signal.aborted, true)
  await h.load()
  const active = h.requests.at(-1)
  await act(async () => old.resolve(response(curves(0.2))))
  assert.equal(h.current.data.curves, null)
  assert.equal(h.current.curvesLoading, true)
  await act(async () => active.resolve(response(curves(0.8))))
  assert.deepEqual(h.current.data.curves, curves(0.8))
})

test('late body parsing is discarded even when returning to the same release', async t => {
  const h = await mount(t)
  await h.load()
  const body = deferred()
  await act(async () => h.requests.at(-1).resolve({ ok: true, json: () => body.promise }))
  await h.select('September')
  await h.select('June')
  await h.load()
  await act(async () => h.requests.at(-1).resolve(response(curves(0.7))))
  await act(async () => body.resolve(curves(0.1)))
  assert.deepEqual(h.current.data.curves, curves(0.7))
  assert.equal(h.current.error, null)
})

test('stale errors and essential responses cannot contaminate a new release', async t => {
  const h = await mount(t, 'June', true)
  const oldMeta = h.requests.find(r => r.name === 'meta.json')
  await h.load()
  const oldCurves = h.requests.at(-1)
  await h.select('September')
  const newMeta = h.requests.find(r => r.releaseId === 'September' && r.name === 'meta.json')
  await act(async () => {
    newMeta.resolve(response(essential('September', 'meta.json')))
    oldMeta.resolve(response(essential('June', 'meta.json')))
    oldCurves.reject(new Error('old request failed'))
  })
  assert.equal(h.current.data.meta.releaseId, 'September')
  assert.equal(h.current.loading, false)
  assert.equal(h.current.error, null)
})

test('deselecting or unmounting aborts requests and clears selection state', async t => {
  const h = await mount(t)
  await h.load()
  const old = h.requests.at(-1)
  await h.select(null)
  await act(async () => old.resolve(response(curves(0.2))))
  assert.equal(old.signal.aborted, true)
  assert.deepEqual(h.current.data, { meta: null, methods: null, best: null, curves: null })
  assert.equal(h.current.loading, false)
  assert.equal(h.current.curvesLoading, false)
  assert.equal(h.current.error, null)
  await h.select('September')
  await h.load()
  const active = h.requests.at(-1)
  await h.unmount()
  assert.equal(active.signal.aborted, true)
  await act(async () => active.reject(new Error('after unmount')))
})

test('duplicate curve loads share one request and current errors are reported', async t => {
  const h = await mount(t)
  await h.load()
  await h.load()
  assert.equal(h.requests.filter(r => r.name === 'curves.json').length, 1)
  await act(async () => h.requests.at(-1).resolve({ ok: false }))
  assert.equal(h.current.error, 'Failed to load curves data')
  assert.equal(h.current.curvesLoading, false)
  // App's loading effect must not spin on a failed request.
  await h.load()
  assert.equal(h.requests.filter(r => r.name === 'curves.json').length, 1)
})
