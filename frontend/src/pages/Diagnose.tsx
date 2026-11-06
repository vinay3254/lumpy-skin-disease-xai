import { useEffect, useRef, useState } from 'react'
import { api, assetUrl } from '../api'
import type { ClassCount, ModelsResponse, PredictionResponse } from '../types'
import { EmptyState, ErrorToast, Figure, Icon, PageHeading, Spinner } from '../components/UI'

function classLabel(name: string) { return name.replace(/[_-]/g, ' ').replace(/\blsd\b/gi, 'LSD') }
function classTone(name: string) { return name.toLowerCase() === 'healthy' ? 'healthy' : name.toLowerCase() === 'lsd' ? 'disease' : 'other' }

export default function Diagnose() {
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [samples, setSamples] = useState<string[]>([])
  const [loadingModels, setLoadingModels] = useState(true)
  const [sampleError, setSampleError] = useState(false)
  const [mode, setMode] = useState<ClassCount>('2')
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState('')
  const [result, setResult] = useState<PredictionResponse | null>(null)
  const [shap, setShap] = useState('')
  const [busy, setBusy] = useState(false)
  const [sampleBusy, setSampleBusy] = useState(false)
  const [shapBusy, setShapBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState('')
  const input = useRef<HTMLInputElement>(null)
  const version = useRef(0)
  const locked = busy || shapBusy || sampleBusy
  const run = models?.defaults[mode] || models?.runs.find((item) => item.classes === Number(mode))?.run || ''

  function loadModels() {
    setLoadingModels(true)
    api.models().then(setModels).catch((reason: Error) => setError(reason.message)).finally(() => setLoadingModels(false))
  }
  useEffect(() => {
    let active = true
    api.models().then((data) => { if (active) setModels(data) }).catch((reason: Error) => { if (active) setError(reason.message) }).finally(() => { if (active) setLoadingModels(false) })
    api.samples().then((data) => { if (active) setSamples(data) }).catch(() => { if (active) setSampleError(true) })
    return () => { active = false }
  }, [])
  useEffect(() => () => { if (preview) URL.revokeObjectURL(preview) }, [preview])
  function chooseFile(next: File) {
    if (locked) return
    if (!next.type.startsWith('image/')) { setError('Please select an image file, such as JPG, PNG, or WebP.'); return }
    version.current += 1
    setFile(next); setPreview(URL.createObjectURL(next)); setResult(null); setShap(''); setError('')
  }
  async function chooseSample(url: string) {
    const current = ++version.current
    setSampleBusy(true)
    try {
      const next = await api.sample(url)
      if (current === version.current) { setFile(next); setPreview(URL.createObjectURL(next)); setResult(null); setShap(''); setError('') }
    } catch (reason) { setError(reason instanceof Error ? reason.message : 'Could not load sample.') }
    finally { setSampleBusy(false) }
  }
  async function analyze() {
    if (!file || !run || locked) return
    setBusy(true); setResult(null); setShap(''); setError('')
    try { setResult(await api.predict(file, run)) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'Analysis failed. Please try again.') }
    finally { setBusy(false) }
  }
  async function explain() {
    if (!file || !result || locked) return
    setShapBusy(true); setError('')
    try { setShap((await api.shap(file, result.run)).shap) }
    catch (reason) { setError(reason instanceof Error ? reason.message : 'SHAP explanation failed. Please try again.') }
    finally { setShapBusy(false) }
  }
  return <>
    <PageHeading eyebrow="SEE THE PREDICTION. UNDERSTAND THE WHY." title="A clearer picture of cattle health.">Analyze a cattle image with a hybrid vision model and explore the evidence behind its prediction.</PageHeading>
    <div className="diagnose-layout">
      <section className="card input-card" aria-labelledby="image-title">
        <div className="section-heading"><span className="step-number">01</span><div><h2 id="image-title">Choose an image</h2><p>Start with a clear photograph of the animal.</p></div></div>
        <div className={`drop-zone ${dragging ? 'dragging' : ''} ${preview ? 'has-preview' : ''}`} onDragOver={(event) => { event.preventDefault(); if (!locked) setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); if (event.dataTransfer.files[0]) chooseFile(event.dataTransfer.files[0]) }}>
          {preview ? <><img className="upload-preview" src={preview} alt="Selected cattle image" /><button className="replace-image" disabled={locked} onClick={() => input.current?.click()}>Change image</button></> : <><span className="upload-icon"><Icon name="upload" size={28} /></span><h3>Drop your image here</h3><p>or choose a file from your device</p><button className="button secondary" disabled={locked} onClick={() => input.current?.click()}>Browse images</button><span className="file-hint">JPG, PNG, WebP and other image formats</span></>}
          <input ref={input} type="file" accept="image/*" className="sr-only" aria-label="Upload a cattle image" disabled={locked} onChange={(event) => { if (event.target.files?.[0]) chooseFile(event.target.files[0]); event.target.value = '' }} />
        </div>
        {file && <p className="filename" title={file.name}>{file.name}<span>{(file.size / 1024).toFixed(0)} KB</span></p>}
        <div className="sample-heading"><h3>Or try a sample</h3><span>Test dataset</span></div>
        {sampleBusy ? <Spinner label="Loading sample…" /> : samples.length ? <div className="sample-strip">{samples.map((url, index) => <button key={url} disabled={locked} onClick={() => void chooseSample(url)} aria-label={`Use sample image ${index + 1}`}><img src={assetUrl(url)} alt={`Cattle sample ${index + 1}`} loading="lazy" /></button>)}</div> : <p className="small muted sample-empty">{sampleError ? 'Sample images are unavailable. Upload an image to get started.' : 'Sample images will appear here when available.'}</p>}
        <div className="divider" />
        <div className="section-heading"><span className="step-number">02</span><div><h2>Choose a model</h2><p>Select the classification task.</p></div></div>
        <div className="segmented" aria-label="Classification model">{(['2', '3'] as const).map((value) => <button key={value} disabled={locked} aria-pressed={mode === value} className={mode === value ? 'selected' : ''} onClick={() => { setMode(value); setResult(null); setShap('') }}>{value === '2' ? 'Binary' : '3-class'}<span>{value === '2' ? 'Healthy / LSD' : 'Healthy / LSD / Other'}</span></button>)}</div>
        <p className="model-note">{loadingModels ? <Spinner label="Finding available models…" /> : run ? <><span className="status-dot" /><span title={run}>{run.replaceAll('_', ' ')}</span></> : <>No model available. <button className="text-button" onClick={loadModels}>Retry connection</button></>}</p>
        <button className="button primary analyze-button" disabled={!file || !run || loadingModels || locked} onClick={() => void analyze()}>{busy ? <Spinner label="Analyzing image…" /> : <><Icon name="scan" size={19} />Analyze image<Icon name="arrow" size={18} /></>}</button>
        <p className="privacy-note">Images are processed by your local research server.</p>
      </section>
      <section className="card prediction-card" aria-labelledby="prediction-title"><div className="panel-title"><div><p className="eyebrow">MODEL INSIGHTS</p><h2 id="prediction-title">Prediction overview</h2></div><span className="tag">{mode === '2' ? 'Binary' : '3-class'} model</span></div>
        {busy ? <div className="analysis-loading"><Spinner label="Examining the image…" /><p>Generating a prediction, Grad-CAM, and LIME explanations. This may take a moment.</p></div> : result ? <div className="prediction-content"><div className={`prediction-summary ${classTone(result.pred)}`}><span className="small">Predicted class</span><div><span className={`prediction-badge ${classTone(result.pred)}`}>{classLabel(result.pred)}</span><strong>{(result.confidence * 100).toFixed(1)}<span>%</span></strong></div><p>Model confidence</p></div><h3 className="probability-title">Class probabilities</h3><div className="probabilities">{result.class_names.map((name) => <div key={name}><div className="bar-label"><span>{classLabel(name)}</span><strong>{((result.probs[name] ?? 0) * 100).toFixed(1)}%</strong></div><div className="probability-track"><div className={`probability-fill ${classTone(name)}`} style={{ width: `${Math.max(0, Math.min(100, (result.probs[name] ?? 0) * 100))}%` }} /></div></div>)}</div><p className="result-run">Run: {result.run}</p><div className="interpretation-note"><Icon name="info" size={18} /><p>Confidence reflects the model’s prediction, not a clinical diagnosis. Review the visual explanations below.</p></div></div> : <EmptyState title="Your analysis starts here">Upload an image and select Analyze image to see the predicted class, confidence, and visual explanations.</EmptyState>}
        <div className="insight-footer"><span className="status-dot" />EfficientNet-B0 + DeiT-Tiny<span>Explainable by design</span></div>
      </section>
    </div>
    <section className="explanations"><div className="section-title-row"><div><p className="eyebrow">BEHIND THE PREDICTION</p><h2>Visual explanations</h2><p>Explore which regions of the image influenced the model.</p></div>{result && <button className="button secondary" disabled={locked || !!shap} onClick={() => void explain()}>{shapBusy ? <Spinner label="Computing SHAP…" /> : shap ? 'SHAP explanation loaded' : <>Explain deeper (SHAP)<Icon name="arrow" size={17} /></>}</button>}</div>
      {result ? <><div className="xai-grid"><Figure key={`original-${preview}`} src={preview} title="Original image" caption="The image provided to the model." /><Figure key={result.gradcam} src={result.gradcam} title="Grad-CAM" caption="Highlights regions that activate the model." /><Figure key={result.lime} src={result.lime} title="LIME" caption="Shows local regions supporting the prediction." /></div>{shapBusy && <div className="card shap-loading"><Spinner label="Computing SHAP feature attributions…" /><p>This deeper explanation can take longer. Keep this page open.</p></div>}{shap && <div className="shap-panel"><Figure key={shap} src={shap} title="SHAP explanation" caption="Pixel contributions relative to a background of training images." /></div>}</> : <div className="explanation-placeholder"><Icon name="info" size={18} /><p>Original, Grad-CAM, and LIME views will appear after analysis.</p></div>}
    </section>
    {error && <ErrorToast message={error} onDismiss={() => setError('')} />}
  </>
}
