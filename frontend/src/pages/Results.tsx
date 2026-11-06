import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'
import type { AblationRow, ResultsResponse } from '../types'
import { EmptyState, ErrorToast, Figure, PageHeading, Spinner } from '../components/UI'

const columns = [['params_M', 'Params (M)'], ['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1'], ['roc_auc', 'ROC AUC']] as const
function numeric(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined || value === '') return null
  const result = Number(value)
  return Number.isFinite(result) ? result : null
}
function format(value: number | string | null | undefined, percent = true) {
  const number = numeric(value)
  return number === null ? '—' : percent ? `${(number * 100).toFixed(2)}%` : number.toFixed(2)
}
export default function Results() {
  const [data, setData] = useState<ResultsResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [selected, setSelected] = useState('')
  const [attempt, setAttempt] = useState(0)
  useEffect(() => {
    let active = true
    api.results().then((response) => {
      if (!active) return
      setData(response)
      setSelected(Object.keys(response.figures)[0] || '')
    }).catch((reason: Error) => { if (active) setError(reason.message) }).finally(() => { if (active) setLoading(false) })
    return () => { active = false }
  }, [attempt])
  const rows = data?.ablation || []
  const best = rows.reduce<AblationRow | null>((current, row) => numeric(row.f1) !== null && (current === null || (numeric(row.f1) ?? -1) > (numeric(current.f1) ?? -1)) ? row : current, null)
  const chart = rows.map((row) => ({ run: row.run, Accuracy: numeric(row.accuracy) === null ? null : Number(row.accuracy) * 100, F1: numeric(row.f1) === null ? null : Number(row.f1) * 100 }))
  const figures = selected ? data?.figures[selected] : undefined
  return <>
    <PageHeading eyebrow="MEASURED. COMPARED. EXPLAINED." title="Performance, in perspective.">Compare model variants and inspect their evaluation figures. The highlighted run has the highest F1 score.</PageHeading>
    {loading ? <div className="card results-loading"><Spinner label="Loading experiment results…" /></div> : !data ? <div className="card"><EmptyState title="Results could not be loaded">Connect to the local API to explore the completed experiments.</EmptyState><div className="retry-row"><button className="button secondary" onClick={() => { setLoading(true); setError(''); setAttempt((value) => value + 1) }}>Try again</button></div></div> : <>
      <div className="stat-grid"><div className="card stat-card"><p>EXPERIMENTS</p><strong>{rows.length.toString().padStart(2, '0')}</strong><span>Evaluated model variants</span></div><div className="card stat-card"><p>HIGHEST F1</p><strong>{format(best?.f1)}</strong><span>{best?.run || 'No metrics available'}</span></div><div className="card stat-card"><p>ACCURACY OF BEST F1 RUN</p><strong>{format(best?.accuracy)}</strong><span>Use class count when comparing runs</span></div></div>
      <section className="card results-table-card"><div className="panel-title"><div><h2>Ablation study</h2><p>Architecture variants and their evaluation metrics</p></div><span className="tag green">Best F1 highlighted</span></div>{rows.length ? <div className="table-scroll" tabIndex={0} role="region" aria-label="Ablation metrics table"><table><thead><tr><th scope="col">Model run</th>{columns.map(([key, label]) => <th scope="col" key={key}>{label}</th>)}</tr></thead><tbody>{rows.map((row) => <tr key={row.run} className={best?.run === row.run ? 'best-row' : ''}><th scope="row"><span>{row.run}</span>{best?.run === row.run && <span className="best-marker">BEST</span>}</th>{columns.map(([key]) => <td key={key}>{format(row[key], key !== 'params_M')}</td>)}</tr>)}</tbody></table></div> : <EmptyState title="No experiments yet">Evaluation metrics will appear here once trained runs are available.</EmptyState>}</section>
      {rows.length > 0 && <section className="card chart-card"><div className="panel-title"><div><h2>Model comparison</h2><p>Accuracy and F1 score across experiment runs (%)</p></div></div><div className="chart-container"><ResponsiveContainer width="100%" height="100%"><BarChart data={chart} margin={{ top: 15, right: 12, left: 0, bottom: 85 }} accessibilityLayer><CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e7ece9" /><XAxis dataKey="run" angle={-30} textAnchor="end" height={90} interval={0} tick={{ fontSize: 10, fill: '#63736b' }} tickFormatter={(value: string) => value.replaceAll('_', ' ')} /><YAxis domain={[0, 100]} tick={{ fontSize: 12 }} unit="%" width={48} /><Tooltip formatter={(value) => `${Number(value).toFixed(2)}%`} contentStyle={{ borderRadius: 12, borderColor: '#e1e9e4' }} /><Legend verticalAlign="top" height={34} /><Bar dataKey="Accuracy" fill="#207958" radius={[4, 4, 0, 0]} maxBarSize={36} /><Bar dataKey="F1" fill="#96c9ad" radius={[4, 4, 0, 0]} maxBarSize={36} /></BarChart></ResponsiveContainer></div></section>}
      <section className="figure-section"><div className="section-title-row"><div><p className="eyebrow">A CLOSER LOOK</p><h2>Evaluation gallery</h2><p>Inspect the confusion matrix, ROC, and training curves for a run.</p></div>{Object.keys(data.figures).length > 0 && <label className="run-select">Model run<select value={selected} onChange={(event) => setSelected(event.target.value)}>{Object.keys(data.figures).map((run) => <option key={run} value={run}>{run}</option>)}</select></label>}</div>{figures ? <div className="xai-grid"><Figure key={`${selected}-confusion`} src={figures.confusion} title="Confusion matrix" caption="Predicted versus true class labels." /><Figure key={`${selected}-roc`} src={figures.roc} title="ROC curve" caption="Discrimination across decision thresholds." /><Figure key={`${selected}-curves`} src={figures.curves} title="Training curves" caption="Learning progress across training epochs." /></div> : <div className="card"><EmptyState title="No evaluation figures yet">Run evaluation to make the per-model figures available.</EmptyState></div>}</section>
    </>}
    {error && <ErrorToast message={error} onDismiss={() => setError('')} />}
  </>
}
