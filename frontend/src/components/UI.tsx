import { useState } from 'react'
import type { ReactNode } from 'react'
import { assetUrl } from '../api'

export function Icon({ name = 'leaf', size = 22 }: { name?: 'leaf' | 'upload' | 'scan' | 'chart' | 'info' | 'arrow'; size?: number }) {
  const paths = {
    leaf: <><path d="M20 4c-9-1-16 2-16 9a6 6 0 0 0 6 6c7 0 10-7 10-15Z" /><path d="M4 21 15 10M9 16v-5m0 5h5" /></>,
    upload: <><path d="M12 16V3m-5 5 5-5 5 5M4 15v5a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5" /></>,
    scan: <><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5M3 12h18" /><circle cx="12" cy="12" r="4" /></>,
    chart: <><path d="M4 3v18h17M9 16v-5m5 5V7m5 9V4" /></>,
    info: <><circle cx="12" cy="12" r="9" /><path d="M12 11v6m0-10v.1" /></>,
    arrow: <path d="M4 12h16m-6-6 6 6-6 6" />,
  }
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[name]}</svg>
}
export function Spinner({ label = 'Loading…' }: { label?: string }) {
  return <span className="loading" role="status"><span className="spinner" />{label}</span>
}
export function EmptyState({ title, children }: { title: string; children: ReactNode }) {
  return <div className="empty-state"><span className="empty-icon"><Icon name="scan" size={30} /></span><h3>{title}</h3><p>{children}</p></div>
}
export function ErrorToast({ message, onDismiss }: { message: string; onDismiss: () => void }) {
  return <div className="error-toast" role="alert"><Icon name="info" /><span>{message}</span><button onClick={onDismiss} aria-label="Dismiss error">×</button></div>
}
export function Figure({ src, title, caption }: { src?: string | null; title: string; caption: string }) {
  const [failed, setFailed] = useState(false)
  return <figure className="figure-card"><div className="figure-image">{src && !failed ? <a href={assetUrl(src)} target="_blank" rel="noreferrer" aria-label={`Open ${title}`}><img src={assetUrl(src)} alt={title} onError={() => setFailed(true)} /></a> : <p className="muted">{failed ? 'Image could not be loaded.' : 'No figure available yet.'}</p>}</div><figcaption><h3>{title}</h3><p>{caption}</p></figcaption></figure>
}
export function PageHeading({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return <div className="page-heading"><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p>{children}</p></div>
}
