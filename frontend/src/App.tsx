import { lazy, Suspense, useState } from 'react'
import { Icon, Spinner } from './components/UI'
import Diagnose from './pages/Diagnose'
import About from './pages/About'

const Results = lazy(() => import('./pages/Results'))

const tabs = ['Diagnose', 'Results', 'About'] as const
type Tab = typeof tabs[number]
export default function App() {
  const [tab, setTab] = useState<Tab>('Diagnose')
  return <div className="app-shell">
    <a className="skip-link" href="#main">Skip to content</a>
    <header className="site-header"><div className="header-inner">
      <a className="brand" href="#" onClick={(event) => { event.preventDefault(); setTab('Diagnose') }}><span className="brand-mark"><Icon size={27} /></span><span><strong>LSD-XAI<span className="brand-dot">.</span></strong><span className="brand-subtitle">Hybrid CNN–Transformer for Lumpy Skin Disease</span></span></a>
      <nav aria-label="Main navigation">{tabs.map((item) => <button key={item} className={`nav-item ${tab === item ? 'active' : ''}`} aria-current={tab === item ? 'page' : undefined} onClick={() => setTab(item)}><Icon name={item === 'Diagnose' ? 'scan' : item === 'Results' ? 'chart' : 'info'} size={17} />{item}</button>)}</nav>
    </div></header>
    <main id="main" className="main-content">
      <div hidden={tab !== 'Diagnose'}><Diagnose /></div>
      {tab === 'Results' && <Suspense fallback={<div className="card results-loading"><Spinner label="Loading results dashboard…" /></div>}><Results /></Suspense>}
      {tab === 'About' && <About />}
    </main>
    <footer className="site-footer"><span><span className="status-dot" />Built for explainable cattle health research</span><span>Research prototype, not a veterinary diagnostic tool</span></footer>
  </div>
}
