import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AnimatePresence, motion } from 'framer-motion'
import {
  Activity, AlertTriangle, ArrowUp, BarChart3, Check, ChevronRight, CircleAlert, FileText, FolderOpen,
  History, Layers3, Lightbulb, LoaderCircle, LogOut, Paperclip, Plus, Search, Sparkles, TrendingUp, Trash2, Upload, Waves, X,
} from 'lucide-react'
import { analyzeTable, askQuestion, deleteDocument, fetchCurrentUser, fetchDocuments, loginAccount, logoutAccount, registerAccount } from './lib/api'
import type { AgentResponse, AnalyticsResponse, RagDocument, RetrievedContext } from './lib/types'
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, Pie, PieChart, Cell } from 'recharts'
import './App.css'

const suggestions = ['Find the key deadlines', 'Summarize the indexed knowledge', 'Compare the uploaded sources']
const sessionStorageKey = 'synapse-rag-sessions'

type SavedSession = {
  id: string
  title: string
  turns: SessionTurn[]
  updatedAt: string
}

type SessionTurn = {
  id: string
  query: string
  response: AgentResponse
  createdAt: string
}

const loadSessions = (): SavedSession[] => {
  try {
    const stored = window.localStorage.getItem(sessionStorageKey)
    if (!stored) return []
    const parsed = JSON.parse(stored) as Array<SavedSession & { query?: string; response?: AgentResponse }>
    return parsed.map((session) => session.turns ? session : {
      id: session.id,
      title: session.title,
      turns: session.response ? [{ id: crypto.randomUUID(), query: session.query ?? session.title, response: session.response, createdAt: session.updatedAt }] : [],
      updatedAt: session.updatedAt,
    })
  } catch {
    return []
  }
}

function App() {
  const [query, setQuery] = useState('')
  const [files, setFiles] = useState<File[]>([])
  const [response, setResponse] = useState<AgentResponse | null>(null)
  const [sessions, setSessions] = useState<SavedSession[]>(loadSessions)
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [turns, setTurns] = useState<SessionTurn[]>([])
  const [selectedContext, setSelectedContext] = useState<RetrievedContext | null>(null)
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const analyticsInputRef = useRef<HTMLInputElement>(null)
  const [analytics, setAnalytics] = useState<AnalyticsResponse | null>(null)
  const [analyticsFile, setAnalyticsFile] = useState<File | null>(null)
  const [selectedFields, setSelectedFields] = useState({ dateColumn: '', measureColumn: '', categoryColumn: '' })
  const [dateRange, setDateRange] = useState('All periods')
  const queryClient = useQueryClient()
  const currentUser = useQuery({ queryKey: ['current-user'], queryFn: fetchCurrentUser, retry: false })
  const [authMode, setAuthMode] = useState<'login' | 'register'>('login')
  const auth = useMutation({ mutationFn: (payload: { mode: 'login' | 'register'; name: string; email: string; password: string }) => payload.mode === 'login' ? loginAccount(payload.email, payload.password) : registerAccount(payload.name, payload.email, payload.password), onSuccess: (user) => queryClient.setQueryData(['current-user'], user) })
  const documents = useQuery({ queryKey: ['documents'], queryFn: fetchDocuments })
  const ask = useMutation({
    mutationFn: ({ submittedQuery, submittedFiles }: { submittedQuery: string; submittedFiles: File[] }) => askQuestion(submittedQuery, submittedFiles),
    onSuccess: (data, variables) => {
      const turn: SessionTurn = { id: crypto.randomUUID(), query: variables.submittedQuery, response: data, createdAt: new Date().toISOString() }
      setSessions((current) => {
        const sessionId = activeSessionId ?? crypto.randomUUID()
        const existing = current.find((session) => session.id === sessionId)
        const updated: SavedSession = existing
          ? { ...existing, turns: [...existing.turns, turn], updatedAt: turn.createdAt }
          : { id: sessionId, title: variables.submittedQuery.slice(0, 48) || 'Untitled analysis', turns: [turn], updatedAt: turn.createdAt }
        setActiveSessionId(sessionId)
        const next = [updated, ...current.filter((session) => session.id !== sessionId)].slice(0, 20)
        window.localStorage.setItem(sessionStorageKey, JSON.stringify(next))
        return next
      })
      setTurns((current) => [...current, turn])
      setResponse(data)
      setQuery('')
      setFiles([])
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })
  const remove = useMutation({ mutationFn: deleteDocument, onSuccess: () => queryClient.invalidateQueries({ queryKey: ['documents'] }) })
  const analyze = useMutation({ mutationFn: ({ file, fields }: { file: File; fields?: typeof selectedFields }) => analyzeTable(file, fields), onSuccess: (data) => { setAnalytics(data); setSelectedFields({ dateColumn: data.date_column ?? '', measureColumn: data.measure_column ?? '', categoryColumn: data.category_column ?? '' }) } })
  const submit = () => { if ((query.trim() || files.length) && !ask.isPending) ask.mutate({ submittedQuery: query.trim(), submittedFiles: files }) }
  const startNewSession = () => { setQuery(''); setFiles([]); setResponse(null); setTurns([]); setActiveSessionId(null); setSelectedContext(null); ask.reset() }
  const openSession = (session: SavedSession) => { setQuery(''); setFiles([]); setTurns(session.turns); setActiveSessionId(session.id); setResponse(session.turns.at(-1)?.response ?? null); setSelectedContext(null); ask.reset() }
  const addFiles = (incoming: FileList | File[]) => setFiles((current) => [...current, ...Array.from(incoming)].filter((file, index, all) => all.findIndex((candidate) => candidate.name === file.name && candidate.size === file.size) === index))
  const totalChunks = documents.data?.reduce((sum, document) => sum + document.chunk_count, 0) ?? 0
  const confidence = response?.retrieved_contexts.length ? Math.min(98, Math.round(response.retrieved_contexts[0].score * 100)) : 0

  if (currentUser.isLoading) return <div className="auth-loading"><LoaderCircle className="spin" size={22} /> Restoring your workspace...</div>
  if (!currentUser.data) return <AuthScreen mode={authMode} onModeChange={(mode) => { setAuthMode(mode); auth.reset() }} error={auth.error?.message} pending={auth.isPending} onSubmit={(payload) => auth.mutate({ ...payload, mode: authMode })} />
  const user = currentUser.data

  return (
    <main className="app-shell">
      <div className="ambient ambient-one" /><div className="ambient ambient-two" />
      <header className="topbar">
        <div className="brand-lockup"><div className="brand-mark"><Waves size={20} /></div><div><span className="overline">Knowledge interface</span><strong>Synapse<span>/</span>RAG</strong></div></div>
        <div className="top-actions"><div className="system-status"><span className="pulse-dot" /> Index online</div><button className="icon-button" title="Search knowledge base" onClick={() => document.querySelector<HTMLTextAreaElement>('.composer textarea')?.focus()}><Search size={18} /></button><button className="account-menu" title="Sign out" onClick={() => logoutAccount().then(() => queryClient.removeQueries({ queryKey: ['current-user'] }))}><span className="avatar">{user.name.slice(0, 2).toUpperCase()}</span><LogOut size={13} /></button></div>
      </header>

      <section className="hero-strip"><div><div className="eyebrow"><Sparkles size={14} /> Analyst workspace</div><h1>Turn raw data into <em>decisions.</em></h1><p>Upload a table to inspect its schema, compare performance, and find the signals worth acting on.</p></div><div className="hero-actions"><div className="hero-metrics"><Metric label="Documents" value={String(documents.data?.length ?? 0).padStart(2, '0')} icon={<FolderOpen size={16} />} /><Metric label="Indexed chunks" value={String(totalChunks).padStart(2, '0')} icon={<Layers3 size={16} />} /><Metric label="Engine" value="HYBRID" icon={<Activity size={16} />} /></div><input ref={analyticsInputRef} type="file" hidden accept=".csv,.xlsx,.xlsm" onChange={(event) => { const file = event.target.files?.[0]; if (file) { analyze.reset(); setAnalyticsFile(file); analyze.mutate({ file }) } event.target.value = '' }} /><button className="insight-button" onClick={() => analyticsInputRef.current?.click()}><BarChart3 size={16} /> {analyze.isPending ? 'Reading table...' : analytics ? 'Analyze another table' : 'Analyze a table'}</button>{analyze.isError && <div className="analytics-error"><CircleAlert size={14} /> {analyze.error.message}</div>}</div></section>

      {analytics && <><AnalyticsBoard analytics={analytics} dateRange={dateRange} onDateRangeChange={setDateRange} selectedFields={selectedFields} onFieldsChange={(fields) => { setSelectedFields(fields); if (analyticsFile) analyze.mutate({ file: analyticsFile, fields }) }} onExport={() => downloadAnalytics(analytics)} /><VisualSummary analytics={analytics} /></>}

      <section className="workspace-grid">
        <aside className="library-panel panel-surface">
          <div className="panel-heading"><div><span className="section-kicker">01 / Library</span><h2>Source atlas</h2></div><button className="round-button" title="Add documents" onClick={() => inputRef.current?.click()}><Plus size={17} /></button></div>
          <div className={`drop-zone ${isDragging ? 'is-dragging' : ''}`} onDragOver={(event) => { event.preventDefault(); setIsDragging(true) }} onDragLeave={() => setIsDragging(false)} onDrop={(event) => { event.preventDefault(); setIsDragging(false); addFiles(event.dataTransfer.files) }} onClick={() => inputRef.current?.click()}><div className="drop-icon"><Upload size={18} /></div><strong>Drop sources here</strong><span>PDF, audio, image or text</span></div>
          <input ref={inputRef} type="file" multiple hidden accept=".txt,.md,.csv,.json,.jpg,.jpeg,.png,.pdf,.mp3,.wav,.m4a" onChange={(event) => event.target.files && addFiles(event.target.files)} />
          <div className="session-list"><div className="session-list-heading"><div className="list-label"><History size={12} /> Saved sessions</div><button className="text-button" onClick={startNewSession}>New</button></div>{sessions.length === 0 && <div className="empty-state">Completed analyses will appear here.</div>}{sessions.map((session) => <SessionRow key={session.id} session={session} onOpen={() => openSession(session)} />)}</div>
          <div className="library-list"><div className="list-label">Indexed now</div>{documents.isLoading && <div className="empty-state"><LoaderCircle className="spin" size={17} /> Loading atlas</div>}{!documents.isLoading && !documents.data?.length && <div className="empty-state">Your first source starts here.</div>}{documents.data?.map((document, index) => <DocumentRow key={document.document_id} document={document} index={index} onDelete={() => remove.mutate(document.document_id)} />)}</div>
        </aside>

        <section className="conversation-panel">
          <div className="conversation-header"><div><span className="section-kicker">02 / Dialogue</span><h2>Research session</h2></div><span className="live-tag"><span className="pulse-dot" /> Live</span></div>
          <div className="conversation-feed"><motion.div className="welcome-card" initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .5 }}><div className="welcome-orbit"><Sparkles size={22} /></div><div><span className="message-label">SYNAPSE / READY</span><h3>{turns.length ? `${turns.length} turns in this research thread.` : 'Your documents are a constellation.'}</h3><p>{turns.length ? 'Continue the investigation and keep every answer connected to its source.' : 'Choose a prompt below or ask a question to trace an answer through your indexed sources.'}</p></div></motion.div>
            <AnimatePresence>{turns.map((turn) => <motion.div className="turn-group" key={turn.id} initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }}><div className="question-card"><span className="message-label">QUESTION</span><p>{turn.query}</p></div><div className="answer-card"><div className="message-label">ANSWER / GROUNDED</div><p className="answer-text">{turn.response.final_output}</p><div className="answer-footer"><span><Check size={14} /> {turn.response.retrieved_contexts.length} sources connected</span><span>{turn.response.cost_estimate.estimated_input_tokens} input tokens</span></div><PerspectiveGrid perspectives={turn.response.perspectives} /></div></motion.div>)}{ask.isPending && <motion.div className="thinking-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }}><LoaderCircle className="spin" size={16} /> Tracing your knowledge graph...</motion.div>}{ask.isError && <div className="error-line"><CircleAlert size={16} /> {ask.error.message}</div>}</AnimatePresence>
          </div>
          <div className="composer-wrap"><div className="suggestion-row">{suggestions.map((suggestion) => <button key={suggestion} className="suggestion" onClick={() => setQuery(suggestion)}>{suggestion}<ChevronRight size={13} /></button>)}</div>{files.length > 0 && <div className="file-pills">{files.map((file) => <span className="file-pill" key={`${file.name}-${file.size}`}><FileText size={14} /> {file.name}<button onClick={() => setFiles((current) => current.filter((item) => item !== file))}><X size={13} /></button></span>)}</div>}<div className="composer"><button className="composer-icon" title="Attach sources" onClick={() => inputRef.current?.click()}><Paperclip size={19} /></button><textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); submit() } }} placeholder="Ask anything across your knowledge..." rows={1} /><button className="send-button" title="Send question" onClick={submit} disabled={ask.isPending || (!query.trim() && !files.length)}>{ask.isPending ? <LoaderCircle className="spin" size={19} /> : <ArrowUp size={19} />}</button></div><div className="composer-hint"><span>Press <kbd>Enter</kbd> to send</span><span>Answers are source-grounded</span></div></div>
        </section>

        <aside className="evidence-panel panel-surface"><div className="panel-heading"><div><span className="section-kicker">03 / Evidence</span><h2>Signal inspector</h2></div><span className="confidence-badge">{confidence || '--'}% <span>match</span></span></div><div className="confidence-visual"><div className="confidence-ring" style={{ '--score': `${confidence}%` } as React.CSSProperties}><strong>{confidence || '--'}</strong><span>confidence</span></div><div className="confidence-copy"><strong>{response ? 'Evidence aligned' : 'Awaiting a query'}</strong><span>{response ? 'The strongest passages are ready to inspect.' : 'Ask a question to reveal source signals.'}</span></div></div><div className="evidence-list"><div className="list-label">Retrieved passages</div>{response?.retrieved_contexts.map((context, index) => <button className={`evidence-card ${selectedContext?.chunk_id === context.chunk_id ? 'selected' : ''}`} key={context.chunk_id} onClick={() => setSelectedContext(context)}><div className="evidence-top"><span className="citation-number">0{index + 1}</span><span>{context.filename}{context.page_number ? ` / p.${context.page_number}` : ''}</span><strong>{Math.round(context.score * 100)}%</strong></div><p>{context.text}</p></button>)}{!response?.retrieved_contexts.length && <div className="empty-state">Retrieved passages will appear here.</div>}</div>{selectedContext && <motion.div className="selected-source" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}><span className="list-label">Selected source</span><strong>{selectedContext.filename}</strong><p>{selectedContext.text}</p></motion.div>}</aside>
      </section>
      <footer className="bottom-line"><span><span className="pulse-dot" /> Chroma persistence active</span><span>Hash embeddings / hybrid retrieval / reranking</span><span>v0.4.0</span></footer>
    </main>
  )
}

function AuthScreen({ mode, onModeChange, onSubmit, error, pending }: { mode: 'login' | 'register'; onModeChange: (mode: 'login' | 'register') => void; onSubmit: (payload: { name: string; email: string; password: string }) => void; error?: string; pending: boolean }) {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  return <main className="auth-shell"><div className="auth-orbit auth-orbit-one" /><div className="auth-orbit auth-orbit-two" /><section className="auth-card"><div className="auth-brand"><div className="brand-mark"><Waves size={20} /></div><div><span className="overline">Knowledge interface</span><strong>Synapse<span>/</span>RAG</strong></div></div><div className="auth-copy"><span className="eyebrow"><Sparkles size={14} /> Evidence-first intelligence</span><h1>{mode === 'login' ? 'Welcome back to your workspace.' : 'Build a clearer view of your business.'}</h1><p>{mode === 'login' ? 'Continue exploring your documents, data, and saved analysis sessions.' : 'Create an account to turn source material into decisions your team can trust.'}</p></div><form className="auth-form" onSubmit={(event) => { event.preventDefault(); onSubmit({ name, email, password }) }}>{mode === 'register' && <label>Full name<input value={name} onChange={(event) => setName(event.target.value)} placeholder="Alex Morgan" autoComplete="name" required minLength={2} /></label>}<label>Email address<input value={email} onChange={(event) => setEmail(event.target.value)} type="email" placeholder="you@company.com" autoComplete="email" required /></label><label>Password<input value={password} onChange={(event) => setPassword(event.target.value)} type="password" placeholder="At least 8 characters" autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength={8} /></label>{error && <div className="auth-error"><CircleAlert size={15} /> {formatAuthError(error)}</div>}<button className="auth-submit" disabled={pending}>{pending ? <LoaderCircle className="spin" size={18} /> : mode === 'login' ? 'Enter workspace' : 'Create account'}</button></form><div className="auth-switch">{mode === 'login' ? 'New to Synapse/RAG?' : 'Already have an account?'} <button type="button" onClick={() => onModeChange(mode === 'login' ? 'register' : 'login')}>{mode === 'login' ? 'Create an account' : 'Sign in'}</button></div><small className="auth-note">Your files remain source-grounded and your workspace is protected by a private session.</small></section></main>
}
function formatAuthError(error: string) { try { const parsed = JSON.parse(error) as { detail?: string }; return parsed.detail ?? error } catch { return error } }
function Metric({ label, value, icon }: { label: string; value: string; icon: React.ReactNode }) { return <div className="metric"><span>{icon}</span><div><small>{label}</small><strong>{value}</strong></div></div> }
function PerspectiveGrid({ perspectives }: { perspectives: AgentResponse['perspectives'] }) { return perspectives.length > 0 ? <div className="perspective-grid">{perspectives.map((perspective) => <article className={`perspective-card perspective-${perspective.key}`} key={perspective.key}><div className="perspective-heading"><strong>{perspective.title}</strong><span>{Math.round(perspective.confidence * 100)}%</span></div><p>{perspective.summary}</p>{perspective.sources.length > 0 && <footer>{perspective.sources.join(' · ')}</footer>}</article>)}</div> : null }
function VisualSummary({ analytics }: { analytics: AnalyticsResponse }) { const colors = ['#78a7f5', '#d9ad69', '#91b6a3', '#b18bd4', '#e88f8b', '#8d9aab']; const perspectives = analytics.insights.map((insight, index) => ({ label: `Insight ${index + 1}`, value: Math.round(insight.confidence * 100) })); return <div className="visual-summary"><ChartPanel title="Value distribution" icon={<BarChart3 size={15} />}><ResponsiveContainer width="100%" height={200}><BarChart data={analytics.distribution}><CartesianGrid stroke="rgba(166,183,204,.14)" vertical={false} /><XAxis dataKey="label" stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={9} /><YAxis stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={9} /><Tooltip contentStyle={{ background: '#182332', border: '1px solid rgba(166,183,204,.32)', color: '#edf2f8' }} /><Bar dataKey="value" fill="#78a7f5" radius={[2, 2, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel><ChartPanel title="Category share" icon={<BarChart3 size={15} />}><ResponsiveContainer width="100%" height={200}><PieChart><Pie data={analytics.categories} dataKey="value" nameKey="label" innerRadius={48} outerRadius={76} paddingAngle={3}>{analytics.categories.map((category, index) => <Cell key={category.label} fill={colors[index % colors.length]} />)}</Pie><Tooltip contentStyle={{ background: '#182332', border: '1px solid rgba(166,183,204,.32)', color: '#edf2f8' }} /></PieChart></ResponsiveContainer></ChartPanel><ChartPanel title="Insight confidence" icon={<Lightbulb size={15} />}><ResponsiveContainer width="100%" height={200}><BarChart data={perspectives} layout="vertical"><CartesianGrid stroke="rgba(166,183,204,.14)" horizontal={false} /><XAxis type="number" domain={[0, 100]} stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={9} /><YAxis dataKey="label" type="category" width={54} stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={9} /><Bar dataKey="value" fill="#d9ad69" radius={[0, 2, 2, 0]} /></BarChart></ResponsiveContainer></ChartPanel></div> }
function AnalyticsBoard({ analytics, dateRange, onDateRangeChange, selectedFields, onFieldsChange, onExport }: { analytics: AnalyticsResponse; dateRange: string; onDateRangeChange: (range: string) => void; selectedFields: { dateColumn: string; measureColumn: string; categoryColumn: string }; onFieldsChange: (fields: { dateColumn: string; measureColumn: string; categoryColumn: string }) => void; onExport: () => void }) {
  const series = dateRange === '6 periods' ? analytics.trend.slice(-6) : dateRange === '3 periods' ? analytics.trend.slice(-3) : analytics.trend
  return <section className="analytics-board"><div className="analytics-heading"><div><span className="section-kicker">04 / Intelligence</span><h2>Insight board</h2><p>{analytics.filename} · {analytics.row_count.toLocaleString()} rows · {analytics.measure_column ?? 'No measure detected'}</p></div><div className="analytics-controls"><button className="export-button" onClick={onExport}>Export JSON</button><span className="control-label">Time window</span><select value={dateRange} onChange={(event) => onDateRangeChange(event.target.value)} aria-label="Date range"><option>All periods</option><option>12 periods</option><option>6 periods</option><option>3 periods</option></select></div></div>{analytics.warnings.length > 0 && <div className="analytics-warning"><AlertTriangle size={16} /> {analytics.warnings.join(' ')}</div>}{analytics.status === 'completed' ? <><div className="schema-strip"><span className="schema-label">Configure fields</span><FieldSelect label="Date" value={selectedFields.dateColumn} options={analytics.columns.filter((column) => column.kind === 'date' || column.kind === 'text').map((column) => column.name)} onChange={(value) => onFieldsChange({ ...selectedFields, dateColumn: value })} /><FieldSelect label="Measure" value={selectedFields.measureColumn} options={analytics.columns.filter((column) => column.kind === 'number').map((column) => column.name)} onChange={(value) => onFieldsChange({ ...selectedFields, measureColumn: value })} /><FieldSelect label="Category" value={selectedFields.categoryColumn} options={analytics.columns.filter((column) => column.kind === 'category' || column.kind === 'text').map((column) => column.name)} onChange={(value) => onFieldsChange({ ...selectedFields, categoryColumn: value })} /></div><div className="kpi-grid"><KpiCard label="Total value" value={formatMetric(analytics.kpis.total)} /><KpiCard label="Average" value={formatMetric(analytics.kpis.average)} /><KpiCard label="High point" value={formatMetric(analytics.kpis.maximum)} accent /><KpiCard label="Anomalies" value={String(analytics.anomalies.length)} /></div><div className="chart-grid"><ChartPanel title="Performance trend" icon={<TrendingUp size={15} />}><ResponsiveContainer width="100%" height={220}><LineChart data={series}><CartesianGrid stroke="rgba(166,183,204,.14)" vertical={false} /><XAxis dataKey="label" stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={10} /><YAxis stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={10} /><Tooltip contentStyle={{ background: '#182332', border: '1px solid rgba(166,183,204,.32)', color: '#edf2f8' }} /><Line type="monotone" dataKey="value" stroke="#78a7f5" strokeWidth={2} dot={{ r: 3, fill: '#78a7f5' }} /></LineChart></ResponsiveContainer></ChartPanel><ChartPanel title={analytics.category_column ? `By ${analytics.category_column}` : 'Category comparison'} icon={<BarChart3 size={15} />}><ResponsiveContainer width="100%" height={220}><BarChart data={analytics.categories}><CartesianGrid stroke="rgba(166,183,204,.14)" vertical={false} /><XAxis dataKey="label" stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={10} /><YAxis stroke="#8d9aab" tickLine={false} axisLine={false} fontSize={10} /><Tooltip contentStyle={{ background: '#182332', border: '1px solid rgba(166,183,204,.32)', color: '#edf2f8' }} /><Bar dataKey="value" fill="#d9ad69" radius={[2, 2, 0, 0]} /></BarChart></ResponsiveContainer></ChartPanel></div><div className="insight-grid">{analytics.insights.map((insight) => <article className="insight-card" key={insight.finding}><div className="insight-card-top"><Lightbulb size={17} /><span>{Math.round(insight.confidence * 100)}% confidence</span></div><h3>{insight.finding}</h3><p><strong>Business impact</strong>{insight.impact}</p><p><strong>Recommended action</strong>{insight.action}</p><footer>Source citation: {insight.source}</footer></article>)}</div></> : <div className="analytics-empty"><BarChart3 size={18} /> Upload a table with at least one numeric column to generate the board.</div>}</section>
}
function FieldSelect({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (value: string) => void }) { return <label className="field-select"><span>{label}</span><select value={value} onChange={(event) => onChange(event.target.value)}><option value="">None</option>{options.map((option) => <option key={option}>{option}</option>)}</select></label> }
function downloadAnalytics(analytics: AnalyticsResponse) { const blob = new Blob([JSON.stringify(analytics, null, 2)], { type: 'application/json' }); const url = URL.createObjectURL(blob); const anchor = document.createElement('a'); anchor.href = url; anchor.download = `${analytics.filename.replace(/\.[^.]+$/, '')}-insights.json`; anchor.click(); URL.revokeObjectURL(url) }
function KpiCard({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) { return <div className={`kpi-card ${accent ? 'accent' : ''}`}><span>{label}</span><strong>{value}</strong></div> }
function ChartPanel({ title, icon, children }: { title: string; icon: React.ReactNode; children: React.ReactNode }) { return <div className="chart-panel"><div className="chart-title">{icon}<strong>{title}</strong></div>{children}</div> }
function formatMetric(value: number) { return new Intl.NumberFormat(undefined, { maximumFractionDigits: 1, notation: Math.abs(value) >= 1000000 ? 'compact' : 'standard' }).format(value) }
function SessionRow({ session, onOpen }: { session: SavedSession; onOpen: () => void }) { return <button className="session-row" onClick={onOpen}><span className="session-icon"><History size={13} /></span><span className="session-info"><strong title={session.title}>{session.title}</strong><span>{new Date(session.updatedAt).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })} / {session.turns.length} turns</span></span><ChevronRight size={13} /></button> }
function DocumentRow({ document, index, onDelete }: { document: RagDocument; index: number; onDelete: () => void }) { return <motion.div className="document-row" initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: index * .06 }}><div className="document-icon"><FileText size={16} /></div><div className="document-info"><strong title={document.filename}>{document.filename}</strong><span>{document.chunk_count} chunks / {document.source_type}</span></div><button className="delete-button" title={`Remove ${document.filename}`} onClick={onDelete}><Trash2 size={14} /></button></motion.div> }

export default App
