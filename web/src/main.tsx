import React, { useEffect, useRef, useState } from 'react'
import { createRoot } from 'react-dom/client'
import * as monaco from 'monaco-editor'

function Editor() {
  const ref = useRef<HTMLDivElement | null>(null)
  const editorRef = useRef<monaco.editor.IStandaloneCodeEditor | null>(null)
  useEffect(() => {
    if (!ref.current) return
    editorRef.current = monaco.editor.create(ref.current, {
      value: '# Notes\n\nType here...',
      language: 'markdown',
      automaticLayout: true
    })
    return () => editorRef.current?.dispose()
  }, [])
  return <div ref={ref} style={{height: 300, border: '1px solid #ddd', borderRadius: 8}} />
}

function RAG() {
  const [q, setQ] = useState('What is in my notes?')
  const [answer, setAnswer] = useState('')
  const [importStatus, setImportStatus] = useState('')

  const doImport = async () => {
    setImportStatus('Importing...')
    const text = (window as any).sampleText || 'This is a sample document about vectors and matrices.'
    const r = await fetch('/api/import', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({title: 'Sample', text})
    })
    const data = await r.json()
    setImportStatus(`Imported doc ${data.document_id} with ${data.chunks} chunks`)
  }

  const ask = async () => {
    setAnswer('Thinking...')
    const r = await fetch('/api/rag/answer', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({question: q})
    })
    const data = await r.json()
    setAnswer(data.answer)
  }

  return (
    <div style={{display: 'grid', gap: 12}}>
      <div style={{display: 'flex', gap: 8}}>
        <button onClick={doImport}>Import Sample</button>
        <span>{importStatus}</span>
      </div>
      <input value={q} onChange={e=>setQ(e.target.value)} placeholder="Ask..." />
      <button onClick={ask}>Ask RAG</button>
      <pre style={{whiteSpace:'pre-wrap'}}>{answer}</pre>
    </div>
  )
}

function App() {
  return (
    <div style={{maxWidth: 900, margin: '40px auto', fontFamily: 'ui-sans-serif'}}>
      <h1>Dev Assistant (Local)</h1>
      <p>Ollama + pgvector + FastAPI + Monaco</p>
      <h2>Notebook</h2>
      <Editor />
      <h2>RAG</h2>
      <RAG />
    </div>
  )
}

const root = createRoot(document.getElementById('root')!)
root.render(<App />)
