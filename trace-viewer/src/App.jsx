import { Routes, Route, BrowserRouter, Navigate, useLocation } from 'react-router-dom';
import './App.css'
import TraceViewer from './TraceViewer';

const DEFAULT  = '?animate=1&mode=audience&session=myclass&trace=var/traces/lecture_linear_classification.json';
const PRESENTER = '?trace=var%2Ftraces%2Flecture_linear_classification.json&step=0&animate=1&mode=presenter&session=myclass&key=mypin';
const AUDIENCE_KEYS = ['mode', 'animate', 'session', 'trace'];
const SESSION_KEY = 'audienceParams';

function Root() {
  const { search } = useLocation();

  // No params at all → redirect to default audience URL
  if (!search || search === '?') {
    return <Navigate to={DEFAULT} replace />;
  }

  const params = new URLSearchParams(search);

  // Short presenter URL: ?p → full presenter params
  if (params.has('p')) {
    return <Navigate to={PRESENTER} replace />;
  }

  const mode = params.get('mode');

  if (mode === 'audience') {
    const allPresent = AUDIENCE_KEYS.every(k => params.get(k));

    if (allPresent) {
      // All required params intact — save them so we can restore after a reload
      const snapshot = {};
      AUDIENCE_KEYS.forEach(k => { snapshot[k] = params.get(k); });
      sessionStorage.setItem(SESSION_KEY, JSON.stringify(snapshot));
    } else {
      // Some required params were removed — restore from snapshot or fall back to DEFAULT
      const saved = sessionStorage.getItem(SESSION_KEY);
      const base = saved ? JSON.parse(saved) : null;
      const restored = new URLSearchParams(search);
      AUDIENCE_KEYS.forEach(k => {
        if (!restored.get(k)) restored.set(k, base?.[k] || new URLSearchParams(DEFAULT).get(k));
      });
      return <Navigate to={`?${restored.toString()}`} replace />;
    }
  } else if (!mode) {
    // mode param was removed entirely — restore from snapshot if we were audience
    const saved = sessionStorage.getItem(SESSION_KEY);
    if (saved) {
      const base = JSON.parse(saved);
      const restored = new URLSearchParams(search);
      for (const [k, v] of Object.entries(base)) {
        if (restored.get(k) !== v) restored.set(k, v);
      }
      return <Navigate to={`?${restored.toString()}`} replace />;
    }
  }
  // mode=presenter or any other explicit mode — don't interfere

  return <TraceViewer />;
}

function App() {
  return (
    <BrowserRouter basename={process.env.NODE_ENV === 'production' ? '/linear-classifiers/' : '/'}>
      <Routes>
        <Route path="/" element={<Root />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
