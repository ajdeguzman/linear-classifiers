import { Routes, Route, BrowserRouter, Navigate, useLocation } from 'react-router-dom';
import './App.css'
import TraceViewer from './TraceViewer';

const DEFAULT = '?animate=1&mode=audience&session=myclass&trace=var/traces/lecture_linear_classification.json';

function Root() {
  const { search } = useLocation();
  if (!search || search === '?') {
    return <Navigate to={DEFAULT} replace />;
  }
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
