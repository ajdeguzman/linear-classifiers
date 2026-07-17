import { Routes, Route, BrowserRouter } from 'react-router-dom';
import './App.css'
import TraceViewer from './TraceViewer';

function App() {
  return (
    <BrowserRouter basename={process.env.NODE_ENV === 'production' ? '/linear-classifiers/' : '/'}>
      <Routes>
        <Route path="/" element={<TraceViewer />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;