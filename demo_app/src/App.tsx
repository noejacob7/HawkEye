// src/App.tsx
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import SelectCarScene from './scenes/SelectCarScene';
import RadarMap from './components/RadarMap'; // Or wherever RadarMap lives

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<SelectCarScene />} />
        <Route path="/radar" element={<RadarMap />} />
      </Routes>
    </Router>
  );
}
