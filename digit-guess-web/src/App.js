import React, { useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import CanvasView from './components/CanvasView';
import GenerationView from './components/GenerationView';

function App() {
  const [activeView, setActiveView] = useState('canvas'); // 'canvas' or 'generation'

  return (
    <div className="App">
      <Sidebar activeView={activeView} setActiveView={setActiveView} />
      <main className="main-content">
        {activeView === 'canvas' && <CanvasView />}
        {activeView === 'generation' && <GenerationView />}
      </main>
    </div>
  );
}

export default App; 