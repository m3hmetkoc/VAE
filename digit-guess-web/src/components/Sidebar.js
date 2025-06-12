import React from 'react';
import './Sidebar.css';

const Sidebar = ({ activeView, setActiveView }) => {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <h2>Dashboard</h2>
      </div>
      <nav className="sidebar-nav">
        <button
          className={`nav-button ${activeView === 'canvas' ? 'active' : ''}`}
          onClick={() => setActiveView('canvas')}
        >
          Digit Predictor
        </button>
        <button
          className={`nav-button ${activeView === 'generation' ? 'active' : ''}`}
          onClick={() => setActiveView('generation')}
        >
          Image Generation
        </button>
      </nav>
    </div>
  );
};

export default Sidebar; 