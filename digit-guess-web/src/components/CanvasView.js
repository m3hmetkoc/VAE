import React, { useRef, useState, useCallback, useEffect } from 'react';
import './CanvasView.css';

const CanvasView = () => {
  const canvasRef = useRef(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [lastDrawTime, setLastDrawTime] = useState(null);
  const predictionTimeoutRef = useRef(null);

  // Canvas dimensions
  const CANVAS_WIDTH = 28;
  const CANVAS_HEIGHT = 28;
  const SCALE = 20; // Scale factor to make the 28x28 canvas visible

  // Initialize canvas
  React.useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = CANVAS_WIDTH;
    canvas.height = CANVAS_HEIGHT;
    
    // Fill with black background
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    
    // Set drawing properties for MNIST-style strokes
    ctx.strokeStyle = '#FFFFFF';
    ctx.lineWidth = 2; // Slightly thicker for better visibility
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalCompositeOperation = 'source-over';
  }, []);

  // Get mouse position relative to canvas
  const getMousePos = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const scaleX = CANVAS_WIDTH / rect.width;
    const scaleY = CANVAS_HEIGHT / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  }, []);

  // Start drawing
  const startDrawing = useCallback((e) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const pos = getMousePos(e);
    
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  }, [getMousePos]);

  // Draw
  const draw = useCallback((e) => {
    if (!isDrawing) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const pos = getMousePos(e);
    
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  }, [isDrawing, getMousePos]);

  // Stop drawing
  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
    // Trigger auto-prediction after stopping drawing
    setLastDrawTime(Date.now());
  }, []);

  // Clear canvas
  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    setPrediction(null);
    setError(null);
    setLastDrawTime(null);
    // Clear any pending prediction
    if (predictionTimeoutRef.current) {
      clearTimeout(predictionTimeoutRef.current);
      predictionTimeoutRef.current = null;
    }
  };

  // Get canvas data with MNIST-style preprocessing
  const getCanvasData = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    const data = imageData.data;
    
    // Convert to grayscale array (0-255)
    let grayscaleArray = [];
    for (let i = 0; i < data.length; i += 4) {
      // Use red channel (since we're only using black/white)
      grayscaleArray.push(data[i]);
    }
    
    return grayscaleArray;
  };

  // Check if canvas has any drawing
  const hasDrawing = () => {
    const canvasData = getCanvasData();
    return canvasData.some(pixel => pixel > 0);
  };

  // Predict digit using the backend API
  const predictDigit = async (isAutomatic = false) => {
    // Don't predict if already loading or if there's no drawing
    if (isLoading || !hasDrawing()) {
      return;
    }

    setIsLoading(true);
    setError(null);
    
    try {
      const canvasData = getCanvasData();
      
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          canvas_data: canvasData
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Prediction failed');
      }
      
      const result = await response.json();
      setPrediction(result);
      
    } catch (err) {
      setError(err.message);
      console.error('Prediction error:', err);
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-prediction with debouncing
  useEffect(() => {
    if (lastDrawTime) {
      // Clear any existing timeout
      if (predictionTimeoutRef.current) {
        clearTimeout(predictionTimeoutRef.current);
      }
      
      // Set new timeout for auto-prediction
      predictionTimeoutRef.current = setTimeout(() => {
        predictDigit(true); // true indicates automatic prediction
      }, 500); // 500ms delay after stopping drawing
    }
    
    return () => {
      if (predictionTimeoutRef.current) {
        clearTimeout(predictionTimeoutRef.current);
      }
    };
  }, [lastDrawTime]);

  // Probability bar component
  const ProbabilityBar = ({ digit, probability, isMax }) => (
    <div className="probability-bar">
      <div className="digit-label">{digit}</div>
      <div className="bar-container">
        <div 
          className={`bar ${isMax ? 'bar-max' : ''}`}
          style={{ width: `${probability * 100}%` }}
        />
      </div>
      <div className="probability-value">
        {(probability * 100).toFixed(1)}%
      </div>
    </div>
  );

  return (
    <div className="canvas-view">
        <header className="view-header">
            <h1>MNIST Digit Drawer</h1>
            <p>Draw a digit in the 28x28 canvas below</p>
        </header>
        
        <div className="main-container">
          <div className="canvas-section">
            <div className="canvas-container">
              <canvas
                ref={canvasRef}
                style={{
                  width: `${CANVAS_WIDTH * SCALE}px`,
                  height: `${CANVAS_HEIGHT * SCALE}px`,
                  border: '2px solid #ccc',
                  cursor: 'crosshair',
                  imageRendering: 'pixelated'
                }}
                onMouseDown={startDrawing}
                onMouseMove={draw}
                onMouseUp={stopDrawing}
                onMouseLeave={stopDrawing}
              />
            </div>
            
            <div className="controls">
              <button onClick={clearCanvas} className="control-button">
                Clear Canvas
              </button>
              <button 
                onClick={() => predictDigit(false)} 
                className="control-button predict-button"
                disabled={isLoading}
              >
                {isLoading ? 'Predicting...' : 'Predict Now'}
              </button>
            </div>
            
            <div className="auto-prediction-info">
              <p>✨ Auto-prediction enabled - predictions update automatically as you draw!</p>
            </div>
          </div>

          <div className="prediction-section">
            {error && (
              <div className="error-message">
                <h3>Error</h3>
                <p>{error}</p>
                <p className="error-hint">
                  Make sure the Flask server is running: <code>python main/prediction_server.py --model-path path/to/your/classifier_model </code>
                </p>
              </div>
            )}

            {prediction && (
              <div className="prediction-results">
                <div className="prediction-header">
                  <h3>Prediction Results</h3>
                  <div className="main-prediction">
                    <span className="predicted-digit">{prediction.predicted_digit}</span>
                    <span className="confidence">
                      Confidence: {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
                
                <div className="probability-chart">
                  <h4>Probability Distribution</h4>
                  {prediction.probabilities.map((prob, digit) => (
                    <ProbabilityBar
                      key={digit}
                      digit={digit}
                      probability={prob}
                      isMax={digit === prediction.predicted_digit}
                    />
                  ))}
                </div>
              </div>
            )}

            {!prediction && !error && !isLoading && (
              <div className="placeholder">
                <h3>Start drawing a digit</h3>
                <p>The AI will automatically predict your digit as you draw! Predictions appear 500ms after you stop drawing.</p>
              </div>
            )}
          </div>
        </div>
        
        <div className="info">
          <p>Canvas Size: 28x28 pixels</p>
          <p>Background: Black (0) • Drawing: White (255)</p>
          <p>AI Model: Neural Network trained on MNIST dataset</p>
        </div>
    </div>
  );
}

export default CanvasView; 