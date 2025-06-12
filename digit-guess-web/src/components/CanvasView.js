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
    // Apply MNIST-style preprocessing

    grayscaleArray = preprocessForMNIST(grayscaleArray);
    
    return grayscaleArray;
  };
  // Preprocess canvas data to match MNIST format

  const preprocessForMNIST = (pixelData) => {

    // Convert to 2D array for easier processing

    const image2D = [];

    for (let i = 0; i < 28; i++) {

      image2D.push(pixelData.slice(i * 28, (i + 1) * 28));

    }

    

    // Find bounding box of the drawn content

    let minX = 28, maxX = -1, minY = 28, maxY = -1;

    for (let y = 0; y < 28; y++) {

      for (let x = 0; x < 28; x++) {

        if (image2D[y][x] > 0) {

          minX = Math.min(minX, x);

          maxX = Math.max(maxX, x);

          minY = Math.min(minY, y);

          maxY = Math.max(maxY, y);

        }

      }

    }

    

    // If no drawing found, return original

    if (maxX === -1) {

      return pixelData;

    }

    

    // Calculate center of mass for better centering (MNIST style)

    let sumX = 0, sumY = 0, totalMass = 0;

    for (let y = minY; y <= maxY; y++) {

      for (let x = minX; x <= maxX; x++) {

        const intensity = image2D[y][x];

        sumX += x * intensity;

        sumY += y * intensity;

        totalMass += intensity;

      }

    }

    

    if (totalMass === 0) return pixelData;

    

    const centerX = sumX / totalMass;

    const centerY = sumY / totalMass;

    

    // Create new centered image

    const centered = Array(28 * 28).fill(0);

    const targetCenterX = 13.5; // Center of 28x28 grid

    const targetCenterY = 13.5;

    

    const offsetX = Math.round(targetCenterX - centerX);

    const offsetY = Math.round(targetCenterY - centerY);

    

    for (let y = 0; y < 28; y++) {

      for (let x = 0; x < 28; x++) {

        const srcX = x - offsetX;

        const srcY = y - offsetY;

        

        if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {

          centered[y * 28 + x] = image2D[srcY][srcX];

        }

      }

    }

    

    // Apply slight gaussian blur to match MNIST smoothing

    const blurred = applyGaussianBlur(centered);

    

    return blurred;

  };



  // Simple 3x3 Gaussian blur to match MNIST preprocessing

  const applyGaussianBlur = (data) => {

    const result = [...data];

    const kernel = [

      [1, 2, 1],

      [2, 4, 2], 

      [1, 2, 1]

    ];

    const kernelSum = 16;

    

    for (let y = 1; y < 27; y++) {

      for (let x = 1; x < 27; x++) {

        let sum = 0;

        for (let ky = -1; ky <= 1; ky++) {

          for (let kx = -1; kx <= 1; kx++) {

            const idx = (y + ky) * 28 + (x + kx);

            sum += data[idx] * kernel[ky + 1][kx + 1];

          }

        }

        result[y * 28 + x] = Math.round(sum / kernelSum);

      }

    }

    

    return result;

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
            </div>
            
            <div className="auto-prediction-info">
              <p>The AI will automatically predict your digit as you draw! Predictions appear 500ms after you stop drawing.</p>
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
    </div>
  );
}

export default CanvasView; 