import React, { useState, useEffect } from 'react';
import './GenerationView.css';

const GenerationView = () => {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('');
  const [label, setLabel] = useState('0');
  const [numSamples, setNumSamples] = useState(9);
  const [generatedImages, setGeneratedImages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Fashion-MNIST class labels
  const fashionMNISTLabels = [
    'T-shirt/top',
    'Trouser', 
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
  ];

  useEffect(() => {
    // Fetch models on component mount
    const fetchModels = async () => {
      setIsLoading(true);
      try {
        const response = await fetch('http://127.0.0.1:5000/models');
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        setModels(data);
        if (data.length > 0) {
          setSelectedModel(data[0].path);
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setIsLoading(false);
      }
    };
    fetchModels();
  }, []);

  const handleGenerate = async () => {
    if (!selectedModel) {
      setError('Please select a model first.');
      return;
    }
    setIsLoading(true);
    setError(null);
    setGeneratedImages([]);

    try {
      const response = await fetch('http://127.0.0.1:5000/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_path: selectedModel,
          num_samples: parseInt(numSamples, 10),
          label: label === 'any' ? null : parseInt(label, 10),
        }),
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to generate images.');
      }

      const data = await response.json();
      setGeneratedImages(data.images);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsLoading(false);
    }
  };
  
  const currentModel = models.find(m => m.path === selectedModel);
  const isVAEModel = currentModel && currentModel.type === 'VAE';
  const isCVAEModel = currentModel && currentModel.type === 'CVAE';
  
  // Check if the model is trained on Fashion-MNIST
  const isFashionMNIST = currentModel && (
    currentModel.name.toLowerCase().includes('fashion') ||
    currentModel.path.toLowerCase().includes('fashion')
  );

  return (
    <div className="generation-view">
      <header className="view-header">
        <h1>Image Generation</h1>
        <p>Generate digits using a trained VAE or CVAE model.</p>
      </header>
      
      <div className="generation-controls">
        <div className="control-group">
          <label htmlFor="model-select">Select Model:</label>
          <select 
            id="model-select" 
            value={selectedModel} 
            onChange={e => setSelectedModel(e.target.value)}
            disabled={isLoading || models.length === 0}
          >
            {models.map(model => (
              <option key={model.path} value={model.path}>
                {model.name} ({model.type})
              </option>
            ))}
          </select>
        </div>

        <div className={`control-group ${isVAEModel ? 'has-tooltip' : ''}`}>
          <label htmlFor="label-select">Class Label:</label>
          <select 
            id="label-select"
            value={label}
            onChange={e => setLabel(e.target.value)}
            disabled={isLoading || isVAEModel}
          >
            {isVAEModel && <option value="any">Any (VAE)</option>}
            {Array.from({ length: 10 }, (_, i) => (
              <option key={i} value={i}>
                {isFashionMNIST && isCVAEModel ? fashionMNISTLabels[i] : i}
              </option>
            ))}
          </select>
          {isVAEModel && (
            <div className="tooltip">
              Cannot select label when VAE selected
            </div>
          )}
        </div>

        <div className="control-group">
          <label htmlFor="samples-input">Number of Samples:</label>
          <input 
            type="number"
            id="samples-input"
            value={numSamples}
            onChange={e => setNumSamples(e.target.value)}
            min="1"
            max="25"
            disabled={isLoading}
          />
        </div>

        <button onClick={handleGenerate} className="generate-button" disabled={isLoading}>
          {isLoading ? 'Generating...' : 'Generate'}
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {generatedImages.length > 0 && (
        <div className="generated-images-grid">
          {generatedImages.map((imgSrc, index) => (
            <img key={index} src={imgSrc} alt={`Generated ${isFashionMNIST ? 'fashion item' : 'digit'} ${index}`} className="generated-image" />
          ))}
        </div>
      )}
    </div>
  );
};

export default GenerationView; 