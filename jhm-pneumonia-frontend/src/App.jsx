import { useState } from 'react';
import './App.css';

// 🔌 CONEXIÓN AUTOMÁTICA PROFESIONAL
// Si existe la variable en Vercel, la usa. Si estás en local, usa tu archivo .env.development.
const API_URL = import.meta.env.VITE_API_URL || 'https://jhm-pneumonia-api.onrender.com/predict';

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Concepto Industrial: Manejo del archivo binario
  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setPreview(URL.createObjectURL(file)); // Crea una URL temporal para previsualizar la foto
      setResult(null);
      setError(null);
    }
  };

  // Concepto Industrial: Petición HTTP asíncrona (AJAX)
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!image) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', image);

    fetch(API_URL, { method: 'POST', body: formData })
      .then((response) => {
        if (!response.ok) throw new Error('Error en el servidor. Código: ' + response.status);
        return response.json();
      })
      .then((data) => {
        setResult(data);
      })
      .catch((err) => {
        setError(err.message);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <div className="app-shell">
      <div className="card">
        <h1 className="card-title">JHM - Análisis de Neumonía</h1>
        <p className="card-subtitle">Sistema Clínico de Diagnóstico por IA</p>

        <form onSubmit={handleSubmit} className="form">
          {/* Zona de Selección de Imagen */}
          <div className="upload-zone">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              style={{ display: 'none' }}
              id="file-upload"
            />
            <label htmlFor="file-upload" className="upload-label">
              {preview ? '🔄 Cambiar radiografía' : '📁 Seleccionar Radiografía de Tórax'}
            </label>
          </div>

          {/* Previsualización de la Imagen */}
          {preview && (
            <div className="preview-wrap">
              <img src={preview} alt="Preview" className="preview-image" />
            </div>
          )}

          {/* Botón de Acción */}
          <button
            type="submit"
            disabled={!image || loading}
            className={`submit-button ${loading || !image ? 'disabled' : 'enabled'}`}
          >
            {loading ? 'Procesando con Red Neuronal...' : 'Iniciar Diagnóstico'}
          </button>
        </form>

        {/* Mensajes de Error */}
        {error && (
          <div className="error-box">
            ⚠️ Error de conexión: {error}. (¿Falta configurar el Backend?)
          </div>
        )}

        {/* Panel de Resultados */}
        {result && (
          <div className="result-box">
            <h3 className="result-title">Resultado del Análisis</h3>
            <p className="result-value">
              {result.prediction === 'PNEUMONIA' ? '🔴 Neumonía' : '🟢 Sano'}
            </p>
            <p className="result-confidence">
              Confianza del modelo: {result.confidence ? result.confidence.toFixed(2) : "0.00"}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;