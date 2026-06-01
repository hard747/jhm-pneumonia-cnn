import React, { useState } from 'react';

// 🔌 CONEXIÓN AUTOMÁTICA PROFESIONAL
// Si existe la variable en Vercel, la usa. Si estás en local, usa el Docker de tu PC de forma automática.
const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000/predict';

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
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!image) return;

    loading(true);
    setError(null);

    // En la industria, las imágenes se envían como Multipart/FormData
    const formData = new FormData();
    formData.append('file', image);

    try {
      // 🚀 Nos conectamos dinámicamente usando la URL activa (local o remota)
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Error en el servidor de IA. Código: ' + response.status);
      }

      const data = await response.json();
      setResult(data); // Guardamos la predicción de la neurona
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col items-center justify-center p-6" style={{ fontFamily: 'sans-serif' }}>
      <div className="bg-gray-800 p-8 rounded-xl shadow-2xl max-w-md w-full border border-gray-700" style={{ backgroundColor: '#1f2937', borderRadius: '0.75rem', padding: '2rem', maxWidth: '28rem', width: '100%' }}>
        <h1 className="text-2xl font-bold text-center mb-2" style={{ color: '#60a5fa', textAlign: 'center', fontSize: '1.5rem', fontWeight: 'bold', margin: '0 0 0.5rem 0' }}>JHM - Análisis de Neumonía</h1>
        <p className="text-sm text-gray-400 text-center mb-6" style={{ color: '#9ca3af', textAlign: 'center', fontSize: '0.875rem', margin: '0 0 1.5rem 0' }}>Sistema Clínico de Diagnóstico por IA</p>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          {/* Zona de Selección de Imagen */}
          <div className="border-2 border-dashed border-gray-600 rounded-lg p-4 text-center cursor-pointer hover:border-blue-500 transition" style={{ border: '2px dashed #4b5563', borderRadius: '0.5rem', padding: '1rem', textAlign: 'center' }}>
            <input 
              type="file" 
              accept="image/*" 
              onChange={handleImageChange} 
              style={{ display: 'none' }} 
              id="file-upload"
            />
            <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'block', fontSize: '0.875rem', color: '#d1d5db' }}>
              {preview ? '🔄 Cambiar radiografía' : '📁 Seleccionar Radiografía de Tórax'}
            </label>
          </div>

          {/* Previsualización de la Imagen */}
          {preview && (
            <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
              <img src={preview} alt="Preview" style={{ maxHeight: '12rem', borderRadius: '0.375rem', border: '1px solid #374151' }} />
            </div>
          )}

          {/* Botón de Acción */}
          <button
            type="submit"
            disabled={!image || loading}
            style={{
              width: '100%',
              padding: '0.75rem',
              borderRadius: '0.5rem',
              fontWeight: 'bold',
              border: 'none',
              cursor: loading || !image ? 'not-allowed' : 'pointer',
              backgroundColor: loading || !image ? '#4b5563' : '#2563eb',
              color: 'white',
              transition: 'background-color 0.2s'
            }}
          >
            {loading ? 'Procesando con Red Neuronal...' : 'Iniciar Diagnóstico'}
          </button>
        </form>

        {/* Mensajes de Error */}
        {error && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: 'rgba(127, 29, 29, 0.5)', border: '1px solid #ef4444', borderRadius: '0.5rem', color: '#fca5a5', fontSize: '0.875rem' }}>
            ⚠️ Error de conexión: {error}. (¿Falta configurar el Backend?)
          </div>
        )}

        {/* Panel de Resultados */}
        {result && (
          <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: 'rgba(30, 58, 138, 0.3)', border: '1px solid rgba(59, 130, 246, 0.5)', borderRadius: '0.5rem', textAlign: 'center' }}>
            <h3 style={{ fontSize: '1.125rem', fontWeight: 'semibold', color: '#93c5fd', margin: '0' }}>Resultado del Análisis</h3>
            <p style={{ fontSize: '1.875rem', fontWeight: '900', margin: '0.5rem 0 0 0', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
              {result.prediction === 'PNEUMONIA' ? '🔴 Neumonía' : '🟢 Sano'}
            </p>
            <p style={{ fontSize: '0.75rem', color: '#9ca3af', margin: '0.5rem 0 0 0' }}>
              Confianza del modelo: {result.confidence ? result.confidence.toFixed(2) : "0.00"}%
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;