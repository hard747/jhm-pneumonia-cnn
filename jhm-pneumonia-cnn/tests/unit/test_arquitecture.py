import pytest
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.model import PneumoniaCNN

def test_model_output_shape():
    """
    UNIT TEST: Verifica el lote con la resolución nativa del modelo (128x128)
    """
    model = PneumoniaCNN()
    model.eval()
    
    # AJUSTADO: Matriz nativa de 128x128
    mock_images = torch.randn(4, 1, 128, 128)
    
    with torch.no_grad():
        output = model(mock_images)
        
    assert output.shape == (4, 2), f"Formato de salida incorrecto: {output.shape}"

def test_model_prediction_logic():
    """
    UNIT TEST: Verifica que las probabilidades sumen 100%.
    """
    model = PneumoniaCNN()
    model.eval()
    
    # AJUSTADO: Matriz nativa de 128x128
    mock_image = torch.randn(1, 1, 128, 128)
    
    with torch.no_grad():
        output = model(mock_image)
        probabilities = torch.softmax(output, dim=1)
        
    assert torch.isclose(probabilities.sum(), torch.tensor(1.0)), "Las probabilidades no suman 100%"

def test_model_invalid_input_fails():
    """
    UNIT TEST: Verifica que el modelo falle si recibe dimensiones incorrectas (ej: 10x10).
    """
    model = PneumoniaCNN()
    invalid_mock = torch.randn(1, 1, 10, 10)
    
    with pytest.raises(RuntimeError):
        model(invalid_mock)