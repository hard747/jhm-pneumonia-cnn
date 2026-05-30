import torch

print("=== TESTE DE INFRAESTRUTURA DE IA ===")
print(f"Versão do PyTorch: {torch.__version__}")
print(f"O CUDA está disponível?: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Placa de Vídeo Detectada: {torch.cuda.get_device_name(0)}")
    print("Sucesso! O pipeline está pronto para rodar em alta performance.")
else:
    print("Aviso: O PyTorch está rodando apenas no Processador (CPU).")