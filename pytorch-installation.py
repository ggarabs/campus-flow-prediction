import torch
import subprocess
import sys

print("Verificando versões do ambiente...")

torch_version = torch.__version__.split('+')[0]
cuda_version = torch.version.cuda.replace('.', '')

whl_url = f"https://data.pyg.org/whl/torch-{torch_version}+cu{cuda_version}.html"
print(f"Buscando binários pré-compilados na URL: \n{whl_url}\n")

comando_pacotes_base = [
    sys.executable, "-m", "pip", "install", 
    "torch-scatter", "torch-sparse", 
    "-f", whl_url
]

comando_pacote_temporal = [
    sys.executable, "-m", "pip", "install", "torch-geometric-temporal"
]

try:
    print("Instalando dependências otimizadas (torch-scatter, etc)...")
    subprocess.check_call(comando_pacotes_base)
    
    print("\nInstalando torch-geometric-temporal...")
    subprocess.check_call(comando_pacote_temporal)
    
    print("\nInstalação concluída com sucesso!")
except subprocess.CalledProcessError as e:
    print(f"\nOcorreu um erro durante a instalação: {e}")