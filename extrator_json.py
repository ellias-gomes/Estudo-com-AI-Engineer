import os
from pathlib import Path
import requests
import json

# Carrega variáveis de ambiente do .env se existir
def carregar_env():
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip("'\""))

carregar_env()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_EXTRACTOR_MODEL", "qwen2.5:1.5b")

url = f"{OLLAMA_BASE_URL}/api/generate"

# O texto desestruturado do mundo real
relato_mecanico = "A fresadora CNC travou a engrenagem hoje cedo. O painel tá piscando e começou a sair uma fumaça preta com cheiro de queimado."

payload = {
    "model": OLLAMA_MODEL,
    
    # ENGRENAGEM SECRETA: Forçando a saída para ser exclusivamente JSON
    "format": "json",
    
    # Modelando a IA para ser um robô de extração
    "system": """Você é um sistema de triagem de manutenção industrial. 
    Analise o relato e extraia as informações estritamente neste formato JSON:
    {
      "cod_maquina": "nome ou tipo do equipamento",
      "sintoma_principal": "resumo do problema",
      "nivel_urgencia": "BAIXA, MEDIA ou ALTA"
    }""",
    
    "prompt": relato_mecanico,
    
    # Temperatura ZERADA. Não queremos criatividade, queremos um JSON exato.
    "temperature": 0.0,
    "stream": False
}

print(f"Analisando o relato do mecânico com {OLLAMA_MODEL}...\n")
response = requests.post(url, json=payload).json()
print(response)

print('---')

# Recebemos a resposta em texto
resposta_texto = response["response"]

# O milagre: Podemos converter a resposta da IA direto para um dicionário Python!
dados_estruturados = json.loads(resposta_texto)

print("✅ DADOS PRONTOS PARA O BANCO DE DADOS:")
print(f"Código/Máquina: {dados_estruturados.get('cod_maquina')}")
print(f"Problema: {dados_estruturados.get('sintoma_principal')}")
print(f"Urgência: {dados_estruturados.get('nivel_urgencia')}")
