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
OLLAMA_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "deepseek-r1:8b")

# A URL do servidor local que o Ollama criou no seu PC
url = f"{OLLAMA_BASE_URL}/api/generate"

# Aqui estão as "engrenagens" que você vai manipular
payload = {
    "model": OLLAMA_MODEL,
    
    # ENGRENAGEM 1: O System Prompt (O cérebro mestre)
    # É aqui que você injeta regras absolutas que a IA não pode quebrar.
    "system": "Você é um arquiteto de dados sênior. Regra estrita: O nome de todas as chaves primárias e estrangeiras nas tabelas deve começar obrigatoriamente com o prefixo 'cod_' em vez de usar 'id'.",
    
    # ENGRENAGEM 2: O User Prompt (O que você quer que ela faça agora)
    "prompt": "Escreva o script MySQL para criar a tabela de máquinas de um sistema de manutenção.",
    
    # ENGRENAGEM 3: A Temperatura (A Criatividade)
    # 0.0 a 0.3 = Muito lógico, focado, matemático.
    # 0.7 a 1.0 = Criativo, vocabulário solto.
    "temperature": 0.1,
    
    # Faz com que a IA mande a resposta inteira de uma vez, e não palavra por palavra.
    "stream": False
}

print(f"Enviando requisição para o {OLLAMA_MODEL}... aguarde o processamento neural!\n")

response = requests.post(url, json=payload)

if response.status_code == 200:
    resultado = response.json()
    print("RESPOSTA DA IA:\n")
    print(resultado["response"])
else:
    print(f"Erro na requisição: {response.status_code}")
