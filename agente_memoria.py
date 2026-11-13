import os
from pathlib import Path
import requests

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
USER_NAME = os.getenv("USER_NAME", "Usuario")

url = f"{OLLAMA_BASE_URL}/api/chat"

# 1. CRIANDO O CÉREBRO: Array que vai armazenar todo o histórico
historico = [
    {
        "role": "system", 
        "content": "Você é um assistente técnico sênior e direto. Nunca use saudações longas. Vá direto ao ponto."
    }
]

print(f"=== Agente DeepSeek Iniciado ({USER_NAME}) - Digite 'sair' para encerrar ===\n")

# 2. O LOOP NEURAL
while True:
    pergunta = input(f"{USER_NAME}: ")
    
    if pergunta.lower() == 'sair':
        print("Encerrando agente...")
        break

    # Passo A: Inserimos a sua pergunta no histórico
    historico.append({"role": "user", "content": pergunta})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": historico, # <- Enviamos a MATRIZ inteira de memória, não só a pergunta!
        "temperature": 0.3,
        "stream": False
    }

    print("[Processando...]")
    response = requests.post(url, json=payload).json()
    
    # Extraindo apenas o texto da resposta da IA
    resposta_ia = response["message"]["content"]
    
    print(f"DeepSeek: {resposta_ia}\n")

    # Passo B: O SEGREDO! Inserimos a resposta da IA de volta no histórico
    # Para que na próxima rodada, ela lembre do que acabou de dizer.
    historico.append({"role": "assistant", "content": resposta_ia})
