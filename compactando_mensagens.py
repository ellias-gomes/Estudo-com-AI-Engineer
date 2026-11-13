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
OLLAMA_COMPRESSION_MODEL = os.getenv("OLLAMA_COMPRESSION_MODEL", "deepseek-v3.1:671b-cloud")
USER_NAME = os.getenv("USER_NAME", "Usuario")

url_chat = f"{OLLAMA_BASE_URL}/api/chat"
url_generate = f"{OLLAMA_BASE_URL}/api/generate"
modelo = OLLAMA_COMPRESSION_MODEL

# LIMITE DE MEMÓRIA (Baixo para você conseguir testar o gatilho rapidamente)
LIMITE_MENSAGENS = 8

# 1. O CÉREBRO INICIAL (A Matriz de Memória)
historico = [
    {
        "role": "system",
        "content": "Você é um assistente técnico direto. Responda de forma concisa e amigável."
    }
]

print(f"=== Agente DeepSeek com Memória Dinâmica ({USER_NAME}) ===")
print("Digite 'sair' para encerrar.\n")

# 2. O LOOP NEURAL
while True:
    pergunta = input(f"{USER_NAME}: ")
    
    if pergunta.lower() == 'sair':
        print("Encerrando agente...")
        break

    # Passo A: Adiciona a pergunta atual na matriz
    historico.append({"role": "user", "content": pergunta})

    # --- O GATILHO DA COMPRESSÃO NEURAL ---
    # Verifica se a matriz ficou maior que o limite estipulado
    if len(historico) > LIMITE_MENSAGENS:
        print("\n[⚙️ SISTEMA: Limite de memória atingido. Iniciando compressão...]")
        
        # Separamos as mensagens antigas para o abate (ignorando o System no índice 0)
        # Vamos pegar da posição 1 até a 5 (deixando as mais recentes intactas)
        mensagens_antigas = historico[1:6]
        texto_para_resumir = json.dumps(mensagens_antigas, ensure_ascii=False)
        
        # Criamos um payload especial na rota /api/generate apenas para resumir
        payload_resumo = {
            "model": modelo,
            "system": "Você é um compressor de memória. Resuma a conversa a seguir em no máximo 2 frases contendo apenas os fatos cruciais. Não converse, apenas entregue o resumo.",
            "prompt": texto_para_resumir,
            "stream": False,
            "temperature": 0.0 # Temperatura zerada: queremos um resumo cirúrgico e exato
        }
        
        print("[⚙️ SISTEMA: Gerando resumo matemático em background...]")
        resposta_resumo = requests.post(url_generate, json=payload_resumo).json()
        resumo_gerado = resposta_resumo["response"]
        
        print(f"[⚙️ SISTEMA: Resumo criado -> {resumo_gerado}]")
        
        # Passo Crucial: Reconstruindo a matriz de memória
        # [System Prompt] + [Novo Resumo Fictício] + [Mensagens recentes que sobraram]
        novo_historico = [historico[0]]
        novo_historico.append({"role": "system", "content": f"Contexto anterior resumido: {resumo_gerado}"})
        novo_historico.extend(historico[6:])
        
        # Substituímos o histórico antigo pelo novo, muito mais leve
        historico = novo_historico
        print("[⚙️ SISTEMA: Matriz limpa e otimizada com sucesso! Voltando ao chat...]\n")
    # ----------------------------------------

    # Passo B: Enviamos a matriz (compactada ou não) para o chat
    payload_chat = {
        "model": modelo,
        "messages": historico,
        "temperature": 0.3,
        "stream": False
    }

    print("[Processando...]")
    response = requests.post(url_chat, json=payload_chat).json()
    
    resposta_ia = response["message"]["content"]
    print(f"\nDeepSeek: {resposta_ia}\n")

    # Passo C: Salva a resposta da IA na memória para o próximo ciclo
    historico.append({"role": "assistant", "content": resposta_ia})
