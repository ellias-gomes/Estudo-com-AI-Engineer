import os
from pathlib import Path
import requests
import json
import datetime

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

url = f"{OLLAMA_BASE_URL}/api/generate"

data_hoje = datetime.datetime.now().strftime('%Y-%m-%d')

input_usuario = 'fui a padaria ontem de manha comprar dois croissants e um sonho de doce de leite para tomar café da manhã e gastei 40 reais'

payload = {
    'model': OLLAMA_MODEL,

    'format': 'json',

    'system': f'''
        Você é um extrator de dados financeiros.
        A data de hoje é {data_hoje}.
        Analise o texto do usuário e devolva EXCLUSIVAMENTE este JSON:
        {{
            'descricao_movimento': 'resumo curto do gasto',
            'valor': 'apenas o número float',
            'tipo_movimento': 'DESPESA ou RECEITA',
            'categoria': 'ex: ALIMENTACAO, LAZER, TRANSPORTE',
            'data_movimento': 'YYYY-MM-DD'
        }}
    ''',

    'prompt': input_usuario,

    'temperature': 0.0,

    'stream': False
}

print(f"Analisando o Relatório do Usuário com {OLLAMA_MODEL}...\n")
response = requests.post(url, json=payload).json()

resposta_texto = response['response']

dados_estruturados = json.loads(resposta_texto)

print('✅ DADOS PRONTOS PARA O BANCO DE DADOS:')
print(f"Descrição do Movimento: {dados_estruturados.get('descricao_movimento')}")
print(f"Valor: {dados_estruturados.get('valor')}")
print(f"Tipo de Movimento: {dados_estruturados.get('tipo_movimento')}")
print(f"Categoria: {dados_estruturados.get('categoria')}")
print(f"Data do Movimento: {dados_estruturados.get('data_movimento')}")
