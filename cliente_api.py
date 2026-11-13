"""Cliente simples de API externa configurado 100% pelo .env.

Provedores suportados (defina API_PROVIDER no .env):
  - gemini  -> usa GEMINI_API_KEY  (e GEMINI_MODEL, opcional)
  - openai  -> usa OPENAI_API_KEY  (e OPENAI_MODEL, opcional)

Uso:
    from cliente_api import perguntar
    print(perguntar("Olá!"))
"""
import os
from pathlib import Path
import requests


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


def perguntar(prompt, provider=None):
    provider = (provider or os.getenv("API_PROVIDER", "gemini")).lower()

    if provider == "gemini":
        chave = os.getenv("GEMINI_API_KEY")
        modelo = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        if not chave:
            raise RuntimeError("Defina GEMINI_API_KEY no .env")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent",
            headers={"x-goog-api-key": chave},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    if provider == "openai":
        chave = os.getenv("OPENAI_API_KEY")
        modelo = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if not chave:
            raise RuntimeError("Defina OPENAI_API_KEY no .env")
        r = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {chave}"},
            json={"model": modelo, "messages": [{"role": "user", "content": prompt}]},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    raise ValueError(f"Provedor desconhecido: {provider}")


if __name__ == "__main__":
    print(perguntar(input("Pergunta: ")))
