import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. OS CAMINHOS
nome_modelo_base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
caminho_pen_drive = "./meu_chatbot_tinyllama" # A pasta que geramos no treino!
pasta_saida = "./meu_modelo_fundido"

print("🧠 1. Baixando/Carregando o Cérebro Base...")
tokenizer = AutoTokenizer.from_pretrained(nome_modelo_base)
modelo_base = AutoModelForCausalLM.from_pretrained(
    nome_modelo_base,
    torch_dtype=torch.float16, # Mantém a matemática leve
    low_cpu_mem_usage=True
)

print("🔌 2. Plugando o seu Conhecimento (LoRA)...")
modelo_ia = PeftModel.from_pretrained(modelo_base, caminho_pen_drive)

print("🔥 3. Fundindo as peças (Isso exige bastante memória RAM)...")
# AQUI ESTÁ O SEGREDO: Derretemos o LoRA permanentemente no modelo base
modelo_fundido = modelo_ia.merge_and_unload()

print(f"💾 4. Salvando o novo modelo completo na pasta '{pasta_saida}'...")
modelo_fundido.save_pretrained(pasta_saida, safe_serialization=True)
tokenizer.save_pretrained(pasta_saida)

print("✅ Fusão completa! Você acabou de criar a sua própria IA offline.")