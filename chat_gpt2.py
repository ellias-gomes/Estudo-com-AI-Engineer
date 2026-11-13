import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 1. CARREGAR AS PEÇAS
# ==========================================
nome_modelo_base = "gpt2"

# ATENÇÃO: Verifique se o nome da pasta abaixo é exatamente onde o arquivo 
# 'adapter_model.safetensors' foi salvo no seu computador!
caminho_pen_drive = "./meu_chatbot_gpt" 

print("🧠 Carregando o Cérebro Base (GPT-2)...")
tokenizer = AutoTokenizer.from_pretrained(nome_modelo_base)
tokenizer.pad_token = tokenizer.eos_token # O GPT-2 precisa dessa configuração

modelo_base = AutoModelForCausalLM.from_pretrained(nome_modelo_base)

print("🔌 Plugando o seu Conhecimento (LoRA)...")
# A MÁGICA ACONTECE AQUI: Fundimos o modelo base com o seu treinamento!
modelo_ia = PeftModel.from_pretrained(modelo_base, caminho_pen_drive)

# ==========================================
# 2. O LOOP DE CONVERSA
# ==========================================
def conversar(pergunta_usuario):
    # Formatamos a pergunta exatamente igual ensinamos no conhecimento.txt
    prompt = f"<pergunta> {pergunta_usuario} <resposta>"
    
    # Converte o texto para a matemática da IA
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # A IA gera a resposta
    with torch.no_grad():
        outputs = modelo_ia.generate(
            **inputs, 
            max_new_tokens=50, 
            temperature=0.1,      # Temperatura baixa para respostas mais diretas
            do_sample=True,       # Necessário para a temperatura funcionar no GPT-2
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Traduz a matemática de volta para texto
    resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Cortamos a string para mostrar apenas o que vem DEPOIS de <resposta>
    if "<resposta>" in resposta_completa:
        return resposta_completa.split("<resposta>")[-1].strip()
    return resposta_completa

print("\n=======================================")
print("🤖 IA LOCAL INICIADA (GPT-2 com LoRA)")
print("=======================================")

while True:
    texto = input("\nVocê: ")
    if texto.lower() == 'sair':
        break
        
    print(f"IA: {conversar(texto.lower())}")