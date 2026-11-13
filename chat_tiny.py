import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import warnings
from transformers import logging

warnings.filterwarnings("ignore")
logging.set_verbosity_error() # Força a biblioteca a mostrar apenas erros fatais, escondendo os avisos

# ==========================================
# 1. CARREGAR AS PEÇAS
# ==========================================
nome_modelo_base = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
caminho_pen_drive = "./meu_chatbot_tinyllama"

print("🧠 Carregando o Cérebro Base (TinyLlama)...")
tokenizer = AutoTokenizer.from_pretrained(nome_modelo_base)
modelo_base = AutoModelForCausalLM.from_pretrained(nome_modelo_base)

print("🔌 Plugando o seu Conhecimento (LoRA)...")
# A MÁGICA ACONTECE AQUI: Fundimos o modelo base com o seu treinamento!
modelo_ia = PeftModel.from_pretrained(modelo_base, caminho_pen_drive)

# Opcional: Se quiser que ela responda mais rápido na CPU, usamos torch.bfloat16
modelo_ia = modelo_ia.to(torch.float32) 

# ==========================================
# 2. O LOOP DE CONVERSA
# ==========================================
def conversar(pergunta_usuario):
    # 1. MUDANÇA NO PROMPT: Forçamos ela a abrir a tag de pensamento primeiro
    prompt = f"<pergunta> {pergunta_usuario.lower()} <pensamento>"
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = modelo_ia.generate(
            **inputs, 
            max_new_tokens=100,      # Aumentamos para dar espaço ao pensamento + resposta
            do_sample=True,
            temperature=0.2,         # Baixamos a temperatura para ela ser menos "doida"
            repetition_penalty=1.3,  # Aumentamos a punição para evitar loops
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 2. LÓGICA DE EXTRAÇÃO
    if "<resposta>" in resposta_completa:
        # Pega apenas o que vem depois da resposta final
        texto_final = resposta_completa.split("<resposta>")[-1].strip()
    else:
        # Se ela esqueceu de fechar o pensamento, tentamos limpar o que sobrou
        texto_final = resposta_completa.split("<pensamento>")[-1].strip()

    # 3. A NOVA GUILHOTINA (Limpando o lixo)
    # Cortamos qualquer tentativa da IA de continuar o diálogo ou repetir tags
    for stop_word in ["\n", "<pergunta>", "Você:", "ella:", "IA:"]:
        texto_final = texto_final.split(stop_word)[0]
        
    return texto_final.strip()

print("\n=======================================")
print("🤖 IA LOCAL INICIADA (CPU Mode)")
print("=======================================")

while True:
    texto = input("\nVocê: ")
    if texto.lower() == 'sair':
        break
        
    print(f"IA: {conversar(texto.lower())}")