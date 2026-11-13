from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# 1. ESCOLHENDO UM MODELO "PEQUENO" DE VERDADE
# O TinyLlama tem 1.1B de parâmetros. Ele cabe perfeitamente em PCs comuns.
nome_do_modelo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Baixando o Cérebro Base (Isso consome uns 2GB a 4GB do seu HD)...")
tokenizer = AutoTokenizer.from_pretrained(nome_do_modelo)
modelo_base = AutoModelForCausalLM.from_pretrained(nome_do_modelo)

# 2. A MÁGICA DO PEFT / LORA (O Pen-Drive de Conhecimento)
# Aqui nós dizemos para o PyTorch: "Não treine o modelo inteiro! Crie apenas
# um desvio minúsculo e treine ele."
configuracao_lora = LoraConfig(
    r=8,               # O "tamanho" do pen-drive (8 é bem leve)
    lora_alpha=16,     # A força com que o pen-drive afeta o cérebro base
    target_modules=["q_proj", "v_proj"], # Em quais engrenagens vamos plugar o pen-drive
    bias="none",
    task_type="CAUSAL_LM"
)

# 3. FUNDINDO O PEN-DRIVE NO CÉREBRO CONGELADO
# O get_peft_model automaticamente congela (requires_grad = False) os 1.1 Bilhões
# de parâmetros do TinyLlama e deixa ativado apenas os ~14 Milhões do LoRA.
modelo_para_treinar = get_peft_model(modelo_base, configuracao_lora)

# Opcional: O PyTorch imprime na tela o quanto de memória você acabou de economizar!
modelo_para_treinar.print_trainable_parameters()

# A PARTIR DAQUI: 
# Você criaria o loop de treinamento (ou usaria o Trainer da Hugging Face)
# passando aquele seu arquivo 'conhecimento.txt' que criamos antes!