import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset

# ==========================================
# 1. CARREGAR O CÉREBRO E O TRADUTOR
# ==========================================
# nome_do_modelo = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
nome_do_modelo = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(nome_do_modelo)

# O TinyLlama não tem um "token de preenchimento" padrão para treinos curtos, 
# então ensinamos ele a usar o token de "fim de frase" para isso.
tokenizer.pad_token = tokenizer.eos_token

print("Carregando o modelo base para a memória...")
modelo_base = AutoModelForCausalLM.from_pretrained(nome_do_modelo)

# ==========================================
# 2. CONFIGURAR O PEN-DRIVE LORA
# ==========================================
configuracao_lora = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"], # <-- A MUDANÇA É SÓ AQUI!
    bias="none",
    task_type="CAUSAL_LM"
)
modelo_para_treinar = get_peft_model(modelo_base, configuracao_lora)

# ==========================================
# 3. PREPARAR OS DADOS (O seu conhecimento.txt)
# ==========================================
class MeuDatasetPersonalizado(Dataset):
    def __init__(self, caminho_arquivo, tokenizer, tamanho_maximo=128):
        with open(caminho_arquivo, 'r', encoding='utf-8') as f:
            # Lê todas as linhas do seu arquivo, ignorando linhas em branco
            self.linhas = [linha.strip() for linha in f if linha.strip()]
        self.tokenizer = tokenizer
        self.tamanho_maximo = tamanho_maximo

    def __len__(self):
        return len(self.linhas)

    def __getitem__(self, idx):
        texto = self.linhas[idx]
        
        # Converte a sua frase em Português para os IDs matemáticos da IA
        codificacao = self.tokenizer(
            texto, 
            truncation=True, 
            padding="max_length", 
            max_length=self.tamanho_maximo, 
            return_tensors="pt"
        )
        
        # Como a IA tenta prever a própria frase, os 'labels' (respostas corretas) 
        # são iguais aos 'input_ids' (a frase lida).
        item = {chave: valor.squeeze(0) for chave, valor in codificacao.items()}
        item["labels"] = item["input_ids"].clone()
        return item

print("Formatando o dataset de conhecimento...")
meu_dataset = MeuDatasetPersonalizado("conhecimento.txt", tokenizer)
colator_de_dados = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ==========================================
# 4. CONFIGURAR O MOTOR DE TREINO (Hugging Face Trainer)
# ==========================================
config_treino = TrainingArguments(
    output_dir="./resultados_lora", # Pasta onde ele salva backups temporários
    num_train_epochs=30,             # Quantas vezes ele vai ler o seu texto
    per_device_train_batch_size=2,  # Lê 2 frases por vez
    save_steps=50,                  # Faz um backup a cada 50 passos
    logging_steps=5,                # Mostra o Loss a cada 5 passos no terminal
    learning_rate=2e-4,             # Velocidade de aprendizado padrão para LoRA
    optim="adamw_torch"
)

treinador = Trainer(
    model=modelo_para_treinar,
    args=config_treino,
    train_dataset=meu_dataset,
    data_collator=colator_de_dados
)

# ==========================================
# 5. INICIAR A SALA DE AULA
# ==========================================
print("\n🚀 Iniciando o Treinamento Neural...")
treinador.train()

# 6. SALVAR O PEN-DRIVE FINAL
# Ele não salva o modelo de 2GB de novo, ele salva SÓ a camada do LoRA (uns 10 MB!)
modelo_para_treinar.save_pretrained("./meu_chatbot_gpt")
print("\n✅ Treinamento concluído! O seu modelo foi salvo na pasta './meu_chatbot_gpt'.")