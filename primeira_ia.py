import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

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

# Configuração e higienização de dados pessoais via .env
TRAINING_USER_NAME = os.getenv("TRAINING_USER_NAME", "usuario")
TRAINING_USER_AGE = os.getenv("TRAINING_USER_AGE", "20")

# # ==========================================
# # 1. O DICIONÁRIO (A Base de Dados da IA)
# # ==========================================
# text = "i like dogs the dog is very interesting"
# words = text.split() # Correção: Agora separamos as palavras corretamente em uma lista

# # Como você já domina banco de dados, pense no 'vocab' como a Chave Primária (cod_)
# # A IA não entende letras, ela só entende esses números associados a cada palavra.
# vocab = {'the': 0, 'a': 1, 'is': 2, 'very': 3, 'interesting': 4, 'dog': 5, 'runs': 6, 'like': 7, 'dogs': 8, 'i': 9}
# itos = {0: 'the', 1: 'a', 2: 'is', 3: 'very', 4: 'interesting', 5: 'dog', 6: 'runs', 7: 'like', 8: 'dogs', 9: 'i'}

# ==========================================
# 1. O DICIONÁRIO AUTOMATIZADO
# ==========================================
default_text = f"ola , meu nome é {TRAINING_USER_NAME} e eu tenho {TRAINING_USER_AGE} anos, gosto muito da minha namorada ! estou interessado em aprender a criar uma IA que seja super inteligente , por isso estou desenvolvendo voce ! como gostaria de ser chamada ?"
text = os.getenv("TRAINING_TEXT", default_text)
words = text.split()

# 1. Pegamos todas as palavras únicas do texto (o 'set' remove as duplicadas)
palavras_unicas = list(set(words))

# 2. O Python cria os dicionários automaticamente para nós!
vocab = {palavra: i for i, palavra in enumerate(palavras_unicas)}
itos = {i: palavra for i, palavra in enumerate(palavras_unicas)}

print("Vocabulário gerado automaticamente:")
print(vocab)
print("-" * 30)


# ==========================================
# 2. A ARQUITETURA DO CÉREBRO (O Fluxo Neural)
# ==========================================
# class LanguageModel(nn.Module):
    # def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
    #     super().__init__()
    #     # Camada 1: EMBEDDING
    #     # Pega o ID da palavra (ex: 5 para 'dog') e transforma em um vetor matemático com várias dimensões.
    #     self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
    #     # Camada 2: RNN (Recurrent Neural Network)
    #     # É a "memória". Ela lê a palavra atual, junta com o que lembra da palavra anterior, e passa adiante.
    #     self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        
    #     # Camada 3: LINEAR (Totalmente Conectada)
    #     # Pega a "confusão" matemática que saiu da RNN e converte de volta para 7 opções (o tamanho do nosso dicionário),
    #     # apontando qual das 7 palavras tem maior probabilidade de ser a próxima.
    #     self.fc = nn.Linear(hidden_dim, output_dim)
class LanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        
        # A LINHA QUE FALTAVA: Salvando a dimensão na própria classe
        self.hidden_dim = hidden_dim 
        
        # Camada 1: EMBEDDING
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Camada 2: RNN
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        
        # Camada 3: LINEAR
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden
    
    def init_hidden(self, batch_size):
        # Sempre zeramos a memória antes de começar uma frase nova
        return torch.zeros(1, batch_size, self.hidden_dim)


# ==========================================
# 3. INSTANCIANDO AS ENGRENAGENS
# ==========================================
# Criamos um modelo com 3 dimensões de reflexão (embedding) e 5 neurônios de memória (hidden_dim)
model = LanguageModel(vocab_size=len(vocab), embedding_dim=3, hidden_dim=5, output_dim=len(vocab))

# O Juiz (Criterion): Mede o quão burra ou inteligente foi a resposta da IA.
criterion = nn.CrossEntropyLoss()

# O Professor (Optimizer): É o algoritmo que entra no cérebro e muda os pesos matemáticos
# para que o modelo não cometa o mesmo erro na próxima vez. (lr = Taxa de Aprendizado)
optimizer = optim.SGD(model.parameters(), lr=0.1)


# ==========================================
# 4. O LOOP DE TREINAMENTO (A Sala de Aula)
# ==========================================
num_epochs = 1000
sequence_length = 3 # Vamos ler 3 palavras por vez para adivinhar a 4ª

print("Iniciando o treinamento do modelo...\n")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    # A IA desliza pela frase como se fosse um leitor de código de barras
    for i in range(len(words) - sequence_length):
        # Pega o Contexto (Ex: 'the', 'dog', 'is')
        seq_words = words[i : i + sequence_length]
        # Pega a Resposta Correta (Ex: 'very')
        target_word = words[i + sequence_length]
        
        # Converte para tensores (matrizes que a placa de vídeo/processador entende)
        inputs = torch.tensor([vocab[w] for w in seq_words]).unsqueeze(0)
        target = torch.tensor([vocab[target_word]])
        
        hidden = model.init_hidden(batch_size=1)
        
        # Passo 1: A IA chuta a resposta (Forward pass)
        output, hidden = model(inputs, hidden)
        
        # Pegamos apenas o que ela pensou sobre a última palavra da sequência
        prediction = output[0, -1, :].unsqueeze(0)
        
        # Passo 2: O Juiz calcula o erro (Loss)
        loss = criterion(prediction, target)
        
        # Passo 3: O Professor ajusta o cérebro (Backward pass)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    # Mostramos a evolução a cada 20 tentativas
    if (epoch + 1) % 20 == 0:
        erro_medio = total_loss / (len(words) - sequence_length)
        print(f'Época [{epoch+1}/{num_epochs}] | Erro (Loss): {erro_medio:.4f}')


# ==========================================
# 5. O TESTE FINAL (Geração de Texto)
# ==========================================
def generate_text(seed_words, num_words_to_predict):
    model.eval() # Congelamos o cérebro (desliga o modo de aprendizado)
    indices = [vocab[w] for w in seed_words]
    input_tensor = torch.tensor(indices).unsqueeze(0)
    hidden = model.init_hidden(1)
    
    generated = seed_words.copy()
    
    # Pedimos para a IA escrever as próximas palavras
    for _ in range(num_words_to_predict):
        output, hidden = model(input_tensor, hidden)
        
        # Transforma os números brutos da rede em porcentagens de probabilidade (Softmax)
        predicted_probs = output[0, -1, :].softmax(dim=0)
        
        # Escolhe a palavra mais provável
        idx = torch.multinomial(predicted_probs, 1).item()
        predicted_word = itos[idx]
        generated.append(predicted_word)
        
        # A palavra recém criada vira a entrada do próximo ciclo!
        input_tensor = torch.tensor([[idx]])
        
    return ' '.join(generated)

print("\n--- Resultado do Treinamento ---")
# print("Prompt inicial: ['the', 'dog', 'is']")
seed_words = ['ola', 'nome', TRAINING_USER_NAME, ',', 'estou', 'interessado', 'criar', 'IA', 'seja', 'inteligente', '!', 'gostaria', 'chamada']
seed_words_validos = [w for w in seed_words if w in vocab]
print("IA gerou:", generate_text(seed_words_validos if seed_words_validos else list(vocab.keys())[:3], num_words_to_predict=2))