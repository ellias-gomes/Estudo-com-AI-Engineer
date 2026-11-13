import torch
import torch.nn as nn
import torch.optim as optim
import os


# ==========================================
# 1. O DATASET DE INSTRUÇÃO (Os Flashcards)
# ==========================================
# Note a estrutura rígida. A IA vai aprender que sempre que vir 
# a palavra <resposta>, ela deve cuspir a solução.
text = """
<pergunta> ola <resposta> oi, eu sou a ia do ellias
<pergunta> qual o seu nome <resposta> meu nome eh ia do ellias
<pergunta> quem te criou <resposta> o ellias me criou no windows
<pergunta> quantos anos voce tem <resposta> eu acabei de nascer no seu pc
"""

words = text.split()
palavras_unicas = list(set(words))
vocab = {palavra: i for i, palavra in enumerate(palavras_unicas)}
itos = {i: palavra for i, palavra in enumerate(palavras_unicas)}


# ==========================================
# 2. A ARQUITETURA DO CÉREBRO
# ==========================================
class ChatbotModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Aumentamos um pouco a memória (hidden_dim) para ela decorar frases maiores
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc(output)
        return prediction, hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim)

# Criando a IA (Um pouco mais robusta)
model = ChatbotModel(vocab_size=len(vocab), embedding_dim=10, hidden_dim=20, output_dim=len(vocab))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01) # Trocamos para o Adam, um professor mais moderno que o SGD


# ==========================================
# 3. O TREINAMENTO (A Sala de Aula)
# ==========================================
num_epochs = 500
sequence_length = 3 

print("Ensinando o chatbot a conversar... aguarde!")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for i in range(len(words) - sequence_length):
        seq_words = words[i : i + sequence_length]
        target_word = words[i + sequence_length]
        
        inputs = torch.tensor([vocab[w] for w in seq_words]).unsqueeze(0)
        target = torch.tensor([vocab[target_word]])
        hidden = model.init_hidden(batch_size=1)
        
        output, hidden = model(inputs, hidden)
        prediction = output[0, -1, :].unsqueeze(0)
        
        loss = criterion(prediction, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    if (epoch + 1) % 50 == 0:
        print(f'Época [{epoch+1}/{num_epochs}] | Erro: {total_loss:.4f}')

# Nome do arquivo de "memória"
ARQUIVO_PESOS = 'chatbot_pesos.pth'

# LÓGICA DE CARREGAMENTO (No início do script)
if os.path.exists(ARQUIVO_PESOS):
    print("[SISTEMA] Memória encontrada! Carregando conhecimentos prévios...")
    model.load_state_dict(torch.load(ARQUIVO_PESOS))
    # Se já carregamos, podemos até pular o treino se quisermos
else:
    print("[SISTEMA] Nenhuma memória encontrada. Iniciando treinamento do zero...")
    # Executa o loop de treino aqui...
    
    # LÓGICA DE SALVAMENTO (Após o treino)
    torch.save(model.state_dict(), ARQUIVO_PESOS)
    print(f"[SISTEMA] Treinamento concluído e salvo em {ARQUIVO_PESOS}")

# # ==========================================
# # 4. A INTERFACE DO USUÁRIO (O Loop de Chat)
# # ==========================================
# def conversar_com_ia(pergunta_usuario):
#     model.eval()
    
#     # O SEGREDO: Formatamos a pergunta do usuário igual ao texto de treino!
#     prompt_formatado = f"<pergunta> {pergunta_usuario} <resposta>"
#     seed_words = prompt_formatado.split()
    
#     # Tratamento de erro: Se o usuário digitar uma palavra que a IA não conhece
#     for w in seed_words:
#         if w not in vocab:
#             return f"[Erro] A IA ainda é um bebê e não conhece a palavra: '{w}'"
            
#     input_tensor = torch.tensor([vocab[w] for w in seed_words]).unsqueeze(0)
#     hidden = model.init_hidden(1)
    
#     resposta_gerada = []
    
#     # Pede para a IA gerar as próximas 6 palavras (tamanho médio das nossas respostas)
#     for _ in range(6):
#         output, hidden = model(input_tensor, hidden)
#         predicted_probs = output[0, -1, :].softmax(dim=0)
        
#         # Como é um chatbot exato, pegamos a palavra com maior probabilidade (argmax) em vez de sortear
#         idx = torch.argmax(predicted_probs).item()
#         predicted_word = itos[idx]
        
#         # Se a IA tentar gerar a tag <pergunta> de novo, nós paramos ela!
#         if predicted_word == "<pergunta>":
#             break
            
#         resposta_gerada.append(predicted_word)
#         input_tensor = torch.tensor([[idx]])
        
#     return ' '.join(resposta_gerada)

# ==========================================
# 4. A INTERFACE DO USUÁRIO (Corrigida com Contexto Total)
# ==========================================
def conversar_com_ia(pergunta_usuario):
    model.eval()
    
    prompt_formatado = f"<pergunta> {pergunta_usuario} <resposta>"
    seed_words = prompt_formatado.split()
    
    for w in seed_words:
        if w not in vocab:
            return f"[Erro] A IA não conhece a palavra: '{w}'"
            
    resposta_gerada = []
    
    # Vamos deixar ela falar até 10 palavras
    for _ in range(10):
        # A MÁGICA: Agora nós recarregamos a frase INTEIRA a cada passo!
        input_tensor = torch.tensor([vocab[w] for w in seed_words]).unsqueeze(0)
        hidden = model.init_hidden(1)
        
        output, hidden = model(input_tensor, hidden)
        predicted_probs = output[0, -1, :].softmax(dim=0)
        
        idx = torch.argmax(predicted_probs).item()
        predicted_word = itos[idx]
        
        if predicted_word == "<pergunta>":
            break
            
        resposta_gerada.append(predicted_word)
        
        # A MÁGICA 2: Colamos a palavra nova no final da frase base, 
        # para ela ler tudo de novo no próximo ciclo!
        seed_words.append(predicted_word) 
        
    return ' '.join(resposta_gerada)


print("\n=======================================")
print("🧠 CHATBOT INICIADO (Digite 'sair' para fechar)")
print("=======================================")

while True:
    texto = input("\nVocê: ")
    if texto.lower() == 'sair':
        break
        
    resposta = conversar_com_ia(texto.lower())
    print(f"IA: {resposta}")