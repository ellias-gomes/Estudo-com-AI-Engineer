import torch
import torch.nn as nn
import torch.optim as optim
import os

# CONFIGURAÇÕES TÉCNICAS
ARQUIVO_TREINO = 'conhecimento_gigante.txt'
ARQUIVO_PESOS = 'chatbot_pesos.pth'
DISPOSITIVO = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. CARREGAMENTO E VOCABULÁRIO ROBUSTO
if not os.path.exists(ARQUIVO_TREINO):
    with open(ARQUIVO_TREINO, 'w', encoding='utf-8') as f:
        f.write("<pergunta> ola <resposta> oi, sou a ia do ellias\n")

with open(ARQUIVO_TREINO, 'r', encoding='utf-8') as f:
    text = f.read().lower()

words = text.split()
palavras_unicas = sorted(list(set(words)))

# O SEGREDO: Reservamos o ID 0 para palavras que a IA não conhece (<UNK>)
vocab = {palavra: i + 1 for i, palavra in enumerate(palavras_unicas)}
vocab['<UNK>'] = 0 
itos = {i: palavra for palavra, i in vocab.items()}

# 2. ARQUITETURA DO MODELO
class ChatbotPro(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        return self.fc(output), hidden
    
    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_dim).to(DISPOSITIVO)

# Instanciando
model = ChatbotPro(len(vocab), 16, 32).to(DISPOSITIVO)

# 3. LÓGICA DE PERSISTÊNCIA (SALVAR/CARREGAR)
def treinar():
    print(f"Ensinando {len(words)} palavras para a IA...")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(300):
        total_loss = 0
        hidden = model.init_hidden(1)
        for i in range(len(words) - 3):
            # Janela deslizante de 3 palavras
            contexto = [vocab[w] for w in words[i:i+3]]
            alvo = vocab[words[i+3]]
            
            inputs = torch.tensor([contexto]).to(DISPOSITIVO)
            target = torch.tensor([alvo]).to(DISPOSITIVO)
            
            optimizer.zero_grad()
            output, _ = model(inputs, hidden)
            loss = criterion(output[0, -1, :].unsqueeze(0), target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 50 == 0:
            print(f"Época {epoch+1} | Erro: {total_loss:.4f}")
    
    torch.save(model.state_dict(), ARQUIVO_PESOS)
    print("Cérebro salvo com sucesso!")

if os.path.exists(ARQUIVO_PESOS):
    print("Carregando memória do HD...")
    model.load_state_dict(torch.load(ARQUIVO_PESOS))
else:
    treinar()

# 4. CHAT COM CONTEXTO E UNK
def responder(frase):
    model.eval()
    tokens = [vocab.get(w, 0) for w in f"<pergunta> {frase.lower()} <resposta>".split()]
    
    # SALVAMOS O TAMANHO DA PERGUNTA ORIGINAL
    tamanho_pergunta = len(tokens) 
    
    for _ in range(15): # Limite de palavras na resposta
        input_tensor = torch.tensor([tokens]).to(DISPOSITIVO)
        output, _ = model(input_tensor, model.init_hidden(1))
        idx = torch.argmax(output[0, -1, :]).item()
        
        if itos.get(idx) == '<pergunta>' or itos.get(idx) == '\n': break
        tokens.append(idx)
        
    # O SEGREDO: Pegamos apenas os tokens gerados DEPOIS da pergunta!
    tokens_resposta = tokens[tamanho_pergunta:]
    return ' '.join([itos[i] for i in tokens_resposta if itos[i] not in ['<pergunta>', '<resposta>']])

import os

print("\n=======================================")
print("🧠 IA PRONTA E ATIVA")
print("💡 Digite '/ensinar' para adicionar novos conhecimentos.")
print("🚪 Digite 'sair' para encerrar.")
print("=======================================")

while True:
    user = input("\nVocê: ")
    
    if user.lower() == 'sair': 
        break
        
    # O SISTEMA DE AUTOALIMENTAÇÃO
    elif user.lower() == '/ensinar':
        print("\n--- MODO DE APRENDIZADO ---")
        nova_pergunta = input("Quando o usuário perguntar: ")
        nova_resposta = input("A IA deve responder: ")
        
        # 1. Salva no arquivo de texto
        with open(ARQUIVO_TREINO, 'a', encoding='utf-8') as f:
            f.write(f"\n<pergunta> {nova_pergunta.lower()} <resposta> {nova_resposta.lower()}")
            
        # 2. Destrói o cérebro antigo para forçar o retreino!
        if os.path.exists(ARQUIVO_PESOS):
            os.remove(ARQUIVO_PESOS)
            
        print("✅ Anotado no banco de dados!")
        print("⚠️ Feche o programa e rode novamente para eu treinar meu novo cérebro.")
        continue
        
    # FLUXO NORMAL DE CONVERSA
    else:
        print(f"IA: {responder(user)}")