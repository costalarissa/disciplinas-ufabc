import pandas as pd
import networkx as nx
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch

# 1. Extrator Avançado de Componentes TPEI
def extrair_componentes_tpei(texto):
    """
    Extrai os componentes TPEI (Teoria, Prática, Extensão, Individual) de um texto.
    """
    componentes = {'T': None, 'P': None, 'E': None, 'I': None}
    
    if 'Teoria' in texto:
        componentes['T'] = texto.split('Teoria')[1].split('.')[0].strip()
    if 'Prática' in texto:
        componentes['P'] = texto.split('Prática')[1].split('.')[0].strip()
    if 'Extensão' in texto:
        componentes['E'] = texto.split('Extensão')[1].split('.')[0].strip()
    if 'Individual' in texto:
        componentes['I'] = texto.split('Individual')[1].split('.')[0].strip()
    
    return componentes

# 2. Processador e Analisador de Pré-requisitos
def criar_grafo_pre_requisitos(df):
    """
    Cria um grafo direcionado para os pré-requisitos entre as disciplinas e calcula a centralidade.
    """
    G = nx.DiGraph()
    
    # Criando os nós (disciplinas) e as arestas (pré-requisitos)
    for _, row in df.iterrows():
        if pd.notna(row['disciplina 1']):
            G.add_edge(row['nome'], row['disciplina 1'])
        if pd.notna(row['disciplina 2']):
            G.add_edge(row['nome'], row['disciplina 2'])
        if pd.notna(row['disciplina 3']):
            G.add_edge(row['nome'], row['disciplina 3'])
        if pd.notna(row['disciplina 4']):
            G.add_edge(row['nome'], row['disciplina 4'])
    
    # Calculando as medidas de centralidade
    centralidade = nx.betweenness_centrality(G)
    
    return G, centralidade

# 3. Analisador de Texto Semântico (BERTimbau)
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

def obter_embedding_bert(texto):
    """
    Obtém o embedding semântico de um texto utilizando BERTimbau.
    """
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Pegando a média das camadas de BERT
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def calcular_similaridade_semantica(df):
    """
    Calcula a similaridade semântica entre as ementas/objetivos das disciplinas.
    """
    embeddings = []
    
    for _, row in df.iterrows():
        ementa = row['nome']  # Usando o nome da disciplina ou ementa
        embeddings.append(obter_embedding_bert(ementa))
    
    similaridade = cosine_similarity(embeddings)
    return pd.DataFrame(similaridade, columns=df['nome'], index=df['nome'])

# Função principal do módulo de enriquecimento de dados
def enriquecer_dados(df):
    """
    Enriquecer os dados com componentes TPEI, pré-requisitos e análise semântica.
    """
    # Extrair componentes TPEI
    df['componentes_tpei'] = df['nome'].apply(extrair_componentes_tpei)

    # Criar grafo de pré-requisitos e calcular centralidade
    grafo, centralidade = criar_grafo_pre_requisitos(df)
    df['centralidade'] = df['nome'].map(centralidade)

    # Calcular similaridade semântica
    similaridade_semantica = calcular_similaridade_semantica(df)

    def formatar_tpei_dict(d):
        return ','.join(f'{k}:{v}' for k, v in d.items() if v is not None)
    
    df['componentes'] = df['componentes_tpei'].apply(formatar_tpei_dict)

    return df, similaridade_semantica, grafo, centralidade
