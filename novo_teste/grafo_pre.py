import pandas as pd
import networkx as nx
import unicodedata
import requests
from io import StringIO
from node2vec import Node2Vec
import numpy as np

# ✅ Normaliza uma string removendo acentos e espaços extras
def normalize_str(s: str) -> str:
    return (
        unicodedata.normalize('NFKD', s)
        .encode('ASCII', 'ignore')
        .decode('utf-8')
        .strip()
    )

# ✅ Versão específica para nomes de disciplinas (já normaliza e coloca lowercase)
def normalizar_nome(nome: str) -> str:
    return normalize_str(nome).lower()

# ✅ Baixa o catálogo da UFABC direto do GitHub
def carregar_catalogo():
    url = "https://raw.githubusercontent.com/angeloodr/disciplinas-ufabc/main/catalogo_disciplinas_graduacao_2024_2025.tsv"
    print("🔄 Baixando catálogo de disciplinas...")
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep='\t')

    # Normaliza os nomes das colunas
    df.columns = [
        normalize_str(col).upper().replace(' ', '_')
        for col in df.columns
    ]
    print("✅ Download bem-sucedido!")
    print("📝 Colunas disponíveis:", df.columns.tolist())
    return df

# ✅ Constrói grafo com arestas de pré-requisito com base na coluna RECOMENDACAO
def construir_grafo(df: pd.DataFrame) -> nx.DiGraph:
    print("📌 Construindo grafo de pré-requisitos...")
    G = nx.DiGraph()
    total_arestas = 0

    # Cria um dicionário de nome normalizado → sigla
    mapping = {
        normalizar_nome(row['DISCIPLINA']): row['SIGLA']
        for _, row in df.iterrows()
        if pd.notna(row['SIGLA'])
    }

    # Adiciona todos os nós com metadados
    for _, row in df.iterrows():
        sigla = row['SIGLA']
        nome = row['DISCIPLINA']
        if pd.isna(sigla): 
            continue
        G.add_node(sigla, nome=nome)

    # Adiciona as arestas baseadas nas recomendações
    for _, row in df.iterrows():
        curso = row['SIGLA']
        if pd.isna(curso):
            continue

        recs = row.get('RECOMENDACAO', '')
        if pd.isna(recs) or not isinstance(recs, str):
            continue

        for rec in recs.split(';'):
            rec = rec.strip()
            if not rec:
                continue
            rec_norm = normalizar_nome(rec)
            if rec_norm in mapping:
                prereq = mapping[rec_norm]
                if not G.has_edge(prereq, curso):
                    G.add_edge(prereq, curso, tipo='pre_requisito')
                    total_arestas += 1

    print(f"✅ Grafo criado com {G.number_of_nodes()} nós e {total_arestas} arestas.")
    return G

# ✅ Remove ciclos do grafo para que seja um DAG (necessário para cálculo de profundidade)
def remover_ciclos(G: nx.DiGraph) -> nx.DiGraph:
    print("🔁 Removendo ciclos (modo eficiente)...")
    G_copy = G.copy()
    removidos = 0
    try:
        while True:
            ciclo = nx.find_cycle(G_copy, orientation='original')
            if ciclo:
                # Remove a primeira aresta do ciclo
                u, v, _ = ciclo[0]
                G_copy.remove_edge(u, v)
                removidos += 1
    except nx.NetworkXNoCycle:
        pass  # Não há mais ciclos

    print(f"✅ Ciclos removidos: {removidos}")
    return G_copy

# ✅ Gera embeddings estruturais com Node2Vec
def gerar_embeddings_node2vec(G: nx.DiGraph, dimensions=64, walk_length=10, num_walks=50, workers=1):
    print("🔄 Gerando embeddings estruturais (Node2Vec)...")
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=workers)
    model = node2vec.fit(window=5, min_count=1, batch_words=4)
    embeddings = {node: model.wv[node] for node in G.nodes()}
    print("✅ Embeddings gerados!")
    return embeddings

# ✅ Calcula profundidade de cada nó no grafo (camada curricular)
def calcular_profundidade(G: nx.DiGraph):
    print("📏 Calculando profundidade no grafo (DAG)...")
    profundidades = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        if not preds:
            profundidades[node] = 0
        else:
            profundidades[node] = max(profundidades[p] for p in preds) + 1
    print("✅ Profundidade calculada!")
    return profundidades

# ✅ Salva os embeddings em CSV
def salvar_embeddings_csv(embeddings, caminho="embeddings_node2vec.csv"):
    df = pd.DataFrame.from_dict(embeddings, orient='index')
    df.index.name = 'DISCIPLINA'
    df.to_csv(caminho)
    print(f"💾 Embeddings salvos em: {caminho}")

# ✅ Fluxo principal
if __name__ == "__main__":
    # Etapa 1 - Carregar catálogo
    df = carregar_catalogo()

    # Etapa 2 - Construir grafo
    grafo = construir_grafo(df)
    nx.write_graphml(grafo, "grafo_pre_requisitos.graphml")
    print("📁 Arquivo salvo: grafo_pre_requisitos.graphml")

    # Etapa 3 - Gerar embeddings estruturais
    embeddings = gerar_embeddings_node2vec(grafo)
    salvar_embeddings_csv(embeddings)

    # Etapa 4 - Remover ciclos e calcular profundidade
    grafo_sem_ciclos = remover_ciclos(grafo)
    profundidades = calcular_profundidade(grafo_sem_ciclos)

    # Etapa 5 - Salvar profundidades em TXT
    with open("profundidade_nos.txt", "w") as f:
        for node, prof in profundidades.items():
            f.write(f"{node}: {prof}\n")
    print("📁 Profundidades salvas em: profundidade_nos.txt")
