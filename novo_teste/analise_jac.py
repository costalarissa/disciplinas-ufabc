import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 📂 Carrega o grafo de pré-requisitos salvo em .graphml
def carregar_grafo(caminho="grafo_pre_requisitos.graphml"):
    return nx.read_graphml(caminho)

# 📂 Carrega os embeddings Node2Vec do CSV
def carregar_embeddings(caminho="embeddings_node2vec.csv"):
    df = pd.read_csv(caminho, index_col='DISCIPLINA')
    return df

# 📂 Carrega profundidades dos nós (níveis no grafo)
def carregar_profundidades(caminho="profundidade_nos.txt"):
    profundidades = {}
    with open(caminho, "r") as f:
        for linha in f:
            if ":" in linha:
                node, valor = linha.strip().split(":")
                profundidades[node.strip()] = int(valor.strip())
    return profundidades

# 🔢 Cálculo da similaridade de Jaccard entre dois conjuntos
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# 🔁 Aplica Jaccard entre predecessores ou sucessores de dois nós
def similaridade_jaccard(G, a, b, tipo='predecessor'):
    try:
        if tipo == 'predecessor':
            set_a = set(G.predecessors(a))
            set_b = set(G.predecessors(b))
        else:
            set_a = set(G.successors(a))
            set_b = set(G.successors(b))
        return jaccard_similarity(set_a, set_b)
    except nx.NetworkXError:
        # Se um dos nós não estiver no grafo, retorna 0
        return 0.0

# 📏 Similaridade baseada na profundidade dos nós
def similaridade_profundidade(prof, a, b):
    if a not in prof or b not in prof:
        return 0.0
    max_p = max(prof.values())
    if max_p == 0:
        return 1.0
    return 1.0 - abs(prof[a] - prof[b]) / max_p

# 🧬 Similaridade vetorial usando cosseno (Node2Vec)
def similaridade_node2vec(embeddings, a, b):
    if a not in embeddings.index or b not in embeddings.index:
        return 0.0
    vec_a = embeddings.loc[a].values.reshape(1, -1)
    vec_b = embeddings.loc[b].values.reshape(1, -1)
    return cosine_similarity(vec_a, vec_b)[0][0]

# 🧠 Combina múltiplas similaridades em um score final
def similaridade_combinada(G, embeddings, profundidades, a, b, pesos=None):
    if pesos is None:
        pesos = {
            'jaccard_pred': 1.0,
            'jaccard_succ': 1.0,
            'profundidade': 1.0,
            'node2vec': 1.0,
        }

    sim_jaccard_pred = similaridade_jaccard(G, a, b, tipo='predecessor')
    sim_jaccard_succ = similaridade_jaccard(G, a, b, tipo='successor')
    sim_profundidade = similaridade_profundidade(profundidades, a, b)
    sim_node2vec = similaridade_node2vec(embeddings, a, b)

    total_peso = sum(pesos.values())
    score = (
        pesos['jaccard_pred'] * sim_jaccard_pred +
        pesos['jaccard_succ'] * sim_jaccard_succ +
        pesos['profundidade'] * sim_profundidade +
        pesos['node2vec'] * sim_node2vec
    ) / total_peso

    return {
        "score_combinado": score,
        "jaccard_pred": sim_jaccard_pred,
        "jaccard_succ": sim_jaccard_succ,
        "profundidade": sim_profundidade,
        "node2vec": sim_node2vec
    }

# 💡 Compara uma disciplina com todas as outras e retorna as mais similares
def mais_similares(G, embeddings, profundidades, disciplina_alvo, top_k=10):
    similares = []

    for outra in G.nodes():
        if outra == disciplina_alvo:
            continue
        sim = similaridade_combinada(G, embeddings, profundidades, disciplina_alvo, outra)
        similares.append((outra, sim["score_combinado"]))

    similares.sort(key=lambda x: x[1], reverse=True)
    return similares[:top_k]

# 🧪 Gera similaridades entre todas as disciplinas e salva em um arquivo TSV
if __name__ == "__main__":
    grafo = carregar_grafo()
    embeddings = carregar_embeddings()
    profundidades = carregar_profundidades()

    resultados = []

    # 🔄 Para cada par (a, b) distinto no grafo, calcula similaridade
    for a in grafo.nodes():
        for b in grafo.nodes():
            if a == b:
                continue
            sim = similaridade_combinada(grafo, embeddings, profundidades, a, b)
            resultados.append({
                "disciplina_a": a,
                "disciplina_b": b,
                "score_combinado": sim["score_combinado"],
                "jaccard_pred": sim["jaccard_pred"],
                "jaccard_succ": sim["jaccard_succ"],
                "profundidade": sim["profundidade"],
                "node2vec": sim["node2vec"]
            })

    # 💾 Salva os resultados em arquivo TSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv("similaridades_disciplinas.tsv", sep="\t", index=False)
    print("✅ Arquivo 'similaridades_disciplinas.tsv' gerado com sucesso.")
