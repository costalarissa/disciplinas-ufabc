import networkx as nx
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Carrega o grafo de pré-requisitos salvo em .graphml
def carregar_grafo(caminho="grafo_pre_requisitos.graphml"):
    return nx.read_graphml(caminho)

# Carrega os embeddings Node2Vec do CSV
def carregar_embeddings(caminho="embeddings_node2vec.csv"):
    df = pd.read_csv(caminho, index_col='DISCIPLINA')
    return df

# Carrega profundidades dos nós (níveis no grafo)
def carregar_profundidades(caminho="profundidade_nos.txt"):
    profundidades = {}
    with open(caminho, "r") as f:
        for linha in f:
            if ":" in linha:
                node, valor = linha.strip().split(":")
                profundidades[node.strip()] = int(valor.strip())
    return profundidades

# Jaccard entre dois conjuntos
def jaccard_similarity(set1, set2):
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

# Aplica Jaccard entre predecessores ou sucessores
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
        return 0.0

# Similaridade de profundidade
def similaridade_profundidade(prof, a, b):
    if a not in prof or b not in prof:
        return 0.0
    max_p = max(prof.values())
    if max_p == 0:
        return 1.0
    return 1.0 - abs(prof[a] - prof[b]) / max_p

# Similaridade por Node2Vec (cosseno)
def similaridade_node2vec(embeddings, a, b):
    if a not in embeddings.index or b not in embeddings.index:
        return 0.0
    vec_a = embeddings.loc[a].values.reshape(1, -1)
    vec_b = embeddings.loc[b].values.reshape(1, -1)
    return cosine_similarity(vec_a, vec_b)[0][0]

# Combina múltiplas similaridades em score final
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

# Filtro TPEI exato: compara atributos T, P, E, I
def filtro_tpei_exato(G, a, b):
    for campo in ['T', 'P', 'E', 'I']:
        if G.nodes[a].get(campo) != G.nodes[b].get(campo):
            return False
    return True

# Escrita incremental com filtro de profundidade e TPEI
if __name__ == "__main__":
    grafo = carregar_grafo()
    embeddings = carregar_embeddings()
    profundidades = carregar_profundidades()

    caminho_saida = "similaridades_disciplinas_filtrado.tsv"
    with open(caminho_saida, mode="w", newline='', encoding='utf-8') as f_out:
        writer = csv.writer(f_out, delimiter="\t")
        # Cabeçalho
        writer.writerow([
            "disciplina_a", "disciplina_b", "score_combinado",
            "jaccard_pred", "jaccard_succ", "profundidade", "node2vec"
        ])

        # Laço otimizado com filtros
        for a in grafo.nodes():
            for b in grafo.nodes():
                if a == b:
                    continue

                # Filtro por profundidade (ex: 2 níveis no máximo)
                if abs(profundidades.get(a, 0) - profundidades.get(b, 0)) > 2:
                    continue

                # Filtro TPEI exato
                if not filtro_tpei_exato(grafo, a, b):
                    continue

                sim = similaridade_combinada(grafo, embeddings, profundidades, a, b)
                writer.writerow([
                    a, b, sim["score_combinado"],
                    sim["jaccard_pred"], sim["jaccard_succ"],
                    sim["profundidade"], sim["node2vec"]
                ])

    print(f"✅ Arquivo '{caminho_saida}' gerado com sucesso.")

# AVISO: O ARQUIVO .TSV CONTENDO O RESULTADO DAS ANÁLISES PODE DEMORAR PARA SER GERADO DEPENDENDO DA MÁQUINA UTILIZADA PARA A EXECUÇÃO!!