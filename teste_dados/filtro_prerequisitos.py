import networkx as nx
import pandas as pd
from itertools import combinations
from collections import defaultdict

def construir_grafo_prerequisitos(df):
    """
    Constrói um grafo direcionado com base nos pré-requisitos entre disciplinas.
    """
    G = nx.DiGraph()

    for _, row in df.iterrows():
        disciplina = row['cod_disciplina_ufabc_edit']
        prereqs = [row[f'disciplina {i}'] for i in range(1, 10) if pd.notna(row.get(f'disciplina {i}'))]
        
        # Adiciona nós
        G.add_node(disciplina)
        for prereq in prereqs:
            G.add_node(prereq)
            G.add_edge(prereq, disciplina)  # prereq → disciplina

    return G

def calcular_metricas_grafo(G):
    """
    Calcula centralidade de grau e betweenness dos nós no grafo.
    """
    grau_centralidade = nx.degree_centrality(G)
    intermediação = nx.betweenness_centrality(G)

    df_metricas = pd.DataFrame({
        'disciplina': list(G.nodes),
        'grau_centralidade': [grau_centralidade.get(n, 0) for n in G.nodes],
        'betweenness': [intermediação.get(n, 0) for n in G.nodes]
    })

    return df_metricas

def similaridade_jaccard(df):
    """
    Calcula similaridade de Jaccard entre disciplinas com base nos pré-requisitos.
    """
    prereq_dict = {}
    for _, row in df.iterrows():
        cod = row['cod_disciplina_ufabc_edit']
        prereqs = set([row[f'disciplina {i}'] for i in range(1, 10) if pd.notna(row.get(f'disciplina {i}'))])
        prereq_dict[cod] = prereqs

    jaccard_results = []
    for a, b in combinations(prereq_dict.keys(), 2):
        intersec = len(prereq_dict[a] & prereq_dict[b])
        uniao = len(prereq_dict[a] | prereq_dict[b])
        sim = intersec / uniao if uniao else 0
        jaccard_results.append((a, b, sim))

    df_jaccard = pd.DataFrame(jaccard_results, columns=['disciplina_A', 'disciplina_B', 'sim_jaccard'])
    return df_jaccard

def filtro_prerequisitos(df):
    """
    Retorna métricas de grafo e similaridade Jaccard para análise de pré-requisitos.
    """
    grafo = construir_grafo_prerequisitos(df)
    metricas = calcular_metricas_grafo(grafo)
    jaccard = similaridade_jaccard(df)
    return grafo, metricas, jaccard
