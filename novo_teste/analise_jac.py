#PARA TESTAR COLOCA ISSO NO TERMINAL:  python jaccard_bulk.py grafo_pre_requisitos.graphml 5

import networkx as nx
import pandas as pd
import itertools
import sys

def compute_jaccard(set_a, set_b):
    """
    Coeficiente de Jaccard entre dois conjuntos A e B:
    |A ∩ B| / |A ∪ B|
    """
    inter = set_a & set_b
    union = set_a | set_b
    if not union:
        return 1.0
    return len(inter) / len(union)

def main(grafo_path, output_matrix='jaccard_matrix.csv', top_n=5, output_top='jaccard_top.csv'):
    # 1. Carrega o grafo
    G = nx.read_graphml(grafo_path)

    # 2. Extrai os predecessores de cada nó
    preds = {n: set(G.predecessors(n)) for n in G.nodes()}

    # 3. Calcula Jaccard para cada par (i,j)
    cursos = list(G.nodes())
    data = []
    for i, j in itertools.combinations(cursos, 2):
        score = compute_jaccard(preds[i], preds[j])
        data.append({'course1': i, 'course2': j, 'jaccard': score})

    df = pd.DataFrame(data)
    
    # 4a. Salva a matriz completa (triangular) — se quiser a matriz completa,
    #    você pode pivotar: df.pivot(index='course1', columns='course2', values='jaccard')
    df.to_csv(output_matrix, index=False)
    print(f"✅ Matriz completa de Jaccard salva em {output_matrix}")

    # 4b. Para cada disciplina, pega os Top N mais similares
    top_list = []
    for curso in cursos:
        # Extrai pares onde aparece como course1 ou course2
        subset = df[(df.course1 == curso) | (df.course2 == curso)].copy()
        # Normaliza para sempre ter colunas [course1, course2] e extrai o outro
        subset['other'] = subset.apply(lambda r: r.course2 if r.course1 == curso else r.course1, axis=1)
        # Ordena por jaccard
        topk = subset.nlargest(top_n, 'jaccard')[['other', 'jaccard']]
        for _, row in topk.iterrows():
            top_list.append({
                'course': curso,
                'similar_course': row['other'],
                'jaccard': row['jaccard']
            })

    df_top = pd.DataFrame(top_list)
    df_top.to_csv(output_top, index=False)
    print(f"✅ Top {top_n} similares para cada curso salvo em {output_top}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python jaccard_bulk.py grafo_pre_requisitos.graphml [top_n]")
        sys.exit(1)
    grafo_path = sys.argv[1]
    top_n = int(sys.argv[2]) if len(sys.argv) >= 3 else 5
    main(grafo_path, top_n=top_n)