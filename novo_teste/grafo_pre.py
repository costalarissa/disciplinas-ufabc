import pandas as pd
import networkx as nx
import unicodedata
import requests
from io import StringIO

# 1. Normalização geral de strings
def normalize_str(s: str) -> str:
    return (
        unicodedata.normalize('NFKD', s)
        .encode('ASCII', 'ignore')
        .decode('utf-8')
        .strip()
    )

# 2. Função para normalizar o nome das disciplinas (remover acentos e colocar em lowercase)
def normalizar_nome(nome: str) -> str:
    return normalize_str(nome).lower()

# 3. Baixa e carrega o catálogo, normalizando colunas
def carregar_catalogo():
    url = "https://raw.githubusercontent.com/angeloodr/disciplinas-ufabc/main/catalogo_disciplinas_graduacao_2024_2025.tsv"
    print("🔄 Baixando catálogo de disciplinas...")
    resp = requests.get(url)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text), sep='\t')
    # Normaliza cabeçalhos: sem acentos, maiúsculas, sem espaços
    df.columns = [
        normalize_str(col).upper().replace(' ', '_')
        for col in df.columns
    ]
    print("✅ Download bem-sucedido!")
    print("📝 Colunas disponíveis:", df.columns.tolist())
    return df

# 4. Função para montar o grafo de pré-requisitos
def construir_grafo(df: pd.DataFrame) -> nx.DiGraph:
    print("📌 Construindo grafo de pré-requisitos...")
    G = nx.DiGraph()
    total_arestas = 0

    # Dicionário nome_normalizado -> SIGLA
    mapping = {
        normalizar_nome(row['DISCIPLINA']): row['SIGLA']
        for _, row in df.iterrows()
        if pd.notna(row['SIGLA'])
    }

    # Adiciona todos os nós
    for _, row in df.iterrows():
        sigla = row['SIGLA']
        nome = row['DISCIPLINA']
        if pd.isna(sigla): 
            continue
        G.add_node(sigla, nome=nome)

    # Para cada disciplina, lê recomendações e cria arestas
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
            # Só cria aresta se a disciplina recomendada existir no catálogo
            if rec_norm in mapping:
                prereq = mapping[rec_norm]
                if not G.has_edge(prereq, curso):
                    G.add_edge(prereq, curso, tipo='pre_requisito')
                    total_arestas += 1

    print(f"✅ Grafo criado com {G.number_of_nodes()} nós e {total_arestas} arestas.")
    return G

# 5. Executa tudo
if __name__ == "__main__":
    df = carregar_catalogo()
    grafo = construir_grafo(df)
    nx.write_graphml(grafo, "grafo_pre_requisitos.graphml")
    print("📁 Arquivo salvo: grafo_pre_requisitos.graphml")
