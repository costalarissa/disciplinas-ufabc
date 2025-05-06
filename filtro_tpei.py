import pandas as pd

def abrir_arquivo(caminho_arquivo):
    df = pd.read_csv(caminho_arquivo, sep='\t', usecols=["disciplina_a", "disciplina_b"])
    df.rename(columns={"disciplina_a": "SIGLA_A", "disciplina_b": "SIGLA_B"}, inplace=True)
    return df

def calcular_creditos_totais(caminho_arquivo):
    df = pd.read_excel(caminho_arquivo, usecols=["SIGLA", "TPEI"])
    df = df.drop_duplicates(subset="SIGLA")
    df[['T', 'P', 'E', 'I']] = df['TPEI'].str.split('-', expand=True)
    df['T'] = pd.to_numeric(df['T'], errors='coerce')
    df['P'] = pd.to_numeric(df['P'], errors='coerce')
    df['total_creditos'] = df['T'] + df['P']
    return df[['SIGLA', 'total_creditos']]


def aplicar_filtro_tpei(df_pares, df_disciplinas):
    print("🔄 Aplicando filtro TPEI...")
    
    # Criar dicionário para mapeamento rápido de sigla -> créditos
    creditos_dict = dict(zip(df_disciplinas['SIGLA'], df_disciplinas['total_creditos']))
    
    pares_filtrados = []
    tpei_dif = {}
    pares_excluidos = []
    
    for _, row in df_pares.iterrows():
        sigla_a = row['SIGLA_A']
        sigla_b = row['SIGLA_B']
        
        if sigla_a not in creditos_dict or sigla_b not in creditos_dict:
            pares_excluidos.append((sigla_a, sigla_b, "Sigla não encontrada"))
            continue
        
        creditos_a = creditos_dict[sigla_a]
        creditos_b = creditos_dict[sigla_b]
        
        if creditos_a > creditos_b:
            pares_excluidos.append((sigla_a, sigla_b, f"Créditos A({creditos_a}) > B({creditos_b})"))
        else:
            pares_filtrados.append(row)
            tpei_dif[(sigla_a, sigla_b)] = creditos_b - creditos_a
    
    df_filtrado = pd.DataFrame(pares_filtrados)
    
    print(f"✅ Filtro TPEI aplicado: {len(df_filtrado)} pares mantidos, {len(pares_excluidos)} excluídos")
    
    return df_filtrado, tpei_dif

caminho_csv = "jaccard_baixado.csv"
caminho_excel = "catalogo_disciplinas_graduacao_2024_2025.xlsx"

df_pares = abrir_arquivo(caminho_csv)
df_disciplinas = calcular_creditos_totais(caminho_excel)

df_filtrado, diferencas = aplicar_filtro_tpei(df_pares, df_disciplinas)
