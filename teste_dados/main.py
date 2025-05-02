import pandas as pd

from preprocessamento import carregar_e_preprocessar_catalogo as preprocessar_tpei
from filtro_tpei import carregar_e_analisar_tsv, filtro_tpei
from enriquecer_dados import enriquecer_dados
from filtro_abnt import extrair_dataset_sobreposicao, treinar_classificador_bibliografia, aplicar_filtro_bibliografia

# 1. Carregar e pré-processar
print("🔄 Carregando e pré-processando dados...")
url = "https://raw.githubusercontent.com/angeloodr/disciplinas-ufabc/refs/heads/main/teste_dados/Dados%20de%20Equival%C3%AAncia.tsv"
df = carregar_e_analisar_tsv(url)

# Garante colunas de bibliografia (caso venham de arquivo externo sem edição)
for col in ["bibliografia_ufabc", "bibliografia_fora"]:
    if col not in df.columns:
        df[col] = ""

df = preprocessar_tpei(df)

# 2. Enriquecimento de dados
print("⚙️ Enriquecendo dados com TPEI, grafo e BERT...")
df_enriquecido, sim_semantica, grafo, centralidade = enriquecer_dados(df)

# 3. Filtro TPEI (baseado em comparação entre disciplinas)
print("🧮 Aplicando filtro TPEI...")
pares_tpei = filtro_tpei(df_enriquecido)
pares_tpei.to_csv("saida_pares_tpei.csv", index=False)
print(f"✔️ {len(pares_tpei)} pares TPEI salvos.")

# 4. Filtro Bibliografia ABNT
if "bibliografia_ufabc" in df.columns and "bibliografia_fora" in df.columns:
    print("📚 Calculando sobreposição bibliográfica e treinando SVM...")
    df_abnt_feat = extrair_dataset_sobreposicao(df)
    modelo_abnt = treinar_classificador_bibliografia(df_abnt_feat)
    result_abnt = aplicar_filtro_bibliografia(df, modelo_abnt)
    result_abnt.to_csv("saida_filtro_abnt.csv", index=False)
    print(f"✔️ Filtro ABNT aplicado a {len(result_abnt)} pares.")
else:
    print("⚠️ Colunas de bibliografia ausentes. Pulando filtro ABNT.")

# 5. Salvar dados enriquecidos
df_enriquecido.to_csv("saida_enriquecida.csv", index=False)
print("✅ Dados enriquecidos salvos em 'saida_enriquecida.csv'")
