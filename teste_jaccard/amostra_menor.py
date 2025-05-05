import pandas as pd

# Carrega o arquivo grande
df = pd.read_csv("similaridades_disciplinas_filtrado.tsv", sep="\t")

# Amostra representativa (ex: 100 mil linhas ALEATÓRIAS!), caso precise ajuste o n para quanto achar melhor.
amostra = df.sample(n=100000, random_state=42)

# Salva em um novo arquivo menor
amostra.to_csv("similaridades_amostra_menor.tsv", sep="\t", index=False)

print("✅ Arquivo reduzido salvo como 'similaridades_amostra_menor.tsv'")
