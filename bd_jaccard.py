import gdown
import pandas as pd

# 1. Baixar o arquivo do Google Drive
file_id = "1wzEmQ4D6EMjPkyMCQK6OPwQbOuNwIr5r"
url = f"https://drive.google.com/uc?id={file_id}"
output = "jaccard_baixado.csv"

gdown.download(url, output, quiet=False)

# 2. Ler o CSV com separador de tabulação
df = pd.read_csv(output, sep='\t')

# 3. Filtrar as colunas desejadas
df_limpo = df[['disciplina_a', 'disciplina_b', 'score_combinado']]

# 4. Mostrar as primeiras linhas
print(df_limpo.head())

# 5. (Opcional) Salvar a tabela limpa
df_limpo.to_csv("disciplinas_score.csv", index=False)

# 6. Preparar listas para uso no modelo (se necessário)
discipline_pairs = list(zip(df_limpo['disciplina_a'], df_limpo['disciplina_b']))
combined_scores = df_limpo['score_combinado'].tolist()

# Exemplo de verificação:
print(f"\nTotal de pares: {len(discipline_pairs)}")
print(f"Exemplo de par: {discipline_pairs[0]}, score: {combined_scores[0]}")
