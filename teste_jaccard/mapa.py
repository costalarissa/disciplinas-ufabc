import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Carrega o TSV
df = pd.read_csv("similaridades_disciplinas_filtrado.tsv", sep="\t")

# Seleciona 50 pares representativos
amostras = pd.concat([
    df.nlargest(10, 'score_combinado'),
    df.nsmallest(10, 'score_combinado'),
    df.nlargest(10, 'node2vec'),
    df[df['jaccard_pred'] > 0].sample(10, random_state=42),
    df.sample(10, random_state=7)
]).drop_duplicates().reset_index(drop=True)

# Indexação legível
amostras.index = amostras["disciplina_a"] + " vs " + amostras["disciplina_b"]

# Métricas para o heatmap
metricas = ["score_combinado", "jaccard_pred", "jaccard_succ", "profundidade", "node2vec"]

# Cria o heatmap
plt.figure(figsize=(14, 14))
sns.heatmap(amostras[metricas], annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

plt.title("Mapa de Calor: Similaridade Entre Disciplinas (50 pares)")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
