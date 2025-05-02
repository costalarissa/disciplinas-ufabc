# ajustar_tsv.py
import pandas as pd

CAMINHO_ARQUIVO = "Dados de Equivalência.tsv"

try:
    df = pd.read_csv(CAMINHO_ARQUIVO, sep="\t", encoding="utf-8")

    for col in ["bibliografia_ufabc", "bibliografia_fora"]:
        if col not in df.columns:
            df[col] = ""

    df.to_csv(CAMINHO_ARQUIVO, sep="\t", index=False, encoding="utf-8")
    print("✅ Colunas 'bibliografia_ufabc' e 'bibliografia_fora' garantidas no arquivo.")
except Exception as e:
    print(f"❌ Erro ao ajustar o arquivo TSV: {e}")
