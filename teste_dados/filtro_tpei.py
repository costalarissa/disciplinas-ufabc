import pandas as pd
import re
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS

# Carregar modelo spacy para português
nlp = spacy.load("pt_core_news_lg")

# ----------------------------
# UTILITÁRIOS DE TEXTO
# ----------------------------

def extrair_sigla_componentes(sigla):
    padrao = re.compile(r'^([A-Z]{3,4})(\d{4})-(\d{2})$')
    match = padrao.match(sigla)
    if match:
        area, numero, versao = match.groups()
        return {'area': area, 'numero': numero, 'versao': versao, 'valido': True}
    return {'valido': False}

def normalizar_texto(texto):
    if not texto or not isinstance(texto, str):
        return ""
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def extrair_referencias_abnt(texto):
    if not isinstance(texto, str):
        return []
    padrao = re.compile(
        r'([A-Z]{2,},\s+[A-ZÀ-ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-ÿ][a-zà-ÿ]+)*\.)'
        r'\s+(.+?)\.\s+'
        r'([A-Z][a-z]+):\s+(.+?),\s+(\d{4})',
        re.MULTILINE
    )
    matches = padrao.findall(texto)
    return [{'autor': a.strip(), 'titulo': t.strip(), 'cidade': c.strip(), 'editora': e.strip(), 'ano': y.strip()}
            for a, t, c, e, y in matches]

# ----------------------------
# PARSE TPEI
# ----------------------------

def parse_tpei(componente_str):
    valores = {'T': 0, 'P': 0, 'E': 0, 'I': 0}
    if not isinstance(componente_str, str):
        return valores
    for par in componente_str.split(','):
        if ':' in par:
            k, v = par.strip().split(':')
            if k in valores:
                valores[k] = int(v)
    return valores

def filtro_tpei(df, delta=10):
    df = df.copy()
    df[['T', 'P', 'E', 'I']] = df['componentes'].apply(lambda x: pd.Series(parse_tpei(x)))
    filtrados = []
    for i in range(len(df)):
        for j in range(i+1, len(df)):
            a, b = df.iloc[i], df.iloc[j]
            if (a['T'], a['P'], a['E'], a['I']) == (b['T'], b['P'], b['E'], b['I']):
                criterio = "exato"
            elif abs((a['T'] + a['P']) - (b['T'] + b['P'])) <= delta and min(a['T'], b['T']) > 0:
                criterio = "redistribuição"
            elif sum(abs(a[k] - b[k]) for k in ['T', 'P', 'E', 'I']) <= delta:
                criterio = "aproximado"
            else:
                continue
            filtrados.append({'sigla_a': a['sigla'], 'sigla_b': b['sigla'], 'criterio': criterio})
    return pd.DataFrame(filtrados)

# ----------------------------
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ----------------------------

def carregar_e_analisar_tsv(caminho_arquivo):
    try:
        df = pd.read_csv(caminho_arquivo, sep="\t", encoding="utf-8", engine='python')
        print("Colunas detectadas:", list(df.columns))

        df = df.rename(columns={
            'cod_disciplina_ufabc_edit': 'sigla',
            'disciplina_ufabc_edit': 'nome',
            'componentes_ufabc': 'componentes'  # <- necessário para TPEI
        })

        componentes_sigla = df['sigla'].apply(extrair_sigla_componentes)
        df['area'] = componentes_sigla.apply(lambda x: x.get('area', '') if x['valido'] else '')
        df['numero'] = componentes_sigla.apply(lambda x: x.get('numero', '') if x['valido'] else '')
        df['versao'] = componentes_sigla.apply(lambda x: x.get('versao', '') if x['valido'] else '')

        df['nome_norm'] = df['nome'].apply(normalizar_texto)

        if 'ch_disciplina_ufabc_edit' in df.columns:
            df['ch_num'] = df['ch_disciplina_ufabc_edit'].str.extract(r'(\d+)').astype(float)
        else:
            df['ch_num'] = None

        if 'bibliografia' in df.columns:
            df['referencias_abnt'] = df['bibliografia'].apply(extrair_referencias_abnt)
        else:
            df['referencias_abnt'] = None

        df_unico = df.drop_duplicates(subset='sigla')

        print("\nTop 5 disciplinas com maior carga horária:")
        print(df_unico[['sigla', 'nome', 'ch_num']].sort_values(by='ch_num', ascending=False).head(5))

        return df_unico

    except Exception as e:
        print(f"Erro ao processar arquivo TSV: {str(e)}")
        return None

# ----------------------------
# EXECUÇÃO
# ----------------------------

if __name__ == "__main__":
    df_limpo = carregar_e_analisar_tsv("Dados de Equivalência.tsv")

    if df_limpo is not None:
        print("\nExecutando filtro TPEI...")
        df_filtrados = filtro_tpei(df_limpo)
        print(df_filtrados)
        df_filtrados.to_csv("pares_filtrados_tpei.csv", index=False, encoding="utf-8")
