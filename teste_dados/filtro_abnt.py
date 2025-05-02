import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def extrair_caracteristicas_abnt(texto):
    """
    Extrai informações bibliográficas como autores, títulos, editoras, anos.
    Retorna um dicionário com essas características.
    """
    autores = re.findall(r'^[A-ZÁÉÍÓÚÇÑ\s]+', texto, flags=re.MULTILINE)
    anos = re.findall(r'\b(19|20)\d{2}\b', texto)
    editoras = re.findall(r':\s?([^.,\n]+)', texto)
    titulos = re.findall(r'\.\s([^.\n]+)\.', texto)

    return {
        'autores': set(autores),
        'anos': set(anos),
        'editoras': set(editoras),
        'titulos': set(titulos)
    }

def calcular_sobreposicao(bib1, bib2):
    """
    Calcula a sobreposição Jaccard de autores, títulos, editoras e anos.
    """
    c1 = extrair_caracteristicas_abnt(bib1)
    c2 = extrair_caracteristicas_abnt(bib2)

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a | b else 0

    return {
        'jaccard_autores': jaccard(c1['autores'], c2['autores']),
        'jaccard_titulos': jaccard(c1['titulos'], c2['titulos']),
        'jaccard_editoras': jaccard(c1['editoras'], c2['editoras']),
        'jaccard_anos': jaccard(c1['anos'], c2['anos']),
    }

def extrair_dataset_sobreposicao(df):
    """
    Para cada linha do DataFrame, calcula os atributos de sobreposição.
    """
    dados = []
    for _, row in df.iterrows():
        if pd.isna(row['bibliografia_ufabc']) or pd.isna(row['bibliografia_fora']):
            continue
        features = calcular_sobreposicao(row['bibliografia_ufabc'], row['bibliografia_fora'])
        features['rotulo'] = row.get('rotulo', None)  # pode ser None se não houver rótulo
        dados.append(features)
    
    return pd.DataFrame(dados)

def treinar_classificador_bibliografia(df_feat):
    """
    Treina um classificador SVM com base nos dados de sobreposição.
    """
    df_train = df_feat.dropna(subset=['rotulo'])  # só usa linhas com rótulo

    X = df_train.drop(columns=['rotulo'])
    y = df_train['rotulo']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = make_pipeline(SVC(kernel='linear', probability=True))
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    return clf

def aplicar_filtro_bibliografia(df, modelo):
    """
    Aplica o classificador a novos pares de disciplinas.
    """
    df_feat = extrair_dataset_sobreposicao(df)
    df_feat['prob_equivalente'] = modelo.predict_proba(df_feat.drop(columns=['rotulo'], errors='ignore'))[:, 1]
    return df_feat
