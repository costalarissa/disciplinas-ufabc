import pandas as pd
import re
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS

# Carregar modelo spaCy para português
nlp = spacy.load("pt_core_news_lg")

def extrair_sigla_componentes(sigla):
    """Extrai componentes da sigla de disciplina usando regex."""
    padrao = re.compile(r'^([A-Z]{3,4})(\d{4})-(\d{2})$')
    match = padrao.match(sigla)

    if match:
        area, numero, versao = match.groups()
        return {
            'area': area,
            'numero': numero,
            'versao': versao,
            'valido': True

        }
    return {'valido': False}

def normalizar_texto(texto):
    """Normaliza texto com remoção de stopwords e lematização."""
    if not texto or not isinstance(texto, str):
        return ""
    doc = nlp(texto.lower())
    tokens = [token.lemma_ for token in doc
              if not token.is_stop and not token.is_punct]
    return " ".join(tokens)

def carregar_e_preprocessar_catalogo(caminho_arquivo):
    """Carrega catálogo de Excel e realiza pré-processamento básico."""
    try:
        # Carregar dados
        df = pd.read_excel("""copiar diretorio""")
        
        # Verificar colunas obrigatórias
        colunas_obrigatorias = ['sigla', 'nome', 'creditos_t', 'creditos_p']
        for coluna in colunas_obrigatorias:
            if coluna not in df.columns:
                raise ValueError(f"Coluna obrigatória ausente: {coluna}")
      
        # Extrair componentes da sigla
        componentes_sigla = df['sigla'].apply(extrair_sigla_componentes)
        df['area'] = componentes_sigla.apply(lambda x: x.get('area', '') if x['valido'] else '')
        df['numero'] = componentes_sigla.apply(lambda x: x.get('numero', '') if x['valido'] else '')
        df['versao'] = componentes_sigla.apply(lambda x: x.get('versao', '') if x['valido'] else '')
        # Normalizar textos
        if 'ementa' in df.columns:
            df['ementa_norm'] = df['ementa'].apply(normalizar_texto)
        if 'objetivos' in df.columns:
            df['objetivos_norm'] = df['objetivos'].apply(normalizar_texto)
        return df
    except Exception as e:
        print(f"Erro ao processar catálogo: {str(e)}")
        return None
