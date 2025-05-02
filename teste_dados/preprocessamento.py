import pandas as pd

def parse_tpei(texto):
    if not isinstance(texto, str):
        return [0, 0, 0, 0]
    partes = {'T': 0, 'P': 0, 'E': 0, 'I': 0}
    for item in texto.split():
        chave = item[0]
        valor = item[1:]
        if chave in partes and valor.isdigit():
            partes[chave] = int(valor)
    return [partes['T'], partes['P'], partes['E'], partes['I']]

def filtro_tpei(df):
    if 'componentes' in df.columns:
        df[['T', 'P', 'E', 'I']] = df['componentes'].apply(lambda x: pd.Series(parse_tpei(x)))
        print("Coluna 'componentes' encontrada. Componentes TPEI extraídos.")
    else:
        df['T'] = 0
        df['P'] = 0
        df['E'] = 0
        df['I'] = 0
        print("Coluna 'componentes' NÃO encontrada. Colunas TPEI preenchidas com zero.")
    return df

# Atenção: não inclua nenhum processamento aqui se for importar esse script em outro lugar.
