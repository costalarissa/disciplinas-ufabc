import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Carregar o modelo e tokenizer do BERTimbau
tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

def obter_embedding_bert(texto, model=model, tokenizer=tokenizer):
    """
    Obtém o embedding semântico de um texto utilizando as camadas de atenção 9-12 do BERTimbau.
    """
    inputs = tokenizer(texto, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    # Pegando as camadas 9-12
    # A camada final de BERT (camada 12) é normalmente a mais informativa, mas podemos usar uma média das camadas intermediárias
    embeddings = outputs.last_hidden_state

    # Extrair a média das camadas 9-12
    # Assumimos que as camadas 9-12 estão nas últimas 4 dimensões do tensor
    layers = embeddings[:, 9:12, :]
    average_embedding = layers.mean(dim=1).squeeze().cpu().numpy()

    return average_embedding

def calcular_similaridade_semantica(df, model=model, tokenizer=tokenizer):
    """
    Calcula a similaridade semântica entre as disciplinas utilizando o BERTimbau.
    """
    embeddings = []

    # Obtemos o embedding para cada disciplina
    for _, row in df.iterrows():
        if pd.isna(row['disciplina_ufabc_edit']):
            embeddings.append(None)
        else:
            embedding = obter_embedding_bert(row['disciplina_ufabc_edit'], model, tokenizer)
            embeddings.append(embedding)
    
    # Adicionando os embeddings ao DataFrame
    df['embedding'] = embeddings
    
    # Calculando a similaridade de cosseno entre todos os pares de disciplinas
    similaridades = cosine_similarity(df['embedding'].tolist())
    return pd.DataFrame(similaridades, columns=df['disciplina_ufabc_edit'], index=df['disciplina_ufabc_edit'])

def aplicar_filtro_semantico(df):
    """
    Aplica o filtro semântico, calculando a similaridade semântica entre as disciplinas.
    """
    similaridade_df = calcular_similaridade_semantica(df)
    return similaridade_df
