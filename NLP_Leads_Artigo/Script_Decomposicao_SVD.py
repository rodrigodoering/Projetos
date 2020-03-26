# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:36:59 2019

@author: rodri
"""
# Importa as biliotecas
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from Database import SQLServer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Instancia o objeto SQLServer
database = SQLServer(dsn='DSN_TCC')

# Acessa a database
database.connect()
database.set_database('DB_LIVE')

# Trabalharemos com a decomposição em três Ns diferentes: 300, 500 e 1000 componentes
n_componentes = [300, 500, 1000]

# Por padrão trunca os dados inseridos como teste nas tabelas criadas no banco de dados
for n_comp in n_componentes:
    database.query('TRUNCATE TABLE TB_SVD_{}_TRAIN'.format(n_comp), commit=True)
    database.query('TRUNCATE TABLE TB_SVD_{}_TEST'.format(n_comp), commit=True)

# Carrega a tabela contendo o texto processado
df = database.select('TB_texto_processado_com_stopwords_sem_stemm')

# Dropa registros com campos missing
if df.texto.isnull().sum() > 0:
    df.dropna(inplace=True)

# Realiza o mapeamento para as classes
classe_proposta = ["Encaminhar","Proposta Data","Proposta Simples","Proposta Simples","Proposta Valor","Indicação"] 
df["class"] = df["class"].apply(lambda termo: 1 if termo in classe_proposta else 0)

# Separa dataset em variável dependente (y) e variável independente (x)
y = df["class"]
x = pd.Series([row.split(',') for row in df["texto"].values])

# Cria conjuntos de treino e teste
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Instancia o vetorizador Tf.Idf 
tfidf = TfidfVectorizer(
    analyzer = 'word',
    tokenizer = lambda x: x,
    preprocessor = lambda x: x,
    token_pattern = None
)

# Realiza transformação com Tf.Idf
x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

for n_comp in n_componentes:
    # Realiza decomposição com SVD
    svd = TruncatedSVD(n_components=n_comp)
    x_train_svd = svd.fit_transform(x_train)
    x_test_svd = svd.transform(x_test)

    # Processo para permitir armazenamento de maneira eficiente no banco de dados
    dense_array_train = x_train_svd.tolist()
    dense_array_test = x_test_svd.tolist()
    
    # Condensa os arrays numéricos e uma string com os valores separados em vírgula
    str_dense_train = list(map(lambda dense_array: ",".join(str(x) for x in dense_array), dense_array_train))
    str_dense_test = list(map(lambda dense_array: ",".join(str(x) for x in dense_array), dense_array_test))

    # Constroí o dataframe que será armazenado no banco de dados
    dense_df_train = pd.DataFrame({'vetor':str_dense_train, 'class':y_train})
    dense_df_test = pd.DataFrame({'vetor':str_dense_test, 'class':y_test})

    # Insere os dataframes nas respectivas tabelas criadas no banco de dados
    database.insert(dense_df_train, 'TB_SVD_{}_TRAIN'.format(n_comp))
    database.insert(dense_df_test, 'TB_SVD_{}_TEST'.format(n_comp))

