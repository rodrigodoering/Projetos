# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:36:59 2019

@author: rodri
"""
# Importa as biliotecas
from time import time
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
from Database import SQLServer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Acessa a database
database = SQLServer(dsn='DSN_TCC')
database.connect()
database.set_database('DB_LIVE')

# Trabalharemos com a decomposição em três Ns diferentes: 300, 500 e 1000 componentes
n_componentes = [300, 500, 1000]

# Carrega a tabela contendo o texto processado
df = database.select('TB_texto_processado')

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

# MLP com hyperparâmetros definidos após busca com GridSearchCV
clf = MLPClassifier(
    hidden_layer_sizes=(50,100,50), 
    learning_rate_init=0.001,
    alpha=0.7, 
    max_iter=500, 
    solver='adam',
    random_state=42, 
    verbose=True,
    activation='relu'
)

# Inicia o processo de modelagem
for n_comp in n_componentes:
    # Realiza decomposição com SVD
    svd = TruncatedSVD(n_components=n_comp)
    x_train_svd = svd.fit_transform(x_train)
    x_test_svd = svd.transform(x_test)

    # String com nome do classificador
    clf_name = clf.__class__.__name__
    
    # Inicia contagem do tempo de treino
    train_start = time()

    # Realiza o treinamento do modelo
    clf.fit(x_train_svd, y_train)

    # Fim do treino
    train_end = time()

    # Computa métricas de avaliação usando os dados de teste
    metricas = computar_metricas(
        clf,
        x_train_svd, 
        x_test_svd, 
        y_train, 
        y_test
    )
    metricas["modelo"] = clf_name
    metricas["tempo_treino"] = train_end - train_start
    metricas["tamanho_amostra"] = len(x_train_svd)
    metricas["descricao_treino"] = 'SVD {} Componentes + Tf.Idf (sublinear_tf = True) - '.format(n_comp) + get_params_clf(clf)
    metricas_tabela = pd.DataFrame({1:metricas}, index=metricas.keys()).T
    
    # Armazena resultados no database
    database.insert(metricas_tabela, 'tb_resultados_gerais_modelagem_final')