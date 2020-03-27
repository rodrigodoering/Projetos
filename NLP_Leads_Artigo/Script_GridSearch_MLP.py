# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:46:46 2019

@author: rodri
"""

# Importação de biliotecas built in e open source
import pandas as pd
from time import time
from sklearn.neural_network import MLPClassifier
from Database import SQLServer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from funcoes_suporte import get_params_clf, computar_metricas

# Instancia o objeto SQLServer
database = SQLServer(dsn='DSN_TCC')

# Acessa a database
database.connect()
database.set_database('DB_LIVE')

# Seleciona o dataset
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

# Grid de parâmetros 
param_grid = {
        'hidden_layer_sizes':[(50,50), (100,200), (50,100,50)],
        'alpha': [.01, .1, .5, .7],
        'random_state':[42],
        'learning_rate_init':[.0001, .001, 0.01],
        'solver':['sgd', 'adam'],
        'activation':['tanh', 'relu', 'logistic']
}

# Define o classificador
clf = MLPClassifier()

# Inicia o processo de modelagem 
clf_name = clf.__class__.__name__

print('Instanciando GridSearch')
clf_gridsearch = GridSearchCV(clf, param_grid, cv=3)

# processo de realizar o treinamento e colher os resultados para cada tamanho de amostra
print('Iniciando busca de parâmetros para %s' % clf_name)
train_start = time()
clf_gridsearch.fit(x_train, y_train)
train_end = time()

# Armazena os resultados do GridSearch
try:
    print('Salvando grid de resultados')
    cv_results = clf_gridsearch.cv_results_
    df_cv_results = pd.DataFrame(cv_results)
    df_cv_results.to_excel('cv_results.xlsx')

except Exception as e:
    print('Erro ao salvar o grid: %s' % e)

# Computa as métricas de avaliação para o modelo otimizado
metricas = computar_metricas(
    clf_gridsearch,
    x_train, 
    x_test, 
    y_train, 
    y_test
)

# Dados descritivos do treino
metricas["modelo"] = clf_name
metricas["tempo_treino"] = train_end - train_start
metricas["tamanho_amostra"] = len(x_train)
metricas["descricao_treino"] = 'Vector Space Model + Tf.Idf - GRIDSEARCH ' + get_params_clf(clf_gridsearch)

# Armazena os resultados na tabela de resultados no banco de dados
metricas_tabela = pd.DataFrame({1:metricas}, index=metricas.keys()).T
database.insert(metricas_tabela, 'tb_resultados_gerais_modelagem_final')
    
print('Fim')

