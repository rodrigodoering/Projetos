# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 21:35:46 2019

@author: rodri
"""

# Importação de biliotecas built in e open source
from time import time
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from funcoes_suporte import get_params_clf, computar_metricas
from Database import SQLServer

# Instancia o objeto SQLServer
database = SQLServer(dsn='DSN_TCC')

# Acessa a database
database.connect()
database.set_database('DB_LIVE')

# Carrega a tabela contendo o texto processado
df = database.select('TB_texto_processado')

# Dropa registros com campos missing
if df.texto.isnull().sum() > 0:
    df.dropna(inplace=True)

classe_proposta = ["Encaminhar","Proposta Data","Proposta Simples","Proposta Simples","Proposta Valor","Indicação"] 
df["class"] = df["class"].apply(lambda termo: 1 if termo in classe_proposta else 0)

y = df["class"]
x = pd.Series([row.split(',') for row in df["texto"].values])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

tfidf = TfidfVectorizer(
    analyzer = 'word',
    tokenizer = lambda x: x,
    preprocessor = lambda x: x,
    token_pattern = None,
    sublinear_tf = True
)

x_train = tfidf.fit_transform(x_train)
x_test = tfidf.transform(x_test)

# Define os classificadores
clf_A = MultinomialNB()
clf_B = LinearSVC(random_state=42)
clf_C = MLPClassifier(
    random_state=42, 
    verbose=True
)

# define o tamanho das amostras   
calcular_amostra = lambda x: int(len(y_train)*x)
amostras = [.25, .5, 1]
lista_amostras = [calcular_amostra(valor) for valor in amostras]

print('Iniciando modelagem')

for clf in [clf_C]:
    
    clf_name = clf.__class__.__name__
    
    resultados = {}
    
    for i, amostra in enumerate(lista_amostras):
        print('Iniciando treinamento com %s para %d amostras' % (clf_name, amostra))
        train_start = time()
        clf.fit(x_train[:amostra], y_train[:amostra])
        train_end = time()
        metricas = computar_metricas(
            clf,
            x_train[:amostra], 
            x_test, 
            y_train[:amostra], 
            y_test
        )
        metricas["modelo"] = clf_name
        metricas["tempo_treino"] = train_end - train_start
        metricas["tamanho_amostra"] = amostra
        metricas["descricao_treino"] = 'Vector Space Model + Tf.Idf (sublinear_tf = True) - ' + get_params_clf(clf)
    
        resultados[i] = metricas
    
    print('Armazenando os resultados de treino para o classificador %s' % clf_name)
    metricas_tabela = pd.DataFrame(resultados).T
    
    database.insert(metricas_tabela, 'tb_resultados_gerais_modelagem_final')

