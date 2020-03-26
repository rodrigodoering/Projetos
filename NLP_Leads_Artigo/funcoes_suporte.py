import pandas as pd
import numpy as np
import seaborn as sb
import string
import re
import nltk
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize
from unicodedata import normalize
from time import time
from sklearn.metrics import confusion_matrix, roc_curve, auc, fbeta_score, accuracy_score, precision_score, recall_score


def limpar(sentenca, ruido, stemmer):
    '''
    Função para realizar todo o tratamento necessária no nível de sentença
    Retorna uma lista contendo o texto limpo
    '''    
    sentenca_sem_acento = normalize("NFKD", sentenca).encode("ASCII", "ignore").decode("ASCII")
    sentenca_normalizada = sentenca_sem_acento.lower().strip(string.punctuation).replace(",", "")
    array_termos = word_tokenize(sentenca_normalizada, language="portuguese")
    texto_filtrado = [termo for termo in array_termos if termo not in ruido and len(termo) > 2]
    limpar = lambda x: True if re.match("^[a-zA-Z0-9_]*$", x) is not None and bool(re.search(r"\d", x)) is False else False
    texto_limpo = list(filter(limpar, texto_filtrado))

    if stemmer:
        stemmer = nltk.stem.RSLPStemmer()
        return list(map(stemmer.stem,texto_limpo))  
    else:
        return texto_limpo
    

def matriz_confusao(y_real, y_pred, title='Matriz de Confusão', size=(10,7)):
    # Gera a matriz de confusão, recebendo o y real e o y previsto pelo modelo
    # Título por padrão é Matriz de confusão mas pode ser alterado
    matriz = confusion_matrix(y_real, y_pred)
    fig, ax = plt.subplots(figsize=size)
    sb.heatmap(matriz, annot=True, ax = ax, fmt='g'); 
    ax.set_xlabel('Previsão');ax.set_ylabel('Real'); 
    ax.set_title(title); 
    ax.xaxis.set_ticklabels([0,1]); ax.yaxis.set_ticklabels([0,1])


def plot_roc_auc(y_real, y_pred, figsize=(10,7)):
    fpr, tpr, thresholds = roc_curve(y_real, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc="lower right")
    plt.show()

def get_params_clf(clf):
    '''
    Cria string com os detalhes do treino para controle
    '''
    params = clf.get_params()
    str_params = list(map(": ".join, zip(*[params.keys(), [str(value) for value in params.values()]])))
    detalhes = 'Classificador: %s - Parâmetros: ' % clf.__class__.__name__
    for param in str_params:
        param = param + ', '
        detalhes += param
    return detalhes


def computar_metricas(clf, x_train, x_test, y_train, y_test):
    '''
    Recebe o modelo, os conjuntos de treino e teste
    Realiza o processo de previsão para teste e treino
    Computa as métricas para teste e treino
    Retorna um dicionário com os resultados, as previsões de treino e de teste
    '''
    metricas = {}
    start = time() 
    pred_train = clf.predict(x_train)
    pred_test = clf.predict(x_test)
    end = time()
    metricas['tempo_previsao'] = end - start 
    metricas['accuracy_teste'] = accuracy_score(y_test, pred_test)
    metricas['accuracy_treino'] = accuracy_score(y_train, pred_train)
    metricas['precision_teste'] =  precision_score(y_test, pred_test, average='binary')
    metricas['precision_treino'] =  precision_score(y_train, pred_train, average='binary')
    metricas['recall_teste'] =  recall_score(y_test, pred_test, average='binary')
    metricas['recall_treino'] =  recall_score(y_train, pred_train, average='binary')  
    metricas['fscore_teste'] = fbeta_score(y_test, pred_test, beta=0.5)
    metricas['fscore_treino'] = fbeta_score(y_train, pred_train, beta=0.5)
    return metricas


#def best_params(clf, param_grid, param_default):
    
    
    
    
    
    
    
    