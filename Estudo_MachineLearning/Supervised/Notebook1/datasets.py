# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:21:46 2021

@author: rodri
"""

# Importa módulos built-ins
import warnings

# Impede mensagens de aviso de serem exibidas no notebook
warnings.filterwarnings("ignore")

# Importa pacotes gerais da comunidade
import numpy as np

# Importa funções específicas
from sklearn.datasets import make_classification


# FUNÇÃO: Load_dataset_R2
def Load_dataset_R2():
    '''
    Descrição:
    ---------
    Carrega um dataset com 𝑋 ∈ ℝ^2
    - Usa a função sklearn.datasets.make_classification para gerar as features
    
    ''' 
    # Cria um dataset sintético com parâmetros pré-definidos 
    X, Y = make_classification(
        n_samples=300, 
        n_features=2,
        n_redundant=0,
        class_sep=2.2,
        n_clusters_per_class=1,
        flip_y=0.00,
        random_state=11,
        n_classes=2
    )
    # retorna os dados
    return X, Y


# FUNÇÃO: Load_dataset_R3
def Load_dataset_R3():
    '''
    Descrição:
    ---------
    Carrega um dataset com 𝑋 ∈ ℝ^3
    - Usa a função sklearn.datasets.make_classification para gerar as features
    - A matriz X é normalizada seguindo a equação do min max scale
    
    ''' 
    # Cria um dataset sintético com parâmetros pré-definidos 
    X, Y = make_classification(
        n_samples=200, 
        n_features=3,
        n_redundant=0,
        class_sep=2.2,
        n_clusters_per_class=1,
        flip_y=0.00,
        random_state=15,
        n_classes=2
    )
    # Dimensões da matriz de features
    n_samples, n_features = X.shape
    
    # Armazena mínimos e máximos das features
    min_array = np.array([X[:, i].min() for i in range(n_features)])
    max_array = np.array([X[:, i].max() for i in range(n_features)])
    
    # Normaliza a matriz
    X = (X - np.tile(min_array, (n_samples, 1))) / (max_array - min_array)
    
    # retorna os dados
    return X, Y
    
    

# FUNÇÃO: Load_dataset_R4    
def Load_dataset_R4():
    '''
    Descrição:
    ---------
    Carrega um dataset sintético com 𝑋 ∈ ℝ^4
    - Usa a função sklearn.datasets.make_classification para gerar as features
    - A matriz X é normalizada seguindo a equação do min max scale
    - As features são reordenadas da menos informativa para a mais informativa
    
    '''  
    # Cria um dataset sintético com parâmetros pré-definidos 
    X, Y = make_classification(
        n_samples=100, n_features=4, n_informative=4,
        n_redundant=0, n_repeated=0, class_sep=1.8,
        n_clusters_per_class=1, flip_y=0.00, random_state=18,
        n_classes=2
    )
    # Dimensões da matriz de features
    n_samples, n_features = X.shape
    
    # Armazena mínimos e máximos das features
    min_array = np.array([X[:, i].min() for i in range(n_features)])
    max_array = np.array([X[:, i].max() for i in range(n_features)])
    
    # Normaliza a matriz
    X = (X - np.tile(min_array, (n_samples, 1))) / (max_array - min_array)
    
    # Cria uma matriz aumentada contendo features e target
    aug_matrix = np.c_[X,Y].T 
    
    # Reordena as features de acordo com sua correlação com o target, do menor para o maior
    permutation = np.argsort(abs(np.corrcoef(aug_matrix)[:-1, -1]))
    X[:, np.arange(len(permutation))] = X[:, permutation]
    
    # Retorna os dados
    return X, Y
     
