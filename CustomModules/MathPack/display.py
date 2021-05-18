# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:03:02 2021

@author: rodrigo doering neves
@email: rodrigodoeringneves@gmail.com
"""

# Importa módulos built-ins
import warnings

# Impede mensagens de aviso de serem exibidas no notebook
warnings.filterwarnings("ignore")

# Importa pacotes gerais da comunidade
import numpy as np
import pandas as pd

# Importa funções específicas
from sklearn.metrics import accuracy_score
from IPython.display import display, Markdown


# FUNÇÃO: numpy_convert 
# Função para testar (e converter se necessário) listas de python em arrays numpy
def numpy_convert(array):
    '''
    Argumentos:
    ----------
    array - lista de valores a ser validada
    '''
    if isinstance(array, list):
        array = np.array(array)
    return array


# FUNÇÃO: number_to_string 
# Converte um array em uma string compatível com a linguagem Latex
def number_to_string(array):
    '''
    Argumentos:
    ----------
    array: numpy.array contendo os vetores que será convertido em string
    '''      
    rounded = (round(x,4) for x in array)
    if array.dtype == np.float64:
        # retorna as duas primeiras casas decimais se forem valores decimais
        return ('%.2f' % val if str(val)[-2:] != '.0' else '%d' % val for val in rounded)
    else:
        return map(str, rounded)


def display_expression(expr):
    '''
    Argumentos:
    ----------
    expr: string - expressão em LaTex para ser exibida
    ''' 
    if not isinstance(expr, str):
        raise('Passe apenas strings')
    display(Markdown(expr))

    
# FUNÇÃO: display_vec
# Printa no Stdout um vetor através de linguagem latex de visualização
def display_vec(V, label=False):
    '''
    Argumentos
    ----------
    V - numpy.array contendo o vetor a ser exibido com Latex
    label - rótulo do vetor, também será exibido em Latex
    '''
    V = numpy_convert(V)
    
    if V.ndim > 1:
        V = V.reshape(-1)
        
    if label:
        latex_code = '<br>${} = '.format(label) + '\\begin{bmatrix}%s\\end{bmatrix}$<br>'
    else:
        latex_code = '<br>$\\begin{bmatrix}%s\\end{bmatrix}$<br>'
        
    # cria a string latex final contendo os valores do vetor
    add = '\\\\'.join(number_to_string(V))
    display_expression(latex_code % add)



# FUNÇÃO: display_matrix
# Printa no Stdout um vetor através de linguagem latex de visualização
def display_matrix(M, n_rows=None, n_cols=None, label=False):
    '''
    Argumentos:
    ----------
    M - Pandas.DataFrame ou 2-D Numpy.array contendo vetor ou matriz a ser plotada com markdown
    n_rows - quantidade máxima de linhas a serem exibidas
    n_cols - quantidade máxima de colunas a serem exibidas
    
    '''
    if isinstance(M, pd.core.frame.DataFrame):
        M = numpy_convert(M.values)
    else:
        M = numpy_convert(M)
    
    # dimensões da matriz
    dims_matriz = M.shape
    
    if n_rows:
        M = M[:n_rows]
        
    if n_cols:
        M = M[:, :n_cols]

    # Gera o código latex
    if label:
        latex_code = '<br>${} = '.format(label) + '\\begin{bmatrix}%s\\end{bmatrix}$<br><br>'
    else:
        latex_code = '<br>$\\begin{bmatrix}%s\\end{bmatrix}$<br><br>'
        
    # cria a string contendo os valores da matriz e exibe a matriz
    M_str = (number_to_string(vec) for vec in M)
    add = '\\\\'.join('&'.join(vec) for vec in M_str)
    display_expression(latex_code % add)
    print('Dimensões da matriz: (%s x %s)' % (dims_matriz[0], dims_matriz[1]))
    print()