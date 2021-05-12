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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

    
# FUNÇÃO: generate_grid_coords
# Gera um grid de coordenadas para plotar uma visualização. Suporta receber uma função para computar
# a última dimensão em função das outras anteriores. Convencionalmente, utiliza-se a função numpy.meshgrid
# para gerar esse grid, mas acredito ser mais direto utilizar diretamente o numpy broadcasting

def generate_grid_coords(n_dims, min_val=-1, max_val=1, n_vals=5, dim_func=None, return_flat=False):
    base_var = np.linspace(min_val, max_val, n_vals)
    update_axis = lambda i: (slice(np.newaxis),) + (np.newaxis,) * i
    # Cria o grid de valores, len(grid) = n_dims
    grid = np.broadcast_arrays(*(base_var[update_axis(i)] for i in range(n_dims)))
    
    # Se for passada uma função para a última dimensão
    if dim_func is not None:
        dim_grid_shape = tuple(n_vals for i in range(n_dims))
        last_dim_grid = dim_func(np.array([vec.flatten() for vec in grid[:-1]]).T).reshape(dim_grid_shape)
        grid[-1] = last_dim_grid
        
    if return_flat:
        # retorna os valores do grid como rowvectors de shape (n_vals^n_dims x n_dims)
        return np.array([vec.flatten() for vec in grid]).T

    return grid


        
        
        
        
        
        
        
        
