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

# Importa funções específicas
from numpy.linalg import norm


# Customizando algumas variáveis
e = np.e #Euler coeff -> 2.71828182...
pi = np.pi #Pi -> 3.141592...
root = lambda x: x**(1/2)
square = lambda x: x**2
cube = lambda x: x**3


# FUNÇÃO: get_angle
def get_angle(u, v, return_degrees=False, **kwargs):
    """
    Descrição:
    ---------
    O dot product (ou inner product) pode ser escrito como: 
    
    ⟨𝑤,𝑥⟩ = ‖𝑤‖‖𝑥‖ cos(𝜃)
    
    Essa função isola e retorna o ângulo 𝜃:
        
    𝜃 = arccos (⟨𝑤,𝑥⟩ / ‖𝑤‖‖𝑥‖)   
    
    Argumentos:
    -----------
    u, v - (numpy.arrays) vetores que formam o ângulo theta
    return_degrees - (bool) se True, retorna theta em graus, se False, em radianos (padrão)
    **kwargs - parâmetros adicionais de np.linalg.norm para computar a norma do vetor
    
    Retorna:
    -------
    o ângulo theta entre os vetores u e v
    """
    
    # Computa o ângulo theta
    theta = np.arccos(np.dot(u, v) / (norm(u, **kwargs) * norm(v, **kwargs)))
    
    if return_degrees:
        # retorna theta em graus, já que por padrão np.arccos retorna em radianos
        return round(np.degrees(theta))
    
    # retorna theta em radianos
    return theta
 


# FUNÇÃO: unit_vec
def unit_vec(vec, l=1, **kwargs):
    '''
    Descrição:
    ---------
    Aplica a equação abaixo para computar o vetor de input com uma norma arbitrária definida por lambda (𝜆)
    
    𝑣' = 𝜆 * (𝑣/‖𝑣‖)
    
    Argumentos:
    ----------
    vec - numpy.array uni-dimensional
    l - coeficiente "lambda" na equação. Por padrão, 𝜆 = 1
    **kwargs - parâmetros adicionais de np.linalg.norm para computar a norma do vetor
    
    Retorna:
    -------
    O vetor com a norma arbitrária. Por padrão retorna-se o vetor unitário ‖𝑣‖ = 1
   
    '''
    # Retorna o vetor na norma desejada. Por padrão é o vetor unitário, |v| = 1
    return l * (vec / norm(vec, **kwargs))



# FUNÇÃO: hyperplane_function
def hyperplane_function(w, b=0, t=0):
    '''
    Descrição:
    ---------
    Cria uma função para desenhar o hiperplano n-dimensional aplicando a equação:
        
    𝑥𝑑 = −⟨𝑤′,𝑥′⟩ + 𝜏−𝑏 / 𝑤𝑑
    
    Onde ⟨𝑤′,𝑥′⟩ = ∑_𝑘 (𝑤𝑘/𝑤𝑑)𝑥𝑘 com 𝑘∈{1, 2 ⋯ 𝑑−1}, 
    
    A equação retorna 𝑥𝑑, que representa o respectivo valor do último componente de
    um vetor 𝑥 = [𝑥1, 𝑥2 ⋯ 𝑥𝑑−1, 𝑥𝑑] que satisfaça a equação ⟨𝑤,𝑥⟩ + 𝑏 = 𝜏, de forma que a função
    final retornada (um callable) deve receber como argumento os 𝑑−1 valores primeiros valores
    de 𝑥 (representados como o vetor 𝑥′)  e retornar o respectivo valor de 𝑥𝑑
    
    Argumentos:
    ----------
    w - vetor de coeficientes (representa 𝑤 na equação)
    b - coeficiente linear / bias do modelo, por padrão 0 (representa 𝑏 na equação)
    t - threshold do modelo, por padrão 0 (representa 𝜏 na equação)
    
    Retorna:
    Função que recebe um vetor ou matriz 𝑥 e retorna os respectivos valores 𝑥𝑑 para desenhar o hiperplano 
    
    '''
    # O array de pesos precisa ser unidimensional
    if w.ndim > 1:
        w = np.squeeze(w)
        
    # Computa o vetor dos 𝑑−1 primeiros componentes e divide o vetor por 𝑤𝑑
    w_reduzido = w[:-1] / w[-1]
    
    # Computa o intercept do hiperplano
    b_reduzido = (t - b) / w[-1]
    
    # Retorna o callable
    return lambda x: -np.dot(x, w_reduzido.T) + b_reduzido   


def z_score(mean, std):
    return lambda x: (x - mean) / std

# FUNÇÃO: Normal_pdf
def normal_prob_density(mean, std):
    return lambda x: 1 / (std * root(2*pi)) * e**(-square(x - mean) / (2 * square(std)))     

def normal_cumulative_density(mean, std):
    z_x = z_score(mean, std)(x)
    integrand = lambda x: e**(-square(x)/2)
    return lambda x: 1/root(2*np.pi) * quad(integrand, -np.inf, z_x)[0]


