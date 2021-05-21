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


# Função SVD_decomp
def SVD_decomp(X, n_comps=2, normalize=True):
    '''
    Descrição
    ---------
    1º Aplica a decomposição de valor singular (Singular Value Decomposition) de uma matriz X: 
    
    𝑋 = 𝑈𝑆𝑉^𝑇
    
    Onde 𝑈 e 𝑉 são os autovetores esquerdos e direitos de 𝑋 
    e 𝑆 é uma matriz diagonal contendo os valores singulares
    
    2º Projeta 𝑋 na nova base multiplicando este pelos autovetores direitos: 𝑋′=𝑋𝑉'
    
    **Se a matriz 𝑋 for ou estiver normalizada com o z-score, e portanto, 
    tiver média = 0 e desvio padrão = 1, então a decomposição com SVD será equivalente 
    à decomposição com PCA, onde os autovalores da eigendecomposition da matriz de
    covariância de 𝑋 serão equivalentes aos valores singulares decompostos de 𝑋:  
    
    cov(𝑋) = 𝑉𝐷𝑉−1 ⇔ 𝐷 = 𝑆²/(𝑛−1)
       
    Onde 𝑛 é o número de samples da matriz 𝑋. As implementações de PCA (como a do próprio sklearn) costumam utilizar o SVD
    por ser um cálculo mais performático e numéricamente estável dado que pula a etapa de criar a matriz de covariância de 𝑋 
  
    
    Argumentos
    ----------
    X - np.ndarray matriz de valores para serem decompostos com shape n_samples x n_dims
    n_comps - dimensionalidade do dataset reduzido. n_comps < n_dims. Basicamente, quantos dos autovetores são usados para decompor X
    normalize - se verdadeiro, aplica o z-score aos dados originais, e a decomposição passa a se comportar como o PCA
    '''

    if isinstance(X, list):
        X = np.array(X)
        
    if normalize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        
    # Realiza a decomposição de X com o SVD
    U,S,V = svd(X)
    X_reduzido = np.dot(X, V.T[:,0:n_comps])
    
    # Computa os autovetores e a variância explicada da decomposição
    autovalores = (S**2)/(len(X) - 1)
    var_exp = sum(autovalores[:n_comps] / sum(autovalores))
    return var_exp, X_reduzido



# FUNÇÕES RELACIONADAS A DISTRIBUIÇÃO NORMAL

def z_score(mean, std):
    
    """Implementa a equação do z-score e retorna um callable que recebe um valor de x. 
    Assume-se X segue uma distribuição normal:
    
    X∼N(μ,σ²)
    
    O z-score é escrito conforme a fórmula abaixo:
    
    z_x = (x − μ) / σ
    
    Argumentos:
    ----------
    mean - média aritmética da distribuição normal
    std - desvio padrão da distribuição normal
    
    Retorna:
    --------
    Callable
    """
    return lambda x: (x - mean) / std


# FUNÇÃO: Normal_pdf
def normal_prob_density(mean, std):
    """Implementa a função de densidade probabilidade de uma distribuição normal
    equivalente à stats.norm.pdf
    Com base na equação abaixo:
    
    f(x)=(σ√2π)^−1 ⋅ e^(−(x−μ)/2σ)²

    Argumentos:
    ----------
    mean - média aritmética da distribuição normal
    std - desvio padrão da distribuição normal
    
    Retorna:
    --------
    Callable
    """
    return lambda x: 1 / (std * root(2*pi)) * e**(-square(z_score(mean, std)(x)) / 2) 


def normal_cumulative_density(mean, std):
    """Implementa a função de densidade cumulativa
    equivalente à stats.norm.cdf

    Φ_μ,σ2(x)= 1√2π ∫ e(−(x−μ)2σ)² dx

    Argumentos:
    ----------
    mean - média aritmética da distribuição normal
    std - desvio padrão da distribuição normal
    
    Retorna:
    --------
    Callable
    """
    pdf = normal_prob_density(mean, std) 
    return lambda x: integrate.quad(pdf, -np.inf, x)[0]


