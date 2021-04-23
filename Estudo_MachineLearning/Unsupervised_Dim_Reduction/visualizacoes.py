

import pandas as pd
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def numpy_convert(array, to_vec=False):
    '''
    Argumentos:
    array: lista de valores a ser validada
    '''
    # Testa se obj é uma lista built in do python ou se é numpy.ndarray
    if isinstance(array, list):
        array = np.array(array)
    # testa de o obj é unidimensional ou bi-dimensional
    # as funções de visualização exigem arrays 2-D
    if to_vec:
        return array
    else:
        if array.ndim == 1:
            return np.array([array])
        else:
            return array

        
def number_to_string(ndarray):
    '''
    Argumentos
    ndarray: array matemático que será convertido em string
    '''
    # Arredonda os valores para 3 casas decimais        
    rounded = (round(x,4) for x in ndarray)
    # testa se existem apenas valores inteiros no vetor
    if ndarray.dtype == np.float64:
        # constroi uma lista contendo os valores como strings já preparados para visualização
        return ('%.2f' % val if str(val)[-2:] != '.0' else '%d' % val for val in rounded)
    else:
        return map(str, rounded)


def display_vec(V):
    V = numpy_convert(V, to_vec=True)
    # Gera o código latex
    latex_code = '<br>$\\begin{bmatrix}%s\\end{bmatrix}$<br><br>'
    # cria a string contendo os valores da matriz
    add = '& '.join(number_to_string(V))
    # Exibe a matriz
    display(Markdown(latex_code % add))



# Função para exibir vetores e matrizes em Latex Markdown
def display_matrix(M, n_rows=None, n_cols=None):
    '''
    Argumentos da função
    M: Pandas.DataFrame ou 2-D Numpy.array contendo vetor ou matriz a ser plotada com markdown
    '''
    # Testa se M é um Pandas.DataFrame
    if isinstance(M, pd.core.frame.DataFrame):
        # Transforma em numpy.array
        M = numpy_convert(M.values)
    else:
        M = numpy_convert(M)

    dims_matriz = M.shape

    if n_rows:
        M = M[:n_rows]
    if n_cols:
        M = M[:, :n_cols]

    # Gera o código latex
    latex_code = '<br>$\\begin{bmatrix}%s\\end{bmatrix}$<br><br>'
    # cria a string contendo os valores da matriz
    M_str = (number_to_string(vec) for vec in M)
    add = '\\\\'.join('&'.join(vec) for vec in M_str)
    # Exibe a matriz
    display(Markdown(latex_code % add))
    print('Dimensões da matriz: (%s x %s)' % (dims_matriz[0], dims_matriz[1]))
    print()


    
# Função para plotar vetores
def plot_vecs(V, labels=None, fontsize=14):
    '''
    Argumentos da função
    V: array de vetores, pode ser lista ou numpy array
    labels: Rótulos posicionais por vetor para o gráfico (se houverem), default é None
    !! len(labels) == len(V) !!
    fontsize: tamanha da fonte dos rótulos, padrão 14 mas pode alterar
    '''
    V = numpy_convert(V)
    # constrói o plano cartesiano e eixos X e Y
    plt.axhline(0, c='black', lw=0.5)
    plt.axvline(0, c='black', lw=0.5)
    
    # Ajusta área de plot de acordo com tamanho dos vetores
    dimensoes = (V.min() - 3, V.max() + 2)
    plt.xlim(dimensoes)
    plt.ylim(dimensoes)
    
    # bloco de código que plota vetores e labels
    
    n = len(V)
    # procedimento para plotar vetor único
    if n == 1:
        x = V[0][0]
        y = V[0][1]
        plt.quiver(0, 0, x, y, range(n), angles='xy', scale_units='xy', scale=1)
        
        # em caso de vetor único, testa que tipo de obj é labels
        if not labels:
            pass
        else:
            # função suporta label como string única
            if isinstance(labels, str):
                label = labels
            else:
                label = labels[0]
            # plota o rótulo
            plt.text(x, y, label, fontsize=fontsize)
            
    # procedimento para plotar uma matriz de vetores
    else:
        U = V.T[0]
        W = V.T[1]
        plt.quiver([0,0], [0,0], U, W, range(n), angles='xy', scale_units='xy', scale=1)
        
        # plota os rótulos dos vetores na ponta de maneira posicional
        # um rótulo para um vetor
        for vetor, label in zip(V, labels):
            #plota o rótulo
            plt.text(vetor[0], vetor[1], label, fontsize=fontsize)
    # exibe o gráfico final pronto
    plt.show()