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
def numpy_convert(array):
    '''
    Descrição:
    ---------
    Função para testar (e converter se necessário) listas de python em arrays numpy
     
    Argumentos:
    ----------
    array - lista de valores a ser validada
    
    Retorna:
    -------
    array numpy
    
    '''
    # Testa se obj é uma lista built in do python ou se é numpy.ndarray
    if isinstance(array, list):
        array = np.array(array)
        
    # Retorna array
    return array


# FUNÇÃO: number_to_string     
def number_to_string(array):
    '''
    Descrição
    ---------
    Converte um array em uma string compatível com a linguagem Latex
    
    Argumentos
    ----------
    array: numpy.array contendo os vetores que será convertido em string
    
    Retorna
    -------
    string formatada a partir do array
    '''
    # Arredonda os valores para 3 casas decimais        
    rounded = (round(x,4) for x in array)
    # testa se existem apenas valores inteiros no vetor
    if array.dtype == np.float64:
        # constroi uma lista contendo os valores como strings já preparados para visualização
        return ('%.2f' % val if str(val)[-2:] != '.0' else '%d' % val for val in rounded)
    else:
        return map(str, rounded)


# FUNÇÃO: display_vec
def display_vec(V, label=False):
    '''
    Descrição:
    ---------
    Printa no Stdout um vetor através de linguagem latex de visualização
    
    Argumentos:
    ----------
    V - numpy.array contendo o vetor a ser exibido com Latex
    label - rótulo do vetor, também será exibido em Latex
    
    '''
    # Testa e converte vetor em numpy.array
    V = numpy_convert(V)
    
    # Testa e adapta dimensões do array. Nessa função espera-se um array unidimensiona
    if V.ndim > 1:
        V = V.reshape(-1)
    
    # Testa se o rótulo foi passado
    if label:
        # gera o código latex
        latex_code = '<br>${} = '.format(label) + '\\begin{bmatrix}%s\\end{bmatrix}$<br>'
    else:
        latex_code = '<br>$\\begin{bmatrix}%s\\end{bmatrix}$<br>'
        
    # cria a string latex final contendo os valores do vetor
    add = '\\\\'.join(number_to_string(V))
    
    # Exibe a matriz
    display(Markdown(latex_code % add))



# FUNÇÃO: display_matrix
def display_matrix(M, n_rows=None, n_cols=None, label=False):
    '''
    Descrição:
    ---------
    Printa no Stdout um vetor através de linguagem latex de visualização
    
    Argumentos:
    ----------
    M - Pandas.DataFrame ou 2-D Numpy.array contendo vetor ou matriz a ser plotada com markdown
    n_rows - quantidade máxima de linhas a serem exibidas
    n_cols - quantidade máxima de colunas a serem exibidas
    
    '''
    # Testa se M é um Pandas.DataFrame
    if isinstance(M, pd.core.frame.DataFrame):
        # Transforma em numpy.array
        M = numpy_convert(M.values)
    else:
        M = numpy_convert(M)
    
    # dimensões da matriz
    dims_matriz = M.shape
    
    # Limita a quantidade de valores a serem printados
    # Improtante para grandes matrizes uma vez que a exibição do código latex é computacionalmente pesado
    if n_rows:
        # limita por quantidade de linhas
        M = M[:n_rows]
        
    if n_cols:
        # limita por quantidade de colunas
        M = M[:, :n_cols]

    # Gera o código latex
    if label:
        latex_code = '<br>${} = '.format(label) + '\\begin{bmatrix}%s\\end{bmatrix}$<br><br>'
    else:
        latex_code = '<br>$\\begin{bmatrix}%s\\end{bmatrix}$<br><br>'
        
    # cria a string contendo os valores da matriz
    M_str = (number_to_string(vec) for vec in M)
    add = '\\\\'.join('&'.join(vec) for vec in M_str)
    
    # Exibe a matriz
    display(Markdown(latex_code % add))
    print('Dimensões da matriz: (%s x %s)' % (dims_matriz[0], dims_matriz[1]))
    print()

    

# FUNÇÃO: plot_vecs
# OBSERVAÇÃO: 
#   -Melhorar essa função em breve acrescentando funcionalidades presentes na função plot_vecs_3d
#   -Incluir argumentos de obj Axes, personalização do estilo do vetor e melhora na personalização do plot em si
def plot_vecs(V, labels=None, fontsize=14, ax_lims=None):
    '''
    Descrição:
    ---------
    Função usada para plotar vetores em R^2
    Converte os arrays em objetos do Numpy por padrão. Não suporta vetores em espaços dimensionais maiores
    
    Argumentos:
    ----------
    V - array de vetores, pode ser lista ou numpy array
    labels - Rótulos posicionais por vetor para o gráfico (se houverem), default é None
    !! len(labels) == len(V) !!
    fontsize - tamanha da fonte dos rótulos, padrão 14 mas pode alterar
    
    '''
    V = numpy_convert(V)
    # constrói o plano cartesiano e eixos X e Y
    plt.axhline(0, c='black', lw=0.5)
    plt.axvline(0, c='black', lw=0.5)
    
    # Ajusta área de plot de acordo com tamanho dos vetores ou com valores pre-definidos
    if ax_lims is not None:
        plt.xlim(ax_lims[0])
        plt.xlim(ax_lims[1])

    else:
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



# FUNÇÃO: plot_vecs_3d  
def plot_vecs_3d(
    vecs, 
    labels=None, 
    label_offset=0.1, 
    ax_offset=0, 
    colors=None, 
    styles=None, 
    arrow_lenght=.08, 
    size=(10,7), 
    elev=10, 
    azim=20, 
    ax_lims=None,
    fontsize=12, 
    row_vec=True,
    ax_obj=None,
    return_ax_obj=False
    ):
    
    '''
    Descrição:
    ---------
    Função adaptada de 'plot_vecs' para plotar vetores em R^3
    
    Argumentos:
    ----------
    vecs - array uni ou bi-dimensional de vetores a serem plotads
    labels - Padrão NoneType, rótulos em código latex dos vetores 
    label_offset - Padrão 0.1, deslocamento do rótulo quando plotado
    ax_offset - Eixo (axis) para deslocar o label (de acordo com label_offset)
    colors - Lista contendo os argumentos de 'color' da função ax.quiver (portanto, caso passado, deve conter um por vetor)
    styles - Lista contendo osoargumentos de 'linestyle' da função ax.quiver
    size - tamanho da figura em pixels
    elev - ângulo de elevação do plot 3D
    azim - ângulo de rotação do plot 3D
    ax_lims - limite dos eixos do plot. Se nada for passado, por padrão min=0 e max=1
    fontsize - tamanho de fonte dos rótulos
    row_vec - Bool, Orientação dos vetores em "vecs". Por padrão espera-se vetores linha; 
             Se os vetores forem passados como colunas, esse argumento deve ser falso.
    ax_obj - Suporta receber um objeto Axes já instanciado ou cria um novo objeto caso seja o primeiro/único plot
    return_ax_obj - Bool, se verdadeiro, retorna o objeto Axes para que o plot possa ser complementado
    
    Retorna:
    -------
    O objeto Axes caso return_ax_obj seja verdadeiro
    
    '''
    # testa e converte array de vetores
    vecs = numpy_convert(vecs)
    
    # Adapta as dimensões do array para o resto da função
    # A função retorna erro com np.arrays unidimensionais
    if vecs.ndim == 1:
        vecs = np.expand_dims(vecs, axis=0)
    
    # Testa a orientação dos vetores
    if not row_vec:
        vecs = vecs.T
    
    # Armazena as dimensões do array
    n_vecs = vecs.shape[0]
    n_dims = vecs.shape[1]
    
    # Estrutura os argumentos personalizáveis por vetor
    colors = ['black' for i in range(n_vecs)] if colors is None else colors
    styles = ['solid' for i in range(n_vecs)] if styles is None else styles
    
    # Testa se um objeto Axes foi passado como argumento
    if ax_obj is None:
        fig = plt.figure(figsize=size)
        # Inicia uma instancia do Axes para projeção em 3D
        ax = fig.gca(projection='3d')
    else:
        ax = ax_obj
    
    # Passa argumentos de elevação e rotação do plot
    ax.view_init(elev=elev, azim=azim)
    
    # For loop para plotar os vetores passando os argumentos desejados
    for vec, color, style in zip(vecs, colors, styles):
        ax.quiver(
            0, 0, 0, vec[0],vec[1],vec[2],
            linestyle=style, color=color, 
            arrow_length_ratio=arrow_lenght
        )
        
    # Testa se rótulos foram passados    
    if labels is not None:
        
        # Estrutura os rótulos dos vetores
        labels = [labels] if isinstance(labels, str) else labels
        
        # For loop para plotar os rótulos em seus respectivos vetores
        for vec, label in zip(vecs, labels):
            x,y,z = (vec[i] + label_offset if i == ax_offset else vec[i] for i in range(n_dims))
            ax.text(x, y, z, label, fontsize=fontsize)
    
    # Define os limites dos eixos
    xlim, ylim, zlim = [(0,1) for i in range(3)] if ax_lims is None else ax_lims
    
    # Aplica os limites
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    
    # Aplica a legenda dos eixos
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    ax.set_zlabel('$x_3$', fontsize=16, rotation=0)
    
    # Retorna instancia do objeto Axes
    if return_ax_obj:
        return ax



# FUNÇÃO: scatter_plot
def scatter_plot(X, Y, colormap=plt.cm.coolwarm, size=(5,3), marker='.', title=None, ax_obj=None, return_ax_obj=False, **kwargs):
    """
    Descrição:
    ---------
    Gera o gráfico de dispersão para dados bidimensionais
    
    Argumentos:
    ----------
    X - numpy.array 
    Y - numpy.array ou lista
    colormap - escala de cor do plot. Argumento "c"
    size - tupla contendo tamanha da figura em pixels, default é 6x4
    marker - Tipo de marcador para representar cada ponto no gráfico
    ax_obj - Suporta receber um objeto Axes já instanciado ou cria um novo objeto caso seja o primeiro/único plot
    return_ax_obj - Bool, se verdadeiro, retorna o objeto Axes para que o plot possa ser complementado
    
    Retorna:
    -------
    O objeto Axes caso return_ax_obj seja verdadeiro    
    """
    if ax_obj is None:
        # gera os subplots
        fig, ax = plt.subplots(figsize=size)
    else:
        ax = ax_obj
        
    # cria plot de dispersão
    ax.scatter(X, Y, cmap=colormap, marker=marker, **kwargs)
    
    # Se uma string for passada em "title", será exibida como título do plot
    if isinstance(title, str):
        ax.set_title(title)
    
    # Retorna instancia do objeto Axes
    if return_ax_obj:
        return ax



def scatter_plot_3d(X, Y, elev=10, azim=20, size=(10,10), ax_obj=None, return_ax_obj=False, cmap=plt.cm.coolwarm, **kwargs):
    '''
    Descrição:
    ---------
    Exibe o gráfico de dispersão tridimensional para os dados passados    
        
    Argumentos:
    ----------
    X - numpy.ndarray de dimensões (n_samples x 3)
    Y - numpy.array de dimensões (n_samples,)
    elev - Elevação do plot
    azim - Rotação do plot
    size - tamanho do plot, em pixels x pixels
    return_ax - se verdadeiro, retorna objeto do plot, para complementar o plot se necessário
    cmap - colormap do gráfico de dispersão, por padrão utilizo o "cool warm", mas é possível alterar
    kwargs - quaisquer outros parâmetros do gráfico de dispersão
    
    Retorna:
    -------
    Objeto Axes do plot, para complementar o plot se necessário se argumento return_ax for verdadeiro
    
    '''
    # Testa se um objeto Axes foi passado como argumento
    if ax_obj is None:
        fig = plt.figure(figsize=size)
        # Inicia uma instancia do Axes para projeção em 3D
        ax = fig.gca(projection='3d')
    else:
        ax = ax_obj
    
    # Gera a visualização dos dados aplicando os parâmetros do scatter plot
    sc = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=Y, cmap=cmap, **kwargs)
    fig.colorbar(sc, fraction=0.012)
    
    # Define angulação do plot, permite rotacionar
    ax.view_init(elev=elev, azim=azim)
    
    # Define os rótulos do gráfico
    ax.set_xlabel('$x_1$', fontsize=16)
    ax.set_ylabel('$x_2$', fontsize=16)
    ax.set_zlabel('$x_3$', fontsize=16, rotation=0) 
    
    # Retorna instancia do obj Axes
    if return_ax_obj:    
        return ax


'''
# FUNÇÃO: get_meshgrid
def get_meshgrid(X, step=.02, limit_values=None):
    """
    Descrição:
    ---------
    Responsável por receber os dados de treino e gerar um grid com todas as combinações de valores (𝑥1,𝑥2)  
    dentro de um range de valores definidos  
    
    Argumentos:
    ----------
    X - Numpy.array de shape (n_samples x 2)
    step - argumento step da função np.arange(start, stop, step), o intervalo entre os valores gerados
    limit_values - valores mínimos e máximos para gerar o grid, se verdadeiro deve ser uma tupla
    
    Retorna:
    -------
    Grid de valores para x1 e x2
    
    """
    # Testa se foram passados valores pré-definidos para gerar o grid
    if limit_values is not None:
        X1, X2 = limit_values
    else: 
        # computa tuplas contendo menor e maior valor para X1 e X2
        X1, X2 = list(map(lambda x: (x.min() - 1, x.max() + 1), [X[:, 0], X[:, 1]]))
    
    # gera grade de valores
    return np.meshgrid(np.arange(X1[0], X1[1], step), np.arange(X2[0], X2[1], step))

'''

# FUNÇÃO: plot_superficie_decisao
def plot_superficie_decisao(
    clf, X, Y, 
    size=(7,5), 
    title='Superfície de Decisão', 
    step=0.02, 
    print_acc=False, 
    levels=False, 
    limit_values=None, 
    plotar_vec=None, 
    vec_label='\vec{v}', 
    plot_grid=False, 
    ax_obj=None,
    return_ax_obj=False,
    **kwargs
    ):
    '''
    Descrição:
    ---------
    Gera um plot de superfície de decisão no espaço R^2 
    A função plot_superficie_decisao espera um objeto classificador que contenha o método predict
    
    Argumentos:
    ----------
    clf - classificador que contenha a função predict e retorne um numpy.array (n_samples, ) 
    X - Numpy.array de shape (n_samples x 2)
    Y - Numpy.array de shape (n_samples, 1)
    size - tupla contendo tamanho do plot em pixels que será passada em plt.subplots, padrão é None,
    title - Título do gráfico
    print_acc - se verdadeiro, exibe a acurácia do modelo no Stdout
    levels - Escala numérica da superfície de decisão. Quanto maior, mais pesado computacionalmente. Valores maiores
            geram gradientes mais difusos para modelos de output contínuo (regressão)
    limit_values - Argumento passado a função "get_meshgrid"
    plotar_vec - A função ainda permite plotar um vetor bidimensional arbitrário partindo da origem
    vec_label - também é possível nomear e rotular o vetor passado no plot em linguagem Latex
    plot_grid - se verdadeiro, exibe a grade no plot
    ax_obj - Suporta receber um objeto Axes já instanciado ou cria um novo objeto caso seja o primeiro/único plot
    return_ax_obj - Bool, se verdadeiro, retorna o objeto Axes para que o plot possa ser complementado
    **kwargs - demais parâmetros que serão passados em ax.scatter
    
    Retorna:
    -------
    O objeto Axes caso return_ax_obj seja verdadeiro    
    
    '''
    # aplica a função get_meshgrid e gera as coordenadas do grid de pontos
    
    # Testa se foram passados valores pré-definidos para gerar o grid
    if limit_values is not None:
        X1, X2 = limit_values
    else: 
        # computa tuplas contendo menor e maior valor para X1 e X2
        X1, X2 = list(map(lambda x: (x.min() - 1, x.max() + 1), [X[:, 0], X[:, 1]]))
    
    # gera grade de valores                  
    coords_x1, coords_x2 = np.meshgrid(np.arange(X1[0], X1[1], step), np.arange(X2[0], X2[1], step))
    
    # realiza as previsões para todas as coordenadas
    preds = clf.predict(np.c_[coords_x1.ravel(), coords_x2.ravel()]).reshape(coords_x1.shape)
    
    # se verdadeiro, printa a acurácia do modelo
    if print_acc:
        acc = accuracy_score(Y, clf.predict(X))
        print('Acurácia do modelo: %.3f\n' % acc)
    
    if ax_obj is None:
        # gera os subplots
        fig, ax = plt.subplots(figsize=size)
    
    # é interessante definir o argumento "levels" com valores altos para modelos de regressão
    if levels:
        ax.contourf(coords_x1, coords_x2, preds, levels, cmap=plt.cm.coolwarm, alpha=0.5)
    else:
        ax.contourf(coords_x1, coords_x2, preds, cmap=plt.cm.coolwarm, alpha=0.5)
    
    # plota dos dados de treino na mesma figura
    sc = ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=15, edgecolors="k", **kwargs)
    
    # exibe a barra de cor do plot
    fig.colorbar(sc, ax=ax)
    
    # define o título da visualização
    ax.set_title(title)
    
    # Se passado, plota um vetor 2D arbitrário
    if plotar_vec is not None:
        ax.quiver(0, 0, plotar_vec[0], plotar_vec[1], angles='xy', scale_units='xy', scale=1, color='black')
        ax.text(plotar_vec[0], plotar_vec[1], vec_label, fontsize=14)
    
    # Exibe a grade do plot
    if plot_grid:
        ax.grid()
    
    # Retorna instancia do obj Axes
    if return_ax_obj:    
        return ax


def generate_grid_coords(n_dims, min_val=-1, max_val=1, n_vals=5, dim_func=None, return_flat=False):
    base_var = np.linspace(min_val, max_val, n_vals)
    update_axis = lambda i: (slice(np.newaxis),) + (np.newaxis,) * i
    grid = np.broadcast_arrays(*(base_var[update_axis(i)] for i in range(n_dims)))
    if dim_func is not None:
        dim_grid_shape = tuple(n_vals for i in range(n_dims))
        last_dim_grid = dim_func(np.array([vec.flatten() for vec in grid]).T).reshape(dim_grid_shape)
        grid[-1] = last_dim_grid
    if return_flat:
        return np.array([vec.flatten() for vec in grid]).T
    return grid
  

def plot_plane(X_coord, Y_coord, Z_coord, figsize=(10,7), label=None, ax=None, return_ax_obj=False, elev=10, azim=20, **kwargs):
    if ax is None:  
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
    s1 = ax.plot_surface(X_coord, Y_coord, Z_coord, label=label, **kwargs)
    s1._facecolors2d=s1._facecolors3d
    s1._edgecolors2d=s1._edgecolors3d
    if label is not None:
        ax.legend()
    ax.view_init(elev=elev, azim=azim)
    if return_ax_obj:
        return ax



     




