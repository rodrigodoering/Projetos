# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:03:02 2021

@author: rodrigo doering neves
@email: rodrigodoeringneves@gmail.com
"""

import pandas as pd
import numpy as np
import seaborn as sb
import functools

import matplotlib.pyplot as plt
from matplotlib.axes import _subplots

from Visuals._utils_._exceptions import *
from Visuals._utils_._type_definitions import *
from Visuals._base_._graph_base import GraphBase
  

class Plot(GraphBase):
    
    """ Representa o gráfico, herda de graph_base """
       

    def __init__(self, current_Axes: _subplots.Axes=None, **kwargs) -> NoReturn:
        # Inicia os parâmetros 
        if current_Axes is not None:
            self.new_plot(
                n_axis=None, 
                current_Axes = current_Axes, 
                **kwargs
            )


    def Annotate(
            self, 
            coords: NumericArray, 
            annotations: Iterable[str], 
            offset: float = 0.1, 
            ax_offset: int = 0, 
            **kwargs
      
        ) -> NoReturn:
        """Função para exibir anotações no gráfico utilizando como base o método _subplots.Axes.text
        Trabalha especialmente com strings
        
        Argumentos:
        -----------
        coords - Coordenadas (x,y,z) para as anotar, um vetor por anotação
        annotations - iterable contendo as anotações como strings
        offset - desloca a anotação com base em um eixo de referência
        ax_offset - eixo de referência
        **kwargs - devem ser passados para  Axes.plot()
        
        """
        coords = GraphBase.numpy_convert(coords)
        
        # Se um axis inexistente for passado, zera o offset
        if ax_offset > self.axObj.n_axis:
            print('AxesInstance: ax_offset não existe no plot, desconsiderando offset')
            offset = 0
        # deve ser passada uma coordenada para cada anotação
        for vec, _str_ in zip(coords, annotations):
            coord_vals = (vec[i] + offset * int(i == ax_offset) for i in range(self.axObj.n_axis))
            self.axObj.ax_text(*coord_vals, text=_str_, **kwargs)  
    
    
    
    def Function(
            self, 
            function: Callable, 
            X: Union[NumericArray, tuple] = None,
            n_samples: int = 10,
            plot_intercept: bool = False,
            label: str = None,            
            **kwargs
      
        ) -> NoReturn:
        """Desenha funções em 2D e 3D com base no método _subplots.Axes.plot
        
        Argumentos:
        -----------
        function - um callable que retorne um array de valores numéricos para servir de dimensão Y em 2D ou Z em 3D
        X - domínio da função, valores para passar como input. X = [x] para plots 2D ou X = [x,y] para plots 3D
        plot_intercept - Se verdadeiro, exibe o ponto [0, function(0)] em 2D ou [0, 0, function(0,0)] em 3D
        label - nome da função, suporta linguagem LaTex
        **kwargs - devem ser passados para  _subplots.Axes.plot()
        
        """
        n_features = self.axObj.n_axis - 1
        
        if X is None:
            # Se X não for passado, o domínio da função será composto por uma variável sintética
            # Por padrão a variável é criada com um range de 0 à 1
            X = np.array(
                [np.linspace(0,1,n_samples) for i in range(n_features)]
                ).reshape(n_samples, n_features)
    
        if isinstance(X, tuple):
            # Se o domínio passado for uma tupla, utiliza os valores para a variável base
            # Cria uma variável X sintética
            _min_ = X[0]
            _max_ = X[1]
            X = np.array(
                [np.linspace(_min_, _max_, n_samples) for i in range(n_features)]
                ).reshape(n_samples, n_features)           
        
        # Aplica a função e computa Y
        Y = function(X)
        
        # Exibe  o plot
        if label is not None:
            self.axObj.ax_plot(*self.iter_params(X, Y), label=label, **kwargs)
            self.enable_legend()
        else:
            self.axObj.ax_plot(*self.iter_params(X, Y), **kwargs)
        
        # Plota o intercept da função
        if plot_intercept:
            zero_vec = np.zeros(n_features)
            intercept = function(zero_vec)
            p0 = (0 if i + 1 < self.axObj.n_axis else intercept[0] for i in range(self.axObj.n_axis))
            self.axObj.ax_scatter(*p0, color='black', marker='X')
        
        

    def Scatter(
            self,
            X: NumericArray, 
            Y: NumericArray = None, 
            Z: NumericArray = None,
            annot: Iterable[str] = None,
            offset: float = 0.1,
            ax_offset: int = 0,
            fontsize: int = 12,
            annot_color: str = 'black',
            **kwargs
            
        ) -> NoReturn:

        self.axObj.ax_scatter(*self.iter_params(X, Y, Z), **kwargs)

        if annot is not None:
            coords = np.stack([*self.iter_params(X,Y,Z)], axis=1)

            self.Annotate(
                coords = coords, 
                annotations = annot, 
                offset = offset, 
                ax_offset = ax_offset, 
                fontsize = fontsize,
                color = annot_color
            )

        
    def Surface(
            self,
            grid: NumericArray = None,
            function: Callable = None,
            min_val: Numeric = -1,
            max_val: Numeric = 1,
            n_samples: int = 5,
            levels: int = None,
            **kwargs
            
        ) -> NoReturn:
        
        if grid is None:
            
            if callable(function):
                base_variable = np.linspace(min_val, max_val, n_samples)
                grid = np.broadcast_arrays(base_variable[np.newaxis,:], base_variable[:,np.newaxis])
                
                Z = GraphBase.numpy_convert(
                    function(GraphBase.flat_grid(grid))
                )
                
                grid.append(Z.reshape(n_samples, n_samples))                
                          
            else:
                raise Error('InsufficientInput')
        
        if self.axObj.n_axis == 2:
            self.axObj.ax_contourf(*grid, levels=levels, **kwargs)
            
        else:
            self.axObj.ax_surface(*grid, **kwargs)
            
            
            
    def Vectors(
            self,
            X: NumericArray, 
            Y: NumericArray = None, 
            Z: NumericArray = None,
            origin: tuple = None,
            colors: Iterable[str] = None,
            annot: Iterable[str] = None,
            offset: float = 0.1,
            ax_offset: int = 0,
            fontsize: int = 12,
            color: str = 'black',
            **kwargs
        
        ) -> NoReturn:
        
        vecs = np.array([*self.iter_params(X,Y,Z)]).T
        n_vecs = vecs.shape[0]
        
        if any(param is not None for param in [colors, annot]):
            for param in [colors, annot]:
                if param is not None:
                    self.validate_sequence(param, n_vecs)
        
        if origin is not None:
            # ponto de origem deve ter a mesma dimensionalidade do plot
            # i.e mesmo número de componentes
            self.axObj.assert_sequence(origin, None)
        
        vec_scale = {'angles':'xy', 'scale_units':'xy', 'scale':1}
        colors = ['black' for i in range(n_vecs)] if colors is None else colors
        
        for vec, color in zip(vecs, colors):
            self.axObj.ax_quiver(*vec, origin=origin, color=color, **vec_scale)
            
        if annot is not None:

            self.Annotate(
                coords = vecs, 
                annotations = annot, 
                offset = offset, 
                ax_offset = ax_offset, 
                fontsize = fontsize,
                color = color
            )
            
        

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                  
