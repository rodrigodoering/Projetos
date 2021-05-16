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


import sys
path = "C:\\Users\\rodri\\Desktop\\Relacionados a Código\\github_Rodrigo\\Notebooks\\DataViz_Obj"
sys.path.append(path)

from Graphics.utils._exceptions import *
from Graphics.utils._type_definitions import *
from Graphics.plots._graph_base import GraphBase
  

class Plot(GraphBase):
    
    """ Representa o gráfico, herda de AxesInstance """
       
    
    def __init__(self, n_axis: int, current_Axes: _subplots.Axes=None, **kwargs) -> NoReturn:
        # Inicia os parâmetros 
        super(Plot, self).__init__(n_axis, current_Axes=current_Axes, **kwargs)
    
    
    def Annotate(
            self, 
            coords: NumericArray, 
            annotations: Iterable[str], 
            offset: float = 0.1, 
            ax_offset: int = 0, 
            **kwargs
        ) -> NoReturn:
        
        coords = GraphBase.numpy_convert(coords)
        
        # Se um axis inexistente for passado, zera o offset
        if ax_offset > self.axObj.n_axis:
            print('AxesInstance: ax_offset não existe no plot, desconsiderando offset')
            offset = 0
        
        for vec, _str_ in zip(coords, annotations):
            coord_vals = (vec[i] + offset * int(i == ax_offset) for i in range(self.axObj.n_axis))
            self.axObj.ax_text(*coord_vals, text=_str_, **kwargs)  
    
    
    
    def Function(
            self, 
            function: Callable, 
            X: Union[NumericArray, tuple] = None,
            n_samples: int = 10,
            plot_intercept: bool = False,
            specs: dict = None,
            function_label: str = '$f(x)$',
            **kwargs
        ) -> NoReturn:
        
        n_features = self.axObj.n_axis - 1
        
        if X is None:
            X = np.array(
                [np.linspace(0,1,n_samples) for i in range(n_features)]
                ).reshape(n_samples, n_features)
    
        elif isinstance(X, tuple):
            _min_ = X[0]
            _max_ = X[1]
            X = np.array(
                [np.linspace(_min_, _max_, n_samples) for i in range(n_features)]
                ).reshape(n_samples, n_features)           
                
        Y = function(X)
        
        self.axObj.ax_plot(*X.T, Y, label=function_label, **kwargs)
        
        if plot_intercept:
            zero_vec = np.zeros(n_features)
            intercept = function(zero_vec)
            p0 = (0 if i + 1 < self.axObj.n_axis else intercept[0] for i in range(self.axObj.n_axis))
            self.axObj.ax_scatter(*p0, color='red', marker='X')
        
        self.control_plot_specs(specs,'Plots.Function')

            
  
    def Scatter(
            self,
            X: NumericArray, 
            Y: NumericArray = None, 
            Z: NumericArray = None,
            annot: Iterable[str] = None,
            annot_offset: float = 0.1,
            annot_ax_offset: int = 0,
            annot_fontsize: int = 12,
            annot_color: str = 'black',
            specs: dict = None,
            **kwargs
            
        ) -> NoReturn:

        self.axObj.ax_scatter(*self.iter_params(X, Y, Z), **kwargs)

        if annot is not None:
            coords = np.stack([*self.iter_params(X,Y,Z)], axis=1)

            self.annotate(
                coords = coords, 
                annotations = annot, 
                offset = annot_offset, 
                ax_offset = annot_ax_offset, 
                fontsize = annot_fontsize,
                color = annot_color
            )
           
        self.control_plot_specs(specs,'Plots.Scatter')
           

        
    def Surface(
            self,
            grid: NumericArray = None,
            function: Callable = None,
            min_val: Numeric = -1,
            max_val: Numeric = 1,
            n_samples: int = 5,
            levels: int = None,
            specs: dict = None,
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
            
        self.control_plot_specs(specs,'Plots.Surface')
        
      
            
            
    def Vectors(
            self,
            X: NumericArray, 
            Y: NumericArray = None, 
            Z: NumericArray = None,
            annot: Iterable[str] = None,
            annot_offset: float = 0.1,
            annot_ax_offset: int = 0,
            annot_fontsize: int = 12,
            annot_color: str = 'black',
            specs: dict = None,
            **kwargs
        
        ) -> NoReturn:
        raise NotImplementedError
        
        

        
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
                  